//! A stream of messages hashed in place by this crate's threads, in the
//! program's own memory, digests handed back in order ([`stream`]).
//!
//! How it runs:
//! - Each message of up to [`PIECE`] bytes is one entry in a ring of
//!   entries; entries gather into tasks of up to [`MEMBERS`] messages (or
//!   TASK_LEN / 4 bytes), hashed as the queue's members are: messages of
//!   one length side by side, the rest one at a time.
//! - A longer message is split into pieces of PIECE bytes, each a whole
//!   subtree at its offset and a task of its own; the program's thread
//!   merges the pieces' chaining values in order as it hands back the
//!   digest.
//! - Tasks go to the pool's task list a task at a time (one lock per task,
//!   no locked instruction per message). The open task is handed over when
//!   it fills, or when the program collects or finishes.
//! - Digests come back in order on the program's thread, from `submit`
//!   (when the ring is full), `collect` and `finish`. A program waiting
//!   hashes tasks no thread has taken.
//!
//! The messages are borrowed for the scope of [`stream`], which waits for
//! every task before it returns, also when its body panics: no thread
//! reads a message after the scope ends.

use crate::lanes::{MEMBERS, Member, OwnLine, TASK_LEN, TASKS, Task};
use crate::platform::Platform;
use crate::{CHUNK_LEN, CVBytes, CVWords, Hash, Mode, OUT_LEN};
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicUsize, Ordering};

/// The longest message hashed as one entry; a longer one goes in pieces
/// this long (a task's subtree).
pub(crate) const PIECE: usize = TASK_LEN;

/// The bytes a stream keeps for messages written in place.
const RING_BYTES: usize = 4 << 20;

/// The most entries (messages and pieces) in flight in one stream (1024:
/// 4 KiB to 16 KiB messages 8-9% faster than at 8192, i7-12700K; 256 lost
/// 26-130% at 64 and 256 B).
const ENTRIES: usize = 1024;

/// One message, or one piece of a long one, from its submission to its
/// delivery: what the program's thread keeps of it. Its task's threads
/// write its `out`, then count down the `left` of the task's first entry
/// (Release, once a task); the program reads `out` once that is zero
/// (Acquire). Kept apart, so that the program's writes here share no
/// line with the threads'.
#[derive(Clone, Copy)]
struct Entry {
    len: usize,
    kind: Kind,
    /// The index of its task's first entry, whose `left` counts the task.
    first: usize,
    /// The stream's ring bytes reserved up to and including this entry
    /// (its own, for a message written in place): free once it is handed
    /// back.
    ring_end: usize,
}

/// What an entry is.
#[derive(Clone, Copy)]
enum Kind {
    /// A whole message: its digest in the entry's `outs`.
    Message,
    /// A piece of a long message: its first chunk's index, and whether it
    /// is the message's last piece.
    Piece { counter: u64, last: bool },
    /// `count` messages of `message_len` bytes laid out as hash_many takes
    /// them: their digests in the ring, from `at`.
    Batch { message_len: usize, count: usize, at: usize },
    /// The end of a long message whose reader failed after some pieces
    /// ([`Stream::submit_reader`]): no digest; its pieces' chaining values
    /// are dropped.
    Abandoned,
}

/// What a task's threads write for an entry: a message's digest, two to a
/// line (the program reads each line once for two messages: 64 B messages
/// 0.249 -> 0.217 ns/B, 256 B x0.935, i7-12700K); or, in `pieces_out`, a piece's chaining value
/// (a piece at chunk 0, of two chunks or more, gives its two children
/// instead) on a line of its own.
#[repr(transparent)]
struct Out<const N: usize>(UnsafeCell<[u8; N]>);

/// Hash a stream of messages on this crate's threads, in place: `body`
/// submits messages that live at least as long as the call
/// ([`Stream::submit`]), and receives their digests in order from
/// `submit`, [`collect`](Stream::collect) and [`finish`](Stream::finish),
/// a slice at a time: the digests of messages finished in a row, or of a
/// batch's part, together.
/// `body` calls `finish` last (a scope that ends with digests not handed
/// back panics). A message of any length; this crate's threads read it
/// where it is, and `stream` returns only after they are done, also when
/// `body` panics.
///
/// The program calls [`crate::initialize_multithreaded`] first. A stream
/// holds up to 1024 messages (a message over 64 KiB counts once per 64 KiB)
/// in flight; when it is full, `submit` waits for room, handing back
/// finished digests and hashing a waiting task itself. A message stays
/// unhashed in a task still open until the task fills or the program
/// calls `collect` or `finish`, so a program that needs a digest soon
/// calls one of them.
///
/// ```
/// # blake3_servil::initialize_multithreaded();
/// use blake3_servil::Mode;
/// let messages: Vec<Vec<u8>> = (0..1000u32).map(|i| i.to_le_bytes().repeat(16)).collect();
/// let mut digests = Vec::new();
/// blake3_servil::stream(Mode::Hash, |s| {
///     for message in &messages {
///         s.submit(message, |hashes| digests.extend_from_slice(hashes));
///     }
///     s.finish(|hashes| digests.extend_from_slice(hashes));
/// });
/// assert_eq!(digests[7], blake3_servil::hash(&messages[7]));
/// ```
pub fn stream<'env, T>(mode: Mode, body: impl for<'s> FnOnce(&'s mut Stream<'env>) -> T) -> T {
    crate::lanes::require();
    let (key, flags) = mode.key_and_flags();
    let s = SPARE.with(|spare| spare.take()).unwrap_or_else(Storage::new);
    let mut stream = Stream {
        key,
        flags,
        s: core::mem::ManuallyDrop::new(s),
        open_first: 0,
        next: 0,
        delivered: 0,
        head: 0,
        tail: 0,
        open: None,
        stack: arrayvec::ArrayVec::new(),
        pieces: 0,
        env: PhantomData,
    };
    let result = body(&mut stream);
    assert_eq!(stream.delivered, stream.next, "a stream's body calls finish last");
    result
}

/// A stream's entries, and room for their tasks in the pool's list: kept
/// for the thread's next stream (a stream of one message up to 4 KiB
/// after a 1 ms pause: 55 us with 248 KiB allocated and faulted in each
/// time, 4-5 us with the spare; one stream kept open, 2.6-4 us. i7-12700K).
struct Storage {
    entries: Box<[Entry]>,
    /// RING_BYTES for messages written in place ([`Stream::submit_with`])
    /// and batches' digests, each at a 64-byte boundary, in the order of
    /// their entries: an allocation of its own, reached by pointer alone
    /// (the program writes some bytes while threads read others).
    ring: *mut u8,
    outs: Box<[Out<OUT_LEN>]>,
    pieces_out: Box<[OwnLine<Out<64>>]>,
    /// For a task's first entry, the count of the task's entries still
    /// unhashed: on lines of their own (128 bytes, as the adjacent-line
    /// prefetcher pairs lines), so that the program polling one shares no
    /// line with the threads writing digests. Zero between streams.
    lefts: Box<[OwnLine<AtomicUsize>]>,
}

std::thread_local! {
    /// The storage of this thread's last stream.
    static SPARE: core::cell::Cell<Option<Storage>> = const { core::cell::Cell::new(None) };
}

impl Storage {
    fn new() -> Storage {
        // Every entry may be a task in flight at once.
        TASKS.make_room(ENTRIES);
        Storage {
            entries: std::vec![Entry { len: 0, kind: Kind::Message, first: 0, ring_end: 0 }; ENTRIES].into_boxed_slice(),
            // Zeroed, untouched pages: only the bytes used are faulted in.
            ring: Box::into_raw(std::vec![0u8; RING_BYTES].into_boxed_slice()) as *mut u8,
            outs: (0..ENTRIES).map(|_| Out(UnsafeCell::new([0; OUT_LEN]))).collect(),
            pieces_out: (0..ENTRIES).map(|_| OwnLine(Out(UnsafeCell::new([0; 64])))).collect(),
            lefts: (0..ENTRIES).map(|_| OwnLine(AtomicUsize::new(0))).collect(),
        }
    }
}

impl Storage {
    /// The ring's byte at `at`.
    fn ring_at(&self, at: usize) -> *mut u8 {
        debug_assert!(at <= RING_BYTES);
        // Sound: `at` is in the ring.
        unsafe { self.ring.add(at) }
    }
}

impl Drop for Storage {
    /// Give back its room and its ring.
    fn drop(&mut self) {
        TASKS.give_room(ENTRIES);
        // Sound: the ring is Storage::new's allocation, and no task reads it
        // (a stream's storage drops only after its tasks are done).
        drop(unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(self.ring, RING_BYTES)) });
    }
}

/// Part `i` of a message in `pieces` parts: the message whole, or a piece.
fn part_kind(i: usize, pieces: usize) -> Kind {
    if pieces == 1 { Kind::Message } else { Kind::Piece { counter: (i * PIECE / CHUNK_LEN) as u64, last: i + 1 == pieces } }
}

/// The messages of one [`stream`] call, borrowed for its scope.
pub struct Stream<'env> {
    key: CVWords,
    flags: u8,
    /// Its own until the stream ends (then the thread's spare).
    s: core::mem::ManuallyDrop<Storage>,
    /// The open task's first entry.
    open_first: usize,
    /// Entries submitted, and entries handed back (each counts on).
    next: usize,
    delivered: usize,
    /// Ring bytes reserved, and freed (each counts on; the ring holds
    /// head - tail).
    head: usize,
    tail: usize,
    /// Short messages gathered, not yet handed to the pool.
    open: Option<Task>,
    /// A long message being handed back: its pieces' chaining values so
    /// far (merged as the Hasher's stack merges), and how many.
    stack: arrayvec::ArrayVec<CVBytes, { crate::MAX_DEPTH + 1 }>,
    pieces: u64,
    /// Invariant in 'env, as std::thread::Scope is: the messages' lifetime.
    env: PhantomData<&'env mut &'env ()>,
}

// Sound: the entries are reached by the program's thread through `&mut
// self`, and by the pool's threads only for entries in tasks in flight,
// across the Release/Acquire pair on each entry's `left`; the messages are
// shared borrows, alive for 'env.
unsafe impl Send for Stream<'_> {}

impl<'env> Stream<'env> {
    /// Hash `message` where it is; `digest` receives any digests this call
    /// hands back.
    pub fn submit(&mut self, message: &'env [u8], mut digest: impl FnMut(&[Hash])) {
        let pieces = message.len().div_ceil(PIECE).max(1);
        for i in 0..pieces {
            let part = &message[i * PIECE..][..(message.len() - i * PIECE).min(PIECE)];
            let at = self.head;
            self.add_part(part.as_ptr(), part.len(), part_kind(i, pieces), at, &mut digest);
        }
    }

    /// Hash a message of `len` bytes that `fill` writes in place, into the
    /// stream's own bytes, in pieces of up to 64 KiB: `fill(at, room)`
    /// writes the message's bytes from `at` into all of `room` (a read
    /// from a file or socket straight into the stream). `digest` receives
    /// any digests this call hands back. The stream reuses the bytes once
    /// the message's digest is handed back; a stream keeps 4 MiB of them,
    /// and `submit_with` waits for room as `submit` does.
    pub fn submit_with(&mut self, len: usize, mut fill: impl FnMut(usize, &mut [u8]), mut digest: impl FnMut(&[Hash])) {
        let pieces = len.div_ceil(PIECE).max(1);
        for i in 0..pieces {
            let part_len = (len - i * PIECE).min(PIECE);
            let at = self.reserve(part_len, &mut digest);
            // Sound: [at, at + part_len) is reserved and free (no entry in
            // flight holds it), and `&mut self` keeps every other writer out.
            let room = unsafe { core::slice::from_raw_parts_mut(self.s.ring_at(at), part_len) };
            fill(i * PIECE, room);
            let end = self.head;
            self.add_part(room.as_ptr(), part_len, part_kind(i, pieces), end, &mut digest);
        }
    }

    /// Hash the message `reader` yields, to its end, read straight into the
    /// stream's own bytes in pieces of up to 64 KiB (a file or socket read
    /// into the stream, overlapping the hashing of earlier pieces): the
    /// message's length on success. `digest` receives any digests this
    /// call hands back. On a read error (other than an interruption, which
    /// reads again) the message gets no digest and the error returns; the
    /// stream goes on with the next message. The length need not be known:
    /// the stream reads one piece ahead to tell the last.
    pub fn submit_reader(&mut self, mut reader: impl std::io::Read, mut digest: impl FnMut(&[Hash])) -> std::io::Result<u64> {
        let start = self.head;
        let (mut at, mut end, mut n, mut read) = self.read_piece(&mut reader, &mut digest);
        let mut i = 0;
        loop {
            let ahead = (n == PIECE && read.is_ok()).then(|| self.read_piece(&mut reader, &mut digest));
            let failed = read.is_err() || ahead.as_ref().is_some_and(|a| a.3.is_err());
            if failed {
                let error = match (read, ahead) {
                    (Err(e), _) => e,
                    (_, Some((_, _, _, Err(e)))) => e,
                    _ => unreachable!("a read failed"),
                };
                // The bytes not yet handed to the pool go; the pieces that
                // were end in an entry that drops their chaining values.
                self.head = if i == 0 { start } else { end - PIECE.div_ceil(64) * 64 };
                if i > 0 {
                    let head = self.head;
                    self.add_entry(0, Kind::Abandoned, head, &mut digest);
                }
                return Err(error);
            }
            let last = match &ahead {
                None => true,
                // Exactly a piece, then the end: give the read-ahead room back.
                Some((_, _, 0, _)) => {
                    self.head = end;
                    true
                }
                Some(_) => false,
            };
            let kind = if i == 0 && last { Kind::Message } else { Kind::Piece { counter: (i * PIECE / CHUNK_LEN) as u64, last } };
            // [at, at + n): this piece's reserved room, written.
            let input = self.s.ring_at(at);
            self.add_part(input, n, kind, end, &mut digest);
            if last {
                return Ok((i * PIECE + n) as u64);
            }
            (at, end, n, read) = ahead.unwrap();
            i += 1;
        }
    }

    /// A piece's room (PIECE bytes, reserved), read into from `reader` until
    /// it is full, the reader ends, or a read fails: its offset, the ring's
    /// head after its reservation, the bytes read, and the read's result.
    fn read_piece(&mut self, reader: &mut impl std::io::Read, digest: &mut impl FnMut(&[Hash])) -> (usize, usize, usize, std::io::Result<()>) {
        let at = self.reserve(PIECE, digest);
        let end = self.head;
        // Sound: [at, at + PIECE) is reserved and free, and `&mut self`
        // keeps every other writer out.
        let room = unsafe { core::slice::from_raw_parts_mut(self.s.ring_at(at), PIECE) };
        let mut n = 0;
        while n < PIECE {
            match reader.read(&mut room[n..]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return (at, end, n, Err(e)),
            }
        }
        (at, end, n, Ok(()))
    }

    /// Hash a batch of messages of `message_len` bytes where it is, laid
    /// out as [`hash_many`](crate::hash_many) takes them (each in whole
    /// blocks, zero after its bytes); `digest` receives any digests this
    /// call hands back, the batch's in its messages' order. The batch goes in parts of
    /// up to 64 KiB, each one entry.
    pub fn submit_batch(&mut self, batch: &'env [u8], message_len: usize, mut digest: impl FnMut(&[Hash])) {
        let slot = crate::many::slot_len(message_len);
        crate::many::assert_slots(batch, message_len, batch.len() / slot);
        for part in batch.chunks((TASK_LEN / slot).max(1) * slot) {
            let count = part.len() / slot;
            let at = self.reserve(count * OUT_LEN, &mut digest);
            let end = self.head;
            self.add_part(part.as_ptr(), part.len(), Kind::Batch { message_len, count, at }, end, &mut digest);
        }
    }

    /// Hash a batch of `count` messages of `message_len` bytes that `fill`
    /// writes in place, laid out as [`hash_many`](crate::hash_many) takes
    /// them: `fill(at, room)` writes the batch's bytes from `at` (padding
    /// included: each message in whole blocks, zero after its bytes) into
    /// all of `room`, a part of up to 64 KiB at a time. As
    /// [`submit_with`](Stream::submit_with) and [`submit_batch`](Stream::submit_batch).
    pub fn submit_batch_with(&mut self, count: usize, message_len: usize, mut fill: impl FnMut(usize, &mut [u8]), mut digest: impl FnMut(&[Hash])) {
        let slot = crate::many::slot_len(message_len);
        let per_part = (TASK_LEN / slot).max(1);
        for first in (0..count).step_by(per_part) {
            let part_count = (count - first).min(per_part);
            let bytes_at = self.reserve(part_count * slot, &mut digest);
            // Sound: as in submit_with.
            let room = unsafe { core::slice::from_raw_parts_mut(self.s.ring_at(bytes_at), part_count * slot) };
            fill(first * slot, room);
            crate::many::assert_slots(room, message_len, part_count);
            let at = self.reserve(part_count * OUT_LEN, &mut digest);
            let end = self.head;
            self.add_part(room.as_ptr(), room.len(), Kind::Batch { message_len, count: part_count, at }, end, &mut digest);
        }
    }

    /// An entry of `kind`, `len` bytes at `input`, its ring bytes reserved
    /// up to `ring_end`: a whole message, or a batch's part, joins the open
    /// task (of its kind: one of another kind is handed over first); a
    /// piece of a long message is a task of its own, a whole subtree at its
    /// offset.
    fn add_part(&mut self, input: *const u8, len: usize, kind: Kind, ring_end: usize, digest: &mut impl FnMut(&[Hash])) {
        let entry = self.add_entry(len, kind, ring_end, digest);
        let (batch, out) = match kind {
            Kind::Message => (None, self.s.outs[entry].0.get() as *mut u8),
            Kind::Abandoned => unreachable!("an abandoned message adds no part"),
            Kind::Batch { message_len, at, .. } => (Some(message_len), self.s.ring_at(at)),
            Kind::Piece { counter, .. } => {
                self.s.lefts[entry].store(1, Ordering::Relaxed);
                // Sound: the bytes stay in place until the entry is handed back.
                let bytes = unsafe { core::slice::from_raw_parts(input, len) };
                let task = Task {
                    key: self.key,
                    flags: self.flags,
                    out: self.s.pieces_out[entry].0.0.get() as *mut u8,
                    left: &*self.s.lefts[entry],
                    ..Task::of(bytes, counter)
                };
                TASKS.push(core::iter::once(task));
                return;
            }
        };
        if self.open.as_ref().is_some_and(|open| open.batch != batch) {
            self.publish();
        }
        let (key, flags) = (self.key, self.flags);
        let open = self.open.get_or_insert_with(|| Task::members(&key, flags, batch));
        // The task's first member's `left` counts all of its members (set
        // as the task is handed over).
        if open.members == 0 {
            self.open_first = entry;
        }
        let first = self.open_first;
        self.s.entries[entry].first = first;
        open.member[open.members] = Member { input, len, out, left: &*self.s.lefts[first] };
        open.members += 1;
        open.len += len;
        // Messages longer than a block hash one at a time: a task of them
        // stops at a quarter of a task's bytes, to spread over more threads
        // (the queue's rule); batches fill a task's bytes.
        if open.members == MEMBERS || open.len >= if batch.is_some() { TASK_LEN } else { TASK_LEN / 4 } {
            self.publish();
        }
    }

    /// The ring offset of `len` free bytes at a 64-byte boundary (whole,
    /// before the ring's end), reserved: waits, handing back digests and
    /// hashing tasks, until entries in flight free them.
    fn reserve(&mut self, len: usize, digest: &mut impl FnMut(&[Hash])) -> usize {
        let slot = len.div_ceil(64) * 64;
        let mut at = self.head % RING_BYTES;
        // A slot that would cross the end starts the next lap instead.
        let skip = if at + slot > RING_BYTES { RING_BYTES - at } else { 0 };
        while RING_BYTES - (self.head - self.tail) < skip + slot {
            self.publish();
            self.help(digest);
        }
        if skip > 0 {
            at = 0;
        }
        self.head += skip + slot;
        at
    }

    /// Hand back every digest finished, in order, without waiting; and hand
    /// the open task to the pool, so what it holds starts hashing.
    pub fn collect(&mut self, mut digest: impl FnMut(&[Hash])) {
        self.publish();
        self.deliver(&mut digest);
    }

    /// Hand back every message's digest, in order: this thread hashes any
    /// task no thread has taken, and waits for the rest.
    pub fn finish(&mut self, mut digest: impl FnMut(&[Hash])) {
        self.publish();
        while self.delivered < self.next {
            self.help(&mut digest);
        }
    }

    /// Hand back what is done; with nothing done, hash a waiting task or
    /// pause.
    fn help(&mut self, digest: &mut impl FnMut(&[Hash])) {
        if self.deliver(digest) == 0 && !crate::lanes::run_one_task() {
            core::hint::spin_loop();
        }
    }

    /// A new entry of `len` bytes, waiting for room when every entry is in
    /// flight.
    #[inline(always)]
    fn add_entry(&mut self, len: usize, kind: Kind, ring_end: usize, digest: &mut impl FnMut(&[Hash])) -> usize {
        if self.next - self.delivered == ENTRIES {
            self.wait_for_room(digest);
        }
        let index = self.next % ENTRIES;
        self.s.entries[index] = Entry { len, kind, first: index, ring_end };
        self.next += 1;
        index
    }

    /// Hand back digests, and hash tasks, until an entry is free (out of
    /// line: a frame of its own, set up only when the stream is full).
    #[cold]
    #[inline(never)]
    fn wait_for_room(&mut self, digest: &mut impl FnMut(&[Hash])) {
        self.publish();
        while self.next - self.delivered == ENTRIES {
            self.help(digest);
        }
    }

    /// Hand the open task to the pool.
    fn publish(&mut self) {
        if let Some(task) = self.open.take() {
            // Before the push, whose lock orders it before the threads' reads.
            self.s.lefts[self.open_first].store(task.members, Ordering::Relaxed);
            TASKS.push(core::iter::once(task));
        }
    }

    /// Hand back every finished entry from the oldest on; how many.
    fn deliver(&mut self, digest: &mut impl FnMut(&[Hash])) -> usize {
        let mut count = 0;
        // Whole messages finished in a row: their digests sit side by side
        // in `outs` (up to its end), handed back as one slice.
        let mut run = (0, 0);
        while self.delivered < self.next {
            let index = self.delivered % ENTRIES;
            let Entry { len, kind, first, ring_end } = self.s.entries[index];
            if self.s.lefts[first].load(Ordering::Acquire) != 0 {
                break;
            }
            self.delivered += 1;
            self.tail = ring_end;
            count += 1;
            if let Kind::Message = kind {
                if run.1 > 0 && run.0 + run.1 == index {
                    run.1 += 1;
                } else {
                    self.hand_back(run, digest);
                    run = (index, 1);
                }
                continue;
            }
            self.hand_back(core::mem::take(&mut run), digest);
            // Sound: the task wrote its outs before counting `left` down;
            // Hash is [u8; 32] (repr(transparent)).
            match kind {
                Kind::Message => unreachable!(),
                Kind::Abandoned => {
                    self.stack.clear();
                    self.pieces = 0;
                }
                Kind::Batch { count, at, .. } => digest(unsafe { core::slice::from_raw_parts(self.s.ring_at(at) as *const Hash, count) }),
                Kind::Piece { counter, last } => {
                    let out = unsafe { *self.s.pieces_out[index].0.0.get() };
                    let (key, flags, platform) = (self.key, self.flags, Platform::detect());
                    let parent = |left: &CVBytes, right: &CVBytes| crate::hasher::parent_output(left, right, &key, flags, platform);
                    let cv: CVBytes = if counter == 0 && len > CHUNK_LEN {
                        parent(out[..OUT_LEN].try_into().unwrap(), out[OUT_LEN..].try_into().unwrap()).chaining_value()
                    } else {
                        out[..OUT_LEN].try_into().unwrap()
                    };
                    // As the Hasher's stack: before pushing piece n, merge
                    // down to as many values as n has 1 bits.
                    while self.stack.len() > self.pieces.count_ones() as usize {
                        let right = self.stack.pop().unwrap();
                        let left = self.stack.pop().unwrap();
                        self.stack.push(parent(&left, &right).chaining_value());
                    }
                    self.stack.push(cv);
                    self.pieces += 1;
                    if last {
                        let mut right = self.stack.pop().unwrap();
                        while self.stack.len() > 1 {
                            right = parent(&self.stack.pop().unwrap(), &right).chaining_value();
                        }
                        digest(&[parent(&self.stack.pop().unwrap(), &right).root_hash()]);
                        self.pieces = 0;
                    }
                }
            }
        }
        self.hand_back(run, digest);
        count
    }

    /// Hand back the digests of `run.1` whole messages from entry `run.0`.
    fn hand_back(&self, run: (usize, usize), digest: &mut impl FnMut(&[Hash])) {
        if run.1 > 0 {
            // Sound: their tasks are done (deliver), Out<32> is [u8; 32] and
            // so is Hash (both repr(transparent)), side by side in `outs`.
            digest(unsafe { core::slice::from_raw_parts((self.s.outs.as_ptr() as *const Hash).add(run.0), run.1) });
        }
    }
}

impl Drop for Stream<'_> {
    /// Wait for every task still reading the messages (also when the body
    /// panicked), then keep the storage as the thread's spare. A stream
    /// lives only inside [`stream`], which owns it, so this runs before the
    /// scope ends.
    fn drop(&mut self) {
        self.publish();
        for i in self.delivered..self.next {
            let left = &self.s.lefts[self.s.entries[i % ENTRIES].first];
            while left.load(Ordering::Acquire) != 0 {
                if !crate::lanes::run_one_task() {
                    core::hint::spin_loop();
                }
            }
        }
        // Sound: `s` is not used again.
        let s = unsafe { core::mem::ManuallyDrop::take(&mut self.s) };
        SPARE.with(|spare| spare.set(Some(s)));
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Messages of every kind the stream handles (empty, short, one length
    /// many times over, a whole piece, a piece and a byte, several pieces
    /// and a partial one), borrowed and written in place, in each mode,
    /// against hash_with, in order; far more entries and bytes than a
    /// stream holds at once, so it fills, wraps its ring, and waits.
    #[test]
    fn test_stream_against_hash_with() {
        let mut input = std::vec![0u8; (1 << 20) + 200];
        crate::test::paint_test_input(&mut input);
        let key = [5u8; 32];
        let lens: std::vec::Vec<usize> =
            (0..12000).map(|i| [0, 1, 63, 64, 64, 64, 100, 1024, 1024, 1500, 4096, PIECE, PIECE + 1, 200_000, (1 << 20) + 7, 64][i % 16]).collect();
        let message = |i: usize| &input[i % 97..][..lens[i]];
        for (mode, ways) in [(Mode::Hash, 0), (Mode::Keyed(&key), 1), (Mode::DeriveKey("stream test"), 2)] {
            let mut got = std::vec::Vec::new();
            stream(mode, |s| {
                for i in 0..lens.len() {
                    // Borrowed, written in place (in reads of up to 1000
                    // bytes), or each in turn.
                    if ways == 0 || (ways == 2 && i % 3 == 0) {
                        s.submit(message(i), |h| got.extend_from_slice(h));
                    } else {
                        s.submit_with(
                            lens[i],
                            |at, room| {
                                for (j, part) in room.chunks_mut(1000).enumerate() {
                                    part.copy_from_slice(&message(i)[at + j * 1000..][..part.len()]);
                                }
                            },
                            |h| got.extend_from_slice(h),
                        );
                    }
                    if i % 500 == 499 {
                        s.collect(|h| got.extend_from_slice(h));
                    }
                }
                s.finish(|h| got.extend_from_slice(h));
            });
            assert_eq!(got.len(), lens.len());
            for i in 0..lens.len() {
                assert_eq!(got[i], crate::hash_with(mode, message(i)), "message {i}, {} bytes", lens[i]);
            }
        }
    }

    /// Batches of every message length kind (empty, short, one block, not
    /// whole blocks, a chunk, longer than a task's bytes), counts from 0
    /// to more than a part holds, borrowed and written in place, between
    /// single messages (the open task changes kind), in each mode, against
    /// hash_with of each message, in order.
    #[test]
    fn test_stream_batches_against_hash_with() {
        let key = [7u8; 32];
        let cases: std::vec::Vec<(usize, usize)> = [0, 1, 64, 100, 1024, 1500, 70_000]
            .iter()
            .flat_map(|&len| [0, 1, 3, 16, 1100].into_iter().filter(move |&count| len < 10_000 || count < 20).map(move |count| (len, count)))
            .collect();
        let batch_of = |len: usize, count: usize| -> std::vec::Vec<u8> {
            let slot = crate::many::slot_len(len);
            let mut batch = std::vec![0u8; slot * count];
            for (i, s) in batch.chunks_mut(slot).enumerate() {
                crate::test::paint_test_input(&mut s[..len]);
                s[..len.min(8)].iter_mut().for_each(|b| *b ^= i as u8);
            }
            batch
        };
        let batches: std::vec::Vec<std::vec::Vec<u8>> = cases.iter().map(|&(len, count)| batch_of(len, count)).collect();
        let lone = [5u8; 300];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("stream batches")] {
            let mut got = std::vec::Vec::new();
            let mut want = std::vec::Vec::new();
            stream(mode, |s| {
                for (k, (&(len, count), batch)) in cases.iter().zip(&batches).enumerate() {
                    if k % 2 == 0 {
                        s.submit_batch(batch, len, |h| got.extend_from_slice(h));
                    } else {
                        s.submit_batch_with(count, len, |at, room| room.copy_from_slice(&batch[at..][..room.len()]), |h| got.extend_from_slice(h));
                    }
                    let slot = crate::many::slot_len(len);
                    want.extend(batch.chunks(slot).map(|m| crate::hash_with(mode, &m[..len])));
                    s.submit(&lone, |h| got.extend_from_slice(h));
                    want.push(crate::hash_with(mode, &lone));
                }
                s.finish(|h| got.extend_from_slice(h));
            });
            assert_eq!(got.len(), want.len());
            assert!(got == want, "a batch's digests in order");
        }
    }

    /// A stream takes more messages after finishing; a body that ends
    /// without finishing panics (after the threads are done with its
    /// messages), and so does one that panics itself.
    #[test]
    fn test_stream_finish_and_panics() {
        let input = std::vec![9u8; 300_000];
        let mut got = std::vec::Vec::new();
        stream(Mode::Hash, |s| {
            for round in 0..3 {
                for len in [0, 65, 70_000] {
                    s.submit(&input[..len], |h| got.extend_from_slice(h));
                }
                s.finish(|h| got.extend_from_slice(h));
                assert_eq!(got.len(), 3 * (round + 1));
            }
        });
        assert_eq!(got[5], crate::hash(&input[..70_000]));
        for unwinding in [false, true] {
            let result = std::panic::catch_unwind(|| {
                stream(Mode::Hash, |s| {
                    for len in [10, 100_000, 64, 300_000] {
                        s.submit(&input[..len], |_| {});
                    }
                    assert!(!unwinding, "the body panics");
                })
            });
            assert!(result.is_err());
        }
    }

    /// A reader that yields `data` in reads of varying sizes (1 byte to
    /// 9000), interrupted now and then, failing after `fail_at` bytes.
    struct Uneven<'a> {
        data: &'a [u8],
        at: usize,
        calls: usize,
        fail_at: Option<usize>,
    }

    impl std::io::Read for Uneven<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.calls += 1;
            if self.calls % 7 == 0 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if self.fail_at.is_some_and(|f| self.at >= f) {
                return Err(std::io::Error::other("the reader fails"));
            }
            let limit = self.fail_at.unwrap_or(self.data.len()).min(self.data.len());
            let n = buf.len().min([1, 9000, 333, 65536, 4096][self.calls % 5]).min(limit - self.at);
            buf[..n].copy_from_slice(&self.data[self.at..][..n]);
            self.at += n;
            Ok(n)
        }
    }

    /// Readers of every length the pieces make special (empty, under,
    /// at, and over a piece, many pieces), in each mode, between borrowed
    /// messages, against hash_with, in order; and readers failing in the
    /// first piece, at a piece's end, and later: no digest for them, an
    /// error back, and the next message's digest still right.
    #[test]
    fn test_stream_readers() {
        let mut input = std::vec![0u8; 5 * PIECE + 77];
        crate::test::paint_test_input(&mut input);
        let key = [9u8; 32];
        let lens = [0, 1, 1000, PIECE - 1, PIECE, PIECE + 1, 2 * PIECE, 3 * PIECE + 5, 5 * PIECE + 77];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("stream readers")] {
            let (mut got, mut want) = (std::vec::Vec::new(), std::vec::Vec::new());
            stream(mode, |s| {
                for _ in 0..3 {
                    for &len in &lens {
                        let n = s.submit_reader(Uneven { data: &input[..len], at: 0, calls: 0, fail_at: None }, |h| got.extend_from_slice(h)).unwrap();
                        assert_eq!(n, len as u64);
                        want.push(crate::hash_with(mode, &input[..len]));
                        s.submit(&input[..100], |h| got.extend_from_slice(h));
                        want.push(crate::hash_with(mode, &input[..100]));
                    }
                }
                for fail_at in [0, 500, PIECE, PIECE + 3, 2 * PIECE, 3 * PIECE + 1] {
                    let r = s.submit_reader(Uneven { data: &input, at: 0, calls: 0, fail_at: Some(fail_at) }, |h| got.extend_from_slice(h));
                    assert!(r.is_err(), "a failing reader's error comes back (failing at {fail_at})");
                    s.submit(&input[..3 * PIECE + 5], |h| got.extend_from_slice(h));
                    want.push(crate::hash_with(mode, &input[..3 * PIECE + 5]));
                }
                s.finish(|h| got.extend_from_slice(h));
            });
            assert!(got == want, "readers' digests in order, none for the failed ones");
        }
    }

    /// A few messages through a stream, small enough for Miri: the handover
    /// of entries to the pool's threads and back, checked for data races.
    #[test]
    fn test_miri_stream() {
        let input = std::vec![3u8; 3 * PIECE + 5];
        let lens = [0, 64, 64, 1000, PIECE + 1, 3 * PIECE + 5];
        let mut got = std::vec::Vec::new();
        stream(Mode::Hash, |s| {
            for len in lens {
                s.submit(&input[..len], |h| got.extend_from_slice(h));
                s.submit_with(len, |at, room| room.copy_from_slice(&input[at..][..room.len()]), |h| got.extend_from_slice(h));
            }
            assert_eq!(s.submit_reader(&input[..2 * PIECE + 1], |h| got.extend_from_slice(h)).unwrap(), 2 * PIECE as u64 + 1);
            assert!(s.submit_reader(Uneven { data: &input, at: 0, calls: 0, fail_at: Some(PIECE + 1) }, |h| got.extend_from_slice(h)).is_err());
            s.submit_batch(&input[..4 * 64], 64, |h| got.extend_from_slice(h));
            s.submit_batch_with(3, 64, |at, room| room.copy_from_slice(&input[at..][..room.len()]), |h| got.extend_from_slice(h));
            s.finish(|h| got.extend_from_slice(h));
        });
        let mut want: std::vec::Vec<Hash> = lens.iter().flat_map(|&l| [crate::hash(&input[..l]); 2]).collect();
        want.push(crate::hash(&input[..2 * PIECE + 1]));
        want.extend([crate::hash(&input[..64]); 7]);
        assert_eq!(got, want);
    }
}
