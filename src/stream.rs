//! A stream of messages hashed in place by this crate's threads, in the
//! program's own memory, each digest written into the program's own memory
//! and that memory handed back in order ([`stream`]).
//!
//! How it runs:
//! - Each message of up to [`PIECE`] bytes is one entry in a ring of
//!   entries; entries gather into tasks of up to [`MEMBERS`] messages (or
//!   TASK_LEN / 4 bytes), hashed as members of a task: messages of
//!   one length side by side, the rest one at a time.
//! - A longer message is split into pieces of PIECE bytes, each a whole
//!   subtree at its offset and a task of its own; the program's thread
//!   merges the pieces' chaining values in order as it hands back the
//!   digest.
//! - The stream is its own source of tasks for the pool's workers
//!   (`lanes::sources`): it writes each task into its source's ring and
//!   publishes it with one store, and workers take tasks in order by
//!   compare-and-swap, with no lock and no shared queue. The open task is
//!   handed over when it fills, or when the program collects or finishes.
//! - Each submission names where its digests go, in the program's memory;
//!   the threads write them there (a long message's, merged on the
//!   program's thread, too). The stream hands that memory back to the
//!   stream's one handler in submission order, on the program's thread,
//!   from a submission (when the stream is full), `collect` and `finish`.
//!   A program waiting hashes tasks no thread has taken.
//!
//! The messages are borrowed for the scope of [`stream`], which waits for
//! every task before it returns, also when its body panics: no thread
//! reads a message after the scope ends.

use crate::lanes::{MEMBERS, Member, OwnLine, TASK_LEN, Task};
use crate::platform::Platform;
use crate::{CHUNK_LEN, CVBytes, CVWords, Hash, Mode, OUT_LEN};
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicUsize, Ordering};

/// The longest message hashed as one entry; a longer one goes in pieces
/// this long (a task's subtree).
pub(crate) const PIECE: usize = TASK_LEN;

/// The bytes a stream keeps for messages and batches written in place:
/// 2 MiB. With the digests in the program's places, 4 MiB made two
/// programs streaming 4 MiB batches 18% slower (the ring, the places, and
/// the programs' own buffers past the L3, i7-12700K); 2 MiB is level there
/// and 2-4 points less fast than 4 MiB on some other cells.
const RING_BYTES: usize = 2 << 20;

/// The most entries (messages and pieces) in flight in one stream (1024:
/// 4 KiB to 16 KiB messages 8-9% faster than at 8192, i7-12700K; 256 lost
/// 26-130% at 64 and 256 B).
const ENTRIES: usize = 1024;

/// One message, one piece of a long one, or one part of a batch, from its
/// submission to its delivery: what the program's thread keeps of it. Its
/// task's threads write its digests, then count down the `left` of the
/// task's first entry (Release, once a task); the program hands them back
/// once that is zero (Acquire). Kept apart, so that the program's writes
/// here share no line with the threads'.
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

/// What an entry is. Each `out` is the program's, borrowed for 'env.
#[derive(Clone, Copy)]
enum Kind {
    /// A whole message: its digest at `out`.
    Message { out: *mut Hash },
    /// A piece of a long message: its first chunk's index, whether it is
    /// the message's last piece, and where the message's digest goes.
    Piece { counter: u64, last: bool, out: *mut Hash },
    /// A part of a batch of messages of `message_len` bytes laid out as
    /// hash_many takes them (an empty batch: one part of none), its
    /// digests written into the batch's space; the batch's last part holds
    /// that whole space, handed back once it is done.
    Batch { message_len: usize, whole: Option<(*mut Hash, usize)> },
    /// The end of a long message whose reader failed after some pieces
    /// ([`Stream::submit_reader`]): no digest; its pieces' chaining values
    /// are dropped.
    Abandoned,
}

/// What a task's threads write for a piece of a long message, in
/// `pieces_out`: its chaining value (a piece at chunk 0, of two chunks or
/// more, gives its two children instead), on a line of its own.
#[repr(transparent)]
struct Out<const N: usize>(UnsafeCell<[u8; N]>);

/// Hash a stream of messages on this crate's threads, in place: `body`
/// submits messages that live at least as long as the call
/// ([`Stream::submit`]), each with the place its digest goes, in the
/// program's memory (a batch's, the place for all its digests). The
/// threads write the digests there, and `done` receives each place back,
/// in submission order, once its digests are in it: from a submission
/// (when the stream is full), [`collect`](Stream::collect) and
/// [`finish`](Stream::finish), on this thread. `body` calls `finish` last
/// (a scope that ends with places not handed back panics). A message of
/// any length; this crate's threads read it where it is, and `stream`
/// returns only after they are done, also when `body` panics.
///
/// The program calls [`crate::initialize_multithreaded`] first. A stream
/// holds up to 1024 messages (a message over 64 KiB counts once per 64 KiB,
/// a batch once per 64 KiB of messages) in flight; when it is full, a
/// submission waits for room, handing places back and hashing a waiting
/// task itself. A message stays unhashed in a task still open until the
/// task fills or the program calls `collect` or `finish`, so a program
/// that needs a digest soon calls one of them. State that `done` and
/// `body` share goes in a [`Cell`](core::cell::Cell) or
/// [`RefCell`](core::cell::RefCell).
///
/// Each message's place comes back alone, about 1-2 ns a message: for
/// many short messages of one length, [`Stream::submit_batch`] hands back
/// one place per batch and hashes them side by side (64-byte messages one
/// at a time ran 4-8% slower than in the stream's earlier design, which
/// handed back runs of digests from its own memory; batches of them,
/// 5-13% faster; i7-12700K).
///
/// ```
/// # blake3_servil::initialize_multithreaded();
/// use blake3_servil::{Hash, Mode};
/// let messages: Vec<Vec<u8>> = (0..1000u32).map(|i| i.to_le_bytes().repeat(16)).collect();
/// let mut digests = vec![Hash::from_bytes([0; 32]); messages.len()];
/// let mut handed_back = 0;
/// blake3_servil::stream(Mode::Hash, |done: &mut [Hash]| handed_back += done.len(), |s| {
///     for (message, digest) in messages.iter().zip(&mut digests) {
///         s.submit(message, digest);
///     }
///     s.finish();
/// });
/// assert_eq!(handed_back, 1000);
/// assert_eq!(digests[7], blake3_servil::hash(&messages[7]));
/// ```
pub fn stream<'env, D: FnMut(&'env mut [Hash]), T>(mode: Mode, done: D, body: impl for<'s> FnOnce(&'s mut Stream<'env, D>) -> T) -> T {
    crate::lanes::require();
    let (key, flags) = mode.key_and_flags();
    let s = Storage::take();
    let slot = s.source.register();
    crate::lanes::sources::opened();
    let mut stream = Stream {
        key,
        flags,
        s,
        slot,
        done,
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
    /// RING_BYTES for messages and batches written in place
    /// ([`Stream::submit_with`]), each at a 64-byte boundary, in the order
    /// of their entries: an allocation of its own, reached by pointer
    /// alone (the program writes some bytes while threads read others).
    ring: *mut u8,
    pieces_out: Box<[OwnLine<Out<64>>]>,
    /// For a task's first entry, the count of the task's entries still
    /// unhashed: on lines of their own (128 bytes, as the adjacent-line
    /// prefetcher pairs lines), so that the program polling one shares no
    /// line with the threads writing digests. Zero between streams.
    lefts: Box<[OwnLine<AtomicUsize>]>,
    /// The workers' view of this storage's tasks: shared with them, so
    /// apart from the stream's own (`&mut`) storage, and never freed.
    source: &'static crate::lanes::sources::Source,
}

/// Streams' storage not in use: recycled, never freed (a worker may read a
/// source's counters after its stream ended).
static FREE: std::sync::Mutex<std::vec::Vec<FreeStorage>> = std::sync::Mutex::new(std::vec::Vec::new());

struct FreeStorage(&'static mut Storage);

// Sound: a storage in the free list is no stream's; the one that takes it
// owns it alone.
unsafe impl Send for FreeStorage {}

impl Storage {
    /// A storage from the free list, or a new one (kept for good).
    fn take() -> &'static mut Storage {
        match FREE.lock().unwrap().pop() {
            Some(FreeStorage(s)) => s,
            None => Box::leak(Box::new(Storage::new())),
        }
    }

    fn new() -> Storage {
        Storage {
            source: Box::leak(Box::new(crate::lanes::sources::Source::new(ENTRIES))),
            entries: std::vec![Entry { len: 0, kind: Kind::Abandoned, first: 0, ring_end: 0 }; ENTRIES].into_boxed_slice(),
            // Zeroed, untouched pages: only the bytes used are faulted in.
            ring: Box::into_raw(std::vec![0u8; RING_BYTES].into_boxed_slice()) as *mut u8,
            pieces_out: (0..ENTRIES).map(|_| OwnLine(Out(UnsafeCell::new([0; 64])))).collect(),
            lefts: (0..ENTRIES).map(|_| OwnLine(AtomicUsize::new(0))).collect(),
        }
    }

    /// The ring's byte at `at`.
    fn ring_at(&self, at: usize) -> *mut u8 {
        debug_assert!(at <= RING_BYTES);
        // Sound: `at` is in the ring.
        unsafe { self.ring.add(at) }
    }
}


/// Part `i` of a message in `pieces` parts, its digest at `out`: the
/// message whole, or a piece.
fn part_kind(i: usize, pieces: usize, out: *mut Hash) -> Kind {
    if pieces == 1 { Kind::Message { out } } else { Kind::Piece { counter: (i * PIECE / CHUNK_LEN) as u64, last: i + 1 == pieces, out } }
}

/// The messages of one [`stream`] call, and the places for their digests,
/// borrowed for its scope; `D` is the stream's handler.
pub struct Stream<'env, D> {
    key: CVWords,
    flags: u8,
    /// Its own until the stream ends (then the thread's spare).
    s: &'static mut Storage,
    /// The source's registration, or None (this thread hashes every task).
    slot: Option<usize>,
    /// The handler that takes each place back.
    done: D,
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
    /// Invariant in 'env, as std::thread::Scope is: the messages' and the
    /// places' lifetime.
    env: PhantomData<&'env mut &'env ()>,
}

// Sound: the entries are reached by the program's thread through `&mut
// self`, and by the pool's threads only for entries in tasks in flight,
// across the Release/Acquire pair on each entry's `left`; the messages are
// shared borrows and the places exclusive ones, alive for 'env, each place
// written only by its task's thread before the Release and handed back
// only after the Acquire.
unsafe impl<D: Send> Send for Stream<'_, D> {}

impl<'env, D: FnMut(&'env mut [Hash])> Stream<'env, D> {
    /// Hash `message` where it is, its digest into `out`.
    pub fn submit(&mut self, message: &'env [u8], out: &'env mut Hash) {
        let out: *mut Hash = out;
        let pieces = message.len().div_ceil(PIECE).max(1);
        for i in 0..pieces {
            let part = &message[i * PIECE..][..(message.len() - i * PIECE).min(PIECE)];
            let at = self.head;
            self.add_part(part.as_ptr(), part.len(), part_kind(i, pieces, out), at, out as *mut u8);
        }
    }

    /// Hash a message of `len` bytes that `fill` writes in place, into the
    /// stream's own bytes, in pieces of up to 64 KiB, its digest into
    /// `out`: `fill(at, room)` writes the message's bytes from `at` into
    /// all of `room` (a read from a file or socket straight into the
    /// stream). The stream reuses the bytes once `out` is handed back; a
    /// stream keeps 4 MiB of them, and `submit_with` waits for room as
    /// `submit` does.
    pub fn submit_with(&mut self, len: usize, mut fill: impl FnMut(usize, &mut [u8]), out: &'env mut Hash) {
        let out: *mut Hash = out;
        let pieces = len.div_ceil(PIECE).max(1);
        for i in 0..pieces {
            let part_len = (len - i * PIECE).min(PIECE);
            let at = self.reserve(part_len);
            // Sound: [at, at + part_len) is reserved and free (no entry in
            // flight holds it), and `&mut self` keeps every other writer out.
            let room = unsafe { core::slice::from_raw_parts_mut(self.s.ring_at(at), part_len) };
            fill(i * PIECE, room);
            let end = self.head;
            self.add_part(room.as_ptr(), part_len, part_kind(i, pieces, out), end, out as *mut u8);
        }
    }

    /// Hash the message `reader` yields, to its end, read straight into the
    /// stream's own bytes in pieces of up to 64 KiB (a file or socket read
    /// into the stream, overlapping the hashing of earlier pieces), its
    /// digest into `out`: the message's length on success. On a read error
    /// (other than an interruption, which reads again) the error returns,
    /// `out` gets no digest and is not handed back, and the stream goes on
    /// with the next message. The length need not be known: the stream
    /// reads one piece ahead to tell the last.
    ///
    /// `size_hint` is the length the caller expects, where it knows one (a
    /// file's size, a protocol's length field): a scheduling hint, never
    /// part of the hash, so a wrong one costs nothing but speed. Today the
    /// stream does not use it; it is here so that a stream that schedules
    /// by it later changes no caller.
    pub fn submit_reader(&mut self, mut reader: impl std::io::Read, size_hint: Option<u64>, out: &'env mut Hash) -> std::io::Result<u64> {
        let _ = size_hint;
        let out: *mut Hash = out;
        let start = self.head;
        let (mut at, mut end, mut n, mut read) = self.read_piece(&mut reader);
        let mut i = 0;
        loop {
            let ahead = (n == PIECE && read.is_ok()).then(|| self.read_piece(&mut reader));
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
                    self.add_entry(0, Kind::Abandoned, head);
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
            let kind = if i == 0 && last { Kind::Message { out } } else { Kind::Piece { counter: (i * PIECE / CHUNK_LEN) as u64, last, out } };
            // [at, at + n): this piece's reserved room, written.
            let input = self.s.ring_at(at);
            self.add_part(input, n, kind, end, out as *mut u8);
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
    fn read_piece(&mut self, reader: &mut impl std::io::Read) -> (usize, usize, usize, std::io::Result<()>) {
        let at = self.reserve(PIECE);
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
    /// blocks, zero after its bytes), a digest per message into `out`, in
    /// the messages' order: `out` holds one per message. The batch goes in
    /// parts of up to 64 KiB, each one entry; `out` comes back whole.
    pub fn submit_batch(&mut self, batch: &'env [u8], message_len: usize, out: &'env mut [Hash]) {
        let slot = crate::many::slot_len(message_len);
        assert_eq!(Some(batch.len()), slot.checked_mul(out.len()), "a batch's digest space holds one digest per message");
        crate::many::assert_slots(batch, message_len, out.len());
        let whole = (out.as_mut_ptr(), out.len());
        let per_part = (TASK_LEN / slot).max(1);
        let parts = out.len().div_ceil(per_part).max(1);
        for p in 0..parts {
            let first = p * per_part;
            let count = (out.len() - first).min(per_part);
            let at = self.head;
            let kind = Kind::Batch { message_len, whole: (p + 1 == parts).then_some(whole) };
            // Sound: `first` is within the batch and its space (count may be 0).
            let (input, output) = unsafe { (batch.as_ptr().add(first * slot), whole.0.add(first)) };
            self.add_part(input, count * slot, kind, at, output as *mut u8);
        }
    }

    /// Hash a batch of `out.len()` messages of `message_len` bytes that
    /// `fill` writes in place, laid out as [`hash_many`](crate::hash_many)
    /// takes them: `fill(at, room)` writes the batch's bytes from `at`
    /// (padding included: each message in whole blocks, zero after its
    /// bytes) into all of `room`, a part of up to 64 KiB at a time. As
    /// [`submit_with`](Stream::submit_with) and [`submit_batch`](Stream::submit_batch).
    pub fn submit_batch_with(&mut self, message_len: usize, mut fill: impl FnMut(usize, &mut [u8]), out: &'env mut [Hash]) {
        let slot = crate::many::slot_len(message_len);
        let whole = (out.as_mut_ptr(), out.len());
        let per_part = (TASK_LEN / slot).max(1);
        let parts = out.len().div_ceil(per_part).max(1);
        for p in 0..parts {
            let first = p * per_part;
            let count = (out.len() - first).min(per_part);
            let at = self.reserve(count * slot);
            // Sound: as in submit_with.
            let room = unsafe { core::slice::from_raw_parts_mut(self.s.ring_at(at), count * slot) };
            fill(first * slot, room);
            crate::many::assert_slots(room, message_len, count);
            let end = self.head;
            let kind = Kind::Batch { message_len, whole: (p + 1 == parts).then_some(whole) };
            // Sound: `first` is within the batch's space.
            let output = unsafe { whole.0.add(first) };
            self.add_part(room.as_ptr(), room.len(), kind, end, output as *mut u8);
        }
    }

    /// An entry of `kind`, `len` bytes at `input`, its digests at `out`,
    /// its ring bytes reserved up to `ring_end`: a whole message, or a
    /// batch's part, joins the open task (of its kind: one of another kind
    /// is handed over first); a piece of a long message is a task of its
    /// own, a whole subtree at its offset; an empty batch's one part is no
    /// task (done as it is added).
    fn add_part(&mut self, input: *const u8, len: usize, kind: Kind, ring_end: usize, out: *mut u8) {
        let entry = self.add_entry(len, kind, ring_end);
        let batch = match kind {
            Kind::Message { .. } => None,
            Kind::Abandoned => unreachable!("an abandoned message adds no part"),
            Kind::Batch { .. } if len == 0 => return,
            Kind::Batch { message_len, .. } => Some(message_len),
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
                self.hand_over(task);
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
            open.left = &*self.s.lefts[entry];
        }
        self.s.entries[entry].first = self.open_first;
        open.member[open.members] = Member { input, len, out };
        open.members += 1;
        open.len += len;
        // Messages longer than a block hash one at a time: a task of them
        // stops at a quarter of a task's bytes, to spread over more threads
        // (spread over more threads); batches fill a task's bytes.
        if open.members == MEMBERS || open.len >= if batch.is_some() { TASK_LEN } else { TASK_LEN / 4 } {
            self.publish();
        }
    }

    /// The ring offset of `len` free bytes at a 64-byte boundary (whole,
    /// before the ring's end), reserved: waits, handing places back and
    /// hashing tasks, until entries in flight free them.
    fn reserve(&mut self, len: usize) -> usize {
        let slot = len.div_ceil(64) * 64;
        let mut at = self.head % RING_BYTES;
        // A slot that would cross the end starts the next lap instead.
        let skip = if at + slot > RING_BYTES { RING_BYTES - at } else { 0 };
        while RING_BYTES - (self.head - self.tail) < skip + slot {
            self.publish();
            self.help();
        }
        if skip > 0 {
            at = 0;
        }
        self.head += skip + slot;
        at
    }

    /// Hand back every place whose digests are done, in order, without
    /// waiting; and hand the open task to the pool, so what it holds
    /// starts hashing.
    pub fn collect(&mut self) {
        self.publish();
        self.deliver();
    }

    /// Hand back every place, in order, once its digests are done: this
    /// thread hashes any task no thread has taken, and waits for the rest.
    pub fn finish(&mut self) {
        self.publish();
        while self.delivered < self.next {
            self.help();
        }
    }

    /// Hand back what is done; with nothing done, hash a waiting task or
    /// pause.
    fn help(&mut self) {
        if self.deliver() == 0 && !self.run_waiting() {
            core::hint::spin_loop();
        }
    }

    /// Hash one of the stream's tasks no thread has taken: whether there
    /// was one.
    fn run_waiting(&mut self) -> bool {
        match self.s.source.take_task() {
            Some(task) => {
                crate::lanes::run_task(task);
                true
            }
            None => false,
        }
    }

    /// A new entry of `len` bytes, waiting for room when every entry is in
    /// flight.
    #[inline(always)]
    fn add_entry(&mut self, len: usize, kind: Kind, ring_end: usize) -> usize {
        if self.next - self.delivered == ENTRIES {
            self.wait_for_room();
        }
        let index = self.next % ENTRIES;
        self.s.entries[index] = Entry { len, kind, first: index, ring_end };
        self.next += 1;
        index
    }

    /// Hand back places, and hash tasks, until an entry is free (out of
    /// line: a frame of its own, set up only when the stream is full).
    #[cold]
    #[inline(never)]
    fn wait_for_room(&mut self) {
        self.publish();
        while self.next - self.delivered == ENTRIES {
            self.help();
        }
    }

    /// Hand the open task to the pool.
    fn publish(&mut self) {
        if let Some(task) = self.open.take() {
            // Before the push, whose lock orders it before the threads' reads.
            self.s.lefts[self.open_first].store(task.members, Ordering::Relaxed);
            self.hand_over(task);
        }
    }

    /// Give `task` to the workers through the stream's source; with no
    /// slot, hash it on this thread now.
    fn hand_over(&mut self, task: Task) {
        match self.slot {
            Some(_) => self.s.source.publish(task),
            None => crate::lanes::run_task(task),
        }
    }

    /// Hand back every finished entry's place from the oldest on (a
    /// batch's once its last part is done); how many entries.
    fn deliver(&mut self) -> usize {
        let mut count = 0;
        while self.delivered < self.next {
            let index = self.delivered % ENTRIES;
            let Entry { len, kind, first, ring_end } = self.s.entries[index];
            // An entry of no task (an empty batch's part, an abandoned
            // message) has its own `left`, zero.
            if self.s.lefts[first].load(Ordering::Acquire) != 0 {
                break;
            }
            self.delivered += 1;
            self.tail = ring_end;
            count += 1;
            // Sound, each place: the program lent it for 'env and the stream
            // hands it back once, after its task wrote it (the Acquire above).
            match kind {
                Kind::Message { out } => (self.done)(core::slice::from_mut(unsafe { &mut *out })),
                Kind::Batch { whole: Some((out, n)), .. } => (self.done)(unsafe { core::slice::from_raw_parts_mut(out, n) }),
                Kind::Batch { whole: None, .. } => {}
                Kind::Abandoned => {
                    self.stack.clear();
                    self.pieces = 0;
                }
                Kind::Piece { counter, last, out } => {
                    // Sound: the task wrote its out before counting `left` down.
                    let piece = unsafe { *self.s.pieces_out[index].0.0.get() };
                    let (key, flags, platform) = (self.key, self.flags, Platform::detect());
                    let parent = |left: &CVBytes, right: &CVBytes| crate::hasher::parent_output(left, right, &key, flags, platform);
                    let cv: CVBytes = if counter == 0 && len > CHUNK_LEN {
                        parent(piece[..OUT_LEN].try_into().unwrap(), piece[OUT_LEN..].try_into().unwrap()).chaining_value()
                    } else {
                        piece[..OUT_LEN].try_into().unwrap()
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
                        let place = unsafe { &mut *out };
                        *place = parent(&self.stack.pop().unwrap(), &right).root_hash();
                        self.pieces = 0;
                        (self.done)(core::slice::from_mut(place));
                    }
                }
            }
        }
        count
    }
}

impl<D> Drop for Stream<'_, D> {
    /// Wait for every task still reading the messages and writing the
    /// places (also when the body panicked), then keep the storage as the
    /// thread's spare. A stream lives only inside [`stream`], which owns
    /// it, so this runs before the scope ends.
    fn drop(&mut self) {
        if let Some(task) = self.open.take() {
            // Handed over, so that the wait below ends: as publish.
            self.s.lefts[self.open_first].store(task.members, Ordering::Relaxed);
            match self.slot {
                Some(_) => self.s.source.publish(task),
                None => crate::lanes::run_task(task),
            }
        }
        for i in self.delivered..self.next {
            while self.s.lefts[self.s.entries[i % ENTRIES].first].load(Ordering::Acquire) != 0 {
                let ran = match self.s.source.take_task() {
                    Some(task) => {
                        crate::lanes::run_task(task);
                        true
                    }
                    None => false,
                };
                if !ran {
                    core::hint::spin_loop();
                }
            }
        }
        if let Some(slot) = self.slot {
            crate::lanes::sources::Source::unregister(slot);
        }
        crate::lanes::sources::closed();
        // Sound: `s` is not used again; the next stream takes it whole.
        let s: &'static mut Storage = unsafe { &mut *(self.s as *mut Storage) };
        FREE.lock().unwrap().push(FreeStorage(s));
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Places for `n` digests, each starting as zeros.
    fn places(n: usize) -> std::vec::Vec<Hash> {
        std::vec![Hash::from_bytes([0; 32]); n]
    }

    /// The next `n` places of `rest`, split off its front.
    fn take<'a>(rest: &mut &'a mut [Hash], n: usize) -> &'a mut [Hash] {
        let (now, after) = core::mem::take(rest).split_at_mut(n);
        *rest = after;
        now
    }

    /// Where each place handed back starts, and its length.
    fn record(handed: &mut std::vec::Vec<(usize, usize)>) -> impl FnMut(&mut [Hash]) + '_ {
        |d: &mut [Hash]| handed.push((d.as_ptr() as usize, d.len()))
    }

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
            let mut got = places(lens.len());
            let want_handed: std::vec::Vec<(usize, usize)> = got.iter().map(|h| (h as *const Hash as usize, 1)).collect();
            let mut handed = std::vec::Vec::new();
            stream(mode, record(&mut handed), |s| {
                for (i, out) in got.iter_mut().enumerate() {
                    // Borrowed, written in place (in reads of up to 1000
                    // bytes), or each in turn.
                    if ways == 0 || (ways == 2 && i % 3 == 0) {
                        s.submit(message(i), out);
                    } else {
                        s.submit_with(
                            lens[i],
                            |at, room| {
                                for (j, part) in room.chunks_mut(1000).enumerate() {
                                    part.copy_from_slice(&message(i)[at + j * 1000..][..part.len()]);
                                }
                            },
                            out,
                        );
                    }
                    if i % 500 == 499 {
                        s.collect();
                    }
                }
                s.finish();
            });
            assert!(handed == want_handed, "every place handed back once, in order");
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
            let total: usize = cases.iter().map(|&(_, count)| count + 1).sum();
            let mut got = places(total);
            let mut want = std::vec::Vec::new();
            let mut want_handed = std::vec::Vec::new();
            let mut handed = std::vec::Vec::new();
            let mut rest: &mut [Hash] = &mut got;
            stream(mode, record(&mut handed), |s| {
                for (k, (&(len, count), batch)) in cases.iter().zip(&batches).enumerate() {
                    let out = take(&mut rest, count);
                    want_handed.push((out.as_ptr() as usize, count));
                    if k % 2 == 0 {
                        s.submit_batch(batch, len, out);
                    } else {
                        s.submit_batch_with(len, |at, room| room.copy_from_slice(&batch[at..][..room.len()]), out);
                    }
                    let slot = crate::many::slot_len(len);
                    want.extend(batch.chunks(slot).map(|m| crate::hash_with(mode, &m[..len])));
                    let out = &mut take(&mut rest, 1)[0];
                    want_handed.push((out as *const Hash as usize, 1));
                    s.submit(&lone, out);
                    want.push(crate::hash_with(mode, &lone));
                }
                s.finish();
            });
            assert!(handed == want_handed, "every batch's space handed back once, whole, in order");
            assert!(got == want, "a batch's digests in its space, in order");
        }
    }

    /// A stream takes more messages after finishing; a body that ends
    /// without finishing panics (after the threads are done with its
    /// messages), and so does one that panics itself.
    #[test]
    fn test_stream_finish_and_panics() {
        let input = std::vec![9u8; 300_000];
        let mut got = places(9);
        let handed = core::cell::Cell::new(0);
        stream(Mode::Hash, |d: &mut [Hash]| handed.set(handed.get() + d.len()), |s| {
            for (round, outs) in got.chunks_mut(3).enumerate() {
                for (len, out) in [0, 65, 70_000].into_iter().zip(outs) {
                    s.submit(&input[..len], out);
                }
                s.finish();
                assert_eq!(handed.get(), 3 * (round + 1));
            }
        });
        assert_eq!(got[5], crate::hash(&input[..70_000]));
        for unwinding in [false, true] {
            let result = std::panic::catch_unwind(|| {
                let mut got = places(4);
                stream(Mode::Hash, |_: &mut [Hash]| {}, |s| {
                    for (len, out) in [10, 100_000, 64, 300_000].into_iter().zip(&mut got) {
                        s.submit(&input[..len], out);
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
        let fails = [0, 500, PIECE, PIECE + 3, 2 * PIECE, 3 * PIECE + 1];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("stream readers")] {
            let mut got = places(3 * 2 * lens.len() + 2 * fails.len());
            let (mut want, mut want_handed, mut handed) = (std::vec::Vec::new(), std::vec::Vec::new(), std::vec::Vec::new());
            let mut rest: &mut [Hash] = &mut got;
            stream(mode, record(&mut handed), |s| {
                for _ in 0..3 {
                    for &len in &lens {
                        let out = &mut take(&mut rest, 1)[0];
                        want_handed.push((out as *const Hash as usize, 1));
                        let n = s.submit_reader(Uneven { data: &input[..len], at: 0, calls: 0, fail_at: None }, None, out).unwrap();
                        assert_eq!(n, len as u64);
                        want.push(crate::hash_with(mode, &input[..len]));
                        let out = &mut take(&mut rest, 1)[0];
                        want_handed.push((out as *const Hash as usize, 1));
                        s.submit(&input[..100], out);
                        want.push(crate::hash_with(mode, &input[..100]));
                    }
                }
                for fail_at in fails {
                    let r = s.submit_reader(Uneven { data: &input, at: 0, calls: 0, fail_at: Some(fail_at) }, None, &mut take(&mut rest, 1)[0]);
                    assert!(r.is_err(), "a failing reader's error comes back (failing at {fail_at})");
                    want.push(Hash::from_bytes([0; 32]));
                    let out = &mut take(&mut rest, 1)[0];
                    want_handed.push((out as *const Hash as usize, 1));
                    s.submit(&input[..3 * PIECE + 5], out);
                    want.push(crate::hash_with(mode, &input[..3 * PIECE + 5]));
                }
                s.finish();
            });
            assert!(handed == want_handed, "every place handed back once, in order, none for the failed readers");
            assert!(got == want, "readers' digests in their places; the failed readers' places untouched");
        }
    }

    /// A few messages through a stream, small enough for Miri: the handover
    /// of entries to the pool's threads and back, checked for data races.
    #[test]
    fn test_miri_stream() {
        let input = std::vec![3u8; 3 * PIECE + 5];
        let lens = [0, 64, 64, 1000, PIECE + 1, 3 * PIECE + 5];
        let mut got = places(2 * lens.len() + 2 + 7);
        let mut handed = std::vec::Vec::new();
        let mut rest: &mut [Hash] = &mut got;
        stream(Mode::Hash, record(&mut handed), |s| {
            for len in lens {
                s.submit(&input[..len], &mut take(&mut rest, 1)[0]);
                s.submit_with(len, |at, room| room.copy_from_slice(&input[at..][..room.len()]), &mut take(&mut rest, 1)[0]);
            }
            assert_eq!(s.submit_reader(&input[..2 * PIECE + 1], None, &mut take(&mut rest, 1)[0]).unwrap(), 2 * PIECE as u64 + 1);
            assert!(s.submit_reader(Uneven { data: &input, at: 0, calls: 0, fail_at: Some(PIECE + 1) }, None, &mut take(&mut rest, 1)[0]).is_err());
            s.submit_batch(&input[..4 * 64], 64, take(&mut rest, 4));
            s.submit_batch_with(64, |at, room| room.copy_from_slice(&input[at..][..room.len()]), take(&mut rest, 3));
            s.finish();
        });
        let mut want: std::vec::Vec<Hash> = lens.iter().flat_map(|&l| [crate::hash(&input[..l]); 2]).collect();
        want.push(crate::hash(&input[..2 * PIECE + 1]));
        want.push(Hash::from_bytes([0; 32]));
        want.extend([crate::hash(&input[..64]); 7]);
        assert_eq!(got, want);
        assert_eq!(handed.len(), 2 * lens.len() + 1 + 2, "a place back per submission, none for the failed reader");
    }
}
