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
//! every task before it returns; a panic inside ends the process
//! (lanes::AbortOnUnwind, as for a one-shot job): no thread reads a message
//! after the scope ends.

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

/// The bytes a stream keeps for the pieces of long messages written in
/// place (LONG or more): 8 MiB, 128 pieces in flight, so the program
/// writing them runs ahead of the threads hashing them. In the 2 MiB ring,
/// 32 pieces in flight, the program waits for the oldest piece to come
/// back: streams of 4 and 16 MiB messages took 0.71-0.82 of the time with
/// this ring (Apple M4 Max, job 1896, three rounds). 64 MiB did no better
/// alone, and with two of a program's threads streaming at once its
/// footprint cost what the depth gave (4 and 16 MiB messages, 16 a stream:
/// 1.01-1.02 of the 2 MiB ring's time, where 8 MiB took 0.93; Mac check
/// 1895: shared-os continuous 4 MiB +19.3%).
const LONG_RING_BYTES: usize = 8 << 20;

/// The shortest message whose pieces go to the long ring: 4 MiB, past the
/// caches where the 2 MiB ring is kept hot.
const LONG: usize = 4 << 20;

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
    /// the message's last piece, whether it lies in the long ring, and
    /// where the message's digest goes. (The long ring's pieces come back
    /// in the order they were reserved, so handing one back frees its room
    /// by reserving it again from the tail: no end kept, so an entry stays
    /// 56 bytes. With the end kept, 64 bytes made multithreaded streams of
    /// 512 B messages 3.6% slower, Mac check 1893.)
    Piece {
        counter: u64,
        last: bool,
        long: bool,
        /// In the GPU ring, not the long ring (where the GPU's code is built).
        #[cfg(blake3_gpu)]
        gpu: bool,
        out: *mut Hash,
    },
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
/// returns only after they are done.
///
/// **A panic inside a stream ends the process**, wherever it starts: in
/// `body`, in `done`, in a `fill` or a reader, or in the stream's own
/// checks (a `body` that ends without `finish`). Its message and backtrace
/// print first, as for any panic; then the process aborts, since this
/// crate's threads may still be reading the messages and writing the
/// places. So `std::panic::catch_unwind` around a stream recovers nothing.
/// A program that must go on past a failing read hashes it with
/// [`Stream::submit_reader`], which returns the read's error and goes on
/// with the next message. Under plain `cargo test`, which runs a test
/// binary's tests as threads of one process, a failing assertion inside a
/// stream ends the whole binary, so the tests after it report nothing;
/// cargo-nextest runs each test in a process of its own.
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
/// **Streams, then other calls.** While a stream is open, this crate's
/// threads keep looking for its next message for a few microseconds before
/// they sleep, which makes streams of messages 3-5% faster. A program that
/// streams and then soon makes a one-shot multithreaded call pays for it:
/// on an i7-12700K under Linux, [`crate::throughput::hash_many_multithreaded`] on 16384
/// messages right after streams of 16-64 KiB messages ran about 5% slower.
/// A program that keeps streaming does not pay it; one that alternates can
/// stream its other work
/// too ([`Stream::submit_batch`] for a batch, a stream per round with
/// [`finish`](Stream::finish) between rounds that depend on each other).
///
/// ```
/// # blake3_servil::initialize_multithreaded(None);
/// use blake3_servil::{Hash, Mode, throughput::Block};
/// // Each message 64 bytes in one block, as the stream borrows messages.
/// let messages: Vec<Block> = (0..1000u32).map(|i| Block(i.to_le_bytes().repeat(16).try_into().unwrap())).collect();
/// let mut digests = vec![Hash::from_bytes([0; 32]); messages.len()];
/// let mut handed_back = 0;
/// blake3_servil::throughput::stream(Mode::Plain, |done: &mut [Hash]| handed_back += done.len(), |s| {
///     for (message, digest) in messages.iter().zip(&mut digests) {
///         s.submit(core::slice::from_ref(message), 64, digest);
///     }
///     s.finish();
/// });
/// assert_eq!(handed_back, 1000);
/// assert_eq!(digests[7], blake3_servil::convenience::hash(&messages[7].0));
/// ```
pub fn stream<'env, D: FnMut(&'env mut [Hash]), T>(mode: Mode, done: D, body: impl for<'s> FnOnce(&'s mut Stream<'env, D>) -> T) -> T {
    let mut stream = Stream::open(mode, done);
    // Dropped first if anything below unwinds: the process ends there, while
    // the threads may still read the messages and write the places.
    let unwinding = crate::lanes::AbortOnUnwind;
    let result = body(&mut stream);
    assert_eq!(stream.delivered, stream.next, "a stream's body calls finish last");
    core::mem::forget(unwinding);
    result
}

impl<'env, D: FnMut(&'env mut [Hash])> Stream<'env, D> {
    /// A stream outside [`stream`]'s scope, for the C API (`capi/`): the
    /// same contract, with the program holding the stream and ending it
    /// (`finish`, then drop). Dropping a stream waits for its tasks. Rust
    /// programs call [`stream`], whose scope holds the contract for them.
    #[doc(hidden)]
    pub fn open(mode: Mode, done: D) -> Self {
        crate::lanes::require();
        let (key, flags) = mode.key_and_flags();
        let s = Storage::take();
        // A slot, where some thread beside this one can take tasks (with one
        // CPU none can: then every task is hashed here as it is handed over,
        // as when every slot is taken).
        let slot = if crate::lanes::splits() { s.source.register() } else { None };
        crate::lanes::sources::opened();
        Stream {
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
            long_head: 0,
            long_tail: 0,
            #[cfg(blake3_gpu)]
            gpu_head: 0,
            #[cfg(blake3_gpu)]
            gpu_tail: 0,
            #[cfg(blake3_gpu)]
            gpu_run: 0..0,
            open: None,
            stack: arrayvec::ArrayVec::new(),
            pieces: 0,
            env: PhantomData,
        }
    }
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
    /// LONG_RING_BYTES for the pieces of long messages written in place
    /// (`Stream::submit_long`), reached as the ring is.
    long_ring: *mut u8,
    /// The storage's GPU ring and commands, made when its first long
    /// message finds the GPU engaged, kept with it.
    #[cfg(blake3_gpu)]
    gpu: Option<crate::gpu::Gpu>,
    /// The storage's run of long messages, which engages the GPU.
    #[cfg(blake3_gpu)]
    run: crate::gpu::Run,
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
            ring: ring(RING_BYTES),
            long_ring: ring(LONG_RING_BYTES),
            #[cfg(blake3_gpu)]
            gpu: None,
            #[cfg(blake3_gpu)]
            run: crate::gpu::Run::new(),
            pieces_out: (0..ENTRIES).map(|_| OwnLine(Out(UnsafeCell::new([0; 64])))).collect(),
            lefts: (0..ENTRIES).map(|_| OwnLine(AtomicUsize::new(0))).collect(),
        }
    }

    /// Hash one of the stream's tasks no thread has taken: whether there
    /// was one. A waiting caller's own help, the one way a lone task runs
    /// while every worker is parked (only collect wakes one for it).
    fn run_waiting(&self) -> bool {
        match self.source.take_task() {
            Some(task) => {
                crate::lanes::run_task(task);
                true
            }
            None => false,
        }
    }

    /// The ring's byte at `at`.
    fn ring_at(&self, at: usize) -> *mut u8 {
        debug_assert!(at <= RING_BYTES);
        // Sound: `at` is in the ring.
        unsafe { self.ring.add(at) }
    }

    /// The long ring's byte at `at`.
    fn long_at(&self, at: usize) -> *mut u8 {
        debug_assert!(at <= LONG_RING_BYTES);
        // Sound: `at` is in the long ring.
        unsafe { self.long_ring.add(at) }
    }
}

/// A ring of `len` bytes, kept for good, as the storage is: zeroed,
/// untouched pages (only the bytes used are faulted in: calloc maps them
/// lazily, where an aligned allocation would write them all), starting at
/// the first 64-byte boundary, as each room's offset is, so every room
/// starts a cache line (glibc puts a 2 MiB block 16 bytes into one).
fn ring(len: usize) -> *mut u8 {
    let block = Box::leak(std::vec![0u8; len + 64].into_boxed_slice());
    let start = block.as_ptr().align_offset(64);
    block[start..].as_mut_ptr()
}

/// Whether a ring of `ring` bytes, `head - tail` of them in flight (both
/// counting on), has `taken` bytes free: each ring's wait for room, here
/// once, so a test holds it to exact values (Sam's whole x86-64 run, #218:
/// the long ring's own copy of it, `-` as `+`, survived, as the ring never
/// filled under the tests).
fn has_room(head: usize, tail: usize, ring: usize, taken: usize) -> bool {
    ring - (head - tail) >= taken
}

/// Where `len` bytes go in a ring of `ring` bytes whose head is at `head`
/// (counting on), at a 64-byte boundary, whole before its end: their
/// offset, and the bytes they take from the head (a room that would cross
/// the end starts the next lap, the rest of this one skipped).
fn room(head: usize, ring: usize, len: usize) -> (usize, usize) {
    let slot = len.div_ceil(64) * 64;
    let at = head % ring;
    if at + slot > ring { (0, ring - at + slot) } else { (at, slot) }
}

/// `fill` called as `Read::read_exact` calls `read`, until `room` is full:
/// each call writes at the start of what is left, the bytes from `at` on,
/// and returns how many it wrote. A short write (a plain read's) would
/// otherwise leave bytes an earlier message left, hashed into a wrong
/// digest with no error (BLAKE3 #204).
fn fill_all(fill: &mut impl FnMut(usize, &mut [u8]) -> usize, at: usize, room: &mut [u8]) {
    let mut done = 0;
    while done < room.len() {
        let left = room.len() - done;
        let n = fill(at + done, &mut room[done..]);
        assert!(n != 0, "fill wrote nothing with {left} of its {} bytes left: its source ended before the message's length", room.len());
        assert!(n <= left, "fill says it wrote {n} bytes into a room of {left}");
        done += n;
    }
}

/// Part `i` of a message in `pieces` parts, its digest at `out`: the
/// message whole, or a piece.
fn part_kind(i: usize, pieces: usize, out: *mut Hash, long: bool) -> Kind {
    if pieces == 1 {
        Kind::Message { out }
    } else {
        Kind::Piece {
            counter: (i * PIECE / CHUNK_LEN) as u64,
            last: i + 1 == pieces,
            long,
            #[cfg(blake3_gpu)]
            gpu: false,
            out,
        }
    }
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
    /// The long ring's bytes reserved, and freed, as `head` and `tail`.
    long_head: usize,
    long_tail: usize,
    /// The GPU ring's bytes reserved, and freed, as `head` and `tail`.
    #[cfg(blake3_gpu)]
    gpu_head: usize,
    #[cfg(blake3_gpu)]
    gpu_tail: usize,
    /// The positions in the source of the latest run of tasks the GPU may
    /// take: whole pieces in its ring, published one after another.
    #[cfg(blake3_gpu)]
    gpu_run: core::ops::Range<usize>,
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
    /// Hash the message of `len` bytes that starts `message`, where it is,
    /// its digest into `out`: a [`Block`](crate::throughput::Block)-aligned
    /// message, as [`hash`](crate::throughput::hash) takes one (the bytes
    /// past its end ignored).
    pub fn submit(&mut self, message: &'env [crate::throughput::Block], len: usize, out: &'env mut Hash) {
        let message: &'env [u8] = &crate::throughput::Block::bytes(message)[..len];
        let out: *mut Hash = out;
        let pieces = message.len().div_ceil(PIECE).max(1);
        for i in 0..pieces {
            let part = &message[i * PIECE..][..(message.len() - i * PIECE).min(PIECE)];
            let at = self.head;
            self.add_part(part.as_ptr(), part.len(), part_kind(i, pieces, out, false), at, out as *mut u8);
        }
    }

    /// Hash a message of `len` bytes that `fill` writes in place, into the
    /// stream's own bytes, in pieces of up to 64 KiB, its digest into
    /// `out`. `fill(at, room)` writes the message's bytes from `at` at the
    /// start of `room` and returns how many it wrote, as `Read::read`
    /// does: the stream calls it again for the rest until the piece is
    /// full, so `|_, room| file.read(room).unwrap()` reads a file or socket
    /// straight into the stream. A `fill` that returns 0 with bytes still
    /// to write stops the program (its source ended before `len`). Each
    /// piece's first `room` starts on a 64-byte boundary, and its bytes on
    /// entry are unspecified: the rooms are the process's, reused across
    /// streams, so a room may hold bytes an earlier message left (a program
    /// that must keep one caller's secrets from another reads none of
    /// `room` before writing it). The stream reuses the bytes once `out`
    /// is handed back; a stream keeps 2 MiB of them, and 8 MiB more for
    /// messages of 4 MiB or more, and `submit_with` waits for room as
    /// `submit` does. On an Apple silicon Mac, a run of messages of 4 MiB or
    /// more is shared with the GPU (the crate's "Which call to use"); their
    /// rooms are then in 64 MiB of memory the GPU shares, on the same terms.
    pub fn submit_with(&mut self, len: usize, mut fill: impl FnMut(usize, &mut [u8]) -> usize, out: &'env mut Hash) {
        let out: *mut Hash = out;
        if len >= LONG {
            return self.submit_long(len, &mut fill, out);
        }
        let pieces = len.div_ceil(PIECE).max(1);
        for i in 0..pieces {
            let part_len = (len - i * PIECE).min(PIECE);
            let at = self.reserve(part_len);
            // Sound: [at, at + part_len) is reserved and free (no entry in
            // flight holds it), and `&mut self` keeps every other writer out.
            let room = unsafe { core::slice::from_raw_parts_mut(self.s.ring_at(at), part_len) };
            fill_all(&mut fill, i * PIECE, room);
            let end = self.head;
            self.add_part(room.as_ptr(), part_len, part_kind(i, pieces, out, false), end, out as *mut u8);
        }
    }

    /// `submit_with` for a long message (LONG bytes or more): its pieces in
    /// the long ring (out of line: the short messages' path keeps only the
    /// length's test).
    #[inline(never)]
    fn submit_long(&mut self, len: usize, fill: &mut impl FnMut(usize, &mut [u8]) -> usize, out: *mut Hash) {
        // With the GPU engaged, the GPU ring, whose whole pieces the GPU may
        // take (crate::gpu); a pool asked for one thread has no slot, and
        // keeps to its CPU core (AGENTS: the GPU wherever it is efficient).
        #[cfg(blake3_gpu)]
        let gpu = self.slot.is_some() && self.s.run.engaged() && crate::gpu::ready();
        #[cfg(blake3_gpu)]
        let gpu_ring = if gpu { self.s.gpu.get_or_insert_with(|| crate::gpu::Gpu::new(crate::gpu::RING_BYTES)).ring } else { core::ptr::null_mut() };
        let pieces = len.div_ceil(PIECE);
        for i in 0..pieces {
            let part_len = (len - i * PIECE).min(PIECE);
            #[cfg(blake3_gpu)]
            let room = if gpu {
                let at = self.reserve_gpu(part_len);
                // Sound: as below, in the GPU ring.
                unsafe { core::slice::from_raw_parts_mut(gpu_ring.add(at), part_len) }
            } else {
                let at = self.reserve_long(part_len);
                // Sound: as below.
                unsafe { core::slice::from_raw_parts_mut(self.s.long_at(at), part_len) }
            };
            #[cfg(not(blake3_gpu))]
            let room = {
                let at = self.reserve_long(part_len);
                // Sound: [at, at + part_len) is reserved and free in the long
                // ring (no entry in flight holds it), and `&mut self` keeps
                // every other writer out.
                unsafe { core::slice::from_raw_parts_mut(self.s.long_at(at), part_len) }
            };
            fill_all(fill, i * PIECE, room);
            // No bytes of the 2 MiB ring: its head is the piece's end there.
            // Not through add_part: one more caller there, and the compiler
            // stopped inlining it into submit_with (multithreaded 64 B and
            // 256 B continuous messages 3.4-4.1% slower, the VM's check).
            let end = self.head;
            #[allow(unused_mut)]
            let mut kind = part_kind(i, pieces, out, true);
            #[cfg(blake3_gpu)]
            if let Kind::Piece { gpu: in_gpu, .. } = &mut kind {
                *in_gpu = gpu;
            }
            let entry = self.add_entry(part_len, kind, end);
            self.hand_over_piece(entry, room.as_ptr(), part_len, (i * PIECE / CHUNK_LEN) as u64);
            #[cfg(blake3_gpu)]
            if gpu {
                if part_len == PIECE {
                    // Published last, at position p - 1: the run of tasks the
                    // GPU may take goes on, or starts there.
                    let p = self.s.source.published();
                    if self.gpu_run.end != p - 1 {
                        self.gpu_run.start = p - 1;
                    }
                    self.gpu_run.end = p;
                }
                self.feed();
            }
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
            let kind = if i == 0 && last {
                Kind::Message { out }
            } else {
                Kind::Piece {
                    counter: (i * PIECE / CHUNK_LEN) as u64,
                    last,
                    long: false,
                    #[cfg(blake3_gpu)]
                    gpu: false,
                    out,
                }
            };
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

    /// Hash a [`Batch`](crate::throughput::Batch) where it is, as
    /// [`hash_many`](crate::throughput::hash_many) takes it, a digest per
    /// message into `out`, in the messages' order: `out` holds one per
    /// message. The batch goes in parts of up to 64 KiB, each one entry;
    /// `out` comes back whole.
    pub fn submit_batch(&mut self, batch: &'env crate::throughput::Batch<'_>, out: &'env mut [Hash]) {
        let message_len = batch.message_len();
        let batch = crate::throughput::Block::bytes(batch.blocks());
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
    /// `fill` writes in place, laid out as [`hash_many`](crate::throughput::hash_many)
    /// takes them: `fill(at, room)` writes the batch's bytes from `at` at
    /// the start of `room` and returns how many it wrote, as in
    /// [`submit_with`](Stream::submit_with), the stream calling it until a
    /// part of up to 64 KiB is full, each message at the start of its slot
    /// of whole blocks. The bytes past each message are the
    /// stream's: whatever `fill` leaves there, the stream zeroes them before
    /// hashing, so a room reused for shorter messages hashes right. As
    /// [`submit_with`](Stream::submit_with) and [`submit_batch`](Stream::submit_batch).
    pub fn submit_batch_with(&mut self, message_len: usize, mut fill: impl FnMut(usize, &mut [u8]) -> usize, out: &'env mut [Hash]) {
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
            fill_all(&mut fill, first * slot, room);
            // The bytes past each message, zero (none when the length is a
            // multiple of 64): a reused room holds an earlier batch's bytes.
            if message_len % 64 != 0 || message_len == 0 {
                for message in room.chunks_exact_mut(slot) {
                    message[message_len..].fill(0);
                    crate::counted(crate::Counted::BatchZeroed);
                }
            }
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
                self.hand_over_piece(entry, input, len, counter);
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

    /// Entry `entry`, a piece of a long message, `len` bytes at `input`
    /// from chunk `counter`: a task of its own, a whole subtree at its
    /// offset, handed over.
    #[inline(always)]
    fn hand_over_piece(&mut self, entry: usize, input: *const u8, len: usize, counter: u64) {
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
    }

    /// The ring offset of `len` free bytes at a 64-byte boundary (whole,
    /// before the ring's end), reserved: waits, handing places back and
    /// hashing tasks, until entries in flight free them.
    fn reserve(&mut self, len: usize) -> usize {
        let (at, taken) = room(self.head, RING_BYTES, len);
        while !has_room(self.head, self.tail, RING_BYTES, taken) {
            self.publish();
            self.help();
        }
        self.head += taken;
        at
    }

    /// `reserve`, in the GPU ring.
    #[cfg(blake3_gpu)]
    fn reserve_gpu(&mut self, len: usize) -> usize {
        let (at, taken) = room(self.gpu_head, crate::gpu::RING_BYTES, len);
        while !has_room(self.gpu_head, self.gpu_tail, crate::gpu::RING_BYTES, taken) {
            self.publish();
            self.help();
        }
        self.gpu_head += taken;
        at
    }

    /// `reserve`, in the long ring.
    fn reserve_long(&mut self, len: usize) -> usize {
        let (at, taken) = room(self.long_head, LONG_RING_BYTES, len);
        while !has_room(self.long_head, self.long_tail, LONG_RING_BYTES, taken) {
            self.publish();
            self.help();
        }
        self.long_head += taken;
        at
    }

    /// Hand back every place whose digests are done, in order, without
    /// waiting; and hand the open task to the pool, so what it holds
    /// starts hashing.
    pub fn collect(&mut self) {
        self.publish();
        // Nothing back: a worker for any task still waiting (the caller
        // takes none here; in steady flow the workers are awake and this
        // costs a comparison). Workers spin 5 us for a stream's next task
        // and then park, and publish wakes one only for two waiting tasks,
        // so without this a caller looping on collect waited forever where
        // no SME2 thread takes stream tasks (tests/stream_collect_parked_workers).
        if self.deliver() == 0 && self.slot.is_some() {
            self.s.source.wake_for_waiting();
        }
    }

    /// Hand back every place, in order, once its digests are done: this
    /// thread hashes any task no thread has taken, and waits for the rest.
    pub fn finish(&mut self) {
        self.publish();
        while self.delivered < self.next {
            // No new GPU batches at the end; and with nothing else to do,
            // the GPU's tasks in flight back to the pool (crate::gpu).
            if self.deliver() == 0 && !self.s.run_waiting() && !self.reclaim_at_finish() {
                core::hint::spin_loop();
            }
        }
    }

    /// `reclaim`, where the GPU's code is built.
    #[inline(always)]
    fn reclaim_at_finish(&mut self) -> bool {
        // A load a command first: the finish's loop asks on every pass, and
        // reclaim's frame holds room for every task the commands may hold.
        #[cfg(blake3_gpu)]
        if self.s.gpu.as_ref().is_some_and(|gpu| gpu.in_flight()) {
            return self.reclaim();
        }
        false
    }

    /// Hand back what is done; with nothing done, hash a waiting task or
    /// pause.
    fn help(&mut self) {
        if self.deliver() == 0 && !self.feed() && !self.s.run_waiting() {
            core::hint::spin_loop();
        }
    }

    /// Give the GPU a batch of the stream's tasks if it may take one now
    /// (`crate::gpu::feed`): whether it did. Nothing where the GPU's code
    /// is not built.
    #[inline(always)]
    fn feed(&mut self) -> bool {
        // First the run's length, the stream's own: a stream without a
        // batch's run of GPU pieces (every stream of short messages) asks
        // nothing of the GPU's commands or the source.
        #[cfg(blake3_gpu)]
        if self.gpu_run.end - self.gpu_run.start >= crate::gpu::BATCH {
            if let Some(gpu) = self.s.gpu.as_ref() {
                return crate::gpu::feed(gpu, self.s.source, self.gpu_run.start, self.gpu_run.end, &self.key, self.flags);
            }
        }
        false
    }

    /// The GPU's tasks still in flight, taken back and published again for
    /// the pool's workers (`crate::gpu::Gpu::reclaim`): whether there were
    /// any. Out of line, as finish's last resort.
    #[cfg(blake3_gpu)]
    #[cold]
    #[inline(never)]
    fn reclaim(&mut self) -> bool {
        let Some(gpu) = self.s.gpu.as_ref() else { return false };
        let mut back = arrayvec::ArrayVec::<crate::gpu::Held, { crate::gpu::HELD }>::new();
        if !gpu.reclaim(|t| back.push(t)) {
            return false;
        }
        for t in back {
            // Sound: the piece stays in the GPU ring, and its output and
            // count in place, until its count is zero.
            let bytes = unsafe { core::slice::from_raw_parts(t.input, PIECE) };
            let task = Task { key: self.key, flags: self.flags, out: t.out, left: t.left, ..Task::of(bytes, t.counter) };
            self.hand_over(task);
        }
        #[cfg(test)]
        crate::gpu::test::RECLAIMS.fetch_add(1, Ordering::Relaxed);
        true
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
                Kind::Piece { counter, last, long, out, .. } => {
                    #[cfg(blake3_gpu)]
                    if let Kind::Piece { gpu: true, .. } = kind {
                        self.gpu_tail += room(self.gpu_tail, crate::gpu::RING_BYTES, len).1;
                    } else if long {
                        self.long_tail += room(self.long_tail, LONG_RING_BYTES, len).1;
                    }
                    #[cfg(not(blake3_gpu))]
                    if long {
                        self.long_tail += room(self.long_tail, LONG_RING_BYTES, len).1;
                    }
                    // Sound: the task wrote its out before counting `left` down.
                    let piece = unsafe { *self.s.pieces_out[index].0.0.get() };
                    let (key, flags, platform) = (self.key, self.flags, Platform::detect());
                    let parent = |left: &CVBytes, right: &CVBytes| crate::hasher::parent_output(left, right, &key, flags, platform);
                    // A piece at chunk 0 is a long message's first, a whole
                    // PIECE: it gave its two children.
                    debug_assert!(counter != 0 || len == PIECE, "a long message's first piece is whole");
                    let cv: CVBytes = if counter == 0 {
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
    /// Keep the storage as the thread's spare. A stream drops finished,
    /// every place handed back: one that unwinds ends the process first
    /// ([`stream`]'s guard), and the C API finishes it before it drops.
    fn drop(&mut self) {
        assert!(self.open.is_none() && self.delivered == self.next, "a stream drops finished");
        if let Some(slot) = self.slot {
            crate::lanes::sources::Source::unregister(slot);
        }
        crate::lanes::sources::closed();
        // Sound: `s` is not used again; the next stream takes it whole.
        let s: &'static mut Storage = unsafe { &mut *(self.s as *mut Storage) };
        FREE.lock().unwrap().push(FreeStorage(s));
    }
}

// The new ownership path is exercised on this same source and worker pool
// before its frontend replaces the synchronous submission frontend.
#[cfg(test)]
#[path = "stream_prototype.rs"]
mod prototype;

#[cfg(test)]
mod test {
    use super::*;

    /// The GPU's batches so far, in this process.
    #[cfg(blake3_gpu)]
    fn gpu_batches() -> usize {
        crate::gpu::test::GPU_BATCHES.load(Ordering::Relaxed)
    }

    /// Message `i` of a GPU test's stream: `len` bytes of `input`, 0, 64 or
    /// 128 bytes in.
    #[cfg(blake3_gpu)]
    fn gpu_message(input: &[u8], i: usize, len: usize) -> &[u8] {
        &input[64 * (i % 3)..][..len]
    }

    /// One stream of messages of `lens` bytes of `input` (gpu_message),
    /// written in reads of up to `read` bytes, in `mode`, back to back:
    /// their digests, every place handed back once, in order.
    #[cfg(blake3_gpu)]
    fn gpu_stream(mode: Mode, input: &[u8], lens: &[usize], read: usize, collect: bool) -> std::vec::Vec<Hash> {
        let mut got = places(lens.len());
        let want_handed: std::vec::Vec<(usize, usize)> = got.iter().map(|h| (h as *const Hash as usize, 1)).collect();
        let mut handed = std::vec::Vec::new();
        stream(mode, record(&mut handed), |s| {
            for (i, out) in got.iter_mut().enumerate() {
                s.submit_with(
                    lens[i],
                    |at, room| {
                        let n = room.len().min(read);
                        room[..n].copy_from_slice(&gpu_message(input, i, lens[i])[at..][..n]);
                        n
                    },
                    out,
                );
                if collect {
                    s.collect();
                }
            }
            s.finish();
            assert_eq!((s.gpu_tail, s.long_tail), (s.gpu_head, s.long_head), "every piece back frees exactly its room");
        });
        assert!(handed == want_handed, "every place handed back once, in order");
        got
    }

    /// `gpu_stream`, each digest checked against the CPU's.
    #[cfg(blake3_gpu)]
    fn gpu_stream_checked(mode: Mode, input: &[u8], lens: &[usize], read: usize, collect: bool) {
        for (i, got) in gpu_stream(mode, input, lens, read, collect).iter().enumerate() {
            assert_eq!(*got, crate::convenience::hash_with(mode, gpu_message(input, i, lens[i])), "message {i}, {} bytes", lens[i]);
        }
    }

    /// Engage every run (crate::gpu::Run: its timing is the machine's, and
    /// a gap of 5 ms between streams, a finish's or a check's, ends one),
    /// wait until the GPU is ready (its start runs on a thread of its own),
    /// then a stream of 64 messages of 16 MiB, in which the GPU takes
    /// batches.
    #[cfg(blake3_gpu)]
    fn gpu_ready(input: &[u8]) {
        crate::gpu::test::ENGAGED.store(true, Ordering::Relaxed);
        let started = std::time::Instant::now();
        while !crate::gpu::ready() {
            assert!(started.elapsed() < std::time::Duration::from_secs(10), "the GPU starts within 10 s");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let before = gpu_batches();
        gpu_stream_checked(Mode::Plain, input, &[16 << 20; 64], usize::MAX, false);
        assert!(gpu_batches() > before, "the GPU took batches in a stream of 1 GiB");
    }

    /// Long messages with the GPU taking batches, in every mode, among
    /// short ones and messages whose last piece is partial, written in
    /// reads of up to 1000 bytes, past the GPU ring's 64 MiB (so it
    /// wraps): each digest the CPU's, handed back once, in order; with
    /// collect after each, and finishing at once. (Job 1911's confirm
    /// baseline: the 1000-byte reads' stream took no batch.)
    #[cfg(blake3_gpu)]
    #[test]
    fn test_gpu_stream_against_hash_with() {
        let mut input = std::vec![0u8; (64 << 20) + 300];
        crate::test::paint_test_input(&mut input);
        gpu_ready(&input);
        let key = [9u8; 32];
        let lens: std::vec::Vec<usize> = (0..48)
            .map(|i| {
                [16 << 20, (4 << 20) + 1, 100, (5 << 20) + 1023, 64 << 20, (4 << 20) - 1, 16 << 20, 16 << 20, (6 << 20) + 7, (64 << 20) + 1, 8 << 20, 0][i % 12]
            })
            .collect();
        for (mode, collect, read) in
            [(Mode::Plain, false, 1000), (Mode::Keyed(&key), true, usize::MAX), (Mode::DeriveKey("gpu stream test"), false, usize::MAX)]
        {
            let before = gpu_batches();
            gpu_stream_checked(mode, &input, &lens, read, collect);
            // The GPU claims what waits unclaimed: a writer of whole pieces
            // runs ahead of the pool, one of 1000-byte reads may not (so
            // there its share is the machine's timing, not asserted).
            if read == usize::MAX {
                assert!(gpu_batches() > before, "the GPU took batches in this mode's stream");
            }
        }
    }

    /// Streams of one long message each, 40 back to back: the digests stay
    /// right with the GPU taking batches in some and each
    /// finish reclaiming what it holds, and the storage's GPU ring and
    /// commands serve each next stream.
    #[cfg(blake3_gpu)]
    #[test]
    fn test_gpu_streams_of_one_message() {
        let mut input = std::vec![0u8; (64 << 20) + 300];
        crate::test::paint_test_input(&mut input);
        gpu_ready(&input);
        let lens = [16 << 20, 64 << 20, 32 << 20];
        let want: std::vec::Vec<Hash> = lens.iter().map(|&len| crate::convenience::hash(gpu_message(&input, 0, len))).collect();
        let before = gpu_batches();
        let got: std::vec::Vec<Hash> = (0..40).map(|i| gpu_stream(Mode::Plain, &input, &[lens[i % 3]], usize::MAX, false)[0]).collect();
        for (i, got) in got.iter().enumerate() {
            assert_eq!(*got, want[i % 3], "stream {i}, {} bytes", lens[i % 3]);
        }
        assert!(gpu_batches() > before, "the GPU took batches in streams of one message");
    }

    /// The GPU ring's arithmetic, exact on every machine (the caller's
    /// own): 16 messages of 4 MiB, 1024 whole pieces, fill the GPU ring to
    /// its end with no lap skipped and no wait for room, the other rings
    /// keep none of their bytes, and the GPU's run holds every piece.
    #[cfg(blake3_gpu)]
    #[test]
    fn test_gpu_ring_fills_to_its_end_and_its_run_holds_every_piece() {
        let mut input = std::vec![0u8; (64 << 20) + 300];
        crate::test::paint_test_input(&mut input);
        gpu_ready(&input);
        let message = &input[..4 << 20];
        let mut got = places(crate::gpu::RING_BYTES / message.len());
        stream(
            Mode::Plain,
            |_| {},
            |s| {
                for out in got.iter_mut() {
                    s.submit_with(
                        message.len(),
                        |at, room| {
                            room.copy_from_slice(&message[at..][..room.len()]);
                            room.len()
                        },
                        out,
                    );
                }
                assert_eq!((s.gpu_head, s.long_head, s.head), (crate::gpu::RING_BYTES, 0, 0), "1024 pieces end the GPU ring's first lap exactly");
                assert_eq!(s.delivered, 0, "and nothing waited for room");
                assert_eq!(s.gpu_run.end - s.gpu_run.start, 1024, "the GPU's run holds every piece");
                s.finish();
                assert_eq!(s.gpu_tail, crate::gpu::RING_BYTES, "every piece back frees exactly its room");
            },
        );
        assert!(got.iter().all(|h| *h == crate::convenience::hash(message)));
    }

    /// A stream asks its GPU for a claim only where its run of GPU pieces
    /// could hold a batch (`feed`'s first test, the stream's own): a long
    /// message of half a batch's pieces, then short messages that wait for
    /// room, which ask nothing of the GPU's commands or the source.
    #[cfg(blake3_gpu)]
    #[test]
    fn test_gpu_feed_asks_only_with_a_batch_s_run() {
        let mut input = std::vec![0u8; (64 << 20) + 300];
        crate::test::paint_test_input(&mut input);
        gpu_ready(&input);
        let lens: std::vec::Vec<usize> = core::iter::once(4 << 20).chain(core::iter::repeat_n(1 << 20, 40)).collect();
        let feeds = crate::gpu::test::FEEDS.load(Ordering::Relaxed);
        gpu_stream_checked(Mode::Plain, &input, &lens, usize::MAX, false);
        assert_eq!(crate::gpu::test::FEEDS.load(Ordering::Relaxed), feeds, "64 GPU pieces, half a batch: no feed");
    }

    /// A stream whose GPU commands do not end until it has finished: its
    /// finish takes their tasks back for the pool (the GPU's outputs, when
    /// they come, dropped), and every digest is right. The held stream fits
    /// in the GPU ring: a GPU that never ends would leave a longer one
    /// waiting for room, its oldest tasks the GPU's.
    #[cfg(blake3_gpu)]
    #[test]
    fn test_gpu_finish_reclaims_the_gpus_tasks() {
        let mut input = std::vec![0u8; (64 << 20) + 300];
        crate::test::paint_test_input(&mut input);
        gpu_ready(&input);
        let lens = [16 << 20; 4];
        let (batches, reclaims) = (gpu_batches(), crate::gpu::test::RECLAIMS.load(Ordering::Relaxed));
        crate::gpu::test::HOLD.store(true, Ordering::Release);
        let got = gpu_stream(Mode::Plain, &input, &lens, usize::MAX, false);
        crate::gpu::test::HOLD.store(false, Ordering::Release);
        assert!(gpu_batches() > batches, "the GPU took batches");
        assert!(crate::gpu::test::RECLAIMS.load(Ordering::Relaxed) > reclaims, "finish took the held batches back");
        for (i, got) in got.iter().enumerate() {
            assert_eq!(*got, crate::convenience::hash(gpu_message(&input, i, lens[i])), "message {i}");
        }
        // The commands end, dropping their outputs; the GPU serves the next stream.
        gpu_stream_checked(Mode::Plain, &input, &lens, usize::MAX, false);
    }

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
    /// Keep: it detects, by hanging, mutants the clock then stops
    /// (tools/mutants/timeouts-x86_64.txt).
    #[test]
    fn test_stream_against_hash_with() {
        let mut input = std::vec![0u8; (1 << 20) + 200];
        crate::test::paint_test_input(&mut input);
        let key = [5u8; 32];
        let lens: std::vec::Vec<usize> =
            (0..12000).map(|i| [0, 1, 63, 64, 64, 64, 100, 1024, 1024, 1500, 4096, PIECE, PIECE + 1, 200_000, (1 << 20) + 7, 64][i % 16]).collect();
        // Borrowed messages start their blocks: 0, 1 or 2 blocks in.
        let blocks = crate::test::message(&input);
        let message = |i: usize| &input[64 * (i % 3)..][..lens[i]];
        for (mode, ways) in [(Mode::Plain, 0), (Mode::Keyed(&key), 1), (Mode::DeriveKey("stream test"), 2)] {
            let mut got = places(lens.len());
            let want_handed: std::vec::Vec<(usize, usize)> = got.iter().map(|h| (h as *const Hash as usize, 1)).collect();
            let mut handed = std::vec::Vec::new();
            stream(mode, record(&mut handed), |s| {
                // With workers to hand tasks to, the stream has its slot
                // among the pool's sources (a stream without one hashes
                // every task on the caller: right, and slower).
                assert_eq!(s.slot.is_some(), std::thread::available_parallelism().map_or(1, |n| n.get()) > 1);
                for (i, out) in got.iter_mut().enumerate() {
                    // Borrowed, written in place (in reads of up to 1000
                    // bytes), or each in turn.
                    if ways == 0 || (ways == 2 && i % 3 == 0) {
                        s.submit(&blocks[i % 3..], lens[i], out);
                    } else {
                        s.submit_with(
                            lens[i],
                            |at, room| {
                                let n = room.len().min(1000);
                                room[..n].copy_from_slice(&message(i)[at..][..n]);
                                n
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
                assert_eq!(got[i], crate::convenience::hash_with(mode, message(i)), "message {i}, {} bytes", lens[i]);
            }
        }
    }

    /// A batch written in place hashes its messages whatever `fill` leaves
    /// past each one: the stream zeroes those bytes, for every length that
    /// leaves some (an empty message's whole block, a partial last block).
    #[test]
    fn test_batch_with_zeroes_past_each_message() {
        for len in [0, 1, 63, 100, 1000] {
            let count = 5;
            let mut got = places(count);
            stream(
                Mode::Plain,
                |_| {},
                |s| {
                    s.submit_batch_with(
                        len,
                        |_, room| {
                            room.fill(0xAB);
                            room.len()
                        },
                        &mut got,
                    );
                    s.finish();
                },
            );
            assert_eq!(got, std::vec![crate::convenience::hash(&std::vec![0xABu8; len]); count], "{len}-byte messages");
        }
    }

    /// A plain `read` as the fill, the code a file or socket suggests
    /// (BLAKE3 #204): reads of 1 to 9000 bytes, some interrupted, the
    /// stream calling again until each piece is full, hash the right
    /// digests, a message and a batch.
    #[test]
    fn test_fill_with_a_plain_read() {
        use std::io::Read;
        let input: std::vec::Vec<u8> = (0..3 * PIECE + 77).map(|i| (i % 251) as u8).collect();
        let mut got = places(1);
        let mut batch = places(40);
        stream(
            Mode::Plain,
            |_| {},
            |s| {
                let mut reader = Uneven::new(&input, None);
                s.submit_with(input.len(), |_, room| retry(|| reader.read(room)), &mut got[0]);
                // 40 messages of 100 bytes, each in its 128-byte slot.
                let slots: std::vec::Vec<u8> = (0..40).flat_map(|m| (0..128).map(move |b| if b < 100 { (m * 7 + b) as u8 } else { 0 })).collect();
                let mut reader = Uneven::new(&slots, None);
                s.submit_batch_with(100, |_, room| retry(|| reader.read(room)), &mut batch);
                s.finish();
            },
        );
        assert_eq!(got[0], crate::convenience::hash(&input), "a message read in short reads");
        for (m, digest) in batch.iter().enumerate() {
            let message: std::vec::Vec<u8> = (0..100).map(|b| (m * 7 + b) as u8).collect();
            assert_eq!(*digest, crate::convenience::hash(&message), "batch message {m}");
        }
    }

    /// `read` again after an interruption, as `read_exact` does.
    fn retry(mut read: impl FnMut() -> std::io::Result<usize>) -> usize {
        loop {
            match read() {
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                other => return other.unwrap(),
            }
        }
    }

    /// A panic inside a stream ends the process, wherever it starts, and
    /// prints its message first: the body's own, the handler's, a fill
    /// that ends before the message's length (it can never hash the room's
    /// stale bytes, BLAKE3 #204), and a body that ends without finishing.
    /// Each case runs in a child process of this test binary, inside
    /// catch_unwind as a careless caller would: the child must die of
    /// SIGABRT, saying why. With the guard gone, it would catch the panic
    /// and exit 0.
    #[cfg(unix)]
    #[test]
    fn test_a_panic_in_a_stream_ends_the_process() {
        const CHILD: &str = "BLAKE3_SERVIL_TEST_STREAM_PANIC";
        let input = std::vec![9u8; 300_000];
        let blocks = crate::test::message(&input);
        if let Ok(case) = std::env::var(CHILD) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut got = places(4);
                stream(
                    Mode::Plain,
                    |_: &mut [Hash]| assert!(case != "handler", "the handler panics"),
                    |s| match case.as_str() {
                        "body" | "handler" => {
                            for (len, out) in [10, 100_000, 64, 300_000].into_iter().zip(&mut got) {
                                s.submit(&blocks, len, out);
                            }
                            if case == "body" {
                                panic!("the body panics");
                            }
                            s.finish();
                        }
                        "fill" => {
                            let mut left = 1000usize;
                            s.submit_with(
                                5000,
                                |_, room| {
                                    let n = room.len().min(left);
                                    left -= n;
                                    n
                                },
                                &mut got[0],
                            );
                            s.finish();
                        }
                        "unfinished" => s.submit(&blocks, 300_000, &mut got[0]),
                        _ => unreachable!(),
                    },
                )
            }));
            std::process::exit(0);
        }
        for (case, says) in [
            ("body", "the body panics"),
            ("handler", "the handler panics"),
            ("fill", "wrote nothing with 4000 of its 5000 bytes left"),
            ("unfinished", "a stream's body calls finish last"),
        ] {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "stream::test::test_a_panic_in_a_stream_ends_the_process", "--nocapture", "--test-threads", "1"])
                .env(CHILD, case)
                .output()
                .expect("the test binary runs again");
            use std::os::unix::process::ExitStatusExt as _;
            let stderr = std::string::String::from_utf8_lossy(&out.stderr);
            assert_eq!(out.status.signal(), Some(6), "{case}: the child aborts (SIGABRT): {:?}\n{stderr}", out.status);
            assert!(stderr.contains(says), "{case}: it says why: {stderr}");
        }
    }

    /// A reader that yields `data` in reads of varying sizes (1 byte to
    /// 9000), interrupted now and then, failing after `fail_at` bytes. Read
    /// into no room, or again after it reported its end or an error, it
    /// panics: each is a read (for a file, a system call) no caller needs.
    struct Uneven<'a> {
        data: &'a [u8],
        at: usize,
        calls: usize,
        fail_at: Option<usize>,
        ended: bool,
    }

    impl<'a> Uneven<'a> {
        fn new(data: &'a [u8], fail_at: Option<usize>) -> Self {
            Uneven { data, at: 0, calls: 0, fail_at, ended: false }
        }
    }

    impl std::io::Read for Uneven<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            assert!(!self.ended, "read again after its end or an error");
            assert!(!buf.is_empty(), "read into no room");
            self.calls += 1;
            if self.calls % 7 == 0 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if self.fail_at.is_some_and(|f| self.at >= f) {
                self.ended = true;
                return Err(std::io::Error::other("the reader fails"));
            }
            let limit = self.fail_at.unwrap_or(self.data.len()).min(self.data.len());
            let n = buf.len().min([1, 9000, 333, 65536, 4096][self.calls % 5]).min(limit - self.at);
            buf[..n].copy_from_slice(&self.data[self.at..][..n]);
            self.at += n;
            self.ended = n == 0;
            Ok(n)
        }
    }

    /// `submit_reader` through a reader of short reads, interrupted now and
    /// then: each message's length and digest, for lengths around a piece.
    /// Then readers that fail at every point (in the first piece; at exactly
    /// a piece, in the read-ahead; one byte into the read-ahead; in the
    /// third piece and the fourth, with pieces already handed to the pool),
    /// each followed by more than two rings of messages, long ones among
    /// them: every digest right, each failed reader's place untouched and
    /// never handed back, the ring's head moved by the handed pieces alone,
    /// and no reader read again after its end or error (Uneven). BLAKE3
    /// #218: the interruption's retry, the error path's ring head and its
    /// abandoned entry, and the read-ahead's condition had no test; the last
    /// two only cost work, which this counts, where a clock would only
    /// guess.
    #[test]
    fn test_submit_reader_reads_and_fails_at_every_point() {
        let input: std::vec::Vec<u8> = (0..4 * PIECE + 99).map(|i| (i % 253) as u8).collect();
        let reader = |len: usize, fail_at: Option<usize>| Uneven::new(&input[..len], fail_at);
        let lens = [0, 1, 1000, PIECE - 1, PIECE, PIECE + 1, 2 * PIECE, 3 * PIECE + 7];
        // Where each reader fails, and how far it leaves the ring's head:
        // the room of the pieces it handed to the pool, not a byte more
        // (the ring's head is the caller's arithmetic alone, whatever the
        // threads do, so this is exact on every machine).
        let fails = [(10, 0), (PIECE, 0), (PIECE + 1, 0), (2 * PIECE + 5, PIECE), (3 * PIECE + 1, 2 * PIECE)];
        // Each handed piece is an entry, and with any, one abandoned entry
        // drops their chaining values; none handed, none added.
        let entries = |kept: usize| kept / PIECE + usize::from(kept > 0);
        let after: std::vec::Vec<usize> = (0..2 * RING_BYTES / PIECE + 3).map(|i| [PIECE + 3, 700, 2 * PIECE][i % 3]).collect();
        let mut want: std::vec::Vec<Hash> = lens.iter().map(|&l| crate::convenience::hash(&input[..l])).collect();
        for _ in fails {
            want.push(Hash::from_bytes([0; 32]));
            want.extend(after.iter().map(|&l| crate::convenience::hash(&input[..l])));
        }
        let mut got = places(want.len());
        let mut handed = std::vec::Vec::new();
        let mut rest: &mut [Hash] = &mut got;
        stream(Mode::Plain, record(&mut handed), |s| {
            for len in lens {
                assert_eq!(s.submit_reader(reader(len, None), None, &mut take(&mut rest, 1)[0]).unwrap(), len as u64, "{len} bytes");
            }
            for (fail, kept) in fails {
                let (before, next) = (s.head, s.next);
                assert!(s.submit_reader(reader(input.len(), Some(fail)), None, &mut take(&mut rest, 1)[0]).is_err(), "fails at {fail}");
                assert_eq!(s.head - before, kept, "a reader failing at {fail} keeps its handed pieces' room alone");
                assert_eq!(s.next - next, entries(kept), "a reader failing at {fail}: its handed pieces' entries, and one to drop them");
                for &len in &after {
                    assert_eq!(s.submit_reader(reader(len, None), None, &mut take(&mut rest, 1)[0]).unwrap(), len as u64);
                }
            }
            s.finish();
        });
        for (i, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(got, want, "place {i}");
        }
        assert_eq!(handed.len(), lens.len() + fails.len() * after.len(), "a place back per message read, none for a failed reader");
    }

    /// Long messages (4 MiB or more, in the long ring) among short ones and
    /// messages just short of 4 MiB, in every mode, written in reads of up
    /// to 1000 bytes, past the long ring's 8 MiB (so it wraps): each
    /// digest the CPU's, handed back once, in order; with collect after
    /// each, and finishing at once.
    #[test]
    fn test_long_messages_against_hash_with() {
        let mut input = std::vec![0u8; (16 << 20) + 300];
        crate::test::paint_test_input(&mut input);
        let key = [9u8; 32];
        let lens = [LONG, LONG + 1, 100, (5 << 20) + 1023, 16 << 20, LONG - 1, PIECE + 1, (16 << 20) + 1, (6 << 20) + 7, 8 << 20, 0, 16 << 20];
        let message = |i: usize| &input[64 * (i % 3)..][..lens[i % lens.len()]];
        for (mode, collect, read) in
            [(Mode::Plain, false, 1000), (Mode::Keyed(&key), true, usize::MAX), (Mode::DeriveKey("long stream test"), false, usize::MAX)]
        {
            let mut got = places(2 * lens.len());
            let want_handed: std::vec::Vec<(usize, usize)> = got.iter().map(|h| (h as *const Hash as usize, 1)).collect();
            let mut handed = std::vec::Vec::new();
            stream(mode, record(&mut handed), |s| {
                for (i, out) in got.iter_mut().enumerate() {
                    s.submit_with(
                        lens[i % lens.len()],
                        |at, room| {
                            let n = room.len().min(read);
                            room[..n].copy_from_slice(&message(i)[at..][..n]);
                            n
                        },
                        out,
                    );
                    if collect {
                        s.collect();
                    }
                }
                assert!(s.long_head > LONG_RING_BYTES, "the long ring wrapped");
                s.finish();
                assert_eq!(s.long_tail, s.long_head, "every long piece back frees exactly its room");
            });
            assert!(handed == want_handed, "every place handed back once, in order");
            for (i, got) in got.iter().enumerate() {
                assert_eq!(*got, crate::convenience::hash_with(mode, message(i)), "message {i}, {} bytes", lens[i % lens.len()]);
            }
        }
    }

    /// The long ring's arithmetic, exact on every machine: 2 messages of
    /// 4 MiB, 128 whole pieces, fill it to its end with no lap skipped and
    /// no wait for room, and the 2 MiB ring keeps none of their bytes.
    #[test]
    fn test_long_messages_fill_the_long_ring_to_its_end() {
        let input = std::vec![5u8; LONG];
        let mut got = places(LONG_RING_BYTES / LONG);
        stream(
            Mode::Plain,
            |_| {},
            |s| {
                for out in got.iter_mut() {
                    s.submit_with(
                        LONG,
                        |at, room| {
                            room.copy_from_slice(&input[at..][..room.len()]);
                            room.len()
                        },
                        out,
                    );
                }
                assert_eq!((s.long_head, s.head), (LONG_RING_BYTES, 0), "128 pieces end the long ring's first lap exactly");
                assert_eq!(s.delivered, 0, "and nothing waited for room");
                s.finish();
                assert_eq!(s.long_tail, LONG_RING_BYTES, "every long piece back frees exactly its room");
            },
        );
        assert!(got.iter().all(|h| *h == crate::convenience::hash(&input)));
    }

    /// A reader failing in its first piece where that piece's room would
    /// start the ring's next lap (`reserve` skipped the lap's end) keeps no
    /// room either: the head goes back to where it was, not to the next
    /// lap's start (BLAKE3 #238's extra mutations: `i == 0` replaced with
    /// `false` kept the skip, and no test failed a reader there: a reader
    /// reserves whole pieces, so readers alone never skip). Copied 700-byte
    /// messages (704-byte slots) bring the head within a piece of the lap's
    /// end; then the failure, and the ring goes on to hash the rest.
    #[test]
    fn test_submit_reader_failing_where_its_piece_skips_the_lap_end() {
        let input: std::vec::Vec<u8> = (0..PIECE + 99).map(|i| (i % 251) as u8).collect();
        let mut lens = std::vec::Vec::new();
        let mut got = places(RING_BYTES / 704 + 40);
        let mut handed = std::vec::Vec::new();
        let mut rest: &mut [Hash] = &mut got;
        stream(Mode::Plain, record(&mut handed), |s| {
            while s.head % RING_BYTES <= RING_BYTES - PIECE {
                lens.push(700);
                s.submit_with(
                    700,
                    |at, room| {
                        room.copy_from_slice(&input[at..][..room.len()]);
                        room.len()
                    },
                    &mut take(&mut rest, 1)[0],
                );
            }
            let (before, next) = (s.head, s.next);
            assert!(s.submit_reader(Uneven::new(&input, Some(10)), None, &mut take(&mut rest, 1)[0]).is_err());
            assert_eq!(
                (s.head - before, s.next - next),
                (0, 0),
                "a reader failing in its first piece at {} keeps no room and adds no entry",
                before % RING_BYTES
            );
            lens.push(usize::MAX);
            for len in [PIECE + 99, 1, PIECE] {
                lens.push(len);
                s.submit_reader(Uneven::new(&input[..len], None), None, &mut take(&mut rest, 1)[0]).unwrap();
            }
            s.finish();
        });
        let mut golden = crate::test::Golden::new("submit_reader_failing_at_the_lap_end");
        for (len, got) in lens.iter().zip(&got) {
            if *len != usize::MAX {
                golden.add(got.as_bytes(), || crate::test::official::hash(Mode::Plain, &input[..*len]).to_vec());
            }
        }
        golden.check(0x6449952b3442e340);
        assert_eq!(handed.len(), lens.len() - 1, "a place back per message read, none for the failed reader");
    }

    /// A ring's room, exact: a 16-byte ring 10 bytes in flight (head 110,
    /// tail 100) has 6 bytes free and not 7, and an empty one all 16.
    #[test]
    fn test_has_room_counts_the_bytes_in_flight() {
        assert!(has_room(110, 100, 16, 6) && !has_room(110, 100, 16, 7));
        assert!(has_room(100, 100, 16, 16) && !has_room(100, 100, 16, 17));
        assert!(has_room(116, 100, 16, 0) && !has_room(116, 100, 16, 1), "a full ring");
    }

    /// The ring's arithmetic, exact on every machine (the caller's own,
    /// whatever the threads do): 32 whole pieces fill the 2 MiB ring to its
    /// end, the last fitting exactly with no lap skipped and no wait for
    /// room (nothing handed back yet). BLAKE3 #218: reserve's `>` as `>=`
    /// skipped a lap, its `<` as `<=` waited, each only costing work.
    #[test]
    fn test_a_reservation_that_fits_to_the_ring_s_end_takes_no_lap_and_no_wait() {
        let input = std::vec![5u8; PIECE];
        let mut got = places(RING_BYTES / PIECE);
        stream(
            Mode::Plain,
            |_| {},
            |s| {
                for out in got.iter_mut() {
                    s.submit_with(
                        PIECE,
                        |at, room| {
                            room.copy_from_slice(&input[at..][..room.len()]);
                            room.len()
                        },
                        out,
                    );
                }
                assert_eq!(s.head, RING_BYTES, "32 pieces end the ring's first lap exactly");
                assert_eq!(s.delivered, 0, "and nothing waited for room");
                s.finish();
            },
        );
        assert!(got.iter().all(|h| *h == crate::convenience::hash(&input)));
    }

    /// A batch is cut into parts of one task's bytes each (the last the
    /// rest), an entry a part, whether borrowed or written in, for message
    /// lengths whose slots divide a task and don't, and counts that fill
    /// parts and don't; every digest right. BLAKE3 #218: the part's size
    /// and the last part's count had no test.
    #[test]
    fn test_a_batch_is_cut_into_task_sized_parts() {
        for (message_len, count) in [(64usize, 2500usize), (100, 1000), (64, 1024), (1000, 70), (5000, 30)] {
            let slot = crate::many::slot_len(message_len);
            let per_part = (TASK_LEN / slot).max(1);
            let parts = count.div_ceil(per_part).max(1);
            let message = |m: usize| -> std::vec::Vec<u8> { (0..message_len).map(|b| (m * 31 + b) as u8).collect() };
            let mut batch = crate::throughput::Batch::new(message_len, count);
            for m in 0..count {
                batch.set(m, &message(m));
            }
            let bytes: std::vec::Vec<u8> = (0..count)
                .flat_map(|m| {
                    let mut v = message(m);
                    v.resize(slot, 0);
                    v
                })
                .collect();
            let (mut lent, mut written) = (places(count), places(count));
            stream(
                Mode::Plain,
                |_| {},
                |s| {
                    let next = s.next;
                    s.submit_batch(&batch, &mut lent);
                    assert_eq!(s.next - next, parts, "{count} messages of {message_len} bytes, lent: {parts} parts");
                    let next = s.next;
                    s.submit_batch_with(
                        message_len,
                        |at, room| {
                            room.copy_from_slice(&bytes[at..][..room.len()]);
                            room.len()
                        },
                        &mut written,
                    );
                    assert_eq!(s.next - next, parts, "{count} messages of {message_len} bytes, written in: {parts} parts");
                    s.finish();
                },
            );
            for (m, (a, b)) in lent.iter().zip(&written).enumerate() {
                let want = crate::convenience::hash(&message(m));
                assert!(*a == want && *b == want, "{message_len}-byte message {m} of {count}");
            }
        }
    }

    /// A batch written in has its bytes past each message zeroed where there
    /// are any (lengths not a multiple of 64, and empty messages, whose slot
    /// is a whole block), and none touched where there are none (counted:
    /// Counted::BatchZeroed); empty messages in a room an earlier stream left full of
    /// bytes hash as empty. BLAKE3 #218: the condition had no test.
    #[test]
    fn test_batch_with_zeroes_where_bytes_follow_a_message_and_nowhere_else() {
        let mut got = places(100);
        // Each fill writes 0xEE over its whole room, as a careless reader
        // might, so every byte past a message starts nonzero.
        let zeroed = |message_len: usize, got: &mut [Hash]| {
            let zeroed = || crate::COUNTED.with(|c| c[crate::Counted::BatchZeroed as usize].get());
            let before = zeroed();
            stream(
                Mode::Plain,
                |_| {},
                |s| {
                    s.submit_batch_with(
                        message_len,
                        |_, room| {
                            room.fill(0xEE);
                            room.len()
                        },
                        got,
                    );
                    s.finish();
                },
            );
            zeroed() - before
        };
        assert_eq!(zeroed(64, &mut got), 0, "64-byte messages fill their slots");
        assert!(got.iter().all(|h| *h == crate::convenience::hash(&[0xEE; 64])));
        assert_eq!(zeroed(128, &mut got[..50]), 0, "128-byte messages fill their slots");
        assert_eq!(zeroed(100, &mut got), 100, "each 100-byte message's last 28 bytes");
        assert!(got.iter().all(|h| *h == crate::convenience::hash(&[0xEE; 100])));
        assert_eq!(zeroed(0, &mut got), 100, "each empty message's whole block");
        assert!(got.iter().all(|h| *h == crate::convenience::hash(b"")), "empty messages hash as empty");
    }

    /// How short messages gather into tasks, exact (the caller's own state):
    /// an empty batch joins no task; 64-byte messages close a task at its
    /// 64 members; 1000-byte ones at a quarter of a task's bytes (the
    /// 17th), to spread over more threads; a batch part closes its own.
    /// BLAKE3 #218: the closing condition's mutants each regrouped work.
    #[test]
    fn test_short_messages_gather_into_tasks_as_designed() {
        let input = std::vec![7u8; 1000];
        let blocks = crate::test::message(&input);
        let members = |s: &Stream<'_, _>| s.open.as_ref().map(|t| (t.members, t.len));
        let mut got = places(64 + 17 + 1);
        let mut rest: &mut [Hash] = &mut got;
        let empty = crate::throughput::Batch::new(64, 0);
        let mut none: [Hash; 0] = [];
        stream(
            Mode::Plain,
            |_| {},
            |s| {
                s.submit_batch(&empty, &mut none);
                assert_eq!(members(s), None, "an empty batch joins no task");
                for k in 1..=64 {
                    s.submit(&blocks, 64, &mut take(&mut rest, 1)[0]);
                    assert_eq!(members(s), (k < 64).then_some((k, 64 * k)), "64-byte message {k}");
                }
                for k in 1..=17 {
                    s.submit(&blocks, 1000, &mut take(&mut rest, 1)[0]);
                    assert_eq!(members(s), (k < 17).then_some((k, 1000 * k)), "1000-byte message {k}");
                }
                s.submit(&blocks, 64, &mut take(&mut rest, 1)[0]);
                assert_eq!(members(s), Some((1, 64)));
                s.finish();
            },
        );
        assert!(got[..64].iter().chain(&got[81..]).all(|h| *h == crate::convenience::hash(&input[..64])));
        assert!(got[64..81].iter().all(|h| *h == crate::convenience::hash(&input)));
    }

    /// A finished stream gives back what it held: its source's slot (with
    /// 64 slots, 70 streams one after another each get one), the open
    /// streams' count (none open after it, so idle workers stop looking),
    /// and its storage. BLAKE3 #218: Drop made a no-op survived the suite.
    #[test]
    fn test_a_finished_stream_gives_back_its_slot_and_storage() {
        if !crate::lanes::splits() {
            return;
        }
        let input = std::vec![1u8; 64];
        let blocks = crate::test::message(&input);
        for round in 0..70 {
            let mut got = places(1);
            stream(
                Mode::Plain,
                |_| {},
                |s| {
                    assert!(s.slot.is_some(), "stream {round} has a slot");
                    s.submit(&blocks, 64, &mut got[0]);
                    s.finish();
                },
            );
            assert!(!crate::lanes::sources::open(), "no stream open after stream {round}");
            assert_eq!(got[0], crate::convenience::hash(&input));
        }
        assert!(!FREE.lock().unwrap().is_empty(), "the storage is the thread's spare again");
    }

    /// A few messages through a stream: the handover of entries to the
    /// pool's threads and back.
    /// Keep: it detects, by hanging, mutants the clock then stops
    /// (tools/mutants/timeouts-x86_64.txt).
    #[test]
    fn test_small_stream() {
        let input = std::vec![3u8; 3 * PIECE + 5];
        let batch_blocks = crate::test::blocks(&input[..4 * 64]);
        let batch = crate::throughput::Batch::from_blocks(&batch_blocks, 64);
        let blocks = crate::test::message(&input);
        let lens = [0, 64, 64, 1000, PIECE + 1, 3 * PIECE + 5];
        let mut got = places(2 * lens.len() + 2 + 7);
        let mut handed = std::vec::Vec::new();
        let mut rest: &mut [Hash] = &mut got;
        stream(Mode::Plain, record(&mut handed), |s| {
            for len in lens {
                s.submit(&blocks, len, &mut take(&mut rest, 1)[0]);
                s.submit_with(
                    len,
                    |at, room| {
                        assert_eq!(room.as_ptr() as usize % 64, 0, "a room starts a cache line");
                        room.copy_from_slice(&input[at..][..room.len()]);
                        room.len()
                    },
                    &mut take(&mut rest, 1)[0],
                );
            }
            assert_eq!(s.submit_reader(&input[..2 * PIECE + 1], None, &mut take(&mut rest, 1)[0]).unwrap(), 2 * PIECE as u64 + 1);
            assert!(s.submit_reader(Uneven::new(&input, Some(PIECE + 1)), None, &mut take(&mut rest, 1)[0]).is_err());
            s.submit_batch(&batch, take(&mut rest, 4));
            s.submit_batch_with(
                64,
                |at, room| {
                    assert_eq!(room.as_ptr() as usize % 64, 0, "a room starts a cache line");
                    room.copy_from_slice(&input[at..][..room.len()]);
                    room.len()
                },
                take(&mut rest, 3),
            );
            s.finish();
        });
        let mut want: std::vec::Vec<Hash> = lens.iter().flat_map(|&l| [crate::convenience::hash(&input[..l]); 2]).collect();
        want.push(crate::convenience::hash(&input[..2 * PIECE + 1]));
        want.push(Hash::from_bytes([0; 32]));
        want.extend([crate::convenience::hash(&input[..64]); 7]);
        assert_eq!(got, want);
        assert_eq!(handed.len(), 2 * lens.len() + 1 + 2, "a place back per submission, none for the failed reader");
    }
}
