//! Multithreaded hashing over every CPU, with concurrent callers served
//! round-robin.
//!
//! This module is crate-internal. The public entry points are
//! [`crate::hash_multithreaded`], [`crate::hash_with`], and their batch
//! forms, whose contracts speak of
//! threads alone; pieces and the pool are how those contracts
//! are met, and this file is where they are explained.
//!
//! # Pieces
//!
//! The input is cut at chunk boundaries into pieces, each a valid BLAKE3
//! subtree (see [`crate::hazmat::left_subtree_len`]), from the front: each piece
//! is about a thread's share of what remains ([`next_piece_len`]), a power
//! of two chunks within [`MIN_PIECE_LEN`] and [`MAX_PIECE_LEN`]. Pieces
//! shrink toward the end, so a thread that is slow (an efficiency core, or
//! one sharing its CPU) holds a small piece when the others run out, and
//! the finish waits on little. Each thread hashes whole subtrees directly
//! through the one-shot hashing code, producing one chaining value per
//! piece. The caller merges these values a level at a time through the
//! SIMD parent kernels and finishes with the root compression. Inputs
//! under [`MIN_SPLIT_LEN`] stay on the calling thread.
//!
//! # Jobs and the pool
//!
//! A call registers a *job* (its pieces, a cursor, an active-thread count)
//! in the pool's slot table, then takes pieces through the cursor until
//! none remain. It unregisters the job and waits for its active threads
//! to finish. Workers,
//! started once per process ([`crate::initialize_multithreaded`]), one per CPU beyond
//! the first, serve the registered jobs round-robin, one piece at a time
//! through the same cursor. Two callers
//! hashing at once therefore each get about half the workers' time, and
//! a slow thread takes fewer pieces than a fast one; there is nothing to
//! tune for fairness or balance.
//!
//! A call arriving when callers already fill the CPUs hashes its input
//! whole on its own thread. This check happens once per call; workers
//! need only the job's cursor for each piece. The fixed pool and the
//! calling threads can overlap; the OS schedules them.
//!
//! # Kernels
//!
//! On a CPU with SME2, a call that gets the SME2 turn hashes a prefix of
//! its input itself on SME2, in whole subtrees of up to 1 MiB back to back
//! (the flat walk), sized to its speed: about 1.65 NEON threads' worth
//! ([`sme2_prefix`]). Every other piece, the caller's after its prefix
//! included, runs on the NEON hybrids, which run at full speed on every
//! core at once. Measured on an M4 Max against the pool on NEON alone
//! (probe/sme2-thread, job 187): 1-64 MiB 24-32% faster with two threads,
//! 14-21% with four, 1-6% with sixteen; the VM on it alike. Pieces on
//! SME2 handed out one at a time lost (8-128 KiB, with the caller's
//! bookkeeping between them): the SME unit's slow state follows about
//! 0.2 µs of other work between its kernels (NOTES-servil.md).
//!
//! # Waiting
//!
//! Nothing keeps a worker awake for work that may come (AGENTS.md, "Serve
//! real programs"): a worker takes pieces and tasks while it finds them,
//! and sleeps on a condition variable as soon as it finds none. So every
//! call meets sleeping workers, and a call wakes only as many as it can use
//! (its pieces less one): it wakes one itself,
//! about 3 µs of the caller's time (a wake of all fifteen at once cost the
//! caller 25-56 µs), and the first to wake wakes the rest. A woken worker
//! arrives 15-45 µs later on a core whose clock the idle time lowered, so
//! inputs below [`MIN_SPLIT_LEN`] stay on the caller's thread. A caller
//! waiting for its last pieces polls, then sleeps after
//! [`SPIN_BEFORE_SLEEP`]. Polls spin in user space and yield the CPU every
//! [`YIELD_EVERY`]: the yield keeps a runnable thread from waiting a
//! scheduler quantum behind a poller (measured on a two-CPU machine: two
//! threads spinning without ever yielding took a 128 KiB split from 29 µs
//! to 2 ms). Beside eight hashing threads on a 16-vCPU VM, eight idle
//! pollers that call `sched_yield` in a loop slowed each hash by 36%, eight
//! that spin by 18%, eight asleep by nothing.

use crate::hazmat::ChainingValue;
#[cfg(test)]
use crate::hazmat::{self, Mode};
#[cfg(test)]
use crate::hazmat::HasherExt;
use crate::platform::Platform;
use crate::{CHUNK_LEN, Hash};
#[cfg(test)]
use crate::{Hasher, KEY_LEN};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Inputs below this length are hashed on the calling thread. Every call
/// meets sleeping workers (nothing keeps them awake between calls), and a
/// woken worker arrives 15-45 µs later on a core whose clock the idle time
/// lowered, so a split pays only for inputs that take the caller longer.
/// Measured with 1 ms of sleep before each call, the split came back
/// sooner from 512 KiB on an M4 Max (136 µs against 158; jobs 364-367)
/// and from 768 KiB in the VM (146 against 181; 512 KiB 154 against
/// 130). Zooko chose 512 KiB, where it pays on the Mac (September 28,
/// 2026; native first): after the gap 512 KiB 0.35 -> 0.19-0.24 ns/B and
/// 8192 64-byte messages 22.5 -> 13.3 ns/msg (jobs 538-541), the VM's
/// 512 KiB 20-30% slower than on the caller's thread.
pub(crate) const MIN_SPLIT_LEN: usize = 512 * 1024;

/// The shortest piece: eight chunks, a hybrid kernel's worth.
const MIN_PIECE_LEN: usize = 8 * CHUNK_LEN;

/// The longest piece.
const MAX_PIECE_LEN: usize = 128 * CHUNK_LEN;

/// Between polls a thread spins in user space, and every YIELD_EVERY it
/// hands the CPU to any runnable thread (a caller the OS has queued
/// behind a poller then waits this long at most, never a quantum).
const YIELD_EVERY: std::time::Duration = std::time::Duration::from_micros(20);

#[inline]
pub(crate) fn poll_pause(yielded: &mut std::time::Instant) {
    for _ in 0..8 {
        std::hint::spin_loop();
    }
    if yielded.elapsed() >= YIELD_EVERY {
        std::thread::yield_now();
        *yielded = std::time::Instant::now();
    }
}

/// Lock `mutex`, polling for it first: its holders keep it for well under
/// a microsecond, and a waiter the std mutex parks costs both sides a
/// system call (the waiter's sleep, the holder's wake: microseconds in a
/// VM, 50000 of them per million 64-byte messages through the queue).
/// Parks only after about LOCK_POLLS failed tries.
pub(crate) fn lock_polling<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    for _ in 0..LOCK_POLLS {
        match mutex.try_lock() {
            Ok(guard) => return guard,
            Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
            Err(std::sync::TryLockError::Poisoned(_)) => panic!("a panic while hashing aborts, so no lock is poisoned"),
        }
    }
    mutex.lock().expect("a panic while hashing aborts, so no lock is poisoned")
}

const LOCK_POLLS: usize = 256;

/// How long a caller polls for its last pieces before sleeping.
pub(crate) const SPIN_BEFORE_SLEEP: std::time::Duration = std::time::Duration::from_micros(200);

/// The length of the next piece when `remaining` bytes are uncut and
/// `threads` threads may share them: about a thread's share of what is
/// left, rounded down to a power of two chunks within the bounds above.
/// `threads` is positive. Pieces therefore shrink toward the end of the input, and the slowest
/// thread's last piece is a small one.
fn next_piece_len(remaining: usize, threads: usize) -> usize {
    debug_assert!(threads > 0, "a cut needs at least one thread");
    let want = (remaining / threads).clamp(MIN_PIECE_LEN, MAX_PIECE_LEN);
    1 << (usize::BITS - 1 - want.leading_zeros())
}

/// Hash `input` over the machine's threads.
#[inline]
pub(crate) fn hash(input: &[u8]) -> Hash {
    hash_with_key(input, crate::IV, 0)
}

/// Hash `input` over the machine's threads in the mode of `key` and
/// `flags` (IV and 0 for plain hashing).
#[inline]
pub(crate) fn hash_with_key(input: &[u8], key: &crate::CVWords, flags: u8) -> Hash {
    // The short path first and inline, so a small input costs what hash()
    // costs; everything the pool needs is behind the call below.
    if input.len() < MIN_SPLIT_LEN {
        return crate::hash_serial(input, key, flags);
    }
    hash_over_pool(input, key, flags)
}

#[inline(never)]
fn hash_over_pool(input: &[u8], key: &crate::CVWords, flags: u8) -> Hash {
    let pool = pool();
    let callers = pool.callers.fetch_add(1, Ordering::SeqCst) + 1;
    let _caller = Caller(&pool.callers);
    // No CPU to spare (every one has a caller, or the process has one
    // CPU): hash as hash() does, as the batch path does.
    if callers >= pool.cpus {
        return crate::hash_serial(input, key, flags);
    }
    let threads = pool.cpus;
    let turn = crate::platform::Sme2Turn::take(Platform::detect(), true);
    let (pieces, own) = cut_with_prefix(input.len(), threads, prefix_for(&turn, input.len(), threads));
    let mut cvs = vec![ChainingValue::default(); pieces.len()];
    let work = Work::Tree {
        input,
        pieces: &pieces,
        cvs: cvs.as_mut_ptr(),
        key: *key,
        counter: 0,
        flags,
    };
    pool.run_job(work, pieces.len(), own, turn.platform());
    drop(turn);
    merge_root(&pieces, &mut cvs, key, flags)
}

/// Whether the pool has a thread that takes queue tasks: a worker, or the
/// SME2 thread. With one CPU to the process (`available_parallelism`, which
/// counts its affinity and quota) and no SME2 it has none, and a queue
/// hashes everything on its delivery thread.
pub(crate) fn takes_tasks() -> bool {
    let pool = pool();
    pool.cpus > 1 || pool.sme2
}

/// The two child chaining values of the subtree `input` at chunk
/// `counter`, over the machine's threads: what
/// `compress_subtree_to_parent_node` returns for a whole subtree.
/// Requires a power-of-two number of chunks, at least MIN_SPLIT_LEN
/// bytes, and `counter` a multiple
/// of that number, as `Hasher::update` hands out.
pub(crate) fn subtree_children(
    input: &[u8],
    key: &crate::CVWords,
    counter: u64,
    flags: u8,
) -> [u8; 2 * crate::OUT_LEN] {
    assert!(input.len().is_power_of_two() && input.len() >= MIN_SPLIT_LEN, "a whole subtree of at least MIN_SPLIT_LEN bytes");
    assert_eq!(counter % (input.len() / CHUNK_LEN) as u64, 0, "a subtree starts at a multiple of its chunk count");
    let pool = pool();
    let callers = pool.callers.fetch_add(1, Ordering::SeqCst) + 1;
    let _caller = Caller(&pool.callers);
    let threads = pool.cpus;
    if callers >= pool.cpus {
        return crate::compress_subtree_to_parent_node::<crate::join::SerialJoin>(input, 0, key, counter, flags, pool_platform());
    }
    let turn = crate::platform::Sme2Turn::take(Platform::detect(), true);
    let (pieces, own) = cut_with_prefix(input.len(), threads, prefix_for(&turn, input.len(), threads));
    let mut cvs = vec![ChainingValue::default(); pieces.len()];
    let work = Work::Tree { input, pieces: &pieces, cvs: cvs.as_mut_ptr(), key: *key, counter, flags };
    pool.run_job(work, pieces.len(), own, turn.platform());
    drop(turn);
    let count = merge_to_children(&pieces, &mut cvs, key, flags);
    debug_assert_eq!(count, 2);
    let mut children = [0u8; 2 * crate::OUT_LEN];
    children[..crate::OUT_LEN].copy_from_slice(&cvs[0]);
    children[crate::OUT_LEN..].copy_from_slice(&cvs[1]);
    children
}

/// [`crate::hash_many`] over the machine's threads: below MIN_SPLIT_LEN
/// of input, or with one message, on this thread.
#[inline]
pub(crate) fn hash_many(input: &[u8], message_len: usize, key: &crate::CVWords, flags: u8, outputs: &mut [[u8; crate::OUT_LEN]]) {
    // The short path first and inline, so a small batch costs what
    // crate::hash_many costs; everything the pool needs is behind the call.
    if outputs.len() < 2 || input.len() < MIN_SPLIT_LEN {
        return crate::hash_many_serial(input, message_len, key, flags, outputs);
    }
    hash_many_over_pool(input, message_len, key, flags, outputs)
}

#[inline(never)]
fn hash_many_over_pool(input: &[u8], message_len: usize, key: &crate::CVWords, flags: u8, outputs: &mut [[u8; crate::OUT_LEN]]) {
    let slot = crate::many::slot_len(message_len);
    assert_eq!(Some(input.len()), slot.checked_mul(outputs.len()), "input holds one slot of whole blocks per output");
    let pool = pool();
    let callers = pool.callers.fetch_add(1, Ordering::SeqCst) + 1;
    let _caller = Caller(&pool.callers);
    if callers >= pool.cpus {
        return crate::hash_many_serial(input, message_len, key, flags, outputs);
    }
    let pieces = cut_messages(outputs.len(), slot, pool.cpus);
    let work = Work::Messages { input, message_len, key: *key, flags, pieces: &pieces, outputs: outputs.as_mut_ptr() };
    pool.run_job(work, pieces.len(), 0, pool_platform());
}

/// Cut `count` messages in slots of `message_len` bytes into ranges for `threads`
/// threads, in order: each range holds about [`next_piece_len`] of the
/// bytes that remain, at least one message, so ranges shrink toward the
/// end as subtree pieces do.
fn cut_messages(count: usize, message_len: usize, threads: usize) -> Vec<Piece> {
    let mut pieces = Vec::with_capacity(64);
    let mut start = 0;
    while start < count {
        let remaining = (count - start) * message_len.max(1);
        let take = (next_piece_len(remaining, threads) / message_len.max(1)).clamp(1, count - start);
        pieces.push(Piece { offset: start, len: take });
        start += take;
    }
    pieces
}

/// A counted call, including its serial fallback. Keeping the count until
/// return lets concurrent callers see that this CPU already has work.
struct Caller<'a>(&'a AtomicUsize);

impl Drop for Caller<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One call's work, on the caller's stack. Workers reach it through the
/// pool's slots while it is registered, and finish the pieces they hold
/// after. Once the slot is cleared and its readers have drained, no new
/// reservation can reach this job. The caller waits for `active == 0`, so
/// every worker access falls inside the caller's frame.
struct Job<'a> {
    work: Work<'a>,
    /// How many pieces the work has; `cursor` counts up to it.
    pieces: usize,
    /// The next piece to take.
    cursor: Line,
    /// The threads holding the job, the caller's included while it takes
    /// pieces: a worker counts itself in before taking a piece and out
    /// after it (or after finding none), so the caller returns once it
    /// reaches zero.
    active: Line,
}

/// What a job's pieces are. Piece `i` writes output slot `i` (a chaining
/// value, or a range of digests) and nothing else, from the thread that
/// took `i` through the cursor.
enum Work<'a> {
    /// Subtrees of one input, in offset order; one chaining value each.
    Tree {
        input: &'a [u8],
        pieces: &'a [Piece],
        cvs: *mut ChainingValue,
        key: crate::CVWords,
        /// The chunk counter at `input`'s first byte.
        counter: u64,
        flags: u8,
    },
    /// Ranges of a batch of messages of one length, each in its slot of
    /// whole blocks in `input` (`offset` and `len` count messages); one
    /// digest each.
    Messages {
        input: &'a [u8],
        message_len: usize,
        key: crate::CVWords,
        flags: u8,
        pieces: &'a [Piece],
        outputs: *mut [u8; crate::OUT_LEN],
    },
}

/// An atomic counter on its own cache line: `cursor` and `active` are
/// each hit by every thread of a job, and sharing a line would make
/// each update evict the others.
#[repr(align(128))]
struct Line(AtomicUsize);

/// Any value on a cache line of its own.
#[repr(align(128))]
pub(crate) struct OwnLine<T>(pub(crate) T);

impl<T> std::ops::Deref for OwnLine<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl std::ops::Deref for Line {
    type Target = AtomicUsize;
    fn deref(&self) -> &AtomicUsize {
        &self.0
    }
}

impl Job<'_> {
    /// Hash piece `index` into its slot. The caller has taken `index`
    /// through `cursor`.
    unsafe fn hash_piece(&self, index: usize, platform: Platform) {
        match &self.work {
            Work::Tree { input, pieces, cvs, key, counter, flags } => {
                let piece = pieces[index];
                let bytes = &input[piece.offset..][..piece.len];
                let counter = counter + (piece.offset / CHUNK_LEN) as u64;
                let cv = crate::hash_all_at_once::<crate::join::SerialJoin>(bytes, key, counter, *flags, platform).chaining_value();
                unsafe { *cvs.add(index) = cv };
            }
            Work::Messages { input, message_len, key, flags, pieces, outputs } => {
                let piece = pieces[index];
                let slot = crate::many::slot_len(*message_len);
                let messages = &input[piece.offset * slot..][..piece.len * slot];
                // Sound: this range of outputs belongs to piece `index` alone.
                let digests = unsafe { core::slice::from_raw_parts_mut(outputs.add(piece.offset), piece.len) };
                crate::many::hash_many_on(messages, *message_len, key, *flags, digests, platform);
            }
        }
    }
}

/// The platform every piece runs on (see the module docs, "Kernels"): on
/// an SME2 CPU the NEON hybrids, which no other core slows down; elsewhere
/// the detected one.
fn pool_platform() -> Platform {
    #[cfg(blake3_sme2)]
    if matches!(Platform::detect(), Platform::SME2) {
        return Platform::NEON;
    }
    Platform::detect()
}

/// One piece of the input: a whole subtree (bytes), or for a batch a
/// range of messages (message indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    offset: usize,
    len: usize,
}

/// Cut `len` bytes into subtrees for `threads` threads, in offset order:
/// each piece is [`next_piece_len`] of what remains, and the last is
/// whatever is left. A piece's length never exceeds the one before it and
/// every piece starts at a multiple of its own length, so each is a whole
/// subtree at its offset. `len` exceeds [`MIN_PIECE_LEN`] and at least
/// two threads share it, so two or more pieces come back.
#[cfg(test)]
fn cut_subtrees(len: usize, threads: usize) -> Vec<Piece> {
    cut_with_prefix(len, threads, 0).0
}

/// [`cut_subtrees`] after a prefix of about `prefix` bytes for the SME2
/// thread: whole subtrees of up to [`MAX_PREFIX_PIECE_LEN`], the prefix
/// rounded down to a multiple of the first piece after it, so the lengths
/// still never increase. Returns the pieces and how many form the prefix
/// (none when `prefix` rounds to nothing). `prefix` is below `len`.
fn cut_with_prefix(len: usize, threads: usize, prefix: usize) -> (Vec<Piece>, usize) {
    assert!(threads >= 2, "a serial call hashes the input whole");
    assert!(len > MIN_PIECE_LEN, "one piece has no parent to merge; hash it whole instead");
    assert!(prefix < len, "the SME2 thread's prefix leaves the pool some input");
    let mut pieces = Vec::with_capacity(64);
    let mut offset = 0;
    let first_after = next_piece_len(len - prefix, threads);
    let prefix = prefix / first_after * first_after;
    while offset < prefix {
        let piece = MAX_PREFIX_PIECE_LEN.min(1 << (usize::BITS - 1 - (prefix - offset).leading_zeros()));
        pieces.push(Piece { offset, len: piece });
        offset += piece;
    }
    let own = pieces.len();
    let mut cap = if own > 0 { pieces[own - 1].len } else { MAX_PIECE_LEN };
    while len - offset > MIN_PIECE_LEN {
        let piece = next_piece_len(len - offset, threads).min(cap);
        if len - offset <= piece {
            break;
        }
        pieces.push(Piece { offset, len: piece });
        offset += piece;
        cap = piece;
    }
    pieces.push(Piece { offset, len: len - offset });
    (pieces, own)
}

/// The SME2 thread's prefix for a call holding `turn`: [`sme2_prefix`]
/// when the turn gave it SME2, else nothing.
fn prefix_for(turn: &crate::platform::Sme2Turn, len: usize, threads: usize) -> usize {
    #[cfg(blake3_sme2)]
    if matches!(turn.platform(), Platform::SME2) {
        return sme2_prefix(len, threads);
    }
    let _ = (turn, len, threads);
    0
}

/// The longest piece of the SME2 thread's prefix: the flat walk's largest
/// subtree, which it hashes on SME2 alone.
const MAX_PREFIX_PIECE_LEN: usize = 1 << 20;

/// The SME2 thread's share of an input that `threads` threads hash,
/// itself included: its speed over theirs, about 1.65 NEON threads' worth
/// (M4 Max and the VM on it: the best split of 1 to 64 MiB put 62% on SME2
/// beside one NEON thread, 38% beside three, 12% beside fifteen;
/// probe/sme2-thread, job 187).
/// A share below MAX_PIECE_LEN is no prefix: too little SME2 work to pay
/// for the turn and the streaming session (VM, servil mt 256 KiB over
/// sixteen threads, a 24 KiB prefix: +18-20%).
#[cfg_attr(not(blake3_sme2), allow(dead_code))]
fn sme2_prefix(len: usize, threads: usize) -> usize {
    let share = len / (100 + (threads - 1) * 10000 / 165) * 100;
    if share >= MAX_PIECE_LEN { share } else { 0 }
}

/// The chaining value of the subtree covering `pieces` (in offset order,
/// tiling one subtree): the piece's own value for one piece, else the
/// parent of its two halves, cut at the subtree's left-subtree boundary.
#[cfg(test)]
fn subtree_cv(pieces: &[Piece], cvs: &[ChainingValue], mode: Mode) -> ChainingValue {
    debug_assert_eq!(pieces.len(), cvs.len());
    if pieces.len() == 1 {
        return cvs[0];
    }
    let offset = pieces[0].offset;
    let len: usize = pieces.iter().map(|p| p.len).sum();
    let boundary = offset + hazmat::left_subtree_len(len as u64) as usize;
    let split = pieces.partition_point(|p| p.offset < boundary);
    assert!(split > 0 && split < pieces.len() && pieces[split].offset == boundary, "pieces do not tile the tree");
    let left = subtree_cv(&pieces[..split], &cvs[..split], mode);
    let right = subtree_cv(&pieces[split..], &cvs[split..], mode);
    hazmat::merge_subtrees_non_root(&left, &right, mode)
}

/// Merge a shrinking cut in place, a level at a time. `pieces` tiles the
/// input from zero, with non-increasing power-of-two lengths except the
/// final (possibly short) piece, as produced by `cut_subtrees`. Each CV
/// hashes its corresponding piece. Two or more pieces are required.
///
/// At each level the longer pieces form an untouched prefix; the suffix
/// contains the children of this level. Hash adjacent pairs together with
/// the platform's SIMD parent kernel, carrying an odd final child upward.
/// The final two CVs belong to the root's children, including a short right
/// subtree, on the pool's platform like the pieces.
fn merge_root(pieces: &[Piece], cvs: &mut [ChainingValue], key: &crate::CVWords, flags: u8) -> Hash {
    let count = merge_to_children(pieces, cvs, key, flags);
    debug_assert_eq!(count, 2);
    crate::parent_node_output(&cvs[0], &cvs[1], key, flags, pool_platform()).root_hash()
}

/// [`merge_root`] up to the root's two children, left in `cvs[0]` and
/// `cvs[1]`; returns 2.
fn merge_to_children(pieces: &[Piece], cvs: &mut [ChainingValue], key: &crate::CVWords, flags: u8) -> usize {
    debug_assert_eq!(pieces.len(), cvs.len());
    debug_assert!(pieces.len() >= 2);
    debug_assert_eq!(pieces[0].offset, 0);
    debug_assert!(pieces.last().unwrap().len > 0);
    debug_assert!(pieces[..pieces.len() - 1].iter().all(|p| p.len >= MIN_PIECE_LEN));
    debug_assert!(pieces.windows(2).all(|w| w[0].len.is_power_of_two()
        && w[0].len >= w[1].len && w[0].offset + w[0].len == w[1].offset));
    let platform = pool_platform();
    let mut count = cvs.len();
    let mut width = MIN_PIECE_LEN;
    let mut out = [0u8; crate::MAX_SIMD_DEGREE_OR_2 * crate::OUT_LEN];
    while count > 2 {
        let prefix = pieces.partition_point(|p| p.len > width);
        debug_assert!(prefix < count);
        let mut read = prefix;
        let mut write = prefix;
        while read + 1 < count {
            let children = (count - read).min(2 * crate::MAX_SIMD_DEGREE_OR_2);
            let parents = crate::compress_parents_parallel(
                cvs[read..read + children].as_flattened(), key, flags, platform, &mut out,
            );
            cvs[write..write + parents].as_flattened_mut().copy_from_slice(&out[..parents * crate::OUT_LEN]);
            read += children;
            write += parents;
        }
        if read < count {
            cvs[write] = cvs[read];
            write += 1;
        }
        count = write;
        width *= 2;
    }
    count
}

/*
 * The pool: cpus - 1 worker threads serving the registered jobs.
 *
 * The slots hold raw pointers to jobs on callers' stacks. A worker counts
 * itself as a reader of a slot before it loads the pointer and until it has
 * moved the job's cursor; a caller clears its slot once every piece is
 * taken and then waits until the slot has no reader, so no worker is left
 * mid-take on a job that is gone. A worker that took a piece touches the
 * job until its decrement of `active`, its last access to the job. Once
 * slot readers have drained, all reservations belong to workers finishing
 * pieces. The caller waits for active == 0 before returning. SeqCst on the
 * slot pointer and reader count orders publication against unregister;
 * the active counter also publishes every finished chaining value.
 *
 * Every step on the way to a piece is an atomic operation; the two locks
 * below serve sleeping and waking alone, and a thread takes one only when
 * it is about to sleep or knows a sleeper is there.
 */
struct Pool {
    slots: [Slot; MAX_JOBS],
    /// Callers inside hash_over_pool.
    callers: AtomicUsize,
    cpus: usize,
    /// Whether the SME2 thread runs (this CPU has SME2).
    sme2: bool,
    /// Workers asleep on `posted`.
    sleepers: AtomicUsize,
    /// Of those, the ones a caller has asked to wake and that have yet to.
    notified: AtomicUsize,
    /// Of those, the ones still to be woken: woken workers wake them.
    owed: AtomicUsize,
    sleep_lock: Mutex<()>,
    posted: Condvar,
    /// Callers asleep on `finished_signal`.
    waiters: AtomicUsize,
    finished: Mutex<()>,
    finished_signal: Condvar,
}

/// Jobs registered at once. A caller that finds every slot taken hashes
/// on its own thread.
const MAX_JOBS: usize = 64;

struct Slot {
    job: AtomicPtr<Job<'static>>,
    /// Workers between loading `job` and finishing their take from it.
    readers: AtomicUsize,
}

/// Start the pool: see [`crate::initialize`]. Every later call returns
/// the same pool at once.
pub(crate) fn initialize() {
    pool();
}

/// Make `mutex` and `condvar` ready now: on Apple's systems the standard
/// library allocates each on its first use, which the pool's threads would
/// otherwise reach at their first sleep, after `initialize_multithreaded`
/// returned (the SME2 thread's: 64 and 48 bytes, probe/lazy-sync, job 987).
pub(crate) fn prepare<T>(mutex: &Mutex<T>, condvar: &Condvar) {
    drop(mutex.lock());
    condvar.notify_one();
}

/// The pool, created on first use. Creation spawns the workers, then
/// returns; [`crate::initialize`] is this function's public face. Workers
/// calling `pool()` wait until the creator returns.
fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let pool = Pool {
            slots: std::array::from_fn(|_| Slot { job: AtomicPtr::new(std::ptr::null_mut()), readers: AtomicUsize::new(0) }),
            callers: AtomicUsize::new(0),
            cpus,
            sme2: cfg!(blake3_sme2) && !matches!(pool_platform(), p if core::mem::discriminant(&p) == core::mem::discriminant(&Platform::detect())),
            sleepers: AtomicUsize::new(0),
            notified: AtomicUsize::new(0),
            owed: AtomicUsize::new(0),
            sleep_lock: Mutex::new(()),
            posted: Condvar::new(),
            waiters: AtomicUsize::new(0),
            finished: Mutex::new(()),
            finished_signal: Condvar::new(),
        };
        prepare(&pool.sleep_lock, &pool.posted);
        prepare(&pool.finished, &pool.finished_signal);
        prepare(&TASKS.sme2_asleep, &TASKS.sme2_wake);
        drop(TASKS.list.lock());
        // Every thread counts itself as it starts, before it reaches
        // pool() (which waits for this creator): the creator returns once
        // all have, so the pool's start ends inside initialize_multithreaded,
        // its allocations too (std's thread start allocates 16 bytes for its
        // stack-overflow handler, which a warm queue's test counted after
        // initialize_multithreaded returned).
        static STARTED: AtomicUsize = AtomicUsize::new(0);
        let threads = usize::from(pool.sme2) + cpus - 1;
        if pool.sme2 {
            std::thread::Builder::new()
                .name("blake3-sme2".into())
                .spawn(|| {
                    STARTED.fetch_add(1, Ordering::SeqCst);
                    sme2_main()
                })
                .expect("spawning the SME2 thread");
        }
        for worker in 1..cpus {
            std::thread::Builder::new()
                .name(format!("blake3-worker-{worker}"))
                .spawn(move || {
                    STARTED.fetch_add(1, Ordering::SeqCst);
                    worker_main()
                })
                .expect("spawning a BLAKE3 worker");
        }
        while STARTED.load(Ordering::SeqCst) < threads {
            std::thread::yield_now();
        }
        pool
    })
}

impl Pool {
    /// Register `work` of `pieces` pieces, take pieces on this thread until
    /// none remain, and return once every piece is finished, on whichever
    /// thread took it. The caller first hashes pieces `0..own` on `own_platform`, back to
    /// back (the SME2 thread's prefix; the workers' cursor starts after
    /// it), then takes pieces as the workers do.
    fn run_job(&self, work: Work, pieces: usize, own: usize, own_platform: Platform) {
        let job = Job {
            work,
            pieces,
            cursor: Line(AtomicUsize::new(own)),
            active: Line(AtomicUsize::new(1)),
        };
        let slot = self.register(&job);
        for index in 0..own {
            // Sound: the cursor starts past these, so this thread alone
            // writes their slots.
            unsafe { job.hash_piece(index, own_platform) };
        }
        loop {
            let index = job.cursor.fetch_add(1, Ordering::SeqCst);
            if index >= pieces {
                break;
            }
            // Sound: this thread took index through the cursor, so it alone
            // writes slot index.
            unsafe { job.hash_piece(index, pool_platform()) };
        }
        job.active.fetch_sub(1, Ordering::SeqCst);
        // Every piece is taken; the slot has nothing more to give from this
        // job. Unregister drains readers still making a reservation; afterwards
        // active reaches zero exactly when the last piece has finished.
        if let Some(slot) = slot {
            self.unregister(slot);
        }
        self.wait_done(&job.active);
    }

    /// Put the job in a free slot. Returns the slot, or None when every
    /// slot is taken. When the workers awake are fewer than the call can
    /// use, the caller wakes one sleeper and leaves the others it needs
    /// owed: a wake costs the waker about 4 µs on the VM (all fifteen at
    /// once, 56 µs), and each woken worker wakes owed sleepers before it
    /// takes a piece, so the wakes spread over the woken.
    fn register(&self, job: &Job) -> Option<usize> {
        // The caller keeps this erased lifetime valid by unregistering,
        // draining readers, and waiting for active == 0 before returning.
        let ptr = job as *const Job as *mut Job<'static>;
        let slot = (0..MAX_JOBS).find(|&i| {
            self.slots[i].job.compare_exchange(std::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::Relaxed).is_ok()
        })?;
        self.wake_for(job.pieces - 1);
        Some(slot)
    }

    /// Make `wanted` workers awake or on their way, waking sleepers as
    /// needed: the caller wakes one, the woken wake the rest.
    fn wake_for(&self, wanted: usize) {
        // Sleepers already notified are on their way (a wake takes tens
        // of microseconds to land); a second caller in that window counts
        // them as awake and pays nothing. The lock-free reads are a hint;
        // the counts are exact under sleep_lock, where every sleeper
        // counted is waiting (see next_piece).
        let unnotified = self.sleepers.load(Ordering::SeqCst).saturating_sub(self.notified.load(Ordering::SeqCst));
        if unnotified > 0 && self.cpus - 1 - unnotified.min(self.cpus - 1) < wanted {
            let _guard = self.sleep_lock.lock().unwrap();
            let sleepers = self.sleepers.load(Ordering::SeqCst);
            let unnotified = sleepers - self.notified.load(Ordering::SeqCst);
            let awake = self.cpus - 1 - unnotified;
            if unnotified > 0 && awake < wanted {
                let wake = (wanted - awake).min(unnotified);
                self.notified.fetch_add(wake, Ordering::SeqCst);
                self.owed.fetch_add(wake - 1, Ordering::SeqCst);
                self.posted.notify_one();
            }
        }
    }

    /// Clear the slot and wait out any worker mid-take on it.
    fn unregister(&self, slot: usize) {
        self.slots[slot].job.store(std::ptr::null_mut(), Ordering::SeqCst);
        while self.slots[slot].readers.load(Ordering::SeqCst) > 0 {
            std::hint::spin_loop();
        }
    }

    /// Wait for the last active thread. The caller has exhausted the cursor,
    /// released its reservation, and unregistered the job (including draining
    /// slot readers), so no new reservations can arrive. Poll first, then
    /// sleep on the condition variable that finishing workers signal.
    fn wait_done(&self, active: &AtomicUsize) {
        let started = std::time::Instant::now();
        let mut yielded = started;
        while started.elapsed() < SPIN_BEFORE_SLEEP {
            if active.load(Ordering::SeqCst) == 0 {
                return;
            }
            poll_pause(&mut yielded);
        }
        let mut guard = self.finished.lock().unwrap();
        self.waiters.fetch_add(1, Ordering::SeqCst);
        while active.load(Ordering::SeqCst) != 0 {
            guard = self.finished_signal.wait(guard).unwrap();
        }
        self.waiters.fetch_sub(1, Ordering::SeqCst);
    }

    /// Release a finished piece's reservation, then wake sleeping callers.
    /// The decrement is this worker's last access to the job.
    fn piece_done(&self, active: &AtomicUsize) {
        active.fetch_sub(1, Ordering::SeqCst);
        if self.waiters.load(Ordering::SeqCst) > 0 {
            let _guard = self.finished.lock().unwrap();
            self.finished_signal.notify_all();
        }
    }

    /// A worker takes one piece from the first job, scanning the slots
    /// from `start`, that has one.
    /// The pointer stays valid until
    /// this worker's `piece_done` (see the pool's comment).
    fn take_piece(&self, start: &mut usize) -> Option<(*const Job<'static>, usize)> {
        for k in 0..MAX_JOBS {
            let at = (*start + k) % MAX_JOBS;
            let slot = &self.slots[at];
            // A load first: an empty slot costs the pollers no writes, so
            // a caller's registration is not fighting fifteen of them for
            // the line.
            if slot.job.load(Ordering::SeqCst).is_null() {
                continue;
            }
            slot.readers.fetch_add(1, Ordering::SeqCst);
            let ptr = slot.job.load(Ordering::SeqCst);
            let mut taken = None;
            if !ptr.is_null() {
                // Sound: a reader of the slot; the caller waits for readers.
                let job = unsafe { &*ptr };
                if job.cursor.load(Ordering::SeqCst) < job.pieces {
                    job.active.fetch_add(1, Ordering::SeqCst);
                    let index = job.cursor.fetch_add(1, Ordering::SeqCst);
                    if index < job.pieces {
                        taken = Some(index);
                    } else {
                        job.active.fetch_sub(1, Ordering::SeqCst);
                    }
                }
            }
            slot.readers.fetch_sub(1, Ordering::SeqCst);
            if let Some(index) = taken {
                *start = at + 1;
                return Some((ptr, index));
            }
        }
        None
    }

    /// The next piece for a worker, hashing the queues' tasks it finds on
    /// the way; asleep whenever it finds neither.
    fn next_piece(&self, start: &mut usize) -> (*const Job<'static>, usize) {
        loop {
            if let Some(task) = TASKS.pop() {
                task.run(pool_platform());
                TASKS.in_flight.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            if let Some(taken) = self.take_piece(start) {
                return taken;
            }
            // Nothing to take: asleep until a wake. The guard is held from
            // the count's increment through the take, the wait, and the
            // decrement, so
            // under sleep_lock every sleeper counted is waiting, and
            // notified never exceeds sleepers: a caller's wake cannot fall
            // between the check and the wait, and a sleeper that leaves
            // without waiting was never counted as notified.
            let mut guard = self.sleep_lock.lock().unwrap();
            self.sleepers.fetch_add(1, Ordering::SeqCst);
            let taken = self.take_piece(start);
            // Asleep only with nothing to take: no piece, no task queued. A
            // queue's tasks can be queued while nothing is registered (the
            // delivery thread takes its hold after the push), and a worker
            // that slept on them then would use up the push's wake.
            let waited = taken.is_none() && TASKS.queued.load(Ordering::SeqCst) == 0;
            if waited {
                guard = self.posted.wait(guard).unwrap();
                // Awake: one fewer notified sleeper on the way (a wake
                // nobody asked for leaves the count where it is).
                // Every write of `notified` holds sleep_lock, as here.
                let notified = self.notified.load(Ordering::SeqCst);
                self.notified.store(notified.saturating_sub(1), Ordering::SeqCst);
            }
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
            drop(guard);
            if waited {
                for _ in 0..self.owed.swap(0, Ordering::SeqCst) {
                    self.posted.notify_one();
                }
            }
            if let Some(taken) = taken {
                return taken;
            }
        }
    }
}

/// The longest part of a queue's input planned as tasks
/// ([`crate::plan_subtrees`]): a task is a whole subtree within one part.
pub(crate) const TASK_LEN: usize = 64 * CHUNK_LEN;

/// One piece of a queue's work (crate::queue), hashed by whichever thread
/// pops it from [`TASKS`]: a whole subtree at chunk `counter` of a message,
/// as [`crate::plan_subtrees`] cuts it, its result a 64-byte block at `out`;
/// or, with `batch` the messages' length, a range of a batch's messages in
/// their slots, their digests at `out`. Then it counts down `left`; the
/// queue keeps the bytes, `out`, and `left` in place until `left` is zero.
pub(crate) struct Task {
    pub(crate) input: *const u8,
    pub(crate) len: usize,
    pub(crate) counter: u64,
    pub(crate) batch: Option<usize>,
    pub(crate) key: crate::CVWords,
    pub(crate) flags: u8,
    pub(crate) out: *mut u8,
    pub(crate) left: *const AtomicUsize,
    /// With `members` above zero, the task is that many separate short
    /// messages instead (or, with `batch`, batches of that length's
    /// messages): each one's digest (digests) to its `out`, then its
    /// `left` counted down; `len` sums their bytes.
    pub(crate) members: usize,
    pub(crate) member: [Member; MEMBERS],
}

/// The most short messages (or small batches) one task takes.
pub(crate) const MEMBERS: usize = 64;

/// A short message in a task of several ([`Task::members`]).
#[derive(Clone, Copy)]
pub(crate) struct Member {
    pub(crate) input: *const u8,
    pub(crate) len: usize,
    pub(crate) out: *mut u8,
    pub(crate) left: *const AtomicUsize,
}

const NO_MEMBER: Member = Member { input: core::ptr::null(), len: 0, out: core::ptr::null_mut(), left: core::ptr::null() };

// Sound: a task's pointers stay valid until its `left` reaches zero (above).
unsafe impl Send for Task {}

impl Task {
    /// A task over `input` at chunk `counter`, its mode, kind, and
    /// destinations still to fill in.
    pub(crate) fn of(input: &[u8], counter: u64) -> Task {
        Task { input: input.as_ptr(), len: input.len(), counter, batch: None, key: [0; 8], flags: 0, out: core::ptr::null_mut(), left: core::ptr::null(), members: 0, member: [NO_MEMBER; MEMBERS] }
    }

    /// An empty task of short messages (with `batch`, batches of messages
    /// of that length) in the mode of `key` and `flags`.
    pub(crate) fn members(key: &crate::CVWords, flags: u8, batch: Option<usize>) -> Task {
        Task { key: *key, flags, batch, ..Task::of(&[], 0) }
    }

    /// Hash on `platform` into `out`: for a subtree of two chunks or more at
    /// chunk zero (it may be the whole message), its pair of child chaining
    /// values; for any other, its chaining value (first 32 bytes). Then
    /// count down.
    pub(crate) fn run(self, platform: Platform) {
        if self.members > 0 {
            return self.run_members(platform);
        }
        // Sound: the queue keeps these in place until `left` is zero.
        let bytes = unsafe { core::slice::from_raw_parts(self.input, self.len) };
        if let Some(message_len) = self.batch {
            let count = self.len / crate::many::slot_len(message_len);
            // Sound: `out` holds a digest per message of the range.
            let digests = unsafe { core::slice::from_raw_parts_mut(self.out as *mut [u8; crate::OUT_LEN], count) };
            crate::many::hash_many_on(bytes, message_len, &self.key, self.flags, digests, platform);
            unsafe { &*self.left }.fetch_sub(1, Ordering::Release);
            return;
        }
        // Sound: a subtree's `out` is one 64-byte block.
        let out = unsafe { &mut *(self.out as *mut [u8; crate::BLOCK_LEN]) };
        if self.counter == 0 && self.len > CHUNK_LEN {
            *out = crate::compress_subtree_to_parent_node::<crate::join::SerialJoin>(bytes, 0, &self.key, 0, self.flags, platform);
        } else {
            out[..crate::OUT_LEN].copy_from_slice(&crate::hash_all_at_once::<crate::join::SerialJoin>(bytes, &self.key, self.counter, self.flags, platform).chaining_value());
        }
        unsafe { &*self.left }.fetch_sub(1, Ordering::Release);
    }
}

impl Task {
    /// The members' digests: messages of one block side by side on the
    /// multi-lane kernels (`hash_many`'s, from a table of pointers); any
    /// others one at a time.
    fn run_members(self, platform: Platform) {
        let members = &self.member[..self.members];
        if let Some(message_len) = self.batch {
            for m in members {
                // Sound: the queue keeps each batch's bytes, digest space,
                // and `left` in place until its `left` is zero.
                let bytes = unsafe { core::slice::from_raw_parts(m.input, m.len) };
                let count = m.len / crate::many::slot_len(message_len);
                let digests = unsafe { core::slice::from_raw_parts_mut(m.out as *mut [u8; crate::OUT_LEN], count) };
                crate::many::hash_many_on(bytes, message_len, &self.key, self.flags, digests, platform);
                unsafe { &*m.left }.fetch_sub(1, Ordering::Release);
            }
            return;
        }
        // Sound: the queue keeps every member's bytes, `out`, and `left` in
        // place until its `left` is zero.
        if members.len() >= 2 && members.iter().all(|m| m.len == crate::BLOCK_LEN) {
            let table: arrayvec::ArrayVec<&[u8; crate::BLOCK_LEN], MEMBERS> =
                members.iter().map(|m| unsafe { &*(m.input as *const [u8; crate::BLOCK_LEN]) }).collect();
            let mut digests = [[0u8; crate::OUT_LEN]; MEMBERS];
            let flags = self.flags | crate::CHUNK_START | crate::CHUNK_END | crate::ROOT;
            platform.hash_many::<{ crate::BLOCK_LEN }>(&table, &self.key, 0, crate::IncrementCounter::No, flags, 0, 0, digests[..members.len()].as_flattened_mut());
            for (m, digest) in members.iter().zip(&digests) {
                unsafe { core::ptr::copy_nonoverlapping(digest.as_ptr(), m.out, crate::OUT_LEN) };
                unsafe { &*m.left }.fetch_sub(1, Ordering::Release);
            }
            return;
        }
        for m in members {
            let bytes = unsafe { core::slice::from_raw_parts(m.input, m.len) };
            let hash = crate::hash_serial_on(bytes, &self.key, self.flags, platform);
            unsafe { core::ptr::copy_nonoverlapping(hash.as_bytes().as_ptr(), m.out, crate::OUT_LEN) };
            unsafe { &*m.left }.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Every queue's tasks waiting for a worker, in the order pushed.
pub(crate) struct Tasks {
    /// Each on a line of its own: threads read `queued`, every finishing
    /// thread writes `in_flight`, and pushers and poppers take the lock.
    list: OwnLine<Mutex<std::collections::VecDeque<Task>>>,
    /// The list's length, read without the lock.
    queued: OwnLine<AtomicUsize>,
    /// Tasks pushed and not yet finished: what pushes wake threads for.
    in_flight: OwnLine<AtomicUsize>,
    /// The room the queues made in the list (make_room), under its lock.
    room: AtomicUsize,
    /// Whether the SME2 thread sleeps (read without the lock), and its wake.
    sme2_asleep: Mutex<bool>,
    sme2_sleeps: std::sync::atomic::AtomicBool,
    sme2_wake: Condvar,
}

pub(crate) static TASKS: Tasks = Tasks {
    list: OwnLine(Mutex::new(std::collections::VecDeque::new())),
    queued: OwnLine(AtomicUsize::new(0)),
    in_flight: OwnLine(AtomicUsize::new(0)),
    room: AtomicUsize::new(0),
    sme2_asleep: Mutex::new(false),
    sme2_sleeps: std::sync::atomic::AtomicBool::new(false),
    sme2_wake: Condvar::new(),
};

/// The SME2 thread, on CPUs with SME2: it hashes tasks only, on SME2 (the
/// SME unit hashes a 64 KiB piece in 14 us where a NEON worker takes 22,
/// Mac, probe/queue-timeline), under the process's turn so that a caller of
/// hash() elsewhere keeps the unit; without the turn, on NEON. It takes
/// tasks while any wait, and sleeps as soon as none does; every push wakes
/// it first.
fn sme2_main() {
    loop {
        {
            let mut asleep = TASKS.sme2_asleep.lock().unwrap();
            while TASKS.queued.load(Ordering::SeqCst) == 0 {
                *asleep = true;
                TASKS.sme2_sleeps.store(true, Ordering::SeqCst);
                // A push between the check above and this store sees the
                // flag, takes the lock after the wait releases it, and wakes.
                if TASKS.queued.load(Ordering::SeqCst) > 0 {
                    break;
                }
                asleep = TASKS.sme2_wake.wait(asleep).unwrap();
            }
            *asleep = false;
            TASKS.sme2_sleeps.store(false, Ordering::SeqCst);
        }
        let mut yielded = std::time::Instant::now();
        while TASKS.queued.load(Ordering::SeqCst) > 0 {
            match TASKS.pop() {
                Some(task) if task.members > 0 => {
                    // Short messages and small batches, gathered, run
                    // faster on NEON than through an SME2 session per task
                    // (Mac: 64-byte messages 14-15% faster, batches of 16
                    // and 64 10-15%); below 16 KiB the other members run
                    // the NEON kernels on either platform.
                    task.run(pool_platform());
                    TASKS.in_flight.fetch_sub(1, Ordering::SeqCst);
                }
                Some(task) => {
                    let turn = crate::platform::Sme2Turn::take(Platform::detect(), true);
                    task.run(turn.platform());
                    drop(turn);
                    TASKS.in_flight.fetch_sub(1, Ordering::SeqCst);
                }
                // Another thread is taking a task: poll on.
                None => poll_pause(&mut yielded),
            }
        }
    }
}

impl Tasks {
    /// Add `tasks`, and wake sleeping threads so that one per task in
    /// flight (waiting or being hashed) is awake or on its way: the SME2
    /// thread first, then workers.
    pub(crate) fn push(&self, tasks: impl ExactSizeIterator<Item = Task>) {
        let pool = pool();
        // The count, which every finishing thread writes, outside the lock
        // (inside, it held pushers and pollers up: two programs' 16 KiB
        // messages 2.0 -> 1.65 us each on the Mac, probe/submit-16k).
        let pushed = tasks.len();
        let in_flight = self.in_flight.fetch_add(pushed, Ordering::SeqCst) + pushed;
        let mut list = lock_polling(&self.list);
        list.extend(tasks);
        let queued = list.len();
        // The queues made room for every task their entries can have in
        // flight (make_room): no growth here, whatever the threads' timing.
        debug_assert!(queued <= self.room.load(Ordering::Relaxed), "the queues made room for every task");
        self.queued.store(queued, Ordering::SeqCst);
        drop(list);
        if pool.sme2 && self.sme2_sleeps.load(Ordering::SeqCst) && *self.sme2_asleep.lock().unwrap() {
            self.sme2_wake.notify_one();
        }
        pool.wake_for(in_flight.saturating_sub(usize::from(pool.sme2)));
    }

    /// Make room in the list for `more` tasks: a queue makes room for the
    /// most tasks its entries can have waiting (its slots times the most
    /// tasks a submission has had), so that the list grows with what the
    /// programs keep in flight, never with the threads' timing (a room of
    /// "every task in flight" counted tasks finished but not yet counted
    /// down, and grew the list after warm-up once in a hundred runs of
    /// tests/queue_no_alloc.rs).
    pub(crate) fn make_room(&self, more: usize) {
        let mut list = lock_polling(&self.list);
        let room = self.room.fetch_add(more, Ordering::Relaxed) + more;
        let len = list.len();
        list.reserve(room.saturating_sub(len));
    }

    /// Give back room a queue made (its capacity stays).
    pub(crate) fn give_room(&self, less: usize) {
        let _list = lock_polling(&self.list);
        self.room.fetch_sub(less, Ordering::Relaxed);
    }

    /// A waiting task, unless none waits or another thread is taking one.
    fn pop(&self) -> Option<Task> {
        if self.queued.load(Ordering::SeqCst) == 0 {
            return None;
        }
        // Another thread popping means this one would wait for it: poll on.
        let mut list = self.list.try_lock().ok()?;
        let task = list.pop_front()?;
        self.queued.store(list.len(), Ordering::SeqCst);
        Some(task)
    }
}

fn worker_main() {
    let pool = pool();
    let mut start = 0;
    loop {
        let (job_ptr, index) = pool.next_piece(&mut start);
        // Sound by the pool's contract: our active reservation keeps the job alive.
        let job = unsafe { &*job_ptr };
        unsafe { job.hash_piece(index, pool_platform()) };
        pool.piece_done(&job.active);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_next_piece_len() {
        assert_eq!(next_piece_len(64 * CHUNK_LEN, 16), MIN_PIECE_LEN);
        assert_eq!(next_piece_len(1 << 20, 16), 64 * CHUNK_LEN);
        assert_eq!(next_piece_len(3 << 20, 16), 128 * CHUNK_LEN, "192 KiB caps at the longest piece");
        assert_eq!(next_piece_len(1 << 30, 16), MAX_PIECE_LEN);
        assert_eq!(next_piece_len((3 << 20) / 4, 16), 32 * CHUNK_LEN, "48 KiB rounds down to 32 KiB");
        for remaining in [MIN_PIECE_LEN + 1, 100_000, 1 << 20, (3 << 20) + 5, 1 << 27] {
            for threads in [2, 3, 16, 64] {
                let piece = next_piece_len(remaining, threads);
                assert!(piece.is_power_of_two() && piece >= MIN_PIECE_LEN && piece <= MAX_PIECE_LEN);
            }
        }
    }

    /// Cuts with an SME2 prefix tile the input with non-increasing whole
    /// subtrees, the prefix's pieces first, at every length and share, and
    /// hash to the serial result.
    #[test]
    fn test_cut_with_prefix_shapes_and_hashes() {
        let mut input = vec![0u8; 3 * (1 << 20) + 12345];
        crate::test::paint_test_input(&mut input);
        for len in [MIN_PIECE_LEN + 1, 64 * CHUNK_LEN, 100 * CHUNK_LEN + 7, 1 << 20, 3 << 20, input.len()] {
            for threads in [2, 3, 4, 16] {
                let prefix = sme2_prefix(len, threads);
                let (pieces, own) = cut_with_prefix(len, threads, prefix);
                assert!(pieces.len() >= 2 && own < pieces.len());
                assert!(pieces[..own].iter().map(|p| p.len).sum::<usize>() <= prefix);
                let mut offset = 0;
                for (i, p) in pieces.iter().enumerate() {
                    assert_eq!(p.offset, offset);
                    if i + 1 < pieces.len() {
                        assert!(p.len.is_power_of_two() && p.offset % p.len == 0 && p.len >= pieces[i + 1].len, "{len} {threads}: {pieces:?}");
                    }
                    offset += p.len;
                }
                assert_eq!(offset, len);
                let mut cvs: Vec<ChainingValue> = pieces
                    .iter()
                    .map(|p| crate::hash_all_at_once::<crate::join::SerialJoin>(&input[p.offset..][..p.len], crate::IV, (p.offset / CHUNK_LEN) as u64, 0, Platform::detect()).chaining_value())
                    .collect();
                assert_eq!(merge_root(&pieces, &mut cvs, crate::IV, 0), crate::hash(&input[..len]), "{len} {threads}");
            }
        }
    }

    #[test]
    fn test_cut_subtrees_shapes() {
        // 64 KiB for 16 threads: eight pieces of the shortest length.
        let pieces = cut_subtrees(64 * CHUNK_LEN, 16);
        assert_eq!(pieces.len(), 8);
        assert!(pieces.iter().all(|p| p.len == MIN_PIECE_LEN));
        // 1 MiB for 16 threads: 64 KiB first, then shrinking, then a tail.
        let pieces = cut_subtrees(1 << 20, 16);
        assert_eq!(pieces[0].len, 64 * CHUNK_LEN);
        assert!(pieces.windows(2).all(|w| w[1].len <= w[0].len), "{pieces:?}");
        assert!(pieces.last().unwrap().len <= MIN_PIECE_LEN);
        for threads in [2, 3, 16, 64] {
            for len in [MIN_PIECE_LEN + 1, 100 * CHUNK_LEN + 7, 1000 * CHUNK_LEN + 3, 1 << 24, (1 << 20) + 1] {
                let pieces = cut_subtrees(len, threads);
                assert!(pieces.len() >= 2);
                assert_eq!(pieces.iter().map(|p| p.len).sum::<usize>(), len);
                for (i, p) in pieces.iter().enumerate() {
                    assert!(p.len <= MAX_PIECE_LEN);
                    if i > 0 {
                        assert_eq!(p.offset, pieces[i - 1].offset + pieces[i - 1].len);
                        let max = hazmat::max_subtree_len(p.offset as u64).unwrap();
                        assert!(p.len as u64 <= max, "piece {i} for {len}: {p:?} exceeds {max}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_hash_matches_serial() {
        let mut input = vec![0u8; (3 << 20) + 12345];
        crate::test::paint_test_input(&mut input);
        for len in [
            0,
            1,
            MIN_SPLIT_LEN - 1,
            MIN_SPLIT_LEN,
            MIN_SPLIT_LEN + 1,
            MIN_SPLIT_LEN + 64,
            MIN_SPLIT_LEN + 1023,
            MIN_SPLIT_LEN + 1024,
            MIN_SPLIT_LEN + 1025,
            2 * MIN_SPLIT_LEN + 999,
            3 << 20,
            input.len(),
        ] {
            let want = crate::hash(&input[..len]);
            assert_eq!(want, hash(&input[..len]), "len = {len}");
            let key = [42u8; KEY_LEN];
            assert_eq!(
                crate::keyed_hash(&key, &input[..len]),
                hash_with_key(&input[..len], &crate::platform::words_from_le_bytes_32(&key), crate::KEYED_HASH),
                "keyed len = {len}"
            );
            let context_key = hazmat::hash_derive_key_context("lanes test");
            assert_eq!(
                *Hasher::new_from_context_key(&context_key).update(&input[..len]).finalize().as_bytes(),
                *hash_with_key(&input[..len], &crate::platform::words_from_le_bytes_32(&context_key), crate::DERIVE_KEY_MATERIAL).as_bytes(),
                "derive len = {len}"
            );
        }
    }

    /// `Hasher::update_multithreaded` against `hash`, fed in pieces of
    /// several sizes after a first piece that leaves the counter unaligned
    /// (so `update` shrinks the subtrees it hands the pool), keyed too.
    #[test]
    fn test_update_multithreaded_matches_hash() {
        let mut input = vec![0u8; (5 * MIN_SPLIT_LEN + 1025).max(3 << 20) + 12345];
        crate::test::paint_test_input(&mut input);
        let key = [42u8; KEY_LEN];
        for len in [0, 1, MIN_SPLIT_LEN, MIN_SPLIT_LEN + 1, 5 * MIN_SPLIT_LEN + 1025, 3 << 20, input.len()] {
            for first in [0, 1, 1024, 3000, MIN_SPLIT_LEN] {
                for piece in [1000, MIN_SPLIT_LEN, 4 * MIN_SPLIT_LEN, 1 << 20, usize::MAX] {
                    let first = first.min(len);
                    let mut hasher = Hasher::new();
                    let mut keyed = Hasher::new_keyed(&key);
                    hasher.update_multithreaded(&input[..first]);
                    keyed.update_multithreaded(&input[..first]);
                    for part in input[first..len].chunks(piece.min(len.max(1))) {
                        hasher.update_multithreaded(part);
                        keyed.update_multithreaded(part);
                    }
                    let what = format!("len {len}, first {first}, pieces of {piece}");
                    assert_eq!(hasher.finalize(), crate::hash(&input[..len]), "{what}");
                    assert_eq!(keyed.finalize(), crate::keyed_hash(&key, &input[..len]), "keyed {what}");
                }
            }
        }
    }

    /// Every piece length the cut can produce, through cut and merge alone
    /// (independent of how many CPUs the test machine has), on both
    /// kernel platforms.
    #[test]
    fn test_merge_every_cut() {
        for len in [70 * CHUNK_LEN + 5, 100 * CHUNK_LEN + 7, 256 * CHUNK_LEN, 1000 * CHUNK_LEN] {
            let mut input = vec![0u8; len];
            crate::test::paint_test_input(&mut input);
            let want = crate::hash(&input);
            for threads in [2, 3, 5, 16, 64] {
                let pieces = cut_subtrees(len, threads);
                for platform in [Platform::detect(), pool_platform()] {
                    let mut cvs: Vec<ChainingValue> = pieces
                        .iter()
                        .map(|p| {
                            let mut hasher = Hasher::new();
                            hasher.set_platform(platform);
                            hasher.set_input_offset(p.offset as u64);
                            hasher.update(&input[p.offset..][..p.len]);
                            hasher.finalize_non_root()
                        })
                        .collect();
                    let split = pieces.partition_point(|p| p.offset < hazmat::left_subtree_len(len as u64) as usize);
                    let left = subtree_cv(&pieces[..split], &cvs[..split], Mode::Hash);
                    let right = subtree_cv(&pieces[split..], &cvs[split..], Mode::Hash);
                    assert_eq!(want, hazmat::merge_subtrees_root(&left, &right, Mode::Hash));
                    assert_eq!(want, merge_root(&pieces, &mut cvs, crate::IV, 0), "len = {len}, threads = {threads}");
                }
            }
        }
    }

    /// Arbitrary CVs isolate the merge's tree shape from chunk hashing.
    /// The larger cuts cross the fixed scratch buffer's batching boundary.
    #[test]
    fn test_simd_merge_matches_recursive() {
        let key = [93u8; KEY_LEN];
        let context_key = hazmat::hash_derive_key_context("parallel merge test");
        for threads in [2, 3, 16, 64, 512] {
            for base in [MIN_SPLIT_LEN, 1 << 20, 8 << 20, 64 << 20] {
                for tail in [0, 1, 63, 64, 1023, 1024, 1025, 8191] {
                    let len = base + tail;
                    let pieces = cut_subtrees(len, threads);
                    let mut cvs = vec![[0u8; crate::OUT_LEN]; pieces.len()];
                    crate::test::paint_test_input(cvs.as_flattened_mut());
                    let split = pieces.partition_point(|p| p.offset < hazmat::left_subtree_len(len as u64) as usize);
                    for mode in [Mode::Hash, Mode::KeyedHash(&key), Mode::DeriveKeyMaterial(&context_key)] {
                        let left = subtree_cv(&pieces[..split], &cvs[..split], mode);
                        let right = subtree_cv(&pieces[split..], &cvs[split..], mode);
                        let want = hazmat::merge_subtrees_root(&left, &right, mode);
                        assert_eq!(want, merge_root(&pieces, &mut cvs.clone(), &mode.key_words(), mode.flags_byte()), "len={len}, threads={threads}");
                    }
                }
            }
        }
    }

    /// Many concurrent callers on one process: every result is right, and
    /// each call waits for its workers and leaves every result complete.
    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "spawns threads, which this target lacks")]
    fn test_concurrent_callers_agree() {
        let mut input = vec![0u8; 8 * MIN_SPLIT_LEN + 1];
        crate::test::paint_test_input(&mut input);
        let want = crate::hash(&input);
        let input = &input[..];
        let barrier = std::sync::Barrier::new(32);
        std::thread::scope(|scope| {
            for _ in 0..32 {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..20 {
                        assert_eq!(want, hash(input));
                    }
                });
            }
        });
        // Global pool counters can change as other tests start calls.
        // Each call's digest assertion above checks its own completion.
    }

    /// Message ranges: every message in exactly one range, ranges shrink
    /// toward the end, and a batch that fits one range stays whole.
    #[test]
    fn test_cut_messages_shapes() {
        for message_len in [crate::BLOCK_LEN, 256, 3 * CHUNK_LEN + 5] {
            let count = 40 * MIN_SPLIT_LEN / message_len;
            let pieces = cut_messages(count, message_len, 16);
            assert!(pieces.len() >= 16, "{} ranges of {message_len}-byte messages", pieces.len());
            assert_eq!(pieces[0].offset, 0);
            assert!(pieces.windows(2).all(|w| w[0].offset + w[0].len == w[1].offset && w[0].len >= w[1].len));
            assert_eq!(pieces.last().unwrap().offset + pieces.last().unwrap().len, count);
            let bytes = |p: &Piece| p.len * message_len;
            assert!(pieces[..pieces.len() - 1].iter().all(|p| bytes(p) <= MAX_PIECE_LEN && bytes(p) >= MIN_PIECE_LEN / 2));
        }
        assert_eq!(cut_messages(100, crate::BLOCK_LEN, 16).len(), 1);
    }

    /// The pool gives the single-threaded digests, for one-block
    /// messages, whole-block messages, and messages longer than a chunk.
    #[test]
    fn test_hash_many_pool_agrees() {
        let mut buffer = vec![0u8; 4 * MIN_SPLIT_LEN];
        crate::test::paint_test_input(&mut buffer);
        for message_len in [crate::BLOCK_LEN, 256, 100, 3 * CHUNK_LEN + 5] {
            let slot = crate::many::slot_len(message_len);
            let count = buffer.len() / slot;
            let mut owned = buffer[..count * slot].to_vec();
            for message in owned.chunks_exact_mut(slot) {
                message[message_len..].fill(0);
            }
            let input = &owned[..];
            let mut want = vec![[0u8; crate::OUT_LEN]; count];
            crate::hash_many(input, message_len, &mut want);
            for (i, digest) in want.iter().enumerate().take(200) {
                assert_eq!(*digest, *crate::hash(&input[i * slot..][..message_len]).as_bytes(), "message {i}");
            }
            let mut got = vec![[0u8; crate::OUT_LEN]; count];
            hash_many(input, message_len, crate::IV, 0, &mut got);
            assert_eq!(want, got, "{message_len}-byte messages");
        }
    }

    /// Concurrent batch callers beside tree callers: every digest right.
    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "spawns threads, which this target lacks")]
    fn test_concurrent_batch_callers_agree() {
        let mut buffer = vec![0u8; 8 * MIN_SPLIT_LEN];
        crate::test::paint_test_input(&mut buffer);
        let count = buffer.len() / crate::BLOCK_LEN;
        let mut want = vec![[0u8; crate::OUT_LEN]; count];
        crate::hash_many(&buffer, crate::BLOCK_LEN, &mut want);
        let tree_want = crate::hash(&buffer);
        let (want, buffer) = (&want[..], &buffer[..]);
        let barrier = std::sync::Barrier::new(16);
        std::thread::scope(|scope| {
            for i in 0..16 {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let mut got = vec![[0u8; crate::OUT_LEN]; count];
                    for _ in 0..10 {
                        if i % 2 == 0 {
                            hash_many(buffer, crate::BLOCK_LEN, crate::IV, 0, &mut got);
                            assert_eq!(want, &got[..]);
                        } else {
                            assert_eq!(tree_want, hash(buffer));
                        }
                    }
                });
            }
        });
    }
}

/// Proofs (Kani, `cargo kani`; QUALITY.md says how to install it):
/// properties the pool's cuts rest on, for every input, where the tests
/// check chosen points.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// A piece is a power of two between the bounds, and at most a
    /// thread's share of what remains unless that share is below the
    /// shortest piece: for every remaining length and 1 to 256 threads.
    #[kani::proof]
    #[kani::solver(cadical)]
    fn next_piece_len_bounds() {
        let remaining: usize = kani::any();
        let threads: usize = kani::any();
        kani::assume(threads > 0 && threads <= 256);
        let piece = next_piece_len(remaining, threads);
        assert!(piece.is_power_of_two() && MIN_PIECE_LEN <= piece && piece <= MAX_PIECE_LEN);
        assert!(piece <= (remaining / threads).max(MIN_PIECE_LEN));
        assert!(piece * 2 > (remaining / threads).clamp(MIN_PIECE_LEN, MAX_PIECE_LEN));
    }

    /// One step of cut_with_prefix's pool loop keeps its invariant: the
    /// offset is a multiple of `cap`, a power of two (the last piece's
    /// length, or MAX_PIECE_LEN at the start). The next piece is a power
    /// of two no longer than `cap`, so the offset is a multiple of it and
    /// it is a whole subtree there, and it becomes the next `cap`. By
    /// induction every piece of the loop is a whole subtree at its offset,
    /// no longer than the one before, for every length.
    #[kani::proof]
    #[kani::solver(cadical)]
    fn cut_step_keeps_whole_subtrees() {
        let len: usize = kani::any();
        let offset: usize = kani::any();
        let cap: usize = kani::any();
        let threads: usize = kani::any();
        kani::assume(threads >= 2 && threads <= 256);
        kani::assume(cap.is_power_of_two() && cap <= MAX_PREFIX_PIECE_LEN.max(MAX_PIECE_LEN));
        kani::assume(offset & (cap - 1) == 0 && offset < len);
        let piece = next_piece_len(len - offset, threads).min(cap);
        assert!(piece.is_power_of_two() && piece <= cap && offset & (piece - 1) == 0);
    }
}
