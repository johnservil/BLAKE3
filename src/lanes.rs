//! Multithreaded hashing over every CPU, with the machine's SME2 units
//! shared out as permits and concurrent callers served round-robin.
//!
//! This module is crate-internal. The public entry points are
//! [`crate::hash_multithreaded`] and
//! [`crate::hash_multithreaded_with_budget`], whose contracts speak of
//! threads alone; pieces, permits, and the pool are how those contracts
//! are met, and this file is where they are explained.
//!
//! # Pieces
//!
//! The input is cut at chunk boundaries into pieces, each a valid BLAKE3
//! subtree (see [`hazmat::left_subtree_len`]), from the front: each piece
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
//! to finish. The same count enforces the call's thread budget. Workers,
//! started once per process ([`crate::initialize`]), one per CPU beyond
//! the first, serve the registered jobs round-robin, one piece at a time
//! through the same cursor. Two callers
//! hashing at once therefore each get about half the workers' time, and
//! a slow thread takes fewer pieces than a fast one; there is nothing to
//! tune for fairness or balance.
//!
//! A call arriving when callers already fill the CPUs hashes its input
//! whole on its own thread. This check happens once per call; workers
//! need only the job's cursor and thread budget for each piece. The fixed
//! pool and the calling threads can overlap; the OS schedules them.
//! [`crate::hash_multithreaded_with_budget`] bounds the threads hashing
//! one call's pieces at once, including its caller.
//!
//! # Kernels
//!
//! On a CPU with SME2, the SME2 kernel runs about half again as fast as
//! the NEON hybrids, and the SME unit is shared by the cores of a cluster:
//! a second SME2 thread on the same cluster adds nothing. The pool holds
//! one permit per SME unit ([`sme_permits`]); a thread takes one before a
//! piece when one is free and hashes that piece with the SME2 kernels,
//! otherwise with the NEON hybrids, which every core runs at full speed.
//! The unit count is the cluster count on Apple systems (`sysctl
//! hw.perflevelN`) and measured once on Linux: the aggregate rate of `n`
//! SME2 threads at once stops growing at the unit count. Without SME2
//! every piece runs on the detected platform.
//!
//! # Waiting
//!
//! A worker between pieces polls the slots, yielding the CPU between polls
//! ([`std::thread::yield_now`]) so a runnable thread on the same CPU goes
//! first, and sleeps on a condition variable once no call has registered
//! a job for [`SPIN_BEFORE_SLEEP`]; a caller waiting for its last pieces
//! polls the same way and then sleeps. Waking a sleeping thread costs the
//! waker ten microseconds or more on some systems and the sleeper arrives
//! tens of microseconds later, so a call whose pieces outnumber the
//! workers awake wakes every sleeper at once, and workers stay awake
//! while calls keep coming; the cost falls on the first call after a
//! pause, once.
//!
//! A thread that spins without yielding holds its CPU for a whole
//! scheduler quantum; callers that spin-wait on something while this
//! module hashes on their behalf delay their own pieces by that much
//! (measured on a two-CPU machine: two spinning threads beside a 128 KiB
//! split took the call from 29 µs to 2 ms; the same two threads yielding
//! between polls left it at 29 µs).

use crate::hazmat::{self, ChainingValue, Mode};
#[cfg(test)]
use crate::hazmat::HasherExt;
use crate::platform::Platform;
use crate::{CHUNK_LEN, Hash};
#[cfg(test)]
use crate::{Hasher, KEY_LEN};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Inputs below this length are hashed on the calling thread. A split
/// hands pieces to workers that are polling, a few microseconds each way;
/// 64 KiB takes about 14 µs on one SME2 thread, and as eight pieces it
/// comes back sooner (VM, beside a copy of itself: 0.19 ns/B against
/// 0.21; 32 KiB and 48 KiB came back later split than whole).
pub(crate) const MIN_SPLIT_LEN: usize = 64 * 1024;

/// The shortest piece: eight chunks, a hybrid kernel's worth.
const MIN_PIECE_LEN: usize = 8 * CHUNK_LEN;

/// The longest piece: one SME2 kernel call of the platform's degree.
const MAX_PIECE_LEN: usize = 128 * CHUNK_LEN;

/// Pieces shorter than this run on the NEON hybrids without asking for
/// an SME2 permit: a piece on SME2 pays one switch out of streaming mode
/// and back (about a microsecond on an M4, several on a virtual machine),
/// which at 8 KiB is what SME2 would gain over NEON.
const MIN_SME_PIECE_LEN: usize = 16 * CHUNK_LEN;

/// How long a worker or a waiting caller spins before sleeping. Long
/// enough to bridge the gap between back-to-back hashes in a busy caller;
/// short enough that an idle process is quiet within it.
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

/// Hash `input` over the machine's threads, at most `max_threads` of them
/// holding a piece at once (the caller's included; at least 1).
#[inline]
pub(crate) fn hash(input: &[u8], max_threads: usize) -> Hash {
    hash_with_mode(input, Mode::Hash, max_threads)
}

/// Hash `input` over the machine's threads in the given mode.
#[inline]
pub(crate) fn hash_with_mode(input: &[u8], mode: Mode, max_threads: usize) -> Hash {
    assert!(max_threads >= 1, "a hash needs at least the calling thread");
    // The short path first and inline, so a small input costs what hash()
    // costs; everything the pool needs is behind the call below.
    if input.len() < MIN_SPLIT_LEN || max_threads == 1 {
        let (key, flags) = (mode.key_words(), mode.flags_byte());
        return crate::hash_serial(input, &key, flags);
    }
    hash_over_pool(input, mode, max_threads)
}

#[inline(never)]
fn hash_over_pool(input: &[u8], mode: Mode, max_threads: usize) -> Hash {
    let pool = pool();
    let callers = pool.callers.fetch_add(1, Ordering::SeqCst) + 1;
    let _caller = Caller(&pool.callers);
    if callers >= pool.cpus {
        return pool.hash_subtree(input, &mode.key_words(), 0, mode.flags_byte()).root_hash();
    }
    let pieces = cut_subtrees(input.len(), pool.cpus.min(max_threads));
    let mut cvs = vec![ChainingValue::default(); pieces.len()];
    let work = Work::Tree {
        input,
        pieces: &pieces,
        cvs: cvs.as_mut_ptr(),
        key: mode.key_words(),
        flags: mode.flags_byte(),
    };
    pool.run_job(work, pieces.len(), max_threads);
    merge_root(&pieces, &mut cvs, mode)
}

/// Hash every message of `inputs` into `outputs` over the machine's
/// threads, at most `max_threads` holding a range at once (the caller's
/// included; at least 1). Same digests as [`crate::hash_many`].
#[inline]
pub(crate) fn hash_many(inputs: &[&[u8]], outputs: &mut [Hash], max_threads: usize) {
    assert!(max_threads >= 1, "a hash needs at least the calling thread");
    assert_eq!(inputs.len(), outputs.len(), "one output per message");
    let total: usize = inputs.iter().map(|m| m.len()).sum();
    if total < MIN_SPLIT_LEN || max_threads == 1 || inputs.len() < 2 {
        return crate::many::hash_many_on(inputs, outputs, Platform::detect());
    }
    hash_many_over_pool(inputs, outputs, total, max_threads)
}

#[inline(never)]
fn hash_many_over_pool(inputs: &[&[u8]], outputs: &mut [Hash], total: usize, max_threads: usize) {
    let pool = pool();
    let callers = pool.callers.fetch_add(1, Ordering::SeqCst) + 1;
    let _caller = Caller(&pool.callers);
    if callers >= pool.cpus {
        return pool.hash_messages(inputs, outputs, total);
    }
    let pieces = cut_messages(inputs, total, pool.cpus.min(max_threads));
    if pieces.len() < 2 {
        return pool.hash_messages(inputs, outputs, total);
    }
    let work = Work::Messages { inputs, pieces: &pieces, outputs: outputs.as_mut_ptr() };
    pool.run_job(work, pieces.len(), max_threads);
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
    /// Thread reservations, including the caller while it takes pieces.
    /// Failed reservations undo their increment before releasing the slot
    /// reader. Accepted reservations stay within max_threads.
    active: Line,
    /// Bound on accepted reservations.
    max_threads: usize,
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
        flags: u8,
    },
    /// Ranges of a batch of messages (`offset` and `len` count messages);
    /// one digest per message.
    Messages {
        inputs: &'a [&'a [u8]],
        pieces: &'a [Piece],
        outputs: *mut Hash,
    },
}

/// An atomic counter on its own cache line: `cursor` and `active` are
/// each hit by every thread of a job, and sharing a line would make
/// each update evict the others.
#[repr(align(128))]
struct Line(AtomicUsize);

impl std::ops::Deref for Line {
    type Target = AtomicUsize;
    fn deref(&self) -> &AtomicUsize {
        &self.0
    }
}

/// Reserve one thread before taking a piece, staying within the cap.
/// The caller releases it after the piece, or after finding no piece.
/// `max_threads` is positive. Failed attempts undo their increment before
/// returning; only successful attempts may hash a piece.
fn reserve_thread(active: &AtomicUsize, max_threads: usize) -> bool {
    debug_assert!(max_threads > 0);
    if active.fetch_add(1, Ordering::SeqCst) < max_threads {
        true
    } else {
        active.fetch_sub(1, Ordering::SeqCst);
        false
    }
}

impl Job<'_> {
    /// Hash piece `index` into its slot. The caller has taken `index`
    /// through `cursor`.
    unsafe fn hash_piece(&self, index: usize) {
        match &self.work {
            Work::Tree { input, pieces, cvs, key, flags } => {
                let piece = pieces[index];
                let bytes = &input[piece.offset..][..piece.len];
                let cv = pool().hash_subtree(bytes, key, (piece.offset / CHUNK_LEN) as u64, *flags).chaining_value();
                unsafe { *cvs.add(index) = cv };
            }
            Work::Messages { inputs, pieces, outputs } => {
                let piece = pieces[index];
                let messages = &inputs[piece.offset..][..piece.len];
                // Sound: this range of outputs belongs to piece `index` alone.
                let digests = unsafe { core::slice::from_raw_parts_mut(outputs.add(piece.offset), piece.len) };
                let bytes = messages.iter().map(|m| m.len()).sum();
                pool().hash_messages(messages, digests, bytes);
            }
        }
    }
}

/// The platform a piece runs on with an SME2 permit: the detected one.
fn fast_platform() -> Platform {
    Platform::detect()
}

/// The platform a piece runs on without a permit. On an SME2 CPU the NEON
/// hybrids, which no other core slows down; elsewhere the detected one.
fn other_platform() -> Platform {
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

/// Cut a batch of messages (`total` bytes in all) into ranges for
/// `threads` threads, in order: each range gathers messages until it
/// holds [`next_piece_len`] of the bytes that remain, so ranges shrink
/// toward the end as subtree pieces do. Every message lands in one range.
fn cut_messages(inputs: &[&[u8]], total: usize, threads: usize) -> Vec<Piece> {
    let mut pieces = Vec::with_capacity(64);
    let mut remaining = total;
    let mut start = 0;
    while start < inputs.len() {
        let target = next_piece_len(remaining, threads);
        let mut end = start;
        let mut bytes = 0;
        while end < inputs.len() && (end == start || bytes + inputs[end].len() <= target) {
            bytes += inputs[end].len();
            end += 1;
        }
        pieces.push(Piece { offset: start, len: end - start });
        remaining -= bytes;
        start = end;
    }
    pieces
}

/// Cut `len` bytes into subtrees for `threads` threads, in offset order:
/// each piece is [`next_piece_len`] of what remains, and the last is
/// whatever is left. A piece's length never exceeds the one before it and
/// every piece starts at a multiple of its own length, so each is a whole
/// subtree at its offset. `len` exceeds [`MIN_PIECE_LEN`] and at least
/// two threads share it, so two or more pieces come back.
fn cut_subtrees(len: usize, threads: usize) -> Vec<Piece> {
    assert!(threads >= 2, "a serial call hashes the input whole");
    assert!(len > MIN_PIECE_LEN, "one piece has no parent to merge; hash it whole instead");
    let mut pieces = Vec::with_capacity(64);
    let mut offset = 0;
    while len - offset > MIN_PIECE_LEN {
        let piece = next_piece_len(len - offset, threads);
        if len - offset <= piece {
            break;
        }
        pieces.push(Piece { offset, len: piece });
        offset += piece;
    }
    pieces.push(Piece { offset, len: len - offset });
    pieces
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
/// subtree. NEON on SME machines avoids a streaming-mode transition for
/// these small batches; other machines use their detected SIMD kernels.
fn merge_root(pieces: &[Piece], cvs: &mut [ChainingValue], mode: Mode) -> Hash {
    debug_assert_eq!(pieces.len(), cvs.len());
    debug_assert!(pieces.len() >= 2);
    debug_assert_eq!(pieces[0].offset, 0);
    debug_assert!(pieces.last().unwrap().len > 0);
    debug_assert!(pieces[..pieces.len() - 1].iter().all(|p| p.len >= MIN_PIECE_LEN));
    debug_assert!(pieces.windows(2).all(|w| w[0].len.is_power_of_two()
        && w[0].len >= w[1].len && w[0].offset + w[0].len == w[1].offset));
    let (key, flags) = (mode.key_words(), mode.flags_byte());
    let platform = other_platform();
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
                cvs[read..read + children].as_flattened(), &key, flags, platform, &mut out,
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
    hazmat::merge_subtrees_root(&cvs[0], &cvs[1], mode)
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
    /// SME2 permits free.
    sme_free: AtomicUsize,
    cpus: usize,
    /// When the last job was registered, in nanoseconds of `epoch`; a
    /// worker sleeps only when this is SPIN_BEFORE_SLEEP old.
    last_job_ns: AtomicU64,
    epoch: std::time::Instant,
    /// Workers asleep on `posted`.
    sleepers: AtomicUsize,
    /// Of those, the ones a caller has notified and that have yet to wake.
    notified: AtomicUsize,
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

/// The pool, created on first use. Creation counts the SME units (a
/// measurement of up to tens of milliseconds where the platform reports
/// no cluster topology, see [`sme_permits`]) and spawns the workers, then
/// returns; [`crate::initialize`] is this function's public face. Workers
/// calling `pool()` wait until the creator returns.
fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let pool = Pool {
            slots: std::array::from_fn(|_| Slot { job: AtomicPtr::new(std::ptr::null_mut()), readers: AtomicUsize::new(0) }),
            callers: AtomicUsize::new(0),
            sme_free: AtomicUsize::new(sme_permits()),
            cpus,
            last_job_ns: AtomicU64::new(0),
            epoch: std::time::Instant::now(),
            sleepers: AtomicUsize::new(0),
            notified: AtomicUsize::new(0),
            sleep_lock: Mutex::new(()),
            posted: Condvar::new(),
            waiters: AtomicUsize::new(0),
            finished: Mutex::new(()),
            finished_signal: Condvar::new(),
        };
        for worker in 1..cpus {
            std::thread::Builder::new()
                .name(format!("blake3-worker-{worker}"))
                .spawn(worker_main)
                .expect("spawning a BLAKE3 worker");
        }
        pool
    })
}

impl Pool {
    /// Hash a subtree with the same kernel policy on callers and workers.
    /// `input` tiles a valid subtree at `counter`, as hash_all_at_once requires.
    /// Even a call hashing alone shares SME permits with the pool's jobs.
    fn hash_subtree(&self, input: &[u8], key: &crate::CVWords, counter: u64, flags: u8) -> crate::Output {
        let permit = input.len() >= MIN_SME_PIECE_LEN && self.take_sme_permit();
        let platform = if permit { fast_platform() } else { other_platform() };
        let output = crate::hash_all_at_once::<crate::join::SerialJoin>(input, key, counter, flags, platform);
        if permit {
            self.release_sme_permit();
        }
        output
    }

    /// Hash a batch of messages (`bytes` in all) with the same kernel
    /// policy as [`Pool::hash_subtree`].
    fn hash_messages(&self, inputs: &[&[u8]], outputs: &mut [Hash], bytes: usize) {
        let permit = bytes >= MIN_SME_PIECE_LEN && self.take_sme_permit();
        let platform = if permit { fast_platform() } else { other_platform() };
        crate::many::hash_many_on(inputs, outputs, platform);
        if permit {
            self.release_sme_permit();
        }
    }

    /// Register `work` of `pieces` pieces, take pieces on this thread until
    /// none remain, and return once every piece is finished, on whichever
    /// thread took it. At most `max_threads` threads hold a piece at once.
    fn run_job(&self, work: Work, pieces: usize, max_threads: usize) {
        let job = Job {
            work,
            pieces,
            cursor: Line(AtomicUsize::new(0)),
            active: Line(AtomicUsize::new(1)),
            max_threads,
        };
        let slot = self.register(&job);
        loop {
            let index = job.cursor.fetch_add(1, Ordering::SeqCst);
            if index >= pieces {
                break;
            }
            // Sound: this thread took index through the cursor, so it alone
            // writes slot index.
            unsafe { job.hash_piece(index) };
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
    /// slot is taken. When the workers awake are fewer than the pieces,
    /// every sleeper is woken, in one system call: a wake costs the waker
    /// ten microseconds or more on some machines and the sleeper arrives
    /// tens of microseconds later, so one call pays once for all, and
    /// the woken workers then stay awake while calls keep coming.
    fn register(&self, job: &Job) -> Option<usize> {
        // The caller keeps this erased lifetime valid by unregistering,
        // draining readers, and waiting for active == 0 before returning.
        let ptr = job as *const Job as *mut Job<'static>;
        let slot = (0..MAX_JOBS).find(|&i| {
            self.slots[i].job.compare_exchange(std::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::Relaxed).is_ok()
        })?;
        self.last_job_ns.store(self.epoch.elapsed().as_nanos() as u64, Ordering::SeqCst);
        // Sleepers already notified are on their way (a wake takes tens
        // of microseconds to land); a second caller in that window counts
        // them as awake and pays nothing. The lock-free reads are a hint;
        // the counts are exact under sleep_lock, where every sleeper
        // counted is waiting (see next_piece).
        if self.sleepers.load(Ordering::SeqCst) > self.notified.load(Ordering::SeqCst) {
            let _guard = self.sleep_lock.lock().unwrap();
            let sleepers = self.sleepers.load(Ordering::SeqCst);
            let unnotified = sleepers - self.notified.load(Ordering::SeqCst);
            if unnotified > 0 && (self.cpus - 1 - unnotified) < job.pieces {
                self.notified.store(sleepers, Ordering::SeqCst);
                self.posted.notify_all();
            }
        }
        Some(slot)
    }

    /// Whether a job was registered within SPIN_BEFORE_SLEEP: while calls
    /// keep coming, workers stay awake. A registration between this
    /// thread's clock read and its load of `last_job_ns` reads as recent.
    fn jobs_recently(&self) -> bool {
        let now = self.epoch.elapsed().as_nanos() as u64;
        let last = self.last_job_ns.load(Ordering::SeqCst);
        now.saturating_sub(last) < SPIN_BEFORE_SLEEP.as_nanos() as u64
    }

    /// Clear the slot and wait out any worker mid-take on it.
    fn unregister(&self, slot: usize) {
        self.slots[slot].job.store(std::ptr::null_mut(), Ordering::SeqCst);
        while self.slots[slot].readers.load(Ordering::SeqCst) > 0 {
            std::hint::spin_loop();
        }
    }

    /// One SME2 permit, when one is free.
    fn take_sme_permit(&self) -> bool {
        let mut free = self.sme_free.load(Ordering::Relaxed);
        while free > 0 {
            match self.sme_free.compare_exchange_weak(free, free - 1, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(now) => free = now,
            }
        }
        false
    }

    fn release_sme_permit(&self) {
        self.sme_free.fetch_add(1, Ordering::AcqRel);
    }

    /// Wait for the last active thread. The caller has exhausted the cursor,
    /// released its reservation, and unregistered the job (including draining
    /// slot readers), so no new reservations can arrive. Poll first, then
    /// sleep on the condition variable that finishing workers signal.
    fn wait_done(&self, active: &AtomicUsize) {
        let started = std::time::Instant::now();
        while started.elapsed() < SPIN_BEFORE_SLEEP {
            if active.load(Ordering::SeqCst) == 0 {
                return;
            }
            std::thread::yield_now();
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
    /// from `start`, that has one and room under its thread cap.
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
                if job.cursor.load(Ordering::SeqCst) < job.pieces
                    && job.active.load(Ordering::SeqCst) < job.max_threads
                {
                    if reserve_thread(&job.active, job.max_threads) {
                        let index = job.cursor.fetch_add(1, Ordering::SeqCst);
                        if index < job.pieces {
                            taken = Some(index);
                        } else {
                            job.active.fetch_sub(1, Ordering::SeqCst);
                        }
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

    /// The next piece for a worker: polled for a while (yielding the CPU
    /// between polls, so a thread that has work on this CPU runs), then
    /// waited for.
    fn next_piece(&self, start: &mut usize) -> (*const Job<'static>, usize) {
        let taken = 'taken: loop {
            let started = std::time::Instant::now();
            while started.elapsed() < SPIN_BEFORE_SLEEP || self.jobs_recently() {
                if let Some(taken) = self.take_piece(start) {
                    break 'taken taken;
                }
                std::thread::yield_now();
            }
            // Asleep until a wake, then back to polling: a woken worker
            // that finds nothing yet stays available for the next call.
            // The guard is held from the count's increment through the
            // take, the wait, and the decrement, so under sleep_lock every
            // sleeper counted is waiting, and notified never exceeds
            // sleepers: a caller's wake cannot fall between the take and
            // the wait, and a sleeper that leaves without waiting was
            // never counted as notified.
            let mut guard = self.sleep_lock.lock().unwrap();
            self.sleepers.fetch_add(1, Ordering::SeqCst);
            let taken = self.take_piece(start);
            if taken.is_none() {
                guard = self.posted.wait(guard).unwrap();
                // Awake: one fewer notified sleeper on the way (a wake
                // nobody asked for leaves the count where it is).
                let _ = self.notified.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| Some(n.saturating_sub(1)));
            }
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
            drop(guard);
            if let Some(taken) = taken {
                break 'taken taken;
            }
        };
        taken
    }
}

fn worker_main() {
    let pool = pool();
    let mut start = 0;
    loop {
        let (job_ptr, index) = pool.next_piece(&mut start);
        // Sound by the pool's contract: our active reservation keeps the job alive.
        let job = unsafe { &*job_ptr };
        unsafe { job.hash_piece(index) };
        pool.piece_done(&job.active);
    }
}

/// Whether pieces take SME2 permits: the detected platform is SME2.
fn uses_sme_permits() -> bool {
    #[cfg(blake3_sme2)]
    if matches!(Platform::detect(), Platform::SME2) {
        return true;
    }
    false
}

/// The number of SME2 permits: the SME units this machine has, when the
/// detected platform is SME2; otherwise nothing takes one, so any value
/// serves and 0 says so. Calls the measurement where one is needed.
fn sme_permits() -> usize {
    if uses_sme_permits() {
        #[cfg(blake3_sme2)]
        return sme_unit_count().max(1);
    }
    0
}

/// See the module docs: the platform's topology where it names clusters,
/// else a measurement.
#[cfg(blake3_sme2)]
fn sme_unit_count() -> usize {
    if let Some(count) = platform_cluster_count() {
        return count;
    }
    measure_sme_units()
}

/// How many threads can run the SME2 kernel at full speed at once: the
/// aggregate rate of `n` threads hashing at the same time, over the rate
/// of one, at the `n` where it stops growing. Each timing covers `REPEAT`
/// hashes of 256 KiB (about a millisecond); the best of `ROUNDS` is
/// kept, which rides out a hypervisor descheduling a virtual CPU. The
/// whole measurement takes about 40 ms on a 16-CPU virtual machine, the
/// "tens of milliseconds" in [`crate::initialize`]'s contract.
#[cfg(blake3_sme2)]
fn measure_sme_units() -> usize {
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};
    const LEN: usize = 256 * CHUNK_LEN;
    const REPEAT: usize = 20;
    const ROUNDS: usize = 2;
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let input: Arc<Vec<u8>> = Arc::new((0..LEN as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect());
    let timed = |input: &[u8]| -> Duration {
        let started = Instant::now();
        for _ in 0..REPEAT {
            let _ = std::hint::black_box(crate::hash(std::hint::black_box(input)));
        }
        started.elapsed()
    };
    // The slowest thread's time, best of ROUNDS, for n threads at once.
    let pair_time = |n: usize| -> Duration {
        let mut best = Duration::MAX;
        for _ in 0..ROUNDS {
            let barrier = Arc::new(Barrier::new(n));
            let threads: Vec<_> = (0..n)
                .map(|_| {
                    let input = Arc::clone(&input);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        timed(&input)
                    })
                })
                .collect();
            let slowest = threads.into_iter().map(|t| t.join().expect("SME probe thread")).max().unwrap();
            best = best.min(slowest);
        }
        best
    };
    timed(&input);
    let solo = pair_time(1).as_nanos() as f64;
    let mut units = 1.0f64;
    let mut n = 2;
    while n <= cpus {
        let aggregate = n as f64 * solo / pair_time(n).as_nanos() as f64;
        units = units.max(aggregate);
        // Past the knee the aggregate stays flat; two more steps confirm it.
        if aggregate < units * 0.9 {
            break;
        }
        n += 1 + n / 4;
    }
    (units.round() as usize).clamp(1, cpus)
}

/// Apple: `hw.nperflevels` performance levels, each with
/// `hw.perflevelN.physicalcpu` cores and `hw.perflevelN.cpusperl2` cores
/// per cluster (the cores that share an L2 share an SME unit). Units are
/// clusters: the sum over levels of physicalcpu / cpusperl2.
#[cfg(all(blake3_sme2, target_vendor = "apple"))]
fn platform_cluster_count() -> Option<usize> {
    fn sysctl_u32(name: &std::ffi::CStr) -> Option<u32> {
        let mut value: u32 = 0;
        let mut size = core::mem::size_of::<u32>();
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut value as *mut u32 as *mut libc::c_void,
                &mut size,
                core::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(value)
    }
    let levels = sysctl_u32(c"hw.nperflevels")?;
    let mut clusters = 0usize;
    for level in 0..levels {
        let cores = std::ffi::CString::new(format!("hw.perflevel{level}.physicalcpu")).unwrap();
        let per_cluster = std::ffi::CString::new(format!("hw.perflevel{level}.cpusperl2")).unwrap();
        let cores = sysctl_u32(&cores)? as usize;
        let per_cluster = sysctl_u32(&per_cluster)?.max(1) as usize;
        clusters += cores.div_ceil(per_cluster);
    }
    (clusters > 0).then_some(clusters)
}

/// Elsewhere no interface reports which cores share an SME unit (Linux's
/// `cluster_cpus_list` describes the guest's view, which a hypervisor
/// invents), so the count is measured.
#[cfg(all(blake3_sme2, not(target_vendor = "apple")))]
fn platform_cluster_count() -> Option<usize> {
    None
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
            assert_eq!(want, hash(&input[..len], usize::MAX), "len = {len}");
            let key = [42u8; KEY_LEN];
            assert_eq!(
                crate::keyed_hash(&key, &input[..len]),
                hash_with_mode(&input[..len], Mode::KeyedHash(&key), usize::MAX),
                "keyed len = {len}"
            );
            let context_key = hazmat::hash_derive_key_context("lanes test");
            assert_eq!(
                *Hasher::new_from_context_key(&context_key).update(&input[..len]).finalize().as_bytes(),
                *hash_with_mode(&input[..len], Mode::DeriveKeyMaterial(&context_key), usize::MAX).as_bytes(),
                "derive len = {len}"
            );
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
                for platform in [fast_platform(), other_platform()] {
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
                    assert_eq!(want, merge_root(&pieces, &mut cvs, Mode::Hash), "len = {len}, threads = {threads}");
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
                        assert_eq!(want, merge_root(&pieces, &mut cvs.clone(), mode), "len={len}, threads={threads}");
                    }
                }
            }
        }
    }

    /// All contenders reserve before any release. Exactly cap - 1
    /// workers fit beside the caller, even when they race for the last place.
    #[test]
    fn test_thread_reservations_obey_cap() {
        for cap in [1, 2, 3, 8, 17] {
            let active = AtomicUsize::new(1);
            let barrier = std::sync::Barrier::new(17);
            std::thread::scope(|scope| {
                for _ in 0..16 {
                    let (active, barrier) = (&active, &barrier);
                    scope.spawn(move || {
                        for _ in 0..50 {
                            barrier.wait();
                            let reserved = reserve_thread(active, cap);
                            barrier.wait();
                            barrier.wait();
                            if reserved {
                                active.fetch_sub(1, Ordering::SeqCst);
                            }
                            barrier.wait();
                        }
                    });
                }
                for _ in 0..50 {
                    barrier.wait();
                    barrier.wait();
                    assert_eq!(active.load(Ordering::SeqCst), cap);
                    barrier.wait();
                    barrier.wait();
                    assert_eq!(active.load(Ordering::SeqCst), 1);
                }
            });
        }
    }

    /// Every thread cap gives hash()'s result, and a cap of one takes the
    /// serial path without touching the pool.
    #[test]
    fn test_budget_caps_agree() {
        let mut input = vec![0u8; 4 * MIN_SPLIT_LEN + 77];
        crate::test::paint_test_input(&mut input);
        let want = crate::hash(&input);
        assert_eq!(want, hash(&input, 1));
        for cap in [2, 3, 4, 64, usize::MAX] {
            assert_eq!(want, hash(&input, cap), "cap = {cap}");
        }
    }

    /// Many concurrent callers on one process: every result is right, and
    /// each call waits for its workers and leaves every result complete.
    #[test]
    fn test_concurrent_callers_agree() {
        let mut input = vec![0u8; 8 * MIN_SPLIT_LEN + 1];
        crate::test::paint_test_input(&mut input);
        let want = crate::hash(&input);
        let input = &input[..];
        let barrier = std::sync::Barrier::new(32);
        std::thread::scope(|scope| {
            for i in 0..32 {
                let cap = [usize::MAX, 3, 2, 5][i % 4];
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..20 {
                        assert_eq!(want, hash(input, cap));
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
        let block = [0u8; crate::BLOCK_LEN];
        let big = [0u8; 3 * CHUNK_LEN + 5];
        let small = [0u8; 3];
        let mut inputs: Vec<&[u8]> = vec![&block[..]; 40_000];
        inputs[7] = &big[..];
        inputs[39_999] = &small[..];
        let total: usize = inputs.iter().map(|m| m.len()).sum();
        let pieces = cut_messages(&inputs, total, 16);
        assert!(pieces.len() >= 16, "{} ranges", pieces.len());
        assert_eq!(pieces[0].offset, 0);
        assert!(pieces.windows(2).all(|w| w[0].offset + w[0].len == w[1].offset));
        assert_eq!(pieces.last().unwrap().offset + pieces.last().unwrap().len, inputs.len());
        let bytes = |p: &Piece| inputs[p.offset..][..p.len].iter().map(|m| m.len()).sum::<usize>();
        assert!(pieces[..pieces.len() - 1].iter().all(|p| bytes(p) <= MAX_PIECE_LEN && bytes(p) >= MIN_PIECE_LEN / 2));
        assert!(bytes(&pieces[0]) >= bytes(&pieces[pieces.len() - 2]));
        assert_eq!(cut_messages(&inputs[200..300], 100 * 64, 16).len(), 1);
    }

    /// Every budget gives hash_many()'s digests, on contiguous and on
    /// scattered messages, with lengths other than one block mixed in.
    #[test]
    fn test_hash_many_budgets_agree() {
        let mut buffer = vec![0u8; 4 * MIN_SPLIT_LEN + 3 * CHUNK_LEN];
        crate::test::paint_test_input(&mut buffer);
        let mut inputs: Vec<&[u8]> = buffer[..4 * MIN_SPLIT_LEN].chunks_exact(crate::BLOCK_LEN).collect();
        inputs[100] = &buffer[4 * MIN_SPLIT_LEN..][..3 * CHUNK_LEN];
        inputs[101] = &buffer[..0];
        inputs[102] = &buffer[..65];
        let mut want = vec![Hash::from_bytes([0; 32]); inputs.len()];
        crate::hash_many(&inputs, &mut want);
        for (i, message) in inputs.iter().enumerate().take(200) {
            assert_eq!(want[i], crate::hash(message), "message {i}");
        }
        for cap in [1, 2, 3, 4, 64, usize::MAX] {
            let mut got = vec![Hash::from_bytes([0; 32]); inputs.len()];
            hash_many(&inputs, &mut got, cap);
            assert_eq!(want, got, "cap = {cap}");
        }
    }

    /// Concurrent batch callers beside tree callers: every digest right.
    #[test]
    fn test_concurrent_batch_callers_agree() {
        let mut buffer = vec![0u8; 8 * MIN_SPLIT_LEN];
        crate::test::paint_test_input(&mut buffer);
        let inputs: Vec<&[u8]> = buffer.chunks_exact(crate::BLOCK_LEN).collect();
        let mut want = vec![Hash::from_bytes([0; 32]); inputs.len()];
        crate::hash_many(&inputs, &mut want);
        let tree_want = crate::hash(&buffer);
        let (inputs, want, buffer) = (&inputs[..], &want[..], &buffer[..]);
        let barrier = std::sync::Barrier::new(16);
        std::thread::scope(|scope| {
            for i in 0..16 {
                let cap = [usize::MAX, 3, 2, 5][i % 4];
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let mut got = vec![Hash::from_bytes([0; 32]); inputs.len()];
                    for _ in 0..10 {
                        if i % 2 == 0 {
                            hash_many(inputs, &mut got, cap);
                            assert_eq!(want, &got[..]);
                        } else {
                            assert_eq!(tree_want, hash(buffer, cap));
                        }
                    }
                });
            }
        });
    }
}
