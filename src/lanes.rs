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
//! the finish waits on little. Every thread hashes a piece with
//! [`hazmat::HasherExt::set_input_offset`] and
//! [`finalize_non_root`](hazmat::HasherExt::finalize_non_root), one
//! chaining value per piece, and the calling thread merges the values back
//! up the tree with [`hazmat::merge_subtrees_non_root`] and
//! [`hazmat::merge_subtrees_root`]. Inputs under [`MIN_SPLIT_LEN`] stay on
//! the calling thread.
//!
//! # Jobs and the pool
//!
//! A call registers a *job* (its pieces, a cursor, a done count) in the
//! pool's list, then takes pieces from its own job through the cursor until
//! none remain and waits for the done count. Workers, started once per
//! process, one per CPU beyond the first, serve the jobs in the list
//! round-robin, one piece at a time through the same cursor. Two callers
//! hashing at once therefore each get about half the workers' time, and
//! a slow thread takes fewer pieces than a fast one; there is nothing to
//! tune for fairness or balance.
//!
//! Threads hashing at once (callers inside a call plus workers holding a
//! piece) stay within the CPU count: a worker takes a piece only while
//! that holds. Two callers on a two-CPU machine run one piece at a time
//! each, on their own threads, which is the right answer there. A caller's
//! thread cap ([`crate::hash_multithreaded_with_budget`]) bounds the
//! threads holding a piece of its job at once, the caller included.
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

use crate::hazmat::{self, ChainingValue, HasherExt, Mode};
use crate::platform::Platform;
use crate::{CHUNK_LEN, Hash, Hasher, KEY_LEN};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Inputs below this length are hashed on the calling thread. A split
/// hands pieces to workers that are polling, a few microseconds each way;
/// 64 KiB takes about 14 µs on one SME2 thread, and as four pieces it
/// comes back sooner.
pub(crate) const MIN_SPLIT_LEN: usize = 64 * 1024;

/// The shortest piece: eight chunks, a hybrid kernel's worth.
const MIN_PIECE_LEN: usize = 8 * CHUNK_LEN;

/// The longest piece: one SME2 kernel call of the platform's degree.
const MAX_PIECE_LEN: usize = 128 * CHUNK_LEN;

/// How long a worker or a waiting caller spins before sleeping. Long
/// enough to bridge the gap between back-to-back hashes in a busy caller;
/// short enough that an idle process is quiet within it.
pub(crate) const SPIN_BEFORE_SLEEP: std::time::Duration = std::time::Duration::from_micros(200);

/// The length of the next piece when `remaining` bytes are uncut and
/// `threads` threads may share them: about a thread's share of what is
/// left, rounded down to a power of two chunks within the bounds above.
/// Pieces therefore shrink toward the end of the input, and the slowest
/// thread's last piece is a small one.
fn next_piece_len(remaining: usize, threads: usize) -> usize {
    let want = (remaining / threads.max(1)).clamp(MIN_PIECE_LEN, MAX_PIECE_LEN);
    1 << (usize::BITS - 1 - want.leading_zeros())
}

/// A hashing mode with its key material owned, so it can cross to a
/// worker without borrowing the caller.
#[derive(Clone, Copy)]
enum OwnedMode {
    Hash,
    KeyedHash([u8; KEY_LEN]),
    DeriveKeyMaterial(hazmat::ContextKey),
}

impl OwnedMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Hash => Self::Hash,
            Mode::KeyedHash(key) => Self::KeyedHash(*key),
            Mode::DeriveKeyMaterial(context_key) => Self::DeriveKeyMaterial(*context_key),
        }
    }

    fn hasher(&self) -> Hasher {
        match self {
            Self::Hash => Hasher::new(),
            Self::KeyedHash(key) => Hasher::new_keyed(key),
            Self::DeriveKeyMaterial(context_key) => Hasher::new_from_context_key(context_key),
        }
    }

    /// The key words and flags this mode hashes with.
    fn key_and_flags(&self) -> (crate::CVWords, u8) {
        match self {
            Self::Hash => (*crate::IV, 0),
            Self::KeyedHash(key) => (crate::platform::words_from_le_bytes_32(key), crate::KEYED_HASH),
            Self::DeriveKeyMaterial(context_key) => {
                (crate::platform::words_from_le_bytes_32(context_key), crate::DERIVE_KEY_MATERIAL)
            }
        }
    }
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
        let (key, flags) = OwnedMode::from(mode).key_and_flags();
        return crate::hash_serial(input, &key, flags);
    }
    hash_over_pool(input, mode, max_threads)
}

#[inline(never)]
fn hash_over_pool(input: &[u8], mode: Mode, max_threads: usize) -> Hash {
    let pool = pool();
    let pieces = cut_subtrees(input.len(), pool.cpus);
    let mut cvs = vec![ChainingValue::default(); pieces.len()];
    let job = Job {
        input,
        pieces: &pieces,
        cvs: cvs.as_mut_ptr(),
        mode: OwnedMode::from(mode),
        cursor: Line(AtomicUsize::new(0)),
        done: Line(AtomicUsize::new(0)),
        active: Line(AtomicUsize::new(1)),
        max_threads,
    };
    pool.callers.fetch_add(1, Ordering::SeqCst);
    let slot = pool.register(&job);
    loop {
        let index = job.cursor.fetch_add(1, Ordering::SeqCst);
        if index >= pieces.len() {
            break;
        }
        // Sound: this thread took index through the cursor, so it alone
        // writes cvs[index].
        unsafe { job.hash_piece(index) };
        job.done.fetch_add(1, Ordering::SeqCst);
    }
    job.active.fetch_sub(1, Ordering::SeqCst);
    // Every piece is taken; the slot has nothing more to give from this
    // job. Workers that hold a piece finish it and count it in `done`.
    if let Some(slot) = slot {
        pool.unregister(slot);
    }
    pool.wait_done(&job.done, pieces.len());
    pool.callers.fetch_sub(1, Ordering::SeqCst);
    merge_root(&pieces, &cvs, mode)
}

/// One call's work, on the caller's stack. Workers reach it through the
/// pool's list while it is registered, and finish the pieces they hold
/// after; the caller returns only when `done` has counted every piece, so
/// every worker access falls inside the caller's frame.
struct Job<'a> {
    input: &'a [u8],
    /// In offset order.
    pieces: &'a [Piece],
    /// One slot per piece; slot `i` is written by the thread that took
    /// piece `i` through `cursor`, and by nobody else.
    cvs: *mut ChainingValue,
    mode: OwnedMode,
    /// The next piece to take.
    cursor: Line,
    /// Pieces finished.
    done: Line,
    /// Threads holding a piece of this job, the caller included while it
    /// still takes pieces.
    active: Line,
    /// Bound on `active`.
    max_threads: usize,
}

/// An atomic counter on its own cache line: `cursor`, `done`, and `active`
/// are each hit by every thread of a job, and sharing a line would make
/// each update evict the others.
#[repr(align(128))]
struct Line(AtomicUsize);

impl std::ops::Deref for Line {
    type Target = AtomicUsize;
    fn deref(&self) -> &AtomicUsize {
        &self.0
    }
}

impl Job<'_> {
    /// Hash piece `index` into its slot. The caller has taken `index`
    /// through `cursor`.
    unsafe fn hash_piece(&self, index: usize) {
        let piece = self.pieces[index];
        let bytes = &self.input[piece.offset..][..piece.len];
        let permit = pool().take_sme_permit();
        let mut hasher = self.mode.hasher();
        hasher.set_platform(if permit { fast_platform() } else { other_platform() });
        hasher.set_input_offset(piece.offset as u64);
        hasher.update(bytes);
        let cv = hasher.finalize_non_root();
        if permit {
            pool().release_sme_permit();
        }
        unsafe { *self.cvs.add(index) = cv };
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

/// One piece of the input: a whole subtree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    offset: usize,
    len: usize,
}

/// Cut `len` bytes into subtrees for `threads` threads, in offset order:
/// each piece is [`next_piece_len`] of what remains, and the last is
/// whatever is left. A piece's length never exceeds the one before it and
/// every piece starts at a multiple of its own length, so each is a whole
/// subtree at its offset. `len` is at least two chunks... it exceeds
/// [`MIN_PIECE_LEN`], so two or more pieces come back.
fn cut_subtrees(len: usize, threads: usize) -> Vec<Piece> {
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

/// Merge the pieces' chaining values up the tree and finish with the root
/// compression. Needs two or more pieces, which is what a cut gives.
fn merge_root(pieces: &[Piece], cvs: &[ChainingValue], mode: Mode) -> Hash {
    assert_eq!(pieces.len(), cvs.len());
    assert!(pieces.len() >= 2, "one piece has no parent to merge; finalize it as the root instead");
    let len: usize = pieces.iter().map(|p| p.len).sum();
    let boundary = hazmat::left_subtree_len(len as u64) as usize;
    let split = pieces.partition_point(|p| p.offset < boundary);
    assert!(pieces[split].offset == boundary, "pieces do not tile the tree");
    let left = subtree_cv(&pieces[..split], &cvs[..split], mode);
    let right = subtree_cv(&pieces[split..], &cvs[split..], mode);
    hazmat::merge_subtrees_root(&left, &right, mode)
}

/*
 * The pool: cpus - 1 worker threads serving the registered jobs.
 *
 * The slots hold raw pointers to jobs on callers' stacks. A worker counts
 * itself as a reader of a slot before it loads the pointer and until it has
 * moved the job's cursor; a caller clears its slot once every piece is
 * taken and then waits until the slot has no reader, so no worker is left
 * mid-take on a job that is gone. A worker that took a piece touches the
 * job until its increment of `done`, which is the last thing it does with
 * the job, and the caller returns only after `done` has counted every
 * piece; so every worker access falls inside the caller's frame.
 *
 * Every step on the way to a piece is an atomic operation; the two locks
 * below serve sleeping and waking alone, and a thread takes one only when
 * it is about to sleep or knows a sleeper is there.
 */
struct Pool {
    slots: [Slot; MAX_JOBS],
    /// Callers inside hash_over_pool.
    callers: AtomicUsize,
    /// Workers holding a piece.
    busy: AtomicUsize,
    /// SME2 permits free.
    sme_free: AtomicUsize,
    /// SME2 permits in all, once the starter has counted them.
    sme_total: OnceLock<usize>,
    cpus: usize,
    /// When the last job was registered, in nanoseconds of `epoch`; a
    /// worker sleeps only when this is SPIN_BEFORE_SLEEP old.
    last_job_ns: AtomicU64,
    epoch: std::time::Instant,
    /// Workers asleep on `posted`.
    sleepers: AtomicUsize,
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

// The pointers are borrows of callers that outlive every use (see above).
unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

/// The pool, created on first use. Creation itself is cheap: a starter
/// thread spawns the workers and, where the SME unit count is measured,
/// runs that measurement; the first calls proceed on the calling thread
/// and whichever workers have arrived, and the permits grow to the
/// measured count when the starter is done. The first call therefore
/// costs about what a call costs, and a caller timing it (a benchmark's
/// calibration, say) sees nothing of the start.
fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let pool = Pool {
            slots: std::array::from_fn(|_| Slot { job: AtomicPtr::new(std::ptr::null_mut()), readers: AtomicUsize::new(0) }),
            callers: AtomicUsize::new(0),
            busy: AtomicUsize::new(0),
            sme_free: AtomicUsize::new(if uses_sme_permits() { 1 } else { 0 }),
            sme_total: OnceLock::new(),
            cpus,
            last_job_ns: AtomicU64::new(0),
            epoch: std::time::Instant::now(),
            sleepers: AtomicUsize::new(0),
            sleep_lock: Mutex::new(()),
            posted: Condvar::new(),
            waiters: AtomicUsize::new(0),
            finished: Mutex::new(()),
            finished_signal: Condvar::new(),
        };
        std::thread::Builder::new()
            .name("blake3-starter".into())
            .spawn(move || {
                for worker in 1..cpus {
                    std::thread::Builder::new()
                        .name(format!("blake3-worker-{worker}"))
                        .spawn(worker_main)
                        .expect("spawning a BLAKE3 worker");
                }
                let pool = POOL.wait();
                let permits = sme_permits();
                if permits > 1 {
                    pool.sme_free.fetch_add(permits - 1, Ordering::SeqCst);
                }
                pool.sme_total.set(permits).expect("the starter sets the permit count once");
            })
            .expect("spawning the BLAKE3 starter");
        pool
    })
}

impl Pool {
    /// Put the job in a free slot. Returns the slot, or None when every
    /// slot is taken. When the workers awake are fewer than the pieces,
    /// every sleeper is woken, in one system call: a wake costs the waker
    /// ten microseconds or more on some machines and the sleeper arrives
    /// tens of microseconds later, so one call pays once for all, and
    /// the woken workers then stay awake while calls keep coming.
    fn register(&self, job: &Job) -> Option<usize> {
        // The 'static is a promise the caller keeps by waiting on `done`.
        let ptr = job as *const Job as *mut Job<'static>;
        let slot = (0..MAX_JOBS).find(|&i| {
            self.slots[i].job.compare_exchange(std::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::Relaxed).is_ok()
        })?;
        self.last_job_ns.store(self.epoch.elapsed().as_nanos() as u64, Ordering::SeqCst);
        let sleepers = self.sleepers.load(Ordering::SeqCst);
        if sleepers > 0 && (self.cpus - 1 - sleepers.min(self.cpus - 1)) < job.pieces.len() {
            // The lock orders the notify after a sleeper's last check.
            let _guard = self.sleep_lock.lock().unwrap();
            self.posted.notify_all();
        }
        Some(slot)
    }

    /// Whether a job was registered within SPIN_BEFORE_SLEEP: while calls
    /// keep coming, workers stay awake.
    fn jobs_recently(&self) -> bool {
        let last = self.last_job_ns.load(Ordering::SeqCst);
        self.epoch.elapsed().as_nanos() as u64 - last < SPIN_BEFORE_SLEEP.as_nanos() as u64
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

    /// Wait until `done` reaches `count`. The caller's own pieces have just
    /// finished, so the workers are usually done or nearly so: spin for a
    /// while, then sleep on the condition variable the workers signal
    /// after a piece while a caller sleeps.
    fn wait_done(&self, done: &AtomicUsize, count: usize) {
        let started = std::time::Instant::now();
        while started.elapsed() < SPIN_BEFORE_SLEEP {
            if done.load(Ordering::SeqCst) == count {
                return;
            }
            std::thread::yield_now();
        }
        let mut guard = self.finished.lock().unwrap();
        self.waiters.fetch_add(1, Ordering::SeqCst);
        while done.load(Ordering::SeqCst) != count {
            guard = self.finished_signal.wait(guard).unwrap();
        }
        self.waiters.fetch_sub(1, Ordering::SeqCst);
    }

    /// A piece is finished: count it, and wake a sleeping caller. The
    /// increment is the worker's last touch of the job.
    fn piece_done(&self, done: &AtomicUsize) {
        done.fetch_add(1, Ordering::SeqCst);
        if self.waiters.load(Ordering::SeqCst) > 0 {
            let _guard = self.finished.lock().unwrap();
            self.finished_signal.notify_all();
        }
    }

    /// A worker takes one piece from the first job, scanning the slots
    /// from `start`, that has one and room under its thread cap, while
    /// hashing threads stay within the CPUs. The pointer stays valid until
    /// this worker's `piece_done` (see the pool's comment).
    fn take_piece(&self, start: &mut usize) -> Option<(*const Job<'static>, usize)> {
        if self.busy.load(Ordering::SeqCst) + self.callers.load(Ordering::SeqCst) >= self.cpus {
            return None;
        }
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
            let mut over_cap = false;
            if !ptr.is_null() {
                // Sound: a reader of the slot; the caller waits for readers.
                let job = unsafe { &*ptr };
                if job.cursor.load(Ordering::SeqCst) < job.pieces.len()
                    && job.active.load(Ordering::SeqCst) < job.max_threads
                {
                    // Reserve a hashing thread, then the piece; both exact.
                    if self.busy.fetch_add(1, Ordering::SeqCst) + self.callers.load(Ordering::SeqCst) >= self.cpus {
                        over_cap = true;
                    } else {
                        let index = job.cursor.fetch_add(1, Ordering::SeqCst);
                        if index < job.pieces.len() {
                            job.active.fetch_add(1, Ordering::SeqCst);
                            taken = Some(index);
                        }
                    }
                    if taken.is_none() {
                        self.busy.fetch_sub(1, Ordering::SeqCst);
                    }
                }
            }
            slot.readers.fetch_sub(1, Ordering::SeqCst);
            if let Some(index) = taken {
                *start = at + 1;
                return Some((ptr, index));
            }
            if over_cap {
                return None;
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
            // The guard is held through the take, so a caller's wake
            // cannot fall between the take and the wait.
            let guard = self.sleep_lock.lock().unwrap();
            self.sleepers.fetch_add(1, Ordering::SeqCst);
            let taken = self.take_piece(start);
            if taken.is_none() {
                drop(self.posted.wait(guard).unwrap());
            }
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
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
        // Sound by the pool's contract: the caller is waiting on `done`.
        let job = unsafe { &*job_ptr };
        unsafe { job.hash_piece(index) };
        job.active.fetch_sub(1, Ordering::SeqCst);
        pool.busy.fetch_sub(1, Ordering::SeqCst);
        pool.piece_done(&job.done);
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
/// hashes of 256 KiB (about two milliseconds); the best of `ROUNDS` is
/// kept, which rides out a hypervisor descheduling a virtual CPU.
#[cfg(blake3_sme2)]
fn measure_sme_units() -> usize {
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};
    const LEN: usize = 256 * CHUNK_LEN;
    const REPEAT: usize = 40;
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
            for threads in [1, 2, 3, 16, 64] {
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
        for threads in [1, 2, 3, 16, 64] {
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
            for threads in [1, 2, 3, 5, 16, 64] {
                let pieces = cut_subtrees(len, threads);
                for platform in [fast_platform(), other_platform()] {
                    let cvs: Vec<ChainingValue> = pieces
                        .iter()
                        .map(|p| {
                            let mut hasher = Hasher::new();
                            hasher.set_platform(platform);
                            hasher.set_input_offset(p.offset as u64);
                            hasher.update(&input[p.offset..][..p.len]);
                            hasher.finalize_non_root()
                        })
                        .collect();
                    assert_eq!(want, merge_root(&pieces, &cvs, Mode::Hash), "len = {len}, threads = {threads}");
                }
            }
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
    /// afterwards the pool is quiet: no caller, no busy worker, no job,
    /// every permit back.
    #[test]
    fn test_concurrent_callers_agree() {
        let mut input = vec![0u8; 8 * MIN_SPLIT_LEN + 1];
        crate::test::paint_test_input(&mut input);
        let want = crate::hash(&input);
        let input = &input[..];
        std::thread::scope(|scope| {
            for cap in [usize::MAX, usize::MAX, 3, 2, usize::MAX, 5, usize::MAX, usize::MAX] {
                scope.spawn(move || {
                    for _ in 0..20 {
                        assert_eq!(want, hash(input, cap));
                    }
                });
            }
        });
        let pool = pool();
        // Other tests in this process may be mid-call; when they are quiet
        // too the counts read zero and the permits are all back.
        let total = *pool.sme_total.wait();
        if pool.callers.load(Ordering::SeqCst) == 0 {
            assert!(pool.slots.iter().all(|s| s.job.load(Ordering::SeqCst).is_null()));
            assert_eq!(pool.busy.load(Ordering::SeqCst), 0);
            assert_eq!(pool.sme_free.load(Ordering::SeqCst), total);
        }
    }
}
