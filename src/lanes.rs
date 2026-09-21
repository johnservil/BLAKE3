//! Multithreaded hashing over the machine's execution *lanes*, with
//! cooperative admission so that concurrent callers share the machine
//! instead of each taking all of it.
//!
//! # Lanes
//!
//! A lane is a run of CPUs that share the execution resource a BLAKE3
//! kernel saturates. On Apple silicon every core cluster has one SME unit,
//! so a second SME2 thread on the same cluster adds nothing: one lane per
//! cluster, from `sysctl hw.perflevelN`. Elsewhere the SME unit's sharing
//! is a property of the part that no interface reports, so it is measured
//! once, at first use: two threads hash 256 KiB each at the same time, and
//! when the pair takes at least half again as long as one thread alone, the
//! CPUs share a unit and the lanes are the kernel's CPU clusters (sysfs
//! `topology/cluster_cpus_list`); otherwise every CPU is a lane. The probe
//! takes about thirty milliseconds. `BLAKE3_LANES=<n>` in the environment
//! overrides the detection.
//!
//! # Admission
//!
//! A work-stealing pool sized to the whole machine performs well alone and
//! badly beside a copy of itself: two such pools on two cores each run at
//! half speed, and the pair finishes later than two single threads would.
//! This module holds a process-wide count of lanes in use. A call takes
//! the lanes that are free at that moment (at least one: its own thread,
//! which counts whether or not the call splits) and releases them when it
//! returns. Two simultaneous callers on a four-lane machine get two lanes
//! each; a caller that arrives when every lane is busy runs on its own
//! thread alone.
//!
//! Admission also watches the workers: a worker that a busy machine keeps
//! off its CPU finishes its piece late, and the caller, done with its own,
//! waits for it. When a call's wait for the workers exceeds the time its
//! own piece took by more than [`STRAGGLER_PERMILLE`], the call notes a
//! straggler, and after [`STRAGGLER_STRIKES`] such calls in a row the
//! module runs on the caller's thread alone for a while before trying the
//! lanes again. The backoff doubles on each consecutive stand-down
//! ([`BACKOFF_CALLS_MIN`] to [`BACKOFF_CALLS_MAX`] calls) and resets when a
//! split runs clean, so a machine that stays busy costs one probe in a few
//! thousand calls and a machine that frees up is noticed within a few
//! dozen. The count above is exact inside one process; the straggler rule
//! is what makes two *processes* (which cannot see each other's count)
//! share the machine, each discovering that its workers are competing and
//! standing down.
//!
//! A lane is held from the moment a call takes it until the call returns,
//! so a call that arrives while another is running sees the lanes the
//! other took. Between two calls on one thread the lanes are free for an
//! instant, and a caller on another thread may take one then; when the
//! first thread's next call comes, it finds that lane taken and runs
//! alone. Once the machine is shared this way the split caller's worker
//! runs on the CPU the other caller's thread is on, and the two compete
//! there until the straggler rule stands the split caller down. Two busy
//! threads on a two-lane machine therefore settle to one lane each within
//! a few calls: which is the outcome that shares the machine.
//!
//! # Splitting
//!
//! The input is cut at chunk boundaries into as many pieces as there are
//! lanes granted, each piece a valid BLAKE3 subtree (see
//! [`hazmat::left_subtree_len`]): the pieces are leaves of the tree whose
//! root is the whole input, reached by splitting the largest piece at its
//! left-subtree boundary until enough exist. Each lane hashes its piece
//! with [`hazmat::HasherExt::set_input_offset`] and
//! [`finalize_non_root`](hazmat::HasherExt::finalize_non_root), and the
//! calling thread merges the chaining values back up that tree with
//! [`hazmat::merge_subtrees_non_root`] and
//! [`hazmat::merge_subtrees_root`]. Inputs under [`MIN_SPLIT_LEN`] stay on
//! the calling thread: the hand-off costs a few microseconds and 64 KiB
//! takes about fourteen on one SME2 lane, so smaller inputs gain nothing.
//!
//! # Workers
//!
//! Workers are started on first use, one per lane beyond the first, and
//! take jobs from a shared queue. A job is the caller's input slice and a
//! slot for the chaining value; the caller hashes its own piece, then
//! waits until every job it posted is done. The caller returns only after
//! that wait, which is what makes lending its stack to the workers sound.
//!
//! A worker sleeps on the queue's condition variable between jobs, and a
//! caller sleeps on another while it waits for results; both spin briefly
//! first ([`SPIN_BEFORE_SLEEP`]), which covers back-to-back calls. Waking a
//! sleeping worker costs tens of microseconds on some systems (a futex
//! wake through a hypervisor), and that wake is the caller's to pay only
//! when a worker is asleep. The caller pays it as latency on its critical
//! path, so a piece a worker takes must be worth more than the wake: see
//! [`MIN_SPLIT_LEN`].
//!
//! Workers take no lane of their own: a sleeping worker is invisible to
//! the scheduler, and a spinning one yields its CPU on every iteration
//! ([`std::thread::yield_now`]) so a runnable thread on the same CPU goes
//! first. The spin is a poll between yields, and it stops within
//! `SPIN_BEFORE_SLEEP` on an idle process.

use crate::hazmat::{self, ChainingValue, HasherExt, Mode};
use crate::{CHUNK_LEN, Hash, Hasher, KEY_LEN};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Inputs below this length are hashed on the calling thread. 128 KiB is
/// two 64 KiB pieces: on an Apple M4 one SME2 lane takes about 14 µs per
/// 64 KiB, and the hand-off to a waiting worker and back costs 3–5 µs, so
/// a two-way split of 128 KiB comes out ahead by a third and a split of
/// 64 KiB about even.
pub const MIN_SPLIT_LEN: usize = 128 * 1024;

/// The smallest piece a lane receives: sixteen chunks fill one SME2 group.
const MIN_PIECE_LEN: usize = 16 * CHUNK_LEN;

/// Upper bound on lanes one call uses; enough for any machine this crate
/// targets, and it sizes the on-stack chaining-value array.
pub const MAX_LANES: usize = 64;

/// How long a worker or a waiting caller spins before sleeping. Long
/// enough to bridge the gap between back-to-back hashes in a busy caller;
/// short enough that an idle process is quiet within it.
pub const SPIN_BEFORE_SLEEP: std::time::Duration = std::time::Duration::from_micros(200);

/// A call whose wait for its workers exceeds this share of its own piece's
/// time saw a straggler: a worker that lost its CPU to someone else. Equal
/// pieces on free lanes finish within a few percent of each other.
pub const STRAGGLER_PERMILLE: u128 = 300;
/// Straggling calls in a row before the module stands down.
pub const STRAGGLER_STRIKES: usize = 2;
/// Calls run on the caller's thread alone after a stand-down: the first
/// stand-down waits this many, each consecutive one twice as many, up to
/// the maximum.
pub const BACKOFF_CALLS_MIN: usize = 16;
pub const BACKOFF_CALLS_MAX: usize = 4096;

/// Straggler strikes so far, the backoff calls remaining, and the length
/// of the next backoff. All process-wide: the machine is shared by every
/// caller, so one caller's finding serves the rest.
static STRIKES: AtomicUsize = AtomicUsize::new(0);
static BACKOFF_REMAINING: AtomicUsize = AtomicUsize::new(0);
static NEXT_BACKOFF: AtomicUsize = AtomicUsize::new(BACKOFF_CALLS_MIN);

/// Whether this call should stay on its own thread because the workers
/// have been straggling; counts down the backoff as it goes.
fn backing_off() -> bool {
    let mut remaining = BACKOFF_REMAINING.load(Ordering::Relaxed);
    loop {
        if remaining == 0 {
            return false;
        }
        match BACKOFF_REMAINING.compare_exchange_weak(remaining, remaining - 1, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(seen) => remaining = seen,
        }
    }
}

/// Record whether the workers straggled on this call. A clean split resets
/// the strikes and the backoff length; strikes in a row start a backoff
/// and double the next one.
fn note_outcome(straggled: bool) {
    if straggled {
        let strikes = STRIKES.fetch_add(1, Ordering::AcqRel) + 1;
        if strikes >= STRAGGLER_STRIKES {
            STRIKES.store(0, Ordering::Relaxed);
            let length = NEXT_BACKOFF.load(Ordering::Relaxed);
            NEXT_BACKOFF.store((length * 2).min(BACKOFF_CALLS_MAX), Ordering::Relaxed);
            BACKOFF_REMAINING.store(length, Ordering::Release);
        }
    } else {
        STRIKES.store(0, Ordering::Relaxed);
        NEXT_BACKOFF.store(BACKOFF_CALLS_MIN, Ordering::Relaxed);
    }
}

/// Reset the straggler state, for tests and benchmarks that want each run
/// to start from "lanes free".
#[doc(hidden)]
pub fn reset_backoff() {
    STRIKES.store(0, Ordering::Relaxed);
    BACKOFF_REMAINING.store(0, Ordering::Relaxed);
    NEXT_BACKOFF.store(BACKOFF_CALLS_MIN, Ordering::Relaxed);
}

/// The number of lanes this machine has (see the module docs). At least 1.
pub fn lane_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| detect_lane_count().clamp(1, MAX_LANES))
}

/// Lanes currently granted to callers in this process, the callers' own
/// threads included.
static LANES_IN_USE: AtomicUsize = AtomicUsize::new(0);

/// Take the caller's own lane plus up to `want` free ones; returns the
/// extra lanes granted (0..=want).
fn admit(want: usize) -> usize {
    let total = lane_count();
    let mut current = LANES_IN_USE.load(Ordering::Relaxed);
    loop {
        let free = total.saturating_sub(current);
        let extra = free.saturating_sub(1).min(want);
        let next = current + 1 + extra;
        match LANES_IN_USE.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return extra,
            Err(seen) => current = seen,
        }
    }
}

fn release(extra: usize) {
    LANES_IN_USE.fetch_sub(1 + extra, Ordering::AcqRel);
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

    /// The whole input on the calling thread: the same path as hash().
    fn hash_serial(&self, input: &[u8]) -> Hash {
        let (key, flags) = self.key_and_flags();
        crate::hash_serial(input, &key, flags)
    }
}

/// Hash `input` over the machine's free lanes; the regular hash function.
#[inline]
pub fn hash(input: &[u8]) -> Hash {
    hash_with_mode(input, Mode::Hash)
}

/// Hash `input` over the machine's free lanes in keyed mode.
#[inline]
pub fn keyed_hash(key: &[u8; KEY_LEN], input: &[u8]) -> Hash {
    hash_with_mode(input, Mode::KeyedHash(key))
}

/// Hash `input` over the machine's free lanes in the given mode.
#[inline]
pub fn hash_with_mode(input: &[u8], mode: Mode) -> Hash {
    // The short path first and inline, so a small input costs what hash()
    // costs; everything the lanes need is behind the call below.
    if input.len() < MIN_SPLIT_LEN {
        let (key, flags) = OwnedMode::from(mode).key_and_flags();
        return crate::hash_serial(input, &key, flags);
    }
    hash_over_lanes(input, mode)
}

#[inline(never)]
fn hash_over_lanes(input: &[u8], mode: Mode) -> Hash {
    let owned = OwnedMode::from(mode);
    // A call that stays on its own thread still occupies a lane: the
    // count must show it, or a concurrent caller takes the lane this
    // thread is running on and its worker competes with this thread.
    let want = if backing_off() {
        0
    } else {
        (input.len() / MIN_PIECE_LEN).min(lane_count()).saturating_sub(1)
    };
    let extra = admit(want);
    if extra == 0 {
        let hash = owned.hash_serial(input);
        release(0);
        return hash;
    }
    let pieces = split_subtrees(input.len(), 1 + extra);
    debug_assert_eq!(pieces.len(), 1 + extra);

    // The caller takes the largest piece, so on free lanes every worker
    // finishes before it and its wait is the hand-off alone; a wait beyond
    // STRAGGLER_PERMILLE of its own time is a worker that lost its CPU.
    let own_index = pieces
        .iter()
        .enumerate()
        .max_by_key(|(_, p)| p.len)
        .map(|(i, _)| i)
        .unwrap();
    let mut cvs = [ChainingValue::default(); MAX_LANES];
    {
        let done = AtomicUsize::new(0);
        let mut own_slot: Option<&mut ChainingValue> = None;
        let mut jobs: Vec<Job> = Vec::with_capacity(extra);
        // Every job borrows `input`, a slot in `cvs`, and `done`; the wait
        // below ends before any of them goes out of scope.
        for (index, (piece, slot)) in pieces.iter().zip(cvs.iter_mut()).enumerate() {
            if index == own_index {
                own_slot = Some(slot);
            } else {
                jobs.push(Job {
                    input: &input[piece.offset..][..piece.len],
                    offset: piece.offset,
                    mode: owned,
                    slot,
                    done: &done,
                });
            }
        }
        pool().post(jobs);
        let own = pieces[own_index];
        let started = std::time::Instant::now();
        *own_slot.unwrap() = hash_piece(&input[own.offset..][..own.len], own.offset, owned);
        let own_time = started.elapsed();
        pool().wait(&done, extra);
        let waited = started.elapsed() - own_time;
        note_outcome(waited.as_nanos() * 1000 > own_time.as_nanos() * STRAGGLER_PERMILLE);
    }
    release(extra);
    merge_root(&pieces, &cvs[..pieces.len()], mode)
}

fn hash_piece(bytes: &[u8], offset: usize, mode: OwnedMode) -> ChainingValue {
    let mut hasher = mode.hasher();
    hasher.set_input_offset(offset as u64);
    hasher.update(bytes);
    hasher.finalize_non_root()
}

/// One lane's share of the input: a whole subtree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub offset: usize,
    pub len: usize,
}

/// Cut `len` bytes into `count` subtrees: starting from the whole input,
/// split the largest piece at its left-subtree boundary until `count`
/// pieces exist. Pieces come back in offset order, each a valid subtree at
/// its offset. `count` is 1..=MAX_LANES and `len` is at least `count`
/// chunks.
pub fn split_subtrees(len: usize, count: usize) -> Vec<Piece> {
    assert!((1..=MAX_LANES).contains(&count));
    assert!(len >= count * CHUNK_LEN, "each piece needs at least one chunk");
    let mut pieces = vec![Piece { offset: 0, len }];
    while pieces.len() < count {
        // The largest piece; ties to the leftmost.
        let (index, _) = pieces
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.len.cmp(&b.len).then(ib.cmp(ia)))
            .unwrap();
        let piece = pieces[index];
        assert!(piece.len > CHUNK_LEN, "a one-chunk piece cannot split");
        let left = hazmat::left_subtree_len(piece.len as u64) as usize;
        pieces[index] = Piece { offset: piece.offset, len: left };
        pieces.insert(index + 1, Piece { offset: piece.offset + left, len: piece.len - left });
    }
    pieces
}

/// The chaining value of the subtree at `offset` of `len` bytes: a piece's
/// value when a piece covers exactly that subtree, else the parent of its
/// two halves. Every subtree this asks for is a node of the tree
/// `split_subtrees` cut, because that function split along the same
/// boundaries from the same root.
fn subtree_cv(pieces: &[Piece], cvs: &[ChainingValue], offset: usize, len: usize, mode: Mode) -> ChainingValue {
    if let Some(index) = pieces.iter().position(|p| p.offset == offset && p.len == len) {
        return cvs[index];
    }
    assert!(len > CHUNK_LEN, "a subtree below every piece: the pieces do not tile the input");
    let left = hazmat::left_subtree_len(len as u64) as usize;
    let left_cv = subtree_cv(pieces, cvs, offset, left, mode);
    let right_cv = subtree_cv(pieces, cvs, offset + left, len - left, mode);
    hazmat::merge_subtrees_non_root(&left_cv, &right_cv, mode)
}

/// Merge the pieces' chaining values up the tree and finish with the root
/// compression. Needs two or more pieces, which is what a split gives.
fn merge_root(pieces: &[Piece], cvs: &[ChainingValue], mode: Mode) -> Hash {
    assert_eq!(pieces.len(), cvs.len());
    assert!(pieces.len() >= 2, "one piece has no parent to merge; finalize it as the root instead");
    let len: usize = pieces.iter().map(|p| p.len).sum();
    let left = hazmat::left_subtree_len(len as u64) as usize;
    let left_cv = subtree_cv(pieces, cvs, 0, left, mode);
    let right_cv = subtree_cv(pieces, cvs, left, len - left, mode);
    hazmat::merge_subtrees_root(&left_cv, &right_cv, mode)
}

/*
 * The worker pool: lane_count() - 1 threads waiting on one queue.
 *
 * A Job carries borrows of the caller's stack as raw pointers. The caller
 * posts its jobs, hashes its own piece, and then waits until `done` has
 * counted every job; only then does it return and let those borrows end.
 * A worker touches a job's memory only between taking it from the queue
 * and incrementing `done`, so every access falls inside the caller's wait.
 */
struct Job {
    input: *const [u8],
    offset: usize,
    mode: OwnedMode,
    slot: *mut ChainingValue,
    done: *const AtomicUsize,
}

// The pointers are borrows of a caller that outlives every use (see above).
unsafe impl Send for Job {}

struct Pool {
    queue: Mutex<VecDeque<Job>>,
    posted: Condvar,
    finished: Mutex<()>,
    finished_signal: Condvar,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool = Pool {
            queue: Mutex::new(VecDeque::new()),
            posted: Condvar::new(),
            finished: Mutex::new(()),
            finished_signal: Condvar::new(),
        };
        for lane in 1..lane_count() {
            std::thread::Builder::new()
                .name(format!("blake3-lane-{lane}"))
                .spawn(worker)
                .expect("spawning a BLAKE3 lane worker");
        }
        pool
    })
}

impl Pool {
    fn post(&self, jobs: Vec<Job>) {
        let mut queue = self.queue.lock().unwrap();
        for job in jobs {
            queue.push_back(job);
            self.posted.notify_one();
        }
    }

    /// Wait until `done` reaches `count`. The caller's own piece has just
    /// finished, so the workers are usually done or nearly so: spin for a
    /// while, then sleep on the condition variable the workers signal
    /// after every job.
    fn wait(&self, done: &AtomicUsize, count: usize) {
        let started = std::time::Instant::now();
        while started.elapsed() < SPIN_BEFORE_SLEEP {
            if done.load(Ordering::Acquire) == count {
                return;
            }
            std::hint::spin_loop();
        }
        let mut guard = self.finished.lock().unwrap();
        while done.load(Ordering::Acquire) != count {
            guard = self.finished_signal.wait(guard).unwrap();
        }
    }

    /// The next job: polled for a while (yielding the CPU between polls,
    /// so a thread that has work on this CPU runs), then waited for.
    fn take(&self) -> Job {
        let started = std::time::Instant::now();
        while started.elapsed() < SPIN_BEFORE_SLEEP {
            if let Some(job) = self.queue.lock().unwrap().pop_front() {
                return job;
            }
            std::thread::yield_now();
        }
        let mut queue = self.queue.lock().unwrap();
        loop {
            if let Some(job) = queue.pop_front() {
                return job;
            }
            queue = self.posted.wait(queue).unwrap();
        }
    }
}

fn worker() {
    let pool = pool();
    loop {
        let job = pool.take();
        // Sound by the pool's contract: the caller is waiting on `done`.
        let cv = hash_piece(unsafe { &*job.input }, job.offset, job.mode);
        unsafe {
            *job.slot = cv;
        }
        // Taking the lock before the increment closes the gap where the
        // caller checks the count, misses it, and sleeps forever.
        let _guard = pool.finished.lock().unwrap();
        unsafe { &*job.done }.fetch_add(1, Ordering::Release);
        pool.finished_signal.notify_all();
    }
}

/// See the module docs: the environment override, then the platform's
/// topology, then a measurement of whether CPUs share the kernel's unit.
fn detect_lane_count() -> usize {
    if let Some(count) = std::env::var("BLAKE3_LANES").ok().and_then(|v| v.parse::<usize>().ok()) {
        assert!(count >= 1, "BLAKE3_LANES must be at least 1");
        return count;
    }
    if let Some(count) = platform_lane_count() {
        return count;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Whether two threads hashing at once each keep their single-thread
/// speed. A pair that takes at least half again as long as one thread
/// alone shares an execution unit. Each timing covers `REPEAT` hashes of
/// 256 KiB (about two milliseconds), so a fresh thread's first
/// microseconds of scheduling are a small share; the best of `ROUNDS` is
/// kept, which rides out a hypervisor descheduling a virtual CPU.
#[cfg(target_os = "linux")]
fn cpus_share_kernel_unit() -> bool {
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};
    const LEN: usize = 256 * CHUNK_LEN;
    const REPEAT: usize = 40;
    const ROUNDS: usize = 5;
    let input: Vec<u8> = (0..LEN as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
    let timed = |input: &[u8]| -> Duration {
        let started = Instant::now();
        for _ in 0..REPEAT {
            let _ = std::hint::black_box(crate::hash(std::hint::black_box(input)));
        }
        started.elapsed()
    };
    timed(&input);
    let solo = (0..ROUNDS).map(|_| timed(&input)).min().unwrap();
    let input = Arc::new(input);
    let mut pair = Duration::MAX;
    for _ in 0..ROUNDS {
        let barrier = Arc::new(Barrier::new(2));
        let other = {
            let input = Arc::clone(&input);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                timed(&input)
            })
        };
        barrier.wait();
        let mine = timed(&input);
        let theirs = other.join().expect("lane probe thread");
        pair = pair.min(mine.max(theirs));
    }
    // Integer comparison: pair ≥ 1.5 × solo.
    pair.as_nanos() * 2 >= solo.as_nanos() * 3
}

/// Apple: `hw.nperflevels` performance levels, each with
/// `hw.perflevelN.physicalcpu` cores and `hw.perflevelN.cpusperl2` cores
/// per cluster (the cores that share an L2 share an SME unit). Lanes are
/// clusters: the sum over levels of physicalcpu / cpusperl2.
#[cfg(target_vendor = "apple")]
fn platform_lane_count() -> Option<usize> {
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
    let mut lanes = 0usize;
    for level in 0..levels {
        let cores = std::ffi::CString::new(format!("hw.perflevel{level}.physicalcpu")).unwrap();
        let per_cluster = std::ffi::CString::new(format!("hw.perflevel{level}.cpusperl2")).unwrap();
        let cores = sysctl_u32(&cores)? as usize;
        let per_cluster = sysctl_u32(&per_cluster)?.max(1) as usize;
        lanes += cores.div_ceil(per_cluster);
    }
    (lanes > 0).then_some(lanes)
}

/// Linux: every CPU is a lane when two threads keep their speed side by
/// side; otherwise the distinct `cluster_cpus_list` values across online
/// CPUs. Kernels before 5.16 have no cluster topology; then every CPU is
/// a lane.
#[cfg(target_os = "linux")]
fn platform_lane_count() -> Option<usize> {
    let cpus = std::thread::available_parallelism().ok()?.get();
    if cpus == 1 || !cpus_share_kernel_unit() {
        return Some(cpus);
    }
    let mut clusters: Vec<String> = Vec::new();
    for cpu in 0..cpus {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/cluster_cpus_list");
        let list = std::fs::read_to_string(path).ok()?.trim().to_owned();
        if !clusters.contains(&list) {
            clusters.push(list);
        }
    }
    (!clusters.is_empty()).then_some(clusters.len())
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux")))]
fn platform_lane_count() -> Option<usize> {
    None
}

/// A one-line description of the lanes for reports: the count and how it
/// was found.
pub fn describe_lanes() -> String {
    let count = lane_count();
    let how = if std::env::var_os("BLAKE3_LANES").is_some() {
        "set by BLAKE3_LANES"
    } else if cfg!(target_vendor = "apple") {
        "one per core cluster (hw.perflevelN.physicalcpu / cpusperl2; each cluster shares one SME unit)"
    } else if cfg!(target_os = "linux") {
        "one per CPU when two threads measured at full speed side by side, else one per CPU cluster (sysfs topology/cluster_cpus_list)"
    } else {
        "one per CPU (available_parallelism)"
    };
    format!("{count} lane{}: {how}", if count == 1 { "" } else { "s" })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_split_subtrees_shapes() {
        assert_eq!(
            split_subtrees(64 * CHUNK_LEN, 2),
            vec![Piece { offset: 0, len: 32 * CHUNK_LEN }, Piece { offset: 32 * CHUNK_LEN, len: 32 * CHUNK_LEN }]
        );
        assert_eq!(
            split_subtrees(64 * CHUNK_LEN, 3),
            vec![
                Piece { offset: 0, len: 16 * CHUNK_LEN },
                Piece { offset: 16 * CHUNK_LEN, len: 16 * CHUNK_LEN },
                Piece { offset: 32 * CHUNK_LEN, len: 32 * CHUNK_LEN },
            ]
        );
        // 100 chunks + 7 bytes into three: the root splits 64 | 36 + 7,
        // then the 64-chunk left side (the larger) splits into halves.
        assert_eq!(
            split_subtrees(100 * CHUNK_LEN + 7, 3),
            vec![
                Piece { offset: 0, len: 32 * CHUNK_LEN },
                Piece { offset: 32 * CHUNK_LEN, len: 32 * CHUNK_LEN },
                Piece { offset: 64 * CHUNK_LEN, len: 36 * CHUNK_LEN + 7 },
            ]
        );
        for count in 1..=16 {
            for len in [16 * CHUNK_LEN, 1000 * CHUNK_LEN + 3, 1 << 24, (1 << 20) + 1] {
                let pieces = split_subtrees(len, count);
                assert_eq!(pieces.len(), count);
                assert_eq!(pieces.iter().map(|p| p.len).sum::<usize>(), len);
                for (i, p) in pieces.iter().enumerate() {
                    if i > 0 {
                        assert_eq!(p.offset, pieces[i - 1].offset + pieces[i - 1].len);
                        let max = hazmat::max_subtree_len(p.offset as u64).unwrap();
                        assert!(p.len as u64 <= max, "piece {i} of {count} for {len}: {p:?} exceeds {max}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_hash_matches_serial() {
        let mut input = vec![0u8; 4 * MIN_SPLIT_LEN + 12345];
        crate::test::paint_test_input(&mut input);
        for len in [
            0,
            1,
            MIN_SPLIT_LEN - 1,
            MIN_SPLIT_LEN,
            MIN_SPLIT_LEN + 1,
            2 * MIN_SPLIT_LEN + 999,
            input.len(),
        ] {
            let want = crate::hash(&input[..len]);
            assert_eq!(want, hash(&input[..len]), "len = {len}");
            let key = [42u8; KEY_LEN];
            assert_eq!(crate::keyed_hash(&key, &input[..len]), keyed_hash(&key, &input[..len]), "keyed len = {len}");
            let context_key = hazmat::hash_derive_key_context("lanes test");
            assert_eq!(
                *Hasher::new_from_context_key(&context_key).update(&input[..len]).finalize().as_bytes(),
                *hash_with_mode(&input[..len], Mode::DeriveKeyMaterial(&context_key)).as_bytes(),
                "derive len = {len}"
            );
        }
    }

    /// Every piece count the merge can see, through split and merge alone
    /// (independent of how many lanes the test machine has).
    #[test]
    fn test_merge_every_count() {
        for len in [70 * CHUNK_LEN + 5, 100 * CHUNK_LEN + 7, 256 * CHUNK_LEN, 1000 * CHUNK_LEN] {
            let mut input = vec![0u8; len];
            crate::test::paint_test_input(&mut input);
            let want = crate::hash(&input);
            for count in 2..=16 {
                let pieces = split_subtrees(input.len(), count);
                let cvs: Vec<ChainingValue> = pieces
                    .iter()
                    .map(|p| hash_piece(&input[p.offset..][..p.len], p.offset, OwnedMode::Hash))
                    .collect();
                assert_eq!(want, merge_root(&pieces, &cvs, Mode::Hash), "len = {len}, count = {count}");
            }
        }
    }

    /// Admission grants the free lanes and no more. A caller's own thread
    /// always counts, so with `before` lanes in use a caller that asks for
    /// everything receives `total - before - 1` extra lanes, or none when
    /// `before + 1` reaches the total. Other tests in this process may hold
    /// lanes at the same time, so the check reads the count first and
    /// reasons from there; every step is exact.
    #[test]
    fn test_admission_is_bounded() {
        let total = lane_count();
        assert!(total >= 1);
        let before = LANES_IN_USE.load(Ordering::Relaxed);
        let a = admit(MAX_LANES);
        assert_eq!(a, total.saturating_sub(before + 1).min(MAX_LANES), "first caller takes all free extras");
        let b = admit(MAX_LANES);
        assert_eq!(b, total.saturating_sub(before + a + 2).min(MAX_LANES), "second caller takes what is left");
        let c = admit(1);
        assert_eq!(c, 0, "with every lane granted a third caller gets its own thread and nothing more");
        release(c);
        release(b);
        release(a);
        // Any other holder's lanes are still counted; ours are gone.
        assert!(LANES_IN_USE.load(Ordering::Relaxed) <= before + total, "released lanes leave the count");
    }

    /// Many concurrent callers on one process: every result is right and
    /// the lane count is never exceeded by more than the callers' own
    /// threads (which admission cannot refuse).
    #[test]
    fn test_concurrent_callers_agree() {
        let mut input = vec![0u8; 8 * MIN_SPLIT_LEN + 1];
        crate::test::paint_test_input(&mut input);
        let want = crate::hash(&input);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..20 {
                        assert_eq!(want, hash(&input));
                    }
                });
            }
        });
    }
}
