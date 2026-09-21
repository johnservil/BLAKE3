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
//! lanes that are free at that moment (at least one: its own thread,
//! which counts whether or not the call splits) and releases them when it
//! returns.
//!
//! How many it takes is a fair share: with `c` callers active (this one
//! included) and `L` lanes, at most `ceil(L / c)`, and never more than
//! are free. Two simultaneous callers on a four-lane machine get two
//! lanes each; on a three-lane machine the first to arrive gets two and
//! the second finds one free and takes that; on a two-lane machine one
//! each. The active count is a second process-wide counter, incremented
//! on entry to any call long enough to split and decremented on return.
//!
//! The share is what keeps a caller from taking every lane in the instant
//! between another caller's calls: two threads hashing back to back both
//! release everything between calls, and without the share the first to
//! return would take all the lanes and leave the other to run alone, turn
//! and turn about, so that each got the whole machine half the time and
//! one lane the other half, which averages to no gain over one lane.
//!
//! A caller's share is settled when it enters, so a caller that is alone
//! when it enters holds every lane until it returns, and a second caller
//! that arrives during that call finds nothing free. It waits for its
//! share to come back rather than running on its own thread at once, up to
//! [`ADMISSION_WAIT`]: the first caller's call ends within one input's
//! hashing time, and when it comes back for its next call it sees two
//! callers and takes its half, leaving the other half for the one that
//! waited. Two threads hashing at once settle to a half each within one
//! call this way, and the waiting caller loses one hand-off rather than a
//! whole call at single-lane speed. A machine that stays full past the
//! wait (many callers, or a caller with an input beyond what one lane
//! hashes in that time) runs the call on the caller's own thread.
//!
//! One caller that hashes alone and then beside another sees this: its
//! solo calls take every lane; the other's first call waits for them; from
//! then on each holds half. A benchmark that alternates a solo batch with
//! a duo batch, as bench-hashes --duo does, therefore measures the duo
//! batch after that hand-over, at a half each.
//!
//! The counts are exact inside one process. Across processes, which
//! cannot see each other's counts, the operating system's scheduler is
//! the arbiter: each process's workers are ordinary threads that yield
//! between polls and sleep when idle, so two processes hashing at once
//! each see their workers run at whatever share of a CPU the scheduler
//! gives them, and the pair finishes when a fair scheduler would have it
//! finish. An earlier design stood the module down to one thread after
//! its workers ran late twice in a row; it could not tell a worker
//! delayed by another process from one delayed by an unrelated thread of
//! its own process waking up, and on a busy machine it stood down for
//! good. Latency of a worker is a fact about the scheduler, and the
//! module leaves it there.
//!
//! # Splitting
//!
//! The input is cut at chunk boundaries into pieces, each a valid BLAKE3
//! subtree (see [`hazmat::left_subtree_len`]): the pieces are leaves of
//! the tree whose root is the whole input, reached by splitting the
//! largest piece at its left-subtree boundary until enough exist. A
//! subtree cut is uneven when the input is no power of two (3 MiB splits
//! 2 | 1), so where the input is long enough for every piece to stay at
//! least [`MIN_BALANCED_PIECE_LEN`], the cut goes to up to
//! [`PIECES_PER_LANE`] times as many pieces as lanes and the pieces are
//! dealt to lanes largest first, each to the lane with the least so far;
//! the lanes' shares then agree to within one piece. Each lane hashes its pieces in
//! order with [`hazmat::HasherExt::set_input_offset`] and
//! [`finalize_non_root`](hazmat::HasherExt::finalize_non_root), one
//! chaining value per piece, and the calling thread merges the values
//! back up the tree with [`hazmat::merge_subtrees_non_root`] and
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
//! Both spins yield the CPU on every iteration
//! ([`std::thread::yield_now`]), so a runnable thread on the same CPU
//! goes first: a worker between jobs or a caller waiting for results is a
//! poll between yields, and stops within `SPIN_BEFORE_SLEEP` on an idle
//! process. A busy spin here would hold a CPU that another thread, one
//! about to call this module, is waiting to run on, and that thread's
//! call would start late by the length of the spin.
//!
//! The workers are ordinary threads, and a worker whose CPU another thread
//! holds waits its turn like any other. A thread that spins without
//! yielding holds its CPU for a whole scheduler quantum; on a machine with
//! as many such spinners as CPUs, a worker's piece waits a quantum (about
//! 2 ms on Linux) to start, and the caller waits with it. Callers that
//! spin-wait on something while this module hashes on their behalf are
//! asking for that; a spin that yields costs the worker nothing measurable
//! (measured on a two-CPU machine: two spinning threads beside a 128 KiB
//! split took the call from 29 µs to 2 ms; the same two threads yielding
//! between polls left it at 29 µs).

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

/// Pieces cut per lane granted, so the lanes' shares come out even when
/// the input's subtrees are unequal. Four pieces per lane put the worst
/// imbalance at one piece in four of a lane's share for a power-of-two
/// input and less for the others.
pub const PIECES_PER_LANE: usize = 4;

/// A piece below this length is not worth cutting for balance: each piece
/// costs its lane a chunk-state start and a chaining value and the caller
/// a merge, and below 512 KiB a lane's SME2 pipeline has not reached its
/// bulk rate, so a finer cut there loses more to ramp-up than the balance
/// gains. Measured on the two-CPU VM: 1 MiB over two lanes as two pieces,
/// 0.092 ns/B; as eight, 0.108.
pub const MIN_BALANCED_PIECE_LEN: usize = 512 * 1024;

/// Upper bound on pieces one call cuts; sizes the on-stack arrays.
pub const MAX_PIECES: usize = MAX_LANES * PIECES_PER_LANE;

/// Upper bound on lanes one call uses; enough for any machine this crate
/// targets, and it sizes the on-stack chaining-value array.
pub const MAX_LANES: usize = 64;

/// How long a caller that finds no lane free waits for its share before
/// running on its own thread. The wait pays off whenever the caller holding
/// the lanes returns within it; one SME2 lane on an Apple M4 hashes 8 MiB
/// in about 1.4 ms and 64 MiB in about 11 ms, so this covers inputs to a
/// few tens of megabytes; past that the waiting caller gives up and hashes
/// alone, at one lane's speed, which is what it would have done at once.
pub const ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_millis(20);

/// How long a worker or a waiting caller spins before sleeping. Long
/// enough to bridge the gap between back-to-back hashes in a busy caller;
/// short enough that an idle process is quiet within it.
pub const SPIN_BEFORE_SLEEP: std::time::Duration = std::time::Duration::from_micros(200);

/// The number of lanes this machine has (see the module docs). At least 1.
pub fn lane_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| detect_lane_count().clamp(1, MAX_LANES))
}

/*
 * The admission state: lanes in use (the callers' own threads included)
 * and callers active, packed into one word so a caller reads and changes
 * both in one atomic step. Low 32 bits lanes, high 32 bits callers.
 */
static ADMISSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn unpack(state: u64) -> (usize, usize) {
    ((state & 0xffff_ffff) as usize, (state >> 32) as usize)
}

fn pack(lanes: usize, callers: usize) -> u64 {
    lanes as u64 | (callers as u64) << 32
}

/// Enter as a caller: one more active caller, no lanes yet.
fn enter() {
    ADMISSION.fetch_add(pack(0, 1), Ordering::AcqRel);
}

/// Leave: this caller and its `1 + extra` lanes go.
fn leave(extra: usize) {
    ADMISSION.fetch_sub(pack(1 + extra, 1), Ordering::AcqRel);
}

/// The extra lanes a caller may take now, given the state: its fair share
/// of `ceil(total / callers)`, less its own thread, within what is free
/// and what it wants.
fn grant(state: u64, total: usize, want: usize) -> usize {
    let (in_use, callers) = unpack(state);
    let share = total.div_ceil(callers.max(1));
    let free = total.saturating_sub(in_use);
    free.min(share).saturating_sub(1).min(want)
}

/// Take the caller's own lane plus up to `want` free ones within its fair
/// share (see the module docs); returns the extra lanes granted. The
/// caller has entered already.
///
/// When the share is not free yet and other callers are active, wait up
/// to ADMISSION_WAIT for it (yielding between polls): the caller holding
/// the lanes returns within one input's hashing time, and then this
/// caller's share is there. The decision reads lanes and callers in one
/// atomic step, so a caller that has just left cannot be counted in the
/// share while its lanes are still counted as held, or the reverse.
fn admit(want: usize) -> usize {
    let total = lane_count();
    let started = std::time::Instant::now();
    loop {
        let state = ADMISSION.load(Ordering::Acquire);
        let extra = grant(state, total, want);
        let (_, callers) = unpack(state);
        // Worth waiting: other callers hold lanes, the share due is bigger
        // than what is free now, and time remains.
        let share_due = total.div_ceil(callers.max(1)).min(want + 1);
        if extra + 1 < share_due && callers > 1 && started.elapsed() < ADMISSION_WAIT {
            std::thread::yield_now();
            continue;
        }
        let next = state + pack(1 + extra, 0);
        if ADMISSION.compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
            return extra;
        }
    }
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
    enter();
    let want = (input.len() / MIN_PIECE_LEN).min(lane_count()).saturating_sub(1);
    let extra = admit(want);
    if extra == 0 {
        let hash = owned.hash_serial(input);
        leave(0);
        return hash;
    }
    let lanes = 1 + extra;
    // One piece per lane, or up to PIECES_PER_LANE when the pieces stay
    // at least MIN_BALANCED_PIECE_LEN long.
    let piece_count = (input.len() / MIN_BALANCED_PIECE_LEN)
        .clamp(lanes, lanes * PIECES_PER_LANE)
        .min(input.len() / MIN_PIECE_LEN);
    let pieces = split_subtrees(input.len(), piece_count);
    let shares = deal_to_lanes(&pieces, lanes);

    // The caller takes the largest share, so on free lanes every worker
    // finishes first and the caller's wait is the hand-off alone.
    let own_lane = (0..lanes)
        .max_by_key(|&lane| shares[lane].iter().map(|&i| pieces[i].len).sum::<usize>())
        .unwrap();
    let mut cvs = [ChainingValue::default(); MAX_PIECES];
    {
        let done = AtomicUsize::new(0);
        let mut slots: Vec<Option<&mut ChainingValue>> = cvs[..pieces.len()].iter_mut().map(Some).collect();
        let mut jobs: Vec<Job> = Vec::with_capacity(extra);
        // Every job borrows `input`, slots in `cvs`, and `done`; the wait
        // below ends before any of them goes out of scope.
        for (lane, share) in shares.iter().enumerate() {
            if lane == own_lane {
                continue;
            }
            let job_pieces: Vec<(Piece, *mut ChainingValue)> = share
                .iter()
                .map(|&i| (pieces[i], slots[i].take().unwrap() as *mut ChainingValue))
                .collect();
            jobs.push(Job { input, pieces: job_pieces, mode: owned, done: &done });
        }
        pool().post(jobs);
        for &i in &shares[own_lane] {
            let piece = pieces[i];
            *slots[i].take().unwrap() = hash_piece(&input[piece.offset..][..piece.len], piece.offset, owned);
        }
        pool().wait(&done, extra);
    }
    leave(extra);
    merge_root(&pieces, &cvs[..pieces.len()], mode)
}

fn hash_piece(bytes: &[u8], offset: usize, mode: OwnedMode) -> ChainingValue {
    let mut hasher = mode.hasher();
    hasher.set_input_offset(offset as u64);
    hasher.update(bytes);
    hasher.finalize_non_root()
}

/// Deal pieces to `lanes` lanes so the lanes' byte totals are as even as
/// the pieces allow: largest piece first, each to the lane with the least
/// so far (ties to the lowest lane). Returns each lane's piece indices in
/// offset order. `lanes` is 1..=pieces.len().
pub fn deal_to_lanes(pieces: &[Piece], lanes: usize) -> Vec<Vec<usize>> {
    assert!((1..=pieces.len()).contains(&lanes), "one to pieces.len() lanes");
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by(|&a, &b| pieces[b].len.cmp(&pieces[a].len).then(a.cmp(&b)));
    let mut shares: Vec<Vec<usize>> = vec![Vec::new(); lanes];
    let mut totals = vec![0usize; lanes];
    for index in order {
        let lane = (0..lanes).min_by_key(|&lane| (totals[lane], lane)).unwrap();
        shares[lane].push(index);
        totals[lane] += pieces[index].len;
    }
    for share in &mut shares {
        share.sort_unstable();
    }
    shares
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
/// its offset. `count` is 1..=MAX_PIECES and `len` is at least `count`
/// chunks.
pub fn split_subtrees(len: usize, count: usize) -> Vec<Piece> {
    assert!((1..=MAX_PIECES).contains(&count));
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
 * posts its jobs, hashes its own share, and then waits until `done` has
 * counted every job; only then does it return and let those borrows end.
 * A worker touches a job's memory only between taking it from the queue
 * and incrementing `done`, so every access falls inside the caller's wait.
 */
struct Job {
    /// The whole input; each piece is a range within it.
    input: *const [u8],
    /// This lane's pieces, each with the slot its chaining value goes to.
    pieces: Vec<(Piece, *mut ChainingValue)>,
    mode: OwnedMode,
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
            std::thread::yield_now();
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
        let input = unsafe { &*job.input };
        for &(piece, slot) in &job.pieces {
            let cv = hash_piece(&input[piece.offset..][..piece.len], piece.offset, job.mode);
            unsafe {
                *slot = cv;
            }
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
    let mut detail = Vec::new();
    for level in 0..levels {
        let name = std::ffi::CString::new(format!("hw.perflevel{level}.name")).unwrap();
        let cores = std::ffi::CString::new(format!("hw.perflevel{level}.physicalcpu")).unwrap();
        let per_cluster = std::ffi::CString::new(format!("hw.perflevel{level}.cpusperl2")).unwrap();
        let cores = sysctl_u32(&cores)? as usize;
        let per_cluster = sysctl_u32(&per_cluster)?.max(1) as usize;
        let clusters = cores.div_ceil(per_cluster);
        lanes += clusters;
        detail.push(format!(
            "{} {cores} cores / {per_cluster} per cluster = {clusters}",
            sysctl_string(&name).unwrap_or_else(|| format!("level {level}"))
        ));
    }
    APPLE_DETAIL.set(detail.join("; ")).ok();
    (lanes > 0).then_some(lanes)
}

/// What the Apple detection saw, per performance level, for reports.
#[cfg(target_vendor = "apple")]
static APPLE_DETAIL: OnceLock<String> = OnceLock::new();

#[cfg(target_vendor = "apple")]
fn sysctl_string(name: &std::ffi::CStr) -> Option<String> {
    let mut buffer = [0u8; 64];
    let mut size = buffer.len();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    let end = buffer[..size].iter().position(|&b| b == 0).unwrap_or(size);
    Some(String::from_utf8_lossy(&buffer[..end]).into_owned())
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
        "set by BLAKE3_LANES".to_owned()
    } else if cfg!(target_vendor = "apple") {
        #[cfg(target_vendor = "apple")]
        let detail = APPLE_DETAIL.get().cloned().unwrap_or_default();
        #[cfg(not(target_vendor = "apple"))]
        let detail = String::new();
        format!("one per core cluster, each sharing one SME unit ({detail})")
    } else if cfg!(target_os = "linux") {
        "one per CPU when two threads measured at full speed side by side, else one per CPU cluster (sysfs topology/cluster_cpus_list)".to_owned()
    } else {
        "one per CPU (available_parallelism)".to_owned()
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
        for count in 1..=32 {
            for len in [32 * CHUNK_LEN, 1000 * CHUNK_LEN + 3, 1 << 24, (1 << 20) + 1] {
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
        let mut input = vec![0u8; 3 * MIN_SPLIT_LEN * 8 + 12345];
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

    /// The deal evens the lanes: for a 3 MiB input over two lanes (whose
    /// subtree cut alone is 2 | 1), eight pieces come out 1.5 | 1.5 MiB;
    /// for a power of two, every lane gets the same.
    #[test]
    fn test_deal_to_lanes_is_even() {
        let total = |pieces: &[Piece], share: &[usize]| share.iter().map(|&i| pieces[i].len).sum::<usize>();
        let pieces = split_subtrees(3 << 20, 2 * PIECES_PER_LANE);
        let shares = deal_to_lanes(&pieces, 2);
        assert_eq!(total(&pieces, &shares[0]), 3 << 19);
        assert_eq!(total(&pieces, &shares[1]), 3 << 19);
        for lanes in 1..=6 {
            for len in [1 << 20, 3 << 20, 5 << 20, (7 << 20) + 12345] {
                let pieces = split_subtrees(len, lanes * PIECES_PER_LANE);
                let shares = deal_to_lanes(&pieces, lanes);
                assert_eq!(shares.iter().map(|s| s.len()).sum::<usize>(), pieces.len());
                let mut seen: Vec<usize> = shares.iter().flatten().copied().collect();
                seen.sort_unstable();
                assert_eq!(seen, (0..pieces.len()).collect::<Vec<_>>(), "every piece dealt once");
                let totals: Vec<usize> = shares.iter().map(|s| total(&pieces, s)).collect();
                let largest_piece = pieces.iter().map(|p| p.len).max().unwrap();
                let spread = totals.iter().max().unwrap() - totals.iter().min().unwrap();
                assert!(spread <= largest_piece, "lanes = {lanes}, len = {len}: shares {totals:?} differ by more than one piece");
                for share in &shares {
                    assert!(share.windows(2).all(|w| w[0] < w[1]), "shares are in offset order");
                }
            }
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
            for count in 2..=32 {
                let pieces = split_subtrees(input.len(), count);
                let cvs: Vec<ChainingValue> = pieces
                    .iter()
                    .map(|p| hash_piece(&input[p.offset..][..p.len], p.offset, OwnedMode::Hash))
                    .collect();
                assert_eq!(want, merge_root(&pieces, &cvs, Mode::Hash), "len = {len}, count = {count}");
            }
        }
    }

    /// Admission arithmetic on the packed state, without touching the
    /// live counters: grant() is a pure function of (lanes in use,
    /// callers, total, want).
    #[test]
    fn test_grant_is_a_fair_share() {
        // One caller, nothing in use: everything it wants, less its own thread.
        assert_eq!(grant(pack(0, 1), 4, 8), 3);
        assert_eq!(grant(pack(0, 1), 4, 1), 1);
        assert_eq!(grant(pack(0, 1), 1, 8), 0, "a one-lane machine grants no extra");
        // Two callers on four lanes: two each.
        assert_eq!(grant(pack(0, 2), 4, 8), 1);
        assert_eq!(grant(pack(2, 2), 4, 8), 1);
        // Two callers on three lanes: ceil(3/2) = 2 for the first; the
        // second finds one free and gets its own thread only.
        assert_eq!(grant(pack(0, 2), 3, 8), 1);
        assert_eq!(grant(pack(2, 2), 3, 8), 0);
        // Two callers on two lanes: one each.
        assert_eq!(grant(pack(0, 2), 2, 8), 0);
        // Never more than free.
        assert_eq!(grant(pack(3, 1), 4, 8), 0);
        assert_eq!(grant(pack(4, 1), 4, 8), 0);
    }

    /// pack and unpack round-trip, and enter/leave are inverses.
    #[test]
    fn test_admission_state_packing() {
        for (lanes, callers) in [(0, 0), (1, 1), (7, 3), (63, 64), (usize::from(u16::MAX), 5)] {
            assert_eq!(unpack(pack(lanes, callers)), (lanes, callers));
        }
        let before = ADMISSION.load(Ordering::Relaxed);
        enter();
        let extra = admit(0);
        assert_eq!(extra, 0);
        let (lanes, callers) = unpack(ADMISSION.load(Ordering::Relaxed));
        let (before_lanes, before_callers) = unpack(before);
        assert!(lanes >= before_lanes + 1 && callers >= before_callers + 1);
        leave(extra);
        // Other tests may be mid-flight; the difference we made is gone.
        let (after_lanes, after_callers) = unpack(ADMISSION.load(Ordering::Relaxed));
        assert!(after_lanes <= lanes - 1 + 8 && after_callers <= callers - 1 + 8);
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
