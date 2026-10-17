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
//! # Sources and the pool
//!
//! All the workers' work comes from *sources* ([`sources`]): a stream's
//! tasks, and each calling thread's one-shot jobs. A source numbers its
//! items; its owner publishes them with one store, and workers claim them
//! in order by compare-and-swap, with no lock and no shared queue. A call
//! publishes its job's pieces at once and hashes pieces itself until none
//! remain, then waits for the last ones in flight. Workers, started once per
//! process ([`crate::initialize_multithreaded`]), one per CPU beyond the
//! first, take from every registered source, one item at a time: two callers
//! hashing at once each get about half the workers' time, and a slow
//! thread takes fewer pieces than a fast one; there is nothing to tune for
//! fairness or balance. A call arriving when callers already fill the CPUs
//! hashes its input whole on its own thread.
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
//! A worker that finds nothing parks on its own thread's token (no two
//! share a lock or a futex word: nineteen workers sleeping through one
//! mutex and condition variable spent 6-15% of all samples in the kernel's
//! futex-bucket spinlock, i7-12700K). While a stream is open it first looks
//! again for [`SPIN`] (more is coming). A call wakes only as many workers
//! as it can use, waking one itself (a system call) and leaving the rest to
//! the woken: a woken worker arrives 15-45 us later on a core whose clock
//! the idle time lowered, so inputs below [`MIN_SPLIT_LEN`] stay on the
//! caller's thread. A caller waiting for its last pieces polls, then parks
//! after [`SPIN_BEFORE_SLEEP`]; its polls yield the CPU every
//! [`YIELD_EVERY`], so a runnable thread never waits a scheduler quantum
//! behind a poller (two threads spinning without yielding took a 128 KiB
//! split from 29 us to 2 ms on a two-CPU machine).

use crate::hazmat::ChainingValue;
#[cfg(test)]
use crate::hazmat::{self, Mode};
#[cfg(test)]
use crate::hazmat::HasherExt;
use crate::platform::Platform;
use crate::{CHUNK_LEN, Hash};
#[cfg(test)]
use crate::{Hasher, KEY_LEN};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

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
    let Some((pool, _caller)) = spare_pool() else { return crate::hash_serial(input, key, flags) };
    let (pieces, mut cvs) = tree_cvs(pool, input, key, 0, flags);
    merge_root(&pieces, &mut cvs, key, flags)
}

/// The pool, this caller counted until the guard drops; or None with no
/// CPU to spare (every one has a caller, or the process has one CPU), when
/// the call hashes on its own thread. The count lets concurrent callers
/// see that this CPU already has work.
fn spare_pool() -> Option<(&'static Pool, Caller<'static>)> {
    let pool = pool();
    let callers = pool.callers.fetch_add(1, Ordering::SeqCst) + 1;
    let caller = Caller(&pool.callers);
    (callers < pool.cpus).then_some((pool, caller))
}

/// The subtree pieces of `input` at chunk `counter` and their chaining
/// values, hashed over the pool (the SME2 thread's prefix first, where it
/// takes one).
fn tree_cvs(pool: &Pool, input: &[u8], key: &crate::CVWords, counter: u64, flags: u8) -> (Vec<Piece>, Vec<ChainingValue>) {
    let threads = pool.cpus;
    let turn = crate::platform::Sme2Turn::take(Platform::detect(), true);
    let (pieces, own) = cut_with_prefix(input.len(), threads, prefix_for(&turn, input.len(), threads));
    let mut cvs = vec![ChainingValue::default(); pieces.len()];
    let work = Work::Tree { input, pieces: &pieces, cvs: cvs.as_mut_ptr(), key: *key, counter, flags };
    pool.run_job(work, pieces.len(), own, turn.platform());
    drop(turn);
    (pieces, cvs)
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
    let Some((pool, _caller)) = spare_pool() else { return crate::compress_subtree_to_parent_node(input, 0, key, counter, flags, pool_platform()) };
    let (pieces, mut cvs) = tree_cvs(pool, input, key, counter, flags);
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
    let Some((pool, _caller)) = spare_pool() else { return crate::hash_many_serial(input, message_len, key, flags, outputs) };
    let pieces = cut_messages(outputs.len(), slot, pool.cpus);
    let work = Work::Messages { input, message_len, key: *key, flags, pieces: &pieces, outputs: outputs.as_mut_ptr() };
    pool.run_job(work, pieces.len(), 0, pool_platform());
}

/// [`crate::hash_each_multithreaded_with`]'s pool job: `items`, every one
/// under MIN_SPLIT_LEN, cut into ranges of about [`next_piece_len`] of the
/// bytes that remain.
pub(crate) fn hash_each(key: &crate::CVWords, flags: u8, items: &[&[u8]], out: &mut [[u8; crate::OUT_LEN]]) {
    assert!(items.iter().all(|i| i.len() < MIN_SPLIT_LEN), "items under MIN_SPLIT_LEN");
    let Some((pool, _caller)) = spare_pool() else {
        let platform = Platform::detect();
        return crate::each::hash_each_on(key, flags, items, out, platform, |item| crate::hash_serial_on(item, key, flags, platform));
    };
    let mut remaining: usize = items.iter().map(|i| i.len()).sum();
    let mut pieces = Vec::with_capacity(64);
    let mut start = 0;
    while start < items.len() {
        let want = next_piece_len(remaining, pool.cpus);
        let (mut end, mut bytes) = (start, 0);
        while end < items.len() && (bytes < want || end == start) {
            bytes += items[end].len();
            end += 1;
        }
        remaining -= bytes;
        pieces.push(Piece { offset: start, len: end - start });
        start = end;
    }
    let work = Work::Each { items, key: *key, flags, pieces: &pieces, outputs: out.as_mut_ptr() };
    pool.run_job(work, pieces.len(), 0, pool_platform());
}

/// The chaining value of each of `groups`, a message's 16 KiB groups from
/// `first_group` on, each whole but the last (crate::outboard), into
/// `cvs`, over the machine's threads: below MIN_SPLIT_LEN of them on this
/// thread.
pub(crate) fn group_cvs(groups: &[&[u8]], first_group: u64, key: &crate::CVWords, flags: u8, cvs: &mut [ChainingValue]) {
    assert_eq!(groups.len(), cvs.len(), "a chaining value for each group");
    let pooled = if groups.len() * crate::outboard::GROUP_LEN < MIN_SPLIT_LEN { None } else { spare_pool() };
    let Some((pool, _caller)) = pooled else { return crate::outboard::groups_cvs_into(groups, first_group, key, flags, Platform::detect(), cvs) };
    let pieces = cut_messages(groups.len(), crate::outboard::GROUP_LEN, pool.cpus);
    let work = Work::Groups { groups, first_group, pieces: &pieces, key: *key, flags, cvs: cvs.as_mut_ptr() };
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

/// A counted call (spare_pool), until it returns, unwinding too.
struct Caller<'a>(&'a AtomicUsize);

impl Drop for Caller<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One call's work, on the caller's stack, reached by workers through the
/// caller's source ([`sources::job_source`]) while the call runs: a
/// worker that took piece `i` hashes it and counts it finished, and the
/// caller returns only once every piece is, so every worker access falls
/// inside the caller's frame.
pub(crate) struct Job<'a> {
    work: Work<'a>,
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
    /// Ranges of a message's 16 KiB groups, wherever each lies (`offset`
    /// and `len` count groups, from `first_group`); one chaining value each
    /// (crate::outboard).
    Groups {
        groups: &'a [&'a [u8]],
        first_group: u64,
        pieces: &'a [Piece],
        key: crate::CVWords,
        flags: u8,
        cvs: *mut ChainingValue,
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
    /// Ranges of a collection's items (`offset` and `len` count items),
    /// each under MIN_SPLIT_LEN; one digest each.
    Each {
        items: &'a [&'a [u8]],
        key: crate::CVWords,
        flags: u8,
        pieces: &'a [Piece],
        outputs: *mut [u8; crate::OUT_LEN],
    },
}

/// Any value on a cache line of its own.
#[repr(align(128))]
pub(crate) struct OwnLine<T>(pub(crate) T);

impl<T> std::ops::Deref for OwnLine<T> {
    type Target = T;
    fn deref(&self) -> &T {
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
                let cv = crate::hash_all_at_once(bytes, key, counter, *flags, platform).chaining_value();
                unsafe { *cvs.add(index) = cv };
            }
            Work::Groups { groups, first_group, pieces, key, flags, cvs } => {
                let piece = pieces[index];
                // Sound: this range of chaining values belongs to piece `index` alone.
                let out = unsafe { core::slice::from_raw_parts_mut(cvs.add(piece.offset), piece.len) };
                crate::outboard::groups_cvs_into(&groups[piece.offset..][..piece.len], first_group + piece.offset as u64, key, *flags, platform, out);
            }
            Work::Messages { input, message_len, key, flags, pieces, outputs } => {
                let piece = pieces[index];
                let slot = crate::many::slot_len(*message_len);
                let messages = &input[piece.offset * slot..][..piece.len * slot];
                // Sound: this range of outputs belongs to piece `index` alone.
                let digests = unsafe { core::slice::from_raw_parts_mut(outputs.add(piece.offset), piece.len) };
                crate::many::hash_many_on(messages, *message_len, key, *flags, digests, platform);
            }
            Work::Each { items, key, flags, pieces, outputs } => {
                let piece = pieces[index];
                // Sound: this range of outputs belongs to piece `index` alone.
                let digests = unsafe { core::slice::from_raw_parts_mut(outputs.add(piece.offset), piece.len) };
                let range = &items[piece.offset..][..piece.len];
                crate::each::hash_each_on(key, *flags, range, digests, platform, |item| crate::hash_serial_on(item, key, *flags, platform));
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

/// [`cut_with_prefix`] with no prefix (the tests').
#[cfg(test)]
fn cut_subtrees(len: usize, threads: usize) -> Vec<Piece> {
    cut_with_prefix(len, threads, 0).0
}

/// Cut `len` bytes into subtrees for `threads` threads, in offset order,
/// after a prefix of about `prefix` bytes for the SME2 thread. Past the
/// prefix, each piece is [`next_piece_len`] of what remains, and the last
/// is whatever is left. The prefix is whole subtrees of up to
/// [`MAX_PREFIX_PIECE_LEN`], rounded down to a multiple of the first piece
/// after it. A piece's length never exceeds the one before it and every
/// piece starts at a multiple of its own length, so each is a whole
/// subtree at its offset. Returns the pieces and how many form the prefix
/// (none when `prefix` rounds to nothing). `prefix` is below `len`; `len`
/// exceeds [`MIN_PIECE_LEN`] and at least two threads share it, so two or
/// more pieces come back.
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
    crate::hasher::parent_output(&cvs[0], &cvs[1], key, flags, pool_platform()).root_hash()
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

/// The pool: cpus - 1 workers taking from the registered sources
/// ([`sources`]), the SME2 thread beside them where the CPU has SME2, and
/// the record of which workers are parked.
struct Pool {
    /// Calls on the pool now ([`spare_pool`]).
    callers: AtomicUsize,
    cpus: usize,
    /// Whether the SME2 thread runs (this CPU has SME2).
    sme2: bool,
    /// Workers parked: bit w of word w / 64 for worker w (0 for
    /// blake3-worker-1). A worker sets its own bit before it parks; a
    /// waker clears one (fetch_and), which makes it the one to unpark that
    /// worker. Each worker parks on its own thread's token (std's park), so
    /// no two share a lock or a futex word.
    parked: Box<[OwnLine<AtomicU64>]>,
    /// Each worker's thread, set as it starts.
    threads: Box<[OnceLock<std::thread::Thread>]>,
    /// Wakes still owed: each woken worker pays them before it works.
    owed: AtomicUsize,
}

/// Create the pool and start its threads in the background, once per
/// process ([`crate::initialize_multithreaded`] is its public face): this
/// thread starts the SME2 thread and worker 1, and worker 1 the other
/// workers. Calls run on the workers started so far.
pub(crate) fn initialize() {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let sme2 = cfg!(blake3_sme2) && !matches!(pool_platform(), p if core::mem::discriminant(&p) == core::mem::discriminant(&Platform::detect()));
    let pool = Pool {
        callers: AtomicUsize::new(0),
        cpus,
        sme2,
        parked: (0..cpus.div_ceil(64)).map(|_| OwnLine(AtomicU64::new(0))).collect(),
        threads: (0..cpus).map(|_| OnceLock::new()).collect(),
        owed: AtomicUsize::new(0),
    };
    assert!(POOL.set(pool).is_ok(), "initialize_multithreaded creates the pool once");
    if sme2 {
        std::thread::Builder::new().name("blake3-sme2".into()).spawn(sme2_main).expect("spawning the SME2 thread");
    }
    if cpus == 1 {
        return;
    }
    std::thread::Builder::new()
        .name("blake3-worker-1".into())
        .spawn(move || {
            for worker in 2..cpus {
                std::thread::Builder::new()
                    .name(format!("blake3-worker-{worker}"))
                    .spawn(move || worker_main(worker - 1))
                    .expect("spawning a BLAKE3 worker");
            }
            worker_main(0)
        })
        .expect("spawning a BLAKE3 worker");
}

/// The contract's check at every multithreaded call and stream: a call
/// before [`crate::initialize_multithreaded`] panics, in every build. It
/// costs one load and a branch the CPU predicts.
#[inline]
pub(crate) fn require() {
    pool();
}

/// The pool, once [`initialize`] has created it.
static POOL: OnceLock<Pool> = OnceLock::new();

/// The pool. The caller must have called
/// [`crate::initialize_multithreaded`] before; a call before it panics. The
/// check is the load of the pool, which every call makes anyway.
#[inline]
fn pool() -> &'static Pool {
    match POOL.get() {
        Some(pool) => pool,
        None => no_pool(),
    }
}

/// A multithreaded call before `initialize_multithreaded`: the caller broke
/// the contract. The crate's own unit tests, many small programs in one,
/// initialize here instead.
#[cold]
#[inline(never)]
fn no_pool() -> &'static Pool {
    #[cfg(test)]
    {
        crate::initialize_multithreaded();
        POOL.get().expect("initialize_multithreaded created the pool")
    }
    #[cfg(not(test))]
    panic!("blake3-servil: call blake3_servil::initialize_multithreaded() once, before any multithreaded call or stream: it starts the threads they use")
}

impl Pool {
    /// Publish `work` of `pieces` pieces as this thread's job source, take
    /// pieces on this thread until none remain, and return once every
    /// piece is finished, on whichever thread took it. The caller first
    /// hashes pieces `0..own` on `own_platform`, back to back (the SME2
    /// thread's prefix; the workers' pieces start after it). A caller that
    /// finds no source slot free hashes every piece itself.
    fn run_job(&self, work: Work, pieces: usize, own: usize, own_platform: Platform) {
        let job = Job { work };
        let source = sources::job_source();
        let slot = source.register();
        // Sound: the caller's frame outlives every access (it waits below).
        let ptr = &job as *const Job as *mut Job<'static>;
        let target = source.publish_job(ptr, pieces, own);
        if slot.is_some() && pieces > own + 1 {
            self.wake_for(pieces - own - 1);
        }
        for index in 0..own {
            // Sound: the workers' pieces start past these.
            unsafe { job.hash_piece(index, own_platform) };
        }
        source.finished(own);
        while let Some(sources::Item::Piece(_, index)) = source.take() {
            // Sound: this thread took `index`, so it alone writes its slot.
            unsafe { job.hash_piece(index, pool_platform()) };
            source.finished(1);
        }
        source.wait(target);
        if let Some(slot) = slot {
            sources::Source::unregister(slot);
        }
    }

    /// Make `wanted` workers awake or on their way, waking parked ones as
    /// needed: the caller wakes one, the woken wake the rest (owed). The
    /// read of `parked` follows the caller's publication of its work (a
    /// locked instruction, or the fence in publish), as a parking worker's
    /// re-check follows its bit: one of the two sees the other.
    pub(crate) fn wake_for(&self, wanted: usize) {
        let parked: usize = self.parked.iter().map(|w| w.load(Ordering::SeqCst).count_ones() as usize).sum();
        if parked == 0 {
            return;
        }
        let awake = self.cpus - 1 - parked;
        if awake < wanted {
            let need = (wanted - awake).min(parked);
            if need > 1 {
                self.owed.fetch_add(need - 1, Ordering::SeqCst);
            }
            self.wake_one();
        }
    }

    /// Wake one parked worker, if any: whether one was.
    pub(crate) fn wake_one(&self) -> bool {
        for (w, word) in self.parked.iter().enumerate() {
            let mut bits = word.load(Ordering::SeqCst);
            while bits != 0 {
                let bit = 1u64 << bits.trailing_zeros();
                let before = word.fetch_and(!bit, Ordering::SeqCst);
                if before & bit != 0 {
                    let worker = w * 64 + bit.trailing_zeros() as usize;
                    self.threads[worker].get().expect("a parked worker has set its thread").unpark();
                    return true;
                }
                bits = before & !bit;
            }
        }
        false
    }

    /// Whether any worker is parked (one load a word).
    pub(crate) fn any_parked(&self) -> bool {
        self.parked.iter().any(|w| w.load(Ordering::Relaxed) != 0)
    }

    /// Worker `me`'s loop: the sources' tasks and pieces; spinning a while
    /// when a stream is open (more is coming), else parked, whenever it
    /// finds nothing.
    fn serve(&self, me: usize) -> ! {
        let (word, bit) = (&self.parked[me / 64], 1u64 << (me % 64));
        loop {
            if sources::run_one() {
                continue;
            }
            if sources::open() && spin_for_work() {
                continue;
            }
            // Park: the bit first, then one more look (a publisher stores,
            // then reads `parked`; this sets the bit, then looks: one of the
            // two sees the other), then the token.
            word.fetch_or(bit, Ordering::SeqCst);
            if sources::visible(false) {
                // Ours to clear unless a waker cleared it first (then its
                // unpark leaves a token, and a later park returns at once).
                word.fetch_and(!bit, Ordering::SeqCst);
                continue;
            }
            while word.load(Ordering::SeqCst) & bit != 0 {
                std::thread::park();
            }
            // Woken: pay the wakes owed before working.
            while self.owed.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_ok() {
                if !self.wake_one() {
                    self.owed.store(0, Ordering::SeqCst);
                    break;
                }
            }
        }
    }
}

/// How long a worker that finds nothing keeps looking while a stream is
/// open (more is coming) before it parks: 5 us. i7-12700K, against the
/// stream on the earlier pool: none cost small messages 9-12% (a park and
/// a wake at every gap); 20 us cost 22% more CPU time a run for no speed;
/// a cap on how many workers spin did not help.
const SPIN: std::time::Duration = std::time::Duration::from_micros(0);

/// Look for work for up to SPIN: whether some appeared.
fn spin_for_work() -> bool {
    let started = std::time::Instant::now();
    loop {
        for _ in 0..16 {
            if sources::visible(false) {
                return true;
            }
            core::hint::spin_loop();
        }
        if started.elapsed() >= SPIN || !sources::open() {
            return false;
        }
    }
}

/// The workers' work: sources, each a numbered sequence of items that its
/// owner publishes and workers claim. One owner writes `published` (one
/// Release store); workers advance `claimed` by compare-and-swap, only
/// below `published`. Both only count up, across everything a source ever
/// carries, so a stale claim is impossible. A stream's source carries its
/// tasks in a ring (a slot written again only after its task was taken,
/// copied out); a calling thread's carries its one-shot jobs' pieces,
/// numbered on from job to job (job k owns [base, base + pieces)). Sources
/// are never freed (recycled), so a worker may read one after its owner is
/// done with it.
pub(crate) mod sources {
    use super::*;
    use core::cell::UnsafeCell;
    use core::mem::MaybeUninit;

    /// Sources registered at once. A stream or a job that finds none free
    /// is hashed on its own thread.
    const SLOTS: usize = 64;

    static REGISTERED: [AtomicPtr<Source>; SLOTS] = [const { AtomicPtr::new(core::ptr::null_mut()) }; SLOTS];
    static ACTIVE: AtomicU64 = AtomicU64::new(0);
    /// Streams open: workers spin a while for more work only when some are.
    static OPEN: OwnLine<AtomicUsize> = OwnLine(AtomicUsize::new(0));

    #[repr(align(128))]
    pub(crate) struct Source {
        claimed: OwnLine<AtomicUsize>,
        published: OwnLine<AtomicUsize>,
        /// A stream's tasks; empty for a caller's jobs.
        ring: Box<[UnsafeCell<MaybeUninit<Task>>]>,
        /// A caller's current job, its first number, and the pieces
        /// finished (all its jobs'); the job stays the claimed numbers'
        /// until `finished` passes them.
        job: AtomicPtr<Job<'static>>,
        base: AtomicUsize,
        finished: OwnLine<AtomicUsize>,
        /// Whether the caller parked waiting for its job's last piece, and
        /// its thread: behind a lock, since the worker finishing the last
        /// piece may read it as the source passes to another thread (an
        /// unpark that reaches the new owner is a harmless spurious wake).
        sleeping: core::sync::atomic::AtomicBool,
        owner: Mutex<Option<std::thread::Thread>>,
    }

    // Sound: a ring slot is written by its stream only above `published`
    // and below `claimed` + len, and read by the one worker whose CAS took
    // it, across the Release store of `published` and the Acquire load;
    // `owner` is behind its lock; the rest is atomics.
    unsafe impl Sync for Source {}
    unsafe impl Send for Source {}

    /// What a claim takes: a stream's task, or a job's piece.
    pub(crate) enum Item {
        Task(Task),
        Piece(*const Job<'static>, usize),
    }

    impl Source {
        /// A source with a ring of `len` tasks (a stream's), or none (a
        /// caller's jobs).
        pub(crate) fn new(len: usize) -> Source {
            Source {
                claimed: OwnLine(AtomicUsize::new(0)),
                published: OwnLine(AtomicUsize::new(0)),
                ring: (0..len).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect(),
                job: AtomicPtr::new(core::ptr::null_mut()),
                base: AtomicUsize::new(0),
                finished: OwnLine(AtomicUsize::new(0)),
                sleeping: core::sync::atomic::AtomicBool::new(false),
                owner: Mutex::new(None),
            }
        }

        /// Register for the workers: the slot, or None when all are taken.
        pub(crate) fn register(&'static self) -> Option<usize> {
            let ptr = self as *const Source as *mut Source;
            let slot = (0..SLOTS).find(|&i| REGISTERED[i].compare_exchange(core::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::Relaxed).is_ok())?;
            ACTIVE.fetch_or(1 << slot, Ordering::SeqCst);
            Some(slot)
        }

        pub(crate) fn unregister(slot: usize) {
            ACTIVE.fetch_and(!(1 << slot), Ordering::SeqCst);
            REGISTERED[slot].store(core::ptr::null_mut(), Ordering::SeqCst);
        }

        /// A stream's task to the workers: one write, one Release store;
        /// then the SME2 thread first, and with two or more waiting and a
        /// worker parked, a worker.
        pub(crate) fn publish(&self, task: Task) {
            let p = self.published.load(Ordering::Relaxed);
            debug_assert!(p - self.claimed.load(Ordering::Relaxed) < self.ring.len(), "a source's ring holds every task not yet taken");
            // Sound: slot p % len is free (its last task was taken), and
            // only this stream writes it.
            unsafe { (*self.ring[p % self.ring.len()].get()).write(task) };
            self.published.store(p + 1, Ordering::Release);
            core::sync::atomic::fence(Ordering::SeqCst);
            let pool = pool();
            if pool.sme2 {
                wake_sme2();
            }
            if pool.any_parked() && p + 1 - self.claimed.load(Ordering::Relaxed) >= 2 {
                pool.wake_one();
            }
        }

        /// Make `job` the caller's current job, its pieces `0..own` the
        /// caller's own: the number `finished` must reach when it is done.
        pub(crate) fn publish_job(&self, job: *mut Job<'static>, pieces: usize, own: usize) -> usize {
            let base = self.published.load(Ordering::Relaxed);
            debug_assert_eq!(self.finished.load(Ordering::Relaxed), base, "the last job is finished");
            self.job.store(job, Ordering::Relaxed);
            self.base.store(base, Ordering::Relaxed);
            // No worker can claim a number yet (claimed == published), so
            // this store alone moves `claimed` past the caller's own pieces.
            self.claimed.store(base + own, Ordering::Relaxed);
            self.published.store(base + pieces, Ordering::Release);
            core::sync::atomic::fence(Ordering::SeqCst);
            base + pieces
        }

        /// Claim the next published item, if any.
        pub(crate) fn take(&self) -> Option<Item> {
            let mut c = self.claimed.load(Ordering::Acquire);
            loop {
                let p = self.published.load(Ordering::Acquire);
                if c >= p {
                    return None;
                }
                match self.claimed.compare_exchange_weak(c, c + 1, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) if self.ring.is_empty() => return Some(Item::Piece(self.job.load(Ordering::Relaxed), c - self.base.load(Ordering::Relaxed))),
                    // Sound: this thread alone took slot c, written before
                    // `published` passed it.
                    Ok(_) => return Some(Item::Task(unsafe { (*self.ring[c % self.ring.len()].get()).assume_init_read() })),
                    Err(now) => c = now,
                }
            }
        }

        /// A stream's next task, if any (this source is a stream's).
        pub(crate) fn take_task(&self) -> Option<Task> {
            match self.take() {
                Some(Item::Task(task)) => Some(task),
                Some(Item::Piece(..)) => unreachable!("a stream's source carries tasks"),
                None => None,
            }
        }

        fn waiting(&self) -> usize {
            self.published.load(Ordering::Acquire).saturating_sub(self.claimed.load(Ordering::Acquire))
        }

        /// Count `n` of the caller's job's pieces finished (their outputs
        /// written before); the last one wakes a caller parked for it.
        pub(crate) fn finished(&self, n: usize) {
            let now = self.finished.fetch_add(n, Ordering::SeqCst) + n;
            if now == self.published.load(Ordering::SeqCst) && self.sleeping.load(Ordering::SeqCst) {
                if let Some(owner) = &*self.owner.lock().unwrap() {
                    owner.unpark();
                }
            }
        }

        /// Wait until `finished` reaches `target`: polling a while, then
        /// parked until the worker finishing the last piece unparks this
        /// thread.
        pub(crate) fn wait(&self, target: usize) {
            let started = std::time::Instant::now();
            let mut yielded = started;
            while started.elapsed() < SPIN_BEFORE_SLEEP {
                if self.finished.load(Ordering::Acquire) >= target {
                    return;
                }
                poll_pause(&mut yielded);
            }
            self.sleeping.store(true, Ordering::SeqCst);
            while self.finished.load(Ordering::SeqCst) < target {
                std::thread::park();
            }
            self.sleeping.store(false, Ordering::SeqCst);
        }
    }

    /// Callers' job sources no thread holds.
    static FREE_JOBS: Mutex<std::vec::Vec<&'static Source>> = Mutex::new(std::vec::Vec::new());

    struct Held(&'static Source);

    impl Drop for Held {
        /// A thread's exit gives its source back, never freed.
        fn drop(&mut self) {
            FREE_JOBS.lock().unwrap().push(self.0);
        }
    }

    std::thread_local! {
        static MINE: std::cell::OnceCell<Held> = const { std::cell::OnceCell::new() };
    }

    /// This thread's source for its one-shot jobs.
    pub(crate) fn job_source() -> &'static Source {
        MINE.with(|mine| {
            mine.get_or_init(|| {
                let source = FREE_JOBS.lock().unwrap().pop().unwrap_or_else(|| Box::leak(Box::new(Source::new(0))));
                *source.owner.lock().unwrap() = Some(std::thread::current());
                Held(source)
            })
            .0
        })
    }

    pub(crate) fn opened() {
        OPEN.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn closed() {
        OPEN.fetch_sub(1, Ordering::SeqCst);
    }

    pub(crate) fn open() -> bool {
        OPEN.load(Ordering::Relaxed) > 0
    }

    /// The first registered source for which `f` says yes.
    fn find(mut f: impl FnMut(&'static Source) -> bool) -> bool {
        let mut active = ACTIVE.load(Ordering::SeqCst);
        while active != 0 {
            let slot = active.trailing_zeros() as usize;
            active &= active - 1;
            let ptr = REGISTERED[slot].load(Ordering::Acquire);
            // Sound: sources are never freed (they are recycled).
            if !ptr.is_null() && f(unsafe { &*ptr }) {
                return true;
            }
        }
        false
    }

    /// Run one item from any source: whether there was one. When two or
    /// more wait and a worker is parked, this one wakes the next first
    /// (the chain).
    pub(crate) fn run_one() -> bool {
        find(|source| {
            let Some(item) = source.take() else { return false };
            if source.waiting() >= 2 && pool().any_parked() {
                pool().wake_one();
            }
            match item {
                Item::Task(task) => task.run(pool_platform()),
                Item::Piece(job, index) => {
                    // Sound: the job lives until its pieces are finished,
                    // this one among them; this thread alone took `index`.
                    unsafe { (*job).hash_piece(index, pool_platform()) };
                    source.finished(1);
                }
            }
            true
        })
    }

    /// A task from any stream (the SME2 thread takes these alone).
    pub(crate) fn take_stream_task() -> Option<Task> {
        let mut found = None;
        find(|source| {
            found = if source.ring.is_empty() { None } else { source.take_task() };
            found.is_some()
        });
        found
    }

    /// Whether any source has work waiting (`streams`: any stream).
    pub(crate) fn visible(streams: bool) -> bool {
        find(|source| (!streams || !source.ring.is_empty()) && source.waiting() > 0)
    }
}

/// The longest part of a stream's input planned as tasks
/// (a stream's piece): a task is a whole subtree within one part.
pub(crate) const TASK_LEN: usize = 64 * CHUNK_LEN;

/// One piece of a stream's work (crate::stream), hashed by whichever thread
/// takes it from the stream's source: a whole subtree at chunk `counter` of a message,
/// as [`crate::hasher::subtree_len`] cuts it, its result a 64-byte block at `out`;
/// or, with `batch` the messages' length, a range of a batch's messages in
/// their slots, their digests at `out`. Then it counts down `left`; the
/// stream keeps the bytes, `out`, and `left` in place until `left` is zero.
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
    /// messages): each one's digest (digests) to its `out`, then `left`
    /// counted down by `members`; `len` sums their bytes.
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
}

const NO_MEMBER: Member = Member { input: core::ptr::null(), len: 0, out: core::ptr::null_mut() };

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
            self.run_members(platform);
            // Sound: the stream keeps `left` in place until it is zero.
            unsafe { &*self.left }.fetch_sub(self.members, Ordering::Release);
            return;
        }
        // Sound: the stream keeps these in place until `left` is zero.
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
            *out = crate::compress_subtree_to_parent_node(bytes, 0, &self.key, 0, self.flags, platform);
        } else {
            out[..crate::OUT_LEN].copy_from_slice(&crate::hash_all_at_once(bytes, &self.key, self.counter, self.flags, platform).chaining_value());
        }
        unsafe { &*self.left }.fetch_sub(1, Ordering::Release);
    }
}

impl Task {
    /// The members' digests: messages of one length side by side on the
    /// multi-lane kernels (`hash_many`'s, from a table of pointers) where
    /// that length is one block, or on x86 two to sixteen blocks of a power
    /// of two (128 B to 1 KiB); any others one at a time.
    fn run_members(&self, platform: Platform) {
        let members = &self.member[..self.members];
        if let Some(message_len) = self.batch {
            for m in members {
                // Sound: the stream keeps each batch's bytes and digest
                // space in place until the task's `left` is zero.
                let bytes = unsafe { core::slice::from_raw_parts(m.input, m.len) };
                let count = m.len / crate::many::slot_len(message_len);
                let digests = unsafe { core::slice::from_raw_parts_mut(m.out as *mut [u8; crate::OUT_LEN], count) };
                crate::many::hash_many_on(bytes, message_len, &self.key, self.flags, digests, platform);
            }
            return;
        }
        // Sound: the stream keeps every member's bytes and `out` in place
        // until the task's `left` is zero.
        if members.len() >= 2 && members.iter().all(|m| m.len == members[0].len) {
            match members[0].len {
                crate::BLOCK_LEN => return self.run_equal_members::<{ crate::BLOCK_LEN }>(platform),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                128 => return self.run_equal_members::<128>(platform),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                256 => return self.run_equal_members::<256>(platform),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                512 => return self.run_equal_members::<512>(platform),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                CHUNK_LEN => return self.run_equal_members::<CHUNK_LEN>(platform),
                _ => {}
            }
        }
        for m in members {
            let bytes = unsafe { core::slice::from_raw_parts(m.input, m.len) };
            let hash = crate::hash_serial_on(bytes, &self.key, self.flags, platform);
            unsafe { core::ptr::copy_nonoverlapping(hash.as_bytes().as_ptr(), m.out, crate::OUT_LEN) };
        }
    }
}

impl Task {
    /// The members' digests, each a message of `N` bytes (whole blocks, at
    /// most a chunk; run_members checked), side by side on the multi-lane
    /// kernel from a table of pointers.
    fn run_equal_members<const N: usize>(&self, platform: Platform) {
        let members = &self.member[..self.members];
        // Sound: the stream keeps every member's N bytes and `out` in place
        // until the task's `left` is zero.
        let table: arrayvec::ArrayVec<&[u8; N], MEMBERS> = members.iter().map(|m| unsafe { &*(m.input as *const [u8; N]) }).collect();
        let mut digests = [[0u8; crate::OUT_LEN]; MEMBERS];
        // One block carries every flag at once, and passes them as the
        // 64-byte batches do (flags_start and flags_end 0): with every
        // caller of hash_many::<64> passing those zeros, LLVM specializes it
        // (AArch64: x5 and x6 left out, `out` in a register; a queue passing
        // its flags here instead cost the VM's lent batches of 16).
        let (flags, start, end) = if N == crate::BLOCK_LEN {
            (self.flags | crate::CHUNK_START | crate::CHUNK_END | crate::ROOT, 0, 0)
        } else {
            (self.flags, crate::CHUNK_START, crate::CHUNK_END | crate::ROOT)
        };
        platform.hash_many::<N>(&table, &self.key, 0, crate::IncrementCounter::No, flags, start, end, digests[..members.len()].as_flattened_mut());
        for (m, digest) in members.iter().zip(&digests) {
            unsafe { core::ptr::copy_nonoverlapping(digest.as_ptr(), m.out, crate::OUT_LEN) };
        }
    }
}

/// The SME2 thread, on CPUs with SME2: it hashes tasks only, on SME2 (the
/// SME unit hashes a 64 KiB piece in 14 us where a NEON worker takes 22,
/// Mac, probe/queue-timeline), under the process's turn so that a caller of
/// hash() elsewhere keeps the unit; without the turn, on NEON. It takes
/// streams' tasks while any wait, and parks as soon as none does; every
/// publish wakes it first.
fn sme2_main() {
    assert!(SME2_THREAD.set(std::thread::current()).is_ok(), "one SME2 thread");
    loop {
        if let Some(task) = sources::take_stream_task() {
            if task.members > 0 {
                // Short messages and small batches, gathered, run faster
                // on NEON than through an SME2 session per task (Mac:
                // 64-byte messages 14-15% faster, batches of 16 and 64
                // 10-15%); below 16 KiB the other members run the NEON
                // kernels on either platform.
                task.run(pool_platform());
            } else {
                let turn = crate::platform::Sme2Turn::take(Platform::detect(), true);
                task.run(turn.platform());
                drop(turn);
            }
            continue;
        }
        // Park as a worker does: the flag, one more look, the token.
        SME2_PARKED.store(true, Ordering::SeqCst);
        if sources::visible(true) {
            SME2_PARKED.store(false, Ordering::SeqCst);
            continue;
        }
        while SME2_PARKED.load(Ordering::SeqCst) {
            std::thread::park();
        }
    }
}

/// The SME2 thread, and whether it is parked: every publish wakes it first.
static SME2_THREAD: OnceLock<std::thread::Thread> = OnceLock::new();
static SME2_PARKED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn wake_sme2() {
    if SME2_PARKED.swap(false, Ordering::SeqCst) {
        if let Some(thread) = SME2_THREAD.get() {
            thread.unpark();
        }
    }
}

/// Run `task` on this thread, as a worker would.
pub(crate) fn run_task(task: Task) {
    task.run(pool_platform());
}

fn worker_main(me: usize) {
    let pool = pool();
    assert!(pool.threads[me].set(std::thread::current()).is_ok(), "each worker sets its thread once");
    pool.serve(me)
}

#[cfg(test)]
mod test {
    use super::*;

    /// Members of one length side by side, against the reference
    /// implementation: every platform this machine runs, the three modes,
    /// equal and mixed lengths, unaligned inputs; the slots beside the
    /// outputs untouched.
    #[test]
    fn equal_members_match_the_reference() {
        #[allow(unused_mut)]
        let mut platforms: Vec<_> = [Some(Platform::Portable)].into_iter().flatten().collect();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        platforms.extend([Platform::sse2(), Platform::sse41(), Platform::avx2()].into_iter().flatten());
        #[cfg(blake3_avx512_ffi)]
        platforms.extend(Platform::avx512());
        platforms.push(Platform::detect());
        let key = [0x5Au8; crate::KEY_LEN];
        let context = "equal queue members";
        let context_key = crate::hazmat::hash_derive_key_context(context);
        let modes = [(*crate::IV, 0), (crate::platform::words_from_le_bytes_32(&key), crate::KEYED_HASH), (crate::platform::words_from_le_bytes_32(&context_key), crate::DERIVE_KEY_MATERIAL)];
        let counts: &[usize] = if cfg!(miri) { &[2, 3] } else { &[2, 3, 4, 7, 8, 15, 16, 17, 63, MEMBERS] };
        let offsets: &[usize] = if cfg!(miri) { &[0, 1] } else { &[0, 1, 7, 63] };
        for len in [64, 128, 192, 256, 512, 1024] {
            for &count in counts {
                for mixed in [false, true] {
                    for &offset in offsets {
                        let inputs: Vec<Vec<u8>> = (0..count)
                            .map(|m| {
                                let length = if mixed && m % 2 == 1 { len + 1 } else { len };
                                (0..offset + length).map(|i| ((m * 1031 + i) % 251) as u8).collect()
                            })
                            .collect();
                        for (mode, (words, flags)) in modes.iter().enumerate() {
                            let expected: Vec<[u8; crate::OUT_LEN]> = inputs
                                .iter()
                                .map(|input| {
                                    let mut h = match mode {
                                        0 => reference_impl::Hasher::new(),
                                        1 => reference_impl::Hasher::new_keyed(&key),
                                        _ => reference_impl::Hasher::new_derive_key(context),
                                    };
                                    h.update(&input[offset..]);
                                    let mut digest = [0; crate::OUT_LEN];
                                    h.finalize(&mut digest);
                                    digest
                                })
                                .collect();
                            for &platform in &platforms {
                                let done = AtomicUsize::new(count);
                                let mut out = vec![[0xAA; crate::OUT_LEN]; count + 2];
                                let mut task = Task::members(words, *flags, None);
                                task.members = count;
                                task.left = &done;
                                // Every destination from one raw base (Stacked Borrows).
                                let base = out.as_mut_ptr();
                                for m in 0..count {
                                    let out = unsafe { base.add(m + 1) }.cast::<u8>();
                                    task.member[m] = Member { input: inputs[m][offset..].as_ptr(), len: inputs[m].len() - offset, out };
                                }
                                task.run(platform);
                                assert_eq!(&out[1..count + 1], &expected[..], "{platform:?}, {count} x {len}, mixed {mixed}, offset {offset}, mode {mode}");
                                assert_eq!(out[0], [0xAA; crate::OUT_LEN]);
                                assert_eq!(out[count + 1], [0xAA; crate::OUT_LEN]);
                                assert_eq!(done.load(Ordering::Acquire), 0);
                            }
                        }
                    }
                }
            }
        }
    }

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
                    .map(|p| crate::hash_all_at_once(&input[p.offset..][..p.len], crate::IV, (p.offset / CHUNK_LEN) as u64, 0, Platform::detect()).chaining_value())
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
                            let mut hasher = Hasher::from_core(crate::HasherCore::new_from(crate::IV, 0, platform));
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
