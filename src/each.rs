//! [`hash_each`]: a collection of messages of any lengths, each hashed,
//! with short messages' chunks side by side in the SIMD lanes.
//!
//! Chunk k of every message has chunk counter k, so the full chunks k of
//! different messages hash in one call of the many-inputs kernel, at counter
//! k for every lane. Messages of up to a chunk share lanes by their block
//! counts (`hash_by_blocks`); NEON with SHA-3 includes their last blocks in
//! the same pass. Messages of 16 chunks or more fill the lanes alone; a
//! longer message's last partial chunk hashes on its own. Each message's
//! chunk chaining values then merge into its root as BLAKE3's tree places them.

use crate::platform::Platform;
use crate::throughput::Block;
use crate::{CHUNK_END, CHUNK_LEN, CHUNK_START, CVBytes, CVWords, ChunkState, Hash, Hasher, IncrementCounter, Mode, OUT_LEN};

/// [`hash_each`], using other CPU cores where that is faster: the
/// same hashes, never slower. Collections of 512 KiB and more in all use
/// them, as [`hash_multithreaded`](crate::throughput::hash_multithreaded) does.
/// Requires `out.len() == items.len()`.
///
/// ```
/// # blake3_servil::initialize_multithreaded(None);
/// use blake3_servil::{Mode, throughput::{self, Block}};
/// let blocks: Vec<Vec<Block>> = (0..100).map(|i| vec![Block([i as u8; 64]); (1000 * i as usize).div_ceil(64)]).collect();
/// let items: Vec<(&[Block], usize)> = blocks.iter().enumerate().map(|(i, b)| (&b[..], 1000 * i)).collect();
/// let (mut a, mut b) = (vec![[0u8; 32]; 100], vec![[0u8; 32]; 100]);
/// throughput::hash_each_multithreaded(Mode::Plain, &items, &mut a);
/// throughput::hash_each(Mode::Plain, &items, &mut b);
/// assert_eq!(a, b);
/// ```
#[cfg(feature = "std")]
pub fn hash_each_multithreaded(mode: Mode, items: &[(&[Block], usize)], out: &mut [[u8; OUT_LEN]]) {
    crate::lanes::require();
    assert_eq!(items.len(), out.len(), "one digest per item");
    let mut room = core::mem::MaybeUninit::uninit();
    let (key, flags) = mode.key(&mut room);
    let alone = |item: &[u8]| item.len() >= crate::lanes::MIN_SPLIT_LEN;
    // The collection with each item that hashes alone set aside (an empty
    // stand-in, its digest written after).
    let small: Vec<&[u8]> = items.iter().map(|i| if alone(i.bytes()) { &[][..] } else { i.bytes() }).collect();
    if small.iter().map(|i| i.len()).sum::<usize>() < crate::lanes::MIN_SPLIT_LEN {
        hash_each_on(key, flags, &small, out, Platform::detect(), |item| crate::hash_serial(item, key, flags));
    } else {
        crate::lanes::hash_each(key, flags, &small, out);
    }
    for (item, out) in items.iter().map(Item::bytes).zip(out.iter_mut()) {
        if alone(item) {
            *out = *crate::lanes::hash_with_key(item, key, flags).as_bytes();
        }
    }
}

// Many messages arriving at once, such as a server's uploads, each with
// its own Hasher: one call takes the pieces that have arrived for all of
// them, which hash faster together than one Hasher::update each.

/// Feed each piece to its hasher: `(i, bytes)` does
/// `hashers[i].update(bytes)`, each hasher's pieces in their order in
/// `pieces`. Call it once with all the pieces your program has in hand
/// (one turn of its event loop): never slower than one
/// [`update`](Hasher::update) each: many messages arriving at once, such
/// as a server's uploads, each with its own [`Hasher`]. Requires each `i < hashers.len()`.
///
/// ```
/// # blake3_servil::initialize();
/// use blake3_servil::throughput::{self, Hasher};
/// let mut hashers = vec![Hasher::new(blake3_servil::Mode::Plain), Hasher::new(blake3_servil::Mode::Plain)];
/// throughput::update_each(&mut hashers, &[(0, b"ab"), (1, b"xyz"), (0, b"c")]);
/// let mut out = [[0u8; 32]; 2];
/// throughput::finalize_each(&hashers, &[0, 1], &mut out);
/// let hash = blake3_servil::convenience::hash;
/// assert_eq!((&out[0], &out[1]), (hash(b"abc").as_bytes(), hash(b"xyz").as_bytes()));
/// ```
pub fn update_each(hashers: &mut [Hasher], pieces: &[(usize, &[u8])]) {
    update_turn(hashers, pieces, false);
}

/// [`update_each`], using other CPU cores where
/// that is faster (pieces of 512 KiB and more): the same result, never
/// slower.
/// Requires each `i < hashers.len()`.
#[cfg(feature = "std")]
pub fn update_each_multithreaded(hashers: &mut [Hasher], pieces: &[(usize, &[u8])]) {
    crate::lanes::require();
    update_turn(hashers, pieces, true);
}

/// The hash of each message `which` names: `out[k]` is
/// `hashers[which[k]].finalize()`. Requires `out.len() == which.len()`
/// and each index `< hashers.len()`.
pub fn finalize_each(hashers: &[Hasher], which: &[usize], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(which.len(), out.len(), "one digest per message named");
    for (&i, out) in which.iter().zip(out) {
        *out = *hashers[i].finalize().as_bytes();
    }
}

/// A turn of pieces for many hashers: each piece gathered into its
/// hasher's stage, and every stage that fills a whole group held, its
/// hasher's later pieces waiting, until the turn's held groups hash
/// together: their chunks back to back, a group per call with nothing in
/// between (the SME unit stays in its fast state), then each level of
/// their parents in one call. Then the waiting pieces, the same way, until
/// none is left.
fn update_turn(hashers: &mut [Hasher], pieces: &[(usize, &[u8])], pooled: bool) {
    let mut turn: Vec<(usize, &[u8])> = pieces.to_vec();
    let mut waiting: Vec<(usize, &[u8])> = Vec::new();
    let mut held: Vec<usize> = Vec::new();
    while !turn.is_empty() {
        for &(i, bytes) in &turn {
            // A hasher holding a group takes no more until the turn's
            // groups are hashed: its pieces wait, in order.
            if hashers[i].held_group().is_some() {
                waiting.push((i, bytes));
                continue;
            }
            let rest = hashers[i].gather_part(bytes, pooled, true);
            if hashers[i].held_group().is_some() {
                held.push(i);
                if !rest.is_empty() {
                    waiting.push((i, rest));
                }
            }
        }
        hash_held(hashers, &held);
        // The loop's variant: every held group was hashed, so each waiting
        // piece gathers on the next pass, and the bytes in play shrink on
        // every pass (a change that kept them would spin here forever;
        // the assert makes it a failure a test sees).
        assert!(held.iter().all(|&i| hashers[i].held_group().is_none()), "a held group outlived its turn");
        let (before, after): (usize, usize) = (turn.iter().map(|p| p.1.len()).sum(), waiting.iter().map(|p| p.1.len()).sum());
        assert!(
            after < before || (after == before && waiting.len() < turn.len()),
            "a turn's pass gathered nothing ({} pieces of {before} bytes before, {} of {after} after)",
            turn.len(),
            waiting.len()
        );
        held.clear();
        core::mem::swap(&mut turn, &mut waiting);
        waiting.clear();
    }
}

/// Hash the whole groups `held` names, one in each of those hashers'
/// stages, together where they share a key and flags, and pass each
/// group's subtree to its hasher's core.
#[cfg(feature = "std")]
fn hash_held(hashers: &mut [Hasher], held: &[usize]) {
    use crate::{BLOCK_LEN, PARENT};
    let Some(&first) = held.first() else { return };
    let (key, flags) = (hashers[first].core.key, hashers[first].core.chunk_state.flags);
    let (alike, others): (Vec<usize>, Vec<usize>) = held.iter().partition(|&&i| hashers[i].core.key == key && hashers[i].core.chunk_state.flags == flags);
    let turn = crate::platform::Sme2Turn::take(hashers[first].core.chunk_state.platform, true);
    let platform = turn.platform();
    let n = alike.len();
    let per = crate::STAGE_LEN / CHUNK_LEN;
    let chunks = n * per;
    let mut level = vec![0u8; chunks * OUT_LEN];
    let refs: Vec<&[u8; CHUNK_LEN]> =
        alike.iter().flat_map(|&i| hashers[i].held_group().expect("a held group").1.chunks_exact(CHUNK_LEN)).map(|c| c.try_into().unwrap()).collect();
    let counters: Vec<u64> = alike.iter().map(|&i| hashers[i].held_group().unwrap().0).collect();
    // On SME2, every group in one kernel call, each at its own counter;
    // elsewhere a call per group.
    #[cfg(blake3_sme2)]
    let on_sme2 = matches!(platform, Platform::SME2);
    #[cfg(not(blake3_sme2))]
    let on_sme2 = false;
    if on_sme2 {
        // Sound: the turn's platform is SME2 only where detect() found it.
        #[cfg(blake3_sme2)]
        unsafe {
            crate::sme2::hash_groups_at(&refs, &counters, &key, flags, CHUNK_START, CHUNK_END, &mut level)
        };
    } else {
        for (k, &counter) in counters.iter().enumerate() {
            platform.hash_many(
                &refs[k * per..(k + 1) * per],
                &key,
                counter,
                IncrementCounter::Yes,
                flags,
                CHUNK_START,
                CHUNK_END,
                &mut level[k * per * OUT_LEN..(k + 1) * per * OUT_LEN],
            );
        }
    }
    // Three levels of parents to each group's two halves, then a fourth to
    // its own value.
    let mut values = chunks;
    let mut next = vec![0u8; chunks / 2 * OUT_LEN];
    let mut halves = vec![0u8; 2 * n * OUT_LEN];
    for depth in 0..4 {
        if depth == 3 {
            halves.copy_from_slice(&level[..2 * n * OUT_LEN]);
        }
        let blocks: Vec<&[u8; BLOCK_LEN]> = level[..values * OUT_LEN].chunks_exact(BLOCK_LEN).map(|b| b.try_into().unwrap()).collect();
        platform.hash_many(&blocks, &key, 0, IncrementCounter::No, flags | PARENT, 0, 0, &mut next[..values / 2 * OUT_LEN]);
        values /= 2;
        level[..values * OUT_LEN].copy_from_slice(&next[..values * OUT_LEN]);
    }
    drop(turn);
    for (k, &i) in alike.iter().enumerate() {
        let (counter, _) = hashers[i].held_group().unwrap();
        let mut result = [0u8; BLOCK_LEN];
        if counter == 0 {
            result.copy_from_slice(&halves[k * BLOCK_LEN..(k + 1) * BLOCK_LEN]);
        } else {
            result[..OUT_LEN].copy_from_slice(&level[k * OUT_LEN..(k + 1) * OUT_LEN]);
        }
        hashers[i].take_held(&result);
    }
    hash_held(hashers, &others);
}

#[cfg(not(feature = "std"))]
fn hash_held(_: &mut [Hasher], held: &[usize]) {
    assert!(held.is_empty(), "nothing held without std");
}

/// Messages of this many chunks or more fill the lanes alone.
const ALONE_CHUNKS: usize = 16;

/// The hash of each message, of any lengths, on the calling thread:
/// `out[i]` is [`hash`](crate::throughput::hash)`(mode, items[i].0,
/// items[i].1)`, each item a message of `items[i].1` bytes starting its
/// blocks (the bytes past its end ignored). Short messages hash together,
/// so a collection of small items (a directory's files, a repository's
/// objects) takes less time than one call each. Requires `out.len() ==
/// items.len()`.
///
/// ```
/// # blake3_servil::initialize();
/// use blake3_servil::{Mode, convenience, throughput::{self, Block}};
/// let blocks: Vec<Vec<Block>> = (0..100).map(|i| vec![Block([i as u8; 64]); (100 * i as usize).div_ceil(64)]).collect();
/// let items: Vec<(&[Block], usize)> = blocks.iter().enumerate().map(|(i, b)| (&b[..], 100 * i)).collect();
/// let mut out = vec![[0u8; 32]; items.len()];
/// throughput::hash_each(Mode::Plain, &items, &mut out);
/// assert_eq!(&out[42], convenience::hash(&[42u8; 4200]).as_bytes());
/// ```
pub fn hash_each(mode: Mode, items: &[(&[Block], usize)], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(items.len(), out.len(), "one digest per item");
    let mut room = core::mem::MaybeUninit::uninit();
    let (key, flags) = mode.key(&mut room);
    hash_each_on(key, flags, items, out, Platform::detect(), |item| crate::hash_serial(item, key, flags));
}

/// The most items [`hash_by_blocks`] sorts at once: a stream task's members.
const BY_BLOCKS_AT_ONCE: usize = crate::lanes::MEMBERS;

/// The digest of each of `items`, every one of up to a chunk, of any
/// lengths, by their blocks: those with as many whole blocks before their
/// last (k) side by side on the platform's hash_many, up to sixteen at a
/// time. On NEON with SHA-3 their last blocks run in the same lanes,
/// each with its own length (the l kernels). Other platforms hash the
/// prefix with CHUNK_START on the first block and no end, leaving each its
/// chaining value; then each one's last block (1 to 64 bytes, zero-padded;
/// an empty message's is empty) compressed on its own with CHUNK_END and
/// ROOT, CHUNK_START too where k is 0. Up to BY_BLOCKS_AT_ONCE items are
/// sorted at once, on the stack. Requires AVX2 or AVX-512 on x86-64; NEON on AArch64.
fn hash_by_blocks(items: &[&[u8]], key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]], platform: Platform) {
    assert_eq!(items.len(), out.len(), "one digest per item");
    assert!(items.iter().all(|i| i.len() <= CHUNK_LEN), "items of up to a chunk");
    for (items, out) in items.chunks(BY_BLOCKS_AT_ONCE).zip(out.chunks_mut(BY_BLOCKS_AT_ONCE)) {
        by_blocks_at_once(items, key, flags, out, platform);
    }
}

fn by_blocks_at_once(items: &[&[u8]], key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]], platform: Platform) {
    const GROUP: usize = 16;
    let whole = |item: &[u8]| item.len().saturating_sub(1) / crate::BLOCK_LEN;
    let mut order: arrayvec::ArrayVec<usize, BY_BLOCKS_AT_ONCE> = (0..items.len()).collect();
    order.sort_unstable_by_key(|&i| whole(items[i]));
    // With SHA-3 the last block stays in the lanes too; other AArch64
    // CPUs keep the prefix kernels followed by scalar last blocks.
    #[cfg(blake3_neon)]
    let last_lanes = crate::neon_hybrid::sha3_detected();
    #[cfg(not(blake3_neon))]
    let last_lanes = false;
    let mut cvs = [*key; BY_BLOCKS_AT_ONCE];
    // chunk_by always advances and returns nonempty groups, even if a
    // changed predicate groups the wrong items; no mutable loop bound.
    for same_blocks in order.chunk_by(|&a, &b| whole(items[a]) == whole(items[b])) {
        let k = whole(items[same_blocks[0]]);
        #[cfg(blake3_neon)]
        if last_lanes {
            for group in same_blocks.chunks(GROUP) {
                let inputs: arrayvec::ArrayVec<*const u8, GROUP> = group.iter().map(|&i| items[i].as_ptr()).collect();
                let mut last_blocks = [[0u8; crate::BLOCK_LEN]; GROUP];
                let mut lengths = [0u32; GROUP];
                for ((&i, block), len) in group.iter().zip(&mut last_blocks).zip(&mut lengths) {
                    let tail = &items[i][k * crate::BLOCK_LEN..];
                    block[..tail.len()].copy_from_slice(tail);
                    *len = tail.len() as u32;
                }
                let lasts: arrayvec::ArrayVec<*const u8, GROUP> = last_blocks[..group.len()].iter().map(|b| b.as_ptr()).collect();
                let mut digests = [0u8; GROUP * OUT_LEN];
                crate::counted(crate::Counted::ByBlocksCall);
                // Sound: SHA-3 is present; these items share k whole blocks
                // before their separately padded last block, 0 to 64 bytes.
                unsafe {
                    crate::neon_hybrid::hash_last_lanes(
                        &inputs,
                        &lasts,
                        &lengths[..group.len()],
                        k + 1,
                        key,
                        flags,
                        CHUNK_START,
                        CHUNK_END | crate::ROOT,
                        &mut digests[..group.len() * OUT_LEN],
                    );
                }
                for (&i, digest) in group.iter().zip(digests.chunks_exact(OUT_LEN)) {
                    out[i].copy_from_slice(digest);
                }
            }
            continue;
        }
        if k > 0 {
            for group in same_blocks.chunks(GROUP) {
                let inputs: arrayvec::ArrayVec<*const u8, GROUP> = group.iter().map(|&i| items[i].as_ptr()).collect();
                let mut lanes = [0u8; GROUP * OUT_LEN];
                crate::counted(crate::Counted::ByBlocksCall);
                // Sound: each item holds more than k whole blocks; the
                // caller selected this platform's block-group path.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    crate::pair::hash_many_blocks(platform, &inputs, k, key, 0, false, flags, CHUNK_START, 0, &mut lanes[..group.len() * OUT_LEN])
                };
                #[cfg(blake3_neon)]
                unsafe {
                    crate::neon_hybrid::hash_messages_raw(
                        inputs.as_ptr(),
                        group.len(),
                        k,
                        key,
                        0,
                        flags,
                        CHUNK_START,
                        0,
                        crate::BLOCK_LEN,
                        &mut lanes[..group.len() * OUT_LEN],
                    )
                };
                for (n, &i) in group.iter().enumerate() {
                    cvs[i] = crate::platform::words_from_le_bytes_32(lanes[n * OUT_LEN..][..OUT_LEN].try_into().unwrap());
                }
            }
        }
    }
    if last_lanes {
        return;
    }
    for &i in &order {
        let item = items[i];
        let k = whole(item);
        let mut cv = cvs[i];
        let mut block = [0u8; crate::BLOCK_LEN];
        let last = item.len() - k * crate::BLOCK_LEN;
        block[..last].copy_from_slice(&item[k * crate::BLOCK_LEN..]);
        let start = if k == 0 { CHUNK_START } else { 0 };
        platform.compress_in_place(&mut cv, &block, last as u8, 0, flags | start | CHUNK_END | crate::ROOT);
        out[i] = crate::hasher::le_bytes(&cv);
    }
}

/// An item of a collection, as hash_each_on reads it: its bytes. The
/// public calls' items are messages in blocks; the pool's ranges, bytes.
pub(crate) trait Item {
    fn bytes(&self) -> &[u8];
}

impl Item for &[u8] {
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        self
    }
}

impl Item for (&[Block], usize) {
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        &Block::bytes(self.0)[..self.1]
    }
}

/// [`hash_each`] in the mode of `key` and `flags`, the lanes on
/// `platform`, each message that fills the lanes alone (or is a chunk or
/// less) through `alone`.
pub(crate) fn hash_each_on<I: Item>(key: &CVWords, flags: u8, items: &[I], out: &mut [[u8; OUT_LEN]], platform: Platform, alone: impl Fn(&[u8]) -> Hash) {
    let key = *key;
    // The messages whose full chunks share the lanes: each one's index, and
    // where its chunk values start in `cvs`.
    let mut shared: Vec<(usize, usize)> = Vec::new();
    let mut slots = 0;
    let mut most_full = 0;
    // Items of up to a chunk, two or more, by their blocks (hash_by_blocks).
    #[cfg(target_arch = "x86_64")]
    let platform_takes = crate::pair::takes_platform(platform);
    // Every supported AArch64 platform has NEON; the kernel family
    // selects SHA-3 or plain instructions itself.
    #[cfg(target_arch = "aarch64")]
    let platform_takes = true;
    let by_blocks = platform_takes && items.iter().filter(|i| i.bytes().len() <= CHUNK_LEN).nth(1).is_some();
    if by_blocks {
        // BY_BLOCKS_AT_ONCE of them at a time, their indices on the stack:
        // a collection of small items allocates nothing.
        let mut small = (0..items.len()).filter(|&i| items[i].bytes().len() <= CHUNK_LEN).peekable();
        while small.peek().is_some() {
            let part: arrayvec::ArrayVec<usize, BY_BLOCKS_AT_ONCE> = small.by_ref().take(BY_BLOCKS_AT_ONCE).collect();
            let bytes: arrayvec::ArrayVec<&[u8], BY_BLOCKS_AT_ONCE> = part.iter().map(|&i| items[i].bytes()).collect();
            let mut digests = [[0u8; OUT_LEN]; BY_BLOCKS_AT_ONCE];
            hash_by_blocks(&bytes, &key, flags, &mut digests[..part.len()], platform);
            for (&i, digest) in part.iter().zip(&digests) {
                out[i] = *digest;
            }
        }
    }
    for (i, item) in items.iter().map(Item::bytes).enumerate() {
        let chunks = item.len().div_ceil(CHUNK_LEN);
        if chunks <= 1 && by_blocks {
            continue;
        }
        if chunks <= 1 || chunks >= ALONE_CHUNKS {
            out[i] = *alone(item).as_bytes();
        } else {
            most_full = most_full.max(item.len() / CHUNK_LEN);
            shared.push((i, slots));
            slots += chunks;
        }
    }
    let mut cvs = vec![[0u8; OUT_LEN]; slots];
    let mut refs: Vec<&[u8; CHUNK_LEN]> = Vec::with_capacity(shared.len());
    let mut owners: Vec<usize> = Vec::with_capacity(shared.len());
    let mut lane_out = vec![0u8; shared.len() * OUT_LEN];
    for k in 0..most_full {
        crate::counted(crate::Counted::EachRound);
        refs.clear();
        owners.clear();
        for &(i, at) in &shared {
            if let Some(chunk) = items[i].bytes().get(k * CHUNK_LEN..(k + 1) * CHUNK_LEN) {
                refs.push(chunk.try_into().unwrap());
                owners.push(at + k);
            }
        }
        let lanes = &mut lane_out[..refs.len() * OUT_LEN];
        platform.hash_many(&refs, &key, k as u64, IncrementCounter::No, flags, CHUNK_START, CHUNK_END, lanes);
        for (j, &slot) in owners.iter().enumerate() {
            cvs[slot].copy_from_slice(&lanes[j * OUT_LEN..(j + 1) * OUT_LEN]);
        }
    }
    for &(i, at) in &shared {
        let item = items[i].bytes();
        let chunks = item.len().div_ceil(CHUNK_LEN);
        let full = item.len() / CHUNK_LEN;
        if full < chunks {
            cvs[at + full] = ChunkState::new(&key, full as u64, flags, platform).update(&item[full * CHUNK_LEN..]).output().chaining_value();
        }
        out[i] = *root(&cvs[at..at + chunks], &key, flags, platform).as_bytes();
    }
}

/// The root of a message of two or more chunks, from its chunk values: the
/// left subtree takes the largest power of two of chunks below the count.
fn root(cvs: &[CVBytes], key: &CVWords, flags: u8, platform: Platform) -> Hash {
    let split = left_len(cvs.len());
    let left = subtree(&cvs[..split], key, flags, platform);
    let right = subtree(&cvs[split..], key, flags, platform);
    crate::hasher::parent_output(&left, &right, key, flags, platform).root_hash()
}

/// A non-root subtree's chaining value.
fn subtree(cvs: &[CVBytes], key: &CVWords, flags: u8, platform: Platform) -> CVBytes {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let split = left_len(cvs.len());
    let left = subtree(&cvs[..split], key, flags, platform);
    let right = subtree(&cvs[split..], key, flags, platform);
    crate::hasher::parent_output(&left, &right, key, flags, platform).chaining_value()
}

/// The largest power of two below `n` (n of 2 or more).
fn left_len(n: usize) -> usize {
    debug_assert!(n >= 2);
    1 << (usize::BITS - 1 - (n - 1).leading_zeros())
}

#[cfg(test)]
mod test {
    use super::*;

    // Expected routes are independent of the implementation's predicate:
    // all supported ARM platforms, AVX2 and AVX-512, but not SSE4.1.
    fn block_groups_expected(platform: Platform) -> bool {
        #[cfg(blake3_sse41_ffi)]
        return !matches!(platform, Platform::SSE41);
        #[cfg(not(blake3_sse41_ffi))]
        {
            let _ = platform;
            true
        }
    }

    /// hash_each_multithreaded takes the pool from MIN_SPLIT_LEN bytes of
    /// items on, exactly, and the calling thread below them. Both give the
    /// same digests, so only where the call ran tells (BLAKE3 #219).
    #[test]
    fn test_hash_each_multithreaded_splits_at_min_split_len_exactly() {
        let min = crate::lanes::MIN_SPLIT_LEN;
        let input = vec![7u8; min];
        for total in [min - 1, min] {
            // Items under the split each, so none hashes alone.
            let items: Vec<&[u8]> = input[..total].chunks(64 * 1024 - 3).collect();
            let (mut a, mut b) = (vec![[0u8; OUT_LEN]; items.len()], vec![[0u8; OUT_LEN]; items.len()]);
            crate::RAN.with(|r| r.set(0));
            crate::throughput::hash_each_multithreaded(Mode::Plain, &crate::test::items(&crate::test::messages(&items), &items), &mut a);
            if total < min {
                assert_eq!(crate::RAN.with(|r| r.get()) & crate::Ran::Pool as u32, 0, "{total} bytes of items stay on the calling thread");
            } else {
                crate::lanes::assert_past_the_split("MIN_SPLIT_LEN bytes of items");
            }
            crate::throughput::hash_each(Mode::Plain, &crate::test::items(&crate::test::messages(&items), &items), &mut b);
            assert_eq!(a, b);
        }
    }

    /// A collection of small items of mixed lengths (empty to a chunk, more
    /// than BY_BLOCKS_AT_ONCE of them, a long one among them): those of up
    /// to a chunk by their blocks where the platform takes them (a
    /// hash_many call counted), every digest hash()'s, held to a golden
    /// fold.
    #[test]
    fn test_hash_each_small_items_by_blocks() {
        let mut golden = crate::test::Golden::new("hash_each_small_items");
        let mut x = 0x9e37_79b9u64;
        for count in [2usize, 7, 63, 64, 65, 200] {
            let mut lens: Vec<usize> = (0..count)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % (CHUNK_LEN as u64 + 1)) as usize
                })
                .collect();
            lens[count / 2] = 5 * CHUNK_LEN + 77;
            let data: Vec<Vec<u8>> = lens.iter().enumerate().map(|(m, &n)| (0..n).map(|i| ((m * 37 + i) % 249) as u8).collect()).collect();
            let items: Vec<&[u8]> = data.iter().map(|d| &d[..]).collect();
            let mut out = vec![[0u8; OUT_LEN]; count];
            for platform in crate::test::every_platform() {
                crate::COUNTED.with(|c| c[crate::Counted::ByBlocksCall as usize].set(0));
                hash_each_on(crate::IV, 0, &items, &mut out, platform, |item| crate::hash_serial(item, crate::IV, 0));
                let calls = crate::COUNTED.with(|c| c[crate::Counted::ByBlocksCall as usize].get());
                // A call where two or more items are of up to a chunk and one of
                // them has a whole block before its last.
                let small: Vec<usize> = lens.iter().copied().filter(|&n| n <= CHUNK_LEN).collect();
                #[cfg(blake3_neon)]
                let last_lanes = crate::neon_hybrid::sha3_detected();
                #[cfg(not(blake3_neon))]
                let last_lanes = false;
                let calls_expected: u64 = if block_groups_expected(platform) && small.len() >= 2 {
                    small
                        .chunks(BY_BLOCKS_AT_ONCE)
                        .map(|part| {
                            let mut buckets = [0u64; 16];
                            for &len in part {
                                buckets[len.saturating_sub(1) / crate::BLOCK_LEN] += 1;
                            }
                            buckets.into_iter().enumerate().filter(|&(k, _)| last_lanes || k > 0).map(|(_, n)| n.div_ceil(16)).sum::<u64>()
                        })
                        .sum()
                } else {
                    0
                };
                assert_eq!(calls, calls_expected, "{count} items on {platform:?}: exact block-group calls");
                let expected: Vec<[u8; OUT_LEN]> = items.iter().map(|i| *crate::convenience::hash(i).as_bytes()).collect();
                assert_eq!(out, expected, "{count} items");
            }
            golden.add(out.as_flattened(), || items.iter().flat_map(|i| crate::test::official::hash(Mode::Plain, i)).collect());
        }
        golden.check(0x8fa73f9dac40c559);
    }

    /// hash_each_on sends a message to `alone` exactly when it has
    /// ALONE_CHUNKS or more, or one chunk or fewer where the platform takes
    /// no items by their blocks (elsewhere two or more such go
    /// by their blocks), and gives the others' lanes as many rounds as the
    /// longest of them has whole chunks. Every other choice gives the same
    /// digests, so only the counts tell (BLAKE3 #219).
    #[test]
    fn test_hash_each_on_counts_its_alone_messages_and_rounds() {
        let input = vec![5u8; (ALONE_CHUNKS + 1) * CHUNK_LEN];
        let lens = [
            0,
            1,
            CHUNK_LEN,
            CHUNK_LEN + 1,
            2 * CHUNK_LEN,
            (ALONE_CHUNKS - 1) * CHUNK_LEN,
            (ALONE_CHUNKS - 1) * CHUNK_LEN + 1,
            ALONE_CHUNKS * CHUNK_LEN,
            ALONE_CHUNKS * CHUNK_LEN + 1,
        ];
        let items: Vec<&[u8]> = lens.iter().map(|&n| &input[..n]).collect();
        let alone = core::cell::RefCell::new(Vec::new());
        let mut out = vec![[0u8; OUT_LEN]; items.len()];
        crate::COUNTED.with(|c| c[crate::Counted::EachRound as usize].set(0));
        hash_each_on(crate::IV, 0, &items, &mut out, Platform::detect(), |item| {
            alone.borrow_mut().push(item.len());
            crate::hash_serial(item, crate::IV, 0)
        });
        let rounds = crate::COUNTED.with(|c| c[crate::Counted::EachRound as usize].get());
        let long = [(ALONE_CHUNKS - 1) * CHUNK_LEN + 1, ALONE_CHUNKS * CHUNK_LEN, ALONE_CHUNKS * CHUNK_LEN + 1];
        let expected: Vec<usize> = if block_groups_expected(Platform::detect()) { long.to_vec() } else { [0, 1, CHUNK_LEN].into_iter().chain(long).collect() };
        assert_eq!(alone.into_inner(), expected);
        // The shared: 1, 2 and ALONE_CHUNKS - 1 whole chunks.
        assert_eq!(rounds, (ALONE_CHUNKS - 1) as u64);
        for (item, out) in items.iter().zip(&out) {
            assert_eq!(out, crate::convenience::hash(item).as_bytes(), "{} bytes", item.len());
        }
    }

    /// The pool's ranges of items (lanes::Work::Each), just past where the
    /// pool takes them, against the calling thread's.
    #[test]
    fn test_hash_each_multithreaded_past_the_split() {
        let mut input = vec![0u8; crate::lanes::MIN_SPLIT_LEN + 8 * 1024];
        crate::test::paint_test_input(&mut input);
        let items: Vec<&[u8]> = input.chunks(crate::lanes::MIN_SPLIT_LEN / 40 + 7).collect();
        let (mut a, mut b) = (vec![[0u8; OUT_LEN]; items.len()], vec![[0u8; OUT_LEN]; items.len()]);
        crate::RAN.with(|r| r.set(0));
        crate::throughput::hash_each_multithreaded(Mode::Plain, &crate::test::items(&crate::test::messages(&items), &items), &mut a);
        crate::lanes::assert_past_the_split("the items");
        crate::throughput::hash_each(Mode::Plain, &crate::test::items(&crate::test::messages(&items), &items), &mut b);
        assert_eq!(a, b);
    }

    /// A collection past the split (items around chunk, lane, and the
    /// split's own length, a few of 512 KiB and more), every mode: the
    /// multithreaded form gives hash_each_with's digests.
    /// Keep: it detects, by hanging, mutants the clock then stops
    /// (tools/mutants/timeouts-x86_64.txt).
    #[test]
    fn test_hash_each_multithreaded() {
        let mut input = vec![0u8; 3 << 20];
        crate::test::paint_test_input(&mut input);
        let lens = [0, 1, 1024, 1025, 3000, 15 * 1024, 16 * 1024, 100_000, 511 * 1024, 512 * 1024, 700_000];
        let items: Vec<&[u8]> = (0..400)
            .map(|k| {
                let n = lens[k * 7 % lens.len()].min(input.len() - k);
                &input[k..k + n]
            })
            .collect();
        let key = [9u8; 32];
        for mode in [Mode::Plain, Mode::Keyed(&key), Mode::DeriveKey("each mt")] {
            let (mut a, mut b) = (vec![[0u8; OUT_LEN]; items.len()], vec![[0u8; OUT_LEN]; items.len()]);
            crate::throughput::hash_each_multithreaded(mode, &crate::test::items(&crate::test::messages(&items), &items), &mut a);
            crate::throughput::hash_each(mode, &crate::test::items(&crate::test::messages(&items), &items), &mut b);
            assert_eq!(a, b);
            // Below the split, on the calling thread: the items of 16 KiB
            // or less, which sum to less than the split's length.
            let small: Vec<usize> = (0..items.len()).filter(|&k| items[k].len() <= 16 * 1024).take(30).collect();
            assert!(small.iter().map(|&k| items[k].len()).sum::<usize>() < crate::lanes::MIN_SPLIT_LEN);
            let small_items: Vec<&[u8]> = small.iter().map(|&k| items[k]).collect();
            let mut c = vec![[0u8; OUT_LEN]; small.len()];
            crate::throughput::hash_each_multithreaded(mode, &crate::test::items(&crate::test::messages(&small_items), &small_items), &mut c);
            assert!(small.iter().zip(&c).all(|(&k, d)| *d == b[k]));
        }
    }

    /// Many hashers fed in turns of interleaved pieces of many lengths:
    /// update_each, its multithreaded twin, and finalize_each give every
    /// message the hash one Hasher per message gives.
    /// Turns in which several hashers hold a group each, under two keys
    /// whose flags agree, beyond their first group and at it (nonzero
    /// counters, and zero): the groups of one key hash together, each
    /// lands in its own hasher.
    #[test]
    fn test_update_each_holds_groups_together() {
        for platform in crate::test::every_platform() {
            holds_groups_together(platform);
        }
    }

    /// Hashers of subtrees at input offsets in update_each's turns,
    /// beside hashers of whole messages: they hold no groups, and each
    /// gives the chaining value one update of its subtree gives.
    #[test]
    fn test_update_each_at_offsets() {
        use crate::test::SubtreeHasher;
        let mut input = vec![0u8; 1 << 20];
        crate::test::paint_test_input(&mut input);
        // (offset, length): subtrees within what their offsets allow.
        let subtrees = [(64 * CHUNK_LEN, 64 * CHUNK_LEN), (16 * CHUNK_LEN, 16 * CHUNK_LEN), (CHUNK_LEN, CHUNK_LEN), (128 * CHUNK_LEN, 100_000)];
        let whole = [70_000usize, 300_000];
        let mut hashers: Vec<Hasher> = subtrees
            .iter()
            .map(|&(offset, _)| {
                let mut h = Hasher::new(crate::Mode::Plain);
                h.set_input_offset(offset as u64);
                h
            })
            .collect();
        hashers.extend(whole.iter().map(|_| Hasher::new(crate::Mode::Plain)));
        let lens: Vec<usize> = subtrees.iter().map(|&(_, len)| len).chain(whole).collect();
        let mut done = vec![0usize; lens.len()];
        let pieces = [1448usize, 31, 16384, 4096, 1];
        let mut k = 0;
        while done.iter().zip(&lens).any(|(d, l)| d < l) {
            let mut turn: Vec<(usize, &[u8])> = Vec::new();
            for _ in 0..9 {
                let i = k % lens.len();
                let n = pieces[k % pieces.len()].min(lens[i] - done[i]);
                turn.push((i, &input[done[i]..done[i] + n]));
                done[i] += n;
                k += 1;
            }
            crate::throughput::update_each(&mut hashers, &turn);
        }
        for (i, &(offset, len)) in subtrees.iter().enumerate() {
            let mut one = Hasher::new(crate::Mode::Plain);
            one.set_input_offset(offset as u64).update(&input[..len]);
            assert_eq!(hashers[i].finalize_non_root(), one.finalize_non_root(), "subtree at {offset}, {len} bytes");
        }
        for (j, &len) in whole.iter().enumerate() {
            assert_eq!(hashers[subtrees.len() + j].finalize(), crate::convenience::hash(&input[..len]), "whole message of {len} bytes");
        }
    }

    fn holds_groups_together(platform: crate::platform::Platform) {
        let mut input = vec![0u8; 3 * crate::STAGE_LEN];
        crate::test::paint_test_input(&mut input);
        let keys = [[1u8; 32], [2u8; 32]];
        let on = |mut h: Hasher| {
            h.core.chunk_state.platform = platform;
            h
        };
        let mut hashers: Vec<Hasher> = (0..6).map(|i| on(Hasher::new(crate::Mode::Keyed(&keys[i % 2])))).collect();
        // Each hasher a group in, its message at a different offset into the
        // input; then one turn of two pieces each, which stage its second
        // group from its start and fill it (held), and 7 bytes more.
        let start = |i: usize| i * 1000;
        let group = crate::STAGE_LEN;
        for (i, h) in hashers.iter_mut().enumerate() {
            h.update(&input[start(i)..start(i) + group]);
        }
        let turn: Vec<(usize, &[u8])> = (0..6)
            .flat_map(|i| [(i, &input[start(i) + group..start(i) + group + 100]), (i, &input[start(i) + group + 100..start(i) + 2 * group + 7])])
            .collect();
        crate::throughput::update_each(&mut hashers, &turn);
        for (i, h) in hashers.iter().enumerate() {
            let message = &input[start(i)..start(i) + 2 * crate::STAGE_LEN + 7];
            assert_eq!(h.finalize(), crate::convenience::hash_with(crate::Mode::Keyed(&keys[i % 2]), message), "hasher {i}, {platform:?}");
        }
        // And each hasher's first group (chunk counter zero, which hands
        // its core the group's two halves), six at once: a first piece
        // longer than a chunk, so it is staged from offset zero.
        let mut fresh: Vec<Hasher> = (0..6).map(|i| on(Hasher::new(crate::Mode::Keyed(&keys[i % 2])))).collect();
        let turn: Vec<(usize, &[u8])> =
            (0..6).flat_map(|i| [(i, &input[start(i)..start(i) + 1100]), (i, &input[start(i) + 1100..start(i) + group + 7])]).collect();
        crate::throughput::update_each(&mut fresh, &turn);
        for (i, h) in fresh.iter().enumerate() {
            assert_eq!(
                h.finalize(),
                crate::convenience::hash_with(crate::Mode::Keyed(&keys[i % 2]), &input[start(i)..start(i) + group + 7]),
                "fresh hasher {i}, {platform:?}"
            );
        }
    }

    #[test]
    fn test_update_each() {
        let mut input = vec![0u8; 1 << 20];
        crate::test::paint_test_input(&mut input);
        let lens = [0, 1, 63, 64, 1023, 1024, 1025, 2048, 4470, 8191, 16384, 16385, 65536, 100_000, 300_000];
        let piece_lens = [1, 1448, 4096, 1448, 16384, 31, 65536];
        for multithreaded in [false, true] {
            let key = [5u8; 32];
            // Every third hasher keyed: a turn's held groups in two kinds.
            let mut hashers: Vec<Hasher> =
                (0..lens.len()).map(|i| if i % 3 == 1 { Hasher::new(crate::Mode::Keyed(&key)) } else { Hasher::new(crate::Mode::Plain) }).collect();
            let mut done = vec![0usize; lens.len()];
            let mut k = 0;
            while done.iter().zip(&lens).any(|(d, l)| d < l) {
                let mut pieces: Vec<(usize, &[u8])> = Vec::new();
                for _ in 0..23 {
                    let i = k % lens.len();
                    let n = piece_lens[k % piece_lens.len()].min(lens[i] - done[i]);
                    pieces.push((i, &input[done[i]..done[i] + n]));
                    done[i] += n;
                    k += 1;
                }
                if multithreaded {
                    crate::throughput::update_each_multithreaded(&mut hashers, &pieces);
                } else {
                    crate::throughput::update_each(&mut hashers, &pieces);
                }
            }
            let which: Vec<usize> = (0..lens.len()).rev().collect();
            let mut out = vec![[0u8; OUT_LEN]; lens.len()];
            crate::throughput::finalize_each(&hashers, &which, &mut out);
            for (k, &i) in which.iter().enumerate() {
                let want = if i % 3 == 1 {
                    crate::convenience::hash_with(crate::Mode::Keyed(&key), &input[..lens[i]])
                } else {
                    crate::convenience::hash(&input[..lens[i]])
                };
                assert_eq!(&out[k], want.as_bytes(), "len {} multithreaded {multithreaded}", lens[i]);
            }
        }
    }

    /// Every length around chunk and lane boundaries, in every mode, mixed in
    /// one call, against `hash_with` on each.
    /// Keep: it detects, by hanging, mutants the clock then stops
    /// (tools/mutants/timeouts-x86_64.txt).
    #[test]
    fn test_hash_each_matches_hash_with() {
        let mut input = vec![0u8; 40 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        let mut lens = vec![
            0,
            1,
            63,
            64,
            65,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
            3000,
            4096,
            5000,
            15 * CHUNK_LEN,
            15 * CHUNK_LEN + 1,
            16 * CHUNK_LEN - 1,
            16 * CHUNK_LEN,
            16 * CHUNK_LEN + 1,
            40 * CHUNK_LEN,
        ];
        for n in 2..16 {
            lens.push(n * CHUNK_LEN);
            lens.push(n * CHUNK_LEN - 7);
        }
        let items: Vec<&[u8]> = lens.iter().enumerate().map(|(j, &l)| &input[j % 3..j % 3 + l.min(input.len() - 2)]).collect();
        let key = [9u8; 32];
        for mode in [Mode::Plain, Mode::Keyed(&key), Mode::DeriveKey("hash_each test")] {
            let mut out = vec![[0u8; OUT_LEN]; items.len()];
            crate::throughput::hash_each(mode, &crate::test::items(&crate::test::messages(&items), &items), &mut out);
            for (item, digest) in items.iter().zip(&out) {
                assert_eq!(digest, crate::convenience::hash_with(mode, item).as_bytes(), "length {}", item.len());
            }
        }
    }
}
