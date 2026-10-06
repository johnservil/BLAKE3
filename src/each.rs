//! [`hash_each_with`]: a collection of messages of any lengths, each hashed,
//! with short messages' chunks side by side in the SIMD lanes.
//!
//! Chunk k of every message has chunk counter k, so the full chunks k of
//! different messages hash in one call of the many-inputs kernel, at counter
//! k for every lane. Messages of one chunk or less, and messages of 16 chunks
//! or more (which fill the lanes alone), go through `hash_with`; a message's
//! last partial chunk hashes on its own. Each message's chunk chaining values
//! then merge into its root as BLAKE3's tree places them.

use crate::platform::Platform;
use crate::{CVBytes, CVWords, ChunkState, Hash, Hasher, IncrementCounter, Mode, CHUNK_END, CHUNK_LEN, CHUNK_START, OUT_LEN};

/// [`hash_each_with`] over several threads, with the same digests: from
/// 512 KiB in all, the items are cut into ranges of about a thread's share
/// of their bytes, which the calling thread and this crate's worker
/// threads hash at once, under the rules of
/// [`hash_multithreaded`](crate::hash_multithreaded); an item of 512 KiB
/// or more as `hash_multithreaded` hashes it, and a smaller collection on
/// the calling thread alone. Allocates what [`hash_each_with`] does, and
/// 16 bytes for each item and 24 for each range, freed when it returns. Requires `out.len() == items.len()`.
///
/// ```
/// use blake3_servil::Mode;
/// let items: Vec<Vec<u8>> = (0..100).map(|i| vec![i as u8; 1000 * i]).collect();
/// let slices: Vec<&[u8]> = items.iter().map(|m| m.as_slice()).collect();
/// let (mut a, mut b) = (vec![[0u8; 32]; 100], vec![[0u8; 32]; 100]);
/// blake3_servil::hash_each_multithreaded_with(Mode::Hash, &slices, &mut a);
/// blake3_servil::hash_each_with(Mode::Hash, &slices, &mut b);
/// assert_eq!(a, b);
/// ```
#[cfg(feature = "std")]
pub fn hash_each_multithreaded_with(mode: Mode, items: &[&[u8]], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(items.len(), out.len(), "one digest per item");
    let (key, flags) = mode.key_and_flags();
    let alone = |item: &[u8]| item.len() >= crate::lanes::MIN_SPLIT_LEN;
    // The collection with each item that hashes alone set aside (an empty
    // stand-in, its digest written after).
    let small: Vec<&[u8]> = items.iter().map(|&i| if alone(i) { &[][..] } else { i }).collect();
    if small.iter().map(|i| i.len()).sum::<usize>() < crate::lanes::MIN_SPLIT_LEN {
        hash_each_on(&key, flags, &small, out, Platform::detect(), |item| crate::hash_with(mode, item));
    } else {
        crate::lanes::hash_each(&key, flags, &small, out);
    }
    for (item, out) in items.iter().zip(out.iter_mut()) {
        if alone(item) {
            *out = *crate::lanes::hash_with_key(item, &key, flags).as_bytes();
        }
    }
}

/// Many messages in progress at once, a server's uploads, each with its own
/// [`Hasher`]: these calls take a whole turn of them, so their bytes can
/// share the SIMD lanes.
impl Hasher {
    /// Feeds each piece to its hasher: `(i, bytes)` is
    /// `hashers[i].update(bytes)`, and a hasher's pieces go in their order
    /// in `pieces`. One call for all the pieces a program has in hand (a
    /// turn of its event loop) runs at least as fast as one
    /// [`update`](Hasher::update) each. Requires each `i < hashers.len()`.
    ///
    /// ```
    /// use blake3_servil::Hasher;
    /// let mut hashers = vec![Hasher::new(), Hasher::new()];
    /// Hasher::update_each(&mut hashers, &[(0, b"ab"), (1, b"xyz"), (0, b"c")]);
    /// let mut out = [[0u8; 32]; 2];
    /// Hasher::finalize_each(&hashers, &[0, 1], &mut out);
    /// assert_eq!((&out[0], &out[1]), (blake3_servil::hash(b"abc").as_bytes(), blake3_servil::hash(b"xyz").as_bytes()));
    /// ```
    pub fn update_each(hashers: &mut [Hasher], pieces: &[(usize, &[u8])]) {
        update_turn::<crate::join::SerialJoin>(hashers, pieces, false);
    }

    /// [`update_each`](Hasher::update_each) allowed this crate's worker
    /// threads, under the rules of
    /// [`update_multithreaded`](Hasher::update_multithreaded): the same
    /// states, and never slower than [`update_each`](Hasher::update_each).
    /// Requires each `i < hashers.len()`.
    #[cfg(feature = "std")]
    pub fn update_each_multithreaded(hashers: &mut [Hasher], pieces: &[(usize, &[u8])]) {
        update_turn::<crate::join::SerialJoin>(hashers, pieces, true);
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
}

/// A turn of pieces for many hashers: each piece gathered into its
/// hasher's stage, and every stage that fills a whole group held, its
/// hasher's later pieces waiting, until the turn's held groups hash
/// together: their chunks back to back, a group per call with nothing in
/// between (the SME unit stays in its fast state), then each level of
/// their parents in one call. Then the waiting pieces, the same way, until
/// none is left.
fn update_turn<J: crate::join::Join>(hashers: &mut [Hasher], pieces: &[(usize, &[u8])], pooled: bool) {
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
            let rest = hashers[i].gather_part::<J>(bytes, pooled, true);
            if hashers[i].held_group().is_some() {
                held.push(i);
                if !rest.is_empty() {
                    waiting.push((i, rest));
                }
            }
        }
        hash_held(hashers, &held);
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
    let refs: Vec<&[u8; CHUNK_LEN]> = alike.iter().flat_map(|&i| hashers[i].held_group().expect("a held group").1.chunks_exact(CHUNK_LEN)).map(|c| c.try_into().unwrap()).collect();
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
        unsafe { crate::sme2::hash_groups_at(&refs, &counters, &key, flags, CHUNK_START, CHUNK_END, &mut level) };
    } else {
        for (k, &counter) in counters.iter().enumerate() {
            platform.hash_many(&refs[k * per..(k + 1) * per], &key, counter, IncrementCounter::Yes, flags, CHUNK_START, CHUNK_END, &mut level[k * per * OUT_LEN..(k + 1) * per * OUT_LEN]);
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

/// Each message's hash in `mode`: `out[i]` is
/// [`hash_with`](crate::hash_with)`(mode, items[i])`, for messages of any
/// lengths. Short messages hash side by side in the SIMD lanes, so a
/// collection of small items (a tree's files, a repository's objects) hashes
/// in less time than one call per item. Requires `out.len() == items.len()`.
/// Allocates 32 bytes for each chunk of the messages from 2 to 15 chunks
/// long, freed when it returns.
///
/// ```
/// use blake3_servil::Mode;
/// let items: Vec<Vec<u8>> = (0..100).map(|i| vec![i as u8; 100 * i]).collect();
/// let slices: Vec<&[u8]> = items.iter().map(|m| m.as_slice()).collect();
/// let mut out = vec![[0u8; 32]; slices.len()];
/// blake3_servil::hash_each_with(Mode::Hash, &slices, &mut out);
/// assert_eq!(&out[42], blake3_servil::hash(&items[42]).as_bytes());
/// ```
pub fn hash_each_with(mode: Mode, items: &[&[u8]], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(items.len(), out.len(), "one digest per item");
    let (key, flags) = mode.key_and_flags();
    hash_each_on(&key, flags, items, out, Platform::detect(), |item| crate::hash_with(mode, item));
}

/// [`hash_each_with`] in the mode of `key` and `flags`, the lanes on
/// `platform`, each message that fills the lanes alone (or is a chunk or
/// less) through `alone`.
pub(crate) fn hash_each_on(key: &CVWords, flags: u8, items: &[&[u8]], out: &mut [[u8; OUT_LEN]], platform: Platform, alone: impl Fn(&[u8]) -> Hash) {
    let key = *key;
    // The messages whose full chunks share the lanes: each one's index, and
    // where its chunk values start in `cvs`.
    let mut shared: Vec<(usize, usize)> = Vec::new();
    let mut slots = 0;
    let mut most_full = 0;
    for (i, item) in items.iter().enumerate() {
        let chunks = item.len().div_ceil(CHUNK_LEN);
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
        refs.clear();
        owners.clear();
        for &(i, at) in &shared {
            if let Some(chunk) = items[i].get(k * CHUNK_LEN..(k + 1) * CHUNK_LEN) {
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
        let item = items[i];
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
    crate::parent_node_output(&left, &right, key, flags, platform).root_hash()
}

/// A non-root subtree's chaining value.
fn subtree(cvs: &[CVBytes], key: &CVWords, flags: u8, platform: Platform) -> CVBytes {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let split = left_len(cvs.len());
    let left = subtree(&cvs[..split], key, flags, platform);
    let right = subtree(&cvs[split..], key, flags, platform);
    crate::parent_node_output(&left, &right, key, flags, platform).chaining_value()
}

/// The largest power of two below `n` (n of 2 or more).
fn left_len(n: usize) -> usize {
    debug_assert!(n >= 2);
    1 << (usize::BITS - 1 - (n - 1).leading_zeros())
}

#[cfg(test)]
mod test {
    use super::*;

    /// The pool's ranges of items (lanes::Work::Each), just past where the
    /// pool takes them, against the calling thread's: small enough for
    /// Miri (the CI's smoketest runs it).
    #[test]
    fn test_miri_hash_each_multithreaded() {
        let mut input = vec![0u8; 520 * 1024];
        crate::test::paint_test_input(&mut input);
        let items: Vec<&[u8]> = input.chunks(13 * 1024 + 7).collect();
        let (mut a, mut b) = (vec![[0u8; OUT_LEN]; items.len()], vec![[0u8; OUT_LEN]; items.len()]);
        hash_each_multithreaded_with(Mode::Hash, &items, &mut a);
        hash_each_with(Mode::Hash, &items, &mut b);
        assert_eq!(a, b);
    }

    /// A collection past the split (items around chunk, lane, and the
    /// split's own length, a few of 512 KiB and more), every mode: the
    /// multithreaded form gives hash_each_with's digests.
    #[test]
    fn test_hash_each_multithreaded() {
        let mut input = vec![0u8; 3 << 20];
        crate::test::paint_test_input(&mut input);
        let lens = [0, 1, 1024, 1025, 3000, 15 * 1024, 16 * 1024, 100_000, 511 * 1024, 512 * 1024, 700_000];
        let items: Vec<&[u8]> = (0..400).map(|k| { let n = lens[k * 7 % lens.len()].min(input.len() - k); &input[k..k + n] }).collect();
        let key = [9u8; 32];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("each mt")] {
            let (mut a, mut b) = (vec![[0u8; OUT_LEN]; items.len()], vec![[0u8; OUT_LEN]; items.len()]);
            hash_each_multithreaded_with(mode, &items, &mut a);
            hash_each_with(mode, &items, &mut b);
            assert_eq!(a, b);
            // Below the split, on the calling thread.
            hash_each_multithreaded_with(mode, &items[..20], &mut a[..20]);
            assert_eq!(a[..20], b[..20]);
        }
    }

    /// Many hashers fed in turns of interleaved pieces of many lengths:
    /// update_each, its multithreaded twin, and finalize_each give every
    /// message the hash one Hasher per message gives.
    #[test]
    fn test_update_each() {
        let mut input = vec![0u8; 1 << 20];
        crate::test::paint_test_input(&mut input);
        let lens = [0, 1, 63, 64, 1023, 1024, 1025, 2048, 4470, 8191, 16384, 16385, 65536, 100_000, 300_000];
        let piece_lens = [1, 1448, 4096, 1448, 16384, 31, 65536];
        for multithreaded in [false, true] {
            let key = [5u8; 32];
            // Every third hasher keyed: a turn's held groups in two kinds.
            let mut hashers: Vec<Hasher> = (0..lens.len()).map(|i| if i % 3 == 1 { Hasher::new_keyed(&key) } else { Hasher::new() }).collect();
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
                    Hasher::update_each_multithreaded(&mut hashers, &pieces);
                } else {
                    Hasher::update_each(&mut hashers, &pieces);
                }
            }
            let which: Vec<usize> = (0..lens.len()).rev().collect();
            let mut out = vec![[0u8; OUT_LEN]; lens.len()];
            Hasher::finalize_each(&hashers, &which, &mut out);
            for (k, &i) in which.iter().enumerate() {
                let want = if i % 3 == 1 { crate::keyed_hash(&key, &input[..lens[i]]) } else { crate::hash(&input[..lens[i]]) };
                assert_eq!(&out[k], want.as_bytes(), "len {} multithreaded {multithreaded}", lens[i]);
            }
        }
    }

    /// Every length around chunk and lane boundaries, in every mode, mixed in
    /// one call, against `hash_with` on each.
    #[test]
    fn test_hash_each_matches_hash_with() {
        let mut input = vec![0u8; 40 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        let mut lens = vec![0, 1, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 2049, 3000, 4096, 5000, 15 * CHUNK_LEN, 15 * CHUNK_LEN + 1, 16 * CHUNK_LEN - 1, 16 * CHUNK_LEN, 16 * CHUNK_LEN + 1, 40 * CHUNK_LEN];
        for n in 2..16 {
            lens.push(n * CHUNK_LEN);
            lens.push(n * CHUNK_LEN - 7);
        }
        let items: Vec<&[u8]> = lens.iter().enumerate().map(|(j, &l)| &input[j % 3..j % 3 + l.min(input.len() - 2)]).collect();
        let key = [9u8; 32];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("hash_each test")] {
            let mut out = vec![[0u8; OUT_LEN]; items.len()];
            hash_each_with(mode, &items, &mut out);
            for (item, digest) in items.iter().zip(&out) {
                assert_eq!(digest, crate::hash_with(mode, item).as_bytes(), "length {}", item.len());
            }
        }
    }
}
