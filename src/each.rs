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

/// [`hash_each_with`] allowed this crate's worker threads, under the rules
/// of [`hash_multithreaded`](crate::hash_multithreaded): the same digests,
/// and never slower than [`hash_each_with`]. Today it hashes on the calling
/// thread, as [`hash_each_with`] does. Requires `out.len() == items.len()`.
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
pub fn hash_each_multithreaded_with(mode: Mode, items: &[&[u8]], out: &mut [[u8; OUT_LEN]]) {
    hash_each_with(mode, items, out)
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
        for &(i, bytes) in pieces {
            hashers[i].update(bytes);
        }
    }

    /// [`update_each`](Hasher::update_each) allowed this crate's worker
    /// threads, under the rules of
    /// [`update_multithreaded`](Hasher::update_multithreaded): the same
    /// states, and never slower than [`update_each`](Hasher::update_each).
    /// Requires each `i < hashers.len()`.
    pub fn update_each_multithreaded(hashers: &mut [Hasher], pieces: &[(usize, &[u8])]) {
        for &(i, bytes) in pieces {
            hashers[i].update_multithreaded(bytes);
        }
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
    let platform = Platform::detect();
    // The messages whose full chunks share the lanes: each one's index, and
    // where its chunk values start in `cvs`.
    let mut shared: Vec<(usize, usize)> = Vec::new();
    let mut slots = 0;
    let mut most_full = 0;
    for (i, item) in items.iter().enumerate() {
        let chunks = item.len().div_ceil(CHUNK_LEN);
        if chunks <= 1 || chunks >= ALONE_CHUNKS {
            out[i] = *crate::hash_with(mode, item).as_bytes();
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
            let mut hashers: Vec<Hasher> = lens.iter().map(|_| Hasher::new()).collect();
            let mut done = vec![0usize; lens.len()];
            let mut k = 0;
            while done.iter().zip(&lens).any(|(d, l)| d < l) {
                let mut pieces: Vec<(usize, &[u8])> = Vec::new();
                for _ in 0..7 {
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
                assert_eq!(&out[k], crate::hash(&input[..lens[i]]).as_bytes(), "len {} multithreaded {multithreaded}", lens[i]);
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
