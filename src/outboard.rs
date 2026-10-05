//! Verified streaming and range reads: a message's outboard, the parent
//! nodes of its tree above 16 KiB groups, and the check of a range of
//! groups against the message's hash through them.
//!
//! A group is 16 aligned chunks of the message (the last may be shorter).
//! The outboard lists, in pre-order (a node, then its left subtree, then its
//! right), each parent node above the groups as its two children's chaining
//! values, 64 bytes a node: `64 × (groups − 1)` bytes in all, about 0.4% of
//! the message. This is the layout of bao-tree's pre-order outboard with
//! 16 KiB blocks, as iroh-blobs stores it.

use crate::platform::Platform;
use crate::{CVBytes, CVWords, Hash, IncrementCounter, Mode, CHUNK_END, CHUNK_LEN, CHUNK_START, PARENT};

/// The length of a group: 16 chunks, 16 KiB.
pub const GROUP_LEN: usize = 16 * CHUNK_LEN;

const GROUP_CHUNKS: u64 = (GROUP_LEN / CHUNK_LEN) as u64;

/// `input`'s hash in `mode` (as [`hash_with`](crate::hash_with)), and its
/// outboard: the parent nodes above its 16 KiB groups, 64 bytes each, in
/// pre-order (this module's docs). A message of one group or less has an
/// empty outboard.
///
/// ```
/// use blake3_servil::{Mode, outboard_with, verify_range_with, GROUP_LEN};
/// let input = vec![7u8; 100_000];
/// let (hash, outboard) = outboard_with(Mode::Hash, &input);
/// assert_eq!(hash, blake3_servil::hash(&input));
/// // The third and fourth groups, as a range read would fetch them.
/// let range = &input[2 * GROUP_LEN..4 * GROUP_LEN];
/// assert!(verify_range_with(Mode::Hash, &hash, input.len() as u64, &outboard, 2, range));
/// ```
pub fn outboard_with(mode: Mode, input: &[u8]) -> (Hash, Vec<u8>) {
    outboard(mode, input, false)
}

/// [`outboard_with`] over several threads, with the same result: from
/// 512 KiB the groups hash on the calling thread and this crate's worker
/// threads at once, under the rules of
/// [`hash_multithreaded`](crate::hash_multithreaded); a shorter message on
/// the calling thread alone. The parent nodes above the groups (one for
/// each 16 KiB) are computed on the calling thread.
///
/// ```
/// use blake3_servil::Mode;
/// let input = vec![9u8; 3 << 20];
/// assert_eq!(blake3_servil::outboard_multithreaded_with(Mode::Hash, &input), blake3_servil::outboard_with(Mode::Hash, &input));
/// ```
pub fn outboard_multithreaded_with(mode: Mode, input: &[u8]) -> (Hash, Vec<u8>) {
    outboard(mode, input, true)
}

fn outboard(mode: Mode, input: &[u8], pooled: bool) -> (Hash, Vec<u8>) {
    let chunks = chunks_of(input.len() as u64);
    if chunks <= GROUP_CHUNKS {
        return (crate::hash_with(mode, input), Vec::new());
    }
    let (key, flags) = mode.key_and_flags();
    let platform = Platform::detect();
    let cvs = if pooled { crate::lanes::group_cvs(input, &key, flags) } else { group_cvs(input, 0, &key, flags, platform) };
    let mut out = Vec::with_capacity(64 * (cvs.len() - 1));
    let (left, right) = node(&cvs, chunks, &key, flags, platform, &mut out);
    (crate::parent_node_output(&left, &right, &key, flags, platform).root_hash(), out)
}

/// Whether `bytes`, the message's bytes from group `first` on (offset
/// `first × GROUP_LEN`), belong to the message of `len` bytes whose hash in
/// `mode` is `hash`, by its `outboard` ([`outboard_with`]). `bytes` ends at a
/// group's end or at the message's end. It hashes the range's groups and
/// checks each parent node on their paths to the root once: a range of a
/// megabyte or more costs about what hashing it costs, and a single group
/// about twice that (its path's parents). Requires an outboard of the length `len`
/// gives, and a range within the message, nonempty unless the message is.
pub fn verify_range_with(mode: Mode, hash: &Hash, len: u64, outboard: &[u8], first: u64, bytes: &[u8]) -> bool {
    let chunks = chunks_of(len);
    let groups = groups_of(chunks);
    assert_eq!(outboard.len() as u64, 64 * (groups - 1), "an outboard of 64 bytes for each group but one");
    let start = first * GROUP_LEN as u64;
    let end = start + bytes.len() as u64;
    assert!(first < groups && end <= len && (end % GROUP_LEN as u64 == 0 || end == len), "a range of whole groups within the message");
    assert!(!bytes.is_empty() || len == 0, "a nonempty range");
    if groups == 1 {
        return crate::hash_with(mode, bytes) == *hash;
    }
    let (key, flags) = mode.key_and_flags();
    let platform = Platform::detect();
    let cvs = group_cvs(bytes, first, &key, flags, platform);
    let range = first..first + cvs.len() as u64;
    check(outboard, 0, 0, chunks, None, &range, &cvs, hash, &key, flags, platform)
}

/// A message received as its encoding, verified as it arrives: the
/// message's groups and parent nodes in pre-order (a node's 64 bytes, then
/// its left subtree, then its right; a group's bytes at each leaf), as
/// bao-tree's `encode_ranges` writes a whole message with 16 KiB blocks,
/// which is iroh-blobs' wire format. The encoding is
/// `len + 64 × (groups − 1)` bytes. Each node and group is checked once,
/// against the hash it must have, so verifying costs about what
/// [`outboard_with`] costs.
///
/// ```
/// use blake3_servil::{Mode, Verifier};
/// # let (input, encoding) = (vec![5u8; 50_000], blake3_servil::test_encoding(Mode::Hash, &vec![5u8; 50_000]));
/// let hash = blake3_servil::hash(&input);
/// let mut verifier = Verifier::new(Mode::Hash, &hash, input.len() as u64);
/// let mut received = Vec::new();
/// for piece in encoding.chunks(1448) {
///     assert!(verifier.update(piece, |bytes| received.extend_from_slice(bytes)));
/// }
/// assert_eq!((verifier.remaining(), received), (0, input));
/// ```
pub struct Verifier {
    key: CVWords,
    flags: u8,
    mode_hash: Hash,
    len: u64,
    remaining: u64,
    /// The nodes still to receive, the next last: (first group, chunks, expected value; `None` for the root).
    nodes: Vec<(u64, u64, Option<CVBytes>)>,
    /// The bytes of the next node or group, gathered while it arrives in pieces.
    gathered: Vec<u8>,
    failed: bool,
}

impl Verifier {
    /// A verifier for the message of `len` bytes whose hash in `mode` is `hash`.
    pub fn new(mode: Mode, hash: &Hash, len: u64) -> Verifier {
        let (key, flags) = mode.key_and_flags();
        let chunks = chunks_of(len);
        Verifier {
            key,
            flags,
            mode_hash: *hash,
            len,
            remaining: len + 64 * (groups_of(chunks) - 1),
            nodes: vec![(0, chunks, None)],
            gathered: Vec::new(),
            failed: false,
        }
    }

    /// The encoding's bytes still to come.
    pub fn remaining(&self) -> u64 {
        self.remaining
    }

    /// Takes the encoding's next bytes, a piece of any length, and gives
    /// each group of the message to `verified` once it is checked, in
    /// order. Returns whether everything received so far is the message's;
    /// once it returns false the verifier is spent. The message is whole
    /// and verified when `update` has returned true and `remaining` is 0
    /// (for the empty message, after one call, with no bytes). Requires at
    /// most `remaining` bytes, and a verifier that has returned true.
    pub fn update(&mut self, encoded: &[u8], mut verified: impl FnMut(&[u8])) -> bool {
        assert!(!self.failed, "a verifier that has found a mismatch is spent");
        assert!(encoded.len() as u64 <= self.remaining, "{} bytes past the encoding's end", encoded.len() as u64 - self.remaining);
        self.remaining -= encoded.len() as u64;
        let platform = Platform::detect();
        let mut input = encoded;
        while let Some(&(g0, size, expected)) = self.nodes.last() {
            let need = if size > GROUP_CHUNKS { 64 } else { (self.len - g0 * GROUP_LEN as u64).min(GROUP_LEN as u64) as usize };
            // The node's bytes: in place in the piece when it holds them
            // whole, else gathered across pieces.
            let mut gathered = core::mem::take(&mut self.gathered);
            let bytes: &[u8] = if gathered.is_empty() && input.len() >= need {
                let (bytes, rest) = input.split_at(need);
                input = rest;
                bytes
            } else {
                let take = (need - gathered.len()).min(input.len());
                gathered.extend_from_slice(&input[..take]);
                input = &input[take..];
                if gathered.len() < need {
                    self.gathered = gathered;
                    return true;
                }
                &gathered
            };
            self.nodes.pop();
            let ok = if size > GROUP_CHUNKS {
                let (left, right): (CVBytes, CVBytes) = (bytes[..32].try_into().unwrap(), bytes[32..].try_into().unwrap());
                let parent = crate::parent_node_output(&left, &right, &self.key, self.flags, platform);
                let split = left_chunks(size);
                self.nodes.push((g0 + split / GROUP_CHUNKS, size - split, Some(right)));
                self.nodes.push((g0, split, Some(left)));
                match expected {
                    None => parent.root_hash() == self.mode_hash,
                    Some(cv) => parent.chaining_value() == cv,
                }
            } else {
                let ok = match expected {
                    None => crate::hash_serial(bytes, &self.key, self.flags) == self.mode_hash,
                    Some(cv) => group_cv(bytes, g0 * GROUP_CHUNKS, &self.key, self.flags) == cv,
                };
                if ok {
                    verified(bytes);
                }
                ok
            };
            gathered.clear();
            self.gathered = gathered;
            if !ok {
                self.failed = true;
                return false;
            }
        }
        true
    }
}

/// A whole message's encoding, as [`Verifier`] reads it: for the docs and
/// tests.
#[doc(hidden)]
pub fn test_encoding(mode: Mode, input: &[u8]) -> Vec<u8> {
    let (_, outboard) = outboard_with(mode, input);
    let mut out = Vec::with_capacity(input.len() + outboard.len());
    let mut parents = outboard.chunks_exact(64);
    let mut nodes = vec![(0u64, chunks_of(input.len() as u64))];
    while let Some((g0, size)) = nodes.pop() {
        if size > GROUP_CHUNKS {
            out.extend_from_slice(parents.next().unwrap());
            let split = left_chunks(size);
            nodes.push((g0 + split / GROUP_CHUNKS, size - split));
            nodes.push((g0, split));
        } else {
            let start = g0 as usize * GROUP_LEN;
            out.extend_from_slice(&input[start..(start + GROUP_LEN).min(input.len())]);
        }
    }
    out
}

/// Check the node over groups from `g0`, `size` chunks, its pair at `at` in
/// the outboard, against its expected chaining value (`None`: the root,
/// against `hash`), then each child the range reaches; at a group, its
/// chaining value against the range's.
#[allow(clippy::too_many_arguments)]
fn check(outboard: &[u8], at: usize, g0: u64, size: u64, expected: Option<CVBytes>, range: &std::ops::Range<u64>, cvs: &[CVBytes], hash: &Hash, key: &CVWords, flags: u8, platform: Platform) -> bool {
    if size <= GROUP_CHUNKS {
        return expected == Some(cvs[(g0 - range.start) as usize]);
    }
    let pair: &[u8; 64] = outboard[at..at + 64].try_into().unwrap();
    let (left, right): (CVBytes, CVBytes) = (pair[..32].try_into().unwrap(), pair[32..].try_into().unwrap());
    let parent = crate::parent_node_output(&left, &right, key, flags, platform);
    let ok = match expected {
        None => parent.root_hash() == *hash,
        Some(cv) => parent.chaining_value() == cv,
    };
    let split = left_chunks(size);
    let left_groups = split / GROUP_CHUNKS;
    let mid = g0 + left_groups;
    ok && (range.start >= mid || check(outboard, at + 64, g0, split, Some(left), range, cvs, hash, key, flags, platform))
        && (range.end <= mid || check(outboard, at + 64 * left_groups as usize, mid, size - split, Some(right), range, cvs, hash, key, flags, platform))
}

/// The node over the groups `cvs` (more than one), `size` chunks in all:
/// its two children's chaining values, with its subtree's parent nodes
/// appended to `out` in pre-order.
fn node(cvs: &[CVBytes], size: u64, key: &CVWords, flags: u8, platform: Platform, out: &mut Vec<u8>) -> (CVBytes, CVBytes) {
    let at = out.len();
    out.extend_from_slice(&[0; 64]);
    let split = left_chunks(size);
    let left_groups = (split / GROUP_CHUNKS) as usize;
    let left = subtree(&cvs[..left_groups], split, key, flags, platform, out);
    let right = subtree(&cvs[left_groups..], size - split, key, flags, platform, out);
    out[at..at + 32].copy_from_slice(&left);
    out[at + 32..at + 64].copy_from_slice(&right);
    (left, right)
}

/// A subtree's chaining value, its parent nodes appended to `out`.
fn subtree(cvs: &[CVBytes], size: u64, key: &CVWords, flags: u8, platform: Platform, out: &mut Vec<u8>) -> CVBytes {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let (left, right) = node(cvs, size, key, flags, platform, out);
    crate::parent_node_output(&left, &right, key, flags, platform).chaining_value()
}

/// Groups hashed in one batch: their chunks in one many-inputs call, then
/// each level of their parents in one more.
const BATCH_GROUPS: usize = 64;

/// The chaining value, as a non-root subtree, of each group of `input`, the
/// message's bytes from group `first` on.
fn group_cvs(input: &[u8], first_group: u64, key: &CVWords, flags: u8, platform: Platform) -> Vec<CVBytes> {
    let mut cvs = vec![[0u8; 32]; input.len().div_ceil(GROUP_LEN).max(1)];
    group_cvs_into(input, first_group, key, flags, platform, &mut cvs);
    cvs
}

/// [`group_cvs`] into `cvs`, one per group. Full groups go in batches
/// through the many-inputs kernels, back to back; a last, shorter group
/// alone.
pub(crate) fn group_cvs_into(input: &[u8], first_group: u64, key: &CVWords, flags: u8, platform: Platform, cvs: &mut [CVBytes]) {
    let full = input.len() / GROUP_LEN;
    let mut done = 0;
    let per_batch = BATCH_GROUPS * GROUP_CHUNKS as usize;
    let mut level = vec![0u8; per_batch * 32];
    let mut next = vec![0u8; per_batch / 2 * 32];
    let mut chunk_refs: Vec<&[u8; CHUNK_LEN]> = Vec::with_capacity(per_batch);
    let mut first = 0;
    while first < full {
        let n = BATCH_GROUPS.min(full - first);
        chunk_refs.clear();
        chunk_refs.extend(input[first * GROUP_LEN..(first + n) * GROUP_LEN].chunks_exact(CHUNK_LEN).map(|c| <&[u8; CHUNK_LEN]>::try_from(c).unwrap()));
        let mut values = chunk_refs.len();
        platform.hash_many(&chunk_refs, key, (first_group + first as u64) * GROUP_CHUNKS, IncrementCounter::Yes, flags, CHUNK_START, CHUNK_END, &mut level[..values * 32]);
        // Four levels of parents: a group's 16 chunk values down to one.
        while values > n {
            let blocks: Vec<&[u8; 64]> = level[..values * 32].chunks_exact(64).map(|b| <&[u8; 64]>::try_from(b).unwrap()).collect();
            platform.hash_many(&blocks, key, 0, IncrementCounter::No, flags | PARENT, 0, 0, &mut next[..values / 2 * 32]);
            values /= 2;
            level[..values * 32].copy_from_slice(&next[..values * 32]);
        }
        for c in level[..n * 32].chunks_exact(32) {
            cvs[done].copy_from_slice(c);
            done += 1;
        }
        first += n;
    }
    if input.len() > full * GROUP_LEN || input.is_empty() {
        cvs[done] = group_cv(&input[full * GROUP_LEN..], (first_group + full as u64) * GROUP_CHUNKS, key, flags);
    }
}

/// A group's chaining value as a non-root subtree at chunk `start`.
fn group_cv(group: &[u8], start: u64, key: &CVWords, flags: u8) -> CVBytes {
    let mut core = crate::HasherCore::new_internal(key, flags);
    core.set_input_offset(start * CHUNK_LEN as u64);
    core.update(group);
    core.final_output().chaining_value()
}

/// Chunks in a message of `len` bytes (one for the empty message).
fn chunks_of(len: u64) -> u64 {
    len.div_ceil(CHUNK_LEN as u64).max(1)
}

fn groups_of(chunks: u64) -> u64 {
    chunks.div_ceil(GROUP_CHUNKS)
}

/// The left subtree of a node over `size` chunks (2 or more): the largest
/// power of two below `size`.
fn left_chunks(size: u64) -> u64 {
    1 << (63 - (size - 1).leading_zeros())
}

#[cfg(test)]
mod test {
    use super::*;

    /// The pool's ranges of groups (lanes::Work::Groups), just past where
    /// the pool takes them, against the calling thread's: small enough for
    /// Miri (the CI's smoketest runs it).
    #[test]
    fn test_miri_outboard_multithreaded() {
        let mut input = vec![0u8; 33 * GROUP_LEN + 100];
        crate::test::paint_test_input(&mut input);
        assert_eq!(outboard_multithreaded_with(Mode::Hash, &input), outboard_with(Mode::Hash, &input));
    }

    /// Every message length around group and tree boundaries, every mode,
    /// pieces of many lengths: the Verifier gives back the message, and
    /// stops at a changed byte of a group or a parent node, or a wrong hash.
    #[test]
    fn test_verifier() {
        let mut input = vec![0u8; 37 * GROUP_LEN + 5];
        crate::test::paint_test_input(&mut input);
        let key = [3u8; 32];
        let lens = [0, 1, CHUNK_LEN, GROUP_LEN - 1, GROUP_LEN, GROUP_LEN + 1, 2 * GROUP_LEN, 3 * GROUP_LEN - 7, 4 * GROUP_LEN, 5 * GROUP_LEN + CHUNK_LEN, 17 * GROUP_LEN + 1, input.len()];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("verifier test")] {
            for &len in &lens {
                let message = &input[..len];
                let hash = crate::hash_with(mode, message);
                let encoding = test_encoding(mode, message);
                assert_eq!(encoding.len() as u64, Verifier::new(mode, &hash, len as u64).remaining());
                let run = |encoding: &[u8], hash: &Hash, piece: usize| -> Option<Vec<u8>> {
                    let mut verifier = Verifier::new(mode, hash, len as u64);
                    let mut received = Vec::new();
                    let mut pieces: Vec<&[u8]> = encoding.chunks(piece).collect();
                    if pieces.is_empty() {
                        pieces.push(&[]);
                    }
                    for p in pieces {
                        if !verifier.update(p, |b| received.extend_from_slice(b)) {
                            return None;
                        }
                    }
                    assert_eq!(verifier.remaining(), 0);
                    Some(received)
                };
                for piece in [1, 63, 64, 1448, GROUP_LEN, GROUP_LEN + 64, 1 << 20] {
                    assert_eq!(run(&encoding, &hash, piece).as_deref(), Some(message), "len {len} pieces of {piece}");
                }
                let mut wrong = *hash.as_bytes();
                wrong[0] ^= 1;
                assert_eq!(run(&encoding, &Hash::from_bytes(wrong), 1448), None, "len {len}, wrong hash");
                for at in [0, encoding.len() / 2, encoding.len().saturating_sub(1)] {
                    if at < encoding.len() {
                        let mut bad = encoding.clone();
                        bad[at] ^= 1;
                        assert_eq!(run(&bad, &hash, 1448), None, "len {len}, byte {at} changed");
                    }
                }
            }
        }
    }

    /// Every message length around group and tree boundaries, every mode:
    /// the hash is hash_with's, the outboard's length is right, every group
    /// and every range verifies, and a changed byte, a changed parent node,
    /// or groups at the wrong place do not.
    #[test]
    fn test_outboard_and_verify() {
        let mut input = vec![0u8; 70 * GROUP_LEN + 5];
        crate::test::paint_test_input(&mut input);
        let key = [3u8; 32];
        let lens = [0, 1, CHUNK_LEN, GROUP_LEN - 1, GROUP_LEN, GROUP_LEN + 1, 2 * GROUP_LEN, 3 * GROUP_LEN - 7, 4 * GROUP_LEN, 5 * GROUP_LEN + CHUNK_LEN, 17 * GROUP_LEN + 1, input.len()];
        for mode in [Mode::Hash, Mode::Keyed(&key), Mode::DeriveKey("outboard test")] {
            for &len in &lens {
                let message = &input[..len];
                let (hash, outboard) = outboard_with(mode, message);
                assert_eq!(hash, crate::hash_with(mode, message), "len {len}");
                assert_eq!((hash, outboard.clone()), outboard_multithreaded_with(mode, message), "multithreaded, len {len}");
                let groups = groups_of(chunks_of(len as u64));
                assert_eq!(outboard.len() as u64, 64 * (groups - 1));
                let range = |a: u64, b: u64| &message[a as usize * GROUP_LEN..(b as usize * GROUP_LEN).min(len)];
                let ok = |a: u64, bytes: &[u8]| verify_range_with(mode, &hash, len as u64, &outboard, a, bytes);
                for a in 0..groups {
                    for b in [a + 1, groups].into_iter().chain([(a + 3).min(groups)]) {
                        let bytes = range(a, b);
                        assert!(ok(a, bytes), "len {len} groups {a}..{b}");
                        if !bytes.is_empty() {
                            let mut bad = bytes.to_vec();
                            let at = bad.len() - 1;
                            bad[at] ^= 1;
                            assert!(!ok(a, &bad), "len {len} groups {a}..{b} changed");
                        }
                    }
                    if a + 1 < groups && a + 2 <= groups {
                        let next = range(a + 1, a + 2);
                        if next.len() == GROUP_LEN && next != range(a, a + 1) {
                            assert!(!ok(a, next), "len {len}: group {} at {a}", a + 1);
                        }
                    }
                }
                if !outboard.is_empty() {
                    for at in [0, outboard.len() - 1] {
                        let mut bad = outboard.clone();
                        bad[at] ^= 1;
                        assert!(!verify_range_with(mode, &hash, len as u64, &bad, 0, message), "len {len} outboard byte {at} changed");
                    }
                }
            }
        }
    }
}
