//! `hash_each_with`: a collection of messages of any lengths, each hashed,
//! with short messages' chunks side by side in the SIMD lanes (probe; Zooko,
//! October 4, 2026, docs/api-design.md "APIs considered", 2).
//!
//! Chunk k of every message has chunk counter k, so the full chunks k of up
//! to a kernel's width of different messages hash in one call of the
//! many-inputs kernel, at counter k for every lane. Messages of one chunk or
//! less, and messages of 16 chunks or more (which fill the lanes alone), go
//! through `hash_with`; a message's last partial chunk hashes on its own.
//! Each message's chunk chaining values then merge into its root as
//! BLAKE3's tree places them.

use crate::platform::Platform;
use crate::{CVBytes, CVWords, ChunkState, Hash, IncrementCounter, Mode, CHUNK_END, CHUNK_LEN, CHUNK_START, OUT_LEN};

/// Messages of this many chunks or more fill the lanes alone.
const ALONE_CHUNKS: usize = 16;

/// `out[i]` is `items[i]`'s hash in `mode`: `hash_with(mode, items[i])`.
/// Requires `out.len() == items.len()`.
pub fn hash_each_with(mode: Mode, items: &[&[u8]], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(items.len(), out.len(), "one digest per item");
    let (key, flags) = mode.key_and_flags();
    let platform = Platform::detect();
    // The messages whose full chunks share the lanes, and each one's chunk values.
    let mut shared: Vec<(usize, Vec<CVBytes>)> = Vec::new();
    let mut most_full = 0;
    for (i, item) in items.iter().enumerate() {
        let chunks = item.len().div_ceil(CHUNK_LEN);
        if chunks <= 1 || chunks >= ALONE_CHUNKS {
            out[i] = *crate::hash_with(mode, item).as_bytes();
        } else {
            let full = item.len() / CHUNK_LEN;
            most_full = most_full.max(full);
            shared.push((i, vec![[0; 32]; chunks]));
        }
    }
    let mut refs: Vec<&[u8; CHUNK_LEN]> = Vec::with_capacity(shared.len());
    let mut owners: Vec<usize> = Vec::with_capacity(shared.len());
    let mut cvs = vec![0u8; shared.len() * OUT_LEN];
    for k in 0..most_full {
        refs.clear();
        owners.clear();
        for (s, (i, _)) in shared.iter().enumerate() {
            if let Some(chunk) = items[*i].get(k * CHUNK_LEN..(k + 1) * CHUNK_LEN) {
                refs.push(chunk.try_into().unwrap());
                owners.push(s);
            }
        }
        let out_cvs = &mut cvs[..refs.len() * OUT_LEN];
        platform.hash_many(&refs, &key, k as u64, IncrementCounter::No, flags, CHUNK_START, CHUNK_END, out_cvs);
        for (j, &s) in owners.iter().enumerate() {
            shared[s].1[k].copy_from_slice(&out_cvs[j * OUT_LEN..(j + 1) * OUT_LEN]);
        }
    }
    for (i, chunk_cvs) in &mut shared {
        let item = items[*i];
        let full = item.len() / CHUNK_LEN;
        if full < chunk_cvs.len() {
            chunk_cvs[full] = ChunkState::new(&key, full as u64, flags, platform).update(&item[full * CHUNK_LEN..]).output().chaining_value();
        }
        out[*i] = *root(chunk_cvs, &key, flags, platform).as_bytes();
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
