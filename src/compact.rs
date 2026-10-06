//! The compact path for a one-shot call after a pause: a message of more
//! than a chunk and less than 16 KiB, hashed by one rolled four-lane NEON
//! kernel (its rounds a loop, about a kilobyte of code) and a little glue,
//! instead of the unrolled kernels (10-24 KB of code at 2-8 KiB). After
//! other work a call's code comes from memory, and fetching it was a third
//! of such a call (NOTES-servil, "How much of a cold one-shot call is its
//! code"); calls back to back keep the unrolled kernels, whose code is
//! near. Chunks, the partial last chunk, every parent, and the root all go
//! through the one kernel.

use crate::{CVWords, Hash, BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, IV, OUT_LEN, PARENT, ROOT};
use core::arch::aarch64::*;

/// The compact path takes messages of more than a chunk and fewer bytes
/// than this: from 16 chunks the SME2 kernels' throughput outweighs their
/// code.
pub(crate) const COMPACT_BELOW: usize = 16 * CHUNK_LEN;

/// The most chaining values a message below COMPACT_BELOW has.
const MOST_CVS: usize = COMPACT_BELOW / CHUNK_LEN;

/// BLAKE3's message permutation, applied after each round.
const PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

#[inline(always)]
fn rot16(x: uint32x4_t) -> uint32x4_t {
    // Sound throughout: NEON is part of every AArch64 target.
    unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

#[inline(always)]
fn rot12(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32(x, 20), x, 12) }
}

#[inline(always)]
fn rot8(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32(x, 24), x, 8) }
}

#[inline(always)]
fn rot7(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32(x, 25), x, 7) }
}

#[inline(always)]
fn g(v: &mut [uint32x4_t; 16], a: usize, b: usize, c: usize, d: usize, x: uint32x4_t, y: uint32x4_t) {
    unsafe {
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), x);
        v[d] = rot16(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = rot12(veorq_u32(v[b], v[c]));
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), y);
        v[d] = rot8(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = rot7(veorq_u32(v[b], v[c]));
    }
}

/// A 4x4 transpose of 32-bit words: row i of the result holds word i of
/// each of the four inputs.
#[inline(always)]
fn transpose(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> [uint32x4_t; 4] {
    unsafe {
        let (t0, t1) = (vtrn1q_u32(a, b), vtrn2q_u32(a, b));
        let (t2, t3) = (vtrn1q_u32(c, d), vtrn2q_u32(c, d));
        let w = |x: uint32x4_t| vreinterpretq_u64_u32(x);
        let u = |x: uint64x2_t| vreinterpretq_u32_u64(x);
        [u(vtrn1q_u64(w(t0), w(t2))), u(vtrn1q_u64(w(t1), w(t3))), u(vtrn2q_u64(w(t0), w(t2))), u(vtrn2q_u64(w(t1), w(t3)))]
    }
}

/// Four inputs of `blocks` blocks each, side by side, one per lane: each
/// lane's chaining value from `key` through its blocks, at its counter.
/// The last block holds `last_len` bytes (the rest of its 64 zero); the
/// first block takes `flags | start`, the last `flags | end`. Every lane's
/// input has `blocks * 64` readable bytes. Rolled (`inline(never)`, its
/// rounds a loop) so that its code stays small.
#[inline(never)]
fn hash4(inputs: [*const u8; 4], blocks: usize, last_len: u32, key: &CVWords, counters: [u64; 4], flags: u8, start: u8, end: u8, out: &mut [[u8; OUT_LEN]; 4]) {
    debug_assert!(blocks >= 1 && last_len as usize <= BLOCK_LEN);
    // Sound: NEON is part of every AArch64 target, and the caller gives
    // `blocks * 64` readable bytes at each input.
    unsafe {
        let mut h = [vdupq_n_u32(0); 8];
        for (i, word) in h.iter_mut().enumerate() {
            *word = vdupq_n_u32(key[i]);
        }
        let low = vld1q_u32([counters[0] as u32, counters[1] as u32, counters[2] as u32, counters[3] as u32].as_ptr());
        let high = vld1q_u32([(counters[0] >> 32) as u32, (counters[1] >> 32) as u32, (counters[2] >> 32) as u32, (counters[3] >> 32) as u32].as_ptr());
        for block in 0..blocks {
            let mut m = [vdupq_n_u32(0); 16];
            for quarter in 0..4 {
                let at = |lane: usize| vld1q_u32(inputs[lane].add(block * BLOCK_LEN + quarter * 16) as *const u32);
                let rows = transpose(at(0), at(1), at(2), at(3));
                m[4 * quarter..][..4].copy_from_slice(&rows);
            }
            let last = block + 1 == blocks;
            let block_flags = flags | if block == 0 { start } else { 0 } | if last { end } else { 0 };
            let len = if last { last_len } else { BLOCK_LEN as u32 };
            let mut v = [
                h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
                vdupq_n_u32(IV[0]), vdupq_n_u32(IV[1]), vdupq_n_u32(IV[2]), vdupq_n_u32(IV[3]),
                low, high, vdupq_n_u32(len), vdupq_n_u32(block_flags as u32),
            ];
            for round in 0..7 {
                g(&mut v, 0, 4, 8, 12, m[0], m[1]);
                g(&mut v, 1, 5, 9, 13, m[2], m[3]);
                g(&mut v, 2, 6, 10, 14, m[4], m[5]);
                g(&mut v, 3, 7, 11, 15, m[6], m[7]);
                g(&mut v, 0, 5, 10, 15, m[8], m[9]);
                g(&mut v, 1, 6, 11, 12, m[10], m[11]);
                g(&mut v, 2, 7, 8, 13, m[12], m[13]);
                g(&mut v, 3, 4, 9, 14, m[14], m[15]);
                if round < 6 {
                    m = core::array::from_fn(|i| m[PERMUTATION[i]]);
                }
            }
            for i in 0..8 {
                h[i] = veorq_u32(v[i], v[i + 8]);
            }
        }
        for half in 0..2 {
            let rows = transpose(h[4 * half], h[4 * half + 1], h[4 * half + 2], h[4 * half + 3]);
            for (lane, row) in rows.iter().enumerate() {
                vst1q_u32(out[lane].as_mut_ptr().add(16 * half) as *mut u32, *row);
            }
        }
    }
}

/// `input`'s hash in the mode of `key` and `flags`, through the compact
/// path. Requires `CHUNK_LEN < input.len() < COMPACT_BELOW`.
#[cold]
#[inline(never)]
pub(crate) fn hash(input: &[u8], key: &CVWords, flags: u8) -> Hash {
    assert!((CHUNK_LEN + 1..COMPACT_BELOW).contains(&input.len()), "the compact path takes more than a chunk and less than 16 KiB");
    let mut cvs = [[0u8; OUT_LEN]; MOST_CVS];
    let mut out = [[0u8; OUT_LEN]; 4];
    let whole = input.len() / CHUNK_LEN;
    let mut n = 0;
    // The whole chunks, four at a time; a group of fewer repeats its last.
    while n < whole {
        let k = (whole - n).min(4);
        let at = |j: usize| input[(n + j.min(k - 1)) * CHUNK_LEN..].as_ptr();
        let counter = |j: usize| (n + j.min(k - 1)) as u64;
        hash4([at(0), at(1), at(2), at(3)], CHUNK_LEN / BLOCK_LEN, BLOCK_LEN as u32, key, [counter(0), counter(1), counter(2), counter(3)], flags, CHUNK_START, CHUNK_END, &mut out);
        cvs[n..n + k].copy_from_slice(&out[..k]);
        n += k;
    }
    // The partial last chunk, zero-padded to whole blocks.
    let rest = &input[whole * CHUNK_LEN..];
    if !rest.is_empty() {
        let mut padded = [0u8; CHUNK_LEN];
        padded[..rest.len()].copy_from_slice(rest);
        let blocks = rest.len().div_ceil(BLOCK_LEN);
        let last_len = (rest.len() - (blocks - 1) * BLOCK_LEN) as u32;
        let p = padded.as_ptr();
        hash4([p; 4], blocks, last_len, key, [whole as u64; 4], flags, CHUNK_START, CHUNK_END, &mut out);
        cvs[n] = out[0];
        n += 1;
    }
    // Each level pairs neighbours, an odd last value rising alone: the
    // left-balanced tree of BLAKE3, as the chunks' order gives it.
    while n > 2 {
        let pairs = n / 2;
        let mut p = 0;
        while p < pairs {
            let k = (pairs - p).min(4);
            let at = |j: usize| cvs[2 * (p + j.min(k - 1))].as_ptr();
            hash4([at(0), at(1), at(2), at(3)], 1, BLOCK_LEN as u32, key, [0; 4], flags | PARENT, 0, 0, &mut out);
            cvs[p..p + k].copy_from_slice(&out[..k]);
            p += k;
        }
        if n % 2 == 1 {
            cvs[pairs] = cvs[n - 1];
        }
        n = pairs + n % 2;
    }
    let root = cvs[0].as_ptr();
    hash4([root; 4], 1, BLOCK_LEN as u32, key, [0; 4], flags | PARENT | ROOT, 0, 0, &mut out);
    Hash::from(out[0])
}

#[cfg(test)]
mod test {
    /// The compact path agrees with the reference at every length it
    /// takes, partial chunks and odd levels among them, in every mode.
    #[test]
    fn test_compact_matches_reference() {
        let mut input = [0u8; super::COMPACT_BELOW];
        crate::test::paint_test_input(&mut input);
        let key = [9u8; 32];
        let key_words = crate::platform::words_from_le_bytes_32(&key);
        let context = crate::hazmat::hash_derive_key_context("compact test");
        let context_words = crate::platform::words_from_le_bytes_32(&context);
        for len in (crate::CHUNK_LEN + 1..super::COMPACT_BELOW).step_by(37).chain([2048, 3072, 4096, 5 * 1024, 8192, 15 * 1024, 16383]) {
            let message = &input[..len];
            let mut reference = reference_impl::Hasher::new();
            reference.update(message);
            let mut want = [0u8; 32];
            reference.finalize(&mut want);
            assert_eq!(super::hash(message, crate::IV, 0).as_bytes(), &want, "len {len}");
            let mut reference = reference_impl::Hasher::new_keyed(&key);
            reference.update(message);
            reference.finalize(&mut want);
            assert_eq!(super::hash(message, &key_words, crate::KEYED_HASH).as_bytes(), &want, "keyed, len {len}");
            let mut reference = reference_impl::Hasher::new_derive_key("compact test");
            reference.update(message);
            reference.finalize(&mut want);
            assert_eq!(super::hash(message, &context_words, crate::DERIVE_KEY_MATERIAL).as_bytes(), &want, "derive, len {len}");
        }
    }
}
