//! Extended output four blocks at a time on NEON (AArch64): every lane
//! compresses the same block at its own counter, as the SME2 kernel does
//! sixteen at a time (`blake3_sme2_xof16_512`). It serves the CPUs without
//! SME2 (Apple M1-M3, most AArch64 cores), and on SME2 the blocks past
//! whole groups of sixteen and calls that find the SME2 turn taken.

use crate::{BLOCK_LEN, CVWords, IV, MSG_SCHEDULE};
use core::arch::aarch64::*;

/// Rotate each lane right by N (L = 32 - N).
#[inline(always)]
unsafe fn rotr<const N: i32, const L: i32>(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32::<N>(vshlq_n_u32::<L>(x), x) }
}

#[inline(always)]
unsafe fn rotr16(x: uint32x4_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

#[inline(always)]
unsafe fn g(v: &mut [uint32x4_t; 16], a: usize, b: usize, c: usize, d: usize, x: uint32x4_t, y: uint32x4_t) {
    unsafe {
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), x);
        v[d] = rotr16(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = rotr::<12, 20>(veorq_u32(v[b], v[c]));
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), y);
        v[d] = rotr::<8, 24>(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = rotr::<7, 25>(veorq_u32(v[b], v[c]));
    }
}

#[inline(always)]
unsafe fn round(v: &mut [uint32x4_t; 16], m: &[uint32x4_t; 16], r: usize) {
    let s = MSG_SCHEDULE[r];
    unsafe {
        g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
}

/// Store four vectors (words w..w+4 of four lanes) transposed: lane j's
/// four words at `out + 64 * j`.
#[inline(always)]
unsafe fn store_transposed(out: *mut u8, a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) {
    unsafe {
        let (t0, t1) = (vtrn1q_u32(a, b), vtrn2q_u32(a, b));
        let (t2, t3) = (vtrn1q_u32(c, d), vtrn2q_u32(c, d));
        let q = |x: uint32x4_t| vreinterpretq_u64_u32(x);
        let rows = [vtrn1q_u64(q(t0), q(t2)), vtrn1q_u64(q(t1), q(t3)), vtrn2q_u64(q(t0), q(t2)), vtrn2q_u64(q(t1), q(t3))];
        for (lane, row) in rows.into_iter().enumerate() {
            vst1q_u8(out.add(lane * BLOCK_LEN), vreinterpretq_u8_u64(row));
        }
    }
}

/// The state of four lanes at `counter` on, before the rounds.
#[inline(always)]
unsafe fn initial(key: &[uint32x4_t; 8], counter: u64, block_len: u8, flags: u8) -> [uint32x4_t; 16] {
    let at: [u64; 4] = core::array::from_fn(|i| counter.wrapping_add(i as u64));
    let low = at.map(|c| c as u32);
    let high = at.map(|c| (c >> 32) as u32);
    unsafe {
        [
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
            vdupq_n_u32(IV[0]),
            vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]),
            vdupq_n_u32(IV[3]),
            vld1q_u32(low.as_ptr()),
            vld1q_u32(high.as_ptr()),
            vdupq_n_u32(block_len as u32),
            vdupq_n_u32(flags as u32),
        ]
    }
}

/// Store four lanes' 64-byte outputs from the state after the rounds.
#[inline(always)]
unsafe fn store(out: *mut u8, v: &[uint32x4_t; 16], key: &[uint32x4_t; 8]) {
    unsafe {
        let o: [uint32x4_t; 16] = core::array::from_fn(|i| if i < 8 { veorq_u32(v[i], v[i + 8]) } else { veorq_u32(v[i], key[i - 8]) });
        for w in (0..16).step_by(4) {
            store_transposed(out.add(4 * w), o[w], o[w + 1], o[w + 2], o[w + 3]);
        }
    }
}

/// Fill `out`'s whole groups of four 64-byte blocks with extended output
/// (block i: `block` under `cv` at `counter + i`, `flags` and `block_len`,
/// both halves of the state); the bytes filled, a multiple of 256. Eight
/// blocks a step where they fit: two independent states, which the core
/// overlaps.
pub fn xof_many(cv: &CVWords, block: &[u8; BLOCK_LEN], block_len: u8, mut counter: u64, flags: u8, out: &mut [u8]) -> usize {
    let words = crate::platform::words_from_le_bytes_64(block);
    let mut done = 0;
    // Sound: NEON is part of every AArch64 CPU; the stores stay within
    // `out`, whole blocks past `done`.
    unsafe {
        let m: [uint32x4_t; 16] = core::array::from_fn(|i| vdupq_n_u32(words[i]));
        let key: [uint32x4_t; 8] = core::array::from_fn(|i| vdupq_n_u32(cv[i]));
        while out.len() - done >= 8 * BLOCK_LEN {
            let mut a = initial(&key, counter, block_len, flags);
            let mut b = initial(&key, counter.wrapping_add(4), block_len, flags);
            for r in 0..7 {
                round(&mut a, &m, r);
                round(&mut b, &m, r);
            }
            let base = out.as_mut_ptr().add(done);
            store(base, &a, &key);
            store(base.add(4 * BLOCK_LEN), &b, &key);
            counter = counter.wrapping_add(8);
            done += 8 * BLOCK_LEN;
        }
        if out.len() - done >= 4 * BLOCK_LEN {
            let mut a = initial(&key, counter, block_len, flags);
            for r in 0..7 {
                round(&mut a, &m, r);
            }
            store(out.as_mut_ptr().add(done), &a, &key);
            done += 4 * BLOCK_LEN;
        }
    }
    done
}

#[cfg(test)]
mod test {
    use super::*;

    /// Against the portable compressor, block by block: counters whose low
    /// word wraps inside a group, flag sets every mode uses, short and full
    /// blocks, lengths that leave blocks past the groups (left untouched).
    #[test]
    fn test_xof_many_matches_portable() {
        let cv: CVWords = core::array::from_fn(|i| 0x9E37_79B9u32.wrapping_mul(i as u32 + 3));
        let mut block = [0u8; BLOCK_LEN];
        crate::test::paint_test_input(&mut block);
        for counter in [0u64, 7, 0xFFFF_FFFE, u64::MAX - 5] {
            for (block_len, flags) in [(64u8, crate::ROOT), (21, crate::ROOT | crate::CHUNK_END), (0, crate::ROOT | crate::KEYED_HASH | crate::PARENT)] {
                for blocks in [4usize, 5, 8, 11, 12, 16, 20] {
                    let mut out = std::vec![0u8; blocks * BLOCK_LEN];
                    let done = xof_many(&cv, &block, block_len, counter, flags, &mut out);
                    assert_eq!(done, blocks / 4 * 4 * BLOCK_LEN);
                    for (i, got) in out[..done].chunks_exact(BLOCK_LEN).enumerate() {
                        let want = crate::portable::compress_xof(&cv, &block, block_len, counter.wrapping_add(i as u64), flags);
                        assert_eq!(got, &want[..], "counter {counter} + {i}, block_len {block_len}, flags {flags:#x}");
                    }
                    assert!(out[done..].iter().all(|&b| b == 0), "nothing past the groups");
                }
            }
        }
    }
}
