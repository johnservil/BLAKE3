//! AArch64 NEON compression of two inputs at once in the "dup" layout, for
//! inputs of two to fifteen chunks where the sixteen-lane SME2 kernel has no
//! full group and upstream's four-lane NEON kernel is latency-bound.
//!
//! # The layout
//!
//! Each 64-bit lane holds one 32-bit state word twice: `w | w << 32`. A
//! 128-bit vector then carries word *i* of input A (low lane) and of input B
//! (high lane). The point is `xar` (EOR then rotate right, from the SHA-3
//! extension): it rotates 64-bit lanes, and a bit pattern with period 32
//! rotated by r < 32 rotates both 32-bit halves by r. So every `d = (d ^ a)
//! >>> 16` in the G function is one two-cycle instruction, where the plain
//! four-lane layout needs `eor` plus a rotate idiom (two to three ops and
//! four cycles). A G becomes ten instructions with a sixteen-cycle
//! dependency chain, against upstream's seventeen instructions and thirty-two
//! cycles.
//!
//! # Why interleave two pairs
//!
//! One pair alone uses two of the four vector pipes; the chain latency is
//! the limit. Two pairs (four inputs) run in the same wall time, which
//! halves the cost per input. Three or more pairs exceed the register file
//! and spill, so four-input calls go here and larger calls take upstream's
//! hash4 kernel, which fills the pipes with throughput-bound work.
//!
//! # Contract
//!
//! Callers guarantee the CPU has NEON and the SHA-3 extension (`xar`); the
//! `Platform` layer checks `sha3` at runtime once. Every input in a call has
//! the same block count; `hash_many` in `platform.rs` guarantees that by the
//! `&[&[u8; N]]` type.

use crate::{BLOCK_LEN, CVWords, IV, IncrementCounter, MSG_SCHEDULE, OUT_LEN};
use core::arch::aarch64::*;

/// Inputs per dup-layout vector.
const PAIR: usize = 2;

#[inline(always)]
unsafe fn dup2(a: u32, b: u32) -> uint32x4_t {
    let da = a as u64 | (a as u64) << 32;
    let db = b as u64 | (b as u64) << 32;
    unsafe { vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(da), vcreate_u64(db))) }
}

#[inline(always)]
unsafe fn dup1(a: u32) -> uint32x4_t {
    unsafe { vdupq_n_u32(a) }
}

/// Sixteen little-endian message words from a 64-byte block of A and of B,
/// each word duplicated within its 64-bit lane.
#[inline(always)]
unsafe fn load_msg(a: *const u8, b: *const u8, m: &mut [uint32x4_t; 16]) {
    unsafe {
        for q in 0..4 {
            let va = vreinterpretq_u32_u8(vld1q_u8(a.add(16 * q)));
            let vb = vreinterpretq_u32_u8(vld1q_u8(b.add(16 * q)));
            let lo = vzip1q_u32(va, vb); // a0 b0 a1 b1
            let hi = vzip2q_u32(va, vb); // a2 b2 a3 b3
            m[4 * q] = vzip1q_u32(lo, lo); // a0 a0 b0 b0
            m[4 * q + 1] = vzip2q_u32(lo, lo);
            m[4 * q + 2] = vzip1q_u32(hi, hi);
            m[4 * q + 3] = vzip2q_u32(hi, hi);
        }
    }
}

#[inline(always)]
unsafe fn xar<const R: i32>(x: uint32x4_t, y: uint32x4_t) -> uint32x4_t {
    unsafe {
        vreinterpretq_u32_u64(vxarq_u64::<R>(
            vreinterpretq_u64_u32(x),
            vreinterpretq_u64_u32(y),
        ))
    }
}

/// One G function. The message add comes first so it overlaps the tail of
/// the previous G (`a` is ready before `b`).
macro_rules! g {
    ($v:ident, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {
        $v[$a] = vaddq_u32(vaddq_u32($v[$a], $mx), $v[$b]);
        $v[$d] = xar::<16>($v[$d], $v[$a]);
        $v[$c] = vaddq_u32($v[$c], $v[$d]);
        $v[$b] = xar::<12>($v[$b], $v[$c]);
        $v[$a] = vaddq_u32(vaddq_u32($v[$a], $my), $v[$b]);
        $v[$d] = xar::<8>($v[$d], $v[$a]);
        $v[$c] = vaddq_u32($v[$c], $v[$d]);
        $v[$b] = xar::<7>($v[$b], $v[$c]);
    };
}

macro_rules! round {
    ($v:ident, $m:ident, $r:expr) => {
        let s = &MSG_SCHEDULE[$r];
        g!($v, 0, 4, 8, 12, $m[s[0]], $m[s[1]]);
        g!($v, 1, 5, 9, 13, $m[s[2]], $m[s[3]]);
        g!($v, 2, 6, 10, 14, $m[s[4]], $m[s[5]]);
        g!($v, 3, 7, 11, 15, $m[s[6]], $m[s[7]]);
        g!($v, 0, 5, 10, 15, $m[s[8]], $m[s[9]]);
        g!($v, 1, 6, 11, 12, $m[s[10]], $m[s[11]]);
        g!($v, 2, 7, 8, 13, $m[s[12]], $m[s[13]]);
        g!($v, 3, 4, 9, 14, $m[s[14]], $m[s[15]]);
    };
}

/// Write the two chaining values held in `h` (A from the low lanes, B from
/// the high lanes) as 32 bytes each.
#[inline(always)]
unsafe fn store_cvs(h: &[uint32x4_t; 8], out: *mut u8) {
    unsafe {
        for i in [0usize, 4] {
            let c0 = vreinterpretq_u64_u32(h[i]);
            let c1 = vreinterpretq_u64_u32(h[i + 1]);
            let c2 = vreinterpretq_u64_u32(h[i + 2]);
            let c3 = vreinterpretq_u64_u32(h[i + 3]);
            let a01 = vreinterpretq_u32_u64(vzip1q_u64(c0, c1)); // A_i A_i A_i+1 A_i+1
            let a23 = vreinterpretq_u32_u64(vzip1q_u64(c2, c3));
            let b01 = vreinterpretq_u32_u64(vzip2q_u64(c0, c1));
            let b23 = vreinterpretq_u32_u64(vzip2q_u64(c2, c3));
            vst1q_u8(out.add(4 * i), vreinterpretq_u8_u32(vuzp1q_u32(a01, a23)));
            vst1q_u8(
                out.add(OUT_LEN + 4 * i),
                vreinterpretq_u8_u32(vuzp1q_u32(b01, b23)),
            );
        }
    }
}

/// The per-pair state that persists across blocks.
#[derive(Clone, Copy)]
struct Pair {
    h: [uint32x4_t; 8],
    ctr_lo: uint32x4_t,
    ctr_hi: uint32x4_t,
    a: *const u8,
    b: *const u8,
}

impl Pair {
    #[inline(always)]
    unsafe fn new(
        a: *const u8,
        b: *const u8,
        key: &CVWords,
        counter_a: u64,
        counter_b: u64,
    ) -> Self {
        unsafe {
            let mut h = [vdupq_n_u32(0); 8];
            for i in 0..8 {
                h[i] = dup1(key[i]);
            }
            Pair {
                h,
                ctr_lo: dup2(counter_a as u32, counter_b as u32),
                ctr_hi: dup2((counter_a >> 32) as u32, (counter_b >> 32) as u32),
                a,
                b,
            }
        }
    }

    #[inline(always)]
    unsafe fn begin_block(
        &self,
        block: usize,
        flags: u8,
        m: &mut [uint32x4_t; 16],
    ) -> [uint32x4_t; 16] {
        unsafe {
            load_msg(
                self.a.add(block * BLOCK_LEN),
                self.b.add(block * BLOCK_LEN),
                m,
            );
            [
                self.h[0],
                self.h[1],
                self.h[2],
                self.h[3],
                self.h[4],
                self.h[5],
                self.h[6],
                self.h[7],
                dup1(IV[0]),
                dup1(IV[1]),
                dup1(IV[2]),
                dup1(IV[3]),
                self.ctr_lo,
                self.ctr_hi,
                dup1(BLOCK_LEN as u32),
                dup1(flags as u32),
            ]
        }
    }

    #[inline(always)]
    unsafe fn end_block(&mut self, v: &[uint32x4_t; 16]) {
        unsafe {
            for i in 0..8 {
                self.h[i] = veorq_u32(v[i], v[i + 8]);
            }
        }
    }
}

/// Hash two inputs of `blocks` whole blocks each.
#[target_feature(enable = "neon,sha3")]
unsafe fn hash2(
    a: *const u8,
    b: *const u8,
    blocks: usize,
    key: &CVWords,
    counter_a: u64,
    counter_b: u64,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
) {
    unsafe {
        let mut p = Pair::new(a, b, key, counter_a, counter_b);
        let mut m = [vdupq_n_u32(0); 16];
        let mut block_flags = flags | flags_start;
        for block in 0..blocks {
            if block + 1 == blocks {
                block_flags |= flags_end;
            }
            let mut v = p.begin_block(block, block_flags, &mut m);
            round!(v, m, 0);
            round!(v, m, 1);
            round!(v, m, 2);
            round!(v, m, 3);
            round!(v, m, 4);
            round!(v, m, 5);
            round!(v, m, 6);
            p.end_block(&v);
            block_flags = flags;
        }
        store_cvs(&p.h, out);
    }
}

/// Hash four inputs of `blocks` whole blocks each, as two pairs whose rounds
/// are interleaved G by G so both chains are in flight together.
#[target_feature(enable = "neon,sha3")]
unsafe fn hash4(
    inputs: [*const u8; 4],
    blocks: usize,
    key: &CVWords,
    counters: [u64; 4],
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
) {
    unsafe {
        let mut p0 = Pair::new(inputs[0], inputs[1], key, counters[0], counters[1]);
        let mut p1 = Pair::new(inputs[2], inputs[3], key, counters[2], counters[3]);
        let mut m0 = [vdupq_n_u32(0); 16];
        let mut m1 = [vdupq_n_u32(0); 16];
        let mut block_flags = flags | flags_start;
        for block in 0..blocks {
            if block + 1 == blocks {
                block_flags |= flags_end;
            }
            let mut v0 = p0.begin_block(block, block_flags, &mut m0);
            let mut v1 = p1.begin_block(block, block_flags, &mut m1);
            // Alternate whole rounds between the pairs. Within one round the
            // four column Gs (then the four diagonal Gs) are independent, so
            // one pair fills the pipes on its own for a half-round; the
            // other pair's round follows while the first's results settle.
            // LLVM interleaves across the boundary as registers allow.
            macro_rules! round2 {
                ($r:expr) => {
                    round!(v0, m0, $r);
                    round!(v1, m1, $r);
                };
            }
            round2!(0);
            round2!(1);
            round2!(2);
            round2!(3);
            round2!(4);
            round2!(5);
            round2!(6);
            p0.end_block(&v0);
            p1.end_block(&v1);
            block_flags = flags;
        }
        store_cvs(&p0.h, out);
        store_cvs(&p1.h, out.add(PAIR * OUT_LEN));
    }
}

/// `hash_many` for two to fifteen inputs. Groups of four go to `hash4`, a
/// remaining pair to `hash2`, and a final single input to the portable
/// compressor, the same way upstream's NEON `hash_many` handles its tail.
///
/// Unsafe because the caller guarantees NEON and SHA-3 support.
pub unsafe fn hash_many<const N: usize>(
    inputs: &[&[u8; N]],
    key: &CVWords,
    counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8],
) {
    assert!(out.len() >= inputs.len() * OUT_LEN);
    let blocks = N / BLOCK_LEN;
    let step = if increment_counter.yes() { 1 } else { 0 };
    let mut i = 0;
    unsafe {
        while inputs.len() - i >= 4 {
            let c = counter + (i * step) as u64;
            hash4(
                [
                    inputs[i].as_ptr(),
                    inputs[i + 1].as_ptr(),
                    inputs[i + 2].as_ptr(),
                    inputs[i + 3].as_ptr(),
                ],
                blocks,
                key,
                [c, c + step as u64, c + 2 * step as u64, c + 3 * step as u64],
                flags,
                flags_start,
                flags_end,
                out[i * OUT_LEN..].as_mut_ptr(),
            );
            i += 4;
        }
        if inputs.len() - i >= 2 {
            let c = counter + (i * step) as u64;
            hash2(
                inputs[i].as_ptr(),
                inputs[i + 1].as_ptr(),
                blocks,
                key,
                c,
                c + step as u64,
                flags,
                flags_start,
                flags_end,
                out[i * OUT_LEN..].as_mut_ptr(),
            );
            i += 2;
        }
    }
    if i < inputs.len() {
        crate::portable::hash_many(
            &inputs[i..],
            key,
            counter + (i * step) as u64,
            increment_counter,
            flags,
            flags_start,
            flags_end,
            &mut out[i * OUT_LEN..],
        );
    }
}

/// True when the CPU has the SHA-3 extension that provides `xar`. Every
/// Apple M-series and Armv8.2+ core with `FEAT_SHA3` does.
pub fn sha3_detected() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_aarch64_feature_detected!("sha3")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "sha3")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hash_many() {
        if !sha3_detected() {
            return;
        }
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }
}
