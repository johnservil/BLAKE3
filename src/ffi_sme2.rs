//! AArch64 SME2 implementation, 512-bit streaming vector length (Apple M4
//! and later). The kernels live in c/blake3_sme2_aarch64.S.
//!
//! The assembly handles exactly one shape: groups of sixteen whole inputs,
//! either 1024-byte chunks (sixteen blocks each, counter incrementing) or
//! 64-byte parent blocks (one block each, counter fixed). This wrapper feeds
//! it every full group and hands anything else to the NEON backend: a
//! remainder of fewer than sixteen inputs, chunk hashing with
//! `IncrementCounter::No`, or parent hashing with `IncrementCounter::Yes`.
//!
//! The kernels return the streaming vector length in 32-bit lanes and do
//! work only when it is 16. `Platform::detect()` checks that once and
//! selects NEON otherwise, so a return value other than 16 here is a
//! contract violation and stops the program.

use crate::{BLOCK_LEN, CHUNK_LEN, CVWords, IncrementCounter, OUT_LEN};

/// Lanes per kernel group: the hardware's real SIMD width.
pub const GROUP: usize = 16;

/// The degree `Platform::SME2` reports. Every `hash_many` call carries up to
/// this many inputs, so the kernel hashes `DEGREE / GROUP` groups per entry
/// into streaming mode. Entering and leaving streaming mode costs about half
/// a microsecond on Apple M4, more than a whole group of parent compressions
/// (~170 ns); eight groups per call amortize it. Measured single-thread
/// `blake3::hash` on a 64 MiB input: degree 16 → 226 ps/B, 64 → 182, 128 → 167
/// (NEON: 424). 128 also keeps `compress_subtree_wide`'s stack arrays at 8 KiB.
pub const DEGREE: usize = 128;

/// Chunk-group and parent-group entry points share one signature.
type Kernel = unsafe extern "C" fn(*const u8, *const u32, u64, u32, *mut u8, u64) -> u64;

// Unsafe because this may only be called on CPUs that report SME2 with a
// 512-bit streaming vector length, and on which NEON is also available.
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

    let full_groups = inputs.len() / GROUP;
    // The chunk kernel takes flags, flags_start, and flags_end packed into
    // one word; the parent kernel applies flags | flags_start | flags_end to
    // its single block, which is what hash_many means for one-block inputs.
    let (kernel, packed_flags): (Option<Kernel>, u32) = if N == CHUNK_LEN && increment_counter.yes()
    {
        (
            Some(ffi::blake3_sme2_hash16_chunks_512),
            flags as u32 | (flags_start as u32) << 8 | (flags_end as u32) << 16,
        )
    } else if N == BLOCK_LEN && !increment_counter.yes() {
        (
            Some(ffi::blake3_sme2_hash16_parents_512),
            (flags | flags_start | flags_end) as u32,
        )
    } else {
        (None, 0)
    };

    let mut done = 0;
    if let (Some(kernel), true) = (kernel, full_groups > 0) {
        let count = full_groups * GROUP;
        let lanes = if N == BLOCK_LEN {
            // The parent kernel reads its sixteen 64-byte blocks from one
            // contiguous run. Every caller in this crate already stores
            // parent inputs that way; anyone else gets a copy.
            let base = inputs[0].as_ptr();
            let contiguous = inputs[..count]
                .iter()
                .enumerate()
                .all(|(i, input)| input.as_ptr() == unsafe { base.add(i * BLOCK_LEN) });
            if contiguous {
                unsafe {
                    kernel(
                        base,
                        key.as_ptr(),
                        counter,
                        packed_flags,
                        out.as_mut_ptr(),
                        full_groups as u64,
                    )
                }
            } else {
                // Gather up to GATHER groups per kernel call, so one entry
                // into streaming mode covers them all.
                const GATHER: usize = DEGREE / GROUP;
                let mut buf = [0u8; GATHER * GROUP * BLOCK_LEN];
                let mut group = 0;
                while group < full_groups {
                    let groups = (full_groups - group).min(GATHER);
                    for (i, input) in inputs[group * GROUP..][..groups * GROUP].iter().enumerate() {
                        buf[i * BLOCK_LEN..][..BLOCK_LEN].copy_from_slice(&input[..]);
                    }
                    let lanes = unsafe {
                        kernel(
                            buf.as_ptr(),
                            key.as_ptr(),
                            counter,
                            packed_flags,
                            out[group * GROUP * OUT_LEN..].as_mut_ptr(),
                            groups as u64,
                        )
                    };
                    assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
                    group += groups;
                }
                16
            }
        } else {
            unsafe {
                kernel(
                    inputs.as_ptr() as *const u8,
                    key.as_ptr(),
                    counter,
                    packed_flags,
                    out.as_mut_ptr(),
                    full_groups as u64,
                )
            }
        };
        assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        done = count;
    }

    if done < inputs.len() {
        let rest_counter = if increment_counter.yes() {
            counter + done as u64
        } else {
            counter
        };
        // The remainder is fewer than sixteen inputs: the NEON kernels
        // (integer + vector hybrids, see neon_hybrid.rs) are the fastest
        // path there. They need the SHA-3 extension; the C kernel is the
        // fallback without it.
        if crate::neon_hybrid::sha3_detected() {
            unsafe {
                crate::neon_hybrid::hash_many(
                    &inputs[done..],
                    key,
                    rest_counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    &mut out[done * OUT_LEN..],
                );
            }
            return;
        }
        unsafe {
            crate::neon::hash_many(
                &inputs[done..],
                key,
                rest_counter,
                increment_counter,
                flags,
                flags_start,
                flags_end,
                &mut out[done * OUT_LEN..],
            );
        }
    }
}

/*
 * A whole subtree on the SME2 kernels alone.
 *
 * On the machines measured, the first NEON instruction after an SME2
 * kernel returns costs microseconds (about 4 on a 16-vCPU VM, about 1 on
 * an M4), while scalar code costs nothing extra; see
 * examples/transition.rs. A thread that hashes with SME2 and never runs a
 * NEON instruction never pays it. `subtree_cv` hashes a subtree that way:
 *
 * - whole chunks through the chunk kernel, a short group's spare lanes
 *   pointed at a static zero chunk and their outputs ignored;
 * - a final partial chunk through the scalar kernel c1 (integer
 *   registers only);
 * - every parent level through the parent kernel, a short level run as a
 *   full group over the padded buffer, an odd child carried up with
 *   scalar loads and stores.
 *
 * The glue between kernel calls stays scalar: buffers live in a `Scratch`
 * zeroed once, when the thread starts; copies go through `copy_cv`, table
 * entries through volatile stores the compiler cannot turn into vector
 * code.
 */

/// The longest subtree `subtree_cv` takes: 128 chunks, eight groups, one
/// chunk-kernel call.
pub const SUBTREE_MAX: usize = DEGREE * CHUNK_LEN;

/// CV slots per buffer: 128 chunks and a partial one, rounded up so a
/// padded parent level (sixteen pairs per group) stays inside.
const CVS: usize = 160;

/// A thread's working memory for `subtree_cv`. Create it before the
/// thread first runs an SME2 kernel: creation zeroes it with whatever
/// instructions the compiler picks.
#[repr(C, align(64))]
pub struct Scratch {
    table: [*const u8; DEGREE],
    a: [[u8; OUT_LEN]; CVS],
    b: [[u8; OUT_LEN]; CVS],
    last: [u8; BLOCK_LEN],
}

impl Scratch {
    pub fn new() -> Box<Self> {
        Box::new(Scratch {
            table: [core::ptr::null(); DEGREE],
            a: [[0; OUT_LEN]; CVS],
            b: [[0; OUT_LEN]; CVS],
            last: [0; BLOCK_LEN],
        })
    }
}

static ZERO_CHUNK: [u8; CHUNK_LEN] = [0; CHUNK_LEN];

/// 32 bytes through four integer registers.
#[inline(always)]
unsafe fn copy_cv(src: *const u8, dst: *mut u8) {
    unsafe {
        core::arch::asm!(
            "ldp {a}, {b}, [{s}]",
            "ldp {c}, {d}, [{s}, #16]",
            "stp {a}, {b}, [{t}]",
            "stp {c}, {d}, [{t}, #16]",
            s = in(reg) src,
            t = in(reg) dst,
            a = out(reg) _,
            b = out(reg) _,
            c = out(reg) _,
            d = out(reg) _,
            options(nostack, preserves_flags),
        );
    }
}

/// The chaining value of `input`, a whole non-root subtree starting at
/// chunk `counter`, into `out`, on the SME2 kernels and scalar code alone.
/// `input` holds 1 to SUBTREE_MAX bytes.
///
/// Unsafe because the CPU must report SME2 with a 512-bit streaming
/// vector length, and `out` must be valid for 32 bytes of writes.
pub unsafe fn subtree_cv(input: &[u8], key: &CVWords, counter: u64, flags: u8, scratch: &mut Scratch, out: *mut u8) {
    assert!(!input.is_empty() && input.len() <= SUBTREE_MAX, "a subtree of 1 to {SUBTREE_MAX} bytes");
    let full = input.len() / CHUNK_LEN;
    let rem = input.len() % CHUNK_LEN;
    let (mut src, mut dst) = (scratch.a.as_mut_ptr() as *mut u8, scratch.b.as_mut_ptr() as *mut u8);
    let mut count = 0;
    if full > 0 {
        let groups = full.div_ceil(GROUP);
        for i in 0..groups * GROUP {
            let lane = if i < full { unsafe { input.as_ptr().add(i * CHUNK_LEN) } } else { ZERO_CHUNK.as_ptr() };
            unsafe { core::ptr::write_volatile(&mut scratch.table[i], lane) };
        }
        let packed = flags as u32 | (crate::CHUNK_START as u32) << 8 | (crate::CHUNK_END as u32) << 16;
        let lanes = unsafe {
            ffi::blake3_sme2_hash16_chunks_512(scratch.table.as_ptr() as *const u8, key.as_ptr(), counter, packed, src, groups as u64)
        };
        assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        count = full;
    }
    if rem > 0 {
        let blocks = rem.div_ceil(BLOCK_LEN);
        let whole = (blocks - 1) * BLOCK_LEN;
        let tail = rem - whole;
        let base = unsafe { input.as_ptr().add(full * CHUNK_LEN) };
        let last: *const [u8; BLOCK_LEN] = if tail == BLOCK_LEN {
            unsafe { base.add(whole) as *const [u8; BLOCK_LEN] }
        } else {
            for i in 0..BLOCK_LEN {
                let byte = if i < tail { unsafe { core::ptr::read_volatile(base.add(whole + i)) } } else { 0 };
                unsafe { core::ptr::write_volatile(&mut scratch.last[i], byte) };
            }
            &scratch.last
        };
        unsafe {
            crate::neon_hybrid::hash_chunk(
                base,
                blocks,
                &*last,
                tail as u8,
                key,
                counter + full as u64,
                flags,
                crate::CHUNK_START,
                crate::CHUNK_END,
                &mut *(src.add(count * OUT_LEN) as *mut [u8; OUT_LEN]),
            );
        }
        count += 1;
    }
    while count > 1 {
        let pairs = count / 2;
        let groups = pairs.div_ceil(GROUP);
        let lanes = unsafe {
            ffi::blake3_sme2_hash16_parents_512(src, key.as_ptr(), 0, (flags | crate::PARENT) as u32, dst, groups as u64)
        };
        assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        if count % 2 == 1 {
            unsafe { copy_cv(src.add((count - 1) * OUT_LEN), dst.add(pairs * OUT_LEN)) };
        }
        count = pairs + count % 2;
        core::mem::swap(&mut src, &mut dst);
    }
    unsafe { copy_cv(src, out) };
}

pub mod ffi {
    unsafe extern "C" {
        /// Sixteen whole 1024-byte chunks per group. `inputs` is a table of
        /// `16 * groups` chunk pointers; `counter` applies to the first chunk
        /// and increments per chunk; `flags` packs `flags | flags_start << 8
        /// | flags_end << 16`. Writes `16 * groups` 32-byte chaining values
        /// to `out`. Returns the streaming vector length in 32-bit lanes.
        pub fn blake3_sme2_hash16_chunks_512(
            inputs: *const u8,
            key: *const u32,
            counter: u64,
            flags: u32,
            out: *mut u8,
            groups: u64,
        ) -> u64;

        /// Sixteen parent compressions per group. `pairs` points at
        /// `16 * groups` adjacent 64-byte parent blocks; `counter` is used
        /// unchanged for every block; `flags` are the parent flags. Writes
        /// `16 * groups` 32-byte chaining values to `out`. Returns the
        /// streaming vector length in 32-bit lanes.
        pub fn blake3_sme2_hash16_parents_512(
            pairs: *const u8,
            key: *const u32,
            counter: u64,
            flags: u32,
            out: *mut u8,
            groups: u64,
        ) -> u64;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hash_many() {
        if !crate::platform::sme2_detected() {
            return;
        }
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }

    /// The SME2-only subtree against the ordinary subtree code, for every
    /// chunk count up to 128 whole or with a partial chunk, at counters
    /// where such a subtree is valid.
    #[test]
    fn test_subtree_cv_matches() {
        if !crate::platform::sme2_detected() {
            return;
        }
        let mut input = vec![0u8; SUBTREE_MAX];
        crate::test::paint_test_input(&mut input);
        let mut scratch = Scratch::new();
        let key = *crate::IV;
        let mut lens: Vec<usize> = (1..=128).map(|c| c * CHUNK_LEN).collect();
        lens.extend([1, 63, 64, 65, 1023, 1025, 2047, 3 * CHUNK_LEN + 700, 17 * CHUNK_LEN + 1, 100 * CHUNK_LEN + 64, SUBTREE_MAX - 1]);
        for len in lens {
            // A counter that is a multiple of the subtree's power-of-two
            // span keeps it a whole subtree at that position.
            let span = len.div_ceil(CHUNK_LEN).next_power_of_two() as u64;
            for counter in [0, span, 5 * span] {
                let want = crate::hash_all_at_once::<crate::join::SerialJoin>(
                    &input[..len], &key, counter, crate::KEYED_HASH, crate::platform::Platform::NEON,
                ).chaining_value();
                let mut got = [0u8; OUT_LEN];
                unsafe { subtree_cv(&input[..len], &key, counter, crate::KEYED_HASH, &mut scratch, got.as_mut_ptr()) };
                assert_eq!(got, want, "len {len}, counter {counter}");
            }
        }
    }
}
