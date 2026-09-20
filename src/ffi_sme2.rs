//! AArch64 SME2 implementation, 512-bit streaming vector length (Apple M4
//! and later). The kernels live in c/blake3_sme2_aarch64.S.
//!
//! The assembly handles exactly one shape: groups of sixteen whole inputs,
//! either 1024-byte chunks (sixteen blocks each, counter incrementing) or
//! 64-byte parent blocks (one block each, counter fixed). This wrapper feeds
//! it every full group and hands anything else to NEON: a remainder of
//! fewer than sixteen inputs, chunk hashing with `IncrementCounter::No`, or
//! parent hashing with `IncrementCounter::Yes`.
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
                let mut buf = [0u8; GROUP * BLOCK_LEN];
                for group in 0..full_groups {
                    for (i, input) in inputs[group * GROUP..][..GROUP].iter().enumerate() {
                        buf[i * BLOCK_LEN..][..BLOCK_LEN].copy_from_slice(&input[..]);
                    }
                    let lanes = unsafe {
                        kernel(
                            buf.as_ptr(),
                            key.as_ptr(),
                            counter,
                            packed_flags,
                            out[group * GROUP * OUT_LEN..].as_mut_ptr(),
                            1,
                        )
                    };
                    assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
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
}
