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

/// Fewest one-block inputs left over after the SME2 groups that go to
/// one more, overlapping, SME2 group rather than the NEON kernels: from 13
/// (p9 + p4 and up), NEON runs past the SME unit's idle threshold. Below it
/// the NEON kernels are cheaper than a second streaming session (M4 Max,
/// probe/neon-cold job 129: 15 left over, overlap -19 to -28% on P-cores;
/// 1 to 12 left over, +2 to +69% where the SME unit stayed fast).
const OVERLAP_MIN: usize = 13;

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

    // One-block inputs after SME2 groups, with OVERLAP_MIN or more left over:
    // the last sixteen as one more group, overlapping the one before, and
    // only the remainder's values kept. The NEON kernels for that many run
    // past the idle time (about a quarter of a microsecond) after which the
    // SME unit drops to a state about 20% slower that lasts tens of
    // microseconds, so the next SME2 work, this call's or the next one's,
    // runs slow (NOTES-servil.md, "SME2 remainders").
    let rest = inputs.len() - done;
    if N == BLOCK_LEN && !increment_counter.yes() && done > 0 && rest >= OVERLAP_MIN {
        let mut last = [0u8; GROUP * OUT_LEN];
        unsafe {
            hash_many(&inputs[inputs.len() - GROUP..], key, counter, increment_counter, flags, flags_start, flags_end, &mut last);
        }
        out[done * OUT_LEN..inputs.len() * OUT_LEN].copy_from_slice(&last[(GROUP - rest) * OUT_LEN..]);
        return;
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

/// Whole subtrees from this many chunks to FLAT_MAX_CHUNKS, powers of two,
/// take the flat walk ([`compress_subtree_flat`]): 32 chunks are the
/// fewest whose first parent level fills a group of 16.
pub const FLAT_MIN_CHUNKS: usize = 32;
/// The fewest chunks, a power of two, that split exactly into groups of 18
/// (the integer lane's kernel) and 16: 256 = 8 x 18 + 7 x 16. Smaller
/// subtrees take groups of 16 alone.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub const LANE_MIN_CHUNKS: usize = 256;
/// The most chunks the flat walk takes; its chaining values sit in 48 KiB
/// of stack. Larger subtrees are split by the tree walk above it.
pub const FLAT_MAX_CHUNKS: usize = 1024;

/// Whether [`compress_subtree_flat`] takes an input of `len` bytes.
#[inline]
pub fn flat_takes(len: usize) -> bool {
    len.is_power_of_two() && (FLAT_MIN_CHUNKS * CHUNK_LEN..=FLAT_MAX_CHUNKS * CHUNK_LEN).contains(&len)
}

/*
 * A whole subtree of 32 to 1024 chunks, bottom up, on SME2 alone: what
 * `compress_subtree_wide` returns for it, its DEGREE chaining values
 * (each covering n / DEGREE chunks). The chunks go to the kernel with an
 * integer lane in groups of 18 (8 of them per 144 chunks), the rest to the
 * plain kernel in groups of 16, with nothing left over (n is a multiple of
 * 16); then each parent level from n / 2 down to DEGREE values goes to the
 * parent kernel, every level a multiple of 16. No NEON work falls between
 * SME2 kernels, so the SME unit never idles on our account (NOTES-servil.md,
 * "SME2 remainders"). Out of line: its 56 KiB of stack arrays belong to
 * large inputs alone. Unsafe because the CPU must have SME2 with 512-bit
 * streaming vectors; `flat_takes(input.len())` must hold, and `out` must
 * have room for DEGREE values.
 */
#[inline(never)]
pub unsafe fn compress_subtree_flat(
    input: &[u8],
    key: &CVWords,
    chunk_counter: u64,
    flags: u8,
    out: &mut [u8],
) -> usize {
    unsafe { flat_walk(input, key, chunk_counter, flags, DEGREE, out) }
}

/// The flat walk down to the subtree's two children: what
/// `compress_subtree_to_parent_node` returns for it. The levels below
/// sixteen parents run as one padded SME2 group each (lanes past the
/// level's parents compress bytes nobody reads), so no NEON or scalar work
/// falls between the SME2 kernels: 0.25 µs of other work between them puts
/// the SME unit in its slow state for the next ones (NOTES-servil.md,
/// "SME2 remainders"). Unsafe for the reasons of [`compress_subtree_flat`].
#[inline(never)]
pub unsafe fn compress_subtree_flat_to_parent(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8) -> [u8; crate::BLOCK_LEN] {
    let mut out = [0u8; crate::BLOCK_LEN];
    unsafe { flat_walk(input, key, chunk_counter, flags, 2, &mut out) };
    out
}

/// The flat walk down to the subtree's own chaining value, for a subtree
/// that cannot be the root (input comes before it). Unsafe for the
/// reasons of [`compress_subtree_flat`].
#[inline(never)]
pub unsafe fn compress_subtree_flat_to_cv(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8) -> crate::CVBytes {
    let mut out = [0u8; OUT_LEN];
    unsafe { flat_walk(input, key, chunk_counter, flags, 1, &mut out) };
    out
}

/// The flat walk until `keep` chaining values remain (DEGREE, 2, or 1),
/// which it copies into `out`.
#[inline(always)]
unsafe fn flat_walk(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8, keep: usize, out: &mut [u8]) -> usize {
    assert!(flat_takes(input.len()), "a whole subtree of 32 to 1024 chunks");
    assert!(keep == DEGREE || keep == 2 || keep == 1, "the flat walk keeps DEGREE, 2, or 1 chaining values");
    assert!(out.len() >= keep * OUT_LEN, "room for the chaining values kept");
    assert!(keep < input.len() / CHUNK_LEN, "the flat walk keeps fewer chaining values than its subtree has chunks");
    let n = input.len() / CHUNK_LEN;
    #[cfg(feature = "std")]
    if let Ok(base) = SCRATCH_BLOCK.try_with(|block| block.get() as *mut u8) {
        // Sound: this thread's block outlives the call, and the walk calls
        // nothing that re-enters it, so the block has one user at a time.
        return unsafe { walk_in(input, key, chunk_counter, flags, keep, out, n, base) };
    }
    // Without std, or in a thread's teardown after its block is gone.
    unsafe { walk_on_stack(input, key, chunk_counter, flags, keep, out, n) }
}

/// [`walk_in`] with the scratch on the stack, out of line so the usual
/// path keeps a small frame.
#[inline(never)]
unsafe fn walk_on_stack(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8, keep: usize, out: &mut [u8], n: usize) -> usize {
    let mut block = Scratch([core::mem::MaybeUninit::<u8>::uninit(); SCRATCH]);
    unsafe { walk_in(input, key, chunk_counter, flags, keep, out, n, block.0.as_mut_ptr() as *mut u8) }
}

/// The flat walk's scratch, on 128-byte lines (Apple's).
#[repr(C, align(128))]
struct Scratch([core::mem::MaybeUninit<u8>; SCRATCH]);

// Each thread's scratch for the flat walk, allocated on its first walk.
// Off the stack: a 64 KiB stack frame is probed with a store into each
// page on every call, and those stores, landing among the buffers the SME
// unit is about to use, put it in its slow state at some stack depths
// (M4 Max, hash(32 KiB) 0.178-0.245 ns/B by depth with the buffers on the
// stack, 0.177 at every depth off it; probe/stack-map, jobs 169-171).
#[cfg(feature = "std")]
std::thread_local! {
    static SCRATCH_BLOCK: Box<core::cell::UnsafeCell<Scratch>> =
        Box::new(core::cell::UnsafeCell::new(Scratch([core::mem::MaybeUninit::uninit(); SCRATCH])));
}

/// The flat walk in `block`, SCRATCH bytes on 128-byte lines, which
/// holds the pointer table at its start and the chaining values where
/// [`scratch_layout`] puts them: 1 KiB and 3 KiB mod 4 KiB, so the three
/// buffers never share an address mod 4 KiB. The kernels' 32-byte stores
/// also want 32-byte boundaries (VM, up to 21% slower off them).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn walk_in(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8, keep: usize, out: &mut [u8], n: usize, base: *mut u8) -> usize {
    let (cvs_at, half_at) = scratch_layout(n);
    let table = base as *mut *const u8;
    for i in 0..n {
        unsafe { table.add(i).write(input.as_ptr().add(i * CHUNK_LEN)) };
    }
    let table = table as *const *const u8;
    let (mut src, mut dst) = unsafe { (base.add(cvs_at), base.add(half_at)) };

    let chunk_flags = flags as u32 | (crate::CHUNK_START as u32) << 8 | (crate::CHUNK_END as u32) << 16;
    let lane_groups = 8 * (n / 144);
    let plain_groups = (n - HYBRID_GROUP * lane_groups) / GROUP;
    debug_assert_eq!(HYBRID_GROUP * lane_groups + GROUP * plain_groups, n);
    unsafe {
        if lane_groups > 0 {
            let lanes = ffi::blake3_sme2x2_hash_chunks_512(table, key.as_ptr(), chunk_counter, chunk_flags, src, lane_groups as u64);
            assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        }
        let done = HYBRID_GROUP * lane_groups;
        let lanes = ffi::blake3_sme2_hash16_chunks_512(
            table.add(done) as *const u8,
            key.as_ptr(),
            chunk_counter + done as u64,
            chunk_flags,
            src.add(done * OUT_LEN),
            plain_groups as u64,
        );
        assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        // Parent levels: pairs of adjacent chaining values, contiguous, from
        // `src` into `dst`, until `keep` values remain. A level of fewer than
        // sixteen parents reads 32 values from `src` and writes 16 into
        // `dst`, within the room scratch_layout leaves.
        let mut count = n;
        while count > keep {
            let parents = count / 2;
            let groups = parents.div_ceil(GROUP);
            let lanes = ffi::blake3_sme2_hash16_parents_512(src, key.as_ptr(), 0, (flags | crate::PARENT) as u32, dst, groups as u64);
            assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
            count = parents;
            core::mem::swap(&mut src, &mut dst);
        }
        core::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), keep * OUT_LEN);
    }
    keep
}

/// Bytes of the flat walk's scratch: the pointer table, the chaining
/// values, and the level above them for FLAT_MAX_CHUNKS, as
/// [`scratch_layout`] places them.
const SCRATCH: usize = 64 * 1024;

/// Where the flat walk of `n` chunks keeps its chaining values and the
/// level above them, in bytes from the scratch's start (the pointer table
/// sits at 0): at 1 KiB and 3 KiB mod 4 KiB, past the table and the
/// values before them, with room for a padded group of 16 parents' inputs
/// (32 values) at either.
fn scratch_layout(n: usize) -> (usize, usize) {
    let cvs = (8 * n).next_multiple_of(4096) + 1024;
    let half = cvs + (32 * n.max(32)).saturating_sub(2048).next_multiple_of(4096) + 2048;
    debug_assert!(half + 16 * n.max(64) <= SCRATCH, "the flat walk's scratch holds its buffers");
    (cvs, half)
}

/// Chunks per group of the kernel with an integer lane: sixteen on SME2,
/// two on the integer units (c/blake3_sme2_hybrid_aarch64.S).
pub const HYBRID_GROUP: usize = 18;

pub mod ffi {
    unsafe extern "C" {
        /// Eighteen whole chunks per group, sixteen on the SME2 lanes and
        /// two on the integer units beside them (generated by
        /// tools/gen_sme2_hybrid.py). `inputs` is a table of `18 * groups`
        /// chunk pointers; otherwise as `blake3_sme2_hash16_chunks_512`.
        pub fn blake3_sme2x2_hash_chunks_512(
            inputs: *const *const u8,
            key: *const u32,
            counter: u64,
            flags: u32,
            out: *mut u8,
            groups: u64,
        ) -> u64;

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

    /// The flat walk against the reference implementation, at every size
    /// it takes and just around them (the tree walk above it, then the
    /// flat walk for the power-of-two subtrees within), in hash, keyed
    /// hash, and a Hasher fed from an unaligned start.
    #[test]
    fn test_flat_walk_matches_reference() {
        if !crate::platform::sme2_detected() {
            return;
        }
        let mut input = vec![0u8; 3 * FLAT_MAX_CHUNKS * CHUNK_LEN + 12345];
        crate::test::paint_test_input(&mut input);
        let key = [7u8; crate::KEY_LEN];
        let reference = |data: &[u8], keyed: bool| {
            let mut hasher = if keyed { reference_impl::Hasher::new_keyed(&key) } else { reference_impl::Hasher::new() };
            hasher.update(data);
            let mut out = [0u8; 32];
            hasher.finalize(&mut out);
            out
        };
        for chunks in [FLAT_MIN_CHUNKS - 1, FLAT_MIN_CHUNKS, FLAT_MIN_CHUNKS + 1, 2 * FLAT_MIN_CHUNKS, LANE_MIN_CHUNKS / 2, LANE_MIN_CHUNKS - 1, LANE_MIN_CHUNKS, 2 * LANE_MIN_CHUNKS, FLAT_MAX_CHUNKS, FLAT_MAX_CHUNKS + 1, 2 * FLAT_MAX_CHUNKS, 3 * FLAT_MAX_CHUNKS] {
            for extra in [0, 1, 1000] {
                let data = &input[..chunks * CHUNK_LEN + extra];
                assert_eq!(*crate::hash(data).as_bytes(), reference(data, false), "{chunks} chunks + {extra}");
                assert_eq!(*crate::keyed_hash(&key, data).as_bytes(), reference(data, true), "keyed, {chunks} chunks + {extra}");
                let mut hasher = crate::Hasher::new();
                hasher.update(&data[..3 * CHUNK_LEN]);
                hasher.update(&data[3 * CHUNK_LEN..]);
                assert_eq!(*hasher.finalize().as_bytes(), reference(data, false), "Hasher, {chunks} chunks + {extra}");
            }
        }
    }

    /// The walk's stack fallback (no std, or a thread's teardown) returns
    /// what the per-thread scratch does, at every size the walk takes.
    #[test]
    fn test_flat_walk_on_stack_matches_scratch() {
        if !crate::platform::sme2_detected() {
            return;
        }
        let mut input = vec![0u8; FLAT_MAX_CHUNKS * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        let key = [3u32; 8];
        let mut chunks = FLAT_MIN_CHUNKS;
        while chunks <= FLAT_MAX_CHUNKS {
            let data = &input[..chunks * CHUNK_LEN];
            for keep in [1, 2, DEGREE].into_iter().filter(|&keep| keep < chunks) {
                let (mut a, mut b) = ([0u8; DEGREE * OUT_LEN], [0u8; DEGREE * OUT_LEN]);
                unsafe {
                    flat_walk(data, &key, 7 * chunks as u64, 0, keep, &mut a);
                    walk_on_stack(data, &key, 7 * chunks as u64, 0, keep, &mut b, chunks);
                }
                assert_eq!(a, b, "{chunks} chunks, keeping {keep}");
            }
            chunks *= 2;
        }
    }

    #[test]
    fn test_hybrid_assembly_matches_generator() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = match std::process::Command::new("python3").arg(root.join("tools/gen_sme2_hybrid.py")).output() {
            Ok(output) if output.status.success() => output.stdout,
            _ => return,
        };
        let committed = std::fs::read(root.join("c/blake3_sme2_hybrid_aarch64.S")).unwrap();
        assert!(output == committed, "c/blake3_sme2_hybrid_aarch64.S is stale; run tools/gen_sme2_hybrid.py");
    }

    #[test]
    fn test_hash_many() {
        if !crate::platform::sme2_detected() {
            return;
        }
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }
}
