//! `hash_many` for AArch64 with NEON and the SHA-3 extension, built from the
//! kernels in c/blake3_neon_hybrid_aarch64.S.
//!
//! Each kernel hashes a fixed number of inputs laid out back to back. Chunk
//! kernels (`k<n>`) take whole 1024-byte chunks with a counter that
//! increments per chunk; parent kernels (`p<n>`) take 64-byte blocks with
//! one shared counter. This wrapper splits an input count into kernel calls
//! that keep the integer and vector units busy together:
//!
//! | inputs | chunk kernels | parent kernels |
//! |--------|---------------|----------------|
//! | 1      | k1            | k1             |
//! | 2      | k2            | p2             |
//! | 3      | k3            | p2 + k1        |
//! | 4      | k4            | p4             |
//! | 5      | k5            | p4 + k1        |
//! | 6      | k6            | p4 + p2        |
//! | 7      | k4 + k3       | p4 + p2 + k1   |
//! | 8      | k8            | p8             |
//! | 9      | k9            | p8 + k1        |
//! | 10     | k10           | p8 + p2        |
//! | 11     | k8 + k3       | p8 + p2 + k1   |
//! | 12     | k8 + k4       | p8 + p4        |
//! | 13     | k8 + k5       | p8 + p4 + k1   |
//! | 14     | k8 + k6       | p8 + p4 + p2   |
//! | 15     | k9 + k6       | p8 + p4 + p2 + k1 |
//!
//! Longer input lists are hashed fifteen at a time. Every caller in this
//! crate passes inputs that sit back to back in memory. Scattered parents
//! are copied into a stack buffer; scattered chunks go one per kernel call.

use crate::{BLOCK_LEN, CHUNK_LEN, CVWords, IncrementCounter, OUT_LEN};

/// The kernels live in c/blake3_neon_hybrid_aarch64.S. Every one has the
/// signature `(base, blocks, key, counter, packed_flags, out)`; the contract
/// is in that file's header.
mod asm {
    unsafe extern "C" {
        pub fn blake3_hybrid_k1(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k2(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k3(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k4(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k5(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k6(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k8(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k9(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k10(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p2(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p4(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p8(
            base: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
    }
}

/// Every kernel shares one signature: `(base, blocks, key, counter,
/// packed_flags, out)`. See the contract in `neon_hybrid_asm.rs`.
type Kernel = unsafe extern "C" fn(*const u8, u64, *const u32, u64, u64, *mut u8);

/// Most inputs one `hash_many` call hands to the kernels at once.
const GROUP: usize = 15;

/// Chunk kernel per exact input count.
const CHUNK_KERNELS: [Option<Kernel>; 11] = [
    None,
    Some(asm::blake3_hybrid_k1),
    Some(asm::blake3_hybrid_k2),
    Some(asm::blake3_hybrid_k3),
    Some(asm::blake3_hybrid_k4),
    Some(asm::blake3_hybrid_k5),
    Some(asm::blake3_hybrid_k6),
    None,
    Some(asm::blake3_hybrid_k8),
    Some(asm::blake3_hybrid_k9),
    Some(asm::blake3_hybrid_k10),
];

/// Kernel sizes per chunk count 1..=15, largest first.
const CHUNK_PLANS: [&[usize]; 16] = [
    &[],
    &[1],
    &[2],
    &[3],
    &[4],
    &[5],
    &[6],
    &[4, 3],
    &[8],
    &[9],
    &[10],
    &[8, 3],
    &[8, 4],
    &[8, 5],
    &[8, 6],
    &[9, 6],
];

/// Parent kernel per exact input count.
const PARENT_KERNELS: [Option<Kernel>; 9] = [
    None,
    Some(asm::blake3_hybrid_k1),
    Some(asm::blake3_hybrid_p2),
    None,
    Some(asm::blake3_hybrid_p4),
    None,
    None,
    None,
    Some(asm::blake3_hybrid_p8),
];

/// Kernel sizes per parent count 1..=15: the binary decomposition.
const PARENT_PLANS: [&[usize]; 16] = [
    &[],
    &[1],
    &[2],
    &[2, 1],
    &[4],
    &[4, 1],
    &[4, 2],
    &[4, 2, 1],
    &[8],
    &[8, 1],
    &[8, 2],
    &[8, 2, 1],
    &[8, 4],
    &[8, 4, 1],
    &[8, 4, 2],
    &[8, 4, 2, 1],
];

/// True when the CPU has the SHA-3 extension that provides `xar`. Every
/// Apple M-series core and every Armv8.2+ core with `FEAT_SHA3` does. The
/// scalar kernel `k1` needs no extension; it is the pair and quad kernels
/// that rotate with `xar`.
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

/// Compress `count` whole 64-byte blocks of one chunk into `cv`, all with
/// the same counter: `flags | flags_start` on the first block, `flags` on
/// the rest. This is `ChunkState::update`'s inner loop as one kernel call;
/// the scalar kernel keeps the whole state in registers and runs at 13.3
/// cycles per G-step against the portable compressor's 14.
///
/// The kernel uses only integer instructions. Unsafe because `blocks` must
/// hold `count * 64` readable bytes and `count` must be in 1..=16.
pub unsafe fn compress_blocks(
    cv: &mut CVWords,
    blocks: *const u8,
    count: usize,
    counter: u64,
    flags: u8,
    flags_start: u8,
) {
    debug_assert!((1..=16).contains(&count));
    let packed = flags as u64 | (flags_start as u64) << 8;
    let mut out = [0u8; OUT_LEN];
    unsafe {
        asm::blake3_hybrid_k1(
            blocks,
            count as u64,
            cv.as_ptr(),
            counter,
            packed,
            out.as_mut_ptr(),
        );
    }
    *cv = crate::platform::words_from_le_bytes_32(&out);
}

/// True when `inputs` are laid out back to back in memory.
fn contiguous<const N: usize>(inputs: &[&[u8; N]]) -> bool {
    let base = inputs[0].as_ptr();
    inputs
        .iter()
        .enumerate()
        .all(|(i, input)| input.as_ptr() == unsafe { base.add(i * N) })
}

/// Run the kernels of `plan` over `count` contiguous inputs at `base`.
unsafe fn run_plan(
    plan: &[usize],
    kernels: &[Option<Kernel>],
    base: *const u8,
    stride: usize,
    blocks: usize,
    key: &CVWords,
    counter: u64,
    counter_step: u64,
    packed_flags: u64,
    out: &mut [u8],
) {
    let mut done = 0;
    for &size in plan {
        let kernel = kernels[size].expect("plan names an existing kernel");
        unsafe {
            kernel(
                base.add(done * stride),
                blocks as u64,
                key.as_ptr(),
                counter + done as u64 * counter_step,
                packed_flags,
                out[done * OUT_LEN..].as_mut_ptr(),
            );
        }
        done += size;
    }
}

/// `hash_many` for whole chunks (`N == CHUNK_LEN`, counter incrementing)
/// and for parent blocks (`N == BLOCK_LEN`, counter fixed). Other shapes
/// are not produced by this crate and stop the program.
///
/// Unsafe because the CPU must have NEON and the SHA-3 extension.
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
    let (plans, kernels, counter_step): (&[&[usize]; 16], &[Option<Kernel>], u64) =
        match (N, increment_counter.yes()) {
            (CHUNK_LEN, true) => (&CHUNK_PLANS, &CHUNK_KERNELS, 1),
            (BLOCK_LEN, false) => (&PARENT_PLANS, &PARENT_KERNELS, 0),
            _ => panic!(
                "hash_many shape the NEON kernels do not cover: N = {N}, increment = {}",
                increment_counter.yes()
            ),
        };
    let packed = flags as u64 | (flags_start as u64) << 8 | (flags_end as u64) << 16;
    let blocks = N / BLOCK_LEN;

    let mut done = 0;
    while done < inputs.len() {
        let count = core::cmp::min(GROUP, inputs.len() - done);
        let group = &inputs[done..done + count];
        let group_counter = counter + done as u64 * counter_step;
        let group_out = &mut out[done * OUT_LEN..];
        if contiguous(group) {
            unsafe {
                run_plan(
                    plans[count],
                    kernels,
                    group[0].as_ptr(),
                    N,
                    blocks,
                    key,
                    group_counter,
                    counter_step,
                    packed,
                    group_out,
                );
            }
        } else if N == BLOCK_LEN {
            // Parents assembled by a caller can arrive scattered; the copy
            // fits a small buffer.
            let mut buf = [0u8; GROUP * BLOCK_LEN];
            for (i, input) in group.iter().enumerate() {
                buf[i * N..][..N].copy_from_slice(&input[..]);
            }
            unsafe {
                run_plan(
                    plans[count],
                    kernels,
                    buf.as_ptr(),
                    N,
                    blocks,
                    key,
                    group_counter,
                    counter_step,
                    packed,
                    group_out,
                );
            }
        } else {
            // Scattered chunks (upstream's benches build them from separate
            // buffers): one kernel call per chunk. Copying 15 KiB would cost
            // more than the kernel saves.
            for (i, input) in group.iter().enumerate() {
                unsafe {
                    run_plan(
                        plans[1],
                        kernels,
                        input.as_ptr(),
                        N,
                        blocks,
                        key,
                        group_counter + i as u64 * counter_step,
                        counter_step,
                        packed,
                        &mut group_out[i * OUT_LEN..],
                    );
                }
            }
        }
        done += count;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{CHUNK_END, CHUNK_START, IV, KEYED_HASH, PARENT};

    #[test]
    fn test_hash_many() {
        if !sha3_detected() {
            return;
        }
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }

    /// Every count 1..=15 for chunks and parents, at four counter values,
    /// so each plan and each kernel is exercised on its own.
    #[test]
    fn test_every_count_against_portable() {
        if !sha3_detected() {
            return;
        }
        let mut input = [0u8; 15 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        for n in 1..=15 {
            let chunks: arrayvec::ArrayVec<&[u8; CHUNK_LEN], 15> = input
                .chunks_exact(CHUNK_LEN)
                .take(n)
                .map(|c| c.try_into().unwrap())
                .collect();
            let parents: arrayvec::ArrayVec<&[u8; BLOCK_LEN], 15> = input
                .chunks_exact(BLOCK_LEN)
                .take(n)
                .map(|c| c.try_into().unwrap())
                .collect();
            for counter in [0u64, u32::MAX as u64, i32::MAX as u64, 1 << 40] {
                let mut want = [0u8; 15 * OUT_LEN];
                let mut got = [0u8; 15 * OUT_LEN];
                crate::portable::hash_many(
                    &chunks,
                    IV,
                    counter,
                    IncrementCounter::Yes,
                    KEYED_HASH,
                    CHUNK_START,
                    CHUNK_END,
                    &mut want,
                );
                unsafe {
                    hash_many(
                        &chunks,
                        IV,
                        counter,
                        IncrementCounter::Yes,
                        KEYED_HASH,
                        CHUNK_START,
                        CHUNK_END,
                        &mut got,
                    )
                };
                assert_eq!(
                    &want[..n * OUT_LEN],
                    &got[..n * OUT_LEN],
                    "chunks n = {n}, counter = {counter}"
                );
                crate::portable::hash_many(
                    &parents,
                    IV,
                    counter,
                    IncrementCounter::No,
                    KEYED_HASH | PARENT,
                    0,
                    0,
                    &mut want,
                );
                unsafe {
                    hash_many(
                        &parents,
                        IV,
                        counter,
                        IncrementCounter::No,
                        KEYED_HASH | PARENT,
                        0,
                        0,
                        &mut got,
                    )
                };
                assert_eq!(
                    &want[..n * OUT_LEN],
                    &got[..n * OUT_LEN],
                    "parents n = {n}, counter = {counter}"
                );
            }
        }
    }

    #[test]
    fn test_scattered_chunks() {
        if !sha3_detected() {
            return;
        }
        let mut input = [0u8; 8 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        // Chunks in reverse order: contiguous in memory, scattered as inputs.
        let chunks: arrayvec::ArrayVec<&[u8; CHUNK_LEN], 8> = input
            .chunks_exact(CHUNK_LEN)
            .rev()
            .map(|c| c.try_into().unwrap())
            .collect();
        let mut want = [0u8; 8 * OUT_LEN];
        let mut got = [0u8; 8 * OUT_LEN];
        crate::portable::hash_many(
            &chunks,
            IV,
            3,
            IncrementCounter::Yes,
            0,
            CHUNK_START,
            CHUNK_END,
            &mut want,
        );
        unsafe {
            hash_many(
                &chunks,
                IV,
                3,
                IncrementCounter::Yes,
                0,
                CHUNK_START,
                CHUNK_END,
                &mut got,
            )
        };
        assert_eq!(want, got);
    }

    #[test]
    fn test_scattered_parents() {
        if !sha3_detected() {
            return;
        }
        let mut input = [0u8; 8 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        // Parent blocks spaced a chunk apart.
        let parents: arrayvec::ArrayVec<&[u8; BLOCK_LEN], 8> = input
            .chunks_exact(CHUNK_LEN)
            .map(|c| c[..BLOCK_LEN].try_into().unwrap())
            .collect();
        let mut want = [0u8; 8 * OUT_LEN];
        let mut got = [0u8; 8 * OUT_LEN];
        crate::portable::hash_many(
            &parents,
            IV,
            7,
            IncrementCounter::No,
            PARENT,
            0,
            0,
            &mut want,
        );
        unsafe {
            hash_many(
                &parents,
                IV,
                7,
                IncrementCounter::No,
                PARENT,
                0,
                0,
                &mut got,
            )
        };
        assert_eq!(want, got);
    }

    /// The committed assembly is the generator's output. Skipped where
    /// Python is absent.
    #[test]
    #[cfg(feature = "std")]
    fn test_assembly_matches_generator() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = match std::process::Command::new("python3")
            .arg(root.join("tools/gen_neon_hybrid.py"))
            .output()
        {
            Ok(output) if output.status.success() => output.stdout,
            _ => return,
        };
        let committed = std::fs::read(root.join("c/blake3_neon_hybrid_aarch64.S")).unwrap();
        assert!(
            output == committed,
            "c/blake3_neon_hybrid_aarch64.S is stale; run tools/gen_neon_hybrid.py"
        );
    }

    #[test]
    fn test_compress_blocks_against_portable() {
        let mut input = [0u8; CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        for count in 1..=16 {
            for counter in [0u64, u32::MAX as u64, 1 << 40] {
                let mut want = *IV;
                let mut flags = KEYED_HASH | CHUNK_START;
                for b in 0..count {
                    crate::portable::compress_in_place(
                        &mut want,
                        input[b * BLOCK_LEN..][..BLOCK_LEN].try_into().unwrap(),
                        BLOCK_LEN as u8,
                        counter,
                        flags,
                    );
                    flags = KEYED_HASH;
                }
                let mut got = *IV;
                unsafe {
                    compress_blocks(
                        &mut got,
                        input.as_ptr(),
                        count,
                        counter,
                        KEYED_HASH,
                        CHUNK_START,
                    )
                };
                assert_eq!(want, got, "count = {count}, counter = {counter}");
            }
        }
    }
}
