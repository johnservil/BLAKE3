//! `hash_many` for two to fifteen whole chunks on AArch64 with NEON and the
//! SHA-3 extension, built from the generated kernels in `neon_hybrid_asm.rs`.
//!
//! Each kernel hashes a fixed number of contiguous chunks. This wrapper
//! splits an input count into kernel calls that keep every unit busy:
//!
//! | chunks | kernels      | chunks | kernels      |
//! |--------|--------------|--------|--------------|
//! | 2      | k2           | 9      | k9           |
//! | 3      | k3           | 10     | k10          |
//! | 4      | k4           | 11     | k8 + k3      |
//! | 5      | k5           | 12     | k8 + k4      |
//! | 6      | k6           | 13     | k8 + k5      |
//! | 7      | k4 + k3      | 14     | k8 + k6      |
//! | 8      | k8           | 15     | k9 + k6      |
//!
//! Anything the kernels do not cover (a lone chunk, partial chunks, parent
//! blocks, non-contiguous inputs, `IncrementCounter::No`) goes to the
//! caller's fallback via `Err`.

use crate::{CHUNK_LEN, CVWords, IncrementCounter, OUT_LEN};

#[path = "neon_hybrid_asm.rs"]
mod asm;

type Kernel = unsafe extern "C" fn(*const u8, u64, *const u32, u64, u64, *mut u8);

/// Kernel per exact chunk count, when one exists.
const KERNELS: [Option<Kernel>; 11] = [
    None,
    None,
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

/// Kernel sizes per chunk count 2..=15, largest first.
const PLANS: [&[usize]; 16] = [
    &[],
    &[],
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

/// Hash `inputs` with the hybrid kernels. Returns `Err(())` when the call
/// does not fit their contract; the caller then uses another path.
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
) -> Result<(), ()> {
    assert!(out.len() >= inputs.len() * OUT_LEN);
    let n = inputs.len();
    if N != CHUNK_LEN || !increment_counter.yes() || n < 2 || n > 15 || !contiguous(inputs) {
        return Err(());
    }
    let packed = flags as u64 | (flags_start as u64) << 8 | (flags_end as u64) << 16;
    let mut done = 0;
    for &size in PLANS[n] {
        let kernel = KERNELS[size].expect("plan names an existing kernel");
        unsafe {
            kernel(
                inputs[done].as_ptr(),
                (N / crate::BLOCK_LEN) as u64,
                key.as_ptr(),
                counter + done as u64,
                packed,
                out[done * OUT_LEN..].as_mut_ptr(),
            );
        }
        done += size;
    }
    debug_assert_eq!(done, n);
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{BLOCK_LEN, CHUNK_END, CHUNK_START, IV, KEYED_HASH};

    #[test]
    fn test_every_count_against_portable() {
        if !crate::neon_dup::sha3_detected() {
            return;
        }
        let mut input = [0u8; 15 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        for n in 2..=15 {
            let chunks: arrayvec::ArrayVec<&[u8; CHUNK_LEN], 15> = input
                .chunks_exact(CHUNK_LEN)
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
                }
                .unwrap();
                assert_eq!(
                    &want[..n * OUT_LEN],
                    &got[..n * OUT_LEN],
                    "n = {n}, counter = {counter}"
                );
            }
        }
        // Partial last chunk (blocks < 16) via a smaller N is not the kernels'
        // job; confirm the wrapper declines parents and single chunks.
        let parents: [&[u8; BLOCK_LEN]; 2] = [
            input[..BLOCK_LEN].try_into().unwrap(),
            input[BLOCK_LEN..2 * BLOCK_LEN].try_into().unwrap(),
        ];
        let mut out = [0u8; 2 * OUT_LEN];
        assert!(
            unsafe { hash_many(&parents, IV, 0, IncrementCounter::No, 0, 0, 0, &mut out) }.is_err()
        );
        let one: [&[u8; CHUNK_LEN]; 1] = [input[..CHUNK_LEN].try_into().unwrap()];
        assert!(
            unsafe { hash_many(&one, IV, 0, IncrementCounter::Yes, 0, 0, 0, &mut out) }.is_err()
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
