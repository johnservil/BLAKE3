//! `hash_many` for AArch64 with NEON and the SHA-3 extension, built from the
//! kernels in c/blake3_neon_hybrid_aarch64.S.
//!
//! Each kernel hashes a fixed number of inputs given as a pointer table. Chunk
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
//! Longer input lists are hashed fifteen at a time.
//!
//! The scalar kernel also serves every single-chunk job: `hash_chunk` runs
//! a whole input of one chunk or less, root compression included, in one
//! call; `compress_blocks` is `ChunkState::update`'s inner loop; and
//! `compress_in_place` is one compression of any block length, which the
//! incremental `Hasher` uses for chunk finalization, parent nodes, and the
//! root. Every AArch64 core runs the scalar kernel, so those need no
//! feature check.

use crate::{BLOCK_LEN, CHUNK_LEN, CVWords, IncrementCounter, OUT_LEN};

/// The kernels live in c/blake3_neon_hybrid_aarch64.S. Every one has the
/// signature `(base, blocks, key, counter, packed_flags, out)`; the contract
/// is in that file's header. `packed_flags` is
/// `flags | flags_start << 8 | flags_end << 16 | last_len << 24`.
mod asm {
    unsafe extern "C" {
        /// k1 entered with the input pointer itself in place of a table,
        /// and the address of the last block as a seventh argument: blocks
        /// 0..blocks-1 come from `input`, the last from `last`.
        pub fn blake3_hybrid_c1(
            input: *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            last: *const u8,
        );
        pub fn blake3_hybrid_k1(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k2(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k3(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k4(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k5(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k6(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k8(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k9(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_k10(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p2(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p4(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p8(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
    }
}

/// Every kernel shares one signature: `(inputs, blocks, key, counter,
/// packed_flags, out)`. See the contract in that file.
type Kernel = unsafe extern "C" fn(*const *const u8, u64, *const u32, u64, u64, *mut u8);

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
    let packed = flags as u64 | (flags_start as u64) << 8 | (BLOCK_LEN as u64) << 24;
    // The kernel reads the whole key in its prologue and stores the output
    // at its end, so `cv` serves as both: the output words land in place,
    // in the little-endian layout CVWords has on this target.
    let cv_ptr = cv.as_mut_ptr();
    unsafe {
        let last = blocks.add((count - 1) * BLOCK_LEN);
        asm::blake3_hybrid_c1(blocks, count as u64, cv_ptr, counter, packed, cv_ptr as *mut u8, last);
    }
}

/// One compression of `block` into `cv`, with every flag in `flags`
/// (start, end, root, and so on are the caller's business) and the given
/// `block_len` recorded in the state. This is `Platform::compress_in_place`
/// on the scalar kernel: the chunk finalizer, the root compression, and
/// every parent node of the incremental `Hasher` go through it. Registers
/// hold the whole state, so it saves the portable compressor's spills.
///
/// The kernel uses only integer instructions and runs on every AArch64
/// core. `block_len` is at most 64; the block is 64 bytes, zero-padded
/// past `block_len` by the caller as the portable compressor requires.
pub fn compress_in_place(
    cv: &mut CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    debug_assert!(block_len as usize <= BLOCK_LEN);
    // The one block is both first and last: the kernel ORs flags_start
    // into the first block and flags_end into the last, so the flags go
    // in as the base and the two per-position fields stay zero.
    let packed = flags as u64 | (block_len as u64) << 24;
    // `cv` is key and output at once; see compress_blocks.
    let cv_ptr = cv.as_mut_ptr();
    // Safe: the block is 64 readable bytes and the count is 1.
    unsafe {
        let block = block.as_ptr();
        asm::blake3_hybrid_c1(block, 1, cv_ptr, counter, packed, cv_ptr as *mut u8, block);
    }
}

/// Hash a whole chunk of `1..=16` blocks in one kernel call and write its
/// 32-byte chaining value (or root hash) to `out`. Blocks `0..blocks - 1`
/// are the first `(blocks - 1) * 64` bytes of `input`; the last block is
/// `last`, zero-padded past `last_len`, so a short final block costs one
/// 64-byte copy. The first block carries `flags | flags_start`, the last
/// `flags | flags_end`, and every block `flags` and `counter`. This is the
/// one-chunk `hash()`: CHUNK_START on the first block, CHUNK_END | ROOT on
/// the last, one kernel call, no chaining value travelling through memory
/// between calls.
///
/// Unsafe because `input` must hold `(blocks - 1) * 64` readable bytes and
/// `blocks` must be in 1..=16.
pub unsafe fn hash_chunk(
    input: *const u8,
    blocks: usize,
    last: &[u8; BLOCK_LEN],
    last_len: u8,
    key: &CVWords,
    counter: u64,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8; OUT_LEN],
) {
    debug_assert!((1..=16).contains(&blocks));
    debug_assert!(last_len as usize <= BLOCK_LEN);
    let packed = flags as u64
        | (flags_start as u64) << 8
        | (flags_end as u64) << 16
        | (last_len as u64) << 24;
    unsafe {
        asm::blake3_hybrid_c1(
            input,
            blocks as u64,
            key.as_ptr(),
            counter,
            packed,
            out.as_mut_ptr(),
            last.as_ptr(),
        );
    }
}

/// Run the kernels of `plan` over `inputs`.
unsafe fn run_plan(
    plan: &[usize],
    kernels: &[Option<Kernel>],
    inputs: *const *const u8,
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
                inputs.add(done),
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
    let packed = flags as u64
        | (flags_start as u64) << 8
        | (flags_end as u64) << 16
        | (BLOCK_LEN as u64) << 24;
    let blocks = N / BLOCK_LEN;
    // `&[&[u8; N]]` is a table of pointers, which is what the kernels take.
    let table = inputs.as_ptr() as *const *const u8;

    let mut done = 0;
    while done < inputs.len() {
        let count = core::cmp::min(GROUP, inputs.len() - done);
        unsafe {
            run_plan(
                plans[count],
                kernels,
                table.add(done),
                blocks,
                key,
                counter + done as u64 * counter_step,
                counter_step,
                packed,
                &mut out[done * OUT_LEN..],
            );
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

    /// Every block length 0..=64, every flag bit, and several counters:
    /// the kernel's one-block path against the portable compressor.
    #[test]
    fn test_compress_in_place_against_portable() {
        let mut input = [0u8; BLOCK_LEN];
        crate::test::paint_test_input(&mut input);
        for block_len in 0..=BLOCK_LEN {
            let mut block = [0u8; BLOCK_LEN];
            block[..block_len].copy_from_slice(&input[..block_len]);
            for flags in [
                0,
                CHUNK_START,
                CHUNK_END,
                CHUNK_START | CHUNK_END | crate::ROOT,
                PARENT,
                PARENT | crate::ROOT | KEYED_HASH,
                crate::DERIVE_KEY_CONTEXT | CHUNK_START | CHUNK_END | crate::ROOT,
                0xff,
            ] {
                for counter in [0u64, 1, u32::MAX as u64, 1 << 40, u64::MAX] {
                    let mut want = *IV;
                    crate::portable::compress_in_place(
                        &mut want,
                        &block,
                        block_len as u8,
                        counter,
                        flags,
                    );
                    let mut got = *IV;
                    compress_in_place(&mut got, &block, block_len as u8, counter, flags);
                    assert_eq!(
                        want, got,
                        "block_len = {block_len}, flags = {flags:#x}, counter = {counter}"
                    );
                }
            }
        }
    }

    /// hash_chunk for every input length 0..=1024 (every block count and
    /// every final block length), with root and non-root flags, against
    /// the portable compressor run block by block.
    #[test]
    fn test_hash_chunk_against_portable() {
        let mut input = [0u8; CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        for len in 0..=CHUNK_LEN {
            let blocks = core::cmp::max(1, (len + BLOCK_LEN - 1) / BLOCK_LEN);
            let whole = (blocks - 1) * BLOCK_LEN;
            let mut last = [0u8; BLOCK_LEN];
            last[..len - whole].copy_from_slice(&input[whole..len]);
            for (flags, flags_end, counter) in [
                (0, CHUNK_END | crate::ROOT, 0u64),
                (KEYED_HASH, CHUNK_END, 5),
                (crate::DERIVE_KEY_MATERIAL, CHUNK_END | crate::ROOT, 1 << 40),
            ] {
                let mut want = *IV;
                for b in 0..blocks {
                    let mut block_flags = flags;
                    if b == 0 {
                        block_flags |= CHUNK_START;
                    }
                    let (block, block_len) = if b + 1 == blocks {
                        block_flags |= flags_end;
                        (&last, (len - whole) as u8)
                    } else {
                        (input[b * BLOCK_LEN..][..BLOCK_LEN].try_into().unwrap(), BLOCK_LEN as u8)
                    };
                    crate::portable::compress_in_place(
                        &mut want,
                        block,
                        block_len,
                        counter,
                        block_flags,
                    );
                }
                let mut got = [0u8; OUT_LEN];
                unsafe {
                    hash_chunk(
                        input.as_ptr(),
                        blocks,
                        &last,
                        (len - whole) as u8,
                        IV,
                        counter,
                        flags,
                        CHUNK_START,
                        flags_end,
                        &mut got,
                    )
                };
                assert_eq!(
                    crate::platform::le_bytes_from_words_32(&want),
                    got,
                    "len = {len}, flags = {flags:#x}, counter = {counter}"
                );
            }
        }
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
