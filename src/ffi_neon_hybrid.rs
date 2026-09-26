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
//! | 3      | k3            | p3             |
//! | 4      | k4            | p4             |
//! | 5      | k5            | p5             |
//! | 6      | k6            | p3 + p3        |
//! | 7      | k7            | p7             |
//! | 8      | k8            | p8             |
//! | 9      | k9            | p9             |
//! | 10     | k10           | p7 + p3        |
//! | 11     | k8 + k3       | p9 + p2        |
//! | 12     | k9 + k3       | p9 + p3        |
//! | 13     | k10 + k3      | p9 + p4        |
//! | 14     | k10 + k4      | p9 + p5        |
//! | 15     | k10 + k5      | p8 + p7        |
//! | 16     | k10 + k6      | p8 + p8        |
//!
//! Longer input lists are hashed sixteen at a time. k10 is the fastest
//! kernel per chunk (two scalar chunks beside eight NEON ones: 0.21 ns/B
//! on a two-CPU SME2 VM against k8's 0.26 and k4's 0.29), so counts
//! from 13 up lead with it, and `Platform::NEON` reports a degree of 16 so
//! the tree walk hands over sixteen chunks at a time (0.23 ns/B in bulk,
//! against 0.29 at a degree of four).
//!
//! Plans are judged on an M4 Max P-core, its E-core, and the VM together.
//! Seven chunks on k7 and twelve on k9 + k3 beat k4 + k3 and k8 + k4 on all
//! three (cycles per byte P / E: 1.00 / 1.51 against 1.19 / 2.08, and
//! 1.02 / 1.52 against 1.10 / 1.89). E-cores have fewer integer units, so
//! kernels with two scalar chunks (k4, k6, k10) run twice the P-core's
//! cycles there, the others 1.1 to 1.6 times. k8 is two scalar chunks
//! beside a quad and a pair, the user's choice over two quads: 18% fewer
//! cycles on a P-core (7180 against 8718 for eight chunks), 6.5% more on an
//! E-core (14170 against 13310).
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
        pub fn blake3_hybrid_k7(
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
        /// q<n>: n whole chunks and a partial chunk, `n + 1` inputs. The
        /// partial chunk is the last input, read as `blocks` whole blocks
        /// (zero-padded), its final block recording `last_len` bytes;
        /// `partial` is `blocks | last_len << 8`.
        pub fn blake3_hybrid_q1(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q2(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q3(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q4(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q7(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q5(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q6(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q8(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_q9(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
            partial: u64,
        );
        pub fn blake3_hybrid_p2(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p3(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p5(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p7(
            inputs: *const *const u8,
            blocks: u64,
            key: *const u32,
            counter: u64,
            packed_flags: u64,
            out: *mut u8,
        );
        pub fn blake3_hybrid_p9(
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
const GROUP: usize = 16;

/// Chunk kernel per exact input count.
const CHUNK_KERNELS: [Option<Kernel>; 11] = [
    None,
    Some(asm::blake3_hybrid_k1),
    Some(asm::blake3_hybrid_k2),
    Some(asm::blake3_hybrid_k3),
    Some(asm::blake3_hybrid_k4),
    Some(asm::blake3_hybrid_k5),
    Some(asm::blake3_hybrid_k6),
    Some(asm::blake3_hybrid_k7),
    Some(asm::blake3_hybrid_k8),
    Some(asm::blake3_hybrid_k9),
    Some(asm::blake3_hybrid_k10),
];

/// Kernel sizes per chunk count 1..=16, largest first.
const CHUNK_PLANS: [&[usize]; 17] = [
    &[],
    &[1],
    &[2],
    &[3],
    &[4],
    &[5],
    &[6],
    &[7],
    &[8],
    &[9],
    &[10],
    &[8, 3],
    &[9, 3],
    &[10, 3],
    &[10, 4],
    &[10, 5],
    &[10, 6],
];

/// Parent kernel per exact input count.
const PARENT_KERNELS: [Option<Kernel>; 10] = [
    None,
    Some(asm::blake3_hybrid_k1),
    Some(asm::blake3_hybrid_p2),
    Some(asm::blake3_hybrid_p3),
    Some(asm::blake3_hybrid_p4),
    Some(asm::blake3_hybrid_p5),
    None,
    Some(asm::blake3_hybrid_p7),
    Some(asm::blake3_hybrid_p8),
    Some(asm::blake3_hybrid_p9),
];

/// Kernel sizes per parent count 1..=16. p3, p5, p7, and p9 hash one
/// block on the integer units beside the NEON ones, as k3, k5, k7, and k9
/// do for chunks. Against the binary decomposition over p2, p4, p8, and k1
/// run one after another, cycles per call on an M4 Max P-core / E-core
/// (probe/mixed-parents, jobs 121-122): 3 messages -37% / -38%, 5 -27 /
/// -27, 6 (p3 + p3) -25 / -11, 7 -31 / -34, 9 -18 / -22, 10 (p7 + p3;
/// p9 + k1 -10 / +1, p5 + p5 -8 / -2) -11 / -8, 11 -13 / -18, 12 -15 / -9,
/// 13 -12 / -15, 14 -17 / -12, 15 -18 / -19; on the VM 6-39% faster. 2,
/// 4, 8, and 16 keep their kernels; 13-16 were level with the other
/// plans tried on the VM. Two scalar lanes stay out: E-cores pay for a
/// second one what P-cores save.
const PARENT_PLANS: [&[usize]; 17] = [
    &[],
    &[1],
    &[2],
    &[3],
    &[4],
    &[5],
    &[3, 3],
    &[7],
    &[8],
    &[9],
    &[7, 3],
    &[9, 2],
    &[9, 3],
    &[9, 4],
    &[9, 5],
    &[8, 7],
    &[8, 8],
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

/// A kernel for whole chunks plus a partial chunk: the arguments of
/// [`Kernel`] and the partial chunk's `blocks | last_len << 8`.
type PartialKernel = unsafe extern "C" fn(*const *const u8, u64, *const u32, u64, u64, *mut u8, u64);

/// The q kernel for `n` whole chunks plus a partial chunk, `n` in 1..=9.
fn partial_kernel(n: usize) -> PartialKernel {
    match n {
        1 => asm::blake3_hybrid_q1,
        2 => asm::blake3_hybrid_q2,
        3 => asm::blake3_hybrid_q3,
        4 => asm::blake3_hybrid_q4,
        5 => asm::blake3_hybrid_q5,
        6 => asm::blake3_hybrid_q6,
        7 => asm::blake3_hybrid_q7,
        8 => asm::blake3_hybrid_q8,
        9 => asm::blake3_hybrid_q9,
        _ => panic!("no q kernel for {n} whole chunks"),
    }
}

/// Whether [`hash_chunks_with_partial`] takes `n` whole chunks: 1 through
/// 15 but 10 (from 16 the SME2 kernels or a second call take the whole
/// chunks).
fn partial_covers(n: usize) -> bool {
    (1..GROUP).contains(&n) && n != 10
}

/// Whether `n` whole chunks and a partial chunk of `len` bytes hash
/// faster through [`hash_chunks_with_partial`] than the whole chunks and
/// then the partial one. Ten whole chunks are one k10 call, the kernel with
/// the fewest cycles per byte; every split into a lead plan and a q kernel
/// costs more than the partial chunk after it (VM, 10 KiB + 500 B: 3097 ns
/// against 2790), where 11 to 15 gain 1-11%. One whole chunk moves from the
/// scalar kernel to q1's NEON pair, whose time does not grow with the
/// partial chunk: from five blocks on, the M4 Max's P-cores are level or
/// faster and its E-cores faster (1300 B: P level, E -14%; 2000 B: P -32%,
/// E -36%); at two blocks (1100 B) P-cores paid 14%.
pub fn partial_pays(n: usize, len: usize) -> bool {
    partial_covers(n) && (n > 1 || len > 4 * BLOCK_LEN)
}

/// The chaining values of `chunks` (whole, at `counter` on) and of
/// `partial` (1 to 1023 bytes, the chunk after them) into `out`: the last
/// up to nine whole chunks and the partial chunk in one q kernel call, the
/// partial chunk on the scalar units beside them; any whole chunks before
/// those through the usual plans. Requires 1 to 15 whole chunks but 10,
/// room in `out` for `chunks.len() + 1` values, and the SHA-3 extension.
pub unsafe fn hash_chunks_with_partial(
    chunks: &[&[u8; CHUNK_LEN]],
    partial: &[u8],
    key: &CVWords,
    counter: u64,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8],
) {
    assert!(partial_covers(chunks.len()), "1 to 15 whole chunks but 10");
    assert!(!partial.is_empty() && partial.len() < CHUNK_LEN, "a partial chunk holds 1 to 1023 bytes");
    assert!(out.len() >= (chunks.len() + 1) * OUT_LEN);
    // Ten whole chunks and more: the first ones through the usual plans,
    // so the q kernel takes the last nine or fewer.
    let lead = chunks.len().saturating_sub(9);
    if lead > 0 {
        let packed = flags as u64 | (flags_start as u64) << 8 | (flags_end as u64) << 16 | (BLOCK_LEN as u64) << 24;
        unsafe {
            run_plan(
                CHUNK_PLANS[lead],
                &CHUNK_KERNELS,
                chunks.as_ptr() as *const *const u8,
                CHUNK_LEN / BLOCK_LEN,
                key,
                counter,
                1,
                packed,
                out,
            );
        }
    }
    let (chunks, counter, out) = (&chunks[lead..], counter + lead as u64, &mut out[lead * OUT_LEN..]);
    let n = chunks.len();
    let kernel = partial_kernel(n);
    if n == 1 {
        // q1's pair takes the whole chunk twice (slots 0 and 2); the
        // partial chunk is slot 1; the duplicate's value is dropped.
        let blocks = partial.len().div_ceil(BLOCK_LEN);
        let last_len = partial.len() - (blocks - 1) * BLOCK_LEN;
        let mut padded = [0u8; CHUNK_LEN];
        padded[..partial.len()].copy_from_slice(partial);
        let table = [chunks[0].as_ptr(), padded.as_ptr(), chunks[0].as_ptr()];
        let mut cvs = [0u8; 3 * OUT_LEN];
        let packed = flags as u64 | (flags_start as u64) << 8 | (flags_end as u64) << 16 | (BLOCK_LEN as u64) << 24;
        unsafe {
            kernel(
                table.as_ptr(),
                (CHUNK_LEN / BLOCK_LEN) as u64,
                key.as_ptr(),
                counter,
                packed,
                cvs.as_mut_ptr(),
                blocks as u64 | (last_len as u64) << 8,
            );
        }
        out[..2 * OUT_LEN].copy_from_slice(&cvs[..2 * OUT_LEN]);
        return;
    }
    let blocks = partial.len().div_ceil(BLOCK_LEN);
    let last_len = partial.len() - (blocks - 1) * BLOCK_LEN;
    // The partial chunk as whole blocks: its bytes, then zeros to the end
    // of its last block.
    let mut padded = [0u8; CHUNK_LEN];
    padded[..partial.len()].copy_from_slice(partial);
    let mut table = [core::ptr::null::<u8>(); 10];
    for (slot, chunk) in table.iter_mut().zip(chunks) {
        *slot = chunk.as_ptr();
    }
    table[n] = padded.as_ptr();
    let packed = flags as u64 | (flags_start as u64) << 8 | (flags_end as u64) << 16 | (BLOCK_LEN as u64) << 24;
    unsafe {
        kernel(
            table.as_ptr(),
            (CHUNK_LEN / BLOCK_LEN) as u64,
            key.as_ptr(),
            counter,
            packed,
            out.as_mut_ptr(),
            blocks as u64 | (last_len as u64) << 8,
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

/// Whether [`hash_many`] takes inputs of N bytes: whole chunks at
/// incrementing counters, or 1 to 16 whole blocks at one counter (parents,
/// and separate messages of whole blocks; the parent kernels run any
/// block count).
pub fn covers(n: usize, increment_counter: IncrementCounter) -> bool {
    (n == CHUNK_LEN && increment_counter.yes()) || (!increment_counter.yes() && n % BLOCK_LEN == 0 && (BLOCK_LEN..=CHUNK_LEN).contains(&n))
}

/// `hash_many` for whole chunks (`N == CHUNK_LEN`, counter incrementing)
/// on the chunk plans, and for inputs of 1 to 16 whole blocks at one
/// counter (parents, separate messages) on the parent plans. Other shapes
/// ([`covers`] false) are not produced by this crate and stop the program.
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
    unsafe { hash_many_last_len(inputs, key, counter, increment_counter, flags, flags_start, flags_end, BLOCK_LEN, out) }
}

/// [`hash_many`] with each input's last block `last_len` bytes long (1 to
/// 64; its bytes past that zero): separate messages in slots of whole
/// blocks. Unsafe because the CPU must have NEON and the SHA-3 extension.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hash_many_last_len<const N: usize>(
    inputs: &[&[u8; N]],
    key: &CVWords,
    counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    last_len: usize,
    out: &mut [u8],
) {
    assert!(out.len() >= inputs.len() * OUT_LEN);
    assert!((1..=BLOCK_LEN).contains(&last_len), "a last block of 1 to 64 bytes");
    let (plans, kernels, counter_step): (&[&[usize]; 17], &[Option<Kernel>], u64) =
        match (N, increment_counter.yes()) {
            (CHUNK_LEN, true) => (&CHUNK_PLANS, &CHUNK_KERNELS, 1),
            _ if covers(N, increment_counter) => (&PARENT_PLANS, &PARENT_KERNELS, 0),
            _ => panic!(
                "hash_many shape the NEON kernels do not cover: N = {N}, increment = {}",
                increment_counter.yes()
            ),
        };
    let packed = flags as u64
        | (flags_start as u64) << 8
        | (flags_end as u64) << 16
        | (last_len as u64) << 24;
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

/// Separate inputs of `blocks` whole blocks (1 to 16) at one `counter`, the
/// last block `last_len` bytes (1 to 64, zero past it), from a table of
/// `count` pointers: the parent plans, as [`hash_many_last_len`] runs them
/// for inputs of a const length. Unsafe because the CPU must have NEON and
/// the SHA-3 extension and every pointer must reach `blocks` blocks.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hash_messages_raw(table: *const *const u8, count: usize, blocks: usize, key: &CVWords, counter: u64, flags_start: u8, flags_end: u8, last_len: usize, out: &mut [u8]) {
    assert!(out.len() >= count * OUT_LEN, "room for every digest");
    assert!((1..=16).contains(&blocks) && (1..=BLOCK_LEN).contains(&last_len), "1 to 16 blocks, the last of 1 to 64 bytes");
    let packed = (flags_start as u64) << 8 | (flags_end as u64) << 16 | (last_len as u64) << 24;
    let mut done = 0;
    while done < count {
        let n = core::cmp::min(GROUP, count - done);
        unsafe { run_plan(PARENT_PLANS[n], &PARENT_KERNELS, table.add(done), blocks, key, counter, 0, packed, &mut out[done * OUT_LEN..]) };
        done += n;
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

    /// Every count 1..=16 for chunks and parents, at four counter values,
    /// so each plan and each kernel is exercised on its own.
    #[test]
    fn test_every_count_against_portable() {
        if !sha3_detected() {
            return;
        }
        let mut input = [0u8; 16 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        for n in 1..=16 {
            let chunks: arrayvec::ArrayVec<&[u8; CHUNK_LEN], 16> = input
                .chunks_exact(CHUNK_LEN)
                .take(n)
                .map(|c| c.try_into().unwrap())
                .collect();
            let parents: arrayvec::ArrayVec<&[u8; BLOCK_LEN], 16> = input
                .chunks_exact(BLOCK_LEN)
                .take(n)
                .map(|c| c.try_into().unwrap())
                .collect();
            for counter in [0u64, u32::MAX as u64, i32::MAX as u64, 1 << 40] {
                let mut want = [0u8; 16 * OUT_LEN];
                let mut got = [0u8; 16 * OUT_LEN];
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
    /// Every q kernel, every partial length: the whole chunks' and the
    /// partial chunk's chaining values against the portable compressor,
    /// with a counter past zero and keyed flags as well as plain ones.
    #[test]
    fn test_partial_kernels_against_portable() {
        if !sha3_detected() {
            return;
        }
        let mut input = [0u8; 16 * CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        let key: CVWords = core::array::from_fn(|i| 0x0102_0304u32.wrapping_mul(i as u32 + 7));
        let chunk_cv = |bytes: &[u8], key: &CVWords, counter: u64, flags: u8| {
            let mut cv = *key;
            let blocks = core::cmp::max(1, bytes.len().div_ceil(BLOCK_LEN));
            for b in 0..blocks {
                let mut block = [0u8; BLOCK_LEN];
                let part = &bytes[b * BLOCK_LEN..core::cmp::min(bytes.len(), (b + 1) * BLOCK_LEN)];
                block[..part.len()].copy_from_slice(part);
                let mut block_flags = flags;
                if b == 0 {
                    block_flags |= CHUNK_START;
                }
                if b + 1 == blocks {
                    block_flags |= CHUNK_END;
                }
                crate::portable::compress_in_place(&mut cv, &block, part.len() as u8, counter, block_flags);
            }
            crate::platform::le_bytes_from_words_32(&cv)
        };
        for n in (1..16).filter(|&n| partial_covers(n)) {
            let chunks: Vec<&[u8; CHUNK_LEN]> =
                input[..n * CHUNK_LEN].chunks_exact(CHUNK_LEN).map(|c| c.try_into().unwrap()).collect();
            for len in 1..CHUNK_LEN {
                let partial = &input[n * CHUNK_LEN..][..len];
                for (key, flags, counter) in [(*IV, 0u8, 0u64), (key, KEYED_HASH, (1u64 << 32) - 2)] {
                    let mut got = vec![0u8; (n + 1) * OUT_LEN];
                    unsafe { hash_chunks_with_partial(&chunks, partial, &key, counter, flags, CHUNK_START, CHUNK_END, &mut got) };
                    for (i, cv) in got.chunks_exact(OUT_LEN).enumerate() {
                        let bytes: &[u8] = if i < n { &chunks[i][..] } else { partial };
                        assert_eq!(cv, chunk_cv(bytes, &key, counter + i as u64, flags), "q{n}, partial of {len} bytes, chunk {i}, flags {flags:#x}");
                    }
                }
            }
        }
    }

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
