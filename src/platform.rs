use crate::{BLOCK_LEN, CVWords, IncrementCounter, portable};

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        cfg_if::cfg_if! {
            if #[cfg(blake3_avx512_ffi)] {
                pub const MAX_SIMD_DEGREE: usize = 16;
            } else {
                pub const MAX_SIMD_DEGREE: usize = 8;
            }
        }
    } else if #[cfg(blake3_sme2)] {
        // See sme2::DEGREE.
        pub const MAX_SIMD_DEGREE: usize = crate::sme2::DEGREE;
    } else if #[cfg(blake3_neon)] {
        pub const MAX_SIMD_DEGREE: usize = 4;
    } else if #[cfg(blake3_wasm32_simd)] {
        pub const MAX_SIMD_DEGREE: usize = 4;
    } else {
        pub const MAX_SIMD_DEGREE: usize = 1;
    }
}

// There are some places where we want a static size that's equal to the
// MAX_SIMD_DEGREE, but also at least 2. Constant contexts aren't currently
// allowed to use cmp::max, so we have to hardcode this additional constant
// value. Get rid of this once cmp::max is a const fn.
cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        cfg_if::cfg_if! {
            if #[cfg(blake3_avx512_ffi)] {
                pub const MAX_SIMD_DEGREE_OR_2: usize = 16;
            } else {
                pub const MAX_SIMD_DEGREE_OR_2: usize = 8;
            }
        }
    } else if #[cfg(blake3_sme2)] {
        pub const MAX_SIMD_DEGREE_OR_2: usize = crate::sme2::DEGREE;
    } else if #[cfg(blake3_neon)] {
        pub const MAX_SIMD_DEGREE_OR_2: usize = 4;
    } else if #[cfg(blake3_wasm32_simd)] {
        pub const MAX_SIMD_DEGREE_OR_2: usize = 4;
    } else {
        pub const MAX_SIMD_DEGREE_OR_2: usize = 2;
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Platform {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE41,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX2,
    #[cfg(blake3_avx512_ffi)]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX512,
    #[cfg(blake3_neon)]
    NEON,
    #[cfg(blake3_sme2)]
    SME2,
    #[cfg(blake3_wasm32_simd)]
    #[allow(non_camel_case_types)]
    WASM32_SIMD,
}

impl Platform {
    #[allow(unreachable_code)]
    pub fn detect() -> Self {
        #[cfg(miri)]
        {
            return Platform::Portable;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(blake3_avx512_ffi)]
            {
                if avx512_detected() {
                    return Platform::AVX512;
                }
            }
            if avx2_detected() {
                return Platform::AVX2;
            }
            if sse41_detected() {
                return Platform::SSE41;
            }
            if sse2_detected() {
                return Platform::SSE2;
            }
        }
        // SME2 is detected at runtime; the kernels also require a 512-bit
        // streaming vector length, which sme2_detected() checks. Without
        // SME2 the NEON backend (integer + vector hybrid kernels) is the
        // selection, and it is a real result in its own right; a benchmark
        // that cares which one ran reads it back from detect().
        #[cfg(blake3_sme2)]
        {
            if sme2_detected() {
                return Platform::SME2;
            }
        }
        // We don't use dynamic feature detection for NEON. If the "neon"
        // feature is on, NEON is assumed to be supported.
        #[cfg(blake3_neon)]
        {
            return Platform::NEON;
        }
        #[cfg(blake3_wasm32_simd)]
        {
            return Platform::WASM32_SIMD;
        }
        Platform::Portable
    }

    pub fn simd_degree(&self) -> usize {
        let degree = match self {
            Platform::Portable => 1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE2 => 4,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => 4,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => 8,
            #[cfg(blake3_avx512_ffi)]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX512 => 16,
            #[cfg(blake3_neon)]
            Platform::NEON => 4,
            #[cfg(blake3_sme2)]
            Platform::SME2 => crate::sme2::DEGREE,
            #[cfg(blake3_wasm32_simd)]
            Platform::WASM32_SIMD => 4,
        };
        debug_assert!(degree <= MAX_SIMD_DEGREE);
        degree
    }

    pub fn compress_in_place(
        &self,
        cv: &mut CVWords,
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) {
        match self {
            Platform::Portable => portable::compress_in_place(cv, block, block_len, counter, flags),
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE2 => unsafe {
                crate::sse2::compress_in_place(cv, block, block_len, counter, flags)
            },
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 | Platform::AVX2 => unsafe {
                crate::sse41::compress_in_place(cv, block, block_len, counter, flags)
            },
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX512 => unsafe {
                crate::avx512::compress_in_place(cv, block, block_len, counter, flags)
            },
            // No NEON compress_in_place() implementation yet.
            #[cfg(blake3_neon)]
            Platform::NEON => portable::compress_in_place(cv, block, block_len, counter, flags),
            // Single compressions stay on the portable path with SME2 too.
            #[cfg(blake3_sme2)]
            Platform::SME2 => portable::compress_in_place(cv, block, block_len, counter, flags),
            #[cfg(blake3_wasm32_simd)]
            Platform::WASM32_SIMD => {
                crate::wasm32_simd::compress_in_place(cv, block, block_len, counter, flags)
            }
        }
    }

    /// Compress `count` whole blocks of one chunk into `cv`: the first block
    /// with `flags | flags_start`, the rest with `flags`, all with `counter`.
    /// `blocks` holds at least `count * BLOCK_LEN` bytes; `count` is 1..=16.
    pub fn compress_blocks(
        &self,
        cv: &mut CVWords,
        blocks: &[u8],
        count: usize,
        counter: u64,
        flags: u8,
        flags_start: u8,
    ) {
        assert!((1..=16).contains(&count));
        assert!(blocks.len() >= count * BLOCK_LEN);
        // The integer-only kernel runs on every AArch64 core.
        #[cfg(blake3_neon_hybrid)]
        {
            unsafe {
                crate::neon_hybrid::compress_blocks(
                    cv,
                    blocks.as_ptr(),
                    count,
                    counter,
                    flags,
                    flags_start,
                );
            }
            return;
        }
        #[allow(unreachable_code)]
        {
            let mut block_flags = flags | flags_start;
            for block in blocks[..count * BLOCK_LEN].chunks_exact(BLOCK_LEN) {
                self.compress_in_place(
                    cv,
                    block.try_into().unwrap(),
                    BLOCK_LEN as u8,
                    counter,
                    block_flags,
                );
                block_flags = flags;
            }
        }
    }

    pub fn compress_xof(
        &self,
        cv: &CVWords,
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) -> [u8; 64] {
        match self {
            Platform::Portable => portable::compress_xof(cv, block, block_len, counter, flags),
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE2 => unsafe {
                crate::sse2::compress_xof(cv, block, block_len, counter, flags)
            },
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 | Platform::AVX2 => unsafe {
                crate::sse41::compress_xof(cv, block, block_len, counter, flags)
            },
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX512 => unsafe {
                crate::avx512::compress_xof(cv, block, block_len, counter, flags)
            },
            // No NEON compress_xof() implementation yet.
            #[cfg(blake3_neon)]
            Platform::NEON => portable::compress_xof(cv, block, block_len, counter, flags),
            #[cfg(blake3_sme2)]
            Platform::SME2 => portable::compress_xof(cv, block, block_len, counter, flags),
            #[cfg(blake3_wasm32_simd)]
            Platform::WASM32_SIMD => {
                crate::wasm32_simd::compress_xof(cv, block, block_len, counter, flags)
            }
        }
    }

    // IMPLEMENTATION NOTE
    // ===================
    // hash_many() applies two optimizations. The critically important
    // optimization is the high-performance parallel SIMD hashing mode,
    // described in detail in the spec. This more than doubles throughput per
    // thread. Another optimization is keeping the state vectors transposed
    // from block to block within a chunk. When state vectors are transposed
    // after every block, there's a small but measurable performance loss.
    // Compressing chunks with a dedicated loop avoids this.

    pub fn hash_many<const N: usize>(
        &self,
        inputs: &[&[u8; N]],
        key: &CVWords,
        counter: u64,
        increment_counter: IncrementCounter,
        flags: u8,
        flags_start: u8,
        flags_end: u8,
        out: &mut [u8],
    ) {
        match self {
            Platform::Portable => portable::hash_many(
                inputs,
                key,
                counter,
                increment_counter,
                flags,
                flags_start,
                flags_end,
                out,
            ),
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE2 => unsafe {
                crate::sse2::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => unsafe {
                crate::sse41::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                crate::avx2::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX512 => unsafe {
                crate::avx512::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Assumed to be safe if the "neon" feature is on.
            #[cfg(blake3_neon)]
            Platform::NEON => unsafe {
                crate::neon::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Safe because detect() checked for SME2 with a 512-bit vector length.
            #[cfg(blake3_sme2)]
            Platform::SME2 => unsafe {
                crate::sme2::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Assumed to be safe if the "wasm32_simd" feature is on.
            #[cfg(blake3_wasm32_simd)]
            Platform::WASM32_SIMD => unsafe {
                crate::wasm32_simd::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
        }
    }

    pub fn xof_many(
        &self,
        cv: &CVWords,
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        mut counter: u64,
        flags: u8,
        out: &mut [u8],
    ) {
        debug_assert_eq!(0, out.len() % BLOCK_LEN, "whole blocks only");
        if out.is_empty() {
            // The current assembly implementation always outputs at least 1 block.
            return;
        }
        match self {
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(unix)]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX512 => unsafe {
                crate::avx512::xof_many(cv, block, block_len, counter, flags, out)
            },
            _ => {
                // For platforms without an optimized xof_many, fall back to a loop over
                // compress_xof. This is still faster than portable code.
                for out_block in out.chunks_exact_mut(BLOCK_LEN) {
                    // TODO: Use array_chunks_mut here once that's stable.
                    let out_array: &mut [u8; BLOCK_LEN] = out_block.try_into().unwrap();
                    *out_array = self.compress_xof(cv, block, block_len, counter, flags);
                    counter += 1;
                }
            }
        }
    }

    // Explicit platform constructors, for benchmarks.

    pub fn portable() -> Self {
        Self::Portable
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn sse2() -> Option<Self> {
        if sse2_detected() {
            Some(Self::SSE2)
        } else {
            None
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn sse41() -> Option<Self> {
        if sse41_detected() {
            Some(Self::SSE41)
        } else {
            None
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn avx2() -> Option<Self> {
        if avx2_detected() {
            Some(Self::AVX2)
        } else {
            None
        }
    }

    #[cfg(blake3_avx512_ffi)]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn avx512() -> Option<Self> {
        if avx512_detected() {
            Some(Self::AVX512)
        } else {
            None
        }
    }

    #[cfg(blake3_neon)]
    pub fn neon() -> Option<Self> {
        // Assumed to be safe if the "neon" feature is on.
        Some(Self::NEON)
    }

    #[cfg(blake3_sme2)]
    pub fn sme2() -> Option<Self> {
        if sme2_detected() {
            Some(Self::SME2)
        } else {
            None
        }
    }

    #[cfg(blake3_wasm32_simd)]
    pub fn wasm32_simd() -> Option<Self> {
        // Assumed to be safe if the "wasm32_simd" feature is on.
        Some(Self::WASM32_SIMD)
    }
}

// Note that AVX-512 is divided into multiple featuresets, and we use two of
// them, F and VL.
#[cfg(blake3_avx512_ffi)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(deprecated)] // TODO: revert this after https://github.com/RustCrypto/utils/pull/1515 is released
pub fn avx512_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A testing-only short-circuit.
    if cfg!(feature = "no_avx512") {
        return false;
    }

    cpufeatures::new!(has_avx512, "avx512f", "avx512vl");
    has_avx512::get()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(deprecated)] // TODO: revert this after https://github.com/RustCrypto/utils/pull/1515 is released
pub fn avx2_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A testing-only short-circuit.
    if cfg!(feature = "no_avx2") {
        return false;
    }

    cpufeatures::new!(has_avx2, "avx2");
    has_avx2::get()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(deprecated)] // TODO: revert this after https://github.com/RustCrypto/utils/pull/1515 is released
pub fn sse41_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A testing-only short-circuit.
    if cfg!(feature = "no_sse41") {
        return false;
    }

    cpufeatures::new!(has_sse41, "sse4.1");
    has_sse41::get()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(deprecated)] // TODO: revert this after https://github.com/RustCrypto/utils/pull/1515 is released
pub fn sse2_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A testing-only short-circuit.
    if cfg!(feature = "no_sse2") {
        return false;
    }

    cpufeatures::new!(has_sse2, "sse2");
    has_sse2::get()
}

#[inline(always)]
pub fn words_from_le_bytes_32(bytes: &[u8; 32]) -> [u32; 8] {
    let mut out = [0; 8];
    out[0] = u32::from_le_bytes(bytes[0 * 4..][..4].try_into().unwrap());
    out[1] = u32::from_le_bytes(bytes[1 * 4..][..4].try_into().unwrap());
    out[2] = u32::from_le_bytes(bytes[2 * 4..][..4].try_into().unwrap());
    out[3] = u32::from_le_bytes(bytes[3 * 4..][..4].try_into().unwrap());
    out[4] = u32::from_le_bytes(bytes[4 * 4..][..4].try_into().unwrap());
    out[5] = u32::from_le_bytes(bytes[5 * 4..][..4].try_into().unwrap());
    out[6] = u32::from_le_bytes(bytes[6 * 4..][..4].try_into().unwrap());
    out[7] = u32::from_le_bytes(bytes[7 * 4..][..4].try_into().unwrap());
    out
}

#[inline(always)]
pub fn words_from_le_bytes_64(bytes: &[u8; 64]) -> [u32; 16] {
    let mut out = [0; 16];
    out[0] = u32::from_le_bytes(bytes[0 * 4..][..4].try_into().unwrap());
    out[1] = u32::from_le_bytes(bytes[1 * 4..][..4].try_into().unwrap());
    out[2] = u32::from_le_bytes(bytes[2 * 4..][..4].try_into().unwrap());
    out[3] = u32::from_le_bytes(bytes[3 * 4..][..4].try_into().unwrap());
    out[4] = u32::from_le_bytes(bytes[4 * 4..][..4].try_into().unwrap());
    out[5] = u32::from_le_bytes(bytes[5 * 4..][..4].try_into().unwrap());
    out[6] = u32::from_le_bytes(bytes[6 * 4..][..4].try_into().unwrap());
    out[7] = u32::from_le_bytes(bytes[7 * 4..][..4].try_into().unwrap());
    out[8] = u32::from_le_bytes(bytes[8 * 4..][..4].try_into().unwrap());
    out[9] = u32::from_le_bytes(bytes[9 * 4..][..4].try_into().unwrap());
    out[10] = u32::from_le_bytes(bytes[10 * 4..][..4].try_into().unwrap());
    out[11] = u32::from_le_bytes(bytes[11 * 4..][..4].try_into().unwrap());
    out[12] = u32::from_le_bytes(bytes[12 * 4..][..4].try_into().unwrap());
    out[13] = u32::from_le_bytes(bytes[13 * 4..][..4].try_into().unwrap());
    out[14] = u32::from_le_bytes(bytes[14 * 4..][..4].try_into().unwrap());
    out[15] = u32::from_le_bytes(bytes[15 * 4..][..4].try_into().unwrap());
    out
}

#[inline(always)]
pub fn le_bytes_from_words_32(words: &[u32; 8]) -> [u8; 32] {
    let mut out = [0; 32];
    *<&mut [u8; 4]>::try_from(&mut out[0 * 4..][..4]).unwrap() = words[0].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[1 * 4..][..4]).unwrap() = words[1].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[2 * 4..][..4]).unwrap() = words[2].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[3 * 4..][..4]).unwrap() = words[3].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[4 * 4..][..4]).unwrap() = words[4].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[5 * 4..][..4]).unwrap() = words[5].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[6 * 4..][..4]).unwrap() = words[6].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[7 * 4..][..4]).unwrap() = words[7].to_le_bytes();
    out
}

#[inline(always)]
pub fn le_bytes_from_words_64(words: &[u32; 16]) -> [u8; 64] {
    let mut out = [0; 64];
    *<&mut [u8; 4]>::try_from(&mut out[0 * 4..][..4]).unwrap() = words[0].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[1 * 4..][..4]).unwrap() = words[1].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[2 * 4..][..4]).unwrap() = words[2].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[3 * 4..][..4]).unwrap() = words[3].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[4 * 4..][..4]).unwrap() = words[4].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[5 * 4..][..4]).unwrap() = words[5].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[6 * 4..][..4]).unwrap() = words[6].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[7 * 4..][..4]).unwrap() = words[7].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[8 * 4..][..4]).unwrap() = words[8].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[9 * 4..][..4]).unwrap() = words[9].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[10 * 4..][..4]).unwrap() = words[10].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[11 * 4..][..4]).unwrap() = words[11].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[12 * 4..][..4]).unwrap() = words[12].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[13 * 4..][..4]).unwrap() = words[13].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[14 * 4..][..4]).unwrap() = words[14].to_le_bytes();
    *<&mut [u8; 4]>::try_from(&mut out[15 * 4..][..4]).unwrap() = words[15].to_le_bytes();
    out
}

/// True when the CPU reports SME2 and the streaming vector length is 512
/// bits, the only length the SME2 kernels support. Detection is per
/// platform: `hw.optional.arm.FEAT_SME2` on Apple systems and
/// `HWCAP2_SME2` on Linux.
///
/// The answer is a property of the machine, so it is computed once and
/// cached. The vector-length check enters and leaves streaming mode, which
/// costs about as much as hashing 64 bytes; doing it on every call would
/// slow small hashes by a third.
#[cfg(blake3_sme2)]
#[inline(always)]
pub fn sme2_detected() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};

    // 0 = unknown, 1 = no, 2 = yes.
    static CACHE: AtomicU8 = AtomicU8::new(0);

    match CACHE.load(Ordering::Relaxed) {
        2 => return true,
        1 => return false,
        _ => {}
    }

    let detected = sme2_detect_uncached();
    CACHE.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
    detected
}

#[cfg(blake3_sme2)]
fn sme2_detect_uncached() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A testing-only short-circuit.
    if cfg!(feature = "no_sme2") {
        return false;
    }

    if !sme2_reported() {
        return false;
    }

    // Vector length check: the kernel reports it without doing any work
    // when asked for zero groups.
    let lanes = unsafe {
        crate::sme2::ffi::blake3_sme2_hash16_chunks_512(
            core::ptr::null(),
            core::ptr::null(),
            0,
            0,
            core::ptr::null_mut(),
            0,
        )
    };
    lanes == crate::sme2::GROUP as u64
}

#[cfg(all(blake3_sme2, target_vendor = "apple"))]
fn sme2_reported() -> bool {
    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.optional.arm.FEAT_SME2".as_ptr(),
            &mut value as *mut u32 as *mut libc::c_void,
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    rc == 0 && value != 0
}

#[cfg(all(blake3_sme2, target_os = "linux"))]
fn sme2_reported() -> bool {
    // AT_HWCAP2 from <elf.h> and HWCAP2_SME2 from <asm/hwcap.h> (Linux 6.4+).
    // The libc crate doesn't expose either for glibc targets yet.
    const AT_HWCAP2: libc::c_ulong = 26;
    const HWCAP2_SME2: libc::c_ulong = 1 << 37;
    let hwcap2 = unsafe { libc::getauxval(AT_HWCAP2) };
    hwcap2 & HWCAP2_SME2 != 0
}
