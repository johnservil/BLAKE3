use crate::{BLOCK_LEN, CVWords, IncrementCounter, portable};

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
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
    } else if #[cfg(blake3_neon_hybrid)] {
        // See neon_hybrid::GROUP.
        pub const MAX_SIMD_DEGREE: usize = 16;
    } else if #[cfg(blake3_neon)] {
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
    if #[cfg(target_arch = "x86_64")] {
        cfg_if::cfg_if! {
            if #[cfg(blake3_avx512_ffi)] {
                pub const MAX_SIMD_DEGREE_OR_2: usize = 16;
            } else {
                pub const MAX_SIMD_DEGREE_OR_2: usize = 8;
            }
        }
    } else if #[cfg(blake3_sme2)] {
        pub const MAX_SIMD_DEGREE_OR_2: usize = crate::sme2::DEGREE;
    } else if #[cfg(blake3_neon_hybrid)] {
        pub const MAX_SIMD_DEGREE_OR_2: usize = 16;
    } else if #[cfg(blake3_neon)] {
        pub const MAX_SIMD_DEGREE_OR_2: usize = 4;
    } else {
        pub const MAX_SIMD_DEGREE_OR_2: usize = 2;
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
/// Each platform's discriminant is fixed, the same in every build (0 means
/// "not yet recorded" to [`Platform::detect`]); tools/verify's whole-call
/// proofs start from the value `initialize` records.
pub enum Platform {
    Portable = 1,
    #[cfg(target_arch = "x86_64")]
    SSE41 = 3,
    #[cfg(target_arch = "x86_64")]
    AVX2 = 4,
    #[cfg(blake3_avx512_ffi)]
    #[cfg(target_arch = "x86_64")]
    AVX512 = 5,
    #[cfg(blake3_neon)]
    NEON = 6,
    #[cfg(blake3_sme2)]
    SME2 = 7,
}

/// One SME2 call at a time per process (all but always: two calls taking
/// the turn at the same instant can both get it): the turn a call takes
/// before it runs SME2 kernels, and the platform it runs on. A call finding
/// another SME2 call running takes NEON instead. Two SME2 threads of a process can
/// share one SME unit, each at half speed or less, slower than NEON; and
/// three of them at once sent the process's other threads to E-cores on an
/// M4 Max (NOTES-servil.md). Calls too small for the SME2 kernels
/// (`sme2_sized` false) leave the turn alone.
pub(crate) struct Sme2Turn {
    platform: Platform,
    #[cfg(blake3_sme2)]
    held: bool,
}

/// The turn's flag, on a 128-byte line of its own (Apple's cache line), so
/// no other state's writes invalidate it.
#[cfg(blake3_sme2)]
#[repr(align(128))]
struct Busy(core::sync::atomic::AtomicBool);

#[cfg(blake3_sme2)]
static SME2_BUSY: Busy = Busy(core::sync::atomic::AtomicBool::new(false));

impl Sme2Turn {
    #[inline]
    pub(crate) fn take(platform: Platform, sme2_sized: bool) -> Self {
        #[cfg(blake3_sme2)]
        if sme2_sized && matches!(platform, Platform::SME2) {
            use core::sync::atomic::Ordering;
            // A load, and a store when the turn is free: two calls that
            // look at the same instant can both take it, which costs what
            // running SME2 at once always did, and nothing else depends on
            // the flag. An atomic swap cost 3-9% for a batch of 24
            // messages (on the VM, 282 ns against 273, and 9-10% in the
            // regression check); acquire and release ordering about 1 µs
            // per call beside the SME2 kernels (256 one-block messages:
            // 3.43 µs against 2.50).
            let held = !SME2_BUSY.0.load(Ordering::Relaxed);
            if held {
                SME2_BUSY.0.store(true, Ordering::Relaxed);
            } else {
                crate::ran(crate::Ran::TurnTaken);
            }
            return Sme2Turn { platform: if held { Platform::SME2 } else { Platform::NEON }, held };
        }
        let _ = sme2_sized;
        Sme2Turn {
            platform,
            #[cfg(blake3_sme2)]
            held: false,
        }
    }

    #[inline]
    pub(crate) fn platform(&self) -> Platform {
        self.platform
    }
}

impl Drop for Sme2Turn {
    #[inline]
    fn drop(&mut self) {
        #[cfg(blake3_sme2)]
        if self.held {
            SME2_BUSY.0.store(false, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// The platform `initialize` detected, as its discriminant (1 and up), or 0
/// before `initialize` has run.
static DETECTED: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// A call before `initialize`: the caller broke the contract. The crate's
/// own unit tests, many small programs in one, initialize here instead.
#[cold]
#[inline(never)]
fn not_initialized() -> u8 {
    #[cfg(test)]
    {
        crate::initialize();
        DETECTED.load(core::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    panic!("blake3-servil: call blake3_servil::initialize() once, before any other call: it runs the startup self-test and detects the CPU")
}

impl Platform {
    /// The platform [`initialize`](crate::initialize) detected for this
    /// process. The caller must have called `initialize` (or
    /// `initialize_multithreaded`) before: a call before it panics, in every
    /// build. The check costs nothing extra: it is the load of the platform,
    /// which every call makes anyway.
    #[inline]
    pub fn detect() -> Self {
        let mut code = DETECTED.load(core::sync::atomic::Ordering::Relaxed);
        if code == 0 {
            code = not_initialized();
        }
        // Sound: DETECTED holds 0 or a Platform's discriminant (`record`).
        unsafe { core::mem::transmute::<u8, Platform>(code) }
    }

    /// Make `self` what [`Platform::detect`] returns from now on, in every
    /// thread: `initialize` does, once its self-test has passed.
    pub(crate) fn record(self) {
        DETECTED.store(self as u8, core::sync::atomic::Ordering::Relaxed);
    }

    /// [`Platform::detect`] without the startup self-test.
    #[allow(unreachable_code)]
    pub(crate) fn detect_unchecked() -> Self {
        #[cfg(miri)]
        {
            return Platform::Portable;
        }

        #[cfg(target_arch = "x86_64")]
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
        Platform::Portable
    }

    /// The selection's name, for reports.
    pub fn name(&self) -> &'static str {
        match self {
            Platform::Portable => "portable",
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 => "SSE4.1",
            #[cfg(target_arch = "x86_64")]
            Platform::AVX2 => "AVX2",
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => "AVX-512",
            #[cfg(blake3_neon)]
            Platform::NEON => "NEON",
            #[cfg(blake3_sme2)]
            Platform::SME2 => "SME2",
        }
    }

    /// The single-block compress kernel's name, for reports on platforms
    /// without the scalar c1 kernel.
    #[allow(dead_code)]
    pub fn compress_name(&self) -> &'static str {
        match self {
            Platform::Portable => "portable compression",
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 | Platform::AVX2 => "SSE4.1 compression",
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => "AVX-512 compression",
            #[cfg(blake3_neon)]
            Platform::NEON => "portable compression",
            #[cfg(blake3_sme2)]
            Platform::SME2 => "portable compression",
        }
    }

    /// The many-chunk kernel's name with its degree, for reports.
    #[allow(dead_code)]
    pub fn hash_many_name(&self) -> &'static str {
        match self {
            Platform::Portable => "portable hash_many",
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 => "SSE4.1 hash_many (4-way)",
            #[cfg(target_arch = "x86_64")]
            Platform::AVX2 => "AVX2 hash_many (8-way)",
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => "AVX-512 hash_many (16-way)",
            #[cfg(all(blake3_neon, blake3_neon_hybrid))]
            Platform::NEON => "integer + NEON hybrid kernels (16 chunks per call)",
            #[cfg(all(blake3_neon, not(blake3_neon_hybrid)))]
            Platform::NEON => "NEON hash_many (4-way)",
            #[cfg(blake3_sme2)]
            Platform::SME2 => "SME2 hash_many",
        }
    }

    pub fn simd_degree(&self) -> usize {
        let degree = match self {
            Platform::Portable => 1,
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 => 4,
            #[cfg(target_arch = "x86_64")]
            Platform::AVX2 => 8,
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => 16,
            // Sixteen chunks per call reach the k10 kernel (see
            // neon_hybrid); the four-lane C kernel takes four.
            #[cfg(all(blake3_neon, blake3_neon_hybrid))]
            Platform::NEON => {
                if crate::neon_hybrid::sha3_detected() {
                    16
                } else {
                    4
                }
            }
            #[cfg(all(blake3_neon, not(blake3_neon_hybrid)))]
            Platform::NEON => 4,
            #[cfg(blake3_sme2)]
            Platform::SME2 => crate::sme2::DEGREE,
        };
        debug_assert!(degree <= MAX_SIMD_DEGREE);
        degree
    }

    pub fn compress_in_place(&self, cv: &mut CVWords, block: &[u8; BLOCK_LEN], block_len: u8, counter: u64, flags: u8) {
        match self {
            Platform::Portable => portable::compress_in_place(cv, block, block_len, counter, flags),
            // Safe because detect() checked for platform support.
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 | Platform::AVX2 => unsafe { crate::sse41::compress_in_place(cv, block, block_len, counter, flags) },
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => unsafe { crate::avx512::compress_in_place(cv, block, block_len, counter, flags) },
            // The scalar kernel k1 keeps the state in registers; it runs on
            // every AArch64 core, so the NEON and SME2 platforms share it.
            #[cfg(all(blake3_neon, blake3_neon_hybrid))]
            Platform::NEON => crate::neon_hybrid::compress_in_place(cv, block, block_len, counter, flags),
            #[cfg(all(blake3_neon, not(blake3_neon_hybrid)))]
            Platform::NEON => portable::compress_in_place(cv, block, block_len, counter, flags),
            #[cfg(blake3_sme2)]
            Platform::SME2 => crate::neon_hybrid::compress_in_place(cv, block, block_len, counter, flags),
        }
    }

    /// Compress `count` whole blocks of one chunk into `cv`: the first block
    /// with `flags | flags_start`, the rest with `flags`, all with `counter`.
    /// `blocks` holds at least `count * BLOCK_LEN` bytes; `count` is 1..=16.
    pub fn compress_blocks(&self, cv: &mut CVWords, blocks: &[u8], count: usize, counter: u64, flags: u8, flags_start: u8) {
        assert!((1..=16).contains(&count));
        assert!(blocks.len() >= count * BLOCK_LEN);
        // The integer-only kernel runs on every AArch64 core.
        #[cfg(blake3_neon_hybrid)]
        {
            unsafe {
                crate::neon_hybrid::compress_blocks(cv, blocks.as_ptr(), count, counter, flags, flags_start);
            }
            return;
        }
        #[allow(unreachable_code)]
        {
            let mut block_flags = flags | flags_start;
            for block in blocks[..count * BLOCK_LEN].chunks_exact(BLOCK_LEN) {
                self.compress_in_place(cv, block.try_into().unwrap(), BLOCK_LEN as u8, counter, block_flags);
                block_flags = flags;
            }
        }
    }

    pub fn compress_xof(&self, cv: &CVWords, block: &[u8; BLOCK_LEN], block_len: u8, counter: u64, flags: u8) -> [u8; 64] {
        match self {
            Platform::Portable => portable::compress_xof(cv, block, block_len, counter, flags),
            // Safe because detect() checked for platform support.
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 | Platform::AVX2 => unsafe { crate::sse41::compress_xof(cv, block, block_len, counter, flags) },
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => unsafe { crate::avx512::compress_xof(cv, block, block_len, counter, flags) },
            // No NEON compress_xof() implementation yet.
            #[cfg(blake3_neon)]
            Platform::NEON => portable::compress_xof(cv, block, block_len, counter, flags),
            #[cfg(blake3_sme2)]
            Platform::SME2 => portable::compress_xof(cv, block, block_len, counter, flags),
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
        // Every kernel hashes whole blocks: an input of another length
        // would lose its tail, so it cannot compile.
        const { assert!(N > 0 && N % BLOCK_LEN == 0, "hash_many inputs are whole blocks") };
        match self {
            Platform::Portable => portable::hash_many(inputs, key, counter, increment_counter, flags, flags_start, flags_end, out),
            // Safe because detect() checked for platform support.
            #[cfg(target_arch = "x86_64")]
            Platform::SSE41 => unsafe { crate::sse41::hash_many(inputs, key, counter, increment_counter, flags, flags_start, flags_end, out) },
            // Safe because detect() checked for platform support.
            #[cfg(target_arch = "x86_64")]
            Platform::AVX2 => unsafe { crate::avx2::hash_many(inputs, key, counter, increment_counter, flags, flags_start, flags_end, out) },
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => unsafe { crate::avx512::hash_many(inputs, key, counter, increment_counter, flags, flags_start, flags_end, out) },
            // Assumed to be safe if the "neon" feature is on.
            #[cfg(blake3_neon)]
            Platform::NEON => unsafe { crate::neon::hash_many(inputs, key, counter, increment_counter, flags, flags_start, flags_end, out) },
            // Safe because detect() checked for SME2 with a 512-bit vector length.
            #[cfg(blake3_sme2)]
            Platform::SME2 => unsafe { crate::sme2::hash_many(inputs, key, counter, increment_counter, flags, flags_start, flags_end, out) },
        }
    }

    pub fn xof_many(&self, cv: &CVWords, block: &[u8; BLOCK_LEN], block_len: u8, mut counter: u64, flags: u8, out: &mut [u8]) {
        debug_assert_eq!(0, out.len() % BLOCK_LEN, "whole blocks only");
        if out.is_empty() {
            // The current assembly implementation always outputs at least 1 block.
            return;
        }
        match self {
            // Safe because detect() checked for platform support.
            #[cfg(blake3_avx512_ffi)]
            #[cfg(unix)]
            #[cfg(target_arch = "x86_64")]
            Platform::AVX512 => unsafe { crate::avx512::xof_many(cv, block, block_len, counter, flags, out) },
            _ => {
                // On SME2, whole groups of sixteen blocks on the extended-
                // output kernel, under the process's turn (M4 Max: 0.70 ns/B
                // block by block on the portable code); the rest below.
                #[cfg(blake3_sme2)]
                let out = {
                    let turn = Sme2Turn::take(*self, out.len() >= 16 * BLOCK_LEN);
                    let done = if matches!(turn.platform(), Platform::SME2) {
                        // Safe: the turn's platform is SME2 only where detect() found it.
                        unsafe { crate::sme2::xof_many(cv, block, block_len, counter, flags, out) }
                    } else {
                        0
                    };
                    counter += (done / BLOCK_LEN) as u64;
                    &mut out[done..]
                };
                // Then four at a time on NEON (AArch64), where the platform
                // has it: CPUs without SME2, and SME2's leftovers.
                #[cfg(all(blake3_neon, target_arch = "aarch64"))]
                let out = if !matches!(self, Platform::Portable) {
                    let done = crate::neon_xof::xof_many(cv, block, block_len, counter, flags, out);
                    counter += (done / BLOCK_LEN) as u64;
                    &mut out[done..]
                } else {
                    out
                };
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

    #[cfg(target_arch = "x86_64")]
    #[cfg(test)]
    pub fn sse41() -> Option<Self> {
        if sse41_detected() { Some(Self::SSE41) } else { None }
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(test)]
    pub fn avx2() -> Option<Self> {
        if avx2_detected() { Some(Self::AVX2) } else { None }
    }

    #[cfg(blake3_avx512_ffi)]
    #[cfg(target_arch = "x86_64")]
    #[cfg(test)]
    pub fn avx512() -> Option<Self> {
        if avx512_detected() { Some(Self::AVX512) } else { None }
    }

    #[cfg(blake3_neon)]
    #[cfg(test)]
    pub fn neon() -> Option<Self> {
        // Assumed to be safe if the "neon" feature is on.
        Some(Self::NEON)
    }

    #[cfg(blake3_sme2)]
    #[cfg(test)]
    pub fn sme2() -> Option<Self> {
        if sme2_detected() { Some(Self::SME2) } else { None }
    }
}

/// The x86-64 features the platforms check, as bits: [`SSE41`], [`AVX2`],
/// [`AVX512`] (F and VL, the two AVX-512 featuresets the kernels use), and
/// [`READ`] once the bits are read. Every check reads them from CPUID once
/// per process: CPUID costs about 640 ns (i7-12700K), and cpufeatures read
/// three leaves per feature checked, six on an AVX2 machine's first
/// detection where two serve. A feature the build assumes needs none.
#[cfg(target_arch = "x86_64")]
mod x86_bits {
    pub const SSE41: u8 = 2;
    pub const AVX2: u8 = 4;
    pub const AVX512: u8 = 8;
    pub const READ: u8 = 128;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn x86_features() -> u8 {
    use core::sync::atomic::{AtomicU8, Ordering};
    static FEATURES: AtomicU8 = AtomicU8::new(0);
    let features = FEATURES.load(Ordering::Relaxed);
    if features != 0 {
        return features;
    }
    let features = read_x86_features();
    FEATURES.store(features, Ordering::Relaxed);
    features
}

/// [`x86_features`] from CPUID: leaf 1, and, where the CPU has AVX and the
/// OS saves its registers (XSAVE, OSXSAVE, AVX, then XCR0), leaf 7. Every
/// CPU with AVX has leaf 7.
#[cfg(target_arch = "x86_64")]
#[cold]
fn read_x86_features() -> u8 {
    use core::arch::x86_64 as arch;
    use x86_bits::*;
    // Sound: every x86-64 CPU has CPUID. (Safe since Rust 1.94; unsafe
    // before.)
    #[allow(unused_unsafe)]
    let leaf1 = unsafe { arch::__cpuid(1) };
    let mut features = READ;
    if leaf1.ecx & (1 << 19) != 0 {
        features |= SSE41;
    }
    let avx = (1 << 26) | (1 << 27) | (1 << 28);
    if leaf1.ecx & avx == avx {
        // Sound: OSXSAVE (bit 27) says the OS enabled XGETBV.
        let xcr0 = unsafe { arch::_xgetbv(0) };
        // Sound: as leaf 1; leaf 7 exists where AVX does.
        #[allow(unused_unsafe)]
        let leaf7 = unsafe { arch::__cpuid_count(7, 0) };
        let (ymm, zmm) = (0b110, 0b1110_0110);
        if xcr0 & ymm == ymm && leaf7.ebx & (1 << 5) != 0 {
            features |= AVX2;
        }
        let f_vl = (1 << 16) | (1 << 31);
        if xcr0 & zmm == zmm && leaf7.ebx & f_vl == f_vl {
            features |= AVX512;
        }
    }
    features
}

#[cfg(blake3_avx512_ffi)]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn avx512_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A build for CPUs that have it (`-C target-cpu=...` or
    // `-C target-feature=...`) needs no check: run elsewhere, it stops
    // at its first such instruction (SIGILL), in the startup self-test.
    if cfg!(all(target_feature = "avx512f", target_feature = "avx512vl")) {
        return true;
    }

    x86_features() & x86_bits::AVX512 != 0
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn avx2_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A build for CPUs that have it (`-C target-cpu=...` or
    // `-C target-feature=...`) needs no check: run elsewhere, it stops
    // at its first such instruction (SIGILL), in the startup self-test.
    if cfg!(target_feature = "avx2") {
        return true;
    }

    x86_features() & x86_bits::AVX2 != 0
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn sse41_detected() -> bool {
    if cfg!(miri) {
        return false;
    }

    // A build for CPUs that have it (`-C target-cpu=...` or
    // `-C target-feature=...`) needs no check: run elsewhere, it stops
    // at its first such instruction (SIGILL), in the startup self-test.
    if cfg!(target_feature = "sse4.1") {
        return true;
    }

    x86_features() & x86_bits::SSE41 != 0
}

/// The core's virtual counter (CNTVCT_EL0, which user code reads
/// directly): control, not measurement.
#[cfg(all(blake3_neon_hybrid, feature = "std"))]
#[inline]
pub(crate) fn counter() -> u64 {
    let now: u64;
    // Sound: readable at EL0 on every AArch64 OS this crate builds the
    // hybrids for (Linux, macOS).
    unsafe { core::arch::asm!("mrs {0}, cntvct_el0", out(reg) now, options(nomem, nostack, preserves_flags)) };
    now
}

/// The counter's ticks in 100 us (CNTFRQ_EL0 / 10,000: 2400 on Apple's
/// cores), read once: reading the frequency costs about 10 cycles on an M4
/// Max (probe/stamp-cost, job 1015), the counter itself next to nothing.
#[cfg(all(blake3_neon_hybrid, feature = "std"))]
#[inline]
pub(crate) fn ticks_in_100_us() -> u64 {
    static TICKS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let ticks = TICKS.load(core::sync::atomic::Ordering::Relaxed);
    if ticks != 0 {
        return ticks;
    }
    let per_second: u64;
    // Sound: as counter().
    unsafe { core::arch::asm!("mrs {0}, cntfrq_el0", out(reg) per_second, options(nomem, nostack, preserves_flags)) };
    let ticks = (per_second / 10_000).max(1);
    TICKS.store(ticks, core::sync::atomic::Ordering::Relaxed);
    ticks
}

/// Prefetch code from `start` to `end` into the L2 cache, a line (128
/// bytes on Apple's cores) at a time. A call after other work otherwise
/// fetches its kernel's lines from DRAM one miss after another: hash() of
/// 8 KiB took 4.8 us so on an M4 Max, and 2.8 with this prefetch (job 971),
/// where code already near costs it a few cycles a line.
#[cfg(all(blake3_neon_hybrid, feature = "std"))]
#[inline]
pub(crate) fn prefetch_code((start, end): (usize, usize)) {
    debug_assert!(start < end && end - start < 64 * 1024, "a kernel's code");
    // Four lines a step (past `end` by up to three: harmless).
    let mut line = start;
    while line < end {
        // Sound: a prefetch never faults, whatever the address.
        unsafe {
            core::arch::asm!(
                "prfm pldl2keep, [{0}]",
                "prfm pldl2keep, [{0}, #128]",
                "prfm pldl2keep, [{0}, #256]",
                "prfm pldl2keep, [{0}, #384]",
                in(reg) line,
                options(nostack, readonly, preserves_flags),
            )
        };
        line += 512;
    }
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

    if !sme2_reported() {
        return false;
    }

    // Vector length check: the kernel reports it without doing any work
    // when asked for zero groups.
    let lanes = unsafe { crate::sme2::ffi::blake3_sme2_hash16_chunks_512(core::ptr::null(), core::ptr::null(), 0, 0, core::ptr::null_mut(), 0) };
    lanes == crate::sme2::GROUP as u64
}

#[cfg(all(blake3_sme2, target_vendor = "apple"))]
fn sme2_reported() -> bool {
    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();
    let rc =
        unsafe { libc::sysctlbyname(c"hw.optional.arm.FEAT_SME2".as_ptr(), &mut value as *mut u32 as *mut libc::c_void, &mut size, core::ptr::null_mut(), 0) };
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

#[cfg(test)]
mod sme2_turn_test {
    use super::*;

    /// While one call holds the SME2 turn, another runs NEON; calls too
    /// small for the SME2 kernels keep their platform. Other tests may hold
    /// the turn meanwhile, so only what holding it guarantees is asserted.
    #[test]
    fn test_one_sme2_call_at_a_time() {
        let detected = Platform::detect();
        let small = Sme2Turn::take(detected, false);
        assert!(core::mem::discriminant(&small.platform()) == core::mem::discriminant(&detected));
        #[cfg(blake3_sme2)]
        if matches!(detected, Platform::SME2) {
            let first = Sme2Turn::take(detected, true);
            if first.held {
                let second = Sme2Turn::take(detected, true);
                assert!(matches!(second.platform(), Platform::NEON) && !second.held);
                assert!(matches!(first.platform(), Platform::SME2));
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod x86_degree_test {
    use super::*;

    /// The detected platform hashes as many chunks at once as the widest
    /// kernel this CPU has (the standard library's detection, an
    /// independent reading, says which): a degree of 1 would hash
    /// correctly, one chunk at a time.
    #[test]
    fn test_detected_degree_is_the_widest_kernel() {
        let expected = if cfg!(blake3_avx512_ffi) && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            16
        } else if std::is_x86_feature_detected!("avx2") {
            8
        } else if std::is_x86_feature_detected!("sse4.1") {
            4
        } else {
            1
        };
        assert_eq!(Platform::detect().simd_degree(), expected);
        assert_eq!(Platform::Portable.simd_degree(), 1);
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod x86_features_test {
    /// The CPUID read finds what the standard library's detection, an
    /// independent reading, finds.
    #[test]
    fn test_x86_features_match_std() {
        use super::x86_bits::*;
        let features = super::read_x86_features();
        assert_eq!(features & READ, READ);
        assert_eq!(features & SSE41 != 0, std::is_x86_feature_detected!("sse4.1"));
        assert_eq!(features & AVX2 != 0, std::is_x86_feature_detected!("avx2"));
        assert_eq!(features & AVX512 != 0, std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl"));
    }
}
