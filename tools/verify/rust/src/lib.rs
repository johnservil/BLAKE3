//! One exported function per code path the proofs cover, each a call of
//! blake3-servil's own `Platform` method, unchanged.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;

#[unsafe(no_mangle)]
pub extern "C" fn verify_portable_compress_in_place(cv: &mut [u32; 8], block: &[u8; 64], len: u8, counter: u64, flags: u8) {
    Platform::Portable.compress_in_place(cv, block, len, counter, flags)
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_portable_compress_xof(cv: &[u32; 8], block: &[u8; 64], len: u8, counter: u64, flags: u8, out: &mut [u8; 64]) {
    *out = Platform::Portable.compress_xof(cv, block, len, counter, flags)
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_neon_compress_in_place(cv: &mut [u32; 8], block: &[u8; 64], len: u8, counter: u64, flags: u8) {
    Platform::NEON.compress_in_place(cv, block, len, counter, flags)
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_sme2_compress_in_place(cv: &mut [u32; 8], block: &[u8; 64], len: u8, counter: u64, flags: u8) {
    Platform::SME2.compress_in_place(cv, block, len, counter, flags)
}

macro_rules! compress_blocks {
    ($name:ident, $platform:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(cv: &mut [u32; 8], blocks: *const u8, len: usize, count: usize, counter: u64, flags: u8, flags_start: u8) {
            // Safe: the caller gives `len` readable bytes.
            let blocks = unsafe { core::slice::from_raw_parts(blocks, len) };
            $platform.compress_blocks(cv, blocks, count, counter, flags, flags_start)
        }
    };
}
compress_blocks!(verify_portable_compress_blocks, Platform::Portable);
compress_blocks!(verify_neon_compress_blocks, Platform::NEON);
compress_blocks!(verify_sme2_compress_blocks, Platform::SME2);

macro_rules! hash_many {
    ($name:ident, $platform:expr, $n:expr, $increment:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(inputs: *const &[u8; $n], count: usize, key: &[u32; 8], counter: u64, flags: u8, flags_start: u8, flags_end: u8, out: *mut u8) {
            // Safe: the caller gives `count` inputs and `count * 32` writable bytes.
            let inputs = unsafe { core::slice::from_raw_parts(inputs, count) };
            let out = unsafe { core::slice::from_raw_parts_mut(out, count * 32) };
            $platform.hash_many(inputs, key, counter, $increment, flags, flags_start, flags_end, out)
        }
    };
}
hash_many!(verify_portable_hash_chunks, Platform::Portable, 1024, IncrementCounter::Yes);
hash_many!(verify_portable_hash_parents, Platform::Portable, 64, IncrementCounter::No);
hash_many!(verify_neon_hash_chunks, Platform::NEON, 1024, IncrementCounter::Yes);
hash_many!(verify_neon_hash_parents, Platform::NEON, 64, IncrementCounter::No);
hash_many!(verify_sme2_hash_chunks, Platform::SME2, 1024, IncrementCounter::Yes);
hash_many!(verify_sme2_hash_parents, Platform::SME2, 64, IncrementCounter::No);

#[unsafe(no_mangle)]
pub extern "C" fn verify_neon_xof_many(cv: &[u32; 8], block: &[u8; 64], len: u8, counter: u64, flags: u8, out: *mut u8, blocks: usize) {
    // Safe: the caller gives `blocks * 64` writable bytes.
    let out = unsafe { core::slice::from_raw_parts_mut(out, blocks * 64) };
    Platform::NEON.xof_many(cv, block, len, counter, flags, out)
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_hasher(input: *const u8, len: usize, out: &mut [u8; 32]) {
    // Safe: the caller gives `len` readable bytes. The Hasher's stack code
    // (HasherCore::merge_cv_stack, final_output) is proved as compiled here.
    let input = unsafe { core::slice::from_raw_parts(input, len) };
    *out = *blake3_servil::Hasher::new().update(input).finalize().as_bytes();
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_hash(input: *const u8, len: usize, out: &mut [u8; 32]) {
    // Safe: the caller gives `len` readable bytes. Whole calls, for the
    // timing proof: what a hash runs depends only on `len`.
    let input = unsafe { core::slice::from_raw_parts(input, len) };
    *out = *blake3_servil::hash(input).as_bytes();
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_keyed_hash(key: &[u8; 32], input: *const u8, len: usize, out: &mut [u8; 32]) {
    // Safe: the caller gives `len` readable bytes.
    let input = unsafe { core::slice::from_raw_parts(input, len) };
    *out = *blake3_servil::keyed_hash(key, input).as_bytes();
}

#[unsafe(no_mangle)]
pub extern "C" fn verify_hasher_split(input: *const u8, len: usize, split: usize, out: &mut [u8; 32]) {
    // Safe: the caller gives `len` readable bytes and split <= len. A
    // Hasher fed the bytes in two updates, cut at `split`.
    let input = unsafe { core::slice::from_raw_parts(input, len) };
    let mut hasher = blake3_servil::Hasher::new();
    hasher.update(&input[..split]);
    hasher.update(&input[split..]);
    *out = *hasher.finalize().as_bytes();
}

/// The test vectors' context string (test_vectors/test_vectors.json).
pub const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

#[unsafe(no_mangle)]
pub extern "C" fn verify_derive_key(input: *const u8, len: usize, out: &mut [u8; 32]) {
    // Safe: the caller gives `len` readable bytes.
    let input = unsafe { core::slice::from_raw_parts(input, len) };
    *out = blake3_servil::derive_key(CONTEXT, input);
}
