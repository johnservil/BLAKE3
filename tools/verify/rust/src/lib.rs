//! One exported function per code path the proofs cover, each a call of
//! blake3-servil's own `Platform` method, unchanged.
use blake3_servil::platform::Platform;

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
pub extern "C" fn verify_neon_xof_many(cv: &[u32; 8], block: &[u8; 64], len: u8, counter: u64, flags: u8, out: *mut u8, blocks: usize) {
    // Safe: the caller gives `blocks * 64` writable bytes.
    let out = unsafe { core::slice::from_raw_parts_mut(out, blocks * 64) };
    Platform::NEON.xof_many(cv, block, len, counter, flags, out)
}
