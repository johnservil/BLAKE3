use crate::{BLOCK_LEN, CVWords, IncrementCounter, OUT_LEN};

// Unsafe because this may only be called on platforms supporting NEON.
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
    // The Rust hash_many implementations do bounds checking on the `out`
    // array, but the C implementations don't. Even though this is an unsafe
    // function, assert the bounds here.
    assert!(out.len() >= inputs.len() * OUT_LEN);
    // On AArch64 cores with the SHA-3 extension, the integer + vector
    // kernels in neon_hybrid.rs are faster than the four-lane C kernel at
    // every input count (see that module). The C kernel stays for
    // 32-bit ARM and for cores without `xar`.
    #[cfg(blake3_neon_hybrid)]
    {
        let shape_covered = crate::neon_hybrid::covers(N, increment_counter);
        if shape_covered && crate::neon_hybrid::sha3_detected() {
            unsafe {
                crate::neon_hybrid::hash_many(
                    inputs,
                    key,
                    counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                );
            }
            return;
        }
    }
    unsafe {
        ffi::blake3_hash_many_neon(
            inputs.as_ptr() as *const *const u8,
            inputs.len(),
            N / BLOCK_LEN,
            key.as_ptr(),
            counter,
            increment_counter.yes(),
            flags,
            flags_start,
            flags_end,
            out.as_mut_ptr(),
        )
    }
}

// blake3_neon.c normally depends on blake3_portable.c, because the NEON
// implementation only provides 4x compression, and it relies on the portable
// implementation for 1x compression. However, we expose the portable Rust
// implementation here instead, to avoid linking in unnecessary code.
#[unsafe(export_name = "blake3_sme2_compress_in_place_portable")]
pub extern "C" fn blake3_compress_in_place_portable(
    cv: *mut u32,
    block: *const u8,
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    unsafe {
        crate::portable::compress_in_place(
            &mut *(cv as *mut [u32; 8]),
            &*(block as *const [u8; 64]),
            block_len,
            counter,
            flags,
        )
    }
}

pub mod ffi {
    unsafe extern "C" {
        #[link_name = "blake3_sme2_hash_many_neon"]
        pub fn blake3_hash_many_neon(
            inputs: *const *const u8,
            num_inputs: usize,
            blocks: usize,
            key: *const u32,
            counter: u64,
            increment_counter: bool,
            flags: u8,
            flags_start: u8,
            flags_end: u8,
            out: *mut u8,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hash_many() {
        // This entire file is gated on feature="neon", so NEON support is
        // assumed here.
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }
}
