//! Sixteen independent final root compressions, one full-round compression
//! per lane. Descriptors borrow existing chunk-state blocks and CVs; all keys,
//! CVs, lengths and mode/structural flags may differ between lanes.
use crate::{BLOCK_LEN, CHUNK_END, Hasher, OUT_LEN, platform::Platform};
#[cfg(test)]
use crate::{ROOT, hasher};

#[repr(C)]
pub(crate) struct Input<'a> {
    block: &'a [u8; BLOCK_LEN],
    cv: &'a [u32; 8],
    len: u8,
    flags: u8,
}
const _: () = {
    assert!(core::mem::size_of::<Input<'static>>() == 24);
    assert!(core::mem::offset_of!(Input<'static>, block) == 0);
    assert!(core::mem::offset_of!(Input<'static>, cv) == 8);
    assert!(core::mem::offset_of!(Input<'static>, len) == 16);
    assert!(core::mem::offset_of!(Input<'static>, flags) == 17);
};
unsafe extern "C" {
    fn blake3_servil_output16(inputs: *const Input<'_>, out: *mut u8);
}

/// A whole message still in its first chunk, with no staged suffix or
/// stacked subtree. Its final block is already padded by ChunkState.
/// Other states keep Hasher's existing finalization path.
pub(crate) fn input(hasher: &Hasher) -> Option<Input<'_>> {
    let c = &hasher.core.chunk_state;
    (hasher.staged == 0 && hasher.core.initial_chunk_counter == 0 && c.chunk_counter == 0 && hasher.core.cv_stack.len() == 0).then(|| Input {
        block: &c.buf,
        cv: &c.cv,
        len: c.buf_len,
        flags: c.flags | c.start_flag() | CHUNK_END,
    })
}

/// Requires AVX-512F+VL on Linux. Each descriptor holds a readable padded
/// block and CV; its block length is 0..64. The output holds 16 digests.
pub(crate) unsafe fn hash_group(inputs: &[Input<'_>; 16], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(out.len(), 16, "sixteen root outputs");
    crate::ran_kernel(blake3_servil_output16 as *const ());
    // Sound: caller supplies the ISA; references and fixed descriptor count
    // provide the readable data, and the checked output holds all stores.
    unsafe { blake3_servil_output16(inputs.as_ptr(), out.as_mut_ptr().cast()) };
}

const SELF_TEST_EXPECTED: u64 = 0x54e3da46af4eb775;
fn self_test_shape(mut hash: impl FnMut(&[Input<'_>; 16], &mut [[u8; OUT_LEN]])) -> u64 {
    let mut lens: [u8; 16] = [0, 1, 31, 63, 64, 7, 32, 55, 3, 15, 48, 60, 2, 33, 62, 64];
    let flags: [u8; 16] = core::array::from_fn(|i| {
        let shape = [3u8, 2, 4][i % 3];
        if shape == 4 {
            lens[i] = 64;
        }
        shape | [0u8, 16, 32, 64][i % 4]
    });
    let cvs: [[u32; 8]; 16] = core::array::from_fn(|i| {
        core::array::from_fn(|w| (0x1234abcdu32.wrapping_add(i as u32 * 0x9e37).wrapping_add(w as u32 * 0x1020304)).wrapping_mul(0x85ebca6b))
    });
    let blocks: [[u8; 64]; 16] = core::array::from_fn(|i| core::array::from_fn(|j| if j < lens[i] as usize { ((i * 37 + j * 17) ^ 0x5a) as u8 } else { 0 }));
    let inputs = core::array::from_fn(|i| Input { block: &blocks[i], cv: &cvs[i], len: lens[i], flags: flags[i] });
    let mut out = [[0u8; OUT_LEN]; 16];
    hash(&inputs, &mut out);
    crate::self_test::fold(out.as_flattened())
}
fn check_self_test(answer: u64) {
    assert_eq!(answer, SELF_TEST_EXPECTED, "blake3-servil startup self-test failed: sixteen root outputs");
}
pub(crate) fn self_test(platform: Platform) -> usize {
    if !matches!(platform, Platform::AVX512) {
        return 0;
    }
    let answer = self_test_shape(|inputs, out| {
        // Sound: initialize detected this platform before testing its kernels.
        unsafe { hash_group(inputs, out) };
    });
    check_self_test(answer);
    1
}
#[cfg(test)]
pub(crate) fn entry() -> *const () {
    blake3_servil_output16 as *const ()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn startup_answers_match_independent_compression() {
        crate::initialize();
        let answer = self_test_shape(|inputs, out| {
            for (input, digest) in inputs.iter().zip(out) {
                let mut cv = *input.cv;
                unsafe { crate::sse41::compress_in_place(&mut cv, input.block, input.len, 0, input.flags | ROOT) };
                *digest = hasher::le_bytes(&cv);
            }
        });
        assert_eq!(answer, SELF_TEST_EXPECTED);
        if matches!(Platform::detect(), Platform::AVX512) {
            assert_eq!(self_test_shape(|inputs, out| unsafe { hash_group(inputs, out) }), answer);
        }
    }
    #[test]
    fn startup_rejects_corrupted_answer() {
        assert!(std::panic::catch_unwind(|| check_self_test(SELF_TEST_EXPECTED ^ 1)).is_err());
    }

    #[test]
    fn incremental_roots_batch_all_modes_and_selected_indices() {
        crate::initialize();
        for count in [0usize, 1, 15, 16, 17, 31, 32, 65] {
            let mut hashers = Vec::new();
            let mut expected = Vec::new();
            for i in 0..count {
                let len = [0usize, 1, 31, 63, 64, 65, 127, 255, 511, 1023, 1024][i % 11];
                let data: Vec<u8> = (0..len).map(|j| (i * 37 + j * 17) as u8).collect();
                let key = [i as u8; 32];
                let mode = match i % 3 {
                    0 => crate::Mode::Plain,
                    1 => crate::Mode::Keyed(&key),
                    _ => crate::Mode::DeriveKey("BLAKE3-Servil bulk-root test"),
                };
                let mut h = Hasher::new(mode);
                for part in data.chunks(17) {
                    h.update(part);
                }
                expected.push(crate::test::official::hash(mode, &data));
                hashers.push(h);
            }
            let which: Vec<usize> = (0..count).rev().collect();
            let mut out = vec![[0u8; 32]; count];
            crate::RAN_ENTRIES.with(|r| r.borrow_mut().clear());
            crate::each::finalize_each(&hashers, &which, &mut out);
            let wanted: Vec<_> = which.iter().map(|&i| expected[i]).collect();
            assert_eq!(out, wanted);
            let reached = crate::RAN_ENTRIES.with(|r| r.borrow().contains(&(entry() as usize)));
            assert_eq!(reached, count >= 16 && matches!(Platform::detect(), Platform::AVX512));
            crate::each::finalize_each(&hashers, &which, &mut out);
            assert_eq!(out, wanted, "finalization leaves the input states reusable");
            if count > 0 {
                let repeated = vec![0usize; count];
                crate::each::finalize_each(&hashers, &repeated, &mut out);
                assert!(out.iter().all(|x| *x == expected[0]));
            }
        }
    }

    #[test]
    fn staged_long_roots_keep_the_existing_path() {
        crate::initialize();
        let mut hashers = Vec::new();
        let mut expected = Vec::new();
        for i in 0..32 {
            let data = vec![i as u8; if i == 3 || i == 20 { 4097 } else { 129 }];
            let mut h = Hasher::new(crate::Mode::Plain);
            for piece in data.chunks(23) {
                h.update(piece);
            }
            expected.push(crate::test::official::hash(crate::Mode::Plain, &data));
            hashers.push(h);
        }
        let which: Vec<_> = (0..32).collect();
        let mut out = vec![[0u8; 32]; 32];
        crate::RAN_ENTRIES.with(|r| r.borrow_mut().clear());
        crate::each::finalize_each(&hashers, &which, &mut out);
        assert_eq!(out, expected);
        assert!(!crate::RAN_ENTRIES.with(|r| r.borrow().contains(&(entry() as usize))));
    }
}
