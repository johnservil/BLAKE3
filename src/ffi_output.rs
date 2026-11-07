//! Eight or sixteen independent counter-zero compressions, one full-round compression
//! per lane. Descriptors borrow blocks and CVs; keys, lengths and flags may
//! differ between lanes. ROOT selects root digests; other blocks produce CVs.
#[cfg(test)]
use crate::hasher;
use crate::{BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, Hasher, OUT_LEN, ROOT, platform::Platform};

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
#[repr(C, align(64))]
struct Params8 {
    msg: [*const u8; 8],
    cvs: [[u32; 8]; 8],
    lens: [u32; 8],
    flags: [u32; 8],
}
const _: () = {
    assert!(core::mem::size_of::<Params8>() == 384);
    assert!(core::mem::offset_of!(Params8, cvs) == 64);
    assert!(core::mem::offset_of!(Params8, lens) == 320);
    assert!(core::mem::offset_of!(Params8, flags) == 352);
};
unsafe extern "C" {
    fn blake3_servil_output16(inputs: *const Input<'_>, out: *mut u8);
    fn blake3_servil_output8(params: *const Params8, out: *mut u8);
}

pub(crate) fn lanes(platform: Platform) -> usize {
    match platform {
        Platform::AVX512 => 16,
        Platform::AVX2 => 8,
        _ => 0,
    }
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
        flags: c.flags | c.start_flag() | CHUNK_END | ROOT,
    })
}

/// Requires AVX2 for W=8, AVX-512F+VL for W=16, on Linux. Every descriptor
/// borrows a readable padded block/CV, length 0..64. All counters are zero.
pub(crate) unsafe fn hash_group<const W: usize>(inputs: &[Input<'_>; W], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(out.len(), W, "one compression output per lane");
    if W == 16 {
        crate::ran_kernel(blake3_servil_output16 as *const ());
        // Sound: caller supplies the ISA; fixed descriptors and checked
        // output provide readable inputs and room for every store.
        unsafe { blake3_servil_output16(inputs.as_ptr(), out.as_mut_ptr().cast()) };
    } else {
        assert_eq!(W, 8, "eight or sixteen compression lanes");
        let p = Params8 {
            msg: core::array::from_fn(|i| inputs[i].block.as_ptr()),
            cvs: core::array::from_fn(|w| core::array::from_fn(|i| inputs[i].cv[w])),
            lens: core::array::from_fn(|i| inputs[i].len as u32),
            flags: core::array::from_fn(|i| inputs[i].flags as u32),
        };
        crate::ran_kernel(blake3_servil_output8 as *const ());
        // Sound: caller supplies AVX2; all borrowed blocks reach64bytes,
        // and the parameter layout and output size are checked above.
        unsafe { blake3_servil_output8(&p, out.as_mut_ptr().cast()) };
    }
}

/// Gather a bulk caller's first chunk without serial prefix compression.
/// The existing chunk buffer holds its first block; the existing stage holds
/// the suffix. The next bulk finalization compresses the prefixes together.
pub(crate) fn gather_first(hasher: &mut Hasher, bytes: &[u8]) -> bool {
    let c = &mut hasher.core.chunk_state;
    if hasher.core.initial_chunk_counter != 0
        || c.chunk_counter != 0
        || c.blocks_compressed != 0
        || hasher.core.cv_stack.len() != 0
        || (hasher.staged != 0 && c.buf_len as usize != BLOCK_LEN)
        || c.buf_len as usize + hasher.staged + bytes.len() > CHUNK_LEN
    {
        return false;
    }
    let take = bytes.len().min(BLOCK_LEN - c.buf_len as usize);
    let start = c.buf_len as usize;
    c.buf[start..start + take].copy_from_slice(&bytes[..take]);
    c.buf_len += take as u8;
    hasher.stage(&bytes[take..]);
    true
}

/// Finalize a full SIMD group of first-chunk states whose uncompressed suffixes wait in
/// their stages. Hashers remain immutable. Finished lanes keep their digest
/// while longer lanes complete their prefix blocks.
pub(crate) unsafe fn staged_group<const W: usize>(hashers: &[Hasher], which: &[usize; W], out: &mut [[u8; OUT_LEN]]) -> bool {
    let states = which.map(|i| &hashers[i]);
    if states.iter().any(|h| {
        let c = &h.core.chunk_state;
        h.core.initial_chunk_counter != 0
            || c.chunk_counter != 0
            || c.blocks_compressed != 0
            || h.core.cv_stack.len() != 0
            || (h.staged != 0 && c.buf_len as usize != BLOCK_LEN)
            || h.staged + c.buf_len as usize > CHUNK_LEN
    }) {
        return false;
    }
    let ends: [usize; W] = core::array::from_fn(|i| (states[i].core.chunk_state.buf_len as usize + states[i].staged).saturating_sub(1) / BLOCK_LEN);
    let lens: [u8; W] = core::array::from_fn(|i| (states[i].core.chunk_state.buf_len as usize + states[i].staged - ends[i] * BLOCK_LEN) as u8);
    let mut tails = [[0u8; BLOCK_LEN]; W];
    for i in 0..W {
        if states[i].staged == 0 {
            tails[i] = states[i].core.chunk_state.buf;
        } else {
            let start = (ends[i] - 1) * BLOCK_LEN;
            tails[i][..lens[i] as usize].copy_from_slice(&states[i].staged_bytes()[start..]);
        }
    }
    let mut cvs: [[u32; 8]; W] = core::array::from_fn(|i| states[i].core.chunk_state.cv);
    let mut values = [[0u8; OUT_LEN]; W];
    for block in 0..=*ends.iter().max().unwrap() {
        let inputs: [Input<'_>; W] = core::array::from_fn(|i| {
            let last = block >= ends[i];
            let bytes = if last {
                &tails[i]
            } else if block == 0 {
                &states[i].core.chunk_state.buf
            } else {
                states[i].staged_bytes()[(block - 1) * BLOCK_LEN..block * BLOCK_LEN].try_into().unwrap()
            };
            Input {
                block: bytes,
                cv: &cvs[i],
                len: if last { lens[i] } else { BLOCK_LEN as u8 },
                flags: states[i].core.chunk_state.flags | if block == 0 { CHUNK_START } else { 0 } | if last { CHUNK_END | ROOT } else { 0 },
            }
        });
        // Sound: caller supplies AVX-512F+VL; every pointer borrows a readable
        // full block and CV, including the padded final blocks above.
        unsafe { hash_group(&inputs, &mut values) };
        for i in 0..W {
            if block == ends[i] {
                out[i] = values[i];
            } else if block < ends[i] {
                cvs[i] = core::array::from_fn(|w| u32::from_le_bytes(values[i][4 * w..4 * w + 4].try_into().unwrap()));
            }
        }
    }
    true
}

const SELF_TEST_EXPECTED: u64 = 0x153e0dfbeac2bb44;
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
    let inputs = core::array::from_fn(|i| Input { block: &blocks[i], cv: &cvs[i], len: lens[i], flags: flags[i] | if (i / 4) % 2 == 0 { ROOT } else { 0 } });
    let mut out = [[0u8; OUT_LEN]; 16];
    hash(&inputs, &mut out);
    crate::self_test::fold(out.as_flattened())
}
fn check_self_test(answer: u64) {
    assert_eq!(answer, SELF_TEST_EXPECTED, "blake3-servil startup self-test failed: sixteen root outputs");
}
unsafe fn hash_eight_groups(inputs: &[Input<'_>; 16], out: &mut [[u8; OUT_LEN]]) {
    for (inputs, out) in inputs.chunks_exact(8).zip(out.chunks_mut(8)) {
        unsafe { hash_group::<8>(inputs.try_into().unwrap(), out) };
    }
}
unsafe fn hash_on_platform(platform: Platform, inputs: &[Input<'_>; 16], out: &mut [[u8; OUT_LEN]]) {
    if lanes(platform) == 16 {
        unsafe { hash_group::<16>(inputs, out) };
    } else {
        assert_eq!(lanes(platform), 8);
        unsafe { hash_eight_groups(inputs, out) };
    }
}
pub(crate) fn self_test(platform: Platform) -> usize {
    if lanes(platform) == 0 {
        return 0;
    }
    let answer = self_test_shape(|inputs, out| {
        // Sound: initialize detected this platform before testing its kernels.
        unsafe { hash_on_platform(platform, inputs, out) };
        if lanes(platform) == 16 && crate::platform::avx2_detected() {
            let mut eight = [[0u8; OUT_LEN]; 16];
            unsafe { hash_eight_groups(inputs, &mut eight) };
            check_self_test(crate::self_test::fold(eight.as_flattened()));
        }
    });
    check_self_test(answer);
    if lanes(platform) == 16 && crate::platform::avx2_detected() { 2 } else { 1 }
}
#[cfg(test)]
pub(crate) fn entry(width: usize) -> *const () {
    if width == 8 { blake3_servil_output8 as *const () } else { blake3_servil_output16 as *const () }
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
                unsafe { crate::sse41::compress_in_place(&mut cv, input.block, input.len, 0, input.flags) };
                *digest = hasher::le_bytes(&cv);
            }
        });
        assert_eq!(answer, SELF_TEST_EXPECTED);
        if lanes(Platform::detect()) != 0 {
            assert_eq!(self_test_shape(|inputs, out| unsafe { hash_on_platform(Platform::detect(), inputs, out) }), answer);
        }
    }
    #[test]
    fn independent_blocks_cvs_descriptors_and_output_against_guard_pages() {
        use crate::test::guard_pages::Guarded;
        crate::initialize();
        if lanes(Platform::detect()) == 0 {
            return;
        }
        for at_end in [false, true] {
            let answer = self_test_shape(|inputs, out| {
                let mut blocks: Vec<_> = (0..16).map(|_| Guarded::new(64, at_end)).collect();
                let mut cvs: Vec<_> = (0..16).map(|_| Guarded::new(32, at_end)).collect();
                let mut table = Guarded::new(core::mem::size_of_val(inputs), at_end);
                let table_ptr = table.bytes().as_mut_ptr().cast::<Input<'_>>();
                for i in 0..16 {
                    blocks[i].bytes().copy_from_slice(inputs[i].block);
                    cvs[i].bytes().copy_from_slice(&hasher::le_bytes(inputs[i].cv));
                    // Page placement aligns CVs to u32 and descriptors to
                    // Input. The guarded mappings outlive every borrowed view.
                    let block = unsafe { &*blocks[i].bytes().as_ptr().cast::<[u8; 64]>() };
                    let cv = unsafe { &*cvs[i].bytes().as_ptr().cast::<[u32; 8]>() };
                    unsafe { table_ptr.add(i).write(Input { block, cv, len: inputs[i].len, flags: inputs[i].flags }) };
                }
                for page in blocks.iter_mut().chain(cvs.iter_mut()) {
                    page.read_only();
                }
                table.read_only();
                let descriptors = unsafe { &*table_ptr.cast::<[Input<'_>; 16]>() };
                let mut output = Guarded::new(16 * OUT_LEN, at_end);
                let guarded_out = output.bytes().as_chunks_mut::<OUT_LEN>().0;
                unsafe { hash_on_platform(Platform::detect(), descriptors, guarded_out) };
                let expected = crate::self_test::fold(guarded_out.as_flattened());
                if crate::platform::avx2_detected() {
                    unsafe { hash_eight_groups(descriptors, guarded_out) };
                    assert_eq!(crate::self_test::fold(guarded_out.as_flattened()), expected);
                }
                out.copy_from_slice(guarded_out);
            });
            assert_eq!(answer, SELF_TEST_EXPECTED, "guard placement at_end={at_end}");
        }
    }

    #[test]
    fn startup_rejects_corrupted_answer() {
        assert!(std::panic::catch_unwind(|| check_self_test(SELF_TEST_EXPECTED ^ 1)).is_err());
    }

    #[test]
    fn incremental_roots_batch_all_modes_and_selected_indices() {
        crate::initialize();
        for count in [0usize, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 65] {
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
            let maximum = lanes(Platform::detect());
            for width in [8, 16] {
                let reached = crate::RAN_ENTRIES.with(|r| r.borrow().contains(&(entry(width) as usize)));
                let expected = match (maximum, width) {
                    (16, 16) => count >= 16,
                    (16, 8) => count % 16 >= 8 && crate::platform::avx2_detected(),
                    (8, 8) => count >= 8,
                    _ => false,
                };
                assert_eq!(reached, expected, "maximum={maximum} count={count} width={width}");
            }
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
    fn bulk_first_chunk_prefixes_all_modes_and_continuations() {
        crate::initialize();
        for count in [2, 15, 16, 17, 31, 32, 65] {
            let lens = [0usize, 1, 63, 64, 65, 127, 128, 255, 512, 1023, 1024];
            let keys: Vec<_> = (0..count).map(|i| [i as u8; 32]).collect();
            let data: Vec<Vec<u8>> = (0..count).map(|i| (0..lens[i % lens.len()]).map(|j| (i * 37 + j * 11) as u8).collect()).collect();
            let mode = |i: usize| match i % 3 {
                0 => crate::Mode::Plain,
                1 => crate::Mode::Keyed(&keys[i]),
                _ => crate::Mode::DeriveKey("BLAKE3-Servil bulk prefix test"),
            };
            let mut hashers: Vec<_> = (0..count).map(|i| Hasher::new(mode(i))).collect();
            // Ordinary and grouped updates can alternate on the same state.
            for (h, bytes) in hashers.iter_mut().zip(&data) {
                h.update(&bytes[..bytes.len().min(3)]);
            }
            for from in (3..1024).step_by(17) {
                let pieces: Vec<_> = data.iter().enumerate().map(|(i, bytes)| (i, &bytes[from.min(bytes.len())..(from + 17).min(bytes.len())])).collect();
                crate::each::update_each(&mut hashers, &pieces);
            }
            let which: Vec<_> = (0..count).rev().collect();
            let wanted: Vec<_> = which.iter().map(|&i| crate::test::official::hash(mode(i), &data[i])).collect();
            let mut out = vec![[0u8; OUT_LEN]; count];
            crate::each::finalize_each(&hashers, &which, &mut out);
            assert_eq!(out, wanted);
            crate::each::finalize_each(&hashers, &which, &mut out);
            assert_eq!(out, wanted, "immutable reusable staged states");
            for (i, h) in hashers.iter().enumerate() {
                assert_eq!(h.count(), data[i].len() as u64);
                assert_eq!(h.finalize().as_bytes(), &crate::test::official::hash(mode(i), &data[i]));
            }
            // Continue across the chunk and 16-KiB staging boundaries.
            let suffix = vec![0xacu8; 17003];
            let pieces: Vec<_> = (0..count).map(|i| (i, suffix.as_slice())).collect();
            crate::each::update_each(&mut hashers, &pieces);
            crate::each::finalize_each(&hashers, &which, &mut out);
            for (slot, &i) in which.iter().enumerate() {
                let mut whole = data[i].clone();
                whole.extend_from_slice(&suffix);
                assert_eq!(out[slot], crate::test::official::hash(mode(i), &whole));
            }
        }
    }

    #[test]
    fn independently_ending_staged_prefixes_at_every_position() {
        crate::initialize();
        for rotation in 0..16 {
            for phase in [0usize, 1, 7, 31, 62, 63] {
                let mut hashers: Vec<_> = (0..16).map(|_| Hasher::new(crate::Mode::Plain)).collect();
                let data: Vec<Vec<u8>> = (0..16)
                    .map(|i| {
                        let len = if i == 0 && phase == 0 { 0 } else { 64 * ((i + rotation) % 16) + 1 + (i + phase) % 64 };
                        (0..len).map(|j| (i * 73 + j * 11) as u8).collect()
                    })
                    .collect();
                for start in (0..1024).step_by(17) {
                    let pieces: Vec<_> = data.iter().enumerate().map(|(i, bytes)| (i, &bytes[start.min(bytes.len())..(start + 17).min(bytes.len())])).collect();
                    crate::each::update_each(&mut hashers, &pieces);
                }
                let which: Vec<_> = (0..16).rev().collect();
                let mut out = [[0u8; OUT_LEN]; 16];
                crate::each::finalize_each(&hashers, &which, &mut out);
                for (slot, &i) in which.iter().enumerate() {
                    assert_eq!(out[slot], crate::test::official::hash(crate::Mode::Plain, &data[i]), "rotation={rotation} phase={phase} lane={i}");
                }
            }
        }
    }

    #[test]
    fn staged_long_roots_keep_the_existing_path() {
        crate::initialize();
        let mut hashers = Vec::new();
        let mut expected = Vec::new();
        for i in 0..32 {
            let data = vec![i as u8; if i % 8 == 3 { 4097 } else { 129 }];
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
        for width in [8, 16] {
            assert!(!crate::RAN_ENTRIES.with(|r| r.borrow().contains(&(entry(width) as usize))));
        }
    }
}
