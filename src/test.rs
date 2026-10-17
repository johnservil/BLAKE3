use crate::{BLOCK_LEN, CHUNK_LEN, CVBytes, CVWords, IncrementCounter, OUT_LEN};
use arrayvec::ArrayVec;

// Interesting input lengths to run tests on.
pub const TEST_CASES: &[usize] = &[
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    BLOCK_LEN - 1,
    BLOCK_LEN,
    BLOCK_LEN + 1,
    2 * BLOCK_LEN - 1,
    2 * BLOCK_LEN,
    2 * BLOCK_LEN + 1,
    CHUNK_LEN - 1,
    CHUNK_LEN,
    CHUNK_LEN + 1,
    2 * CHUNK_LEN,
    2 * CHUNK_LEN + 1,
    3 * CHUNK_LEN,
    3 * CHUNK_LEN + 1,
    4 * CHUNK_LEN,
    4 * CHUNK_LEN + 1,
    5 * CHUNK_LEN,
    5 * CHUNK_LEN + 1,
    6 * CHUNK_LEN,
    6 * CHUNK_LEN + 1,
    7 * CHUNK_LEN,
    7 * CHUNK_LEN + 1,
    8 * CHUNK_LEN,
    8 * CHUNK_LEN + 1,
    16 * CHUNK_LEN - 1,
    16 * CHUNK_LEN, // AVX512's bandwidth
    16 * CHUNK_LEN + 1,
    31 * CHUNK_LEN - 1,
    31 * CHUNK_LEN, // 16 + 8 + 4 + 2 + 1
    31 * CHUNK_LEN + 1,
    100 * CHUNK_LEN, // subtrees larger than MAX_SIMD_DEGREE chunks
];

pub const TEST_CASES_MAX: usize = 100 * CHUNK_LEN;

// There's a test to make sure these two are equal below.
pub const TEST_KEY: CVBytes = *b"whats the Elvish word for friend";
pub const TEST_KEY_WORDS: CVWords = [
    1952540791, 1752440947, 1816469605, 1752394102, 1919907616, 1868963940, 1919295602, 1684956521,
];

// Paint the input with a repeating byte pattern. We use a cycle length of 251,
// because that's the largest prime number less than 256. This makes it
// unlikely to swapping any two adjacent input blocks or chunks will give the
// same answer.
pub fn paint_test_input(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
}

type CompressInPlaceFn =
    unsafe fn(cv: &mut CVWords, block: &[u8; BLOCK_LEN], block_len: u8, counter: u64, flags: u8);

type CompressXofFn = unsafe fn(
    cv: &CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) -> [u8; 64];

// A shared helper function for platform-specific tests.
pub fn test_compress_fn(compress_in_place_fn: CompressInPlaceFn, compress_xof_fn: CompressXofFn) {
    let initial_state = TEST_KEY_WORDS;
    let block_len: u8 = 61;
    let mut block = [0; BLOCK_LEN];
    paint_test_input(&mut block[..block_len as usize]);
    // Use a counter with set bits in both 32-bit words.
    let counter = (5u64 << 32) + 6;
    let flags = crate::CHUNK_END | crate::ROOT | crate::KEYED_HASH;

    let portable_out =
        crate::portable::compress_xof(&initial_state, &block, block_len, counter as u64, flags);

    let mut test_state = initial_state;
    unsafe { compress_in_place_fn(&mut test_state, &block, block_len, counter as u64, flags) };
    let test_state_bytes = crate::hasher::le_bytes(&test_state);
    let test_xof =
        unsafe { compress_xof_fn(&initial_state, &block, block_len, counter as u64, flags) };

    assert_eq!(&portable_out[..32], &test_state_bytes[..]);
    assert_eq!(&portable_out[..], &test_xof[..]);
}

type HashManyFn<A> = unsafe fn(
    inputs: &[&A],
    key: &CVWords,
    counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8],
);

// A shared helper function for platform-specific tests.
pub fn test_hash_many_fn(
    hash_many_chunks_fn: HashManyFn<[u8; CHUNK_LEN]>,
    hash_many_parents_fn: HashManyFn<[u8; 2 * OUT_LEN]>,
) {
    // Test a few different initial counter values.
    // - 0: The base case.
    // - u32::MAX: The low word of the counter overflows for all inputs except the first.
    // - i32::MAX: *No* overflow. But carry bugs in tricky SIMD code can screw this up, if you XOR
    //   when you're supposed to ANDNOT...
    let initial_counters = [0, u32::MAX as u64, i32::MAX as u64];
    for counter in initial_counters {
        #[cfg(feature = "std")]
        dbg!(counter);

        // 31 (16 + 8 + 4 + 2 + 1) inputs
        const NUM_INPUTS: usize = 31;
        let mut input_buf = [0; CHUNK_LEN * NUM_INPUTS];
        crate::test::paint_test_input(&mut input_buf);

        // First hash chunks.
        let mut chunks = ArrayVec::<&[u8; CHUNK_LEN], NUM_INPUTS>::new();
        for i in 0..NUM_INPUTS {
            chunks.push(
                (&input_buf[i * CHUNK_LEN..][..CHUNK_LEN])
                    .try_into()
                    .unwrap(),
            );
        }
        let mut portable_chunks_out = [0; NUM_INPUTS * OUT_LEN];
        crate::portable::hash_many(
            &chunks,
            &TEST_KEY_WORDS,
            counter,
            IncrementCounter::Yes,
            crate::KEYED_HASH,
            crate::CHUNK_START,
            crate::CHUNK_END,
            &mut portable_chunks_out,
        );

        let mut test_chunks_out = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_chunks_fn(
                &chunks[..],
                &TEST_KEY_WORDS,
                counter,
                IncrementCounter::Yes,
                crate::KEYED_HASH,
                crate::CHUNK_START,
                crate::CHUNK_END,
                &mut test_chunks_out,
            );
        }
        for n in 0..NUM_INPUTS {
            #[cfg(feature = "std")]
            dbg!(n);
            assert_eq!(
                &portable_chunks_out[n * OUT_LEN..][..OUT_LEN],
                &test_chunks_out[n * OUT_LEN..][..OUT_LEN]
            );
        }

        // Then hash parents.
        let mut parents = ArrayVec::<&[u8; 2 * OUT_LEN], NUM_INPUTS>::new();
        for i in 0..NUM_INPUTS {
            parents.push(
                (&input_buf[i * 2 * OUT_LEN..][..2 * OUT_LEN])
                    .try_into()
                    .unwrap(),
            );
        }
        let mut portable_parents_out = [0; NUM_INPUTS * OUT_LEN];
        crate::portable::hash_many(
            &parents,
            &TEST_KEY_WORDS,
            counter,
            IncrementCounter::No,
            crate::KEYED_HASH | crate::PARENT,
            0,
            0,
            &mut portable_parents_out,
        );

        let mut test_parents_out = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_parents_fn(
                &parents[..],
                &TEST_KEY_WORDS,
                counter,
                IncrementCounter::No,
                crate::KEYED_HASH | crate::PARENT,
                0,
                0,
                &mut test_parents_out,
            );
        }
        for n in 0..NUM_INPUTS {
            #[cfg(feature = "std")]
            dbg!(n);
            assert_eq!(
                &portable_parents_out[n * OUT_LEN..][..OUT_LEN],
                &test_parents_out[n * OUT_LEN..][..OUT_LEN]
            );
        }
    }
}

#[allow(unused)]
type XofManyFunction = unsafe fn(
    cv: &CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
    out: &mut [u8],
);

// A shared helper function for platform-specific tests.
#[allow(unused)]
pub fn test_xof_many_fn(xof_many_function: XofManyFunction) {
    let mut block = [0; BLOCK_LEN];
    let block_len = 42;
    crate::test::paint_test_input(&mut block[..block_len]);
    let cv = [40, 41, 42, 43, 44, 45, 46, 47];
    let flags = crate::KEYED_HASH;

    // Test a few different initial counter values.
    // - 0: The base case.
    // - u32::MAX: The low word of the counter overflows for all inputs except the first.
    // - i32::MAX: *No* overflow. But carry bugs in tricky SIMD code can screw this up, if you XOR
    //   when you're supposed to ANDNOT...
    let initial_counters = [0, u32::MAX as u64, i32::MAX as u64];
    for counter in initial_counters {
        #[cfg(feature = "std")]
        dbg!(counter);

        // 31 (16 + 8 + 4 + 2 + 1) outputs
        const OUTPUT_SIZE: usize = 31 * BLOCK_LEN;

        let mut portable_out = [0u8; OUTPUT_SIZE];
        for (i, out_block) in portable_out.chunks_exact_mut(64).enumerate() {
            out_block.copy_from_slice(&crate::portable::compress_xof(
                &cv,
                &block,
                block_len as u8,
                counter + i as u64,
                flags,
            ));
        }

        let mut test_out = [0u8; OUTPUT_SIZE];
        unsafe {
            xof_many_function(&cv, &block, block_len as u8, counter, flags, &mut test_out);
        }

        assert_eq!(portable_out, test_out);
    }

    // Test that xof_many doesn't write more blocks than requested. Note that the current assembly
    // implementation always outputs at least one block, so we don't test the zero case.
    for block_count in 1..=32 {
        let mut array = [0; BLOCK_LEN * 33];
        let output_start = 17;
        let output_len = block_count * BLOCK_LEN;
        let output_end = output_start + output_len;
        let output = &mut array[output_start..output_end];
        unsafe {
            xof_many_function(&cv, &block, block_len as u8, 0, flags, output);
        }
        for i in 0..array.len() {
            if i < output_start || output_end <= i {
                assert_eq!(0, array[i], "index {i}");
            }
        }
    }
}

#[test]
fn test_key_bytes_equal_key_words() {
    assert_eq!(
        TEST_KEY_WORDS,
        crate::platform::words_from_le_bytes_32(&TEST_KEY),
    );
}

#[test]
fn test_reference_impl_size() {
    // Because the Rust compiler optimizes struct layout, it's possible that
    // some future version of the compiler will produce a different size. If
    // that happens, we can either disable this test, or test for multiple
    // expected values. For now, the purpose of this test is to make sure we
    // notice if that happens.
    assert_eq!(1880, core::mem::size_of::<reference_impl::Hasher>());
}

#[test]
fn test_counter_words() {
    let counter: u64 = (1 << 32) + 2;
    assert_eq!(crate::counter_low(counter), 2);
    assert_eq!(crate::counter_high(counter), 1);
}

#[test]
fn test_subtree_len() {
    // The largest power of two at most `remaining` dividing the bytes so far.
    for counter in [0u64, 1, 2, 3, 4, 6, 8, 12, 1 << 20, (1 << 20) + 3, (1 << 54) - 1] {
        for remaining in [1025usize, 2047, 2048, 2049, 4096, 1 << 20, (1 << 20) + 1, usize::MAX] {
            let mut want = 1usize;
            while want <= remaining / 2 && (counter * 1024) % (2 * want as u64) == 0 {
                want *= 2;
            }
            assert_eq!(crate::hasher::subtree_len(counter, remaining), want, "{counter} {remaining}");
        }
    }
}

#[test]
fn test_compare_reference_impl() {
    const OUT: usize = 303; // more than 64, not a multiple of 4
    let mut input_buf = [0; TEST_CASES_MAX];
    paint_test_input(&mut input_buf);
    for &case in TEST_CASES {
        let input = &input_buf[..case];
        #[cfg(feature = "std")]
        dbg!(case);

        // regular
        {
            let mut reference_hasher = reference_impl::Hasher::new();
            reference_hasher.update(input);
            let mut expected_out = [0; OUT];
            reference_hasher.finalize(&mut expected_out);

            // all at once
            let test_out = crate::hash(input);
            assert_eq!(
                test_out,
                *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
            );
            // incremental
            let mut hasher = crate::Hasher::new();
            hasher.update(input);
            assert_eq!(
                hasher.finalize(),
                *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
            );
            assert_eq!(hasher.finalize(), test_out);
            // xof
            let mut extended = [0; OUT];
            hasher.finalize_xof().fill(&mut extended);
            assert_eq!(extended, expected_out);
        }

        // keyed
        {
            let mut reference_hasher = reference_impl::Hasher::new_keyed(&TEST_KEY);
            reference_hasher.update(input);
            let mut expected_out = [0; OUT];
            reference_hasher.finalize(&mut expected_out);

            // all at once
            let test_out = crate::keyed_hash(&TEST_KEY, input);
            assert_eq!(
                test_out,
                *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
            );
            // incremental
            let mut hasher = crate::Hasher::new_keyed(&TEST_KEY);
            hasher.update(input);
            assert_eq!(
                hasher.finalize(),
                *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
            );
            assert_eq!(hasher.finalize(), test_out);
            // xof
            let mut extended = [0; OUT];
            hasher.finalize_xof().fill(&mut extended);
            assert_eq!(extended, expected_out);
        }

        // derive_key
        {
            let context = "BLAKE3 2019-12-27 16:13:59 example context (not the test vector one)";
            let mut reference_hasher = reference_impl::Hasher::new_derive_key(context);
            reference_hasher.update(input);
            let mut expected_out = [0; OUT];
            reference_hasher.finalize(&mut expected_out);

            // all at once
            let test_out = crate::derive_key(context, input);
            assert_eq!(test_out, expected_out[..32]);
            // incremental
            let mut hasher = crate::Hasher::new_derive_key(context);
            hasher.update(input);
            assert_eq!(
                hasher.finalize(),
                *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
            );
            assert_eq!(
                hasher.finalize(),
                *<&[u8; 32]>::try_from(&test_out[..32]).unwrap()
            );
            // xof
            let mut extended = [0; OUT];
            hasher.finalize_xof().fill(&mut extended);
            assert_eq!(extended, expected_out);
        }
    }
}

#[test]
fn test_compare_reference_impl_long_xof() {
    let mut reference_output = [0u8; 32 * BLOCK_LEN - 1];
    let mut reference_hasher = reference_impl::Hasher::new_keyed(&TEST_KEY);
    reference_hasher.update(b"hello world");
    reference_hasher.finalize(&mut reference_output);

    let mut test_output = [0u8; 32 * BLOCK_LEN - 1];
    let mut test_hasher = crate::Hasher::new_keyed(&TEST_KEY);
    test_hasher.update(b"hello world");
    test_hasher.finalize_xof().fill(&mut test_output);

    assert_eq!(reference_output, test_output);
}

#[test]
fn test_xof_partial_blocks() {
    const OUT_LEN: usize = 6 * BLOCK_LEN;
    let mut reference_out = [0u8; OUT_LEN];
    reference_impl::Hasher::new().finalize(&mut reference_out);

    let mut all_at_once_out = [0u8; OUT_LEN];
    crate::Hasher::new()
        .finalize_xof()
        .fill(&mut all_at_once_out);
    assert_eq!(reference_out, all_at_once_out);

    let mut partial_out = [0u8; OUT_LEN];
    let partial_start = 32;
    let partial_end = OUT_LEN - 32;
    let mut xof = crate::Hasher::new().finalize_xof();
    xof.fill(&mut partial_out[..partial_start]);
    xof.fill(&mut partial_out[partial_start..partial_end]);
    xof.fill(&mut partial_out[partial_end..]);
    assert_eq!(reference_out, partial_out);
}

pub(crate) fn reference_hash(input: &[u8]) -> crate::Hash {
    let mut hasher = reference_impl::Hasher::new();
    hasher.update(input);
    let mut bytes = [0; 32];
    hasher.finalize(&mut bytes);
    bytes.into()
}

#[test]
fn test_compare_update_multiple() {
    // Don't use all the long test cases here, since that's unnecessarily slow
    // in debug mode.
    let mut short_test_cases = TEST_CASES;
    while *short_test_cases.last().unwrap() > 4 * CHUNK_LEN {
        short_test_cases = &short_test_cases[..short_test_cases.len() - 1];
    }
    assert_eq!(*short_test_cases.last().unwrap(), 4 * CHUNK_LEN);

    let mut input_buf = [0; 2 * TEST_CASES_MAX];
    paint_test_input(&mut input_buf);

    for &first_update in short_test_cases {
        #[cfg(feature = "std")]
        dbg!(first_update);
        let first_input = &input_buf[..first_update];
        let mut test_hasher = crate::Hasher::new();
        test_hasher.update(first_input);

        for &second_update in short_test_cases {
            #[cfg(feature = "std")]
            dbg!(second_update);
            let second_input = &input_buf[first_update..][..second_update];
            let total_input = &input_buf[..first_update + second_update];

            // Clone the hasher with first_update bytes already written, so
            // that the next iteration can reuse it.
            let mut test_hasher = test_hasher.clone();
            test_hasher.update(second_input);
            let expected = reference_hash(total_input);
            assert_eq!(expected, test_hasher.finalize());
        }
    }
}

/// Hashers fed three updates against the reference implementation, each
/// update's length one of the cases the staging (Hasher::gather_part)
/// and the core distinguish: none, a byte, either side of a chunk, either
/// side of the stage (crate::STAGE_LEN), and past two stages. Every order
/// of three, so each update meets every staging state the others leave.
#[test]
fn test_hasher_update_cases() {
    let lens = [0, 1, CHUNK_LEN - 1, CHUNK_LEN, CHUNK_LEN + 1, crate::STAGE_LEN - 1, crate::STAGE_LEN, crate::STAGE_LEN + 1, 2 * crate::STAGE_LEN + 1];
    let mut input = vec![0u8; 3 * (2 * crate::STAGE_LEN + 1)];
    paint_test_input(&mut input);
    for &a in &lens {
        for &b in &lens {
            for &c in &lens {
                let mut hasher = crate::Hasher::new();
                hasher.update(&input[..a]).update(&input[a..][..b]).update(&input[a + b..][..c]);
                assert_eq!(hasher.finalize(), reference_hash(&input[..a + b + c]), "updates of {a}, {b}, {c} bytes");
            }
        }
    }
}

/// Extended output against the reference implementation at every case its
/// reader distinguishes: a root of one block, of a whole multi-block chunk
/// (BLOCK_LEN + 1 bytes), and a parent; a read starting at a block's start,
/// its second byte, and its last; then 1 to 33 whole blocks (below, at,
/// and past SME2's group of 16 and NEON's groups of 8 and 4, every
/// remainder) and none, one, or BLOCK_LEN - 1 bytes more.
#[test]
fn test_xof_cases() {
    let mut input = [0u8; CHUNK_LEN + 1];
    paint_test_input(&mut input);
    for input_len in [0, BLOCK_LEN, BLOCK_LEN + 1, CHUNK_LEN + 1] {
        let mut reference = [0u8; 35 * BLOCK_LEN];
        let mut reference_hasher = reference_impl::Hasher::new();
        reference_hasher.update(&input[..input_len]);
        reference_hasher.finalize(&mut reference);
        let mut hasher = crate::Hasher::new();
        hasher.update(&input[..input_len]);
        for start in [0, 1, BLOCK_LEN - 1] {
            for blocks in 1..=33 {
                for tail in [0, 1, BLOCK_LEN - 1] {
                    let mut xof = hasher.finalize_xof();
                    let mut out = [0u8; 35 * BLOCK_LEN];
                    let end = start + blocks * BLOCK_LEN + tail;
                    xof.fill(&mut out[..start]);
                    xof.fill(&mut out[start..end]);
                    assert_eq!(out[..end], reference[..end], "{input_len} B input, from {start}, {blocks} blocks + {tail}");
                }
            }
        }
    }
}

#[test]
fn test_xof_seek() {
    let mut out = [0; 533];
    let mut hasher = crate::Hasher::new();
    hasher.update(b"foo");
    hasher.finalize_xof().fill(&mut out);
    assert_eq!(hasher.finalize().as_bytes(), &out[0..32]);

    let mut reader = hasher.finalize_xof();
    reader.set_position(303);
    let mut out2 = [0; 102];
    reader.fill(&mut out2);
    assert_eq!(&out[303..][..102], &out2[..]);

    #[cfg(feature = "std")]
    {
        use std::io::prelude::*;
        let mut reader = hasher.finalize_xof();
        reader.seek(std::io::SeekFrom::Start(303)).unwrap();
        let mut out3 = Vec::new();
        reader.by_ref().take(102).read_to_end(&mut out3).unwrap();
        assert_eq!(&out[303..][..102], &out3[..]);

        assert_eq!(
            reader.seek(std::io::SeekFrom::Current(0)).unwrap(),
            303 + 102
        );
        reader.seek(std::io::SeekFrom::Current(-5)).unwrap();
        assert_eq!(
            reader.seek(std::io::SeekFrom::Current(0)).unwrap(),
            303 + 102 - 5
        );
        let mut out4 = [0; 17];
        assert_eq!(reader.read(&mut out4).unwrap(), 17);
        assert_eq!(&out[303 + 102 - 5..][..17], &out4[..]);
        assert_eq!(
            reader.seek(std::io::SeekFrom::Current(0)).unwrap(),
            303 + 102 - 5 + 17
        );
        assert!(reader.seek(std::io::SeekFrom::End(0)).is_err());
        assert!(reader.seek(std::io::SeekFrom::Current(-1000)).is_err());
        // Back to the start exactly, and one byte before it; past the end
        // of u64 stops at its end.
        let at = reader.position() as i64;
        assert_eq!(reader.seek(std::io::SeekFrom::Current(-at)).unwrap(), 0);
        assert!(reader.seek(std::io::SeekFrom::Current(-1)).is_err());
        reader.seek(std::io::SeekFrom::Start(u64::MAX - 1)).unwrap();
        assert_eq!(reader.seek(std::io::SeekFrom::Current(2)).unwrap(), u64::MAX);
        assert_eq!(std::format!("{reader:?}"), "OutputReader { position: 18446744073709551615 }");
    }
}

#[test]
fn test_msg_schedule_permutation() {
    let permutation = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

    let mut generated = [[0; 16]; 7];
    generated[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    for round in 1..7 {
        for i in 0..16 {
            generated[round][i] = generated[round - 1][permutation[i]];
        }
    }

    assert_eq!(generated, crate::MSG_SCHEDULE);
}

#[test]
fn test_reset() {
    let mut hasher = crate::Hasher::new();
    hasher.update(&[42; 3 * CHUNK_LEN + 7]);
    hasher.reset();
    hasher.update(&[42; CHUNK_LEN + 3]);
    assert_eq!(hasher.finalize(), crate::hash(&[42; CHUNK_LEN + 3]));

    let key = &[99; crate::KEY_LEN];
    let mut keyed_hasher = crate::Hasher::new_keyed(key);
    keyed_hasher.update(&[42; 3 * CHUNK_LEN + 7]);
    keyed_hasher.reset();
    keyed_hasher.update(&[42; CHUNK_LEN + 3]);
    assert_eq!(
        keyed_hasher.finalize(),
        crate::keyed_hash(key, &[42; CHUNK_LEN + 3]),
    );

    let context = "BLAKE3 2020-02-12 10:20:58 reset test";
    let mut kdf = crate::Hasher::new_derive_key(context);
    kdf.update(&[42; 3 * CHUNK_LEN + 7]);
    kdf.reset();
    kdf.update(&[42; CHUNK_LEN + 3]);
    let expected = crate::derive_key(context, &[42; CHUNK_LEN + 3]);
    assert_eq!(kdf.finalize(), expected);
}

#[test]
fn test_hex_encoding_decoding() {
    let digest_str = "04e0bb39f30b1a3feb89f536c93be15055482df748674b00d26e5a75777702e9";
    let mut hasher = crate::Hasher::new();
    hasher.update(b"foo");
    let digest = hasher.finalize();
    assert_eq!(digest.to_hex().as_str(), digest_str);
    #[cfg(feature = "std")]
    assert_eq!(digest.to_string(), digest_str);

    // Test round trip
    let digest = crate::Hash::from_hex(digest_str).unwrap();
    assert_eq!(digest.to_hex().as_str(), digest_str);

    // Test uppercase
    let digest = crate::Hash::from_hex(digest_str.to_uppercase()).unwrap();
    assert_eq!(digest.to_hex().as_str(), digest_str);

    // Test string parsing via FromStr
    let digest: crate::Hash = digest_str.parse().unwrap();
    assert_eq!(digest.to_hex().as_str(), digest_str);

    // Test errors
    let bad_len = "04e0bb39f30b1";
    let _result = crate::Hash::from_hex(bad_len).unwrap_err();
    #[cfg(feature = "std")]
    assert_eq!(_result.to_string(), "expected 64 hex bytes, received 13");

    let bad_char = "Z4e0bb39f30b1a3feb89f536c93be15055482df748674b00d26e5a75777702e9";
    let _result = crate::Hash::from_hex(bad_char).unwrap_err();
    #[cfg(feature = "std")]
    assert_eq!(_result.to_string(), "invalid hex character: 'Z'");

    let _result = crate::Hash::from_hex([128; 64]).unwrap_err();
    #[cfg(feature = "std")]
    assert_eq!(_result.to_string(), "invalid hex character: 0x80");
}

#[test]
fn test_hash_conversions() {
    let bytes1 = [42; 32];
    let hash1: crate::Hash = bytes1.into();
    let bytes2: [u8; 32] = hash1.into();
    assert_eq!(bytes1, bytes2);

    let bytes3 = *hash1.as_bytes();
    assert_eq!(bytes1, bytes3);

    let hash2 = crate::Hash::from_bytes(bytes1);
    assert_eq!(hash1, hash2);

    let hex = hash1.to_hex();
    let hash3 = crate::Hash::from_hex(hex.as_bytes()).unwrap();
    assert_eq!(hash1, hash3);

    let slice1: &[u8] = bytes1.as_slice();
    let hash4 = crate::Hash::from_slice(slice1).expect("correct length");
    assert_eq!(hash1, hash4);

    let slice2 = hash1.as_slice();
    assert_eq!(slice1, slice2);

    assert!(crate::Hash::from_slice(&[]).is_err());
    assert!(crate::Hash::from_slice(&[42]).is_err());
    assert!(crate::Hash::from_slice([42; 31].as_slice()).is_err());
    assert!(crate::Hash::from_slice([42; 33].as_slice()).is_err());
    assert!(crate::Hash::from_slice([42; 100].as_slice()).is_err());
}

#[test]
const fn test_hash_const_conversions() {
    let bytes = [42; 32];
    let hash = crate::Hash::from_bytes(bytes);
    _ = hash.as_bytes();
}

#[test]
fn test_block_buffer_alignment() {
    // ChunkState.buf and Output.block are 64-byte aligned so that wide vector stores
    // into them (e.g. from memcpy) can store-to-load forward regardless of
    // stack layout. See the comment on Output in hasher.rs, the measurements in
    // https://github.com/zooko/bench-hashes/issues/2, and the before/after
    // benchmarks in https://github.com/BLAKE3-team/BLAKE3/pull/582.
    assert_eq!(64, core::mem::align_of::<crate::Output>());
    assert_eq!(0, core::mem::offset_of!(crate::Output, block));
    assert_eq!(64, core::mem::align_of::<crate::ChunkState>());
    assert_eq!(0, core::mem::offset_of!(crate::ChunkState, buf));
}

#[cfg(feature = "zeroize")]
#[test]
fn test_zeroize() {
    use zeroize::Zeroize;

    let mut hash = crate::Hash([42; 32]);
    hash.zeroize();
    assert_eq!(hash.0, [0u8; 32]);

    // A Hasher: its core, and the bytes it has staged.
    let mut hasher = crate::Hasher::new_keyed(&[42; 32]);
    hasher.update(&[42; 3000]);
    assert!(hasher.staged > 0);
    hasher.zeroize();
    assert_eq!(hasher.staged, 0);
    assert!(hasher.stage.iter().all(|b| unsafe { b.assume_init() } == 0));
    assert_eq!(hasher.core.key, [0; 8]);

    let mut hasher = crate::HasherCore {
        chunk_state: crate::hasher::ChunkState {
            cv: [42; 8],
            chunk_counter: 42,
            buf: [42; 64],
            buf_len: 42,
            blocks_compressed: 42,
            flags: 42,
            platform: crate::Platform::Portable,
        },
        initial_chunk_counter: 42,
        key: [42; 8],
        cv_stack: crate::hasher::CvStack::new(),
    };
    for _ in 0..crate::MAX_DEPTH + 1 {
        hasher.cv_stack.push([42; 32]);
    }
    hasher.zeroize();
    assert_eq!(hasher.chunk_state.cv, [0; 8]);
    assert_eq!(hasher.chunk_state.chunk_counter, 0);
    assert_eq!(hasher.chunk_state.buf, [0; 64]);
    assert_eq!(hasher.chunk_state.buf_len, 0);
    assert_eq!(hasher.chunk_state.blocks_compressed, 0);
    assert_eq!(hasher.chunk_state.flags, 0);
    assert!(matches!(
        hasher.chunk_state.platform,
        crate::Platform::Portable
    ));
    assert_eq!(hasher.initial_chunk_counter, 0);
    assert_eq!(hasher.key, [0; 8]);
    assert_eq!(hasher.cv_stack.len(), 0);
    assert_eq!(hasher.cv_stack.array(), Some(&[[0u8; 32]; crate::MAX_DEPTH + 1]));

    let mut output_reader = crate::OutputReader {
        inner: crate::Output {
            input_chaining_value: [42; 8],
            block: [42; 64],
            counter: 42,
            block_len: 42,
            flags: 42,
            platform: crate::Platform::Portable,
        },
        position_within_block: 42,
    };

    output_reader.zeroize();
    assert_eq!(output_reader.inner.input_chaining_value, [0; 8]);
    assert_eq!(output_reader.inner.block, [0; 64]);
    assert_eq!(output_reader.inner.counter, 0);
    assert_eq!(output_reader.inner.block_len, 0);
    assert_eq!(output_reader.inner.flags, 0);
    assert!(matches!(
        output_reader.inner.platform,
        crate::Platform::Portable
    ));
    assert_eq!(output_reader.position_within_block, 0);
}

#[test]
#[cfg(feature = "std")]
fn test_update_reader() -> Result<(), std::io::Error> {
    // This is a brief test, since update_reader() is mostly a wrapper around update(), which already
    // has substantial testing.
    let mut input = vec![0; 1_000_000];
    paint_test_input(&mut input);
    assert_eq!(
        crate::Hasher::new().update_reader(&input[..])?.finalize(),
        crate::hash(&input),
    );
    Ok(())
}

#[test]
fn test_long_xof_matches_reference() {
    // Extended output long enough for the platform's wide xof_many (SME2:
    // groups of sixteen blocks), from block-aligned and unaligned positions,
    // in every mode, against the reference implementation.
    let mut input = [0; 3000];
    paint_test_input(&mut input);
    for (mode, len) in [(0, 0usize), (0, 1), (0, 1024), (0, 3000), (1, 100), (2, 777)] {
        let (mut ours, mut reference) = match mode {
            0 => (crate::Hasher::new(), reference_impl::Hasher::new()),
            1 => (crate::Hasher::new_keyed(&TEST_KEY), reference_impl::Hasher::new_keyed(&TEST_KEY)),
            _ => (crate::Hasher::new_derive_key("long xof"), reference_impl::Hasher::new_derive_key("long xof")),
        };
        ours.update(&input[..len]);
        reference.update(&input[..len]);
        let mut want = vec![0u8; 5 * 1024 + 300];
        reference.finalize(&mut want);
        for (skip, out_len) in [(0usize, 1024usize), (0, 1088), (64, 2048), (37, 4096), (1000, 4300), (0, 5 * 1024 + 300)] {
            let out_len = out_len.min(want.len() - skip);
            let mut got = vec![0u8; out_len];
            let mut reader = ours.finalize_xof();
            reader.set_position(skip as u64);
            reader.fill(&mut got);
            assert_eq!(got, want[skip..][..out_len], "mode {mode}, {len} input bytes, {out_len} bytes from {skip}");
        }
    }
}

#[test]
#[cfg(feature = "std")]
fn test_update_reader_keeps_every_byte_across_errors() {
    // A reader that fails (WouldBlock) after given offsets, in short reads,
    // and a caller that retries until the end: every byte read before each
    // error is hashed, inside the first 64 KiB and inside the 1 MiB buffer.
    use std::io;
    struct Failing<'a> {
        slice: &'a [u8],
        at: usize,
        fail_at: Vec<usize>,
    }
    impl io::Read for Failing<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.fail_at.first() == Some(&self.at) {
                self.fail_at.remove(0);
                return Err(io::Error::from(io::ErrorKind::WouldBlock));
            }
            let next = self.fail_at.first().copied().unwrap_or(usize::MAX);
            let take = buf.len().min(self.slice.len()).min(next - self.at).min(50_000);
            buf[..take].copy_from_slice(&self.slice[..take]);
            self.slice = &self.slice[take..];
            self.at += take;
            Ok(take)
        }
    }
    let mut input = vec![0; (3 << 20) + 12345];
    paint_test_input(&mut input);
    for fail_at in [vec![], vec![1000], vec![65536], vec![70_000, 1 << 20], vec![10, 2_000_000, 3_000_000]] {
        let mut reader = Failing { slice: &input, at: 0, fail_at: fail_at.clone() };
        let mut hasher = crate::Hasher::new();
        let mut errors = 0;
        while let Err(e) = hasher.update_reader(&mut reader) {
            assert_eq!(e.kind(), io::ErrorKind::WouldBlock);
            errors += 1;
        }
        assert_eq!(errors, fail_at.len());
        assert_eq!(hasher.finalize(), crate::hash(&input), "failing at {fail_at:?}");
    }
}

#[test]
#[cfg(feature = "std")]
fn test_update_reader_interrupted() -> std::io::Result<()> {
    use std::io;
    struct InterruptingReader<'a> {
        already_interrupted: bool,
        slice: &'a [u8],
    }
    impl<'a> InterruptingReader<'a> {
        fn new(slice: &'a [u8]) -> Self {
            Self {
                already_interrupted: false,
                slice,
            }
        }
    }
    impl<'a> io::Read for InterruptingReader<'a> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if !self.already_interrupted {
                self.already_interrupted = true;
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let take = std::cmp::min(self.slice.len(), buf.len());
            buf[..take].copy_from_slice(&self.slice[..take]);
            self.slice = &self.slice[take..];
            Ok(take)
        }
    }

    let input = b"hello world";
    let mut reader = InterruptingReader::new(input);
    let mut hasher = crate::Hasher::new();
    hasher.update_reader(&mut reader)?;
    assert_eq!(hasher.finalize(), crate::hash(input));
    Ok(())
}

#[test]
#[cfg(feature = "mmap")]
// NamedTempFile isn't Miri-compatible
#[cfg(not(miri))]
fn test_mmap() -> Result<(), std::io::Error> {
    // This is a brief test, since update_mmap() is mostly a wrapper around update(), which already
    // has substantial testing.
    use std::io::prelude::*;
    let mut input = vec![0; 1_000_000];
    paint_test_input(&mut input);
    let mut tempfile = tempfile::NamedTempFile::new()?;
    tempfile.write_all(&input)?;
    tempfile.flush()?;
    assert_eq!(
        crate::Hasher::new()
            .update_mmap(tempfile.path())?
            .finalize(),
        crate::hash(&input),
    );
    Ok(())
}

#[test]
#[cfg(feature = "mmap")]
#[cfg(target_os = "linux")]
fn test_mmap_virtual_file() -> Result<(), std::io::Error> {
    // Virtual files like /proc/version can't be mmapped, because their contents don't actually
    // exist anywhere in memory. Make sure we fall back to regular file IO in these cases.
    let virtual_filepath = "/proc/version";
    let mut mmap_hasher = crate::Hasher::new();
    // We'll fail right here if the fallback doesn't work.
    mmap_hasher.update_mmap(virtual_filepath)?;
    let mut read_hasher = crate::Hasher::new();
    read_hasher.update_reader(std::fs::File::open(virtual_filepath)?)?;
    assert_eq!(mmap_hasher.finalize(), read_hasher.finalize());
    Ok(())
}

#[test]
#[cfg(feature = "std")]
#[cfg(feature = "serde")]
fn test_serde() {
    // Henrik suggested that we use 0xfe / 254 for byte test data instead of 0xff / 255, due to the
    // fact that 0xfe is not a well formed CBOR item.
    let hash: crate::Hash = [0xfe; 32].into();

    let json = serde_json::to_string(&hash).unwrap();
    assert_eq!(
        json,
        "[254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254]",
    );
    let hash2: crate::Hash = serde_json::from_str(&json).unwrap();
    assert_eq!(hash, hash2);

    let mut cbor = Vec::<u8>::new();
    ciborium::into_writer(&hash, &mut cbor).unwrap();
    assert_eq!(
        cbor,
        [
            0x98, 0x20, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe,
            0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe,
            0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe,
            0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe,
            0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe, 0x18, 0xfe,
        ]
    );
    let hash_from_cbor: crate::Hash = ciborium::from_reader(&cbor[..]).unwrap();
    assert_eq!(hash_from_cbor, hash);

    // Version 1.5.2 of this crate changed the default serialization format to a bytestring
    // (instead of an array/list) to save bytes on the wire. That was a backwards compatibility
    // mistake for non-self-describing formats, and it's been reverted. Since some small number of
    // serialized bytestrings will probably exist forever in the wild, we should test that we can
    // still deserialize these from self-describing formats.
    let bytestring_cbor: &[u8] = &[
        0x58, 0x20, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe,
        0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe,
        0xfe, 0xfe, 0xfe, 0xfe,
    ];
    let hash_from_bytestring_cbor: crate::Hash = ciborium::from_reader(bytestring_cbor).unwrap();
    assert_eq!(hash_from_bytestring_cbor, hash);
}

// `cargo +nightly miri test` currently works, but it takes forever, because some of our test
// inputs are quite large. Most of our unsafe code is platform specific and incompatible with Miri
// anyway, but we'd like it to be possible for callers to run their own tests under Miri, assuming
// they don't use incompatible features like Rayon or mmap. This test should get reasonable
// coverage of our public API without using any large inputs, so we can run it in CI and catch
// obvious breaks. (For example, constant_time_eq is not compatible with Miri.)
#[test]
fn test_miri_smoketest() {
    let mut hasher = crate::Hasher::new_derive_key("Miri smoketest");
    hasher.update(b"foo");
    #[cfg(feature = "std")]
    hasher.update_reader(&b"bar"[..]).unwrap();
    assert_eq!(hasher.finalize(), hasher.finalize());
    let mut reader = hasher.finalize_xof();
    reader.set_position(999999);
    reader.fill(&mut [0]);
}

/// The kernel reports' contract: kernels in ascending `from_len`, the
/// first at 0, each named and explained (where each kernel runs is
/// test_kernel_reports_match_the_dispatch's).
#[test]
#[cfg(feature = "std")]
fn test_kernel_reports() {
    let check = |report: crate::KernelReport| {
        assert!(!report.platform.is_empty());
        assert_eq!(report.kernels[0].from_len, 0, "{report:?}");
        assert!(report.kernels.windows(2).all(|pair| pair[0].from_len < pair[1].from_len), "{report:?}");
        assert!(report.kernels.iter().all(|k| !k.name.is_empty() && !k.why.is_empty()), "{report:?}");
    };
    check(crate::kernel_report());
    check(crate::kernel_report_multithreaded());
    for message_len in [1, 63, 64, 65, 128, 256, 1000, 1024, 1025, 40_000] {
        check(crate::kernel_report_many(message_len));
        check(crate::kernel_report_many_multithreaded(message_len));
    }
}

/// The lengths, 1 to `max`, that reach every case of a message's block
/// structure: for each count of blocks, its last block one byte and
/// whole. Lengths within one block count run the same code, differing
/// only in how many bytes the last block's padding covers.
pub(crate) fn block_cases(max: usize) -> impl Iterator<Item = usize> {
    (1..=max.div_ceil(BLOCK_LEN)).flat_map(|b| [(b - 1) * BLOCK_LEN + 1, b * BLOCK_LEN]).filter(move |&len| len <= max)
}

/// One-shot hashes of 1 to 17 chunks against the reference
/// implementation, on this machine's platform: for each count of whole
/// chunks (each picks its own whole-plus-partial kernel and plan), a
/// partial chunk of each block count (each a different kernel shape), its
/// last block one byte and whole, and the count's whole chunks alone.
#[test]
fn test_one_to_seventeen_chunks_against_reference() {
    let mut input = vec![0u8; 17 * CHUNK_LEN];
    paint_test_input(&mut input);
    let mut lens = vec![17 * CHUNK_LEN];
    for whole in 1..=16 {
        lens.push(whole * CHUNK_LEN);
        lens.extend(block_cases(CHUNK_LEN).map(|partial| whole * CHUNK_LEN + partial));
    }
    lens.sort_unstable();
    lens.dedup();
    for len in lens {
        let mut reference = reference_impl::Hasher::new();
        reference.update(&input[..len]);
        let mut want = [0u8; OUT_LEN];
        reference.finalize(&mut want);
        assert_eq!(crate::hash(&input[..len]).as_bytes(), &want, "len = {len}");
    }
}

// The fork's unsafe paths at sizes Miri can run in minutes (under the
// `pure` feature, where no assembly takes part): the batch tables and
// padded lanes of many.rs, the pool's jobs from concurrent callers (Miri
// with -Zmiri-num-cpus=4 spawns three workers), and a stream past one
// buffer. Each against hash() or the reference implementation. Also run
// natively, where they are quick.
#[cfg(feature = "std")]
mod unsafe_paths {
    use crate::*;

    fn input(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u32).wrapping_mul(0x9E37_79B1).to_le_bytes()[3] ^ seed).collect()
    }

    #[test]
    fn test_batches_small() {
        for len in [0, 1, 64, 100, 128, 192, 1024, 1088, 2048, 3000] {
            for count in [0, 1, 2, 3, 4, 5, 9, 17] {
                let slot = crate::many::slot_len(len);
                let mut data = input(slot * count, len as u8);
                for message in data.chunks_exact_mut(slot) {
                    message[len..].fill(0);
                }
                let mut out = vec![[0u8; OUT_LEN]; count];
                hash_many(&data, len, &mut out);
                for (i, digest) in out.iter().enumerate() {
                    assert_eq!(*digest, *hash(&data[i * slot..][..len]).as_bytes(), "{count} x {len} B, message {i}");
                }
            }
        }
    }

    /// A batch and an input, from two threads at once through the
    /// multithreaded full forms.
    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "spawns threads, which this target lacks")]
    fn test_pool_from_two_callers() {
        let batch = input(1024 * 64, 1);
        let tree = input(65 * 1024 + 3, 2);
        let mut expected = vec![[0u8; OUT_LEN]; 1024];
        hash_many(&batch, 64, &mut expected);
        let expected_tree = hash(&tree);
        std::thread::scope(|scope| {
            for caller in 0..2 {
                let (batch, tree, expected) = (&batch, &tree, &expected);
                scope.spawn(move || {
                    let mut out = vec![[0u8; OUT_LEN]; 1024];
                    hash_many_multithreaded_with(Mode::Hash, batch, 64, &mut out);
                    assert_eq!(&out, expected, "batch, caller {caller}");
                    assert_eq!(hash_multithreaded_with(Mode::Hash, tree), expected_tree, "tree, caller {caller}");
                });
            }
        });
    }

}

// Every platform's hash_many against the portable one, for every input
// shape the public Platform API accepts on this CPU: blocks of 64 B to a
// whole chunk, with and without counter increments, keyed flags, counters
// whose low word overflows inside a batch, and pointer tables in memory
// order, reversed, and strided (the SME2 parent kernel's gather path).
#[cfg(feature = "std")]
mod platform_hash_many {
    use crate::platform::Platform;
    use crate::{IncrementCounter, OUT_LEN};

    fn platforms() -> Vec<Platform> {
        #[allow(unused_mut)]
        let mut all = vec![Platform::detect()];
        #[cfg(blake3_neon)]
        all.push(Platform::neon().unwrap());
        #[cfg(blake3_sme2)]
        all.extend(Platform::sme2());
        all
    }

    fn check<const N: usize>() {
        let mut buf = vec![0u8; N * 2 * 140];
        crate::test::paint_test_input(&mut buf);
        let (flags, start, end) = (crate::KEYED_HASH, crate::CHUNK_START, crate::CHUNK_END);
        for count in (0..=40).chain([127, 128, 129, 130]) {
            let forward: Vec<&[u8; N]> = (0..count).map(|i| buf[i * N..][..N].try_into().unwrap()).collect();
            let reversed: Vec<&[u8; N]> = forward.iter().rev().copied().collect();
            let strided: Vec<&[u8; N]> = (0..count).map(|i| buf[2 * i * N..][..N].try_into().unwrap()).collect();
            for table in [&forward, &reversed, &strided] {
                for increment in [IncrementCounter::Yes, IncrementCounter::No] {
                    for counter in [0, u32::MAX as u64 - 5] {
                        let mut want = vec![0u8; count * OUT_LEN];
                        crate::portable::hash_many(table, &crate::test::TEST_KEY_WORDS, counter, increment, flags, start, end, &mut want);
                        for platform in platforms() {
                            let mut got = vec![0xAAu8; count * OUT_LEN + OUT_LEN];
                            platform.hash_many(table, &crate::test::TEST_KEY_WORDS, counter, increment, flags, start, end, &mut got[..count * OUT_LEN]);
                            assert_eq!(&got[..count * OUT_LEN], &want[..], "{platform:?}, N = {N}, {count} inputs, increment {}, counter {counter}", increment.yes());
                            assert_eq!(got[count * OUT_LEN..], [0xAA; OUT_LEN], "{platform:?} wrote past its outputs");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_every_platform_every_shape() {
        check::<64>();
        check::<128>();
        check::<192>();
        check::<256>();
        check::<1024>();
    }
}

// Every kernel's reads and writes against guard pages: inputs placed to end
// exactly where an inaccessible page begins (and, again, to start right
// after one), outputs likewise, so a kernel that reads or writes one byte
// outside its buffers stops the test with a fault. The assembly kernels'
// accesses are invisible to the sanitizers; these are not.
#[cfg(all(feature = "std", target_arch = "aarch64", any(target_vendor = "apple", target_os = "linux")))]
mod guard_pages {
    use crate::*;

    /// `len` bytes with an inaccessible page before and after them, the
    /// bytes flush against the page after (`at_end`) or before.
    struct Guarded {
        base: *mut u8,
        total: usize,
        start: usize,
        len: usize,
    }

    impl Guarded {
        fn new(len: usize, at_end: bool) -> Self {
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let body = len.div_ceil(page).max(1) * page;
            let total = body + 2 * page;
            let base = unsafe { libc::mmap(core::ptr::null_mut(), total, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
            assert_ne!(base, libc::MAP_FAILED, "mmap");
            let base = base as *mut u8;
            unsafe {
                assert_eq!(libc::mprotect(base as *mut _, page, libc::PROT_NONE), 0);
                assert_eq!(libc::mprotect(base.add(page + body) as *mut _, page, libc::PROT_NONE), 0);
            }
            let start = if at_end { page + body - len } else { page };
            Guarded { base, total, start, len }
        }

        fn bytes(&mut self) -> &mut [u8] {
            unsafe { core::slice::from_raw_parts_mut(self.base.add(self.start), self.len) }
        }
    }

    impl Drop for Guarded {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.base as *mut _, self.total) };
        }
    }

    fn filled(len: usize, at_end: bool) -> Guarded {
        let mut g = Guarded::new(len, at_end);
        crate::test::paint_test_input(g.bytes());
        g
    }

    #[test]
    fn test_hash_inside_guard_pages() {
        let mut lens: Vec<usize> = (0..=2 * CHUNK_LEN + 70).collect();
        for chunks in [3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 128, 256, 1024, 1025] {
            for delta in [-65i64, -64, -1, 0, 1, 63, 64, 65] {
                lens.push(((chunks * CHUNK_LEN) as i64 + delta) as usize);
            }
        }
        for len in lens {
            for at_end in [true, false] {
                let mut g = filled(len, at_end);
                let data = g.bytes().to_vec();
                assert_eq!(hash(g.bytes()), crate::test::reference_hash(&data), "{len} bytes, at end {at_end}");
                if len >= 64 * CHUNK_LEN && len % 7 == 0 {
                    assert_eq!(hash_multithreaded(g.bytes()), hash(&data), "mt, {len} bytes");
                }
            }
        }
    }

    #[test]
    fn test_hash_many_inside_guard_pages() {
        for len in [1, 63, 64, 100, 128, 192, 200, 256, 640, 1000, 1024, 1088, 2000, 2048, 3072, 4096, 4100, 8192, 15360, 15359, 16384] {
            for count in [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 15, 16, 17, 21, 24, 31, 33, 127, 128, 129] {
                for at_end in [true, false] {
                    let slot = crate::many::slot_len(len);
                    let mut input = filled(slot * count, at_end);
                    for message in input.bytes().chunks_exact_mut(slot) {
                        message[len..].fill(0);
                    }
                    let mut out = Guarded::new(count * OUT_LEN, !at_end);
                    let outputs: &mut [[u8; OUT_LEN]] = unsafe { core::slice::from_raw_parts_mut(out.bytes().as_mut_ptr() as *mut [u8; OUT_LEN], count) };
                    hash_many(input.bytes(), len, outputs);
                    let data = input.bytes().to_vec();
                    for (i, digest) in outputs.iter().enumerate() {
                        assert_eq!(*digest, *hash(&data[i * slot..][..len]).as_bytes(), "{count} x {len} B, message {i}, at end {at_end}");
                    }
                    if slot * count >= 64 * CHUNK_LEN {
                        let mut mt = vec![[0u8; OUT_LEN]; count];
                        hash_many_multithreaded(input.bytes(), len, &mut mt);
                        assert_eq!(&mt[..], &outputs[..], "mt, {count} x {len} B");
                    }
                }
            }
        }
    }

    #[test]
    fn test_xof_inside_guard_pages() {
        // Extended output written flush against a guard page after the
        // buffer, then before it: the wide xof_many (SME2: groups of
        // sixteen blocks) writes no byte outside it, from aligned and
        // unaligned positions.
        let mut reference = vec![0u8; 5 * 1024 + 200];
        crate::Hasher::new().update(b"guarded").finalize_xof().fill(&mut reference);
        for len in [1, 64, 1023, 1024, 1025, 1088, 2048, 3000, 4096, 5 * 1024] {
            for skip in [0usize, 64, 37] {
                for at_end in [true, false] {
                    let mut out = Guarded::new(len, at_end);
                    let mut reader = crate::Hasher::new().update(b"guarded").finalize_xof();
                    reader.set_position(skip as u64);
                    reader.fill(out.bytes());
                    assert_eq!(out.bytes(), &reference[skip..][..len], "{len} B from {skip}, at end {at_end}");
                }
            }
        }
    }
}

/// The Hasher's stage: pieces of many lengths across 16 KiB boundaries,
/// update and update_multithreaded mixed, finalize midway and after more
/// input, and subtrees at input offsets (hazmat), against one-shot hashes.
#[test]
#[cfg(feature = "std")]
fn test_hasher_gathers() {
    use crate::hazmat::HasherExt;
    let mut input = vec![0u8; 300_000];
    paint_test_input(&mut input);
    let pieces = [1, 1448, 4096, 1448, 16384, 31, 16383, 65536, 16385, 7, 1000];
    for start in 0..pieces.len() {
        let mut hasher = crate::Hasher::new();
        let mut done = 0;
        let mut k = start;
        while done < input.len() {
            let n = pieces[k % pieces.len()].min(input.len() - done);
            if k % 3 == 0 {
                hasher.update_multithreaded(&input[done..done + n]);
            } else {
                hasher.update(&input[done..done + n]);
            }
            done += n;
            k += 1;
            if k % 5 == 0 {
                assert_eq!(hasher.finalize(), crate::hash(&input[..done]), "after {done} bytes");
                assert_eq!(hasher.count(), done as u64);
            }
        }
        assert_eq!(hasher.finalize(), crate::hash(&input), "pieces from {start}");
    }
    // A subtree of 64 chunks at offset 64 KiB, in small pieces, against one update.
    let sub = &input[..64 * CHUNK_LEN];
    let mut whole = crate::Hasher::new();
    whole.set_input_offset(64 * CHUNK_LEN as u64).update(sub);
    let mut pieced = crate::Hasher::new();
    pieced.set_input_offset(64 * CHUNK_LEN as u64);
    for piece in sub.chunks(1448) {
        pieced.update(piece);
    }
    assert_eq!(pieced.finalize_non_root(), whole.finalize_non_root());
}

/// Whole chunks and a partial one in one pass (rust_avx2's
/// hash_chunks_with_partial, on AVX2 or AVX-512): 1 to 7 whole chunks
/// (each a different count of lanes beside the partial chunk's) with a
/// partial chunk of each block count (block_cases), and some past them
/// (the partial chunk then sits at a later counter), in each of the three
/// modes, against the reference implementation.
#[test]
#[cfg(feature = "std")]
fn test_chunks_with_partial_against_reference() {
    let mut input = vec![0u8; 40 * CHUNK_LEN];
    paint_test_input(&mut input);
    let key = [7u8; crate::KEY_LEN];
    let mut lens: Vec<usize> = (1..8).flat_map(|whole| block_cases(CHUNK_LEN - 1).map(move |partial| whole * CHUNK_LEN + partial)).collect();
    lens.extend([9 * CHUNK_LEN + 1, 12 * CHUNK_LEN + 700, 16 * CHUNK_LEN + 64, 23 * CHUNK_LEN + 1023, 39 * CHUNK_LEN + 5]);
    if cfg!(miri) {
        lens = vec![CHUNK_LEN + 1, 2 * CHUNK_LEN + 65, 7 * CHUNK_LEN + 1023];
    }
    for len in lens {
        let m = &input[..len];
        let expect = |mut r: reference_impl::Hasher| {
            r.update(m);
            let mut out = [0u8; OUT_LEN];
            r.finalize(&mut out);
            out
        };
        assert_eq!(crate::hash(m).as_bytes(), &expect(reference_impl::Hasher::new()), "hash, len {len}");
        assert_eq!(crate::keyed_hash(&key, m).as_bytes(), &expect(reference_impl::Hasher::new_keyed(&key)), "keyed_hash, len {len}");
        assert_eq!(crate::derive_key("partial test", m), expect(reference_impl::Hasher::new_derive_key("partial test")), "derive_key, len {len}");
    }
}

/// Hash's comparisons (constant time) and formats: equal and unequal
/// against a Hash, a 32-byte array, and a slice, and a slice of another
/// length; Display as the hex digest, Debug as Hash("hex").
#[test]
fn test_hash_comparisons_and_formats() {
    let a = crate::hash(b"a");
    let b = crate::hash(b"b");
    assert!(a == a && a != b);
    assert!(a == *a.as_bytes() && a != *b.as_bytes());
    assert!(a == a.as_bytes()[..] && a != b.as_bytes()[..] && a != a.as_bytes()[..31]);
    let hex = a.to_hex();
    assert_eq!(hex.as_str(), "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f");
    #[cfg(feature = "std")]
    {
        assert_eq!(std::format!("{a}"), hex.as_str());
        assert_eq!(std::format!("{a:?}"), std::format!("Hash({:?})", hex.as_str()));
    }
}

/// A Hasher's Debug form names its flags and platform, and leaves out
/// its bytes and chaining values (they may be secret).
#[test]
#[cfg(feature = "std")]
fn test_debug_leaves_out_the_bytes() {
    let mut hasher = crate::Hasher::new_keyed(&[0xab; 32]);
    hasher.update(&[0xab; 70]);
    let shown = std::format!("{hasher:?}");
    assert!(shown.starts_with("Hasher {") && shown.contains("flags: 16") && shown.contains("platform"), "{shown}");
    assert!(!shown.contains("171"), "{shown}");
}

/// A Hasher as an io::Write: write takes the whole input, and the digest
/// is update's.
#[test]
#[cfg(feature = "std")]
fn test_hasher_io_write() {
    use std::io::Write;
    let mut input = [0u8; 3000];
    paint_test_input(&mut input);
    let mut hasher = crate::Hasher::new();
    assert_eq!(hasher.write(&input[..1000]).unwrap(), 1000);
    hasher.write_all(&input[1000..]).unwrap();
    hasher.flush().unwrap();
    assert_eq!(hasher.finalize(), crate::hash(&input));
}

/// Each kernel report against the dispatch: at each kernel's first length
/// and the length below the next kernel's, the calls run the kernel the
/// report names, and below the next one, not that one. For `hash`,
/// `hash_multithreaded`, and the batch reports at each kind of message
/// length the batch dispatch distinguishes (a block, blocks to a chunk,
/// two chunks, more).
#[test]
#[cfg(all(feature = "std", blake3_neon_hybrid))]
fn test_kernel_reports_match_the_dispatch() {
    use crate::{Ran, RAN};
    let kind = |name: &str| -> u32 {
        (match name {
            "integer code, one call" | "integer code, one message per call" => Ran::Scalar,
            "integer and NEON code side by side"
            | "integer and NEON code, messages side by side"
            | "integer and NEON code, up to nine messages at a time" => Ran::Hybrid,
            "NEON vectors, four chunks at a time" | "NEON vectors, four messages at a time" => Ran::Neon,
            "SME2, sixteen chunks at a time" | "SME2, sixteen messages side by side" | "SME2, sixteen messages at a time" => Ran::Sme2,
            "SME2 with integer code beside it" => Ran::Sme2x2,
            "split over threads" | "split over threads, SME2 on one for large inputs" => Ran::Pool,
            other => panic!("a kernel this test knows: {other}"),
        }) as u32
    };
    // The kinds one call ran on this thread; again when another test's
    // call held the SME2 turn (this call then ran NEON in its place), or
    // when the call followed a pause and took the pair kernel (the reports
    // describe calls back to back), or when other tests' calls filled the
    // pool. The call again comes right after.
    let ran = |call: &dyn Fn()| loop {
        RAN.with(|r| r.set(0));
        call();
        let kinds = RAN.with(|r| r.get());
        if kinds & (Ran::TurnTaken as u32 | Ran::AfterPause as u32 | Ran::PoolBusy as u32) == 0 {
            return kinds;
        }
    };
    // Kernel i of `report` runs at units from_len / unit up to the next's,
    // and every kind a call runs is one the report names at or below it
    // (the last kernel's checked at twice its start, too).
    // `also`: the kinds a call may run beside the report's (for a batch,
    // what hash() of one message runs).
    let check = |report: crate::KernelReport, unit: usize, also: u32, far: usize, at: &dyn Fn(usize) -> u32| {
        // Any call above one chunk compresses its root on the integer
        // kernel, so every kernel brings the integer kind with it.
        let named = |n: usize| report.kernels.iter().filter(|k| k.from_len <= n * unit).fold(also, |m, k| m | kind(k.name) | Ran::Scalar as u32);
        let ran = |n: usize| {
            let kinds = at(n);
            assert!(kinds & !named(n) == 0, "at {n} x {unit}, kinds {kinds:#b} beyond the report's {:#b}: {report:?}", named(n));
            kinds
        };
        for (i, k) in report.kernels.iter().enumerate() {
            let first = (k.from_len / unit).max(1);
            assert!(ran(first) & kind(k.name) != 0, "{} at {first} x {unit}: {report:?}", k.name);
            match report.kernels.get(i + 1) {
                Some(next) => {
                    let below = next.from_len / unit - 1;
                    let kinds = ran(below);
                    assert!(kinds & kind(k.name) != 0, "{} at {below} x {unit}: {report:?}", k.name);
                    if kind(next.name) != kind(k.name) {
                        assert!(kinds & kind(next.name) == 0, "{} below its start, {below} x {unit}: {report:?}", next.name);
                    }
                }
                None => assert!(ran(2 * first) & kind(k.name) != 0, "{} at {} x {unit}: {report:?}", k.name, 2 * first),
            }
        }
        // And far past the last start, where every kernel any length or
        // count reaches runs: nothing beyond the report there either.
        ran(far);
    };
    let mut input = vec![0u8; 2 << 20];
    paint_test_input(&mut input);
    // What hash() of `len` bytes may run, by its report.
    let one = |len: usize| crate::kernel_report().kernels.iter().filter(|k| k.from_len <= len).fold(Ran::Scalar as u32, |m, k| m | kind(k.name));
    check(crate::kernel_report(), 1, 0, input.len(), &|len| ran(&|| { crate::hash(&input[..len]); }));
    check(crate::kernel_report_multithreaded(), 1, 0, input.len(), &|len| ran(&|| { crate::hash_multithreaded(&input[..len]); }));
    for message_len in [64, 1000, 1024, 2000, 3000, 15 * 1024, 300_000] {
        let slot = crate::many::slot_len(message_len);
        let batch = |n: usize| -> std::vec::Vec<u8> {
            let mut b = std::vec![0u8; n * slot];
            for m in 0..n {
                b[m * slot..][..message_len].copy_from_slice(&input[m % 7..][..message_len]);
            }
            b
        };
        // 64 messages pass every batch threshold below the split.
        let far = 64.min(input.len() / message_len.max(1));
        check(crate::kernel_report_many(message_len), message_len, one(message_len), far, &|n| {
            let b = batch(n);
            let out = std::vec![[0u8; 32]; n];
            ran(&|| crate::hash_many(&b, message_len, &mut out.clone()))
        });
        check(crate::kernel_report_many_multithreaded(message_len), message_len, one(message_len), far, &|n| {
            let b = batch(n);
            let out = std::vec![[0u8; 32]; n];
            ran(&|| crate::hash_many_multithreaded(&b, message_len, &mut out.clone()))
        });
    }
}

/// A context hashed once (Mode::DeriveKeyMaterial) derives what the
/// context string derives (Mode::DeriveKey), one-shot and in a batch.
#[test]
fn test_derive_key_material_mode() {
    let context = "BLAKE3 2026-10-16 derive key material test";
    let context_key = crate::hazmat::hash_derive_key_context(context);
    let (by_string, by_key) = (crate::Mode::DeriveKey(context), crate::Mode::DeriveKeyMaterial(&context_key));
    let mut input = [0u8; 3000];
    paint_test_input(&mut input);
    assert_eq!(crate::hash_with(by_key, &input), crate::hash_with(by_string, &input));
    assert_eq!(crate::hash_with(by_key, &input), crate::derive_key(context, &input));
    let (mut a, mut b) = ([[0u8; 32]; 4], [[0u8; 32]; 4]);
    crate::hash_many_with(by_key, &input[..4 * 128], 128, &mut a);
    crate::hash_many_with(by_string, &input[..4 * 128], 128, &mut b);
    assert_eq!(a, b);
}
