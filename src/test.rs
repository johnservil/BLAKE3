use crate::{BLOCK_LEN, CHUNK_LEN, CVBytes, CVWords, IncrementCounter, OUT_LEN};
use arrayvec::ArrayVec;
use rand::prelude::*;

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
    let test_state_bytes = crate::platform::le_bytes_from_words_32(&test_state);
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
fn test_largest_power_of_two_leq() {
    let input_output = &[
        // The zero case is nonsensical, but it does work.
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 8),
        // the largest possible usize
        (usize::MAX, (usize::MAX >> 1) + 1),
    ];
    for &(input, output) in input_output {
        assert_eq!(
            output,
            crate::largest_power_of_two_leq(input),
            "wrong output for n={}",
            input
        );
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
            // incremental (rayon)
            #[cfg(feature = "rayon")]
            {
                let mut hasher = crate::Hasher::new();
                hasher.update_rayon(input);
                assert_eq!(
                    hasher.finalize(),
                    *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
                );
                assert_eq!(hasher.finalize(), test_out);
            }
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
            // incremental (rayon)
            #[cfg(feature = "rayon")]
            {
                let mut hasher = crate::Hasher::new_keyed(&TEST_KEY);
                hasher.update_rayon(input);
                assert_eq!(
                    hasher.finalize(),
                    *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
                );
                assert_eq!(hasher.finalize(), test_out);
            }
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
            // incremental (rayon)
            #[cfg(feature = "rayon")]
            {
                let mut hasher = crate::Hasher::new_derive_key(context);
                hasher.update_rayon(input);
                assert_eq!(
                    hasher.finalize(),
                    *<&[u8; 32]>::try_from(&expected_out[..32]).unwrap()
                );
                assert_eq!(
                    hasher.finalize(),
                    *<&[u8; 32]>::try_from(&test_out[..32]).unwrap()
                );
            }
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

#[test]
fn test_fuzz_hasher() {
    const INPUT_MAX: usize = 4 * CHUNK_LEN;
    let mut input_buf = [0; 3 * INPUT_MAX];
    paint_test_input(&mut input_buf);

    // Don't do too many iterations in debug mode, to keep the tests under a
    // second or so. CI should run tests in release mode also. Provide an
    // environment variable for specifying a larger number of fuzz iterations.
    let num_tests = if cfg!(debug_assertions) { 100 } else { 10_000 };

    // Use a fixed RNG seed for reproducibility.
    let mut rng = chacha20::ChaCha8Rng::from_seed([1; 32]);
    for _num_test in 0..num_tests {
        #[cfg(feature = "std")]
        dbg!(_num_test);
        let mut hasher = crate::Hasher::new();
        let mut total_input = 0;
        // For each test, write 3 inputs of random length.
        for _ in 0..3 {
            let input_len = rng.random_range(0..(INPUT_MAX + 1));
            #[cfg(feature = "std")]
            dbg!(input_len);
            let input = &input_buf[total_input..][..input_len];
            hasher.update(input);
            total_input += input_len;
        }
        let expected = reference_hash(&input_buf[..total_input]);
        assert_eq!(expected, hasher.finalize());
    }
}

#[test]
fn test_fuzz_xof() {
    let mut input_buf = [0u8; 3 * BLOCK_LEN];
    paint_test_input(&mut input_buf);

    // Don't do too many iterations in debug mode, to keep the tests under a
    // second or so. CI should run tests in release mode also. Provide an
    // environment variable for specifying a larger number of fuzz iterations.
    let num_tests = if cfg!(debug_assertions) { 100 } else { 2500 };

    // Use a fixed RNG seed for reproducibility.
    let mut rng = chacha20::ChaCha8Rng::from_seed([1; 32]);
    for _num_test in 0..num_tests {
        #[cfg(feature = "std")]
        dbg!(_num_test);
        // 31 (16 + 8 + 4 + 2 + 1) outputs
        let mut output_buf = [0; 31 * CHUNK_LEN];
        let input_len = rng.random_range(0..input_buf.len());
        let mut xof = crate::Hasher::new()
            .update(&input_buf[..input_len])
            .finalize_xof();
        let partial_start = rng.random_range(0..output_buf.len());
        let partial_end = rng.random_range(partial_start..output_buf.len());
        xof.fill(&mut output_buf[..partial_start]);
        xof.fill(&mut output_buf[partial_start..partial_end]);
        xof.fill(&mut output_buf[partial_end..]);

        let mut reference_buf = [0; 31 * CHUNK_LEN];
        let mut reference_hasher = reference_impl::Hasher::new();
        reference_hasher.update(&input_buf[..input_len]);
        reference_hasher.finalize(&mut reference_buf);

        assert_eq!(reference_buf, output_buf);
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

// This test is a mimized failure case for the Windows SSE2 bug described in
// https://github.com/BLAKE3-team/BLAKE3/issues/206.
//
// Before that issue was fixed, this test would fail on Windows in the following configuration:
//
//     cargo test --features=no_avx512,no_avx2,no_sse41 --release
//
// Bugs like this one (stomping on a caller's register) are very sensitive to the details of
// surrounding code, so it's not especially likely that this test will catch another bug (or even
// the same bug) in the future. Still, there's no harm in keeping it.
#[test]
fn test_issue_206_windows_sse2() {
    // This stupid loop has to be here to trigger the bug. I don't know why.
    for _ in &[0] {
        // The length 65 (two blocks) is significant. It doesn't repro with 64 (one block). It also
        // doesn't repro with an all-zero input.
        let input = &[0xff; 65];
        let expected_hash = [
            183, 235, 50, 217, 156, 24, 190, 219, 2, 216, 176, 255, 224, 53, 28, 95, 57, 148, 179,
            245, 162, 90, 37, 121, 0, 142, 219, 62, 234, 204, 225, 161,
        ];

        // This throwaway call has to be here to trigger the bug.
        crate::Hasher::new().update(input);

        // This assert fails when the bug is triggered.
        assert_eq!(crate::Hasher::new().update(input).finalize(), expected_hash);
    }
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
    // ChunkState.buf and Output.block are Aligned64 so that wide vector stores
    // into them (e.g. from memcpy) can store-to-load forward regardless of
    // stack layout. See the comment on Aligned64 in lib.rs, the measurements in
    // https://github.com/zooko/bench-hashes/issues/2, and the before/after
    // benchmarks in https://github.com/BLAKE3-team/BLAKE3/pull/582.
    assert_eq!(64, core::mem::align_of::<crate::Aligned64>());
    assert_eq!(0, core::mem::offset_of!(crate::Aligned64, 0));
}

#[cfg(feature = "zeroize")]
#[test]
fn test_zeroize() {
    use zeroize::Zeroize;

    let mut hash = crate::Hash([42; 32]);
    hash.zeroize();
    assert_eq!(hash.0, [0u8; 32]);

    let mut hasher = crate::Hasher {
        chunk_state: crate::ChunkState {
            cv: [42; 8],
            chunk_counter: 42,
            buf: crate::Aligned64([42; 64]),
            buf_len: 42,
            blocks_compressed: 42,
            flags: 42,
            platform: crate::Platform::Portable,
        },
        initial_chunk_counter: 42,
        key: [42; 8],
        cv_stack: [[42; 32]; { crate::MAX_DEPTH + 1 }].into(),
    };
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
    assert_eq!(&*hasher.cv_stack, &[[0u8; 32]; 0]);

    let mut output_reader = crate::OutputReader {
        inner: crate::Output {
            input_chaining_value: [42; 8],
            block: crate::Aligned64([42; 64]),
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
#[cfg(feature = "mmap")]
#[cfg(feature = "rayon")]
// NamedTempFile isn't Miri-compatible
#[cfg(not(miri))]
fn test_mmap_rayon() -> Result<(), std::io::Error> {
    // This is a brief test, since update_mmap_rayon() is mostly a wrapper around update_rayon(),
    // which already has substantial testing.
    use std::io::prelude::*;
    let mut input = vec![0; 1_000_000];
    paint_test_input(&mut input);
    let mut tempfile = tempfile::NamedTempFile::new()?;
    tempfile.write_all(&input)?;
    tempfile.flush()?;
    assert_eq!(
        crate::Hasher::new()
            .update_mmap_rayon(tempfile.path())?
            .finalize(),
        crate::hash(&input),
    );
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

// I had to move these tests out of the deprecated guts module, because leaving them there causes
// an un-silenceable warning: https://github.com/rust-lang/rust/issues/47238
#[cfg(test)]
#[allow(deprecated)]
mod guts_tests {
    use crate::guts::*;

    #[test]
    fn test_chunk() {
        assert_eq!(
            crate::hash(b"foo"),
            ChunkState::new(0).update(b"foo").finalize(true)
        );
    }

    #[test]
    fn test_parents() {
        let mut hasher = crate::Hasher::new();
        let mut buf = [0; crate::CHUNK_LEN];

        buf[0] = 'a' as u8;
        hasher.update(&buf);
        let chunk0_cv = ChunkState::new(0).update(&buf).finalize(false);

        buf[0] = 'b' as u8;
        hasher.update(&buf);
        let chunk1_cv = ChunkState::new(1).update(&buf).finalize(false);

        hasher.update(b"c");
        let chunk2_cv = ChunkState::new(2).update(b"c").finalize(false);

        let parent = parent_cv(&chunk0_cv, &chunk1_cv, false);
        let root = parent_cv(&parent, &chunk2_cv, true);
        assert_eq!(hasher.finalize(), root);
    }
}

/// The kernel reports are well formed: ascending from 0, and the
/// multithreaded report is the single-threaded one below the split, then
/// the split entry, which starts where hash_multithreaded may leave the
/// calling thread.
#[test]
#[cfg(feature = "std")]
fn test_kernel_reports() {
    let single = crate::kernel_report();
    assert!(!single.platform.is_empty());
    assert_eq!(single.kernels[0].from_len, 0);
    assert!(single.kernels.windows(2).all(|pair| pair[0].from_len < pair[1].from_len));
    let multi = crate::kernel_report_multithreaded();
    assert_eq!(multi.platform, single.platform);
    let below: Vec<_> = single.kernels.iter().filter(|kernel| kernel.from_len < crate::lanes::MIN_SPLIT_LEN).copied().collect();
    assert_eq!(&multi.kernels[..multi.kernels.len() - 1], &below[..]);
    assert_eq!(multi.kernels.last().unwrap().from_len, crate::lanes::MIN_SPLIT_LEN);
    for kernel in &multi.kernels {
        assert!(!kernel.name.is_empty() && !kernel.why.is_empty());
    }
    // The batch reports, for every message length a batch can take: the
    // same shape, the split last, where a batch may leave the calling thread.
    for message_len in [0, 1, 63, 64, 65, 128, 256, 1000, 1024, 1025, 40_000] {
        let many = crate::kernel_report_many(message_len);
        let many_multi = crate::kernel_report_many_multithreaded(message_len);
        assert_eq!(many.kernels[0].from_len, 0, "{message_len}-byte messages");
        assert!(many.kernels.windows(2).all(|pair| pair[0].from_len < pair[1].from_len), "{message_len}-byte messages");
        assert_eq!(&many_multi.kernels[..many.kernels.len()], &many.kernels[..]);
        assert_eq!(many_multi.kernels.len(), many.kernels.len() + 1);
        assert!(many_multi.kernels.last().unwrap().from_len > many.kernels.last().unwrap().from_len);
        assert!(many_multi.kernels.last().unwrap().from_len >= crate::lanes::MIN_SPLIT_LEN);
    }
}

/// Every length from one whole chunk and a byte to seventeen chunks,
/// against the reference implementation: the whole-plus-partial kernels
/// and the plans around them, on this machine's platform.
#[test]
fn test_every_length_1_to_17_chunks_against_reference() {
    let mut input = vec![0u8; 17 * CHUNK_LEN];
    paint_test_input(&mut input);
    for len in CHUNK_LEN + 1..=17 * CHUNK_LEN {
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

    /// A batch and an input large enough for the pool (64 KiB), from two
    /// threads at once, with small budgets.
    #[test]
    fn test_pool_from_two_callers() {
        let batch = input(1024 * 64, 1);
        let tree = input(65 * 1024 + 3, 2);
        let mut expected = vec![[0u8; OUT_LEN]; 1024];
        hash_many(&batch, 64, &mut expected);
        let expected_tree = hash(&tree);
        std::thread::scope(|scope| {
            for budget in [2, 4] {
                let (batch, tree, expected) = (&batch, &tree, &expected);
                scope.spawn(move || {
                    let mut out = vec![[0u8; OUT_LEN]; 1024];
                    hash_many_multithreaded_with_budget(batch, 64, &mut out, budget);
                    assert_eq!(&out, expected, "batch, budget {budget}");
                    assert_eq!(hash_multithreaded_with_budget(tree, budget), expected_tree, "tree, budget {budget}");
                });
            }
        });
    }

    /// A stream of one buffer and a little more, written in odd pieces.
    #[test]
    fn test_stream_past_one_buffer() {
        let data = input(crate::stream::BUFFER_LEN + 1000, 3);
        let mut stream = Stream::new();
        let mut at = 0;
        while at < data.len() {
            let buffer = stream.buffer();
            let n = buffer.len().min(data.len() - at).min(300_001);
            buffer[..n].copy_from_slice(&data[at..][..n]);
            stream.filled(n);
            at += n;
        }
        assert_eq!(stream.finalize(), hash(&data));
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
                lens.push((chunks * CHUNK_LEN) as i64 as usize + delta as usize);
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
}

// A long differential run against the reference implementation, off by
// default: `BLAKE3_DIFF_SECONDS=N cargo test --release --lib -- --ignored
// differential` (seed from BLAKE3_DIFF_SEED, default 1). Each step picks an
// entry point, a length (skewed toward block and chunk boundaries, up to 4
// MiB), bytes, and for batches a message length and count, from a
// xorshift64* stream; one to four threads run steps at once, so the SME2
// turn, the pool, and streams meet each other. Prints the steps run.
#[cfg(feature = "std")]
mod differential {
    use crate::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn len(&mut self, max: usize) -> usize {
            let len = match self.below(6) {
                0 => self.below(200) as usize,
                1 => (self.below(20) as usize) * BLOCK_LEN + [0, 1, 63][self.below(3) as usize],
                2 => (self.below(40) as usize) * CHUNK_LEN + [0, 1, 64, 1023][self.below(4) as usize],
                3 => 1usize << self.below(23),
                4 => (1usize << self.below(23)) + [1, 1023, 1024, 65][self.below(4) as usize],
                _ => self.below(1 << 20) as usize,
            };
            len.min(max)
        }
    }

    fn reference(key: Option<&[u8; KEY_LEN]>, context: Option<&str>, input: &[u8], out: &mut [u8]) {
        let mut hasher = match (key, context) {
            (Some(key), _) => reference_impl::Hasher::new_keyed(key),
            (_, Some(context)) => reference_impl::Hasher::new_derive_key(context),
            _ => reference_impl::Hasher::new(),
        };
        hasher.update(input);
        hasher.finalize(out);
    }

    fn step(rng: &mut Rng, data: &[u8]) {
        let pick = rng.below(9);
        let len = rng.len(data.len());
        let off = rng.below(64) as usize;
        let input = &data[off.min(data.len() - len)..][..len];
        let mut want = [0u8; OUT_LEN];
        match pick {
            0 => {
                reference(None, None, input, &mut want);
                assert_eq!(hash(input).as_bytes(), &want, "hash, {len} B");
            }
            1 => {
                reference(None, None, input, &mut want);
                let budget = 1 + rng.below(17) as usize;
                assert_eq!(hash_multithreaded_with_budget(input, budget).as_bytes(), &want, "hash_multithreaded, {len} B, budget {budget}");
            }
            2 => {
                let key: [u8; KEY_LEN] = core::array::from_fn(|i| i as u8 ^ len as u8);
                reference(Some(&key), None, input, &mut want);
                assert_eq!(keyed_hash(&key, input).as_bytes(), &want, "keyed_hash, {len} B");
            }
            3 => {
                reference(None, Some("differential"), input, &mut want);
                assert_eq!(derive_key("differential", input), want, "derive_key, {len} B");
            }
            4 => {
                reference(None, None, input, &mut want);
                let mut hasher = Hasher::new();
                let mut rest = input;
                while !rest.is_empty() {
                    let n = (rng.len(rest.len()) + 1).min(rest.len());
                    hasher.update(&rest[..n]);
                    rest = &rest[n..];
                }
                assert_eq!(hasher.finalize().as_bytes(), &want, "Hasher, {len} B");
            }
            5 => {
                reference(None, None, input, &mut want);
                let mut stream = if rng.below(2) == 0 { Stream::new() } else { Stream::new_multithreaded() };
                let mut rest = input;
                while !rest.is_empty() {
                    let buffer = stream.buffer();
                    let n = (rng.len(rest.len()) + 1).min(rest.len()).min(buffer.len());
                    buffer[..n].copy_from_slice(&rest[..n]);
                    stream.filled(n);
                    rest = &rest[n..];
                }
                assert_eq!(stream.finalize().as_bytes(), &want, "Stream, {len} B");
            }
            6 => {
                let mut full = vec![0u8; 100 + rng.below(3000) as usize];
                reference(None, None, input, &mut full);
                let mut xof = Hasher::new().update(input).finalize_xof();
                let mut got = vec![0u8; full.len()];
                xof.fill(&mut got);
                assert_eq!(got, full, "xof, {len} B");
            }
            _ => {
                let message_len = match rng.below(4) {
                    0 => BLOCK_LEN * (1 + rng.below(16) as usize),
                    1 => BLOCK_LEN * (17 + rng.below(240) as usize),
                    2 => rng.below(300) as usize,
                    _ => rng.len(1 << 16),
                };
                let slot = crate::many::slot_len(message_len);
                let count = (rng.below(300) as usize).min(data.len() / slot);
                let mut owned = data[..slot * count].to_vec();
                for message in owned.chunks_exact_mut(slot) {
                    message[message_len..].fill(0);
                }
                let input = &owned[..];
                let mut out = vec![[0u8; OUT_LEN]; count];
                let budget = 1 + rng.below(17) as usize;
                if pick == 7 {
                    hash_many(input, message_len, &mut out);
                } else {
                    hash_many_multithreaded_with_budget(input, message_len, &mut out, budget);
                }
                for (i, digest) in out.iter().enumerate() {
                    reference(None, None, &input[i * slot..][..message_len], &mut want);
                    assert_eq!(digest, &want, "hash_many ({pick}), {count} x {message_len} B, message {i}, budget {budget}");
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn differential() {
        let seconds: u64 = std::env::var("BLAKE3_DIFF_SECONDS").map(|s| s.parse().unwrap()).unwrap_or(10);
        let seed: u64 = std::env::var("BLAKE3_DIFF_SEED").map(|s| s.parse().unwrap()).unwrap_or(1);
        let mut data = vec![0u8; 4 << 20];
        crate::test::paint_test_input(&mut data);
        let data = &data;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
        let total = std::sync::atomic::AtomicU64::new(0);
        let mut round = 0u64;
        while std::time::Instant::now() < deadline {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ round.wrapping_add(1));
            let threads = 1 + rng.below(4);
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let total = &total;
                    let mut rng = Rng(rng.next() ^ t);
                    scope.spawn(move || {
                        for _ in 0..8 {
                            step(&mut rng, data);
                            total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    });
                }
            });
            round += 1;
        }
        println!("differential: seed {seed}, {round} rounds, {} steps, all equal to the reference", total.into_inner());
    }
}
