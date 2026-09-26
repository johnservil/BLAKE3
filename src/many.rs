//! Many messages, one digest each: [`crate::hash_many`] and the pool's
//! pieces of it.
//!
//! A message of exactly one block (64 bytes) is one compression with the
//! chunk-start, chunk-end, and root flags at counter zero, which is the
//! shape the parent kernels already compress many lanes at a time: sixteen
//! per group on SME2's 512-bit streaming vectors, eight beside the integer
//! units on the NEON hybrids. A run of such messages is handed to the
//! platform's `hash_many` up to [`TABLE`] at a time, so one entry into
//! streaming mode covers eight groups. Every other message runs through
//! the same code as [`crate::hash`]: the single-chunk kernel up to one
//! chunk, the tree above.

use crate::platform::Platform;
use crate::{BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, Hash, IV, IncrementCounter, OUT_LEN, ROOT};

/// Most one-block messages per platform call: eight SME2 groups per entry
/// into streaming mode, the same as the tree walk's `sme2::DEGREE`.
/// Measured on the 16-vCPU VM at 1024 messages, ns per message: 128 →
/// 9.7, 256 → 10.2, 512 and 1024 → 12.5. The slowdown follows the run
/// length handed to the platform, with the kernel call's own group count,
/// the frame size, the buffers' alignment, and the loops below each ruled
/// out by measurement; what remains is scanning far ahead of the kernel.
/// On an M4 the entry into streaming mode costs about a microsecond, so
/// a longer run may pay there; measure before changing this.
pub(crate) const TABLE: usize = 128;

/// `outputs[i] = hash(inputs[i])` for every message, on `platform`.
/// Requires `inputs.len() == outputs.len()`.
pub(crate) fn hash_many_on(inputs: &[&[u8]], outputs: &mut [Hash], platform: Platform) {
    let done = hash_many_until_longer(inputs, outputs, platform, usize::MAX);
    debug_assert_eq!(done, inputs.len());
}

/// [`hash_many_on`] in order up to the first message longer than `longest`
/// bytes; returns how many messages it hashed (all of them when none is
/// longer). Requires `inputs.len() == outputs.len()`.
pub(crate) fn hash_many_until_longer(
    inputs: &[&[u8]],
    outputs: &mut [Hash],
    platform: Platform,
    longest: usize,
) -> usize {
    assert_eq!(inputs.len(), outputs.len(), "one output per message");
    if inputs.len() == 1 {
        // One message costs what hash() costs; the run scan below is not free.
        if inputs[0].len() > longest {
            return 0;
        }
        outputs[0] = crate::hash_serial_on(inputs[0], IV, 0, platform);
        return 1;
    }
    // Filled up to `run` before each use; an initialised table would cost
    // 8 KiB of stores per call, twice a single message's hash.
    let mut table: [core::mem::MaybeUninit<&[u8; BLOCK_LEN]>; TABLE] = [core::mem::MaybeUninit::uninit(); TABLE];
    let mut i = 0;
    while i < inputs.len() {
        /* A run of equal messages of 2 to 16 whole blocks: one platform call. */
        let len = inputs[i].len();
        if len > BLOCK_LEN && len <= CHUNK_LEN && len % BLOCK_LEN == 0 && len <= longest {
            let run = inputs[i..].iter().take(TABLE).take_while(|message| message.len() == len).count();
            if run > 1 {
                hash_blocks_run(&inputs[i..i + run], &mut outputs[i..i + run], platform);
                i += run;
                continue;
            }
        }
        let run = inputs[i..]
            .iter()
            .take(TABLE)
            .take_while(|message| message.len() == BLOCK_LEN)
            .count();
        if run == 0 {
            if inputs[i].len() > longest {
                return i;
            }
            outputs[i] = crate::hash_serial_on(inputs[i], IV, 0, platform);
            i += 1;
            continue;
        }
        for (slot, message) in table[..run].iter_mut().zip(&inputs[i..i + run]) {
            slot.write(message[..].try_into().expect("a one-block message"));
        }
        // Sound: the first `run` slots were written just above.
        let filled: &[&[u8; BLOCK_LEN]] =
            unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; BLOCK_LEN], run) };
        platform.hash_many::<BLOCK_LEN>(
            filled,
            IV,
            0,
            IncrementCounter::No,
            CHUNK_START | CHUNK_END | ROOT,
            0,
            0,
            hashes_as_bytes_mut(&mut outputs[i..i + run]),
        );
        i += run;
    }
    inputs.len()
}

/// Equal messages of 2 to 16 whole blocks, at most TABLE: each one's hash,
/// in one platform call for their length.
fn hash_blocks_run(inputs: &[&[u8]], outputs: &mut [Hash], platform: Platform) {
    let len = inputs[0].len();
    debug_assert!(inputs.iter().all(|m| m.len() == len) && inputs.len() <= TABLE);
    match len / BLOCK_LEN {
        2 => hash_run::<{ 2 * BLOCK_LEN }>(inputs, outputs, platform),
        3 => hash_run::<{ 3 * BLOCK_LEN }>(inputs, outputs, platform),
        4 => hash_run::<{ 4 * BLOCK_LEN }>(inputs, outputs, platform),
        5 => hash_run::<{ 5 * BLOCK_LEN }>(inputs, outputs, platform),
        6 => hash_run::<{ 6 * BLOCK_LEN }>(inputs, outputs, platform),
        7 => hash_run::<{ 7 * BLOCK_LEN }>(inputs, outputs, platform),
        8 => hash_run::<{ 8 * BLOCK_LEN }>(inputs, outputs, platform),
        9 => hash_run::<{ 9 * BLOCK_LEN }>(inputs, outputs, platform),
        10 => hash_run::<{ 10 * BLOCK_LEN }>(inputs, outputs, platform),
        11 => hash_run::<{ 11 * BLOCK_LEN }>(inputs, outputs, platform),
        12 => hash_run::<{ 12 * BLOCK_LEN }>(inputs, outputs, platform),
        13 => hash_run::<{ 13 * BLOCK_LEN }>(inputs, outputs, platform),
        14 => hash_run::<{ 14 * BLOCK_LEN }>(inputs, outputs, platform),
        15 => hash_run::<{ 15 * BLOCK_LEN }>(inputs, outputs, platform),
        16 => hash_run::<{ 16 * BLOCK_LEN }>(inputs, outputs, platform),
        _ => unreachable!("messages of 2 to 16 whole blocks"),
    }
}

fn hash_run<const N: usize>(inputs: &[&[u8]], outputs: &mut [Hash], platform: Platform) {
    let mut table: [core::mem::MaybeUninit<&[u8; N]>; TABLE] = [core::mem::MaybeUninit::uninit(); TABLE];
    for (slot, message) in table[..inputs.len()].iter_mut().zip(inputs) {
        slot.write(message[..].try_into().expect("messages of N bytes"));
    }
    // Sound: the first inputs.len() slots were written just above.
    let filled: &[&[u8; N]] = unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; N], inputs.len()) };
    platform.hash_many::<N>(filled, IV, 0, IncrementCounter::No, 0, CHUNK_START, CHUNK_END | ROOT, hashes_as_bytes_mut(outputs));
}

/// `outputs[i] = hash(input[i * len..][..len])` for every message, on
/// `platform`: messages of 1 to 16 whole blocks go to the platform's
/// hash_many TABLE at a time, others through the one-message path.
/// Requires `input.len() == len * outputs.len()`.
pub(crate) fn hash_many_equal_on(input: &[u8], len: usize, outputs: &mut [Hash], platform: Platform) {
    assert_eq!(Some(input.len()), len.checked_mul(outputs.len()), "input holds exactly one message of the length per output");
    if outputs.len() < 2 || len == 0 || len > CHUNK_LEN || len % BLOCK_LEN != 0 {
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = crate::hash_serial_on(&input[i * len..][..len], IV, 0, platform);
        }
        return;
    }
    for (messages, digests) in input.chunks(len * TABLE).zip(outputs.chunks_mut(TABLE)) {
        match len / BLOCK_LEN {
            1 => hash_equal_run::<{ BLOCK_LEN }>(messages, digests, platform),
            2 => hash_equal_run::<{ 2 * BLOCK_LEN }>(messages, digests, platform),
            3 => hash_equal_run::<{ 3 * BLOCK_LEN }>(messages, digests, platform),
            4 => hash_equal_run::<{ 4 * BLOCK_LEN }>(messages, digests, platform),
            5 => hash_equal_run::<{ 5 * BLOCK_LEN }>(messages, digests, platform),
            6 => hash_equal_run::<{ 6 * BLOCK_LEN }>(messages, digests, platform),
            7 => hash_equal_run::<{ 7 * BLOCK_LEN }>(messages, digests, platform),
            8 => hash_equal_run::<{ 8 * BLOCK_LEN }>(messages, digests, platform),
            9 => hash_equal_run::<{ 9 * BLOCK_LEN }>(messages, digests, platform),
            10 => hash_equal_run::<{ 10 * BLOCK_LEN }>(messages, digests, platform),
            11 => hash_equal_run::<{ 11 * BLOCK_LEN }>(messages, digests, platform),
            12 => hash_equal_run::<{ 12 * BLOCK_LEN }>(messages, digests, platform),
            13 => hash_equal_run::<{ 13 * BLOCK_LEN }>(messages, digests, platform),
            14 => hash_equal_run::<{ 14 * BLOCK_LEN }>(messages, digests, platform),
            15 => hash_equal_run::<{ 15 * BLOCK_LEN }>(messages, digests, platform),
            16 => hash_equal_run::<{ 16 * BLOCK_LEN }>(messages, digests, platform),
            _ => unreachable!("messages of 1 to 16 whole blocks"),
        }
    }
}

/// Up to TABLE messages of N bytes, back to back in `messages`: each one's
/// hash, in one platform call.
fn hash_equal_run<const N: usize>(messages: &[u8], outputs: &mut [Hash], platform: Platform) {
    let mut table: [core::mem::MaybeUninit<&[u8; N]>; TABLE] = [core::mem::MaybeUninit::uninit(); TABLE];
    for (slot, message) in table[..outputs.len()].iter_mut().zip(messages.chunks_exact(N)) {
        slot.write(message.try_into().expect("messages of N bytes"));
    }
    // Sound: the first outputs.len() slots were written just above.
    let filled: &[&[u8; N]] = unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; N], outputs.len()) };
    let (flags, start, end) = if N == BLOCK_LEN { (CHUNK_START | CHUNK_END | ROOT, 0, 0) } else { (0, CHUNK_START, CHUNK_END | ROOT) };
    platform.hash_many::<N>(filled, IV, 0, IncrementCounter::No, flags, start, end, hashes_as_bytes_mut(outputs));
}

/// The digests as one byte slice, for the kernels to write into. Sound:
/// `Hash` is `repr(transparent)` over `[u8; OUT_LEN]`.
pub(crate) fn hashes_as_bytes_mut(hashes: &mut [Hash]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(hashes.as_mut_ptr() as *mut u8, hashes.len() * OUT_LEN) }
}

#[cfg(test)]
mod test {
    use super::*;

    /// A deterministic message of `len` bytes, distinct per `seed`:
    /// little-endian 64-bit words `seed << 48 | index`, so every block
    /// differs and a kernel that mixed up its lanes would be caught.
    fn message(len: usize, seed: u64) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..len.div_ceil(8) as u64).flat_map(|i| (seed << 48 | i).to_le_bytes()).collect();
        bytes.truncate(len);
        bytes
    }

    fn check(lens: &[usize]) {
        let messages: Vec<Vec<u8>> = lens.iter().enumerate().map(|(i, &len)| message(len, i as u64)).collect();
        let inputs: Vec<&[u8]> = messages.iter().map(|m| m.as_slice()).collect();
        let mut outputs = vec![Hash::from_bytes([0; OUT_LEN]); inputs.len()];
        crate::hash_many(&inputs, &mut outputs);
        for (i, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
            assert_eq!(*output, crate::hash(input), "message {i} of {} bytes in {lens:?}", input.len());
        }
    }

    #[test]
    fn test_hash_many_runs_of_blocks() {
        // Every NEON parent plan (1 to 16), then SME2 groups and their
        // remainders (13 to 15 left over take an overlapping group).
        for count in (0..=17).chain([24, 29, 30, 31, 32, 33, 45, 61, 127, 128, 129, 1021, 1022, 1023, 1024, 1025, 2049]) {
            check(&vec![BLOCK_LEN; count]);
        }
    }

    /// Runs of equal messages of 2 to 16 whole blocks, at counts that fill
    /// SME2 groups and leave remainders, against hash() one at a time.
    #[test]
    fn test_hash_many_runs_of_whole_blocks() {
        for blocks in 2..=16 {
            for count in [2, 3, 15, 16, 17, 31, 32, 33, 127, 128, 129, 300] {
                check(&vec![blocks * BLOCK_LEN; count]);
            }
        }
        check(&[256, 256, 256, 64, 64, 256, 1024, 1024, 128, 128, 128, 192, 1024, 65, 256, 256]);
    }

    /// hash_many_equal at every length a batch can take, whole blocks or
    /// not, and counts around the SME2 groups and TABLE, against hash().
    #[test]
    fn test_hash_many_equal() {
        for len in [0, 1, 63, 64, 65, 128, 191, 192, 256, 1000, 1024, 1025, 3000] {
            for count in [0, 1, 2, 15, 16, 17, 127, 128, 129, 300] {
                let input = message(len * count, len as u64);
                let mut out = vec![[0u8; OUT_LEN]; count];
                crate::hash_many_equal(&input, len, &mut out);
                for (i, digest) in out.iter().enumerate() {
                    assert_eq!(*digest, *crate::hash(&input[i * len..][..len]).as_bytes(), "message {i} of {count}, {len} bytes");
                }
                #[cfg(feature = "std")]
                {
                    let mut mt = vec![[0u8; OUT_LEN]; count];
                    crate::hash_many_equal_multithreaded(&input, len, &mut mt);
                    assert_eq!(mt, out, "multithreaded, {count} messages of {len} bytes");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "exactly one message of the length per output")]
    fn test_hash_many_equal_needs_the_input_it_names() {
        crate::hash_many_equal(&[0u8; 100], 64, &mut [[0u8; OUT_LEN]; 2]);
    }

    #[test]
    fn test_hash_many_mixed_lengths() {
        check(&[0, 1, 63, 64, 65, 64, 64, 1023, 1024, 1025, 64, 4096, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 3 * CHUNK_LEN + 7, 64]);
        check(&[65; 40]);
        check(&[1; 40]);
        check(&[0; 3]);
    }

    /// A batch of one-block messages scattered through memory takes the
    /// same path as a contiguous one and agrees with it.
    #[test]
    fn test_hash_many_scattered_blocks() {
        // 205: the last platform call's 77 messages leave 13 over an SME2 group.
        let contiguous = message(205 * BLOCK_LEN, 7);
        let inputs: Vec<&[u8]> = contiguous.chunks_exact(BLOCK_LEN).collect();
        let copies: Vec<Vec<u8>> = inputs.iter().map(|m| m.to_vec()).collect();
        let scattered: Vec<&[u8]> = copies.iter().map(|m| m.as_slice()).collect();
        let mut a = vec![Hash::from_bytes([0; OUT_LEN]); inputs.len()];
        let mut b = a.clone();
        crate::hash_many(&inputs, &mut a);
        crate::hash_many(&scattered, &mut b);
        assert_eq!(a, b);
        for (digest, input) in a.iter().zip(&inputs) {
            assert_eq!(*digest, crate::hash(input));
        }
    }

    #[test]
    #[should_panic(expected = "one output per message")]
    fn test_hash_many_needs_one_output_per_message() {
        crate::hash_many(&[b"a", b"b"], &mut [Hash::from_bytes([0; OUT_LEN])]);
    }
}
