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
use crate::{BLOCK_LEN, CHUNK_END, CHUNK_START, Hash, IV, IncrementCounter, OUT_LEN, ROOT};

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

/// The digests as one byte slice, for the kernels to write into. Sound:
/// `Hash` is `repr(transparent)` over `[u8; OUT_LEN]`.
pub(crate) fn hashes_as_bytes_mut(hashes: &mut [Hash]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(hashes.as_mut_ptr() as *mut u8, hashes.len() * OUT_LEN) }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::CHUNK_LEN;

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
        for count in [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 2049] {
            check(&vec![BLOCK_LEN; count]);
        }
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
        let contiguous = message(200 * BLOCK_LEN, 7);
        let inputs: Vec<&[u8]> = contiguous.chunks_exact(BLOCK_LEN).collect();
        let copies: Vec<Vec<u8>> = inputs.iter().map(|m| m.to_vec()).collect();
        let scattered: Vec<&[u8]> = copies.iter().map(|m| m.as_slice()).collect();
        let mut a = vec![Hash::from_bytes([0; OUT_LEN]); inputs.len()];
        let mut b = a.clone();
        crate::hash_many(&inputs, &mut a);
        crate::hash_many(&scattered, &mut b);
        assert_eq!(a, b);
        assert_eq!(a[199], crate::hash(inputs[199]));
    }

    #[test]
    #[should_panic(expected = "one output per message")]
    fn test_hash_many_needs_one_output_per_message() {
        crate::hash_many(&[b"a", b"b"], &mut [Hash::from_bytes([0; OUT_LEN])]);
    }
}
