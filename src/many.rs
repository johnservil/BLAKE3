//! Many messages of one length, one digest each: [`crate::hash_many`]
//! and the pool's pieces of it.
//!
//! A message of 1 to 16 whole blocks (64 B to 1 KiB) is one chunk, the
//! shape the chunk kernels already hash many lanes at a time: sixteen per
//! group on SME2's 512-bit streaming vectors, several beside the integer
//! units on the NEON hybrids. The messages are handed to the platform's
//! `hash_many` up to [`TABLE`] at a time, so one entry into streaming mode
//! covers eight groups. Messages of other lengths run through the same
//! code as [`crate::hash`], one at a time.

use crate::platform::Platform;
use crate::{BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, IV, IncrementCounter, OUT_LEN, ROOT};

/// Most messages per platform call: eight SME2 groups per entry into
/// streaming mode, the same as the tree walk's `sme2::DEGREE`. Measured on
/// the 16-vCPU VM at 1024 one-block messages, ns per message: 128 → 9.7,
/// 256 → 10.2, 512 and 1024 → 12.5. The slowdown follows the run length
/// handed to the platform, with the kernel call's own group count, the
/// frame size, the buffers' alignment, and the loops below each ruled out
/// by measurement; what remains is scanning far ahead of the kernel. On an
/// M4 the entry into streaming mode costs about a microsecond, so a longer
/// run may pay there; measure before changing this.
pub(crate) const TABLE: usize = 128;

/// `outputs[i] = hash(input[i * len..][..len])` for every message, on
/// `platform`: messages of 1 to 16 whole blocks go to the platform's
/// hash_many TABLE at a time, others through the one-message path.
/// Requires `input.len() == len * outputs.len()`.
pub(crate) fn hash_many_on(input: &[u8], len: usize, outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    assert_eq!(Some(input.len()), len.checked_mul(outputs.len()), "input holds exactly one message of the length per output");
    if outputs.len() < 2 || len == 0 || len > CHUNK_LEN || len % BLOCK_LEN != 0 {
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = *crate::hash_serial_on(&input[i * len..][..len], IV, 0, platform).as_bytes();
        }
        return;
    }
    for (messages, digests) in input.chunks(len * TABLE).zip(outputs.chunks_mut(TABLE)) {
        match len / BLOCK_LEN {
            1 => hash_run::<{ BLOCK_LEN }>(messages, digests, platform),
            2 => hash_run::<{ 2 * BLOCK_LEN }>(messages, digests, platform),
            3 => hash_run::<{ 3 * BLOCK_LEN }>(messages, digests, platform),
            4 => hash_run::<{ 4 * BLOCK_LEN }>(messages, digests, platform),
            5 => hash_run::<{ 5 * BLOCK_LEN }>(messages, digests, platform),
            6 => hash_run::<{ 6 * BLOCK_LEN }>(messages, digests, platform),
            7 => hash_run::<{ 7 * BLOCK_LEN }>(messages, digests, platform),
            8 => hash_run::<{ 8 * BLOCK_LEN }>(messages, digests, platform),
            9 => hash_run::<{ 9 * BLOCK_LEN }>(messages, digests, platform),
            10 => hash_run::<{ 10 * BLOCK_LEN }>(messages, digests, platform),
            11 => hash_run::<{ 11 * BLOCK_LEN }>(messages, digests, platform),
            12 => hash_run::<{ 12 * BLOCK_LEN }>(messages, digests, platform),
            13 => hash_run::<{ 13 * BLOCK_LEN }>(messages, digests, platform),
            14 => hash_run::<{ 14 * BLOCK_LEN }>(messages, digests, platform),
            15 => hash_run::<{ 15 * BLOCK_LEN }>(messages, digests, platform),
            16 => hash_run::<{ 16 * BLOCK_LEN }>(messages, digests, platform),
            _ => unreachable!("messages of 1 to 16 whole blocks"),
        }
    }
}

/// Up to TABLE messages of N bytes, back to back in `messages`: each one's
/// hash, in one platform call. Kept out of line: with all sixteen lengths
/// inlined into hash_many_on, batches of two 64-byte messages took 30%
/// longer (VM, bench-hashes); out of line they cost what they did before.
#[inline(never)]
fn hash_run<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    // On AArch64 the kernel for 2 to 16 blocks takes four messages at a
    // time and hashes any left over one by one in portable code; the
    // one-message integer kernel is faster for those (256 B, VM: 2 and 3
    // messages took 7% longer than a loop of hash()).
    #[cfg(blake3_neon_hybrid)]
    let outputs = if N > BLOCK_LEN {
        let grouped = outputs.len() / 4 * 4;
        let (grouped_outputs, rest) = outputs.split_at_mut(grouped);
        for (output, message) in rest.iter_mut().zip(messages[grouped * N..].chunks_exact(N)) {
            *output = *crate::hash_serial_on(message, IV, 0, platform).as_bytes();
        }
        grouped_outputs
    } else {
        outputs
    };
    let mut table: [core::mem::MaybeUninit<&[u8; N]>; TABLE] = [core::mem::MaybeUninit::uninit(); TABLE];
    for (slot, message) in table[..outputs.len()].iter_mut().zip(messages.chunks_exact(N)) {
        slot.write(message.try_into().expect("messages of N bytes"));
    }
    // Sound: the first outputs.len() slots were written just above.
    let filled: &[&[u8; N]] = unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; N], outputs.len()) };
    let (flags, start, end) = if N == BLOCK_LEN { (CHUNK_START | CHUNK_END | ROOT, 0, 0) } else { (0, CHUNK_START, CHUNK_END | ROOT) };
    platform.hash_many::<N>(filled, IV, 0, IncrementCounter::No, flags, start, end, outputs.as_flattened_mut());
}

#[cfg(test)]
mod test {
    use super::*;

    /// `count` deterministic messages of `len` bytes, back to back: little-
    /// endian 64-bit words `len << 48 | index`, so every block differs and
    /// a kernel that mixed up its lanes would be caught.
    fn messages(len: usize, count: usize) -> Vec<u8> {
        let total = len * count;
        let mut bytes: Vec<u8> = (0..total.div_ceil(8) as u64).flat_map(|i| ((len as u64) << 48 | i).to_le_bytes()).collect();
        bytes.truncate(total);
        bytes
    }

    /// hash_many (and hash_many_multithreaded) of `count` messages of
    /// `len` bytes against hash() one message at a time.
    fn check(len: usize, count: usize) {
        let input = messages(len, count);
        let mut out = vec![[0u8; OUT_LEN]; count];
        crate::hash_many(&input, len, &mut out);
        for (i, digest) in out.iter().enumerate() {
            assert_eq!(*digest, *crate::hash(&input[i * len..][..len]).as_bytes(), "message {i} of {count}, {len} bytes");
        }
        #[cfg(feature = "std")]
        {
            let mut mt = vec![[0u8; OUT_LEN]; count];
            crate::hash_many_multithreaded(&input, len, &mut mt);
            assert_eq!(mt, out, "multithreaded, {count} messages of {len} bytes");
        }
    }

    /// One-block messages: every NEON parent plan (1 to 16), then SME2
    /// groups and their remainders (13 to 15 left over take an overlapping
    /// group), TABLE's edges, and the pool's split.
    #[test]
    fn test_hash_many_blocks() {
        for count in (0..=17).chain([24, 29, 30, 31, 32, 33, 45, 61, 127, 128, 129, 205, 1021, 1022, 1023, 1024, 1025, 2049]) {
            check(BLOCK_LEN, count);
        }
    }

    /// Messages of 2 to 16 whole blocks, at counts that fill SME2 groups
    /// and leave remainders.
    #[test]
    fn test_hash_many_whole_blocks() {
        for blocks in 2..=16 {
            for count in [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 31, 32, 33, 127, 128, 129, 130, 300] {
                check(blocks * BLOCK_LEN, count);
            }
        }
    }

    /// Lengths other than whole blocks up to a chunk, one message at a time.
    #[test]
    fn test_hash_many_other_lengths() {
        for len in [0, 1, 63, 65, 191, 1000, 1025, 3000, 3 * CHUNK_LEN + 7] {
            for count in [0, 1, 2, 15, 16, 17, 129] {
                check(len, count);
            }
        }
    }

    #[test]
    #[should_panic(expected = "exactly one message of the length per output")]
    fn test_hash_many_needs_the_input_it_names() {
        crate::hash_many(&[0u8; 100], 64, &mut [[0u8; OUT_LEN]; 2]);
    }
}
