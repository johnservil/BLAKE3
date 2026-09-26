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
    #[cfg(blake3_sme2)]
    if chunked(len) && platform_is_sme2(platform) && (outputs.len() >= 16 || outputs.len() >= sme2_chunked_min(len)) {
        return hash_chunked(input, len, outputs, platform);
    }
    if outputs.len() < 2 || len == 0 || len > CHUNK_LEN || len % BLOCK_LEN != 0 {
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = *crate::hash_serial_on(&input[i * len..][..len], IV, 0, platform).as_bytes();
        }
        return;
    }
    for (messages, digests) in input.chunks(len * TABLE).zip(outputs.chunks_mut(TABLE)) {
        match len / BLOCK_LEN {
            1 => hash_run::<{ BLOCK_LEN }>(messages, digests, platform),
            2 => hash_blocks::<{ 2 * BLOCK_LEN }>(messages, digests, platform),
            3 => hash_blocks::<{ 3 * BLOCK_LEN }>(messages, digests, platform),
            4 => hash_blocks::<{ 4 * BLOCK_LEN }>(messages, digests, platform),
            5 => hash_blocks::<{ 5 * BLOCK_LEN }>(messages, digests, platform),
            6 => hash_blocks::<{ 6 * BLOCK_LEN }>(messages, digests, platform),
            7 => hash_blocks::<{ 7 * BLOCK_LEN }>(messages, digests, platform),
            8 => hash_blocks::<{ 8 * BLOCK_LEN }>(messages, digests, platform),
            9 => hash_blocks::<{ 9 * BLOCK_LEN }>(messages, digests, platform),
            10 => hash_blocks::<{ 10 * BLOCK_LEN }>(messages, digests, platform),
            11 => hash_blocks::<{ 11 * BLOCK_LEN }>(messages, digests, platform),
            12 => hash_blocks::<{ 12 * BLOCK_LEN }>(messages, digests, platform),
            13 => hash_blocks::<{ 13 * BLOCK_LEN }>(messages, digests, platform),
            14 => hash_blocks::<{ 14 * BLOCK_LEN }>(messages, digests, platform),
            15 => hash_blocks::<{ 15 * BLOCK_LEN }>(messages, digests, platform),
            16 => hash_blocks::<{ 16 * BLOCK_LEN }>(messages, digests, platform),
            _ => unreachable!("messages of 1 to 16 whole blocks"),
        }
    }
}

/// Fewest messages of 2 to 16 blocks that go to SME2 as one more group of
/// sixteen, its spare lanes reading the last message again: 10 in a batch
/// of fewer than sixteen, 5 left over after the SME2 groups (where NEON
/// work beside the SME2 kernels pays the SME unit's slow state as well).
/// Fewer run on the integer + NEON parent plans. ns per message, the plans
/// against the extra group: VM 256 B, 9 messages 60 / 69, 10 68 / 62, 12
/// 65 / 52; 128 B and 1 KiB cross at the same counts. Mac P-core (jobs
/// 261-264), 21 x 256 B 67.5 / 58.3, 53 x 256 B 51.1 / 45.5, 37 x 1 KiB
/// 201 / 193; the VM level at 21.
pub(crate) const SME2_TAIL_MIN: usize = 10;
pub(crate) const SME2_TAIL_MIN_AFTER_GROUPS: usize = 5;

/// Fewest messages of `len` bytes (whole blocks, 2 to 15 chunks) that SME2
/// hashes side by side as one group of sixteen, padded, by chunk count.
/// Fewer, alone or left over past whole groups, run through hash() one at
/// a time. The count at which a padded group's time equals that many
/// hash() calls, VM: 2 chunks 5.9, 3 8.4, 4 8.1, 6 9.5, 8 10.4, 12 11.4, 15
/// 11.6; Mac P-core (jobs 268-271) about the same or lower. Each entry is
/// the next count up.
const SME2_CHUNKED_MIN: [usize; 16] = [0, 0, 7, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12];

pub(crate) fn sme2_chunked_min(len: usize) -> usize {
    SME2_CHUNKED_MIN[len.div_ceil(CHUNK_LEN)]
}

/// Whether hash_many_on takes messages of `len` bytes side by side on
/// SME2: whole blocks, 2 to 15 chunks.
#[inline(always)]
fn chunked(len: usize) -> bool {
    cfg!(blake3_sme2) && len % BLOCK_LEN == 0 && (CHUNK_LEN + 1..=15 * CHUNK_LEN).contains(&len)
}

/// Messages of 2 to 15 chunks on SME2, sixteen side by side per group;
/// messages left over past whole groups make one more group, its spare
/// lanes repeating the last message, when there are sme2_chunked_min of
/// them, and otherwise run through hash() one at a time, first.
#[cfg(blake3_sme2)]
#[inline(never)]
fn hash_chunked(input: &[u8], len: usize, outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    const GROUP: usize = 16;
    let left = outputs.len() % GROUP;
    // Past whole groups a group pays from two messages sooner: hash() calls
    // beside the SME2 kernels run in their slow state (VM, 22 x 2 KiB: 630
    // ns per message with six alone, 486 with a padded group).
    let min = if outputs.len() > GROUP { sme2_chunked_min(len) - 2 } else { sme2_chunked_min(len) };
    let alone = if left >= min { 0 } else { left };
    let grouped = outputs.len() - alone;
    let (groups, rest) = outputs.split_at_mut(grouped);
    for (i, output) in rest.iter_mut().enumerate() {
        *output = *crate::hash_serial_on(&input[(grouped + i) * len..][..len], IV, 0, platform).as_bytes();
    }
    for (g, digests) in groups.chunks_mut(GROUP).enumerate() {
        let mut lanes = [core::ptr::null::<u8>(); GROUP];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let message = g * GROUP + i.min(digests.len() - 1);
            *lane = input[message * len..][..len].as_ptr();
        }
        // Sound: the caller found SME2 (platform_is_sme2), and every lane
        // points at a whole message of `len` bytes inside `input`.
        unsafe { crate::sme2::hash_chunked_messages(&lanes, len, IV, digests.as_flattened_mut()) };
    }
}

/// Whether a batch of `count` messages of `len` bytes runs SME2 kernels
/// (so takes the SME2 turn): sixteen messages or more; for messages of 2
/// to 16 whole blocks, SME2_TAIL_MIN or more; for messages of 2 to 15
/// chunks, sme2_chunked_min or more; and any message long enough for
/// hash() to take the turn itself.
pub(crate) fn sme2_sized(len: usize, count: usize) -> bool {
    count >= crate::SME2_SIZED_BATCH
        || (len > BLOCK_LEN && len <= CHUNK_LEN && len % BLOCK_LEN == 0 && count >= SME2_TAIL_MIN)
        || (chunked(len) && count >= sme2_chunked_min(len))
        || (len >= crate::SME2_SIZED_LEN && count >= 1)
}

/// Up to TABLE messages of N bytes, back to back in `messages`: each one's
/// hash, in one platform call (one-block messages; hash_blocks takes the
/// rest). Kept out of line: with all sixteen lengths inlined into
/// hash_many_on, batches of two 64-byte messages took 30% longer (VM,
/// bench-hashes); out of line they cost what they did before.
#[inline(never)]
fn hash_run<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    let mut table: [core::mem::MaybeUninit<&[u8; N]>; TABLE] = [core::mem::MaybeUninit::uninit(); TABLE];
    for (slot, message) in table[..outputs.len()].iter_mut().zip(messages.chunks_exact(N)) {
        slot.write(message.try_into().expect("messages of N bytes"));
    }
    // Sound: the first outputs.len() slots were written just above.
    let filled: &[&[u8; N]] = unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; N], outputs.len()) };
    let (flags, start, end) = if N == BLOCK_LEN { (CHUNK_START | CHUNK_END | ROOT, 0, 0) } else { (0, CHUNK_START, CHUNK_END | ROOT) };
    platform.hash_many::<N>(filled, IV, 0, IncrementCounter::No, flags, start, end, outputs.as_flattened_mut());
}

/// [`hash_run`] for messages of 2 to 16 whole blocks (N), on one of four
/// plans:
///
/// - SME2, SME2_TAIL_MIN or more messages in all (below sixteen) or
///   SME2_TAIL_MIN_AFTER_GROUPS or more past the groups of sixteen: whole
///   groups, the last one padded, its spare lanes pointing at the last
///   message again (no bytes move; the kernel computes every lane anyway
///   and stores the real ones alone).
/// - SME2 with fewer left over past the groups: those on the integer +
///   NEON plans first, then the groups (NEON work right after the SME2
///   kernels runs in the SME unit's slow state on the VM: 17 x 256 B, 68
///   ns per message after, 44 before).
/// - The integer + NEON plans (no SME2, or fewer messages): all at once.
/// - The C NEON kernel (no SHA-3 extension): four at a time, a fourth,
///   spare lane for three left over, the integer kernel for one or two.
#[inline(never)]
fn hash_blocks<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    const GROUP: usize = 16;
    let count = outputs.len();
    let sme2 = platform_is_sme2(platform);
    let left = count % GROUP;
    // `lanes`: the table's length (the messages, and spare lanes after
    // them); `first`: messages at the end hashed on NEON before the rest;
    // `scalar`: messages at the end hashed one at a time.
    let (lanes, first, scalar) = if sme2 && left >= if count > GROUP { SME2_TAIL_MIN_AFTER_GROUPS } else { SME2_TAIL_MIN } {
        (count.next_multiple_of(GROUP), 0, 0)
    } else if sme2 && count > GROUP {
        (count, left, 0)
    } else if neon_plans() || !cfg!(blake3_neon_hybrid) {
        (count, 0, 0)
    } else {
        // The C kernel for 2 to 16 blocks takes four messages at a time and
        // hashes any left over one by one in portable code; the one-message
        // integer kernel is faster for one or two of them (256 B, VM: 2
        // messages took 7% longer than a loop of hash()), and a fourth,
        // spare lane for three.
        match count % 4 {
            3 => (count + 1, 0, 0),
            r => (count - r, 0, r),
        }
    };
    let vector = count - scalar;
    let (vector_outputs, rest) = outputs.split_at_mut(vector);
    for (output, message) in rest.iter_mut().zip(messages[vector * N..].chunks_exact(N)) {
        *output = *crate::hash_serial_on(message, IV, 0, platform).as_bytes();
    }
    if vector == 0 {
        return;
    }
    let mut table: [core::mem::MaybeUninit<&[u8; N]>; TABLE + GROUP] = [core::mem::MaybeUninit::uninit(); TABLE + GROUP];
    let mut last: &[u8; N] = &[0; N];
    for (slot, message) in table[..vector].iter_mut().zip(messages.chunks_exact(N)) {
        last = message.try_into().expect("messages of N bytes");
        slot.write(last);
    }
    for slot in &mut table[vector..lanes] {
        slot.write(last);
    }
    // Sound: the first `lanes` slots were written just above.
    let filled: &[&[u8; N]] = unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; N], lanes) };
    let (flags, start, end) = (0, CHUNK_START, CHUNK_END | ROOT);
    #[cfg(blake3_sme2)]
    if sme2 && lanes > vector {
        // Sound: platform_is_sme2 means detect() found SME2 with 512-bit
        // streaming vectors.
        unsafe { crate::sme2::hash_messages::<N>(filled, IV, flags, start, end, vector_outputs.as_flattened_mut()) };
        return;
    }
    if lanes > vector {
        hash_padded(filled, flags, start, end, vector_outputs, platform);
        return;
    }
    let (groups, tail) = vector_outputs.split_at_mut(vector - first);
    if first > 0 {
        #[cfg(blake3_neon)]
        Platform::neon().expect("NEON beside SME2").hash_many::<N>(&filled[vector - first..], IV, 0, IncrementCounter::No, flags, start, end, tail.as_flattened_mut());
    }
    #[cfg(not(blake3_neon))]
    let _ = tail;
    platform.hash_many::<N>(&filled[..vector - first], IV, 0, IncrementCounter::No, flags, start, end, groups.as_flattened_mut());
}

/// `platform.hash_many` of `lanes` (the spare ones last) into a scratch
/// array, then the first `outputs.len()` digests to `outputs`. Out of line,
/// so only calls with spare lanes carry the scratch array in their frame.
#[inline(never)]
fn hash_padded<const N: usize>(lanes: &[&[u8; N]], flags: u8, start: u8, end: u8, outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    let mut spare = [[0u8; OUT_LEN]; 4];
    assert!(lanes.len() <= outputs.len() + 1 && lanes.len() % 4 == 0, "one spare lane, in a group of four");
    let rest = lanes.len() - 4;
    platform.hash_many::<N>(&lanes[..rest], IV, 0, IncrementCounter::No, flags, start, end, outputs[..rest].as_flattened_mut());
    platform.hash_many::<N>(&lanes[rest..], IV, 0, IncrementCounter::No, flags, start, end, spare.as_flattened_mut());
    let left = outputs.len() - rest;
    outputs[rest..].copy_from_slice(&spare[..left]);
}

/// Whether this CPU runs the integer + NEON kernels (hash_many's plans for
/// messages of whole blocks at one counter).
#[inline(always)]
fn neon_plans() -> bool {
    #[cfg(blake3_neon_hybrid)]
    return crate::neon_hybrid::sha3_detected();
    #[cfg(not(blake3_neon_hybrid))]
    false
}

#[inline(always)]
fn platform_is_sme2(platform: Platform) -> bool {
    #[cfg(blake3_sme2)]
    return matches!(platform, Platform::SME2);
    #[cfg(not(blake3_sme2))]
    {
        let _ = platform;
        false
    }
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

    /// Messages of 2 to 16 whole blocks on every platform this CPU has
    /// (SME2's padded last group, NEON's spare fourth lane, the integer
    /// kernel beside them), at every count to 40 and around TABLE, against
    /// hash() one message at a time.
    #[test]
    fn test_hash_many_padded_groups_every_platform() {
        #[allow(unused_mut)]
        let mut platforms = vec![Platform::detect(), Platform::Portable];
        #[cfg(blake3_neon)]
        platforms.push(Platform::neon().expect("NEON on AArch64"));
        for blocks in [2, 3, 4, 7, 16] {
            let len = blocks * BLOCK_LEN;
            for count in (0..=40).chain([122, 123, 127, 128, 129, 131, 133, 134, 143, 144, 150]) {
                let input = messages(len, count);
                for &platform in &platforms {
                    // Sixteen sentinels past the end catch a spare lane's store.
                    let mut out = vec![[0xAAu8; OUT_LEN]; count + 16];
                    hash_many_on(&input, len, &mut out[..count], platform);
                    assert!(out[count..].iter().all(|d| *d == [0xAA; OUT_LEN]), "{platform:?}, {count} x {len} B: a store past the last output");
                    for (i, digest) in out[..count].iter().enumerate() {
                        assert_eq!(*digest, *crate::hash(&input[i * len..][..len]).as_bytes(), "{platform:?}, message {i} of {count}, {len} bytes");
                    }
                }
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

    /// Messages of 2 to 15 chunks (whole blocks) on every platform this
    /// CPU has: on SME2 sixteen side by side, every tree shape from two
    /// chunks to fifteen, the last chunk whole or short, at counts that
    /// leave padded groups.
    #[test]
    fn test_hash_many_chunked_every_platform() {
        #[allow(unused_mut)]
        let mut platforms = vec![Platform::detect(), Platform::Portable];
        #[cfg(blake3_neon)]
        platforms.push(Platform::neon().expect("NEON on AArch64"));
        let lens = (2..=15).flat_map(|c| [c * CHUNK_LEN, (c - 1) * CHUNK_LEN + BLOCK_LEN]).chain([CHUNK_LEN + 192, 7 * CHUNK_LEN + 960, 16 * CHUNK_LEN]);
        for len in lens {
            for count in [0, 1, 2, 7, 8, 9, 15, 16, 17, 23, 24, 33] {
                let input = messages(len, count);
                for &platform in &platforms {
                    let mut out = vec![[0xAAu8; OUT_LEN]; count + 16];
                    hash_many_on(&input, len, &mut out[..count], platform);
                    assert!(out[count..].iter().all(|d| *d == [0xAA; OUT_LEN]), "{platform:?}, {count} x {len} B: a store past the last output");
                    for (i, digest) in out[..count].iter().enumerate() {
                        assert_eq!(*digest, *crate::hash(&input[i * len..][..len]).as_bytes(), "{platform:?}, message {i} of {count}, {len} bytes");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "exactly one message of the length per output")]
    fn test_hash_many_needs_the_input_it_names() {
        crate::hash_many(&[0u8; 100], 64, &mut [[0u8; OUT_LEN]; 2]);
    }
}
