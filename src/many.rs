//! Many messages of one length, one digest each: [`crate::hash_many`]
//! and the pool's pieces of it.
//!
//! Each message sits in a slot of whole blocks ([`slot_len`]), zero past
//! its end, so every kernel reads whole blocks in place and records the
//! last block's true length. A message of up to a chunk is one chunk, the
//! shape the kernels hash many lanes at a time: sixteen per group on
//! SME2's 512-bit streaming vectors, several beside the integer units on
//! the NEON hybrids, up to [`TABLE`] per call so one entry into streaming
//! mode covers eight groups. Messages of 2 to 15 chunks go side by side on
//! SME2; others through the same code as [`crate::hash`], one at a time.

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

/// The bytes each message of `len` bytes takes in a batch: `len` rounded up
/// to whole blocks, one block for an empty message.
#[inline(always)]
pub(crate) fn slot_len(len: usize) -> usize {
    len.next_multiple_of(BLOCK_LEN).max(BLOCK_LEN)
}

/// `outputs[i] = hash(input[i * slot..][..len])` for every message, with
/// `slot = slot_len(len)`, on `platform`: messages of up to a chunk go to
/// the kernels TABLE at a time, 2 to 15 chunks side by side on SME2, others
/// through the one-message path. Requires `input.len() == slot *
/// outputs.len()` and every slot's bytes past `len` zero (checked in debug
/// builds; a nonzero byte there changes that message's digest).
pub(crate) fn hash_many_on(input: &[u8], len: usize, outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    // One-block messages first, on the path they took before slots (a
    // shared entry cost 64 B x 3 7% in the regression check).
    if len == BLOCK_LEN && outputs.len() >= 2 {
        assert_eq!(Some(input.len()), BLOCK_LEN.checked_mul(outputs.len()), "input holds one slot of whole blocks per output");
        for (messages, digests) in input.chunks(BLOCK_LEN * TABLE).zip(outputs.chunks_mut(TABLE)) {
            hash_run::<{ BLOCK_LEN }>(messages, digests, platform);
        }
        return;
    }
    let slot = slot_len(len);
    assert_eq!(Some(input.len()), slot.checked_mul(outputs.len()), "input holds one slot of whole blocks per output");
    debug_assert!(len % BLOCK_LEN == 0 || input.chunks_exact(slot).all(|s| s[len..].iter().all(|&b| b == 0)), "every slot's bytes past its message are zero");
    if len == 0 {
        outputs.fill(*crate::hash_serial_on(&[], IV, 0, platform).as_bytes());
        return;
    }
    #[cfg(blake3_sme2)]
    if chunked(len) && platform_is_sme2(platform) && (outputs.len() >= 16 || outputs.len() >= sme2_chunked_min(len)) {
        return hash_chunked(input, len, outputs, platform);
    }
    #[cfg(blake3_neon_hybrid)]
    if (CHUNK_LEN + 1..=2 * CHUNK_LEN).contains(&len) && outputs.len() >= 2 && neon_plans() {
        return hash_two_chunks(input, len, outputs);
    }
    if outputs.len() < 2 || len > CHUNK_LEN {
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = *crate::hash_serial_on(&input[i * slot..][..len], IV, 0, platform).as_bytes();
        }
        return;
    }
    let plans = neon_plans();
    let last_len = len - (slot - BLOCK_LEN);
    for (messages, digests) in input.chunks(slot * TABLE).zip(outputs.chunks_mut(TABLE)) {
        match slot / BLOCK_LEN {
            1 => hash_run_short::<{ BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            2 => hash_blocks::<{ 2 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            3 => hash_blocks::<{ 3 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            4 => hash_blocks::<{ 4 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            5 => hash_blocks::<{ 5 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            6 => hash_blocks::<{ 6 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            7 => hash_blocks::<{ 7 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            8 => hash_blocks::<{ 8 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            9 => hash_blocks::<{ 9 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            10 => hash_blocks::<{ 10 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            11 => hash_blocks::<{ 11 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            12 => hash_blocks::<{ 12 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            13 => hash_blocks::<{ 13 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            14 => hash_blocks::<{ 14 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            15 => hash_blocks::<{ 15 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            16 => hash_blocks::<{ 16 * BLOCK_LEN }>(messages, digests, platform, plans, last_len),
            _ => unreachable!("messages of up to a chunk"),
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
/// SME2: 2 to 15 chunks.
#[inline(always)]
fn chunked(len: usize) -> bool {
    cfg!(blake3_sme2) && (CHUNK_LEN + 1..=15 * CHUNK_LEN).contains(&len)
}

/// Messages of 2 to 15 chunks on SME2, sixteen side by side per group;
/// messages left over past whole groups make one more group, its spare
/// lanes repeating the last message, when there are sme2_chunked_min of
/// them, and otherwise run through hash() one at a time, first.
#[cfg(blake3_sme2)]
#[inline(never)]
fn hash_chunked(input: &[u8], len: usize, outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    const GROUP: usize = 16;
    let slot = slot_len(len);
    let left = outputs.len() % GROUP;
    // Past whole groups a group pays from two messages sooner: hash() calls
    // beside the SME2 kernels run in their slow state (VM, 22 x 2 KiB: 630
    // ns per message with six alone, 486 with a padded group).
    let min = if outputs.len() > GROUP { sme2_chunked_min(len) - 2 } else { sme2_chunked_min(len) };
    let alone = if left >= min { 0 } else { left };
    let grouped = outputs.len() - alone;
    let (groups, rest) = outputs.split_at_mut(grouped);
    for (i, output) in rest.iter_mut().enumerate() {
        *output = *crate::hash_serial_on(&input[(grouped + i) * slot..][..len], IV, 0, platform).as_bytes();
    }
    for (g, digests) in groups.chunks_mut(GROUP).enumerate() {
        let mut lanes = [core::ptr::null::<u8>(); GROUP];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let message = g * GROUP + i.min(digests.len() - 1);
            *lane = input[message * slot..][..slot].as_ptr();
        }
        // Sound: the caller found SME2 (platform_is_sme2), and every lane
        // points at a whole slot inside `input`, zero past its message.
        unsafe { crate::sme2::hash_chunked_messages(&lanes, len, IV, digests.as_flattened_mut()) };
    }
}

/// Messages of two chunks (1025 to 2048 bytes) side by side on the integer
/// + NEON parent plans, sixteen at a time: every message's first chunk in
/// one plan call at counter 0, every second chunk in another at counter 1,
/// then every root. hash() gives each message a NEON pair with a spare
/// lane (q1) or a pair alone (k2); side by side the plans fill their
/// lanes with the batch. VM, ns per message, one hash() each / side by
/// side: 2000 B x 2 1018 / 927, x 3 1036 / 653, x 4 1050 / 697, x 8 690
/// (SME2 from 7); 1100 B x 3 442 / 379. Commonware's four NEON lanes: 2000
/// B x 4 824 (Mac P-core 914 against servil's 1050 before, job 283).
#[cfg(blake3_neon_hybrid)]
#[inline(never)]
fn hash_two_chunks(input: &[u8], len: usize, outputs: &mut [[u8; OUT_LEN]]) {
    use crate::PARENT;
    let slot = slot_len(len);
    let second = len - CHUNK_LEN;
    let second_blocks = second.div_ceil(BLOCK_LEN);
    let last_len = second - (second_blocks - 1) * BLOCK_LEN;
    // Sixteen at a time, as the plans take them: small arrays to zero.
    const GROUP: usize = 16;
    for (messages, digests) in input.chunks(slot * GROUP).zip(outputs.chunks_mut(GROUP)) {
        let count = digests.len();
        let mut firsts: [core::mem::MaybeUninit<&[u8; CHUNK_LEN]>; GROUP] = [core::mem::MaybeUninit::uninit(); GROUP];
        let mut seconds: [core::mem::MaybeUninit<*const u8>; GROUP] = [core::mem::MaybeUninit::uninit(); GROUP];
        for (i, message) in messages.chunks_exact(slot).enumerate() {
            firsts[i].write(message[..CHUNK_LEN].try_into().expect("a whole first chunk"));
            seconds[i].write(message[CHUNK_LEN..].as_ptr());
        }
        let mut cvs = [[0u8; OUT_LEN]; 2 * GROUP];
        let mut pairs = [[0u8; BLOCK_LEN]; GROUP];
        // Sound: the first `count` entries of both tables were written just
        // above; each second-chunk pointer reaches `second_blocks` whole
        // blocks inside its slot, zero past the message; the caller found
        // the SHA-3 extension (neon_plans).
        unsafe {
            let firsts = core::slice::from_raw_parts(firsts.as_ptr() as *const &[u8; CHUNK_LEN], count);
            crate::neon_hybrid::hash_many_last_len(firsts, IV, 0, IncrementCounter::No, 0, CHUNK_START, CHUNK_END, BLOCK_LEN, cvs[..count].as_flattened_mut());
            crate::neon_hybrid::hash_messages_raw(seconds.as_ptr() as *const *const u8, count, second_blocks, IV, 1, CHUNK_START, CHUNK_END, last_len, cvs[count..2 * count].as_flattened_mut());
        }
        for (i, pair) in pairs[..count].iter_mut().enumerate() {
            pair[..OUT_LEN].copy_from_slice(&cvs[i]);
            pair[OUT_LEN..].copy_from_slice(&cvs[count + i]);
        }
        let parents: arrayvec::ArrayVec<&[u8; BLOCK_LEN], GROUP> = pairs[..count].iter().collect();
        // Sound: as above.
        unsafe { crate::neon_hybrid::hash_many_last_len(&parents, IV, 0, IncrementCounter::No, PARENT, 0, ROOT, BLOCK_LEN, digests.as_flattened_mut()) };
    }
}

/// Whether a batch of `count` messages of `len` bytes runs SME2 kernels
/// (so takes the SME2 turn): sixteen messages or more; one-block messages
/// from ONE_BLOCK_PAD_MIN; messages of up to a chunk from SME2_TAIL_MIN;
/// messages of 2 to 15 chunks from sme2_chunked_min; and any message long
/// enough for hash() to take the turn itself.
pub(crate) fn sme2_sized(len: usize, count: usize) -> bool {
    count >= crate::SME2_SIZED_BATCH
        || (len > 0 && len <= BLOCK_LEN && count >= ONE_BLOCK_PAD_MIN)
        || (len > BLOCK_LEN && len <= CHUNK_LEN && count >= SME2_TAIL_MIN)
        || (chunked(len) && count >= sme2_chunked_min(len))
        || (len >= crate::SME2_SIZED_LEN && count >= 1)
}

/// Up to TABLE one-block messages (N = BLOCK_LEN, whole blocks) in
/// `messages`: whole SME2 groups on the message kernel, the last padded,
/// where the count calls for it; otherwise the platform's `hash_many` (the
/// parent kernels and the NEON plans). Kept out of line:
/// with all sixteen lengths inlined into hash_many_on, batches of two
/// 64-byte messages took 30% longer (VM, bench-hashes); out of line they
/// cost what they did before.
#[inline(never)]
fn hash_run<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    #[cfg(blake3_sme2)]
    if platform_is_sme2(platform) && outputs.len() % 16 >= if outputs.len() > 16 { ONE_BLOCK_PAD_AFTER_GROUPS } else { ONE_BLOCK_PAD_MIN } {
        return hash_run_padded::<N>(messages, outputs, BLOCK_LEN);
    }
    let mut table: [core::mem::MaybeUninit<&[u8; N]>; TABLE] = [core::mem::MaybeUninit::uninit(); TABLE];
    for (slot, message) in table[..outputs.len()].iter_mut().zip(messages.chunks_exact(N)) {
        slot.write(message.try_into().expect("messages of N bytes"));
    }
    // Sound: the first outputs.len() slots were written just above.
    let filled: &[&[u8; N]] = unsafe { core::slice::from_raw_parts(table.as_ptr() as *const &[u8; N], outputs.len()) };
    platform.hash_many::<N>(filled, IV, 0, IncrementCounter::No, CHUNK_START | CHUNK_END | ROOT, 0, 0, outputs.as_flattened_mut());
}

/// One-block messages on SME2, every group on the message kernel, the
/// last one padded, their last block `last_len` bytes.
#[cfg(blake3_sme2)]
#[inline(never)]
fn hash_run_padded<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], last_len: usize) {
    let mut slots: Slots<N> = [core::mem::MaybeUninit::uninit(); TABLE + 16];
    let lanes = fill_table(&mut slots, messages, outputs.len(), outputs.len().next_multiple_of(16));
    // Sound: the caller found SME2 (platform_is_sme2).
    unsafe { crate::sme2::hash_messages::<N>(lanes, IV, 0, CHUNK_START, CHUNK_END | ROOT, last_len, outputs.as_flattened_mut()) };
}

/// [`hash_run`] for messages of 1 to 63 bytes (one short block): on SME2
/// the message kernel from ONE_BLOCK_PAD_MIN messages (the parent kernel
/// records 64 for every block), else the NEON plans, or one call each
/// without them.
#[inline(never)]
fn hash_run_short<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], platform: Platform, plans: bool, last_len: usize) {
    #[cfg(blake3_sme2)]
    if platform_is_sme2(platform) && outputs.len() >= ONE_BLOCK_PAD_MIN {
        return hash_run_padded::<N>(messages, outputs, last_len);
    }
    if plans {
        let mut slots: Slots<N> = [core::mem::MaybeUninit::uninit(); TABLE + 16];
        let lanes = fill_table(&mut slots, messages, outputs.len(), outputs.len());
        neon_plans_last_len::<N>(lanes, last_len, outputs);
    } else {
        for (output, message) in outputs.iter_mut().zip(messages.chunks_exact(N)) {
            *output = *crate::hash_serial_on(&message[..last_len], IV, 0, platform).as_bytes();
        }
    }
}

/// Fewest one-block messages that SME2 hashes as whole groups on the
/// message kernel, the last one padded: 11 in a batch of fewer than
/// sixteen (fewer run on the NEON parent plans), 5 left over past whole
/// groups (fewer left over run on the NEON plans). VM, ns per message,
/// before / padded: 10 18.3 / 20.9, 11 19.1 / 15.0, 15 18.2 / 10.8.
/// Past whole groups the padded group replaces the plans and the overlap
/// group. The plans after a group run fast while the SME unit stays in
/// its fast state and 2.5x slower when it sits in its slow one; the
/// padded group runs at one speed. Mac M4 Max P-cores, ns per message,
/// left-overs from 13 / from 5: in a sweep where the unit sat slow at
/// 20 and 22-28 messages (probe onepad-after5, jobs 279-281), 22-28
/// 25.5-31.5 / 11.4-15.7 and 53 18.4 / 12.2; slower at 21 (13.6 / 15.6)
/// and 37-40 (10.1-11.8 / 11.9-13.0). In the benchmark, every sample
/// fast (jobs 291-294), 24 10.6 / 13.3 (official 21.0-21.7). The VM, in
/// the plans' fast state, 22, 24, 26 12-16% slower (their slow state is
/// twice as slow). The worst case halves; the fast case pays up to 25%.
pub(crate) const ONE_BLOCK_PAD_MIN: usize = 11;
#[cfg_attr(not(blake3_sme2), allow(dead_code))]
pub(crate) const ONE_BLOCK_PAD_AFTER_GROUPS: usize = 5;

/// Room for a pointer table: TABLE messages and a padded group's spare
/// lanes, uninitialised on the caller's stack.
type Slots<'a, const N: usize> = [core::mem::MaybeUninit<&'a [u8; N]>; TABLE + 16];

/// Fill `slots` with pointers to the first `count` slots of N bytes in
/// `messages`, then spare lanes repeating the last up to `lanes`; the
/// table's filled part.
#[inline(always)]
fn fill_table<'a, 's, const N: usize>(slots: &'s mut Slots<'a, N>, messages: &'a [u8], count: usize, lanes: usize) -> &'s [&'a [u8; N]] {
    assert!(0 < count && count <= lanes && lanes <= TABLE + 16 && messages.len() >= count * N, "a table of 1 to TABLE + 16 lanes over the messages");
    let mut last: &[u8; N] = &[0; N];
    for (slot, message) in slots[..count].iter_mut().zip(messages.chunks_exact(N)) {
        last = message.try_into().expect("slots of N bytes");
        slot.write(last);
    }
    for slot in &mut slots[count..lanes] {
        slot.write(last);
    }
    // Sound: the first `lanes` slots were written just above.
    unsafe { core::slice::from_raw_parts(slots.as_ptr() as *const &[u8; N], lanes) }
}

/// The integer + NEON parent plans over `lanes`, each message's last block
/// `last_len` bytes.
#[inline(always)]
fn neon_plans_last_len<const N: usize>(lanes: &[&[u8; N]], last_len: usize, outputs: &mut [[u8; OUT_LEN]]) {
    #[cfg(blake3_neon_hybrid)]
    // Sound: the caller found the SHA-3 extension (neon_plans).
    unsafe {
        crate::neon_hybrid::hash_many_last_len(lanes, IV, 0, IncrementCounter::No, 0, CHUNK_START, CHUNK_END | ROOT, last_len, outputs.as_flattened_mut())
    };
    #[cfg(not(blake3_neon_hybrid))]
    {
        let _ = (lanes, last_len, outputs);
        unreachable!("the NEON plans exist on AArch64 alone");
    }
}

/// [`hash_run`] for messages of 2 to 16 blocks (N), the last block
/// `last_len` bytes, on one of four plans:
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
/// - The C NEON kernel (`plans` false: no SHA-3 extension), whole last
///   blocks: four at a time, a fourth, spare lane for three left over, the
///   integer kernel for one or two; with a short last block, one call each.
#[inline(never)]
fn hash_blocks<const N: usize>(messages: &[u8], outputs: &mut [[u8; OUT_LEN]], platform: Platform, plans: bool, last_len: usize) {
    const GROUP: usize = 16;
    let count = outputs.len();
    if count == 0 {
        return;
    }
    let sme2 = platform_is_sme2(platform);
    let left = count % GROUP;
    let padded = sme2 && left >= if count > GROUP { SME2_TAIL_MIN_AFTER_GROUPS } else { SME2_TAIL_MIN };
    // Messages hashed on SME2 (the first `on_sme2`), and the table's lanes.
    let on_sme2 = if padded { count } else if sme2 { count - left } else { 0 };
    let lanes = if padded { count.next_multiple_of(GROUP) } else { count };
    let mut slots: Slots<N> = [core::mem::MaybeUninit::uninit(); TABLE + 16];
    let table = fill_table(&mut slots, messages, count, lanes);
    let (groups, rest) = outputs.split_at_mut(on_sme2);
    if !rest.is_empty() {
        let rest_lanes = &table[on_sme2..count];
        if plans {
            neon_plans_last_len::<N>(rest_lanes, last_len, rest);
        } else if last_len == BLOCK_LEN {
            hash_four_at_a_time::<N>(rest_lanes, rest, platform);
        } else {
            for (output, message) in rest.iter_mut().zip(rest_lanes) {
                *output = *crate::hash_serial_on(&message[..(N - BLOCK_LEN) + last_len], IV, 0, platform).as_bytes();
            }
        }
    }
    #[cfg(blake3_sme2)]
    if on_sme2 > 0 {
        // Sound: platform_is_sme2 means detect() found SME2 with 512-bit
        // streaming vectors.
        let sme2_lanes = &table[..if padded { lanes } else { on_sme2 }];
        unsafe { crate::sme2::hash_messages::<N>(sme2_lanes, IV, 0, CHUNK_START, CHUNK_END | ROOT, last_len, groups.as_flattened_mut()) };
    }
    #[cfg(not(blake3_sme2))]
    let _ = groups;
}

/// The C NEON kernel's arrangement for messages of whole blocks: groups of
/// four through the platform's `hash_many`, a fourth, spare lane for three
/// left over (its value kept in a scratch array), the integer kernel for
/// one or two (256 B, VM: 2 messages took 7% longer than a loop of hash()
/// on the kernel, which hashes a remainder in portable code).
#[inline(never)]
fn hash_four_at_a_time<const N: usize>(lanes: &[&[u8; N]], outputs: &mut [[u8; OUT_LEN]], platform: Platform) {
    let (flags, start, end) = (0, CHUNK_START, CHUNK_END | ROOT);
    let fours = lanes.len() / 4 * 4;
    platform.hash_many::<N>(&lanes[..fours], IV, 0, IncrementCounter::No, flags, start, end, outputs[..fours].as_flattened_mut());
    match lanes.len() - fours {
        3 => {
            let spare = [lanes[fours], lanes[fours + 1], lanes[fours + 2], lanes[fours + 2]];
            let mut digests = [[0u8; OUT_LEN]; 4];
            platform.hash_many::<N>(&spare, IV, 0, IncrementCounter::No, flags, start, end, digests.as_flattened_mut());
            outputs[fours..].copy_from_slice(&digests[..3]);
        }
        _ => {
            for (output, message) in outputs[fours..].iter_mut().zip(&lanes[fours..]) {
                *output = *crate::hash_serial_on(&message[..], IV, 0, platform).as_bytes();
            }
        }
    }
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

    /// `count` deterministic messages of `len` bytes, each in its slot
    /// (slot_len), zero past its end: little-endian 64-bit words `len << 48
    /// | index`, so every block differs and a kernel that mixed up its lanes
    /// would be caught.
    fn messages(len: usize, count: usize) -> Vec<u8> {
        let slot = slot_len(len);
        let mut bytes: Vec<u8> = (0..(slot * count).div_ceil(8) as u64).flat_map(|i| ((len as u64) << 48 | i).to_le_bytes()).collect();
        bytes.truncate(slot * count);
        for message in bytes.chunks_exact_mut(slot) {
            message[len..].fill(0);
        }
        bytes
    }

    /// Message `i` of a batch of `len`-byte messages.
    fn message(input: &[u8], len: usize, i: usize) -> &[u8] {
        &input[i * slot_len(len)..][..len]
    }

    /// hash_many (and hash_many_multithreaded) of `count` messages of
    /// `len` bytes against hash() one message at a time.
    fn check(len: usize, count: usize) {
        let input = messages(len, count);
        let mut out = vec![[0u8; OUT_LEN]; count];
        crate::hash_many(&input, len, &mut out);
        for (i, digest) in out.iter().enumerate() {
            assert_eq!(*digest, *crate::hash(message(&input, len, i)).as_bytes(), "message {i} of {count}, {len} bytes");
        }
        #[cfg(feature = "std")]
        {
            let mut mt = vec![[0u8; OUT_LEN]; count];
            crate::hash_many_multithreaded(&input, len, &mut mt);
            assert_eq!(mt, out, "multithreaded, {count} messages of {len} bytes");
        }
    }

    /// One-block messages: every NEON parent plan (1 to 16), then SME2
    /// groups and their remainders (5 to 15 left over take a padded
    /// group), TABLE's edges, and the pool's split.
    #[test]
    fn test_hash_many_blocks() {
        for count in (0..=17).chain([20, 21, 24, 29, 30, 31, 32, 33, 45, 61, 127, 128, 129, 205, 1021, 1022, 1023, 1024, 1025, 2049]) {
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
        for blocks in [1, 2, 3, 4, 7, 16] {
            let len = blocks * BLOCK_LEN;
            for count in (0..=40).chain([122, 123, 127, 128, 129, 131, 133, 134, 143, 144, 150]) {
                let input = messages(len, count);
                for &platform in &platforms {
                    // Sixteen sentinels past the end catch a spare lane's store.
                    let mut out = vec![[0xAAu8; OUT_LEN]; count + 16];
                    hash_many_on(&input, len, &mut out[..count], platform);
                    assert!(out[count..].iter().all(|d| *d == [0xAA; OUT_LEN]), "{platform:?}, {count} x {len} B: a store past the last output");
                    for (i, digest) in out[..count].iter().enumerate() {
                        assert_eq!(*digest, *crate::hash(message(&input, len, i)).as_bytes(), "{platform:?}, message {i} of {count}, {len} bytes");
                    }
                }
            }
        }
    }

    /// Lengths other than whole blocks, from an empty message to past 15
    /// chunks, in their slots.
    #[test]
    fn test_hash_many_other_lengths() {
        for len in [0, 1, 2, 63, 65, 100, 127, 129, 191, 255, 257, 1000, 1023, 1025, 2000, 3000, 3 * CHUNK_LEN + 7, 15 * CHUNK_LEN - 1, 15 * CHUNK_LEN + 1, 16 * CHUNK_LEN + 5] {
            for count in [0, 1, 2, 3, 5, 9, 10, 15, 16, 17, 21, 129] {
                check(len, count);
            }
        }
    }

    /// Every length from 0 to 3000 bytes at a few counts, and every last
    /// block length at counts that reach each kernel, on every platform
    /// this CPU has.
    #[test]
    fn test_hash_many_every_length() {
        #[allow(unused_mut)]
        let mut platforms = vec![Platform::detect(), Platform::Portable];
        #[cfg(blake3_neon)]
        platforms.push(Platform::neon().expect("NEON on AArch64"));
        for len in 0..=3000 {
            let counts: &[usize] = if len % 97 == 0 || len <= 130 || (1020..=1030).contains(&len) { &[2, 3, 7, 10, 16, 21, 33] } else { &[3, 16] };
            for &count in counts {
                let input = messages(len, count);
                for &platform in &platforms {
                    let mut out = vec![[0xAAu8; OUT_LEN]; count + 16];
                    hash_many_on(&input, len, &mut out[..count], platform);
                    assert!(out[count..].iter().all(|d| *d == [0xAA; OUT_LEN]), "{platform:?}, {count} x {len} B: a store past the last output");
                    for (i, digest) in out[..count].iter().enumerate() {
                        assert_eq!(*digest, *crate::test::reference_hash(message(&input, len, i)).as_bytes(), "{platform:?}, message {i} of {count}, {len} bytes");
                    }
                }
            }
        }
    }

    /// The C NEON kernel's arrangement (the one CPUs without the SHA-3
    /// extension run: groups of four, a spare fourth lane for three left
    /// over, the integer kernel for one or two), on NEON and SME2.
    #[cfg(blake3_neon_hybrid)]
    #[test]
    fn test_hash_blocks_without_plans() {
        #[allow(unused_mut)]
        let mut platforms = vec![Platform::neon().expect("NEON on AArch64")];
        #[cfg(blake3_sme2)]
        platforms.extend(Platform::sme2());
        for count in 0..=40 {
            let input = messages(4 * BLOCK_LEN, count);
            for &platform in &platforms {
                let mut out = vec![[0xAAu8; OUT_LEN]; count + 16];
                hash_blocks::<{ 4 * BLOCK_LEN }>(&input, &mut out[..count], platform, false, BLOCK_LEN);
                assert!(out[count..].iter().all(|d| *d == [0xAA; OUT_LEN]), "{platform:?}, {count}: a store past the last output");
                for (i, digest) in out[..count].iter().enumerate() {
                    assert_eq!(*digest, *crate::hash(message(&input, 256, i)).as_bytes(), "{platform:?}, message {i} of {count}");
                }
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
                        assert_eq!(*digest, *crate::hash(message(&input, len, i)).as_bytes(), "{platform:?}, message {i} of {count}, {len} bytes");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "input holds one slot of whole blocks per output")]
    fn test_hash_many_needs_the_input_it_names() {
        crate::hash_many(&[0u8; 100], 64, &mut [[0u8; OUT_LEN]; 2]);
    }

    #[test]
    #[should_panic(expected = "input holds one slot of whole blocks per output")]
    fn test_hash_many_needs_whole_slots() {
        crate::hash_many(&[0u8; 200], 100, &mut [[0u8; OUT_LEN]; 2]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "every slot's bytes past its message are zero")]
    fn test_hash_many_needs_zero_padding() {
        let mut input = [0u8; 256];
        input[120] = 1;
        crate::hash_many(&input, 100, &mut [[0u8; OUT_LEN]; 2]);
    }
}

/// Proofs (Kani, `cargo kani`; QUALITY.md says how to run them).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// A message's slot is whole blocks, at least one, and holds the
    /// message with less than a block to spare: for every length a slice
    /// can have.
    #[kani::proof]
    fn slot_len_is_whole_blocks() {
        let len: usize = kani::any();
        kani::assume(len <= isize::MAX as usize);
        let slot = slot_len(len);
        assert!(slot % BLOCK_LEN == 0 && slot >= BLOCK_LEN && slot >= len);
        assert!(len == 0 || slot - len < BLOCK_LEN);
    }

    /// fill_table's unsafe slice covers only written slots: the first
    /// `count` point at the messages in order, the rest repeat the last,
    /// for every count and lane count it accepts (one-block messages).
    #[kani::proof]
    #[kani::unwind(22)]
    #[kani::solver(cadical)]
    fn fill_table_writes_every_lane() {
        const MAX: usize = TABLE + 16;
        let messages = [0u8; MAX * BLOCK_LEN];
        let count: usize = kani::any();
        let lanes: usize = kani::any();
        kani::assume(0 < count && count <= lanes && lanes <= 20);
        let mut slots: Slots<BLOCK_LEN> = [core::mem::MaybeUninit::uninit(); MAX];
        let table = fill_table(&mut slots, &messages, count, lanes);
        assert!(table.len() == lanes);
        let i: usize = kani::any();
        kani::assume(i < lanes);
        let expected = i.min(count - 1) * BLOCK_LEN;
        assert!(core::ptr::eq(table[i].as_ptr(), messages[expected..].as_ptr()));
    }
}
