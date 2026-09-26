//! The startup self-test: before a process first hashes, one pass over
//! inputs chosen so that each goes through a different kernel and code
//! path on this machine, every result compared with the reference
//! implementation's (checked in below, verified by a unit test). A
//! miscompiled kernel, a faulty vector or matrix unit, or a broken build
//! then stops the program with a message naming the path, instead of
//! producing wrong digests. It takes about 100 microseconds, once.
//!
//! As Niels Ferguson suggested, the cases chain: each writes the first 32
//! bytes of its output over the start of the next case's input (and keys
//! the keyed case), so a fault anywhere changes every value after it; and
//! they use unusual alignments, lengths, and padding, where platform,
//! compiler, and hardware faults hide from ordinary tests. The first case
//! that differs names the failing path: the cases before it agreed.
//!
//! Platform::detect(), which every entry point calls, runs it through
//! `ensure`. The test calls the public entry points, so it re-enters
//! detect(): a thread-local flag lets its own calls through, and other
//! threads wait for it (std::sync::Once). Builds without std, and Miri,
//! skip it.

/// One case: an input shape and the entry point that hashes it. Every
/// case reads its input from the start of the shared buffer, `offset`
/// bytes in where it has one.
#[derive(Clone, Copy)]
pub(crate) enum Case {
    /// hash() of `len` bytes starting `offset` bytes into the input.
    Hash { offset: usize, len: usize },
    /// hash_many() of `count` messages of `len` bytes, in slots of whole
    /// blocks, zero between messages, the batch's base `offset` bytes past
    /// an allocation's start.
    Many { offset: usize, len: usize, count: usize },
    /// The chaining values of `chunks` whole chunks at counter 1, keyed by
    /// the previous case's output: on SME2, one call of the kernel with
    /// integer lanes beside the matrix unit (18 chunks a group), which
    /// hash() reaches only from 256 KiB; elsewhere the platform's chunk
    /// kernels, with the same values.
    ChunkValues { chunks: usize },
    /// keyed_hash() of `len` bytes, keyed by the previous case's output.
    Keyed { len: usize },
    /// derive_key() over `len` bytes.
    Derive { len: usize },
    /// A Hasher fed `len` bytes in pieces of `piece` bytes.
    Incremental { len: usize, piece: usize },
    /// `out` bytes of extended output from hashing `len` bytes, read
    /// from output position `skip`.
    Xof { len: usize, skip: u64, out: usize },
}

use Case::*;

const KIB: usize = 1024;

/// The cases, each with the path it is there for, assembly first (on
/// Apple M4 and later; other machines take their own paths through the
/// same entry points). Every assembly entry of the AArch64 kernels runs:
/// the hybrids' k1-k10, q1-q9, p2-p9, and c1, and SME2's chunk, message,
/// parent, and integer-lane kernels (gdb breakpoints, NOTES-servil.md).
pub(crate) const CASES: &[(Case, &str)] = &[
    (Hash { offset: 0, len: 0 }, "the empty input"),
    (Hash { offset: 1, len: 63 }, "one partial block, unaligned"),
    (Hash { offset: 3, len: 65 }, "a block and a byte, unaligned"),
    (Hash { offset: 5, len: KIB }, "one whole chunk, unaligned"),
    (Hash { offset: 0, len: KIB + 300 }, "a chunk and a partial one: q1"),
    (Hash { offset: 7, len: 2 * KIB }, "two chunks, unaligned: k2"),
    (Derive { len: 3 * KIB - 1 }, "key derivation, two chunks and a partial one: q2"),
    (Hash { offset: 0, len: 3 * KIB + 100 }, "three chunks and a partial one: q3"),
    (Hash { offset: 1, len: 4 * KIB + 65 }, "four chunks and a partial one, unaligned: q4"),
    (Hash { offset: 0, len: 5 * KIB + 1000 }, "five chunks and a partial one: q5"),
    (Hash { offset: 3, len: 6 * KIB + 129 }, "six chunks and a partial one, unaligned: q6"),
    (Hash { offset: 0, len: 8 * KIB - 1 }, "seven chunks and a partial one: q7"),
    (Hash { offset: 5, len: 8 * KIB + 200 }, "eight chunks and a partial one, unaligned: q8"),
    (Hash { offset: 0, len: 10 * KIB }, "ten chunks: k10"),
    (Hash { offset: 1, len: 16 * KIB - 1 }, "fifteen chunks and a partial one, unaligned: k6 and q9"),
    (Hash { offset: 0, len: 4 * KIB }, "four chunks: k4"),
    (Hash { offset: 3, len: 5 * KIB }, "five chunks, unaligned: k5"),
    (Hash { offset: 0, len: 7 * KIB }, "seven chunks: k7"),
    (Hash { offset: 1, len: 8 * KIB }, "eight chunks, unaligned: k8"),
    (Hash { offset: 0, len: 9 * KIB }, "nine chunks: k9"),
    (Many { offset: 0, len: KIB, count: 10 }, "ten one-chunk messages: the SME2 message kernel"),
    (Hash { offset: 0, len: 16 * KIB }, "one SME2 group of sixteen chunks"),
    (Hash { offset: 7, len: 19 * KIB + 5 }, "an SME2 group, a remainder on NEON, unaligned: k3"),
    (Hash { offset: 0, len: 32 * KIB }, "two SME2 groups: the SME2 parent kernel"),
    (ChunkValues { chunks: 18 }, "keyed chaining values: the SME2 kernel with integer lanes"),
    (Many { offset: 1, len: 64, count: 2 }, "a pair of 64-byte messages: the parent plans"),
    (Many { offset: 0, len: 64, count: 11 }, "eleven 64-byte messages: a padded SME2 group"),
    (Many { offset: 3, len: 64, count: 21 }, "an SME2 group and five more in a padded group"),
    (Many { offset: 0, len: 1, count: 12 }, "one-byte messages, each in a block of padding"),
    (Many { offset: 7, len: 100, count: 13 }, "messages of 100 bytes: partial blocks in slots"),
    (Many { offset: 0, len: 256, count: 3 }, "three 256-byte messages on the parent plans"),
    (Many { offset: 1, len: 128, count: 9 }, "nine two-block messages: p9"),
    (Many { offset: 1, len: 255, count: 17 }, "an SME2 group of 255-byte messages and one more"),
    (Many { offset: 5, len: 2000, count: 3 }, "two-chunk messages side by side on NEON"),
    (Many { offset: 0, len: 1100, count: 7 }, "two-chunk messages side by side on SME2"),
    (Keyed { len: 2 * KIB + 1 }, "keyed hashing"),
    (Incremental { len: 5 * KIB + 7, piece: 700 }, "a Hasher fed in uneven pieces"),
    (Incremental { len: 2 * KIB, piece: 63 }, "a Hasher fed less than a block at a time"),
    (Xof { len: 100, skip: 37, out: 300 }, "extended output of several blocks, from an unaligned position"),
];

/// The largest input any case reads, offset included.
pub(crate) const INPUT_LEN: usize = 32 * KIB + 8;

/// Fill `input` with the starting bytes: little-endian words i * 0x9E3779B97F4A7C15.
pub(crate) fn fill(input: &mut [u8]) {
    for (i, word) in input.chunks_mut(8).enumerate() {
        let bytes = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes();
        word.copy_from_slice(&bytes[..word.len()]);
    }
}

/// The context derive_key uses.
pub(crate) const CONTEXT: &str = "blake3-servil self-test";

/// A case's outputs as one 64-bit word: FNV-1a over their bytes. It needs
/// no hash of its own; a wrong output changes it but for a 2^-64 chance.
pub(crate) fn fold(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// Run the cases in order, chained: before each, the first 32 bytes of the
/// previous case's output (zeros before the first) overwrite the start of
/// its input (of every message, in a batch), as much as it reads, and are
/// the key a keyed case takes; so every output depends on every earlier
/// one. `outputs` hashes one case; `check` sees each case's fold as it
/// comes.
#[cfg(feature = "std")]
pub(crate) fn chain(
    mut outputs: impl FnMut(Case, &[u8], &[u8; 32]) -> std::vec::Vec<u8>,
    mut check: impl FnMut(usize, u64),
) {
    let mut input = std::vec![0u8; INPUT_LEN];
    fill(&mut input);
    let mut previous = [0u8; 32];
    for (index, &(case, _)) in CASES.iter().enumerate() {
        let (start, len, count) = match case {
            Hash { offset, len } => (offset, len, 1),
            Many { len, count, .. } => (0, len, count),
            ChunkValues { chunks } => (0, chunks * crate::CHUNK_LEN, 1),
            Keyed { len } | Derive { len } | Incremental { len, .. } | Xof { len, .. } => (0, len, 1),
        };
        let n = len.min(32);
        for message in 0..count {
            input[start + message * len..][..n].copy_from_slice(&previous[..n]);
        }
        let output = outputs(case, &input, &previous);
        check(index, fold(&output));
        previous.copy_from_slice(&output[..32]);
    }
}

/// Each case's fold, from the reference implementation (the unit test
/// `self_test_matches_the_reference` checks them, and prints this table
/// when it differs).
const EXPECTED: [u64; 39] = [
    0x90e4e563714f7c48,
    0x80003c83df0cf1a6,
    0xf4ffcff9e99d31ba,
    0xb4bb0ea2c72a1ab1,
    0xa7903018757d10ab,
    0xc9d2c00ed1c20580,
    0xed99f187435619b2,
    0xf0bb0553f78db276,
    0xaf5ff9ff4790a2fb,
    0xab47f312598807b6,
    0xa20d5c093d025c29,
    0xddefca2f181fba05,
    0xb50d992295f65767,
    0x03f7aba2333521ce,
    0xcf1b84d7e1e8f634,
    0xae24028e1625d272,
    0x1182e5688548eec5,
    0xfaede175c68c8a44,
    0x9d9dd93e18400fcd,
    0x726458b7818bd991,
    0x7e5a46052c19e913,
    0xb2d9e0c0afd2bd13,
    0x85b92449ae3bb6d6,
    0xd306834db3ef623a,
    0xfcab2c0c730749f0,
    0x9e8638ebeff4b28a,
    0x3fb12d675ce0ef8d,
    0x65a0d42d9d58e25b,
    0x6456a5098c002235,
    0x9af343b2ecd71d2b,
    0x0e5cb7bf68527612,
    0x0243bf5c211ea02c,
    0x2da3874c1806f4dc,
    0xbc178d10e6d9c2bd,
    0x366a22ceb651ca6a,
    0x53f113d67d7ac3e5,
    0x9b802b34b48bb23a,
    0x0eaabf3bcc0790e7,
    0x7bf131603d28d0e7,
];

/// A case's outputs from this build, through the public entry points.
#[cfg(feature = "std")]
pub(crate) fn outputs(case: Case, input: &[u8], previous: &[u8; 32]) -> std::vec::Vec<u8> {
    use std::vec;
    match case {
        Hash { offset, len } => crate::hash(&input[offset..][..len]).as_bytes().to_vec(),
        Many { offset, len, count } => {
            let slot = crate::many::slot_len(len);
            let mut buffer = vec![0u8; offset + slot * count];
            let messages = &mut buffer[offset..];
            for (i, message) in messages.chunks_mut(slot).enumerate() {
                message[..len].copy_from_slice(&input[i * len..][..len]);
            }
            let mut digests = vec![[0u8; crate::OUT_LEN]; count];
            crate::hash_many(messages, len, &mut digests);
            digests.concat()
        }
        ChunkValues { chunks } => chunk_values(&input[..chunks * crate::CHUNK_LEN], previous),
        Keyed { len } => crate::keyed_hash(previous, &input[..len]).as_bytes().to_vec(),
        Derive { len } => crate::derive_key(CONTEXT, &input[..len]).to_vec(),
        Incremental { len, piece } => {
            let mut hasher = crate::Hasher::new();
            for part in input[..len].chunks(piece) {
                hasher.update(part);
            }
            hasher.finalize().as_bytes().to_vec()
        }
        Xof { len, skip, out } => {
            let mut bytes = vec![0u8; out];
            let mut reader = crate::Hasher::new().update(&input[..len]).finalize_xof();
            reader.set_position(skip);
            reader.fill(&mut bytes);
            bytes
        }
    }
}

/// The chaining values of the whole chunks in `input`, at counter 1 on,
/// keyed by `key` (see Case::ChunkValues).
#[cfg(feature = "std")]
fn chunk_values(input: &[u8], key: &[u8; 32]) -> std::vec::Vec<u8> {
    let key = crate::platform::words_from_le_bytes_32(key);
    let chunks: std::vec::Vec<&[u8; crate::CHUNK_LEN]> =
        input.chunks_exact(crate::CHUNK_LEN).map(|c| c.try_into().expect("whole chunks")).collect();
    let mut out = std::vec![0u8; chunks.len() * crate::OUT_LEN];
    #[cfg(blake3_sme2)]
    if chunks.len() % crate::sme2::HYBRID_GROUP == 0 {
        let turn = crate::platform::Sme2Turn::take(crate::platform::Platform::detect_unchecked(), true);
        if !matches!(turn.platform(), crate::platform::Platform::SME2) {
            return platform_chunk_values(&chunks, &key, out);
        }
        let flags = crate::KEYED_HASH as u32 | (crate::CHUNK_START as u32) << 8 | (crate::CHUNK_END as u32) << 16;
        // Sound: SME2 was detected, and the table holds 18 chunk pointers per group.
        let lanes = unsafe {
            crate::sme2::ffi::blake3_sme2x2_hash_chunks_512(
                chunks.as_ptr() as *const *const u8,
                key.as_ptr(),
                1,
                flags,
                out.as_mut_ptr(),
                (chunks.len() / crate::sme2::HYBRID_GROUP) as u64,
            )
        };
        assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        return out;
    }
    platform_chunk_values(&chunks, &key, out)
}

/// [`chunk_values`] through the platform's own chunk kernels.
#[cfg(feature = "std")]
fn platform_chunk_values(chunks: &[&[u8; crate::CHUNK_LEN]], key: &crate::CVWords, mut out: std::vec::Vec<u8>) -> std::vec::Vec<u8> {
    crate::platform::Platform::detect_unchecked().hash_many(
        chunks,
        key,
        1,
        crate::IncrementCounter::Yes,
        crate::KEYED_HASH,
        crate::CHUNK_START,
        crate::CHUNK_END,
        &mut out,
    );
    out
}

/// Run every case and panic on the first that differs from the reference.
#[cfg(feature = "std")]
fn run() {
    chain(outputs, |index, folded| {
        assert!(
            folded == EXPECTED[index],
            "blake3-servil startup self-test failed: case {index}, {} (platform {}): this build or this CPU \
             computes wrong BLAKE3 digests on that path",
            CASES[index].1,
            crate::platform::Platform::detect_unchecked().name(),
        );
    });
}

#[cfg(all(feature = "std", not(miri)))]
std::thread_local! {
    static RUNNING: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[cfg(all(feature = "std", not(miri)))]
static PASSED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Run the self-test unless it has passed: once per process, the first
/// caller running it and any other waiting. It publishes no data, so a
/// relaxed load suffices on the fast path.
#[inline]
pub(crate) fn ensure() {
    #[cfg(all(feature = "std", not(miri)))]
    if !PASSED.load(core::sync::atomic::Ordering::Relaxed) {
        ensure_slow();
    }
}

#[cfg(all(feature = "std", not(miri)))]
#[cold]
#[inline(never)]
fn ensure_slow() {
    if RUNNING.with(|running| running.get()) {
        return; // the self-test's own calls
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        RUNNING.with(|running| running.set(true));
        run();
        RUNNING.with(|running| running.set(false));
        PASSED.store(true, core::sync::atomic::Ordering::Relaxed);
    });
}

/// The self-test's own time, for probes: run it again (it has passed).
#[cfg(all(feature = "std", not(miri)))]
pub(crate) fn run_again() {
    ensure();
    RUNNING.with(|running| running.set(true));
    run();
    RUNNING.with(|running| running.set(false));
}

#[cfg(test)]
mod test {
    use super::*;

    /// A case's outputs from the reference implementation.
    fn reference(case: Case, input: &[u8], previous: &[u8; 32]) -> std::vec::Vec<u8> {
        let digest = |mut hasher: reference_impl::Hasher, bytes: &[u8], skip: usize, out: usize| {
            hasher.update(bytes);
            let mut output = std::vec![0u8; skip + out];
            hasher.finalize(&mut output);
            output.split_off(skip)
        };
        match case {
            Hash { offset, len } => digest(reference_impl::Hasher::new(), &input[offset..][..len], 0, 32),
            Many { len, count, .. } => {
                (0..count).flat_map(|i| digest(reference_impl::Hasher::new(), &input[i * len..][..len], 0, 32)).collect()
            }
            ChunkValues { chunks } => {
                // The portable Rust kernel, which shares no code with the assembly.
                let key = crate::platform::words_from_le_bytes_32(previous);
                let table: std::vec::Vec<&[u8; crate::CHUNK_LEN]> =
                    input[..chunks * crate::CHUNK_LEN].chunks_exact(crate::CHUNK_LEN).map(|c| c.try_into().unwrap()).collect();
                let mut out = std::vec![0u8; chunks * crate::OUT_LEN];
                crate::portable::hash_many(
                    &table,
                    &key,
                    1,
                    crate::IncrementCounter::Yes,
                    crate::KEYED_HASH,
                    crate::CHUNK_START,
                    crate::CHUNK_END,
                    &mut out,
                );
                out
            }
            Keyed { len } => digest(reference_impl::Hasher::new_keyed(previous), &input[..len], 0, 32),
            Derive { len } => digest(reference_impl::Hasher::new_derive_key(CONTEXT), &input[..len], 0, 32),
            Incremental { len, .. } => digest(reference_impl::Hasher::new(), &input[..len], 0, 32),
            Xof { len, skip, out } => digest(reference_impl::Hasher::new(), &input[..len], skip as usize, out),
        }
    }

    /// EXPECTED is the reference implementation's answer for every case.
    /// On a difference the test prints the table the reference gives;
    /// pasting it in is a reviewed change.
    #[test]
    fn self_test_matches_the_reference() {
        let mut folds = std::vec::Vec::new();
        chain(reference, |_, folded| folds.push(folded));
        let table: std::string::String = folds.iter().map(|f| std::format!("    0x{f:016x},\n")).collect();
        assert!(folds == EXPECTED, "EXPECTED differs from the reference implementation; its table:\n{table}");
    }

    /// This build agrees with the reference on every case (as the startup
    /// run, which already passed for this test process, found).
    #[test]
    fn self_test_passes() {
        chain(outputs, |index, folded| assert_eq!(folded, EXPECTED[index], "case {index}, {}", CASES[index].1));
    }

    /// Chaining carries a difference forward: a wrong first output changes
    /// every fold after it.
    #[test]
    fn self_test_chain_carries_a_fault_forward() {
        let mut good = std::vec::Vec::new();
        chain(reference, |_, folded| good.push(folded));
        let mut faulty = std::vec::Vec::new();
        let mut first = true;
        chain(
            |case, input, previous| {
                let mut output = reference(case, input, previous);
                if std::mem::take(&mut first) {
                    output[31] ^= 1;
                }
                output
            },
            |_, folded| faulty.push(folded),
        );
        let same: std::vec::Vec<usize> = (0..good.len()).filter(|&i| good[i] == faulty[i]).collect();
        assert!(same.is_empty(), "every fold changes after a fault in the first case; unchanged: {same:?}");
    }
}
