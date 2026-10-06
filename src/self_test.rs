//! The startup self-test: before a process first hashes, one pass over
//! inputs chosen so that each goes through a different kernel and code
//! path on this machine, every result compared with the reference
//! implementation's (checked in below, verified by a unit test). A
//! miscompiled kernel, a faulty vector or matrix unit, or a broken build
//! then stops the program with a message naming the path, instead of
//! producing wrong digests. It takes 0.1 to 0.2 ms, once (Apple M4 Max:
//! 93 µs warm, 140-200 µs in a fresh process; the VM: 95 and 125): most
//! of it is running every assembly kernel, 390 KB of unrolled code, whose
//! first fetch dominates the fresh-process cost (Zooko chose this budget
//! over a smaller one that leaves kernels out, September 26, 2026).
//!
//! As Niels Ferguson suggested, the cases chain: each writes the first 32
//! bytes of its output over the start of the next case's input (and keys
//! the keyed case), so a fault anywhere changes every value after it; and
//! they use unusual alignments, lengths, and padding, where platform,
//! compiler, and hardware faults hide from ordinary tests. The first case
//! that differs names the failing path: the cases before it agreed.
//!
//! Platform::detect(), which every entry point calls, runs it through
//! `ensure`, in every build (std or not); other threads wait for it,
//! spinning (once per process, for 0.1 to 0.2 ms). The cases call the
//! entry points' bodies from the undetected platform (`hash_serial_from`
//! and its like), just past detect(), so the test never re-enters its own
//! check. It allocates nothing: its buffers (about 44 KiB) live on the
//! calling thread's stack while it runs. Miri skips it (too slow there).

// Under Miri nothing runs the cases.
#![cfg_attr(miri, allow(dead_code))]

use crate::platform::Platform;

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
    /// hash() of `len` bytes on the compact path, which calls after a
    /// pause take (hash() itself where the build has none).
    Compact { len: usize },
}

use Case::*;

const KIB: usize = 1024;

/// The cases, each with the path it is there for, assembly first (on
/// Apple M4 and later; other machines take their own paths through the
/// same entry points). Every assembly entry of the AArch64 kernels runs:
/// the hybrids' k1-k10, q1-q9, p2-p9, and c1, and SME2's chunk, message,
/// parent, and integer-lane kernels (tools/self_test_coverage.py counts
/// them with gdb breakpoints). The hybrids' k kernels hash whole chunks at
/// rising counters, so only whole inputs reach them; batches of one-chunk
/// messages, all at counter 0, take the parent plans (p).
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
    (Xof { len: 1500, skip: 64, out: 1100 }, "extended output of sixteen blocks and more (the SME2 extended-output kernel)"),
    (Compact { len: 7 * KIB + 100 }, "the compact path after a pause: whole chunks, a partial one, an odd level"),
];

/// The largest input any case reads, offset included.
pub(crate) const INPUT_LEN: usize = 32 * KIB + 8;

/// The most output any case writes (the second extended output's).
pub(crate) const OUTPUT_LEN: usize = 1100;

/// The largest batch a Many case lays out, offset included (ten one-chunk
/// messages).
const BATCH_LEN: usize = 10 * KIB;

/// The most messages a Many case holds.
const BATCH_COUNT: usize = 21;

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
/// one. `outputs` hashes one case into its buffer and returns how many
/// bytes it wrote; `check` sees each case's fold as it comes.
pub(crate) fn chain(
    mut outputs: impl FnMut(Case, &[u8], &[u8; 32], &mut [u8]) -> usize,
    mut check: impl FnMut(usize, u64),
) {
    let mut input = [0u8; INPUT_LEN];
    fill(&mut input);
    let mut out = [0u8; OUTPUT_LEN];
    let mut previous = [0u8; 32];
    for (index, &(case, _)) in CASES.iter().enumerate() {
        let (start, len, count) = match case {
            Hash { offset, len } => (offset, len, 1),
            Many { len, count, .. } => (0, len, count),
            ChunkValues { chunks } => (0, chunks * crate::CHUNK_LEN, 1),
            Keyed { len } | Derive { len } | Incremental { len, .. } | Xof { len, .. } | Compact { len } => (0, len, 1),
        };
        let n = len.min(32);
        for message in 0..count {
            input[start + message * len..][..n].copy_from_slice(&previous[..n]);
        }
        let written = outputs(case, &input, &previous, &mut out);
        let output = &out[..written];
        check(index, fold(output));
        previous.copy_from_slice(&output[..32]);
    }
}

/// Each case's fold, from the reference implementation (the unit test
/// `self_test_matches_the_reference` checks them, and prints this table
/// when it differs).
const EXPECTED: [u64; 41] = [
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
    0xa7e6fd013dbeb762,
    0xbbacdca594227d28,
];

/// A case's outputs from this build, into `out`, through the entry points'
/// bodies from `platform`; returns their length.
pub(crate) fn outputs(platform: Platform, case: Case, input: &[u8], previous: &[u8; 32], out: &mut [u8]) -> usize {
    let words = |key: &[u8; 32]| crate::platform::words_from_le_bytes_32(key);
    let mut digest = |hash: crate::Hash| {
        out[..32].copy_from_slice(hash.as_bytes());
        32
    };
    match case {
        Hash { offset, len } => digest(crate::hash_serial_from(platform, &input[offset..][..len], crate::IV, 0)),
        Many { offset, len, count } => many(platform, &input[..len * count], offset, len, &mut out[..count * crate::OUT_LEN]),
        ChunkValues { chunks } => chunk_values(platform, &input[..chunks * crate::CHUNK_LEN], previous, out),
        Keyed { len } => digest(crate::hash_serial_from(platform, &input[..len], &words(previous), crate::KEYED_HASH)),
        Derive { len } => {
            let context = crate::hash_serial_from(platform, CONTEXT.as_bytes(), crate::IV, crate::DERIVE_KEY_CONTEXT);
            digest(crate::hash_serial_from(platform, &input[..len], &words(context.as_bytes()), crate::DERIVE_KEY_MATERIAL))
        }
        Incremental { len, piece } => digest(incremental(platform, &input[..len], piece)),
        Xof { len, skip, out: n } => xof(platform, &input[..len], skip, &mut out[..n]),
        Compact { len } => {
            #[cfg(all(blake3_neon_hybrid, feature = "std"))]
            let hash = crate::compact::hash(&input[..len], crate::IV, 0);
            #[cfg(not(all(blake3_neon_hybrid, feature = "std")))]
            let hash = crate::hash_serial_from(platform, &input[..len], crate::IV, 0);
            digest(hash)
        }
    }
}

/// A Hasher on `platform` fed `input` in pieces of `piece` bytes (out of
/// line, as `xof` and `many`: their buffers take the stack only in turn).
#[inline(never)]
fn incremental(platform: Platform, input: &[u8], piece: usize) -> crate::Hash {
    let mut hasher = crate::Hasher::from_core(crate::HasherCore::new_from(crate::IV, 0, platform));
    for part in input.chunks(piece) {
        hasher.update(part);
    }
    hasher.finalize()
}

/// Extended output of `input` from position `skip`, into `out`.
#[inline(never)]
fn xof(platform: Platform, input: &[u8], skip: u64, out: &mut [u8]) -> usize {
    let mut hasher = crate::Hasher::from_core(crate::HasherCore::new_from(crate::IV, 0, platform));
    let mut reader = hasher.update(input).finalize_xof();
    reader.set_position(skip);
    reader.fill(out);
    out.len()
}

/// hash_many() of `messages`, `len` bytes each, laid out in slots `offset`
/// bytes into a buffer of its own; the digests in `out`.
#[inline(never)]
fn many(platform: Platform, messages: &[u8], offset: usize, len: usize, out: &mut [u8]) -> usize {
    let slot = crate::many::slot_len(len);
    let count = out.len() / crate::OUT_LEN;
    let mut buffer = [0u8; BATCH_LEN];
    let batch = &mut buffer[offset..][..slot * count];
    for (message, bytes) in batch.chunks_mut(slot).zip(messages.chunks(len.max(1))) {
        message[..len].copy_from_slice(&bytes[..len]);
    }
    let mut digests = [[0u8; crate::OUT_LEN]; BATCH_COUNT];
    crate::hash_many_serial_from(platform, batch, len, crate::IV, 0, &mut digests[..count]);
    for (to, digest) in out.chunks_mut(crate::OUT_LEN).zip(&digests) {
        to.copy_from_slice(digest);
    }
    out.len()
}

/// The chaining values of the whole chunks in `input`, at counter 1 on,
/// keyed by `key` (see Case::ChunkValues), into `out`.
fn chunk_values(platform: Platform, input: &[u8], key: &[u8; 32], out: &mut [u8]) -> usize {
    let key = crate::platform::words_from_le_bytes_32(key);
    let mut chunks = arrayvec::ArrayVec::<&[u8; crate::CHUNK_LEN], 18>::new();
    chunks.extend(input.chunks_exact(crate::CHUNK_LEN).map(|c| c.try_into().expect("whole chunks")));
    let out = &mut out[..chunks.len() * crate::OUT_LEN];
    #[cfg(blake3_sme2)]
    if chunks.len() % crate::sme2::HYBRID_GROUP == 0 {
        let turn = crate::platform::Sme2Turn::take(platform, true);
        if matches!(turn.platform(), Platform::SME2) {
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
            return out.len();
        }
    }
    platform.hash_many(
        &chunks,
        &key,
        1,
        crate::IncrementCounter::Yes,
        crate::KEYED_HASH,
        crate::CHUNK_START,
        crate::CHUNK_END,
        out,
    );
    out.len()
}

/// Run every case and panic on the first that differs from the reference.
fn run() {
    let platform = Platform::detect_unchecked();
    chain(
        |case, input, previous, out| outputs(platform, case, input, previous, out),
        |index, folded| {
            assert!(
                folded == EXPECTED[index],
                "blake3-servil startup self-test failed: case {index}, {} (platform {}): this build or this CPU \
                 computes wrong BLAKE3 digests on that path",
                CASES[index].1,
                platform.name(),
            );
        },
    );
}

/// The self-test's progress in this process.
const UNTESTED: u8 = 0;
const RUNNING: u8 = 1;
const PASSED: u8 = 2;
const FAILED: u8 = 3;
static STATE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(UNTESTED);

/// Run the self-test unless it has passed: once per process, the first
/// caller running it and any other waiting. It publishes no data, so a
/// relaxed load suffices on the fast path.
#[inline]
pub(crate) fn ensure() {
    #[cfg(not(miri))]
    if STATE.load(core::sync::atomic::Ordering::Relaxed) != PASSED {
        ensure_slow();
    }
}

#[cfg(not(miri))]
#[cold]
#[inline(never)]
fn ensure_slow() {
    use core::sync::atomic::Ordering::Relaxed;
    if STATE.compare_exchange(UNTESTED, RUNNING, Relaxed, Relaxed).is_ok() {
        /// A failure (the panic below, or one inside a kernel) leaves
        /// FAILED, so that the waiting threads stop too.
        struct Failed;
        impl Drop for Failed {
            fn drop(&mut self) {
                STATE.store(FAILED, Relaxed);
            }
        }
        let failed = Failed;
        run();
        core::mem::forget(failed);
        STATE.store(PASSED, Relaxed);
        return;
    }
    loop {
        match STATE.load(Relaxed) {
            PASSED => return,
            FAILED => panic!("blake3-servil startup self-test failed on another thread: this build or this CPU computes wrong BLAKE3 digests"),
            _ => core::hint::spin_loop(),
        }
    }
}

/// The self-test's own time, for probes: run it again (it has passed).
#[cfg(not(miri))]
pub(crate) fn run_again() {
    ensure();
    run();
}

#[cfg(all(test, feature = "std"))]
mod test {
    use super::*;

    /// `outputs` as `chain` takes it: the bytes written into its buffer.
    fn into(
        mut outputs: impl FnMut(Case, &[u8], &[u8; 32]) -> std::vec::Vec<u8>,
    ) -> impl FnMut(Case, &[u8], &[u8; 32], &mut [u8]) -> usize {
        move |case, input, previous, out| {
            let output = outputs(case, input, previous);
            out[..output.len()].copy_from_slice(&output);
            output.len()
        }
    }

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
            Incremental { len, .. } | Compact { len } => digest(reference_impl::Hasher::new(), &input[..len], 0, 32),
            Xof { len, skip, out } => digest(reference_impl::Hasher::new(), &input[..len], skip as usize, out),
        }
    }

    /// EXPECTED is the reference implementation's answer for every case.
    /// On a difference the test prints the table the reference gives;
    /// pasting it in is a reviewed change.
    #[test]
    fn self_test_matches_the_reference() {
        let mut folds = std::vec::Vec::new();
        chain(into(reference), |_, folded| folds.push(folded));
        let table: std::string::String = folds.iter().map(|f| std::format!("    0x{f:016x},\n")).collect();
        assert!(folds == EXPECTED, "EXPECTED differs from the reference implementation; its table:\n{table}");
    }

    /// This build agrees with the reference on every case (as the startup
    /// run, which already passed for this test process, found).
    #[test]
    fn self_test_passes() {
        let platform = Platform::detect();
        chain(|case, input, previous, out| outputs(platform, case, input, previous, out), |index, folded| assert_eq!(folded, EXPECTED[index], "case {index}, {}", CASES[index].1));
    }

    /// Chaining carries a difference forward: a wrong first output changes
    /// every fold after it.
    #[test]
    fn self_test_chain_carries_a_fault_forward() {
        let mut good = std::vec::Vec::new();
        chain(into(reference), |_, folded| good.push(folded));
        let mut faulty = std::vec::Vec::new();
        let mut first = true;
        chain(
            into(|case, input, previous| {
                let mut output = reference(case, input, previous);
                if std::mem::take(&mut first) {
                    output[31] ^= 1;
                }
                output
            }),
            |_, folded| faulty.push(folded),
        );
        let same: std::vec::Vec<usize> = (0..good.len()).filter(|&i| good[i] == faulty[i]).collect();
        assert!(same.is_empty(), "every fold changes after a fault in the first case; unchanged: {same:?}");
    }
}
