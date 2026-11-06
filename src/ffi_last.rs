//! Items of equal block count side by side, last blocks included, each the
//! root of one chunk at counter 0: the generated kernels of
//! `c/blake3_last_x86-64_unix.S` (tools/gen_x86.py --last, BLAKE3-Servil
//! #247), eight lanes on AVX2, sixteen on AVX-512. They share the partial
//! kernels' rounds with two changes: they hash END/64 blocks, not 16, and
//! read block SWITCH/64 of each lane through its own pointer (LASTMSG),
//! to a zero-padded copy of the item's last block, so no lane reads past
//! its item. The last block's flags carry CHUNK_END and ROOT, so each
//! lane's chaining value after it is its digest.

use crate::{BLOCK_LEN, CHUNK_END, CHUNK_START, CVWords, IV, OUT_LEN, ROOT};

/// The kernels' parameter block for `W` lanes, laid out as gen_x86.py's
/// `layout(w, last=True)` says (the test below checks the offsets).
#[repr(C, align(64))]
pub struct Params<const W: usize> {
    msg: [*const u8; W],
    key: [[u32; W]; 8],
    iv: [[u32; W]; 4],
    rot16: [u32; W],
    rot8: [u32; W],
    ctr: [[u32; W]; 2],
    lens: [[u32; W]; 16],
    flags: [[u32; W]; 16],
    last: [u32; 16],
    h: [[u32; W]; 8],
    snap: [[u32; W]; 8],
    lastmsg: [*const u8; W],
    end: u32,
    switch: u32,
}

unsafe extern "C" {
    fn blake3_servil_last_avx2(params: *mut Params<8>);
}

/// AVX-512's compact parameter block (gen_x86.py's last16): common
/// counter 0 and flags, per-lane last length, input and last-block pointer.
#[repr(C, align(64))]
struct Params16 {
    msg: [*const u8; 16],
    lastmsg: [*const u8; 16],
    lens: [u32; 16],
    key: CVWords,
    k: u32,
    flags: u32,
    out: *mut [u8; OUT_LEN],
}

unsafe extern "C" {
    fn blake3_servil_last_avx512(params: *const Params16);
}

/// AVX-512F's sixteen-lane kernel: `items` have `k` full blocks before
/// their last, and `out` holds one digest each. Unsafe: the CPU has AVX-512F.
unsafe fn hash_group16(items: &[&[u8]], k: usize, key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]]) {
    let n = items.len();
    assert!((1..=16).contains(&n) && out.len() == n && k < 16, "1 to 16 items of up to a chunk");
    assert!(items.iter().all(|i| i.len().saturating_sub(1) / BLOCK_LEN == k), "items of {k} whole blocks before their last");
    let mut last = [[0u8; BLOCK_LEN]; 16];
    let lens = core::array::from_fn(|j| {
        let tail = &items[if j < n { j } else { 0 }][k * BLOCK_LEN..];
        last[j][..tail.len()].copy_from_slice(tail);
        tail.len() as u32
    });
    let mut digests = [[0u8; OUT_LEN]; 16];
    let p = Params16 {
        msg: core::array::from_fn(|j| items[if j < n { j } else { 0 }].as_ptr()),
        lastmsg: core::array::from_fn(|j| last[j].as_ptr()),
        lens,
        key: *key,
        k: k as u32,
        flags: flags as u32,
        out: digests.as_mut_ptr(),
    };
    // Sound: every input reaches its k whole blocks and every LASTMSG a
    // padded 64-byte block. The kernel stores 16 digests, including the
    // unread duplicate lanes, into distinct caller-owned scratch (no
    // sharing with workers). The caller checked AVX-512F.
    crate::ran_kernel(blake3_servil_last_avx512 as *const ());
    unsafe { blake3_servil_last_avx512(&p) };
    out.copy_from_slice(&digests[..n]);
}

/// Per-lane ending offsets extend the equal-length kernel's compact
/// parameter block. Keeping ends on its own line also makes mask loads
/// independent of the common scalar fields.
#[repr(C, align(64))]
struct ParamsMixed {
    common: Params16,
    ends: [u32; 16],
}

unsafe extern "C" {
    fn blake3_servil_mixed_avx512(params: *const ParamsMixed);
}

/// Sixteen or fewer messages of arbitrary lengths up to a chunk. The
/// kernel records a lane's root at its own last block; later steps use
/// the padded block as a dummy input and leave that digest untouched.
/// Unsafe: the CPU supports AVX-512F.
pub(crate) unsafe fn hash_mixed16(items: &[&[u8]], key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]]) {
    let n = items.len();
    assert!((1..=16).contains(&n) && out.len() == n, "1 to 16 items, one digest each");
    assert!(items.iter().all(|i| i.len() <= crate::CHUNK_LEN), "items up to one chunk");
    let ends: [u32; 16] = core::array::from_fn(|j| (items[if j < n { j } else { 0 }].len().saturating_sub(1) / BLOCK_LEN * BLOCK_LEN) as u32);
    let mut last = [[0u8; BLOCK_LEN]; 16];
    let lens = core::array::from_fn(|j| {
        let item = items[if j < n { j } else { 0 }];
        let tail = &item[ends[j] as usize..];
        last[j][..tail.len()].copy_from_slice(tail);
        tail.len() as u32
    });
    let mut words = [[0u32; 16]; 8];
    let p = ParamsMixed {
        common: Params16 {
            msg: core::array::from_fn(|j| items[if j < n { j } else { 0 }].as_ptr()),
            lastmsg: core::array::from_fn(|j| last[j].as_ptr()),
            lens,
            key: *key,
            k: *ends.iter().max().unwrap() / BLOCK_LEN as u32,
            flags: flags as u32,
            out: words.as_mut_ptr().cast(),
        },
        ends,
    };
    // Sound: each lane reads its whole blocks only before ends[j], then
    // the padded last block; output is a caller-owned word-major array of
    // 8 * 16 words, with each lane written once at its ending step. No
    // shared mutable state, and the caller checked AVX-512F.
    crate::ran_kernel(blake3_servil_mixed_avx512 as *const ());
    unsafe { blake3_servil_mixed_avx512(&p) };
    for (j, digest) in out.iter_mut().enumerate() {
        for (i, word) in words.iter().enumerate() {
            digest[4 * i..][..4].copy_from_slice(&word[j].to_le_bytes());
        }
    }
}

/// AVX2's mixed-lane ending offsets follow the existing parameter block,
/// whose size includes its 64-byte alignment padding.
#[repr(C, align(64))]
struct ParamsMixed8 {
    common: Params<8>,
    ends: [u32; 8],
}

unsafe extern "C" {
    fn blake3_servil_mixed_avx2(params: *mut ParamsMixed8);
}

/// AVX2 version of hash_mixed16: up to eight lanes, a per-lane snapshot at
/// its last block. Unsafe: the CPU supports AVX2.
pub(crate) unsafe fn hash_mixed8(items: &[&[u8]], key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]]) {
    let n = items.len();
    assert!((1..=8).contains(&n) && out.len() == n, "1 to 8 items, one digest each");
    assert!(items.iter().all(|i| i.len() <= crate::CHUNK_LEN), "items up to one chunk");
    let ends: [u32; 8] = core::array::from_fn(|j| (items[if j < n { j } else { 0 }].len().saturating_sub(1) / BLOCK_LEN * BLOCK_LEN) as u32);
    let mut last = [[0u8; BLOCK_LEN]; 8];
    let lens: [u32; 8] = core::array::from_fn(|j| {
        let item = items[if j < n { j } else { 0 }];
        let tail = &item[ends[j] as usize..];
        last[j][..tail.len()].copy_from_slice(tail);
        tail.len() as u32
    });
    let f = flags as u32;
    let mut p = ParamsMixed8 {
        common: Params {
            msg: core::array::from_fn(|j| items[if j < n { j } else { 0 }].as_ptr()),
            lastmsg: core::array::from_fn(|j| last[j].as_ptr()),
            key: core::array::from_fn(|i| [key[i]; 8]),
            iv: core::array::from_fn(|i| [IV[i]; 8]),
            rot16: crate::ffi_partial::rotation(2),
            rot8: crate::ffi_partial::rotation(1),
            ctr: [[0; 8]; 2],
            lens: [[64; 8]; 16],
            flags: [[f; 8]; 16],
            last: [0; 16],
            h: [[0; 8]; 8],
            snap: [[0; 8]; 8],
            end: *ends.iter().max().unwrap() + BLOCK_LEN as u32,
            switch: 0,
        },
        ends,
    };
    for j in 0..8 {
        let k = ends[j] as usize / BLOCK_LEN;
        p.common.lens[k][j] = lens[j];
        p.common.flags[0][j] |= CHUNK_START as u32;
        p.common.flags[k][j] |= (CHUNK_END | ROOT) as u32;
    }
    // Sound: each lane reads only its whole blocks before ends[j], then
    // a padded last block. The snapshot retains its digest thereafter;
    // all parameter storage is initialized and owned by this caller.
    crate::ran_kernel(blake3_servil_mixed_avx2 as *const ());
    unsafe { blake3_servil_mixed_avx2(&mut p) };
    for (j, digest) in out.iter_mut().enumerate() {
        for (i, word) in p.common.snap.iter().enumerate() {
            digest[4 * i..][..4].copy_from_slice(&word[j].to_le_bytes());
        }
    }
}

/// Finishing width for this platform, or zero for SSE4.1's serial path.
pub(crate) fn lanes(platform: crate::platform::Platform) -> usize {
    match platform {
        crate::platform::Platform::AVX512 => 16,
        crate::platform::Platform::AVX2 => 8,
        crate::platform::Platform::SSE41 => 0,
    }
}

/// The digest of each of `items` into `out` (one each), in the mode of
/// `key` and `flags`: 1 to lanes() items, each with `k` whole blocks
/// before its last (1 to 64 bytes; none for an empty item, k = 0).
pub(crate) fn hash_group(platform: crate::platform::Platform, items: &[&[u8]], k: usize, key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]]) {
    if matches!(platform, crate::platform::Platform::AVX512) {
        // Sound: the selected platform guarantees AVX-512F.
        unsafe { hash_group16(items, k, key, flags, out) }
    } else if !matches!(platform, crate::platform::Platform::SSE41) {
        // Sound: the CPU has AVX2.
        unsafe { hash_group_on::<8>(blake3_servil_last_avx2, items, k, key, flags, out) }
    } else {
        unreachable!("SSE4.1 uses the serial collection path")
    }
}

/// [`hash_group`] on `W` lanes. Unsafe: the CPU must have `kernel`'s
/// instructions.
unsafe fn hash_group_on<const W: usize>(
    kernel: unsafe extern "C" fn(*mut Params<W>),
    items: &[&[u8]],
    k: usize,
    key: &CVWords,
    flags: u8,
    out: &mut [[u8; OUT_LEN]],
) {
    let n = items.len();
    assert!((1..=W).contains(&n) && out.len() == n && k < 16, "1 to {W} items of up to a chunk");
    assert!(items.iter().all(|i| i.len().saturating_sub(1) / BLOCK_LEN == k), "items of {k} whole blocks before their last");
    // Each lane's last block, zero-padded; the lanes past `n` hash item 0
    // again, their digests unread.
    let mut last = [[0u8; BLOCK_LEN]; W];
    let lens: [u32; W] = core::array::from_fn(|j| {
        let item = items[if j < n { j } else { 0 }];
        let tail = &item[k * BLOCK_LEN..];
        last[j][..tail.len()].copy_from_slice(tail);
        tail.len() as u32
    });
    let flags = flags as u32;
    let start = if k == 0 { CHUNK_START as u32 } else { 0 };
    let mut p = Params::<W> {
        msg: core::array::from_fn(|j| items[if j < n { j } else { 0 }].as_ptr()),
        key: core::array::from_fn(|i| [key[i]; W]),
        iv: core::array::from_fn(|i| [IV[i]; W]),
        rot16: crate::ffi_partial::rotation(2),
        rot8: crate::ffi_partial::rotation(1),
        ctr: [[0; W]; 2],
        lens: [[BLOCK_LEN as u32; W]; 16],
        flags: [[flags; W]; 16],
        last: [0; 16],
        h: [[0; W]; 8],
        snap: [[0; W]; 8],
        lastmsg: core::array::from_fn(|j| last[j].as_ptr()),
        end: (BLOCK_LEN * (k + 1)) as u32,
        switch: (BLOCK_LEN * k) as u32,
    };
    p.lens[k] = lens;
    p.flags[0] = [flags | CHUNK_START as u32; W];
    p.flags[k] = [flags | start | (CHUNK_END | ROOT) as u32; W];
    // Sound: each lane's pointer reaches its k whole blocks (an item of
    // more than k of them, or item 0 again), LASTMSG a whole block each,
    // and the caller's CPU has the kernel's instructions.
    crate::ran_kernel(kernel as *const ());
    unsafe { kernel(&mut p) };
    for (lane, digest) in out.iter_mut().enumerate() {
        for (i, word) in p.h.iter().enumerate() {
            digest[4 * i..][..4].copy_from_slice(&word[lane].to_le_bytes());
        }
    }
}

/// Fixed-key startup answers, from the independent reference test below:
/// two shapes (equal block counts and mixed) at widths eight and sixteen.
/// A fixed seed keeps these checks independent of the other startup
/// chain's case count; each kernel is still checked before user hashing.
const SELF_TEST_EXPECTED: [[u64; 2]; 2] = [[0x9c9bccc8becc1867, 0xece70b823c776e95], [0xfb48780a8346cae2, 0x153fbf85d61a34f8]];
const SELF_TEST_KEY: [u8; 32] = *b"whats the Elvish word for friend";

fn self_test_shape(width: usize, mixed: bool, mut hash: impl FnMut(&[&[u8]], &CVWords, u8, &mut [[u8; OUT_LEN]])) -> u64 {
    let mut bytes = [0u8; 2048];
    crate::self_test::fill(&mut bytes);
    let refs: [&[u8]; 16] = core::array::from_fn(|j| {
        let len = if mixed { [0, 1024, 1, 63, 64, 129, 511, 700, 128, 192, 256, 512, 768, 1023, 600, 255][j] } else { 193 + j * 3 };
        &bytes[j * 13..][..len]
    });
    let mut out = [[0u8; OUT_LEN]; 16];
    let key = crate::platform::words_from_le_bytes_32(&SELF_TEST_KEY);
    hash(&refs[..width], &key, crate::KEYED_HASH, &mut out[..width]);
    crate::self_test::fold(out[..width].as_flattened())
}

fn check_self_test(width: usize, mixed: bool, folded: u64) {
    let row = if width == 16 { 1 } else { 0 };
    assert_eq!(folded, SELF_TEST_EXPECTED[row][mixed as usize], "blake3-servil startup self-test failed: {width}-lane last blocks, mixed={mixed}");
}

pub(crate) fn self_test(platform: crate::platform::Platform) -> usize {
    let width = lanes(platform);
    if width < 8 {
        return 0;
    }
    for mixed in [false, true] {
        let folded = self_test_shape(width, mixed, |items, key, flags, out| {
            if !mixed {
                hash_group(platform, items, 3, key, flags, out);
            } else if width == 16 {
                // Sound: the platform is AVX-512.
                unsafe { hash_mixed16(items, key, flags, out) };
            } else {
                // Sound: the platform is AVX2.
                unsafe { hash_mixed8(items, key, flags, out) };
            }
        });
        check_self_test(width, mixed, folded);
    }
    2
}

#[cfg(test)]
pub(crate) fn entries(platform: crate::platform::Platform) -> [usize; 2] {
    if lanes(platform) == 16 {
        [blake3_servil_last_avx512 as *const () as usize, blake3_servil_mixed_avx512 as *const () as usize]
    } else {
        [blake3_servil_last_avx2 as *const () as usize, blake3_servil_mixed_avx2 as *const () as usize]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_startup_answers_against_official() {
        let mut answers = [[0u64; 2]; 2];
        for (row, width) in [8, 16].into_iter().enumerate() {
            for (column, mixed) in [false, true].into_iter().enumerate() {
                answers[row][column] = self_test_shape(width, mixed, |items, key, flags, out| {
                    assert_eq!(*key, crate::platform::words_from_le_bytes_32(&SELF_TEST_KEY));
                    assert_eq!(flags, crate::KEYED_HASH);
                    for (item, digest) in items.iter().zip(out) {
                        *digest = crate::test::official::hash(crate::Mode::Keyed(&SELF_TEST_KEY), item);
                    }
                });
            }
        }
        assert_eq!(answers, SELF_TEST_EXPECTED, "official startup table: {answers:016x?}");
    }

    #[test]
    fn test_startup_check_rejects_corruption() {
        for (row, width) in [8, 16].into_iter().enumerate() {
            for mixed in [false, true] {
                assert!(
                    std::panic::catch_unwind(|| check_self_test(width, mixed, SELF_TEST_EXPECTED[row][mixed as usize] ^ 1)).is_err(),
                    "a corrupt {width}-lane answer was accepted"
                );
            }
        }
    }

    #[test]
    fn test_lane_widths_are_the_platforms() {
        use crate::platform::Platform;
        assert_eq!(lanes(Platform::SSE41), 0);
        assert_eq!(lanes(Platform::AVX2), 8);
        assert_eq!(lanes(Platform::AVX512), 16);
    }

    /// The parameter block's offsets are the generator's (gen_x86.py's
    /// `layout(w, last=True)`).
    #[test]
    fn test_params_layout_is_the_generators() {
        fn check<const W: usize>() {
            let v = 4 * W;
            let snap = 8 * W + 48 * v + 64 + 8 * v;
            let got = [
                core::mem::offset_of!(Params<W>, lens),
                core::mem::offset_of!(Params<W>, flags),
                core::mem::offset_of!(Params<W>, h),
                core::mem::offset_of!(Params<W>, snap),
                core::mem::offset_of!(Params<W>, lastmsg),
                core::mem::offset_of!(Params<W>, end),
                core::mem::offset_of!(Params<W>, switch),
            ];
            assert_eq!(
                got,
                [8 * W + 16 * v, 8 * W + 32 * v, 8 * W + 48 * v + 64, snap, snap + 8 * v, snap + 8 * v + 8 * W, snap + 8 * v + 8 * W + 4],
                "{W} lanes"
            );
        }
        check::<8>();
        assert_eq!(core::mem::offset_of!(Params16, lastmsg), 128);
        assert_eq!(core::mem::offset_of!(Params16, lens), 256);
        assert_eq!(core::mem::offset_of!(Params16, key), 320);
        assert_eq!(core::mem::offset_of!(Params16, k), 352);
        assert_eq!(core::mem::offset_of!(Params16, flags), 356);
        assert_eq!(core::mem::offset_of!(Params16, out), 360);
        assert_eq!(core::mem::offset_of!(ParamsMixed, common), 0);
        assert_eq!(core::mem::offset_of!(ParamsMixed, ends), 384);
        assert_eq!(core::mem::offset_of!(ParamsMixed8, common), 0);
        assert_eq!(core::mem::offset_of!(ParamsMixed8, ends), 2304);
    }

    /// Every lane can end at a different step. Exercise all block counts,
    /// every partial-block length, empty lanes beside full chunks, all
    /// widths 1..16 and three modes, against the independent reference.
    #[test]
    fn test_mixed_lanes_against_official() {
        type Kernel = unsafe fn(&[&[u8]], &CVWords, u8, &mut [[u8; OUT_LEN]]);
        let kernels: [(usize, Kernel, bool); 2] = [(8, hash_mixed8, crate::platform::avx2_detected()), (16, hash_mixed16, crate::platform::avx512_detected())];
        let input: std::vec::Vec<u8> = (0..32768usize).map(|i| (i * 37 + i / 991) as u8).collect();
        let key: [u8; 32] = core::array::from_fn(|j| (j * 13 + 7) as u8);
        for mode in [crate::Mode::Plain, crate::Mode::Keyed(&key), crate::Mode::DeriveKey("mixed last blocks")] {
            let mut room = core::mem::MaybeUninit::uninit();
            let (words, flags) = mode.key(&mut room);
            for (w, kernel, present) in kernels {
                if !present {
                    continue;
                }
                for tail in 0..=64usize {
                    for n in 1..=w {
                        let items: std::vec::Vec<&[u8]> = (0..n)
                            .map(|j| {
                                let k = (j * 7 + tail) % 16;
                                let len = if j == 0 {
                                    0
                                } else if j == 1 {
                                    1024
                                } else {
                                    k * 64 + tail.max(1)
                                };
                                &input[j * 1027..][..len]
                            })
                            .collect();
                        let mut out = std::vec![[0u8; OUT_LEN]; n];
                        // Sound: this kernel's CPU features were detected.
                        unsafe { kernel(&items, words, flags, &mut out) };
                        for (j, item) in items.iter().enumerate() {
                            assert_eq!(out[j], crate::test::official::hash(mode, item), "{w} lanes, tail {tail}, n {n}, lane {j}, {} bytes", item.len());
                        }
                    }
                }
            }
        }
    }

    /// Every block count (k of 0 to 15), last blocks of every length (0,
    /// one byte, 63 and a whole block), 1 to the lanes' count of items of
    /// mixed last lengths, in the three modes, on both kernels the CPU
    /// has: each digest hash_with's.
    #[test]
    fn test_hash_group_against_hash_with() {
        let mut input = std::vec![0u8; 16 * crate::CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        let key: [u8; 32] = core::array::from_fn(|j| (j * 13 + 7) as u8);
        let kernels: [(usize, unsafe fn(&[&[u8]], usize, &CVWords, u8, &mut [[u8; OUT_LEN]]), bool); 2] = [
            (8, |i, k, key, f, o| unsafe { hash_group_on::<8>(blake3_servil_last_avx2, i, k, key, f, o) }, crate::platform::avx2_detected()),
            (16, |i, k, key, f, o| unsafe { hash_group16(i, k, key, f, o) }, crate::platform::avx512_detected()),
        ];
        for mode in [crate::Mode::Plain, crate::Mode::Keyed(&key), crate::Mode::DeriveKey("last blocks")] {
            let mut room = core::mem::MaybeUninit::uninit();
            let (words, flags) = mode.key(&mut room);
            for (w, kernel, present) in kernels {
                if !present {
                    continue;
                }
                for k in 0..16usize {
                    for n in 1..=w {
                        let lens: std::vec::Vec<usize> = (0..n)
                            .map(|j| {
                                let tail = [1, 63, 64, if k == 0 { 0 } else { 2 }][(j + k) % 4];
                                k * BLOCK_LEN + tail
                            })
                            .collect();
                        // Each item at its own offset, so no two lanes share a read.
                        let items: std::vec::Vec<&[u8]> = lens.iter().enumerate().map(|(j, &l)| &input[j * 997 % 4096..][..l]).collect();
                        let mut out = std::vec![[0u8; OUT_LEN]; n];
                        // Sound: the kernel's CPU feature is present.
                        unsafe { kernel(&items, k, words, flags, &mut out) };
                        for (j, item) in items.iter().enumerate() {
                            assert_eq!(
                                out[j],
                                *crate::convenience::hash_with(mode, item).as_bytes(),
                                "{w} lanes, k {k}, {n} items, item {j} of {} B",
                                item.len()
                            );
                        }
                        for (digest, item) in out.iter().zip(&items) {
                            assert_eq!(*digest, crate::test::official::hash(mode, item), "independent reference, {w} lanes, k {k}, {n} items");
                        }
                    }
                }
            }
        }
    }
}
