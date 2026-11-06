//! Items of equal block count side by side, last blocks included, each the
//! root of one chunk at counter 0: the generated kernels of
//! `c/blake3_last_x86-64_unix.S` (tools/gen_x86.py --last, BLAKE3-Servil
//! #247), four lanes on SSE4.1, eight on AVX2. They are the partial
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
    fn blake3_servil_last_sse41(params: *mut Params<4>);
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
    unsafe { blake3_servil_last_avx512(&p) };
    out.copy_from_slice(&digests[..n]);
}

/// The widest last-block kernel the selected platform takes.
pub(crate) fn lanes(platform: crate::platform::Platform) -> usize {
    match platform {
        crate::platform::Platform::AVX512 => 16,
        crate::platform::Platform::AVX2 => 8,
        crate::platform::Platform::SSE41 => 4,
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
        // Sound: every x86-64 CPU this crate runs on has SSE4.1 (detect()).
        unsafe { hash_group_on::<4>(blake3_servil_last_sse41, items, k, key, flags, out) }
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
    unsafe { kernel(&mut p) };
    for (lane, digest) in out.iter_mut().enumerate() {
        for (i, word) in p.h.iter().enumerate() {
            digest[4 * i..][..4].copy_from_slice(&word[lane].to_le_bytes());
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
        check::<4>();
        check::<8>();
        assert_eq!(core::mem::offset_of!(Params16, lastmsg), 128);
        assert_eq!(core::mem::offset_of!(Params16, lens), 256);
        assert_eq!(core::mem::offset_of!(Params16, key), 320);
        assert_eq!(core::mem::offset_of!(Params16, k), 352);
        assert_eq!(core::mem::offset_of!(Params16, flags), 356);
        assert_eq!(core::mem::offset_of!(Params16, out), 360);
    }

    /// Every block count (k of 0 to 15), last blocks of every length (0,
    /// one byte, 63 and a whole block), 1 to the lanes' count of items of
    /// mixed last lengths, in the three modes, on both kernels the CPU
    /// has: each digest hash_with's.
    #[test]
    fn test_hash_group_against_hash_with() {
        let mut input = std::vec![0u8; 16 * crate::CHUNK_LEN];
        crate::test::paint_test_input(&mut input);
        let key = [9u8; 32];
        let kernels: [(usize, unsafe fn(&[&[u8]], usize, &CVWords, u8, &mut [[u8; OUT_LEN]]), bool); 3] = [
            (4, |i, k, key, f, o| unsafe { hash_group_on::<4>(blake3_servil_last_sse41, i, k, key, f, o) }, crate::platform::sse41_detected()),
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
