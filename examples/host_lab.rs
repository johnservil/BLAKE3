//! Probe (never merged): the flat walk's scratch (pointer table, chaining
//! values, the padded levels) as one contiguous block placed at every
//! 64-byte offset from the input, mod 4 KiB, with the raw SME2 kernels as
//! the flat walk calls them and the root compression after: which
//! placements keep the SME unit in its fast state. 32 KiB and 64 KiB
//! pieces over 4 MiB, wall and cycles per ns.
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;
unsafe extern "C" {
    fn blake3_sme2_hash16_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
    fn blake3_sme2_hash16_parents_512(i: *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
}
const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

/// One piece of `n` chunks: table at `s`, CVs at s + 8n (n values), half
/// at s + 8n + 32n; levels down to two values, then the root compression.
unsafe fn walk(piece: &[u8], n: usize, s: *mut u8, platform: blake3_servil::platform::Platform) {
    unsafe {
        let table = s as *mut *const u8;
        for i in 0..n { *table.add(i) = piece.as_ptr().add(i * 1024); }
        let (mut src, mut dst) = (s.add(8 * n), s.add(8 * n + 32 * n));
        blake3_sme2_hash16_chunks_512(table, IV.as_ptr(), 0, 1 << 8 | 2 << 16, src, (n / 16) as u64);
        let mut count = n;
        while count > 2 {
            blake3_sme2_hash16_parents_512(src, IV.as_ptr(), 0, 4, dst, (count / 2).div_ceil(16) as u64);
            count /= 2;
            std::mem::swap(&mut src, &mut dst);
        }
        let mut cv = IV;
        platform.compress_in_place(&mut cv, &*(src as *const [u8; 64]), 64, 0, 4 | 8);
        black_box(cv);
    }
}

#[inline(never)]
fn at_depth(depth: usize, f: &mut dyn FnMut() -> f64) -> (usize, f64) {
    let pad = [0u8; 64];
    black_box(&pad);
    if depth == 0 {
        let at = black_box(&pad) as *const _ as usize;
        return (at, f());
    }
    let r = at_depth(depth - 1, f);
    black_box(&pad);
    r
}

fn main() {
    clocks::set_qos(clocks::USER_INTERACTIVE);
    let platform = blake3_servil::platform::Platform::detect();
    let input: Vec<u8> = (0..4usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    let mut arena = vec![0u8; 64 << 10];
    let base = arena.as_mut_ptr();
    let input_at = input.as_ptr() as usize;
    println!("probe/stack-vs-scratch: platform {}, input at {:#x}", blake3_servil::kernel_report().platform, input_at);
    let s = unsafe { base.add((input_at + 1024 + 4096 - base as usize % 4096) % 4096) };
    for round in 0..2 {
        let mut rows = Vec::new();
        for depth in 0..48 {
            let mut f = || {
                let m = clocks::measure(5, 8_000, || {
                    for piece in black_box(&input).chunks(32 * 1024) { unsafe { walk(piece, 32, s, platform) } }
                }).fastest();
                m.per_ns()
            };
            rows.push(at_depth(depth, &mut f));
        }
        rows.sort_by_key(|r| r.0 % 4096);
        println!("round {round}, 32 KiB, heap scratch, by stack address mod 4K: {}", rows.iter().map(|(a, c)| format!("{}:{:.2}", a % 4096, c)).collect::<Vec<_>>().join(" "));
        let mut rows = Vec::new();
        for depth in 0..48 {
            let mut f = || {
                let m = clocks::measure(5, 8_000, || {
                    for piece in black_box(&input).chunks(32 * 1024) { black_box(blake3_servil::hash(piece)); }
                }).fastest();
                m.per_ns()
            };
            rows.push(at_depth(depth, &mut f));
        }
        rows.sort_by_key(|r| r.0 % 4096);
        println!("round {round}, 32 KiB, hash(), by stack address mod 4K: {}", rows.iter().map(|(a, c)| format!("{}:{:.2}", a % 4096, c)).collect::<Vec<_>>().join(" "));
    }
}
