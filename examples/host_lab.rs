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

fn main() {
    clocks::set_qos(clocks::USER_INTERACTIVE);
    let platform = blake3_servil::platform::Platform::detect();
    let input: Vec<u8> = (0..4usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    let mut arena = vec![0u8; 64 << 10];
    let base = arena.as_mut_ptr();
    let input_at = input.as_ptr() as usize;
    println!("probe/scratch-offset: platform {}, input at {:#x}", blake3_servil::kernel_report().platform, input_at);
    for n in [32usize, 64] {
        for round in 0..2 {
            let mut line = format!("{} KiB round {round}:", n);
            for step in 0..64 {
                let delta = step * 64;
                // scratch ≡ input + delta (mod 4096), 128-byte aligned where delta allows.
                let want = (input_at + delta) % 4096;
                let off = (want + 4096 - (base as usize % 4096)) % 4096;
                let s = unsafe { base.add(off) };
                let m = clocks::measure(5, 8_000, || {
                    for piece in black_box(&input).chunks(n * 1024) { unsafe { walk(piece, n, s, platform) } }
                }).fastest();
                line += &format!(" {delta}:{:.2}", m.per_ns());
                if step == 0 { line += &format!("[{:.4} ns/B]", m.ns / input.len() as f64); }
            }
            println!("{line}");
        }
    }
}
