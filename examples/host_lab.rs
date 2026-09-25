//! Probe (never merged): hash(32 KiB) in a loop by stack depth, a pad of
//! 0 to 16 KiB before the call in steps of 256 bytes, with the stack
//! address at the call, so the address dependence of the flat walk's
//! stack buffers can be mapped. Wall and cycles per ns.
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;

#[inline(never)]
fn run(input: &[u8], piece: usize) -> (usize, clocks::Sample) {
    let marker = 0u8;
    let at = black_box(&marker) as *const u8 as usize;
    let s = clocks::measure(5, 10_000, || {
        for p in black_box(input).chunks(piece) { black_box(blake3_servil::hash(black_box(p))); }
    }).fastest();
    (at, s)
}

#[inline(never)]
fn padded(input: &[u8], piece: usize, pad: usize) -> (usize, clocks::Sample) {
    // A variable-length stack allocation, as alloca would make.
    let mut v = [0u8; 16640];
    black_box(&mut v);
    if pad == 0 { return run(input, piece); }
    let r = padded_inner(input, piece, pad - 256);
    black_box(&v);
    r
}

#[inline(never)]
fn padded_inner(input: &[u8], piece: usize, pad: usize) -> (usize, clocks::Sample) {
    let v = black_box([0u8; 256]);
    let r = if pad == 0 { run(input, piece) } else { padded_inner(input, piece, pad - 256) };
    black_box(v);
    r
}

fn main() {
    clocks::set_qos(clocks::USER_INTERACTIVE);
    let input: Vec<u8> = (0..4usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    println!("probe/stack-map: platform {}, input at {:#x}", blake3_servil::kernel_report().platform, input.as_ptr() as usize);
    for piece in [32usize << 10, 64 << 10] {
        for step in 0..64 {
            let pad = step * 256;
            let (at, s) = padded(&input, piece, pad);
            println!("{} KiB pad {pad:5} stack {:#x} (mod 16K {:#6x}): {:.4} ns/B {:.2}/ns", piece >> 10, at, at & 0x3fff, s.ns / input.len() as f64, s.per_ns());
        }
    }
}
