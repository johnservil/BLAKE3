//! Probe (never merged): hash() in a loop at 32 KiB to 1 MiB and
//! Hasher::update in 64 KiB and 256 KiB pieces, 8 MiB per batch, wall
//! and cycles (examples/support/clocks.rs), five processes' worth of
//! stack depths by a pad before each measurement.
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;

#[inline(never)]
fn at_depth<const PAD: usize>(input: &[u8], piece: usize, hasher: bool) -> clocks::Sample {
    let pad = black_box([0u8; PAD]);
    let s = clocks::measure(7, 20_000, || {
        if hasher {
            let mut h = blake3_servil::Hasher::new();
            for p in black_box(input).chunks(piece) { h.update(p); }
            black_box(h.finalize());
        } else {
            for p in black_box(input).chunks(piece) { black_box(blake3_servil::hash(black_box(p))); }
        }
    }).fastest();
    black_box(pad);
    s
}

fn main() {
    clocks::set_qos(clocks::USER_INTERACTIVE);
    let input: Vec<u8> = (0..8usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    println!("probe/hash-sizes: platform {}", blake3_servil::kernel_report().platform);
    for round in 0..2 {
        for (label, piece, hasher) in [("hash 32 KiB", 32usize << 10, false), ("hash 64 KiB", 64 << 10, false), ("hash 128 KiB", 128 << 10, false), ("hash 256 KiB", 256 << 10, false), ("hash 1 MiB", 1 << 20, false), ("Hasher, 64 KiB pieces", 64 << 10, true), ("Hasher, 256 KiB pieces", 256 << 10, true)] {
            let r = [at_depth::<16>(&input, piece, hasher), at_depth::<48>(&input, piece, hasher), at_depth::<1040>(&input, piece, hasher), at_depth::<4112>(&input, piece, hasher)];
            let cells: Vec<String> = r.iter().map(|s| format!("{:.4} ({:.2}/ns)", s.ns / input.len() as f64, s.per_ns())).collect();
            println!("round {round} {label:<24} {}", cells.join("  "));
        }
    }
}
