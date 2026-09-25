//! Probe (branch probe/piece-sizes, never merged): what the "for best
//! performance" advice rests on. 8 MiB through Hasher::update and
//! update_multithreaded in pieces of 16 KiB to 8 MiB, against hash and
//! hash_multithreaded on the whole; 1024 one-block messages through
//! hash_many against a loop of hash; the first hash_multithreaded of a
//! process with and without initialize() first. Wall time, and cycles
//! where the OS counts them (examples/support/clocks.rs).
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("probe/piece-sizes: platform {}", blake3_servil::kernel_report().platform);
    // The first multithreaded call, cold (before initialize), then warm.
    let input: Vec<u8> = (0..8usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    let small = &input[..1 << 20];
    let t = Instant::now();
    black_box(blake3_servil::hash_multithreaded(black_box(small)));
    let cold = t.elapsed().as_nanos();
    let t = Instant::now();
    black_box(blake3_servil::hash_multithreaded(black_box(small)));
    let warm = t.elapsed().as_nanos();
    println!("first hash_multithreaded(1 MiB) of the process {cold} ns; the second {warm} ns");
    clocks::set_qos(clocks::USER_INTERACTIVE);
    let whole = blake3_servil::hash(&input);
    for round in 0..2 {
        println!("round {round}, 8 MiB, ns/B (fastest of 9 batches: wall, cycles)");
        let s = clocks::measure(9, 20_000, || { black_box(blake3_servil::hash(black_box(&input))); }).fastest();
        println!("  hash whole                          {:.4}  [{}]", s.ns / input.len() as f64, s.show());
        let s = clocks::measure(9, 20_000, || { black_box(blake3_servil::hash_multithreaded(black_box(&input))); }).fastest();
        println!("  hash_multithreaded whole            {:.4}  [{}]", s.ns / input.len() as f64, s.show());
        for piece in [16usize << 10, 64 << 10, 256 << 10, 1 << 20, 8 << 20] {
            let mut h = blake3_servil::Hasher::new();
            for p in input.chunks(piece) { h.update(p); }
            assert_eq!(h.finalize(), whole);
            let s = clocks::measure(9, 20_000, || {
                let mut h = blake3_servil::Hasher::new();
                for p in black_box(&input).chunks(piece) { h.update(p); }
                black_box(h.finalize());
            }).fastest();
            let m = clocks::measure(9, 20_000, || {
                let mut h = blake3_servil::Hasher::new();
                for p in black_box(&input).chunks(piece) { h.update_multithreaded(p); }
                black_box(h.finalize());
            }).fastest();
            println!("  pieces of {:>5} KiB: update {:.4} [{}]   update_multithreaded {:.4}", piece >> 10, s.ns / input.len() as f64, s.show(), m.ns / input.len() as f64);
        }
        let messages: Vec<[u8; 64]> = (0..1024).map(|i| [i as u8; 64]).collect();
        let refs: Vec<&[u8]> = messages.iter().map(|m| &m[..]).collect();
        let mut outs = vec![blake3_servil::Hash::from([0; 32]); 1024];
        let s = clocks::measure(9, 20_000, || { blake3_servil::hash_many(black_box(&refs), &mut outs); black_box(&outs); }).fastest();
        let l = clocks::measure(9, 20_000, || { for (m, o) in refs.iter().zip(outs.iter_mut()) { *o = blake3_servil::hash(black_box(m)); } black_box(&outs); }).fastest();
        println!("  1024 x 64 B: hash_many {:.1} ns/msg, a loop of hash {:.1} ns/msg", s.ns / 1024.0, l.ns / 1024.0);
    }
}
