//! blake3_servil::hash rate by size on the detected platform.
use std::time::Instant;
fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let sizes = if sizes.is_empty() { vec![2048, 4096, 8192, 16384, 65536, 1 << 20, 8 << 20] } else { sizes };
    println!("platform {}", blake3_servil::platform::Platform::detect().name());
    for len in sizes {
        let input: Vec<u8> = vec![0x5a; len];
        let reps = ((64 << 20) / len).max(1);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(blake3_servil::hash(std::hint::black_box(&input)));
            }
            best = best.min(start.elapsed().as_nanos() as f64 / (reps * len) as f64);
        }
        println!("{len:>9} B: {best:.3} ns/B");
    }
}
