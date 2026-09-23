// Architecture-specific diagnostic; other builds keep the example runnable.
#[cfg(blake3_neon)]
mod probe {
    //! N threads at once, all SME2 or all NEON: aggregate throughput by N.
    use blake3_servil::platform::Platform;
    use std::time::Instant;
    const CHUNK: usize = 1024;
    fn rate(platform: Platform, input: &[u8], reps: usize) -> f64 {
        let chunks: Vec<&[u8; CHUNK]> = input.chunks_exact(CHUNK).map(|c| c.try_into().unwrap()).collect();
        let degree = platform.simd_degree().max(2);
        let mut out = vec![0u8; degree * 32];
        let key = blake3_servil::platform::words_from_le_bytes_32(&[0u8; 32]);
        let start = Instant::now();
        for _ in 0..reps {
            for group in chunks.chunks(degree) {
                platform.hash_many::<CHUNK>(group, &key, 0, blake3_servil::IncrementCounter::Yes, 0, 1, 2, &mut out);
            }
        }
        start.elapsed().as_nanos() as f64 / (reps * input.len()) as f64
    }
    pub fn run() {
        let len = 1 << 20;
        let input: Vec<u8> = vec![0x5a; len];
        let reps = 32;
        for platform in [Platform::detect(), Platform::NEON] {
            for n in [1, 2, 3, 4, 6, 8, 12, 16] {
                let rates: Vec<f64> = std::thread::scope(|s| {
                    let hs: Vec<_> = (0..n).map(|_| s.spawn(|| rate(platform, &input, reps))).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let worst = rates.iter().cloned().fold(0.0, f64::max);
                let agg: f64 = rates.iter().map(|r| 1.0 / r).sum();
                println!("{} x{:2}: worst {:.3} ns/B, aggregate {:.2} GB/s", platform.name(), n, worst, agg);
            }
        }
    }
}

fn main() {
    #[cfg(blake3_neon)]
    probe::run();
    #[cfg(not(blake3_neon))]
    eprintln!("This diagnostic requires an AArch64 NEON build.");
}
