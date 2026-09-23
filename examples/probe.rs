// Architecture-specific diagnostic; other builds keep the example runnable.
#[cfg(blake3_neon)]
mod probe {
    //! Raw kernel rates: SME2 vs NEON, alone and side by side, on this machine.
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
        let ns = start.elapsed().as_nanos() as f64;
        ns / (reps * input.len()) as f64
    }

    pub fn run() {
        let len: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(1 << 20);
        let input: Vec<u8> = vec![0x5a; len];
        let reps = (64 << 20) / len;
        let sme2 = Platform::detect();
        let neon = Platform::NEON;
        println!("detected {}", sme2.name());
        for _ in 0..3 {
            println!("sme2 alone {:.3}   neon alone {:.3}", rate(sme2, &input, reps), rate(neon, &input, reps));
        }
        for (a, b) in [(sme2, sme2), (sme2, neon), (neon, neon)] {
            let (ra, rb) = std::thread::scope(|s| {
                let ta = s.spawn(|| rate(a, &input, reps));
                let tb = s.spawn(|| rate(b, &input, reps));
                (ta.join().unwrap(), tb.join().unwrap())
            });
            println!("pair {}+{}: {:.3} / {:.3}", a.name(), b.name(), ra, rb);
        }
    }
}

fn main() {
    #[cfg(blake3_neon)]
    probe::run();
    #[cfg(not(blake3_neon))]
    eprintln!("This diagnostic requires an AArch64 NEON build.");
}
