// Architecture-specific diagnostic; other builds keep the example runnable.
#[cfg(blake3_neon)]
mod probe {
    use blake3_servil::platform::Platform;
    use std::time::Instant;
    const CHUNK: usize = 1024;
    pub fn run() {
        let len = 1 << 20;
        let input: Vec<u8> = vec![0x5a; len];
        let chunks: Vec<&[u8; CHUNK]> = input.chunks_exact(CHUNK).map(|c| c.try_into().unwrap()).collect();
        let key = blake3_servil::platform::words_from_le_bytes_32(&[0u8; 32]);
        for degree in [2usize, 3, 4, 5, 6, 8, 9, 10, 15, 16] {
            let mut out = vec![0u8; 16 * 32];
            let start = Instant::now();
            for _ in 0..32 {
                for group in chunks.chunks(degree) {
                    Platform::NEON.hash_many::<CHUNK>(group, &key, 0, blake3_servil::IncrementCounter::Yes, 0, 1, 2, &mut out);
                }
            }
            println!("neon degree {degree}: {:.3} ns/B", start.elapsed().as_nanos() as f64 / (32.0 * len as f64));
        }
    }
}

fn main() {
    #[cfg(blake3_neon)]
    probe::run();
    #[cfg(not(blake3_neon))]
    eprintln!("This diagnostic requires an AArch64 NEON build.");
}
