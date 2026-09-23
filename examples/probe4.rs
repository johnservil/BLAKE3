// Architecture-specific diagnostic; other builds keep the example runnable.
#[cfg(blake3_neon)]
mod probe {
    //! Per-piece cost: a subtree piece of each size hashed as lanes does it, on SME2 and NEON.
    use blake3_servil::hazmat::HasherExt;
    use blake3_servil::platform::Platform;
    use std::time::Instant;
    pub fn run() {
        let len = 8 << 20;
        let input: Vec<u8> = vec![0x5a; len];
        for platform in [Platform::detect(), Platform::NEON] {
            for piece in [8 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20] {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let start = Instant::now();
                    for (i, bytes) in input.chunks_exact(piece).enumerate() {
                        let mut h = blake3_servil::Hasher::new_with_platform(platform);
                        h.set_input_offset((i * piece) as u64);
                        h.update(bytes);
                        std::hint::black_box(h.finalize_non_root());
                    }
                    best = best.min(start.elapsed().as_nanos() as f64 / len as f64);
                }
                println!("{} piece {:>5} KiB: {:.3} ns/B", platform.name(), piece >> 10, best);
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
