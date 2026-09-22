// Architecture-specific diagnostic; other builds keep the example runnable.
#[cfg(blake3_neon)]
mod probe {
    use blake3_servil::platform::Platform;
    use blake3_servil::IncrementCounter as IC;
    use std::time::Instant;
    const CHUNK: usize = 1024;
    pub fn run() {
        let input: Vec<u8> = (0..(128u32 << 10)).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
        let chunks: Vec<&[u8; CHUNK]> = input.chunks_exact(CHUNK).map(|c| c.try_into().unwrap()).collect();
        let key = blake3_servil::platform::words_from_le_bytes_32(&[0u8; 32]);
        let sme2 = Platform::detect();
        let neon = Platform::NEON;
        let mut out = vec![0u8; 128 * 32];
        let mut out2 = vec![0u8; 128 * 32];
        let reps = 20000;
        let mut time = |name: &str, f: &mut dyn FnMut(&mut [u8], &mut [u8])| {
            let start = Instant::now();
            for _ in 0..reps { f(&mut out, &mut out2); }
            println!("{name}: {:.2} us", start.elapsed().as_nanos() as f64 / reps as f64 / 1000.0);
        };
        time("chunks only", &mut |out, _| { sme2.hash_many::<CHUNK>(&chunks, &key, 0, IC::Yes, 0, 1, 2, out); });
        time("chunks + scalar hash(64 B)", &mut |out, _| {
            sme2.hash_many::<CHUNK>(&chunks, &key, 0, IC::Yes, 0, 1, 2, out);
            std::hint::black_box(blake3_servil::hash(&input[..64]));
        });
        time("chunks + NEON 1 chunk", &mut |out, out2| {
            sme2.hash_many::<CHUNK>(&chunks, &key, 0, IC::Yes, 0, 1, 2, out);
            neon.hash_many::<CHUNK>(&chunks[..1], &key, 0, IC::Yes, 0, 1, 2, out2);
        });
        time("chunks + NEON 16 chunks", &mut |out, out2| {
            sme2.hash_many::<CHUNK>(&chunks, &key, 0, IC::Yes, 0, 1, 2, out);
            neon.hash_many::<CHUNK>(&chunks[..16], &key, 0, IC::Yes, 0, 1, 2, out2);
        });
        time("NEON 16 chunks alone", &mut |_, out2| {
            neon.hash_many::<CHUNK>(&chunks[..16], &key, 0, IC::Yes, 0, 1, 2, out2);
        });
        time("chunks + memcpy 4 KiB", &mut |out, out2| {
            sme2.hash_many::<CHUNK>(&chunks, &key, 0, IC::Yes, 0, 1, 2, out);
            out2.copy_from_slice(&out[..]);
        });
        time("chunks + f64 math", &mut |out, _| {
            sme2.hash_many::<CHUNK>(&chunks, &key, 0, IC::Yes, 0, 1, 2, out);
            let x = std::hint::black_box(1.5f64);
            std::hint::black_box(x * x + 2.0);
        });
    }
}

fn main() {
    #[cfg(blake3_neon)]
    probe::run();
    #[cfg(not(blake3_neon))]
    eprintln!("This diagnostic requires an AArch64 NEON build.");
}
