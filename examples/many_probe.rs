//! Time hash_many on batches of one-block messages, single-threaded, solo.
use std::time::Instant;
fn main() {
    let total = 1 << 20;
    let buffer: Vec<u8> = (0..total as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
    for &count in &[1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096, 16384] {
        let inputs: Vec<&[u8]> = buffer.chunks_exact(64).take(count).collect();
        let mut out = vec![blake3_servil::Hash::from_bytes([0; 32]); count];
        // one-at-a-time baseline
        let reps = (2_000_000 / count).max(20);
        let t = Instant::now();
        for _ in 0..reps { for (m, o) in inputs.iter().zip(out.iter_mut()) { *o = blake3_servil::hash(std::hint::black_box(m)); } }
        let each = t.elapsed().as_nanos() as f64 / (reps * count) as f64;
        let t = Instant::now();
        for _ in 0..reps { blake3_servil::hash_many(std::hint::black_box(&inputs), &mut out); }
        let many = t.elapsed().as_nanos() as f64 / (reps * count) as f64;
        let t = Instant::now();
        for _ in 0..reps { blake3_servil::hash_many_multithreaded(std::hint::black_box(&inputs), &mut out); }
        let mt = t.elapsed().as_nanos() as f64 / (reps * count) as f64;
        println!("{count:>6} msgs: hash() each {each:6.1} ns/msg   hash_many {many:6.1}   hash_many_multithreaded {mt:6.1}");
    }
}
