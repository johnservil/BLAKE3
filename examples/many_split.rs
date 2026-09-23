use std::time::Instant;
fn main() {
    let buffer: Vec<u8> = vec![0x5a; 1 << 20];
    blake3_servil::initialize();
    for &count in &[1024usize, 1536, 2048, 4096] {
        let inputs: Vec<&[u8]> = buffer.chunks_exact(64).take(count).collect();
        let mut out = vec![blake3_servil::Hash::from_bytes([0; 32]); count];
        let reps = 3000;
        print!("{count:>6} msgs ({:>4} KiB):", count * 64 / 1024);
        for cap in [1usize, 2, 4, 8, 16, usize::MAX] {
            let t = Instant::now();
            for _ in 0..reps { blake3_servil::hash_many_multithreaded_with_budget(std::hint::black_box(&inputs), &mut out, cap); }
            print!("  cap {cap:>2}: {:5.1} µs", t.elapsed().as_nanos() as f64 / reps as f64 / 1000.0);
        }
        let t = Instant::now();
        for _ in 0..reps { blake3_servil::hash_multithreaded(std::hint::black_box(&buffer[..count * 64])); }
        println!("   tree mt: {:5.1} µs", t.elapsed().as_nanos() as f64 / reps as f64 / 1000.0);
    }
}
