//! Duo rate by size: two threads each hash their own copy at once, the
//! later finish is the sample; median of samples. Columns: serial, mt.
use std::sync::{Arc, Barrier};
use std::time::Instant;
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn duo(len: usize, f: fn(&[u8]) -> blake3_servil::Hash) -> f64 {
    let input: Arc<Vec<u8>> = Arc::new((0..len as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect());
    let reps = ((8 << 20) / len).clamp(4, 512);
    let samples = 21;
    let barrier = Arc::new(Barrier::new(2));
    let run = |input: Arc<Vec<u8>>, barrier: Arc<Barrier>| {
        move || {
            let mut times = Vec::with_capacity(samples);
            for _ in 0..samples {
                barrier.wait();
                let start = Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(f(std::hint::black_box(&input)));
                }
                times.push(start.elapsed().as_nanos() as f64 / (reps * len) as f64);
            }
            times
        }
    };
    let a = std::thread::spawn(run(input.clone(), barrier.clone()));
    let b = std::thread::spawn(run(input.clone(), barrier.clone()));
    let (a, b) = (a.join().unwrap(), b.join().unwrap());
    median(a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect())
}
fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let sizes = if sizes.is_empty() { vec![32 << 10, 64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20] } else { sizes };
    println!("{:>9}  {:>8}  {:>8}  {:>8}", "size", "serial", "mt", "mt-solo");
    for len in sizes {
        let serial = duo(len, blake3_servil::hash);
        let mt = duo(len, blake3_servil::hash_multithreaded);
        // Solo: one thread only, for reference.
        let input: Vec<u8> = (0..len as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
        let reps = ((8 << 20) / len).clamp(4, 512);
        let solo = median((0..21).map(|_| {
            let start = Instant::now();
            for _ in 0..reps { std::hint::black_box(blake3_servil::hash_multithreaded(std::hint::black_box(&input))); }
            start.elapsed().as_nanos() as f64 / (reps * len) as f64
        }).collect());
        println!("{:>9}  {:>8.3}  {:>8.3}  {:>8.3}", len, serial, mt, solo);
    }
}
