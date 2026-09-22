//! The bencher's pattern: a duo batch of mt calls, then a few ms of other
//! hashing on the callers' threads, repeated. Median batch rate.
use std::sync::{Arc, Barrier};
use std::time::Instant;
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let sizes = if sizes.is_empty() { vec![64 << 10, 128 << 10, 256 << 10, 1 << 20, 8 << 20] } else { sizes };
    for len in sizes {
        let input: Arc<Vec<u8>> = Arc::new((0..len as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect());
        let iterations = ((4 << 20) / len).max(1); // about a 1 ms batch of mt calls, as the bencher calibrates
        let rounds = 15;
        let barrier = Arc::new(Barrier::new(2));
        let run = |input: Arc<Vec<u8>>, barrier: Arc<Barrier>| move || {
            let mut times = Vec::new();
            for _ in 0..rounds {
                // Other contenders: ~4 ms of serial hashing.
                let start = Instant::now();
                while start.elapsed().as_millis() < 4 { std::hint::black_box(blake3_servil::hash(&input[..len.min(1 << 20)])); }
                barrier.wait();
                let start = Instant::now();
                for _ in 0..iterations { std::hint::black_box(blake3_servil::hash_multithreaded(&input)); }
                times.push(start.elapsed().as_nanos() as f64 / (iterations * len) as f64);
            }
            times
        };
        let a = std::thread::spawn(run(input.clone(), barrier.clone()));
        let b = std::thread::spawn(run(input.clone(), barrier.clone()));
        let (a, b) = (a.join().unwrap(), b.join().unwrap());
        let later: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect();
        println!("{len:>9}  interleaved duo mt {:.3} ns/B  (min {:.3} max {:.3})", median(later.clone()), later.iter().cloned().fold(f64::MAX, f64::min), later.iter().cloned().fold(0.0, f64::max));
    }
}
