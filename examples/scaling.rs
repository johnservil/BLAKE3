//! Threads of one process hashing at once: n threads, each hashing its own
//! input back to back with `hash` (1 MiB) and `hash_many` (4096 one-block
//! messages); per-thread time, median and slowest thread. The case where
//! threads would share an SME unit; compare with `--features no_sme2`.
//!
//!     cargo run --release --example scaling
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

fn per_thread(n: usize, len: usize, batch: bool) -> (f64, f64) {
    let barrier = Arc::new(Barrier::new(n));
    let threads: Vec<_> = (0..n)
        .map(|k| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let input: Vec<u8> = (0..len).map(|i| (i * 31 + k) as u8).collect();
                let messages: Vec<&[u8]> = input.chunks_exact(64).collect();
                let mut digests = vec![blake3_servil::Hash::from_bytes([0; 32]); messages.len()];
                let mut hash = || {
                    if batch {
                        blake3_servil::hash_many(std::hint::black_box(&messages), &mut digests);
                    } else {
                        std::hint::black_box(blake3_servil::hash(std::hint::black_box(&input)));
                    }
                };
                for _ in 0..3 {
                    hash();
                }
                barrier.wait();
                let started = Instant::now();
                let mut count = 0u64;
                while started.elapsed() < Duration::from_millis(150) {
                    hash();
                    count += 1;
                }
                let units = if batch { len / 64 } else { len };
                started.elapsed().as_nanos() as f64 / (count as f64 * units as f64)
            })
        })
        .collect();
    let mut times: Vec<f64> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[times.len() - 1])
}

fn main() {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let counts: Vec<usize> = [1, 2, 4, 8, 12, 16].into_iter().filter(|&n| n <= cpus).collect();
    println!("platform {}; per thread, median/slowest", blake3_servil::kernel_report().platform);
    for (label, len, batch) in [("hash 1 MiB, ns/B", 1 << 20, false), ("hash_many 4096 x 64 B, ns/msg", 4096 * 64, true)] {
        let mut line = format!("{label:30}");
        for &n in &counts {
            let (median, slowest) = per_thread(n, len, batch);
            line += &format!("  n={n:<2} {median:.3}/{slowest:.3}");
        }
        println!("{line}");
    }
}
