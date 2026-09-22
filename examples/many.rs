//! Diagnostic throughput with several independent callers; ns/B of one copy.
use std::sync::Barrier;
use std::time::Instant;
fn main() {
    for n in [2, 4, 8, 16, 32] {
        for len in [256usize << 10, 1 << 20, 8 << 20] {
            let barrier = Barrier::new(n);
            let times = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n).map(|id| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let input: Vec<_> = (0..len).map(|i| (i.wrapping_mul(131) ^ id) as u8).collect();
                        let want = blake3_servil::hash(&input);
                        assert_eq!(want, blake3_servil::hash_multithreaded(&input));
                        let reps = ((8 << 20) / len).max(4);
                        (0..31).map(|_| {
                            barrier.wait();
                            let start = Instant::now();
                            for _ in 0..reps { std::hint::black_box(blake3_servil::hash_multithreaded(std::hint::black_box(&input))); }
                            start.elapsed().as_nanos() as f64 / (reps * len) as f64
                        }).collect::<Vec<_>>()
                    })
                }).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
            });
            let mut times: Vec<_> = (0..31).map(|k| times.iter().map(|t| t[k]).fold(0.0, f64::max)).collect();
            times.sort_by(f64::total_cmp);
            println!("callers={n:2} size={len:8} median={:.3} ns/B", times[15]);
        }
    }
}
