//! BLAKE3 #213: do two simultaneous callers of the pool get fair shares?
//! Two threads, released together each round, each hashing its own input
//! with `hash_multithreaded`; each round records the slower caller's time
//! over the faster's, and which caller won. Then both callers run free for
//! a second, call after call: how many calls each completed (fair sharing
//! completes about as many each). `fair_probe LEN ROUNDS`.
use blake3_servil::Mode;
use blake3_servil::throughput::{Block, hash_multithreaded};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let len: usize = std::env::args().nth(1).map_or(1 << 26, |a| a.parse().unwrap());
    let rounds: usize = std::env::args().nth(2).map_or(100, |a| a.parse().unwrap());
    let barrier = Arc::new(Barrier::new(2));
    let times: Arc<[Vec<AtomicUsize>; 2]> = Arc::new([(0..rounds).map(|_| AtomicUsize::new(0)).collect(), (0..rounds).map(|_| AtomicUsize::new(0)).collect()]);
    let handles: Vec<_> = (0..2)
        .map(|who| {
            let barrier = barrier.clone();
            let times = times.clone();
            std::thread::spawn(move || {
                let mut blocks = vec![Block([who as u8 + 1; 64]); len / 64];
                for (i, b) in blocks.iter_mut().enumerate() {
                    b.0[0] = i as u8;
                }
                let mut out = [0u8; 32];
                for r in 0..=rounds {
                    barrier.wait();
                    let t = Instant::now();
                    let h = hash_multithreaded(Mode::Plain, &blocks, len);
                    out.copy_from_slice(h.as_bytes());
                    std::hint::black_box(&out);
                    if r > 0 {
                        times[who][r - 1].store(t.elapsed().as_nanos() as usize, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let mut ratios = Vec::new();
    let mut wins = [0usize; 2];
    for r in 0..rounds {
        let a = times[0][r].load(Ordering::Relaxed) as f64;
        let b = times[1][r].load(Ordering::Relaxed) as f64;
        ratios.push(a.max(b) / a.min(b));
        wins[(b < a) as usize] += 1;
    }
    ratios.sort_by(f64::total_cmp);
    let med = |v: &[f64]| v[v.len() / 2];
    let a: Vec<f64> = (0..rounds).map(|r| times[0][r].load(Ordering::Relaxed) as f64 / 1e6).collect();
    let b: Vec<f64> = (0..rounds).map(|r| times[1][r].load(Ordering::Relaxed) as f64 / 1e6).collect();
    let mut sa = a.clone();
    sa.sort_by(f64::total_cmp);
    let mut sb = b.clone();
    sb.sort_by(f64::total_cmp);
    let mut fast: Vec<f64> = (0..rounds).map(|r| a[r].min(b[r])).collect();
    let mut slow: Vec<f64> = (0..rounds).map(|r| a[r].max(b[r])).collect();
    fast.sort_by(f64::total_cmp);
    slow.sort_by(f64::total_cmp);
    // Alone, for reference: the same call with no neighbour.
    let mut blocks = vec![Block([7u8; 64]); len / 64];
    for (i, b) in blocks.iter_mut().enumerate() {
        b.0[0] = i as u8;
    }
    let mut alone: Vec<f64> = (0..rounds)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(hash_multithreaded(Mode::Plain, &blocks, len));
            t.elapsed().as_nanos() as f64 / 1e6
        })
        .collect();
    alone.sort_by(f64::total_cmp);
    let _ = (&sa, &sb);
    // Free-running: each caller hashes call after call for a second.
    let counts: Arc<[AtomicUsize; 2]> = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let go = Arc::new(Barrier::new(2));
    let free: Vec<_> = (0..2)
        .map(|who| {
            let counts = counts.clone();
            let go = go.clone();
            std::thread::spawn(move || {
                let mut blocks = vec![Block([who as u8 + 3; 64]); len / 64];
                for (i, b) in blocks.iter_mut().enumerate() {
                    b.0[0] = i as u8;
                }
                go.wait();
                let end = Instant::now() + std::time::Duration::from_secs(1);
                while Instant::now() < end {
                    std::hint::black_box(hash_multithreaded(Mode::Plain, &blocks, len));
                    counts[who].fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    for h in free {
        h.join().unwrap();
    }
    let (c0, c1) = (counts[0].load(Ordering::Relaxed), counts[1].load(Ordering::Relaxed));
    println!(
        "len {len}: alone {:.2} ms; beside a neighbour the faster caller {:.2} ms, the slower {:.2} ms (slower/faster median {:.3}, p90 {:.3}); caller 0 faster in {} of {rounds}; free-running a second: {c0} and {c1} calls ({:.2})",
        med(&alone),
        med(&fast),
        med(&slow),
        med(&ratios),
        ratios[ratios.len() * 9 / 10],
        wins[0],
        c0.max(c1) as f64 / c0.min(c1).max(1) as f64
    );
}
