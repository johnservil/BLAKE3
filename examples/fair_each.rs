//! BLAKE3 #213, the collection shape: two callers each hashing a
//! collection like the bench's "git objects" (2048 items, 24 B to 768 KiB,
//! mixed) with `hash_each_multithreaded`, free-running for two seconds:
//! how many collections each completed, and each call's time.
use blake3_servil::Mode;
use blake3_servil::throughput::{Block, hash_each_multithreaded};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const CLASSES: &[(usize, usize)] = &[
    (1, 24),
    (13, 48),
    (12, 96),
    (42, 192),
    (182, 384),
    (237, 768),
    (135, 1536),
    (141, 3072),
    (173, 6144),
    (447, 12288),
    (321, 24576),
    (231, 49152),
    (84, 98304),
    (22, 196608),
    (4, 393216),
    (3, 786432),
];

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let secs: u64 = std::env::args().nth(1).map_or(2, |a| a.parse().unwrap());
    let lengths: Vec<usize> = CLASSES.iter().flat_map(|&(n, l)| std::iter::repeat_n(l, n)).collect();
    let order: Vec<usize> = (0..2048).map(|j| lengths[j * 389 % 2048]).collect();
    let counts: Arc<[AtomicUsize; 2]> = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let go = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|who| {
            let counts = counts.clone();
            let go = go.clone();
            let order = order.clone();
            std::thread::spawn(move || {
                let blocks: Vec<Vec<Block>> = order.iter().enumerate().map(|(i, &l)| vec![Block([(i + who) as u8; 64]); l.div_ceil(64).max(1)]).collect();
                let items: Vec<(&[Block], usize)> = blocks.iter().zip(&order).map(|(b, &l)| (&b[..], l)).collect();
                let mut out = vec![[0u8; 32]; items.len()];
                let mut times = Vec::new();
                go.wait();
                let end = Instant::now() + Duration::from_secs(secs);
                while Instant::now() < end {
                    let t = Instant::now();
                    hash_each_multithreaded(Mode::Plain, &items, &mut out);
                    std::hint::black_box(&out);
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                    counts[who].fetch_add(1, Ordering::Relaxed);
                }
                times.sort_by(f64::total_cmp);
                times
            })
        })
        .collect();
    let times: Vec<Vec<f64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let (c0, c1) = (counts[0].load(Ordering::Relaxed), counts[1].load(Ordering::Relaxed));
    let med = |v: &[f64]| v[v.len() / 2];
    println!(
        "collections in {secs} s: {c0} and {c1} ({:.2}); a call's median {:.1} ms and {:.1} ms",
        c0.max(c1) as f64 / c0.min(c1).max(1) as f64,
        med(&times[0]),
        med(&times[1])
    );
}
