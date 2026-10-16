//! A one-shot call while a stream runs: one thread streams 1 KiB messages
//! nonstop (16 KiB tasks of members), another calls hash_multithreaded on
//! 4 MiB every millisecond. Each side alone, then together: the one-shot
//! call's time (median, p90 over its calls) and the stream's throughput.
//!   contention_probe [SECONDS]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn stream_side(stop: &AtomicBool, bytes: &AtomicU64) {
    let messages: Vec<Vec<u8>> = (0..1024).map(|m| vec![m as u8; 1024]).collect();
    // Places cycled as the stream hands them back: a stream that runs on
    // and on reuses a fixed set.
    let mut places = vec![blake3_servil::Hash::from_bytes([0; 32]); 2048];
    let free = std::cell::RefCell::new(places.iter_mut().collect::<Vec<_>>());
    blake3_servil::stream(
        blake3_servil::Mode::Hash,
        |d: &mut [blake3_servil::Hash]| {
            std::hint::black_box(&d[0]);
            free.borrow_mut().push(&mut d[0]);
        },
        |s| {
            while !stop.load(Ordering::Relaxed) {
                for m in &messages {
                    let out = loop {
                        if let Some(out) = free.borrow_mut().pop() {
                            break out;
                        }
                        s.collect();
                    };
                    s.submit(m, out);
                }
                s.collect();
                bytes.fetch_add(1 << 20, Ordering::Relaxed);
            }
            s.finish();
        },
    );
}

fn oneshot_side(stop: &AtomicBool) -> Vec<u64> {
    let input = vec![9u8; 4 << 20];
    let mut times = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let t = clocks::now();
        std::hint::black_box(blake3_servil::hash_multithreaded(&input));
        times.push(clocks::since_ns(t));
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    times
}

fn report(name: &str, mut times: Vec<u64>, bytes: u64, ns: u64) {
    times.sort_unstable();
    let pct = |p: usize| times.get(times.len() * p / 100).copied().unwrap_or(0) / 1000;
    let rate = if bytes > 0 { format!("stream {}.{:02} GB/s", bytes / ns, bytes % ns * 100 / ns) } else { "stream -".into() };
    println!("{name:9} one-shot 4 MiB: {} calls, median {} us, p90 {} us; {rate}", times.len(), pct(50), pct(90));
}

fn main() {
    blake3_servil::initialize_multithreaded();
    let secs: u64 = std::env::args().nth(1).map_or(3, |s| s.parse().unwrap());
    for case in ["stream", "one-shot", "together"] {
        let stop = AtomicBool::new(false);
        let bytes = AtomicU64::new(0);
        let started = clocks::now();
        let times = std::thread::scope(|t| {
            if case != "one-shot" {
                t.spawn(|| stream_side(&stop, &bytes));
            }
            let h = (case != "stream").then(|| t.spawn(|| oneshot_side(&stop)));
            std::thread::sleep(std::time::Duration::from_secs(secs));
            stop.store(true, Ordering::Relaxed);
            h.map_or(Vec::new(), |h| h.join().unwrap())
        });
        report(case, times, bytes.load(Ordering::Relaxed), clocks::since_ns(started));
    }
}
