//! Cost of waking a sleeping thread: notify_one's time on the waker, and
//! the latency until the sleeper runs.
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
fn main() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let woke_at = Arc::new(AtomicU64::new(0));
    let epoch = Instant::now();
    let p2 = pair.clone(); let w2 = woke_at.clone();
    std::thread::spawn(move || loop {
        let mut g = p2.0.lock().unwrap();
        while !*g { g = p2.1.wait(g).unwrap(); }
        *g = false;
        w2.store(epoch.elapsed().as_nanos() as u64, Ordering::SeqCst);
    });
    let mut notify = vec![]; let mut latency = vec![];
    for _ in 0..200 {
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t0 = epoch.elapsed().as_nanos() as u64;
        { let mut g = pair.0.lock().unwrap(); *g = true; pair.1.notify_one(); }
        let t1 = epoch.elapsed().as_nanos() as u64;
        loop { let w = woke_at.load(Ordering::SeqCst); if w > t0 { latency.push((w - t0) as f64 / 1000.0); break; } }
        notify.push((t1 - t0) as f64 / 1000.0);
    }
    notify.sort_by(|a, b| a.partial_cmp(b).unwrap()); latency.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("notify_one on the waker: median {:.1} us, p90 {:.1}", notify[100], notify[180]);
    println!("sleeper running after:   median {:.1} us, p90 {:.1}", latency[100], latency[180]);
}
