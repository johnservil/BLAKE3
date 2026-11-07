//! Probe for bench-hashes #142: how macOS's `host_statistics` CPU ticks
//! arrive beside a known load. Every 50 ms for 4 s, print the interval's
//! machine busy ticks (user+system+nice, all CPUs) and this process's own
//! CPU time; from 1 s to 2 s, `threads` threads spin. Lumped accounting
//! shows as intervals whose busy ticks exceed what their wall time allows.
//!
//!     host_lab [threads]
#[cfg(target_vendor = "apple")]
fn main() {
    use std::sync::atomic::{AtomicBool, Ordering};
    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics(host: u32, flavor: i32, info: *mut u32, count: *mut u32) -> i32;
        fn sysconf(name: i32) -> i64;
        fn getrusage(who: i32, usage: *mut i64) -> i32;
    }
    fn busy_ticks() -> u64 {
        let mut t = [0u32; 4];
        let mut n = 4u32;
        assert_eq!(unsafe { host_statistics(mach_host_self(), 3, t.as_mut_ptr(), &mut n) }, 0);
        u64::from(t[0]) + u64::from(t[1]) + u64::from(t[3])
    }
    fn own_ns() -> u64 {
        let mut w = [0i64; 18];
        assert_eq!(unsafe { getrusage(0, w.as_mut_ptr()) }, 0);
        (w[0] as u64 * 1_000_000 + w[1] as u64) * 1000 + (w[2] as u64 * 1_000_000 + w[3] as u64) * 1000
    }
    let threads: usize = std::env::args().nth(1).map_or(12, |a| a.parse().expect("a thread count"));
    let per_second = unsafe { sysconf(3) } as u64;
    let cpus = std::thread::available_parallelism().map_or(0, |n| n.get());
    println!("ticks/s {per_second}, CPUs {cpus}, load threads {threads}: 1 s to 2 s");
    println!("ms  busy_ticks  busy_cpu_s  own_cpu_s  excess_cpu_s(busy-own)");
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        let start = std::time::Instant::now();
        let (mut last_ticks, mut last_own) = (busy_ticks(), own_ns());
        let mut spinners = Vec::new();
        let mut started = false;
        for i in 1..=80 {
            while start.elapsed() < std::time::Duration::from_millis(50 * i) {
                std::hint::spin_loop();
            }
            if !started && i >= 20 {
                started = true;
                for _ in 0..threads {
                    spinners.push(s.spawn(|| {
                        let mut x = 1u64;
                        while !stop.load(Ordering::Relaxed) {
                            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                            std::hint::black_box(x);
                        }
                    }));
                }
            }
            if i == 40 {
                stop.store(true, Ordering::Relaxed);
            }
            let (ticks, own) = (busy_ticks(), own_ns());
            let busy_s = (ticks - last_ticks) as f64 / per_second as f64;
            let own_s = (own - last_own) as f64 / 1e9;
            println!("{:>4}  {:>10}  {:>10.3}  {:>9.3}  {:>+10.3}", 50 * i, ticks - last_ticks, busy_s, own_s, busy_s - own_s);
            (last_ticks, last_own) = (ticks, own);
        }
    });
}

#[cfg(not(target_vendor = "apple"))]
fn main() {
    println!("this probe reads macOS's host_statistics; nothing to do here");
}
