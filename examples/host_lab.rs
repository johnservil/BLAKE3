//! Everything the pool's design rests on, measured on this machine: run it on
//! a native host and in a VM, and compare. Both are targets.
//!
//!     cargo run --release --example host_lab
//!
//! Prints a report and writes it to `host-lab.<unix-seconds>.txt` in the
//! current directory. Takes about two minutes. Sections:
//!
//! 1. primitives: sched_yield, the clock, a cache-line ping-pong, how soon
//!    a waiting thread sees a posted word (spin, yield, condvar), and what
//!    a wake costs the waker;
//! 2. scaling: n threads hashing at once, per-thread time;
//! 3. idle waiters: hashing threads beside idle threads that spin, yield,
//!    or sleep;
//! 4. SME2 → NEON: the cost of the first NEON work after an SME2 kernel
//!    (SME2 machines only);
//! 5. the pool: hash_multithreaded and hash_many_multithreaded against the
//!    serial calls, alone and two callers at once, by size;
//! 6. two SME2 callers at once: whether two threads of one process running
//!    SME2 at the same time share one SME unit (each at half speed), before
//!    and right after a burst of work on every core;
//! 7. WFE (AArch64): hand-off latency and how often WFE returns unprompted.
//!    Last, since a platform that traps WFE at EL0 would stop the program.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

/// `len` copies of one byte. Speed does not depend on the bytes; distinct
/// seeds give concurrent callers distinct buffers.
fn filled(len: usize, seed: u64) -> Vec<u8> {
    vec![seed as u8 ^ 0x5a; len]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Report(String);
impl Report {
    fn line(&mut self, text: String) {
        println!("{text}");
        self.0.push_str(&text);
        self.0.push('\n');
    }
}

/// Spin for `d` without touching the scheduler.
fn busy(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Wait {
    Spin,
    Yield,
    Condvar,
    #[cfg(target_arch = "aarch64")]
    Wfe,
}

/// Wait until `word != old`, idling in WFE between checks.
#[cfg(target_arch = "aarch64")]
fn wfe_wait(word: &AtomicU64, old: u64) -> u64 {
    loop {
        let v: u64;
        unsafe {
            std::arch::asm!("ldaxr {v}, [{a}]", v = out(reg) v, a = in(reg) word.as_ptr(), options(nostack));
        }
        if v != old {
            unsafe { std::arch::asm!("clrex", options(nostack, nomem)) };
            return v;
        }
        unsafe { std::arch::asm!("wfe", options(nostack, nomem)) };
    }
}

/// Nanoseconds from a post to a waiter of `style` seeing it, averaged.
fn hand_off(style: Wait) -> f64 {
    let word = Arc::new(AtomicU64::new(0));
    let seen = Arc::new(AtomicU64::new(0));
    let pair = Arc::new((Mutex::new(0u64), Condvar::new()));
    let epoch = Instant::now();
    let reps = 2000u64;
    let (w2, s2, p2) = (word.clone(), seen.clone(), pair.clone());
    let waiter = std::thread::spawn(move || {
        let mut total = 0u64;
        for k in 1..=reps {
            let v = match style {
                Wait::Spin => loop {
                    let v = w2.load(Ordering::Acquire);
                    if v >> 40 == k {
                        break v;
                    }
                    std::hint::spin_loop();
                },
                Wait::Yield => loop {
                    let v = w2.load(Ordering::Acquire);
                    if v >> 40 == k {
                        break v;
                    }
                    std::thread::yield_now();
                },
                Wait::Condvar => {
                    let mut g = p2.0.lock().unwrap();
                    while *g != k {
                        g = p2.1.wait(g).unwrap();
                    }
                    w2.load(Ordering::Acquire)
                }
                #[cfg(target_arch = "aarch64")]
                Wait::Wfe => {
                    let mut v = w2.load(Ordering::Acquire);
                    while v >> 40 != k {
                        v = wfe_wait(&w2, v);
                    }
                    v
                }
            };
            total += epoch.elapsed().as_nanos() as u64 - (v & 0xff_ffff_ffff);
            s2.store(k, Ordering::Release);
        }
        total
    });
    for k in 1..=reps {
        busy(Duration::from_micros(30));
        let stamp = k << 40 | epoch.elapsed().as_nanos() as u64;
        if style == Wait::Condvar {
            let mut g = pair.0.lock().unwrap();
            word.store(stamp, Ordering::Release);
            *g = k;
            pair.1.notify_one();
        } else {
            word.store(stamp, Ordering::Release);
        }
        while seen.load(Ordering::Acquire) != k {
            std::hint::spin_loop();
        }
    }
    waiter.join().unwrap() as f64 / reps as f64
}

/// Per-thread ns per hash of `len` bytes with `hashers` threads hashing at
/// once for 40 ms, beside `idlers` idle threads waiting in `style` (None:
/// no idle threads): (median, slowest).
fn hashers_beside(len: usize, hashers: usize, idlers: usize, style: Option<Wait>) -> (f64, f64) {
    let stop = Arc::new(AtomicBool::new(false));
    let word = Arc::new(AtomicU64::new(0));
    let idle: Vec<_> = (0..if style.is_some() { idlers } else { 0 })
        .map(|_| {
            let (stop, word) = (stop.clone(), word.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match style.unwrap() {
                        Wait::Spin => {
                            for _ in 0..64 {
                                std::hint::black_box(word.load(Ordering::Acquire));
                                std::hint::spin_loop();
                            }
                        }
                        Wait::Yield => {
                            std::hint::black_box(word.load(Ordering::Acquire));
                            std::thread::yield_now();
                        }
                        Wait::Condvar => std::thread::sleep(Duration::from_millis(1)),
                        #[cfg(target_arch = "aarch64")]
                        Wait::Wfe => {
                            let old = word.load(Ordering::Acquire);
                            if !stop.load(Ordering::Relaxed) {
                                wfe_wait(&word, old);
                            }
                        }
                    }
                }
            })
        })
        .collect();
    // A writer bounds every WFE wait, whatever the platform's event stream.
    let ticker = {
        let (stop, word) = (stop.clone(), word.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(2));
                word.fetch_add(1, Ordering::SeqCst);
            }
        })
    };
    std::thread::sleep(Duration::from_millis(10));
    let barrier = Arc::new(Barrier::new(hashers));
    let threads: Vec<_> = (0..hashers)
        .map(|k| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let input = filled(len, k as u64);
                for _ in 0..50 {
                    std::hint::black_box(blake3_servil::hash(&input));
                }
                barrier.wait();
                let t = Instant::now();
                let mut count = 0u64;
                while t.elapsed() < Duration::from_millis(40) {
                    for _ in 0..4 {
                        std::hint::black_box(blake3_servil::hash(std::hint::black_box(&input)));
                    }
                    count += 4;
                }
                t.elapsed().as_nanos() as f64 / count as f64
            })
        })
        .collect();
    let times: Vec<f64> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    stop.store(true, Ordering::SeqCst);
    word.fetch_add(1, Ordering::SeqCst);
    for t in idle {
        t.join().unwrap();
    }
    ticker.join().unwrap();
    let slowest = times.iter().cloned().fold(0.0, f64::max);
    (median(times), slowest)
}

/// Nanoseconds per call of `f` over `input`, back to back, from `callers`
/// threads at once (each its own input), the slowest caller's figure.
fn per_call(len: usize, callers: usize, f: fn(&[u8])) -> f64 {
    let budget = Duration::from_millis(60);
    let barrier = Arc::new(Barrier::new(callers));
    let threads: Vec<_> = (0..callers)
        .map(|k| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let input = filled(len, 100 + k as u64);
                for _ in 0..3 {
                    f(&input);
                }
                barrier.wait();
                let t = Instant::now();
                let mut count = 0u64;
                while t.elapsed() < budget || count < 5 {
                    f(std::hint::black_box(&input));
                    count += 1;
                }
                t.elapsed().as_nanos() as f64 / count as f64
            })
        })
        .collect();
    threads.into_iter().map(|t| t.join().unwrap()).fold(0.0, f64::max)
}

fn tree_serial(input: &[u8]) {
    std::hint::black_box(blake3_servil::hash(input));
}
fn tree_mt(input: &[u8]) {
    std::hint::black_box(blake3_servil::hash_multithreaded(input));
}
fn batch_serial(input: &[u8]) {
    let mut out = vec![[0u8; 32]; input.len() / 64];
    blake3_servil::hash_many(input, 64, &mut out);
    std::hint::black_box(&out);
}
fn batch_mt(input: &[u8]) {
    let mut out = vec![[0u8; 32]; input.len() / 64];
    blake3_servil::hash_many_multithreaded(input, 64, &mut out);
    std::hint::black_box(&out);
}

#[cfg(blake3_sme2)]
fn transition(report: &mut Report) {
    use blake3_servil::platform::Platform;
    use blake3_servil::IncrementCounter as IC;
    let detected = Platform::detect();
    if !matches!(detected, Platform::SME2) {
        report.line("  no SME2 on this CPU; nothing to measure".to_owned());
        return;
    }
    let input = filled(128 << 10, 7);
    let chunks: Vec<&[u8; 1024]> = input.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
    let key = blake3_servil::platform::words_from_le_bytes_32(&[0u8; 32]);
    let mut out = vec![0u8; 128 * 32];
    let mut out2 = vec![0u8; 128 * 32];
    let mut scratch = vec![0u8; 4096];
    let reps = 5000;
    let mut time = |name: &str, f: &mut dyn FnMut(&mut [u8], &mut [u8], &mut [u8])| -> f64 {
        let t = Instant::now();
        for _ in 0..reps {
            f(&mut out, &mut out2, &mut scratch);
        }
        let us = t.elapsed().as_nanos() as f64 / reps as f64 / 1000.0;
        let _ = name;
        us
    };
    let alone = time("sme", &mut |o, _, _| detected.hash_many::<1024>(&chunks, &key, 0, IC::Yes, 0, 1, 2, o));
    let scalar = time("sme+scalar", &mut |o, _, _| {
        detected.hash_many::<1024>(&chunks, &key, 0, IC::Yes, 0, 1, 2, o);
        std::hint::black_box(blake3_servil::hash(&input[..64]));
    });
    let neon = time("sme+neon", &mut |o, o2, _| {
        detected.hash_many::<1024>(&chunks, &key, 0, IC::Yes, 0, 1, 2, o);
        Platform::NEON.hash_many::<1024>(&chunks[..1], &key, 0, IC::Yes, 0, 1, 2, o2);
    });
    let neon_alone = time("neon", &mut |_, o2, _| Platform::NEON.hash_many::<1024>(&chunks[..1], &key, 0, IC::Yes, 0, 1, 2, o2));
    let memcpy = time("sme+memcpy", &mut |o, _, s| {
        detected.hash_many::<1024>(&chunks, &key, 0, IC::Yes, 0, 1, 2, o);
        s.copy_from_slice(std::hint::black_box(&o[..4096]));
    });
    report.line(format!("  128 chunks on SME2 alone {alone:.2} µs; then a scalar 64 B hash +{:.2} µs; then one NEON chunk (alone {neon_alone:.2} µs) +{:.2} µs; then a 4 KiB memcpy +{:.2} µs", scalar - alone, neon - alone, memcpy - alone));
    report.line(format!("  first-NEON penalty after SME2 ≈ {:.2} µs", neon - alone - neon_alone));
}

#[cfg(not(blake3_sme2))]
fn transition(report: &mut Report) {
    report.line("  this build has no SME2 kernel; nothing to measure".to_owned());
}

fn main() {
    let mut report = Report(String::new());
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    report.line(format!(
        "host_lab · fork {}{} · {} · {} CPUs · kernel platform {}",
        commit.trim(),
        if dirty { " (dirty)" } else { "" },
        std::env::consts::OS,
        cpus,
        blake3_servil::kernel_report().platform
    ));

    report.line("1. primitives".to_owned());
    let n = 100_000;
    let t = Instant::now();
    for _ in 0..n {
        std::thread::yield_now();
    }
    report.line(format!("  sched_yield on an idle CPU      {:7.0} ns", t.elapsed().as_nanos() as f64 / n as f64));
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(Instant::now());
    }
    report.line(format!("  Instant::now                    {:7.0} ns", t.elapsed().as_nanos() as f64 / n as f64));
    for style in [Wait::Spin, Wait::Yield, Wait::Condvar] {
        report.line(format!("  hand-off to a {:>8} waiter    {:7.0} ns", format!("{style:?}"), hand_off(style)));
    }
    {
        let pair = Arc::new((Mutex::new(0u64), Condvar::new()));
        let p2 = pair.clone();
        let reps = 2000u64;
        let sleeper = std::thread::spawn(move || {
            let mut g = p2.0.lock().unwrap();
            let mut seen = 0;
            while seen < reps {
                while *g == seen {
                    g = p2.1.wait(g).unwrap();
                }
                seen = *g;
            }
        });
        let mut waker = Duration::ZERO;
        for k in 0..reps {
            busy(Duration::from_micros(50));
            let t = Instant::now();
            let mut g = pair.0.lock().unwrap();
            *g = k + 1;
            pair.1.notify_one();
            drop(g);
            waker += t.elapsed();
        }
        sleeper.join().unwrap();
        report.line(format!("  notify_one, cost to the waker   {:7.0} ns", waker.as_nanos() as f64 / reps as f64));
    }

    report.line("2. scaling: n threads hashing at once, ns per hash per thread (median / slowest)".to_owned());
    for len in [8 << 10, 256 << 10] {
        let mut line = format!("  {:>4} KiB:", len >> 10);
        let mut n = 1;
        while n <= cpus {
            let (med, max) = hashers_beside(len, n, 0, None);
            write!(line, "  n={n} {med:.0}/{max:.0}").unwrap();
            n = if n < 4 { n * 2 } else { n + 4.min(cpus - n).max(1) };
        }
        report.line(line);
    }

    report.line("3. idle waiters: 8 KiB hashes, ns per hash per thread (median / slowest)".to_owned());
    let half = (cpus / 2).max(1);
    for (hashers, idlers) in [(half, cpus - half), (cpus.saturating_sub(2).max(1), 2.min(cpus - 1))] {
        let mut line = format!("  {hashers} hashers, {idlers} idle:");
        let (m, x) = hashers_beside(8 << 10, hashers, idlers, None);
        write!(line, "  none {m:.0}/{x:.0}").unwrap();
        for style in [Wait::Spin, Wait::Yield, Wait::Condvar] {
            let (m, x) = hashers_beside(8 << 10, hashers, idlers, Some(style));
            write!(line, "  {style:?} {m:.0}/{x:.0}").unwrap();
        }
        report.line(line);
    }

    report.line("4. SME2 → NEON on one thread".to_owned());
    transition(&mut report);

    report.line("5. the pool: ns per call (µs), serial vs multithreaded, one caller / two callers at once".to_owned());
    blake3_servil::initialize();
    for len in [64usize << 10, 128 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20] {
        let (s1, m1, s2, m2) = (per_call(len, 1, tree_serial), per_call(len, 1, tree_mt), per_call(len, 2, tree_serial), per_call(len, 2, tree_mt));
        report.line(format!(
            "  hash {:>6} KiB: serial {:8.1}  mt {:8.1}  | two callers: serial {:8.1}  mt {:8.1}   (mt/serial {:.2}, {:.2})",
            len >> 10, s1 / 1e3, m1 / 1e3, s2 / 1e3, m2 / 1e3, m1 / s1, m2 / s2
        ));
    }
    for messages in [512usize, 1024, 2048, 4096, 16384, 65536, 262144] {
        let len = messages * 64;
        let (s1, m1, s2, m2) = (per_call(len, 1, batch_serial), per_call(len, 1, batch_mt), per_call(len, 2, batch_serial), per_call(len, 2, batch_mt));
        report.line(format!(
            "  hash_many {messages:>6} × 64 B: serial {:8.1}  mt {:8.1}  | two callers: serial {:8.1}  mt {:8.1}   (mt/serial {:.2}, {:.2})",
            s1 / 1e3, m1 / 1e3, s2 / 1e3, m2 / 1e3, m1 / s1, m2 / s2
        ));
    }

    report.line("6. two SME2 callers at once: per-thread time, two at once over one alone (x2.0 = sharing one SME unit)".to_owned());
    for (name, f) in [("hash_many 4096 x 64 B", batch_serial as fn(&[u8])), ("hash 1 MiB", tree_serial)] {
        let len = if name.starts_with("hash_many") { 4096 * 64 } else { 1 << 20 };
        let alone = per_call(len, 1, f);
        let mut quiet = Vec::new();
        let mut after_burst = Vec::new();
        for _ in 0..6 {
            quiet.push(per_call(len, 2, f) / alone);
            // Every core busy for 50 ms, as a multithreaded contender would
            // leave it; then two callers at once, straight away.
            let burst: Vec<_> = (0..cpus)
                .map(|_| std::thread::spawn(|| busy(Duration::from_millis(50))))
                .collect();
            for t in burst {
                t.join().unwrap();
            }
            after_burst.push(per_call(len, 2, f) / alone);
        }
        let show = |v: &[f64]| v.iter().map(|r| format!("x{r:.2}")).collect::<Vec<_>>().join(" ");
        report.line(format!("  {name}: alone {:.1} µs; two at once, quiet: {}; right after an all-core burst: {}", alone / 1e3, show(&quiet), show(&after_burst)));
    }

    #[cfg(target_arch = "aarch64")]
    {
        report.line("7. WFE".to_owned());
        report.line(format!("  hand-off to a WFE waiter        {:7.0} ns", hand_off(Wait::Wfe)));
        let word = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let (w, s) = (word.clone(), stop.clone());
            std::thread::spawn(move || {
                while !s.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(20));
                    w.fetch_add(1, Ordering::SeqCst);
                }
            })
        };
        let t = Instant::now();
        let mut returns = 0u64;
        while t.elapsed() < Duration::from_millis(200) {
            let v: u64;
            unsafe {
                std::arch::asm!("ldaxr {v}, [{a}]", "wfe", v = out(reg) v, a = in(reg) word.as_ptr(), options(nostack));
            }
            std::hint::black_box(v);
            returns += 1;
        }
        stop.store(true, Ordering::SeqCst);
        ticker.join().unwrap();
        report.line(format!(
            "  WFE with no writer returns every {:.1} µs (a writer ticks every 20 ms); a real idle would read near 20000",
            200_000.0 / returns as f64
        ));
        let mut line = format!("  {half} hashers beside {} WFE waiters:", cpus - half);
        let (m, x) = hashers_beside(8 << 10, half, cpus - half, Some(Wait::Wfe));
        write!(line, " {m:.0}/{x:.0} ns per 8 KiB hash").unwrap();
        report.line(line);
    }

    let path = format!(
        "host-lab.{}.txt",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    );
    std::fs::write(&path, &report.0).unwrap();
    println!("report written to {path}");
}
