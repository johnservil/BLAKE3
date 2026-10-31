//! Probe (branch probe/mac-load): which part of "the machine's busy time
//! less this process's own" misbooks our own work as other programs'
//! load on macOS (BLAKE3 #142: every busy window of Mac jobs 1672-1676 sat
//! where our threads burst or woke). Hypothesis: the kernel's work on our
//! behalf (wakes, interrupts, IPIs) is system time charged to no process,
//! while other programs that disturb a benchmark burn user time; so the
//! machine's user ticks less our user time would read our own bursts as
//! none and a real other program as itself.
//!
//! Conditions, WINDOWS windows of 1 s each:
//!   idle      this thread sleeps
//!   wakes     multithreaded one-shots of 1 MiB with 200 us sleeps between
//!             (the pool parks and wakes at every call)
//!   nonstop   multithreaded one-shots of 1 MiB back to back
//!   spawn     a process a call: /usr/bin/true, waited (reaped children
//!             count as ours)
//!   spinner   a shell loop (another program, never reaped until the end)
//!             while this thread sleeps
//!   wakes+spinner, spawn+spinner
//! Each window prints, in milli-CPUs: the machine's user+nice, system;
//! ours user, system (getrusage self + children); and the residuals:
//! total (today's rule), user, system.
use std::time::{Duration, Instant};

const WINDOWS: usize = 5;

#[repr(C)]
struct Rusage([i64; 18]);

unsafe extern "C" {
    fn mach_host_self() -> u32;
    fn host_statistics(host: u32, flavor: i32, info: *mut u32, count: *mut u32) -> i32;
    fn sysconf(name: i32) -> i64;
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}

/// user+nice, system ticks of the whole machine.
fn machine() -> (u64, u64) {
    let mut t = [0u32; 4];
    let mut n = 4u32;
    assert_eq!(unsafe { host_statistics(mach_host_self(), 3, t.as_mut_ptr(), &mut n) }, 0);
    (u64::from(t[0]) + u64::from(t[3]), u64::from(t[1]))
}

/// our user, system ns (self and reaped children).
fn ours() -> (u64, u64) {
    let tv = |r: &Rusage, i: usize| r.0[i] as u64 * 1_000_000_000 + (r.0[i + 1] as u64 & 0xffff_ffff) * 1000;
    let mut s = (0, 0);
    for who in [0, -1] {
        let mut r = Rusage([0; 18]);
        assert_eq!(unsafe { getrusage(who, &mut r) }, 0);
        s.0 += tv(&r, 0);
        s.1 += tv(&r, 2);
    }
    s
}

fn window(name: &str, mut work: impl FnMut()) {
    let tick_ns = 1_000_000_000 / unsafe { sysconf(3) } as u64;
    for w in 0..WINDOWS {
        let (mu0, ms0) = machine();
        let (ou0, os0) = ours();
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(1) {
            work();
        }
        let (mu1, ms1) = machine();
        let (ou1, os1) = ours();
        let ns = t0.elapsed().as_nanos() as u64;
        let m = |x: u64| x * 1000 / ns;
        let (mu, ms) = ((mu1 - mu0) * tick_ns, (ms1 - ms0) * tick_ns);
        let (ou, os) = (ou1 - ou0, os1 - os0);
        let res = |a: u64, b: u64| m(a) as i64 - m(b) as i64;
        println!(
            "{name}\t{w}\tmachine user {}\tsystem {}\tours user {}\tsystem {}\tresidual total {}\tuser {}\tsystem {}",
            m(mu), m(ms), m(ou), m(os), res(mu + ms, ou + os), res(mu, ou), res(ms, os)
        );
    }
}

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let message = vec![blake3_servil::performance::Block::ZERO; 1 << 14];
    let len = message.len() * 64;
    let mut sink = 0u8;
    let mut hash = || {
        let h = blake3_servil::performance::hash_multithreaded(blake3_servil::Mode::Plain, &message, len);
        sink ^= h.as_bytes()[0];
    };
    println!("# probe/mac-load: milli-CPUs per 1 s window; message {len} bytes; ticks/s {}", unsafe { sysconf(3) });
    window("idle", || std::thread::sleep(Duration::from_millis(50)));
    window("wakes", || {
        hash();
        std::thread::sleep(Duration::from_micros(200));
    });
    window("nonstop", &mut hash);
    let spawn = || {
        std::process::Command::new("/usr/bin/true").status().expect("true runs");
    };
    window("spawn", spawn);
    let mut spinner = std::process::Command::new("/bin/sh").args(["-c", "while :; do :; done"]).spawn().expect("sh");
    std::thread::sleep(Duration::from_millis(200));
    window("spinner", || std::thread::sleep(Duration::from_millis(50)));
    window("wakes+spinner", || {
        hash();
        std::thread::sleep(Duration::from_micros(200));
    });
    window("spawn+spinner", spawn);
    spinner.kill().expect("kill");
    spinner.wait().expect("reap");
    println!("# done {sink}");
}
