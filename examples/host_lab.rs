//! Probe (branch probe/mac-load): which part of "the machine's busy time
//! less this process's own" misbooks our own work as other programs'
//! load on macOS (BLAKE3 #142: every busy window of Mac jobs 1672-1676 sat
//! where our threads burst or woke). Hypothesis: the kernel's work on our
//! behalf (wakes, interrupts, IPIs) is system time charged to no process,
//! while other programs that disturb a benchmark burn user time; so the
//! machine's user ticks less our user time would read our own bursts as
//! none and a real other program as itself.
//!
//! Run 2 (the first run's idle, wakes, nonstop, spawn and spinner read as
//! expected: our own wakes and spawns stayed near the machine's idle 0.3
//! CPUs, a spinner read 1.0): the two places the checks' busy windows
//! fell (jobs 1672-1676). Calls after pauses, as the after-idling phase
//! makes them; and a freshly linked program's first launch, which the OS
//! may scan. WINDOWS windows of 1 s each, milli-CPUs: the machine's
//! user+nice and system, ours (self and reaped children), and the
//! residuals: total (today's rule), user, system.
use std::time::{Duration, Instant};

const WINDOWS: usize = 4;

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
    let small = vec![7u8; 4096];
    let mut sink = 0u8;
    let mut mt = || {
        let h = blake3_servil::performance::hash_multithreaded(blake3_servil::Mode::Plain, &message, len);
        sink ^= h.as_bytes()[0];
    };
    println!("# probe/mac-load 2: milli-CPUs per 1 s window; ticks/s {}", unsafe { sysconf(3) });
    window("idle", || std::thread::sleep(Duration::from_millis(50)));
    // Calls after pauses, as the after-idling phase makes them.
    window("mt after 2 ms", || {
        mt();
        std::thread::sleep(Duration::from_millis(2));
    });
    window("mt after 20 ms", || {
        mt();
        std::thread::sleep(Duration::from_millis(20));
    });
    window("st after 1 ms", || {
        let h = blake3_servil::convenience::hash(&small);
        sink ^= h.as_bytes()[0];
        std::thread::sleep(Duration::from_millis(1));
    });
    // A freshly linked program's first launch: does the OS's scan of a new
    // executable read as other programs' load, and for how long?
    for round in 0..3 {
        let dir = std::env::temp_dir().join(format!("mac-load-fresh-{}-{round}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("fresh.c");
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::fs::write(&src, format!("int main(void) {{ volatile unsigned long x = {nonce}UL; return (int)(x & 0); }}\n")).unwrap();
        let exe = dir.join("fresh");
        assert!(std::process::Command::new("cc").arg("-O1").arg("-o").arg(&exe).arg(&src).status().expect("cc").success());
        window(&format!("after compile {round}"), || std::thread::sleep(Duration::from_millis(50)));
        let t = Instant::now();
        assert!(std::process::Command::new(&exe).status().expect("fresh runs").success());
        println!("# fresh launch {round}: {} us", t.elapsed().as_micros());
        window(&format!("after fresh launch {round}"), || std::thread::sleep(Duration::from_millis(50)));
    }
    println!("# done {sink}");
}
