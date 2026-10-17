//! What macOS's per-process energy estimate (clocks::energy::process_nj)
//! sees: a sleep, plain integer work on one core, and hashing 1 GiB on one
//! thread and on the pool. Each line: wall time, CPU time, energy, the
//! P-cores' share, watts per busy core (energy over CPU time), and energy
//! per byte. If one thread's hashing on SME2 (the default build) draws no
//! more per busy core than integer work, the SME unit's power is probably
//! outside the estimate; compare with `--features no_sme2` (NEON).
//!
//!     cargo run --release --example energy_probe
const LEN: usize = 256 << 20;
const ROUNDS: usize = 4;

fn report(what: &str, bytes: usize, work: impl FnOnce()) {
    let Some((e0, p0)) = clocks::energy::process_nj() else {
        println!("{what}: no energy estimate on this platform");
        return;
    };
    let cpu0 = clocks::process_cpu_ns();
    let started = clocks::now();
    work();
    let wall = clocks::since_ns(started);
    let cpu = clocks::process_cpu_ns() - cpu0;
    let (e1, p1) = clocks::energy::process_nj().unwrap();
    let (energy, p) = (e1 - e0, p1 - p0);
    let permille = |a: u64, b: u64| if b == 0 { 0 } else { (a * 1000 + b / 2) / b };
    let watts_milli = if cpu == 0 { 0 } else { (u128::from(energy) * 1000 / u128::from(cpu)) as u64 };
    let per_byte = if bytes == 0 { String::new() } else { format!(", {} pJ/byte", (u128::from(energy) * 1000 / bytes as u128)) };
    println!("{what:<34} wall {:>6} ms  cpu {:>6} ms  energy {:>7} mJ  P-cores {:>4} permille  {:>5} mW a busy core{per_byte}",
        wall / 1_000_000, cpu / 1_000_000, energy / 1_000_000, permille(p, energy), watts_milli);
}

fn main() {
    let input: Vec<u8> = (0..LEN).map(|i| (i * 31) as u8).collect();
    blake3_servil::initialize_multithreaded();
    std::hint::black_box(blake3_servil::hash(&input[..1 << 20]));
    report("sleep 1 s", 0, || std::thread::sleep(std::time::Duration::from_secs(1)));
    report("integer work, one core, 1 s", 0, || clocks::busy_work(1_000_000_000));
    report("hash 1 GiB, one thread", LEN * ROUNDS, || {
        for _ in 0..ROUNDS {
            std::hint::black_box(blake3_servil::hash(std::hint::black_box(&input)));
        }
    });
    report("hash_multithreaded 1 GiB", LEN * ROUNDS, || {
        for _ in 0..ROUNDS {
            std::hint::black_box(blake3_servil::hash_multithreaded(std::hint::black_box(&input)));
        }
    });
}
