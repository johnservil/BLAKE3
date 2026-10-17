//! probe/sme-units: the pool's first K workers on SME2 (each perhaps its
//! own SME unit), and with `e` the first of them on the E-cores. Each
//! phase hashes back to back for 5 s after a 3 s rest; each line leads
//! with its start (Unix seconds), to line up with powermetrics.
//!
//!     cargo run --release --example sme_units -- K [e]
const PHASE_NS: u64 = 5_000_000_000;

fn phase(what: &str, bytes_a_call: usize, mut call: impl FnMut()) {
    std::thread::sleep(std::time::Duration::from_secs(3));
    let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let cpu0 = clocks::process_cpu_ns();
    let started = clocks::now();
    let mut calls = 0u64;
    while calls == 0 || clocks::since_ns(started) < PHASE_NS {
        call();
        calls += 1;
    }
    let wall = clocks::since_ns(started);
    let cpu = clocks::process_cpu_ns() - cpu0;
    let bytes = u128::from(calls) * bytes_a_call as u128;
    println!("{at} {what:<26} {calls:>6} calls  {:>6} MB/s  wall {:>5} ms  cpu {:>6} ms",
        bytes * 1000 / u128::from(wall), wall / 1_000_000, cpu / 1_000_000);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let k: usize = args.first().map_or(0, |a| a.parse().expect("K, the workers on SME2"));
    let e = args.get(1).is_some_and(|a| a == "e");
    // Sound: one thread so far, before the pool reads them.
    unsafe {
        std::env::set_var("BLAKE3_SME2_WORKERS", k.to_string());
        std::env::set_var("BLAKE3_SME2_E", if e { "1" } else { "0" });
    }
    blake3_servil::initialize_multithreaded();
    println!("K = {k} SME2 workers{}", if e { ", the first on the E-cores" } else { "" });
    let big: Vec<u8> = (0..64usize << 20).map(|i| (i * 31) as u8).collect();
    let mut out = vec![[0u8; 32]; 1 << 20];
    phase("hash_multithreaded 4 MiB", 4 << 20, || {
        std::hint::black_box(blake3_servil::hash_multithreaded(std::hint::black_box(&big[..4 << 20])));
    });
    phase("hash_multithreaded 64 MiB", 64 << 20, || {
        std::hint::black_box(blake3_servil::hash_multithreaded(std::hint::black_box(&big)));
    });
    phase("hash_many_mt 1M x 64 B", 64 << 20, || {
        blake3_servil::hash_many_multithreaded(std::hint::black_box(&big), 64, &mut out);
    });
}
