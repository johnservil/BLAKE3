//! probe/rayon-vs-pool, round two: update_rayon (b3sum's path) hashing on
//! NEON on every Rayon thread from B3_RAYON_NEON_FROM bytes (never: every
//! thread on SME2, as servil; 0; 2, 4, 8 MiB), each setting in a process of
//! its own, beside the fork's pool (update_multithreaded). In memory, back
//! to back, and after 1 ms asleep (a file hashed now and then); ns/B.
use std::hint::black_box;
use std::io::Write;

fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{:.4}{}", (s.median as f64) / 2f64.powi(64), if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}

fn child() -> String {
    blake3_servil::initialize_multithreaded();
    let big = vec![3u8; 64 << 20];
    let mut out = String::new();
    for len in [1usize << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20, 64 << 20] {
        let input = &big[..len];
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..9 {
            let t = clocks::now(); let mut h = blake3_servil::Hasher::new(); h.update_rayon(black_box(input)); black_box(h.finalize()); a.push(clocks::speeds::per_unit(clocks::since_ns(t), len as u64));
            std::thread::sleep(std::time::Duration::from_millis(1));
            let t = clocks::now(); let mut h = blake3_servil::Hasher::new(); h.update_rayon(black_box(input)); black_box(h.finalize()); b.push(clocks::speeds::per_unit(clocks::since_ns(t), len as u64));
            let t = clocks::now(); let mut h = blake3_servil::Hasher::new(); h.update_multithreaded(black_box(input)); black_box(h.finalize()); c.push(clocks::speeds::per_unit(clocks::since_ns(t), len as u64));
        }
        out += &format!("  {:3} MiB: update_rayon {}  after 1 ms asleep {}  (pool {})\n", len >> 20, show(&mut a), show(&mut b), show(&mut c));
    }
    out
}

fn main() {
    if std::env::var("B3_RAYON_NEON_FROM").is_ok() {
        print!("{}", child());
        return;
    }
    let mut report = String::from("probe/rayon-vs-pool round two: ns/B\n");
    for (name, from) in [("never (servil)", usize::MAX), ("0", 0), ("2 MiB", 2 << 20), ("4 MiB", 4 << 20), ("8 MiB", 8 << 20)] {
        let output = std::process::Command::new(std::env::current_exe().unwrap()).env("B3_RAYON_NEON_FROM", from.to_string()).output().unwrap();
        report += &format!("NEON on every Rayon thread from {name}:\n{}", String::from_utf8_lossy(&output.stdout));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
