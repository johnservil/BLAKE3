//! probe/xof: OutputReader::fill after hashing a short key, 256 B to 1 MiB
//! of output; ns per byte, speeds. Built before (servil 099a292) and after
//! (18d1b3e: the SME2 extended-output kernel; 4e91fa0: NEON eight blocks at a time).
use std::hint::black_box;
use std::io::Write;

fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{:.4}{}", (s.median as f64) / 2f64.powi(64), if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}

fn main() {
    blake3_servil::initialize();
    let mut report = String::from("probe/xof: OutputReader::fill, ns/B\n");
    for n in [256usize, 1024, 4096, 65536, 1 << 20] {
        let mut out = vec![0u8; n];
        let mut v = Vec::new();
        for b in clocks::measure(15, 2_000_000, || {
            let mut r = blake3_servil::Hasher::new().update(b"key material").finalize_xof();
            r.fill(black_box(&mut out));
        }) {
            v.push(clocks::speeds::per_unit(b.wall_ns, b.calls * n as u64));
        }
        report += &format!("  {n:8} B: {}\n", show(&mut v));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
