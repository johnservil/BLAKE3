//! probe/hasher-prefetch: a Hasher per message after the benchmark's busy
//! gap (clocks::Gap::Busy, 128 MiB) and nonstop, 2-16 KiB; ns per call.
//! Built on servil 3a678cf (before) and 7510d44 (a fresh Hasher's first
//! update prefetches its code after a pause).
use std::hint::black_box;
use std::io::Write;
fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{}{}", (s.median + (1 << 63)) >> 64, if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}
fn main() {
    blake3_servil::initialize();
    let mut report = String::new();
    let work = vec![1u8; 128 << 20];
    for len in [2048usize, 4096, 5000, 8192, 16384] {
        let source: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
        let mut input = vec![0u8; len];
        let (mut cold, mut warm) = (Vec::new(), Vec::new());
        for _ in 0..48 {
            let m = clocks::measure_after_gaps_prepared(4, clocks::Gap::Busy(&work), 1_000_000, &mut input, |i| i.copy_from_slice(black_box(&source)), |i| {
                let mut h = blake3_servil::Hasher::new();
                h.update(black_box(i));
                black_box(h.finalize());
            });
            cold.push(clocks::speeds::per_unit(m.calls.wall_ns, 4));
        }
        for b in clocks::measure(15, 2_000_000, || { let mut h = blake3_servil::Hasher::new(); h.update(black_box(&input)); black_box(h.finalize()); }) {
            warm.push(clocks::speeds::per_unit(b.wall_ns, b.calls));
        }
        report += &format!("Hasher {len:5} B: after other work {} ns, nonstop {} ns\n", show(&mut cold), show(&mut warm));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
