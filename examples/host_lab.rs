//! probe/neon-only-cold: hash() and a Hasher per message after the
//! benchmark's busy gap, 2-32 KiB, and nonstop; ns per call. Built
//! NEON-only (the job's no_sme2: the path of Apple M1-M3), before
//! (b132f8c) and after (ff8f203) the night's code prefetch.
use std::hint::black_box;
use std::io::Write;

fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{}{}", (s.median + (1 << 63)) >> 64, if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}

fn main() {
    blake3_servil::initialize();
    let work = vec![1u8; 128 << 20];
    let mut report = String::from("probe/neon-only-cold: ns/call\n");
    for len in [2048usize, 4096, 8192, 16384, 32768] {
        let source: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
        let mut input = vec![0u8; len];
        let (mut h, mut hh, mut warm) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..48 {
            let m = clocks::measure_after_gaps_prepared(4, clocks::Gap::Busy(&work), 1_000_000, &mut input, |i| i.copy_from_slice(black_box(&source)), |i| { black_box(blake3_servil::hash(black_box(i))); });
            h.push(clocks::speeds::per_unit(m.calls.wall_ns, 4));
            let m = clocks::measure_after_gaps_prepared(4, clocks::Gap::Busy(&work), 1_000_000, &mut input, |i| i.copy_from_slice(black_box(&source)), |i| { let mut x = blake3_servil::Hasher::new(); x.update(black_box(i)); black_box(x.finalize()); });
            hh.push(clocks::speeds::per_unit(m.calls.wall_ns, 4));
        }
        for b in clocks::measure(15, 2_000_000, || { black_box(blake3_servil::hash(black_box(&input))); }) {
            warm.push(clocks::speeds::per_unit(b.wall_ns, b.calls));
        }
        report += &format!("  {len:6} B: hash after other work {}  Hasher after other work {}  hash nonstop {}\n", show(&mut h), show(&mut hh), show(&mut warm));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
