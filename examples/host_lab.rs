//! probe/tail-pad: batches whose count leaves 1-15 messages past whole
//! groups of sixteen, with the last group padded from B3_PAD_AFTER left
//! over (5 as on servil; 1 pads every tail; 3 between). Each setting in a
//! process of its own (this program runs itself). Two patterns: back to
//! back, and with the program's own work between calls (a read of every
//! digest and 2 us of integer work). ns per message, speeds.
use std::hint::black_box;
use std::io::Write;

fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{:.1}{}", (s.median as f64) / 2f64.powi(64), if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}

fn child() -> String {
    blake3_servil::initialize();
    let mut out = String::new();
    for len in [64usize, 256, 1024] {
        let slot = len.next_multiple_of(64);
        for gap in [false, true] {
            let mut line = format!("  {len:4} B {}:", if gap { "work between" } else { "back to back" });
            for n in [16usize, 17, 18, 19, 20, 22, 24, 32, 33, 34, 36, 40, 48, 49, 52, 100, 129] {
                let input: Vec<u8> = (0..n * slot).map(|i| if i % slot < len { (i * 7 + 1) as u8 } else { 0 }).collect();
                let mut digests = vec![[0u8; 32]; n];
                let mut v = Vec::new();
                for b in clocks::measure(15, 1_000_000, || {
                    blake3_servil::hash_many(black_box(&input), len, &mut digests);
                    if gap {
                        let mut s = 0u64;
                        for d in &digests { s = s.wrapping_add(u64::from(d[0])); }
                        black_box(s);
                        clocks::busy_work(2000);
                    }
                }) {
                    v.push(clocks::speeds::per_unit(b.wall_ns, b.calls * n as u64));
                }
                line += &format!(" {n}:{}", show(&mut v));
            }
            out += &line;
            out.push('\n');
        }
    }
    out
}

fn main() {
    if std::env::var("B3_PAD_AFTER").is_ok() {
        print!("{}", child());
        return;
    }
    let mut report = String::from("probe/tail-pad: hash_many, ns per message (2 us of work inside the 'work between' figures)\n");
    for after in ["5", "3", "1", "5"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap()).env("B3_PAD_AFTER", after).output().unwrap();
        report += &format!("pad from {after} left over:\n{}", String::from_utf8_lossy(&output.stdout));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
