//! probe/rayon-vs-pool: b3sum hashes files through Hasher::update_mmap_rayon
//! (update_rayon, Rayon's join over the tree); the fork's own pool serves
//! hash_multithreaded and update_multithreaded. The same inputs both ways,
//! in memory, back to back; ns per byte, speeds. (The probe branch makes
//! `rayon` a default feature: the runner's example jobs take no others.)
use std::hint::black_box;
use std::io::Write;

fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{:.4}{}", (s.median as f64) / 2f64.powi(64), if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}

fn main() {
    blake3_servil::initialize_multithreaded();
    let big = vec![3u8; 1 << 30];
    let mut report = String::from("probe/rayon-vs-pool: ns/B\n");
    for len in [1usize << 20, 16 << 20, 128 << 20, 1 << 30] {
        let input = &big[..len];
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..9 {
            let t = clocks::now(); let mut h = blake3_servil::Hasher::new(); h.update_rayon(black_box(input)); black_box(h.finalize()); a.push(clocks::speeds::per_unit(clocks::since_ns(t), len as u64));
            let t = clocks::now(); black_box(blake3_servil::hash_multithreaded(black_box(input))); b.push(clocks::speeds::per_unit(clocks::since_ns(t), len as u64));
            let t = clocks::now(); let mut h = blake3_servil::Hasher::new(); h.update_multithreaded(black_box(input)); black_box(h.finalize()); c.push(clocks::speeds::per_unit(clocks::since_ns(t), len as u64));
        }
        report += &format!("{:5} MiB: update_rayon {}  hash_multithreaded {}  update_multithreaded {}\n", len >> 20, show(&mut a), show(&mut b), show(&mut c));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
