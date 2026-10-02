//! probe/reader-buffer: Hasher::update_reader over files in the page cache
//! (written here, then read once untimed), 100 KB to 64 MiB; ns per byte,
//! speeds. Built before (servil 1d2b652) and after (a39fb7c: a 1 MiB
//! buffer once a reader fills 64 KiB).
use std::hint::black_box;
use std::io::Write;

fn show(v: &mut Vec<u128>) -> String {
    v.sort_unstable();
    let n = v.len();
    clocks::speeds::speeds(v).into_iter().map(|s| format!("{:.4}{}", (s.median as f64) / 2f64.powi(64), if s.count < n { format!("({}%)", s.count * 100 / n) } else { String::new() })).collect::<Vec<_>>().join("|")
}

fn main() {
    blake3_servil::initialize();
    let data: Vec<u8> = (0..64usize << 20).map(|i| (i * 13 + (i >> 11)) as u8).collect();
    let mut report = String::from("probe/reader-buffer: update_reader, ns/B\n");
    for len in [100_000usize, 1 << 20, 8 << 20, 64 << 20] {
        let path = format!("reader-{len}.bin");
        std::fs::write(&path, &data[..len]).unwrap();
        let expect = blake3_servil::hash(&data[..len]);
        let mut v = Vec::new();
        for i in 0..13 {
            let t = clocks::now();
            let mut h = blake3_servil::Hasher::new();
            h.update_reader(std::fs::File::open(&path).unwrap()).unwrap();
            let got = h.finalize();
            let ns = clocks::since_ns(t);
            assert_eq!(got, expect);
            if i > 0 {
                v.push(clocks::speeds::per_unit(ns, len as u64));
            }
            black_box(got);
        }
        std::fs::remove_file(&path).unwrap();
        report += &format!("  {len:9} B: {}\n", show(&mut v));
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
