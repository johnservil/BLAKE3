//! Equal-length batch probe: hash_many at 128 B-4 KiB messages. Times via
//! clocks::measure (2 ms batches, producer outside timing); summaries and
//! verdicts via clocks::summary. `run OUT` writes ns/units per cell;
//! `compare OLD... -- NEW...` pairs runs in order and prints each cell's
//! median ratio and sign-test verdict.
use std::fmt::Write as _;

const LENGTHS: [usize; 6] = [128, 256, 512, 1024, 2048, 4096];
const COUNTS: [usize; 3] = [6, 16, 64];

fn run(out: &str) {
    let mut text = String::from("# x86 batch probe v1 (clocks::summary mean)\n");
    for len in LENGTHS {
        for count in COUNTS {
            let input: Vec<u8> = (0..len * count).map(|i| (i % 251) as u8).collect();
            let mut digests = vec![[0u8; 32]; count];
            // Reference anchor: batch digests equal one-shot hashes.
            blake3_servil::hash_many(&input, len, &mut digests);
            for (i, d) in digests.iter().enumerate() {
                assert_eq!(d, blake3_servil::hash(&input[i * len..][..len]).as_bytes());
            }
            let batches = clocks::measure(32, 2_000_000, || {
                blake3_servil::hash_many(std::hint::black_box(&input), len, &mut digests);
                std::hint::black_box(&digests);
            });
            let samples: Vec<String> = batches.iter().map(|b| format!("{}/{}", b.wall_ns, b.calls * count as u64)).collect();
            writeln!(text, "{len}x{count}\t{}", samples.join(",")).unwrap();
        }
    }
    let windows = clocks::load::windows();
    writeln!(text, "# load: {}", clocks::load::describe(&windows)).unwrap();
    std::fs::write(out, text).unwrap();
}

fn read(path: &str) -> (Vec<(String, u128)>, String) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("# x86 batch probe v1 (clocks::summary mean)"), "{path}: format");
    let mut cells = Vec::new();
    let mut load = String::new();
    for line in lines {
        if let Some(l) = line.strip_prefix("# load: ") {
            load = l.to_owned();
            continue;
        }
        let (key, samples) = line.split_once('\t').unwrap();
        let mean = clocks::summary::mean(samples.split(',').map(|s| {
            let (ns, units) = s.split_once('/').unwrap();
            (ns.parse().unwrap(), units.parse().unwrap())
        }));
        cells.push((key.to_owned(), mean));
    }
    (cells, load)
}

fn compare(args: &[String]) {
    let split = args.iter().position(|a| a == "--").expect("compare OLD... -- NEW...");
    let (old, new) = (&args[..split], &args[split + 1..]);
    assert_eq!(old.len(), new.len(), "runs pair in order");
    let olds: Vec<_> = old.iter().map(|p| read(p)).collect();
    let news: Vec<_> = new.iter().map(|p| read(p)).collect();
    for (_, l) in olds.iter().chain(&news) {
        if !l.starts_with("quiet") {
            println!("qualified: a run's load was {l}");
        }
    }
    for (i, (key, _)) in olds[0].0.iter().enumerate() {
        let ratios: Vec<u64> = olds.iter().zip(&news).map(|(o, n)| clocks::summary::ratio_permille(n.0[i].1, o.0[i].1)).collect();
        let median = clocks::summary::median_permille(&ratios);
        let v = clocks::summary::verdict(&ratios, 30);
        println!("{key}: new/old median {}.{:03} {v:?} ratios {ratios:?}", median / 1000, median % 1000);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1]),
        Some("compare") => compare(&args[1..]),
        _ => panic!("x86_batch_probe run OUT | compare OLD... -- NEW..."),
    }
}
