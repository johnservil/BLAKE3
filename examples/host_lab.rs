//! probe/cold-code-share: how much of a one-shot call after other work is
//! its code being cold? Each call follows the benchmark's busy gap
//! (clocks::Gap::Busy over 128 MiB, 1 ms), then the producer's write of
//! the input. "cold": the call as the benchmark times it. "warm code": the
//! preparation (untimed) also hashes a different buffer of the same length
//! first, so the call meets its code, glue, and branch history warm and
//! only its data as usual. The difference bounds what a compact path for
//! calls after a pause could recover. Rounds alternate the variants.
use clocks::{Gap, measure_after_gaps_prepared};

fn main() {
    let work = vec![1u8; 128 << 20];
    let lens = [2048usize, 3072, 4096, 8192];
    // The length whose call warms the glue but not this length's kernels.
    let glue_len = |len: usize| if len == 2048 { 3072 } else { 2048 };
    let rounds = 12;
    let calls = 20;
    let hashes: [(&str, fn(&[u8]) -> [u8; 32]); 1] = [
        ("servil", |m| *blake3_servil::hash(m).as_bytes()),
    ];
    for (name, h) in hashes {
    let mut sums = vec![[(0u64, 0u64, 0u64); 3]; lens.len()];
    for round in 0..rounds {
        for (li, &len) in lens.iter().enumerate() {
            for v in [round % 3, (round + 1) % 3, (round + 2) % 3] {
                let mut input = vec![0u8; len];
                let other = vec![3u8; if v == 2 { glue_len(len) } else { len }];
                let mut n = 0u8;
                let b = measure_after_gaps_prepared(calls, Gap::Busy(&work), 1_000_000, &mut input[..], |i: &mut [u8]| {
                    n = n.wrapping_add(1);
                    for (k, w) in i.iter_mut().enumerate() { *w = (k as u8) ^ n; }
                    if v >= 1 { std::hint::black_box(h(std::hint::black_box(&other))); clocks::busy_work(120_000); }
                }, |i: &[u8]| { std::hint::black_box(h(std::hint::black_box(i))); });
                let c = b.calls.counts.map_or(0, |c| c.p.cycles + c.e.cycles);
                let s = &mut sums[li][v];
                s.0 += b.calls.wall_ns; s.1 += c; s.2 += b.calls.calls;
            }
        }
    }
    println!("{name}");
    println!("len      cold ns   glue-warm ns   all-warm ns   (cycles cold, glue warm, all warm)");
    for (li, &len) in lens.iter().enumerate() {
        let [c, w, g] = sums[li];
        println!("{len:6} {:10} {:13} {:13}   ({} {} {})", c.0 / c.2, g.0 / g.2, w.0 / w.2, c.1 / c.2, g.1 / g.2, w.1 / w.2);
    }
    }
}
