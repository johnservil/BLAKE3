//! probe/cold-split2: hash() after other work (the benchmark's gap), with
//! the untimed preparation warming nothing, what a 64-byte hash touches
//! (the crate's shared state and the one-chunk kernel), or everything (a
//! hash of the same length): what a cold call of 2-8 KiB waits for.
fn main() {
    blake3_servil::initialize();
    let work: Vec<u8> = (0..(128u32 << 20)).map(|i| i as u8).collect();
    let scratch = vec![3u8; 8192];
    for round in 0..2 {
        for len in [64usize, 2048, 4096, 8192] {
            let mut input = vec![0u8; len];
            let mut t = [0u64; 3];
            for (k, warm) in [0usize, 64, len].into_iter().enumerate() {
                let m = clocks::measure_after_gaps_prepared(200, clocks::Gap::Busy(&work), 1_000_000, &mut input,
                    |i| { i.fill(round as u8 + 1); if warm > 0 { std::hint::black_box(blake3_servil::hash(std::hint::black_box(&scratch[..warm]))); } },
                    |i| { std::hint::black_box(blake3_servil::hash(std::hint::black_box(i))); });
                t[k] = m.calls.wall_ns / 200;
            }
            println!("round {round} {len:5} B: cold {:5} ns, shared state warm {:5} ns, all warm {:5} ns", t[0], t[1], t[2]);
        }
    }
}
