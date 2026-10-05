//! probe/cold-split: hash() of one message after other work (the
//! benchmark's gap: clocks::Gap::Busy over 128 MiB, 1 ms), as the
//! benchmark calls it, and with its code warmed in the untimed preparation
//! (a hash of a scratch buffer of the same length), against nonstop: how
//! much of a cold call is its code.
fn main() {
    blake3_servil::initialize();
    let work: Vec<u8> = (0..(128u32 << 20)).map(|i| i as u8).collect();
    let mut scratch = vec![3u8; 8192];
    for round in 0..2 {
        for len in [64usize, 1024, 2048, 2304, 4096, 8192] {
            let mut input = vec![0u8; len];
            let cold = clocks::measure_after_gaps_prepared(200, clocks::Gap::Busy(&work), 1_000_000, &mut input,
                |i| i.fill(round as u8 + 1), |i| { std::hint::black_box(blake3_servil::hash(std::hint::black_box(i))); });
            let warm = clocks::measure_after_gaps_prepared(200, clocks::Gap::Busy(&work), 1_000_000, &mut input,
                |i| { i.fill(round as u8 + 1); std::hint::black_box(blake3_servil::hash(std::hint::black_box(&scratch[..len]))); },
                |i| { std::hint::black_box(blake3_servil::hash(std::hint::black_box(i))); });
            let hot = clocks::measure(5, 100_000_000, || { std::hint::black_box(blake3_servil::hash(std::hint::black_box(&input[..]))); });
            let hc: u64 = hot.iter().map(|b| b.calls).sum(); let hw: u64 = hot.iter().map(|b| b.wall_ns).sum();
            scratch[0] ^= 1;
            println!("round {round} {len:5} B: after other work {:5} ns, code warmed {:5} ns, nonstop {:5} ns", cold.calls.wall_ns / 200, warm.calls.wall_ns / 200, hw / hc);
        }
    }
}
