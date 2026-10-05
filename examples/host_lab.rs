//! probe/no-q: nonstop hash() at lengths with a partial last chunk, on
//! the fork as built and without its partial-chunk (q) kernels.
fn main() {
    blake3_servil::initialize();
    let input = vec![5u8; 1 << 16];
    for round in 0..2 {
        let mut line = format!("round {round}:");
        for len in [1100usize, 2304, 3839, 4470, 6000, 7935, 10_000, 15_000, 17_000, 30_000, 60_000] {
            let b = clocks::measure(5, 100_000_000, || { std::hint::black_box(blake3_servil::hash(std::hint::black_box(&input[..len]))); });
            let c: u64 = b.iter().map(|b| b.calls).sum(); let w: u64 = b.iter().map(|b| b.wall_ns).sum();
            line += &format!(" {len}:{}", w / c);
        }
        println!("{line} (ns per call)");
    }
}
