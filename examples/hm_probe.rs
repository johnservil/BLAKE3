//! hash_many of equal-length messages over a chunk: ns per message for each length and count (clocks::measure 2 ms batches, clocks::summary::mean). Run old and new in alternating processes.
fn main() {
    blake3_servil::initialize();
    for len in [1500usize, 2048, 3000, 3839, 4096, 6000, 7935, 8192, 12000, 15000] {
        for count in [4usize, 6, 8, 16, 64] {
            let slot = len.div_ceil(64) * 64;
            let mut input = vec![0u8; slot * count];
            for (i, b) in input.iter_mut().enumerate() {
                if i % slot < len {
                    *b = (i % 251) as u8;
                }
            }
            let mut out = vec![[0u8; 32]; count];
            let b = clocks::measure(32, 2_000_000, || blake3_servil::hash_many(std::hint::black_box(&input), len, &mut out));
            let m = clocks::summary::mean(b.iter().map(|x| (x.wall_ns, x.calls * count as u64)));
            println!("{len}x{count}\t{}", ((m * 10 + (1 << 63)) >> 64) as f64 / 10.0);
        }
    }
}
