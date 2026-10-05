//! probe/sme2-worth: what SME2 buys, nonstop. The same calls on the fork as
//! built (probe/sme2-worth-with) and with build.rs's no_sme2 forced
//! (probe/sme2-worth-without); each line ns per byte (or per message).
fn per(b: &[clocks::Batch], units: u64) -> u64 { let c: u64 = b.iter().map(|b| b.calls).sum(); let w: u64 = b.iter().map(|b| b.wall_ns).sum(); w * 10_000 / (c * units) }
fn main() {
    blake3_servil::initialize_multithreaded();
    let input: Vec<u8> = (0..(64u32 << 20)).map(|i| i as u8).collect();
    let mut out = vec![[0u8; 32]; 65536];
    for round in 0..2 {
        let mut line = format!("round {round}:");
        for len in [16usize << 10, 64 << 10, 1 << 20, 64 << 20] {
            let b = clocks::measure(5, 200_000_000, || { std::hint::black_box(blake3_servil::hash(&input[..len])); });
            line += &format!(" hash {}K {}", len >> 10, per(&b, len as u64));
        }
        for len in [1usize << 20, 64 << 20] {
            let b = clocks::measure(5, 200_000_000, || { std::hint::black_box(blake3_servil::hash_multithreaded(&input[..len])); });
            line += &format!(" mt {}K {}", len >> 10, per(&b, len as u64));
        }
        for n in [16usize, 64, 1024, 65536] {
            let b = clocks::measure(5, 200_000_000, || blake3_servil::hash_many(&input[..64 * n], 64, &mut out[..n]));
            line += &format!(" many{n} {}", per(&b, n as u64));
        }
        let b = clocks::measure(5, 200_000_000, || { std::hint::black_box(blake3_servil::outboard_with(blake3_servil::Mode::Hash, &input[..64 << 20])); });
        line += &format!(" outboard64M {}", per(&b, 64 << 20));
        let b = clocks::measure(5, 200_000_000, || { let mut h = blake3_servil::Hasher::new(); for p in input.chunks(64 << 10) { h.update(p); } std::hint::black_box(h.finalize()); });
        line += &format!(" stream64M {}", per(&b, 64 << 20));
        println!("{line}   (ten-thousandths of a ns)");
    }
}
