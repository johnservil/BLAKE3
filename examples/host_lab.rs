//! probe/sme2-workers: hash_multithreaded and hash_many_multithreaded with
//! the first K pool workers on SME2 (K = 0 to 3; each its own process).
fn main() {
    if let Ok(k) = std::env::var("BLAKE3_SME2_WORKERS") {
        blake3_servil::initialize_multithreaded();
        let input = vec![5u8; 64 << 20];
        let mut out = vec![[0u8; 32]; 1 << 20];
        let mut line = format!("K={k}:");
        for round in 0..2 {
            line.clear(); line += &format!("K={k} round {round}:");
            for len in [4usize << 20, 16 << 20, 64 << 20] {
                let b = clocks::measure(5, 300_000_000, || { std::hint::black_box(blake3_servil::hash_multithreaded(&input[..len])); });
                let c: u64 = b.iter().map(|b| b.calls).sum(); let w: u64 = b.iter().map(|b| b.wall_ns).sum();
                line += &format!(" mt{}M {}", len >> 20, w * 10_000 / (c * len as u64));
            }
            let n = 1 << 20;
            let b = clocks::measure(5, 300_000_000, || blake3_servil::hash_many_multithreaded(&input[..64 * n], 64, &mut out[..n]));
            let c: u64 = b.iter().map(|b| b.calls).sum(); let w: u64 = b.iter().map(|b| b.wall_ns).sum();
            line += &format!(" many1M {}", w * 1000 / (c * n as u64));
            println!("{line}   (ten-thousandths of ns per byte; many: thousandths of ns per message)");
        }
        return;
    }
    for k in [0, 1, 2, 3, 0] {
        let status = std::process::Command::new(std::env::current_exe().unwrap()).env("BLAKE3_SME2_WORKERS", k.to_string()).status().unwrap();
        assert!(status.success());
    }
}
