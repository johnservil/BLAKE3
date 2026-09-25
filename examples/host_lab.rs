//! Probe (never merged): hash_multithreaded_with_budget at 1-64 MiB and
//! budgets 2, 4, 8, and all, fastest of 7 batches, and the caller's
//! cycles per ns.
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;
fn main() {
    clocks::set_qos(clocks::USER_INTERACTIVE);
    blake3_servil::initialize();
    let cpus = std::thread::available_parallelism().unwrap().get();
    let input: Vec<u8> = (0..64usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    println!("probe/budgets: platform {}", blake3_servil::kernel_report().platform);
    for round in 0..2 {
        for mib in [1usize, 2, 8, 64] {
            let data = &input[..mib << 20];
            let mut line = format!("round {round} {mib:>2} MiB:");
            for b in [1usize, 2, 4, 8, cpus] {
                let s = clocks::measure(7, 20_000, || { black_box(blake3_servil::hash_multithreaded_with_budget(black_box(data), b)); }).fastest();
                line += &format!("  b{b} {:.4}", s.ns / data.len() as f64);
            }
            println!("{line}");
        }
    }
}
