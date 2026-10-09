//! A Hasher's whole life on a short message: new, update(64 bytes),
//! finalize. Wall time per call (and cycles, where the platform reports
//! them), a run's mean over its batches, through `clocks`.
fn main() {
    let n: usize = std::env::args().nth(1).map_or(64, |a| a.parse().unwrap());
    let input = vec![7u8; n];
    blake3_servil::initialize();
    let batches = clocks::measure(64, 20_000_000, || {
        let mut h = blake3_servil::Hasher::new();
        h.update(core::hint::black_box(&input));
        core::hint::black_box(h.finalize());
    });
    let ns = clocks::summary::mean(batches.iter().map(|b| (b.wall_ns, b.calls)));
    let cycles: Option<(u64, u64)> = batches.iter().map(|b| b.counts.map(|c| (c.p.cycles, c.e.cycles)))
        .try_fold((0, 0), |(p, e), x| x.map(|(xp, xe)| (p + xp, e + xe)));
    let calls: u64 = batches.iter().map(|b| b.calls).sum();
    let whole = ns >> 64;
    let frac = ((ns & ((1u128 << 64) - 1)) * 100) >> 64;
    match cycles {
        Some((p, e)) => println!("{whole}.{frac:02} ns per call, {} P-core and {} E-core cycles per call",
            (p + calls / 2) / calls, (e + calls / 2) / calls),
        None => println!("{whole}.{frac:02} ns per call (this platform reports no per-thread cycle counts)"),
    }
}
