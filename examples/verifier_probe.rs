//! Verifier::update against update_multithreaded: a 64 MiB message's
//! encoding fed in pieces of PIECE bytes, nonstop (clocks::measure,
//! clocks::summary::mean), ns per message byte. `verifier_probe MODE PIECE`.
fn main() {
    blake3_servil::initialize_multithreaded(None);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (pooled, piece): (bool, usize) = (args[0] == "mt", args[1].parse().unwrap());
    let len = 64 << 20;
    let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let (hash, encoding) = (blake3_servil::hash(&input), blake3_servil::test_encoding(blake3_servil::Mode::Hash, &input));
    let mut out = vec![0u8; len];
    let batches = clocks::measure(16, 50_000_000, || {
        let mut v = blake3_servil::Verifier::new(blake3_servil::Mode::Hash, &hash, len as u64);
        let mut at = 0;
        for p in encoding.chunks(piece) {
            let ok = if pooled { v.update_multithreaded(p, |b| { out[at..at + b.len()].copy_from_slice(b); at += b.len(); }) } else { v.update(p, |b| { out[at..at + b.len()].copy_from_slice(b); at += b.len(); }) };
            assert!(ok);
        }
    });
    let mean = clocks::summary::mean(batches.iter().map(|b| (b.wall_ns, b.calls * len as u64)));
    let milli = (mean * 1000 + (1 << 63)) >> 64;
    println!("{} pieces of {piece}: {}.{:03} ns/B; {}", args[0], milli / 1000, milli % 1000, clocks::load::describe(&clocks::load::windows()));
}
