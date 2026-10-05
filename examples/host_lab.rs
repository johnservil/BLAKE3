//! probe/one-update: a Hasher per message, the whole message in one
//! update, then finalize (the digest traits' shape), nonstop, for messages
//! of 64 B to 64 KiB, each written into the program's buffer first: the
//! cost of the Hasher's 16 KiB stage for a message shorter than it.

fn main() {
    blake3_servil::initialize();
    let source: Vec<u8> = (0..(1u32 << 20)).map(|i| i.wrapping_mul(2654435761).to_le_bytes()[3]).collect();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut slot = [0u8; 32];
    for round in 0..2 {
        for len in [64usize, 1024, 1500, 4096, 8192, 15000, 16384, 65536] {
            let per = (1 << 20) / len;
            let batches = clocks::measure(5, 200_000_000, || {
                for k in 0..per {
                    let at = (k * 4099) % (source.len() - len);
                    buffer[..len].copy_from_slice(&source[at..at + len]);
                    let mut hasher = blake3_servil::Hasher::new();
                    hasher.update(std::hint::black_box(&buffer[..len]));
                    slot = *hasher.finalize().as_bytes();
                }
            });
            let calls: u64 = batches.iter().map(|b| b.calls).sum();
            let wall: u64 = batches.iter().map(|b| b.wall_ns).sum();
            let t = (wall as u128 * 10_000 / (calls as u128 * (per * len) as u128)) as u64;
            println!("round {round}  {len:6} B  {}.{:04} ns/B   [{}]", t / 10000, t % 10000, batches[0].show());
        }
    }
    std::hint::black_box(slot);
    eprintln!("host_lab: {}", clocks::load::describe(&clocks::load::windows()));
}
