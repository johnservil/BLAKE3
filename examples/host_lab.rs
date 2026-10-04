//! probe/gather-16k: the benchmark's "many messages at once" schedule
//! (bench-hashes INTERLEAVED: 256 open messages, pieces dealt in turn,
//! piece lengths cycling 1448, 1448, 4096, 1448, 16384 B, message lengths
//! cycling 64 B to 16 MiB), each piece written into the program's buffer
//! first, then hashed by a Hasher per message: `update` per piece, against
//! each message's pieces gathered into a 16 KiB buffer of its own (the copy
//! charged) and `update` once per 16 KiB and at the message's end.

const OPEN: usize = 256;
const PIECES: [usize; 5] = [1448, 1448, 4096, 1448, 16 * 1024];
const MESSAGES: [usize; 7] = [64, 1000, 4470, 16 * 1024, 100_000, 1024 * 1024, 16 * 1024 * 1024];

struct Open { hasher: blake3_servil::Hasher, left: usize, gathered: Vec<u8> }

fn run(gather: usize, source: &[u8], bytes: usize, state: &mut (Vec<Open>, usize, usize, usize), slots: &mut [[u8; 32]]) {
    let (open, piece, opened, at) = state;
    let mut buffer = [0u8; 16 * 1024];
    let mut budget = bytes;
    while budget > 0 {
        let slot = *piece % OPEN;
        let len = PIECES[*piece % PIECES.len()].min(open[slot].left).min(budget);
        *piece += 1;
        if *at + len > source.len() { *at = 0; }
        buffer[..len].copy_from_slice(&source[*at..*at + len]);
        *at += len;
        let m = &mut open[slot];
        if gather > 0 {
            m.gathered.extend_from_slice(&buffer[..len]);
            // Exactly 16 KiB at a time, so every update starts on a 16-chunk boundary.
            while m.gathered.len() >= gather {
                m.hasher.update(&m.gathered[..gather]);
                m.gathered.drain(..gather);
            }
        } else {
            m.hasher.update(&buffer[..len]);
        }
        m.left -= len;
        budget -= len;
        if m.left == 0 {
            if gather > 0 && !m.gathered.is_empty() { m.hasher.update(&m.gathered); m.gathered.clear(); }
            slots[*opened % slots.len()] = *m.hasher.finalize().as_bytes();
            *m = Open { hasher: blake3_servil::Hasher::new(), left: MESSAGES[*opened % MESSAGES.len()], gathered: std::mem::take(&mut m.gathered) };
            *opened += 1;
        }
    }
}

fn main() {
    blake3_servil::initialize();
    let source: Vec<u8> = (0..(4u32 << 20)).map(|i| i.wrapping_mul(2654435761).to_le_bytes()[3]).collect();
    let mut slots = vec![[0u8; 32]; 1024];
    let unit = 1 << 20;
    let fresh = || -> (Vec<Open>, usize, usize, usize) {
        ((0..OPEN).map(|n| Open { hasher: blake3_servil::Hasher::new(), left: MESSAGES[n % MESSAGES.len()], gathered: Vec::with_capacity(64 * 1024 + 16384) }).collect(), 0, OPEN, 0)
    };
    // The same digests both ways.
    let (mut a, mut b) = (fresh(), fresh());
    let (mut sa, mut sb) = (vec![[0u8; 32]; 1 << 16], vec![[0u8; 32]; 1 << 16]);
    run(0, &source, 64 << 20, &mut a, &mut sa);
    run(16 * 1024, &source, 64 << 20, &mut b, &mut sb);
    assert_eq!(a.2, b.2);
    assert!(sa == sb, "the same hashes either way");
    for round in 0..2 {
        for (name, gather) in [("update per piece", 0), ("gathered into 16 KiB", 16 * 1024), ("gathered into 64 KiB", 64 * 1024)] {
            let mut state = fresh();
            run(gather, &source, 32 << 20, &mut state, &mut slots); // into the steady state
            let batches = clocks::measure(5, 300_000_000, || run(gather, &source, unit, &mut state, &mut slots));
            let calls: u64 = batches.iter().map(|b| b.calls).sum();
            let wall: u64 = batches.iter().map(|b| b.wall_ns).sum();
            let t = (wall as u128 * 10_000 / (calls as u128 * unit as u128)) as u64;
            println!("round {round}  {name:22} {}.{:04} ns/B   [{}]", t / 10000, t % 10000, batches[0].show());
        }
    }
    eprintln!("host_lab: {}", clocks::load::describe(&clocks::load::windows()));
}
