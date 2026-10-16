//! A fresh process's first calls, while the pool's threads are still
//! starting (initialize_multithreaded returns before they do): a stream of
//! messages of every shape it gathers and a multithreaded hash past the
//! split, each checked against hash(). Exits 0 when
//! every digest matches; a mismatch panics. Run many times (a fresh process
//! each) to meet the threads at every stage of their start.
use blake3_servil::{Hash, Mode};

fn main() {
    blake3_servil::initialize_multithreaded();
    let seed: usize = std::env::args().nth(1).map_or(0, |s| s.parse().unwrap());
    let data: Vec<u8> = (0..(2 << 20) + 4096).map(|i| ((i * 31 + seed) % 251) as u8).collect();
    // The stream first: messages of 64 B to 1 KiB (gathered members), and longer.
    let lens = [64usize, 64, 128, 256, 512, 1024, 64, 1000, 4096, 70_000, 64, 64];
    let count = 200 + seed % 300;
    let message = |i: usize| &data[i..i + lens[(i + seed) % lens.len()]];
    let mut got = vec![Hash::from_bytes([0; 32]); count];
    blake3_servil::stream(Mode::Hash, |_: &mut [Hash]| {}, |s| {
        for (i, place) in got.iter_mut().enumerate() {
            s.submit(message(i), place);
        }
        s.finish();
    });
    for (i, digest) in got.iter().enumerate() {
        assert_eq!(*digest, blake3_servil::hash(message(i)), "stream message {i}, {} bytes, seed {seed}", message(i).len());
    }
    // A multithreaded hash past the split.
    let big = &data[seed % 4096..][..(1 << 20) + 777];
    assert_eq!(blake3_servil::hash_multithreaded(big), blake3_servil::hash(big), "hash_multithreaded, seed {seed}");
}
