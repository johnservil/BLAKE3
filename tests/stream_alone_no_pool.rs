//! `performance::stream` needs no pool (bench-hashes #131): in a process
//! that never calls initialize_multithreaded, it hashes, and the process
//! has as many threads after it as before (this crate started none).

use blake3_servil::{Hash, Mode, performance::Block};

/// This process's threads, from /proc/self/status.
fn threads() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    status.lines().find_map(|l| l.strip_prefix("Threads:")).expect("a Threads line").trim().parse().expect("a count")
}

#[test]
fn stream_alone_needs_no_pool() {
    blake3_servil::initialize();
    let before = threads();
    let messages: Vec<Block> = (0..5000u32).map(|i| Block(i.to_le_bytes().repeat(16).try_into().unwrap())).collect();
    let mut digests = vec![Hash::from_bytes([0; 32]); messages.len()];
    let mut handed_back = 0;
    blake3_servil::performance::stream(
        Mode::Plain,
        |done: &mut [Hash]| handed_back += done.len(),
        |s| {
            for (i, (message, digest)) in messages.iter().zip(&mut digests).enumerate() {
                s.submit(core::slice::from_ref(message), 1 + i % 64, digest);
            }
            s.finish();
        },
    );
    assert_eq!(handed_back, messages.len());
    for (i, digest) in digests.iter().enumerate() {
        assert_eq!(*digest, blake3_servil::convenience::hash(&messages[i].0[..1 + i % 64]), "message {i}");
    }
    assert_eq!(threads(), before, "no thread started");
}
