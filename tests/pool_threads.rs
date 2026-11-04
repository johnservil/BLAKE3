//! initialize_multithreaded(Some(n)): the multithreaded calls hash on n
//! threads. A process of its own (the number is process-wide): 1 starts no
//! workers, the calls still hash rightly, a later call with None keeps the
//! number, and another number panics.
#![cfg(feature = "std")]

use core::num::NonZeroUsize;

#[path = "common/golden.rs"]
#[allow(dead_code)]
mod golden;

/// The process's threads (Linux), from /proc.
#[cfg(target_os = "linux")]
fn threads() -> usize {
    std::fs::read_dir("/proc/self/task").expect("/proc/self/task").count()
}

#[test]
fn one_thread_starts_no_workers() {
    blake3_servil::initialize();
    #[cfg(target_os = "linux")]
    let before = threads();
    blake3_servil::initialize_multithreaded(Some(NonZeroUsize::MIN));
    let input: Vec<u8> = (0..8u32 << 20).map(|i| (i * 31 + i / 509) as u8).collect();
    let blocks: Vec<blake3_servil::performance::Block> = input.chunks_exact(64).map(|b| blake3_servil::performance::Block(b.try_into().unwrap())).collect();
    assert_eq!(blake3_servil::performance::hash_multithreaded(blake3_servil::Mode::Plain, &blocks, input.len()), blake3_servil::convenience::hash(&input));
    blake3_servil::initialize_multithreaded(None);
    blake3_servil::initialize_multithreaded(Some(NonZeroUsize::MIN));
    assert_eq!(blake3_servil::performance::hash_multithreaded(blake3_servil::Mode::Plain, &blocks, input.len()), blake3_servil::convenience::hash(&input));
    // A collection on a pool of one: no spare thread, so lanes::hash_each
    // hashes it on the caller (BLAKE3 #238: no test took that path). Its
    // items below MIN_SPLIT_LEN (512 KiB) total 775 KB, over it, so they go
    // to lanes::hash_each; the 600,000-byte item hashes alone.
    let items: Vec<Vec<u8>> =
        [0usize, 1, 64, 1025, 4096, 70_000, 300_000, 400_000, 600_000].iter().enumerate().map(|(k, &n)| (0..n).map(|i| (i * 7 + k) as u8).collect()).collect();
    let held: Vec<Vec<blake3_servil::performance::Block>> = items
        .iter()
        .map(|x| {
            let mut b = vec![blake3_servil::performance::Block::ZERO; x.len().div_ceil(64).max(1)];
            blake3_servil::performance::Block::bytes_mut(&mut b)[..x.len()].copy_from_slice(x);
            b
        })
        .collect();
    let refs: Vec<(&[blake3_servil::performance::Block], usize)> = held.iter().zip(&items).map(|(b, x)| (&b[..], x.len())).collect();
    let mut out = vec![[0u8; 32]; items.len()];
    blake3_servil::performance::hash_each_multithreaded(blake3_servil::Mode::Plain, &refs, &mut out);
    let mut g = golden::Golden::new("hash_each_on_a_pool_of_one");
    g.add(out.as_flattened(), || items.iter().flat_map(|x| *blake3::hash(x).as_bytes()).collect());
    g.check(0x36d8838896a1a882);
    // A Hasher's multithreaded update on a pool of one: each whole subtree
    // of MIN_SPLIT_LEN or more goes to lanes::subtree_children, which has no
    // spare thread and hashes it on the caller (BLAKE3 #238: no test took
    // that path). 1 MiB from zero (one subtree of 1 MiB), and 3 MiB + 5
    // bytes after a 1-byte update (subtrees cut at other counters).
    let big: Vec<u8> = (0..3usize << 20 | 6).map(|i| (i * 13 + i / 4093) as u8).collect();
    let mut g = golden::Golden::new("hasher_on_a_pool_of_one");
    let mut h = blake3_servil::performance::Hasher::new(blake3_servil::Mode::Plain);
    h.update_multithreaded(&big[..1 << 20]);
    g.add(h.finalize().as_bytes(), || blake3::hash(&big[..1 << 20]).as_bytes().to_vec());
    let mut h = blake3_servil::performance::Hasher::new(blake3_servil::Mode::Plain);
    h.update_multithreaded(&big[..1]).update_multithreaded(&big[1..]);
    g.add(h.finalize().as_bytes(), || blake3::hash(&big).as_bytes().to_vec());
    g.check(0x43fd488273159b1a);
    // No other thread: no workers, no SME2 thread.
    #[cfg(target_os = "linux")]
    assert!(threads() <= before, "{before} threads before, {} after", threads());
    // Where panics unwind.
    #[cfg(panic = "unwind")]
    {
        let other = std::panic::catch_unwind(|| blake3_servil::initialize_multithreaded(NonZeroUsize::new(3)));
        assert!(other.is_err(), "another number panics");
    }
}
