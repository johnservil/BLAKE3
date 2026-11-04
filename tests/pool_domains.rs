//! The pool across L3 domains (probe/l3-pool, #196), on any machine: the
//! CPUs split into two domains before the pool starts, so the paths only a
//! machine with several L3s takes run here too. Callers on four threads
//! hash every kind of work the pool takes (one message under and over
//! EVERY_LEN, a batch, a collection, a stream), each checked against the
//! single-threaded digest; then every worker parks, so no work is left
//! for a domain whose workers sleep.
#![cfg(feature = "std")]

use blake3_servil::performance::{Batch, Block, hash, hash_each_multithreaded, hash_many, hash_many_multithreaded, hash_multithreaded, stream};
use blake3_servil::{Hash, Mode};

fn blocks(len: usize, seed: usize) -> Vec<Block> {
    let mut b = vec![Block::ZERO; len.div_ceil(64).max(1)];
    for (i, byte) in Block::bytes_mut(&mut b)[..len].iter_mut().enumerate() {
        *byte = (i.wrapping_mul(31).wrapping_add(seed * 7)) as u8;
    }
    b
}

fn every_kind(seed: usize) {
    for len in [(512 << 10) + 1, 1 << 20, (3 << 20) + 5, 33 << 20] {
        let b = blocks(len, seed);
        assert_eq!(hash_multithreaded(Mode::Plain, &b, len), hash(Mode::Plain, &b, len), "one message of {len} B");
    }
    let batch_blocks = blocks(64 * 8192, seed);
    let batch = Batch::from_blocks(&batch_blocks, 64);
    let (mut many, mut one) = (vec![[0u8; 32]; 8192], vec![[0u8; 32]; 8192]);
    hash_many_multithreaded(Mode::Plain, &batch, &mut many);
    hash_many(Mode::Plain, &batch, &mut one);
    assert_eq!(many, one, "a batch of 8192 x 64 B");
    let items: Vec<(Vec<Block>, usize)> = (0..300).map(|i| (blocks(1 + i * 997 % 40000, seed + i), 1 + i * 997 % 40000)).collect();
    let refs: Vec<(&[Block], usize)> = items.iter().map(|(b, l)| (b.as_slice(), *l)).collect();
    let mut each = vec![[0u8; 32]; refs.len()];
    hash_each_multithreaded(Mode::Plain, &refs, &mut each);
    for ((b, l), d) in items.iter().zip(&each) {
        assert_eq!(*d, *hash(Mode::Plain, b, *l).as_bytes(), "a collection's item of {l} B");
    }
    let messages: Vec<(Vec<Block>, usize)> = (0..400).map(|i| (blocks(1 + i * 4099 % 300000, seed + 3 * i), 1 + i * 4099 % 300000)).collect();
    let mut places = vec![Hash::from([0; 32]); messages.len()];
    stream(
        Mode::Plain,
        |_: &mut [Hash]| {},
        |s| {
            for ((b, l), place) in messages.iter().zip(places.iter_mut()) {
                s.submit(b, *l, place);
            }
            s.finish();
        },
    );
    for ((b, l), got) in messages.iter().zip(&places) {
        assert_eq!(*got, hash(Mode::Plain, b, *l), "a stream's message of {l} B");
    }
}

#[test]
fn every_kind_of_work_is_right_across_two_domains_and_the_workers_park() {
    blake3_servil::__split_domains_for_tests(2);
    blake3_servil::initialize_multithreaded(None);
    let callers: Vec<_> = (0..4)
        .map(|t| {
            std::thread::spawn(move || {
                for round in 0..3 {
                    every_kind(t * 10 + round)
                }
            })
        })
        .collect();
    for c in callers {
        c.join().expect("a caller");
    }
    let started = std::time::Instant::now();
    while !blake3_servil::__all_workers_parked() {
        assert!(started.elapsed() < std::time::Duration::from_secs(10), "every worker parks once the work is done");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
