//! A domain with no worker (probe/l3-pool, #196): a pool of two threads,
//! one worker, over four domains leaves three with none. On the EPYC 9255
//! (four L3 domains) a caller on one of their CPUs panicked in a one-shot
//! ("a serial call hashes the input whole") and waited forever in a
//! stream's `collect`, for work only its own domain's workers may take
//! (Sam's runs, November 2). Here, on any machine of four CPUs or more:
//! the CPUs split into four domains, the caller pinned to the last CPU,
//! whose domain gets no worker, and every kind of work checked against the
//! single-threaded digest. A binary of its own, since the pool is the
//! process's; where it would hang, nextest's clock kills and names it.
#![cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]

use blake3_servil::performance::{Batch, Block, hash, hash_each_multithreaded, hash_many, hash_many_multithreaded, hash_multithreaded, stream};
use blake3_servil::{Hash, Mode};

fn blocks(len: usize, seed: usize) -> Vec<Block> {
    let mut b = vec![Block::ZERO; len.div_ceil(64).max(1)];
    for (i, byte) in Block::bytes_mut(&mut b)[..len].iter_mut().enumerate() {
        *byte = (i.wrapping_mul(31).wrapping_add(seed * 7)) as u8;
    }
    b
}

/// This thread, from now on, on `cpu` alone.
fn pin_to(cpu: usize) {
    // Sound: a zeroed cpu_set_t is the empty set; CPU_SET writes within it,
    // and sched_setaffinity reads it for this thread (pid 0).
    unsafe {
        let mut set: libc::cpu_set_t = core::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set), 0, "pinned to CPU {cpu}");
    }
}

#[test]
fn a_caller_in_a_domain_with_no_worker_hashes_every_kind_of_work() {
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    if cpus < 4 {
        eprintln!("skipped: {cpus} CPUs, and the split needs four domains");
        return;
    }
    blake3_servil::__split_domains_for_tests(4);
    blake3_servil::initialize_multithreaded(Some(std::num::NonZeroUsize::new(2).unwrap()));
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // The last CPU is in the last domain, and one worker goes to the first.
        pin_to(cpus - 1);
        for len in [(512 << 10) + 1, 1 << 20, (3 << 20) + 5] {
            let b = blocks(len, 1);
            assert_eq!(hash_multithreaded(Mode::Plain, &b, len), hash(Mode::Plain, &b, len), "one message of {len} B");
        }
        let batch_blocks = blocks(64 * 8192, 2);
        let batch = Batch::from_blocks(&batch_blocks, 64);
        let (mut many, mut one) = (vec![[0u8; 32]; 8192], vec![[0u8; 32]; 8192]);
        hash_many_multithreaded(Mode::Plain, &batch, &mut many);
        hash_many(Mode::Plain, &batch, &mut one);
        assert_eq!(many, one, "a batch of 8192 x 64 B");
        let items: Vec<(Vec<Block>, usize)> = (0..100).map(|i| (blocks(1 + i * 997 % 40000, 3 + i), 1 + i * 997 % 40000)).collect();
        let refs: Vec<(&[Block], usize)> = items.iter().map(|(b, l)| (b.as_slice(), *l)).collect();
        let mut each = vec![[0u8; 32]; refs.len()];
        hash_each_multithreaded(Mode::Plain, &refs, &mut each);
        for ((b, l), d) in items.iter().zip(&each) {
            assert_eq!(*d, *hash(Mode::Plain, b, *l).as_bytes(), "a collection's item of {l} B");
        }
        // A stream whose caller loops on collect() for a free place.
        let messages: Vec<(Vec<Block>, usize)> = (0..500).map(|i| (blocks(1 + i * 7 % 3000, 5 + i), 1 + i * 7 % 3000)).collect();
        let mut places = vec![Hash::from([0; 32]); 4];
        let free = std::cell::RefCell::new(places.iter_mut().collect::<Vec<_>>());
        let got = std::cell::RefCell::new(Vec::new());
        stream(
            Mode::Plain,
            |place: &mut [Hash]| {
                got.borrow_mut().push(place[0]);
                free.borrow_mut().push(&mut place[0]);
            },
            |s| {
                for (b, l) in &messages {
                    let place = loop {
                        if let Some(place) = free.borrow_mut().pop() {
                            break place;
                        }
                        s.collect();
                    };
                    s.submit(b, *l, place);
                }
                s.finish();
            },
        );
        let got = got.into_inner();
        assert_eq!(got.len(), messages.len());
        for (i, (b, l)) in messages.iter().enumerate() {
            assert_eq!(got[i], hash(Mode::Plain, b, *l), "the stream's message {i}");
        }
        done.send(()).unwrap();
    });
    finished.recv().expect("every kind of work finishes, the stream's collect() included");
}
