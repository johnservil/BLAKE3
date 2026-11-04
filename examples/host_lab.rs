//! PROBE (probe/long-ring-size): the long ring's size, solo and shared.
//! Streams of N messages of L bytes, each written in with `submit_with`
//! from the thread's own input, as bench-hashes' stream cells write it:
//! one thread (solo), or two at once on the one pool, each with its own
//! input (the bench's shared scenario, placed by the OS). Long rings of 8,
//! 16, 32 and 64 MiB, and none (messages in the 2 MiB ring, as servil).
//! Per cell, the mean stream's time over every thread's streams; three
//! rounds, the configurations interleaved.
//!
//!     cargo run --release --example host_lab
use blake3_servil::{Hash, Mode};
use core::sync::atomic::Ordering;
use std::sync::Barrier;

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let inputs: Vec<Vec<u8>> = (0..2).map(|t| (0..16usize << 20).map(|i| (i % 251 + t) as u8).collect()).collect();
    let configs: [(&str, usize, usize); 5] =
        [("2MiB", usize::MAX, 64 << 20), ("8MiB", 4 << 20, 8 << 20), ("16MiB", 4 << 20, 16 << 20), ("32MiB", 4 << 20, 32 << 20), ("64MiB", 4 << 20, 64 << 20)];
    for _ in 0..100 {
        one(&inputs[0], 4 << 20, 4);
    }
    println!("# round\tconfig\tscenario\tlen_MiB\tmessages\tstreams\tstream_us\tGB/s_per_thread");
    for round in 0..3 {
        for len in [4usize << 20, 16 << 20] {
            for n in [1usize, 4, 16] {
                for threads in [1usize, 2] {
                    for &(name, long, ring) in &configs {
                        blake3_servil::PROBE_LONG.store(long, Ordering::Relaxed);
                        blake3_servil::PROBE_LONG_RING.store(ring, Ordering::Relaxed);
                        let streams = ((512usize << 20) / (len * n)).max(6);
                        let barrier = Barrier::new(threads);
                        let totals: Vec<u64> = std::thread::scope(|scope| {
                            let handles: Vec<_> = (0..threads)
                                .map(|t| {
                                    let (input, barrier) = (&inputs[t], &barrier);
                                    scope.spawn(move || {
                                        for _ in 0..3 {
                                            one(input, len, n);
                                        }
                                        barrier.wait();
                                        (0..streams).map(|_| one(input, len, n)).sum::<u64>()
                                    })
                                })
                                .collect();
                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        });
                        clocks::load::tick();
                        let k = (streams * threads) as u64;
                        let total: u64 = totals.iter().sum();
                        let bytes = (len * n) as u64 * k;
                        println!(
                            "{round}\t{name}\t{}\t{}\t{n}\t{streams}\t{}\t{}",
                            if threads == 1 { "solo" } else { "shared" },
                            len >> 20,
                            (total + k * 500) / (k * 1000),
                            (bytes + total / 2) / total
                        );
                    }
                }
            }
        }
    }
    clocks::load::tick();
    println!("# load: {}", clocks::load::describe(&clocks::load::windows()));
}

/// One stream of `n` messages of `len` bytes of `input`: its ns.
fn one(input: &[u8], len: usize, n: usize) -> u64 {
    let message = &input[..len];
    let mut places = vec![Hash::from([0u8; 32]); n];
    let start = clocks::now();
    blake3_servil::performance::stream(
        Mode::Plain,
        |_: &mut [Hash]| {},
        |s| {
            for place in places.iter_mut() {
                s.submit_with(
                    len,
                    |at, room| {
                        room.copy_from_slice(&message[at..][..room.len()]);
                        room.len()
                    },
                    place,
                );
            }
            s.finish();
        },
    );
    let total = clocks::since_ns(start);
    assert_eq!(places[n - 1], blake3_servil::convenience::hash(message));
    total
}
