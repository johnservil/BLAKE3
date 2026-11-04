//! PROBE (probe/gpu-4): the GPU as one more worker, on the landed 8 MiB
//! long ring. Streams of N messages of L bytes, each written in with
//! `submit_with` from the thread's own input, as bench-hashes' stream cells
//! write it, one thread (solo) or two at once on the one pool (shared).
//! Configurations: servil's 8 MiB long ring (cpu8); a 64 MiB GPU ring the
//! GPU never claims from (ring64); and claim rules, tasks a batch + tasks
//! kept behind it. Per cell: the mean stream's time, the process's CPU
//! time a stream, and the GPU's share of the pieces. Three rounds, the
//! configurations interleaved, each cell after 20 ms of its own warm-up.
//!
//!     cargo run --release --example host_lab
use blake3_servil::{Hash, Mode};
use core::sync::atomic::Ordering;
use std::sync::Barrier;

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let inputs: Vec<Vec<u8>> = (0..2).map(|t| (0..16usize << 20).map(|i| (i % 251 + t) as u8).collect()).collect();
    // (name, GPU ring, batch, keep)
    let configs: [(&str, bool, usize, usize); 6] = [
        ("cpu8", false, 256, 0),
        ("ring64", true, 256, 1 << 30),
        ("256+0", true, 256, 0),
        ("256+128", true, 256, 128),
        ("256+512", true, 256, 512),
        ("128+512", true, 128, 512),
    ];
    for _ in 0..50 {
        one(&inputs[0], 16 << 20, 4);
    }
    blake3_servil::__gpu_probe::GPU.store(true, Ordering::Relaxed);
    let started = std::time::Instant::now();
    while started.elapsed() < std::time::Duration::from_secs(2) {
        one(&inputs[0], 16 << 20, 4);
    }
    println!("# round\tconfig\tscenario\tlen_MiB\tmessages\tstreams\tstream_us\tGB/s_per_thread\tcpu_ms_per_stream\tgpu_permille");
    for round in 0..3 {
        for len in [4usize << 20, 16 << 20] {
            for n in [1usize, 4, 16, 64] {
                for threads in [1usize, 2] {
                    for &(name, gpu, batch, keep) in &configs {
                        blake3_servil::__gpu_probe::GPU.store(gpu, Ordering::Relaxed);
                        blake3_servil::__gpu_probe::BATCH.store(batch, Ordering::Relaxed);
                        blake3_servil::__gpu_probe::KEEP.store(keep, Ordering::Relaxed);
                        let streams = ((512usize << 20) / (len * n)).max(3);
                        let barrier = Barrier::new(threads);
                        let warm = std::time::Instant::now();
                        while warm.elapsed() < std::time::Duration::from_millis(20) {
                            one(&inputs[0], len, n.min(4));
                        }
                        blake3_servil::__gpu_probe::TASKS.swap(0, Ordering::Relaxed);
                        let cpu0 = clocks::process_cpu_ns();
                        let totals: Vec<u64> = std::thread::scope(|scope| {
                            let handles: Vec<_> = (0..threads)
                                .map(|t| {
                                    let (input, barrier) = (&inputs[t], &barrier);
                                    scope.spawn(move || {
                                        barrier.wait();
                                        (0..streams).map(|_| one(input, len, n)).sum::<u64>()
                                    })
                                })
                                .collect();
                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        });
                        let cpu = clocks::process_cpu_ns() - cpu0;
                        let tasks = blake3_servil::__gpu_probe::TASKS.swap(0, Ordering::Relaxed) as u64;
                        clocks::load::tick();
                        let k = (streams * threads) as u64;
                        let total: u64 = totals.iter().sum();
                        let bytes = (len * n) as u64 * k;
                        let pieces = bytes / 65536;
                        println!(
                            "{round}\t{name}\t{}\t{}\t{n}\t{streams}\t{}\t{}\t{}\t{}",
                            if threads == 1 { "solo" } else { "shared" },
                            len >> 20,
                            (total + k * 500) / (k * 1000),
                            (bytes + total / 2) / total,
                            (cpu + k * 500_000) / (k * 1_000_000),
                            (tasks * 1000 + pieces / 2) / pieces
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
