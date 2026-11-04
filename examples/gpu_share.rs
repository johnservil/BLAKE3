//! PROBE (probe/gpu-3-share): a stream's GPU share under each claim rule.
//! Streams of N messages of L bytes, back to back, each written in with
//! `submit_with` from one buffer, as bench-hashes' stream cells write it,
//! under: the GPU path off (the 2 MiB ring, the CPU alone), the GPU ring
//! with no claims, and claim rules (tasks a batch, tasks kept behind).
//! Per cell: the mean stream's time, GB/s, the GPU's share of the pieces,
//! the largest backlog a feed saw, the process's CPU time a stream, and the
//! caller's clock. Configurations interleave, three rounds.
//!
//!     cargo run --release --example gpu_share
use blake3_servil::{Hash, Mode};
use core::sync::atomic::Ordering;

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let input: Vec<u8> = (0..64usize << 20).map(|i| (i % 251) as u8).collect();
    let configs: [(&str, bool, usize, usize); 7] = [
        ("off", true, 128, 0),
        ("ring", false, 128, 1 << 30),
        ("128+128", false, 128, 128),
        ("128+32", false, 128, 32),
        ("64+64", false, 64, 64),
        ("256+64", false, 256, 64),
        ("256+0", false, 256, 0),
    ];
    // Warm: the pool, and the GPU's start and engaging (a run of 10 ms).
    for _ in 0..300 {
        one(&input, 16 << 20, 4);
    }
    println!("# round\tconfig\tlen_MiB\tmessages\tstreams\tstream_us\tGB/s\tgpu_permille\tbacklog\tcpu_ms_per_stream\tcaller_MHz");
    for round in 0..3 {
        for len in [4usize << 20, 16 << 20, 64 << 20] {
            for n in [1usize, 4, 16] {
                for &(name, off, batch, keep) in &configs {
                    blake3_servil::__gpu_probe::OFF.store(off, Ordering::Relaxed);
                    blake3_servil::__gpu_probe::BATCH.store(batch, Ordering::Relaxed);
                    blake3_servil::__gpu_probe::KEEP.store(keep, Ordering::Relaxed);
                    let streams = ((1usize << 30) / (len * n)).max(8);
                    // Engage again after a configuration switch (each run of 10 ms).
                    let warm = std::time::Instant::now();
                    while warm.elapsed() < std::time::Duration::from_millis(15) {
                        one(&input, len, n);
                    }
                    blake3_servil::__gpu_probe::take();
                    let cpu0 = clocks::process_cpu_ns();
                    let counts0 = clocks::Counts::read();
                    let mut total = 0u64;
                    for _ in 0..streams {
                        total += one(&input, len, n);
                        clocks::load::tick();
                    }
                    let cpu = clocks::process_cpu_ns() - cpu0;
                    let mhz = match (counts0, clocks::Counts::read()) {
                        (Some(a), Some(b)) => b.since(a).mhz().to_string(),
                        _ => "-".into(),
                    };
                    let (tasks, backlog) = blake3_servil::__gpu_probe::take();
                    let k = streams as u64;
                    let bytes = (len * n * streams) as u64;
                    let pieces = bytes / 65536;
                    println!(
                        "{round}\t{name}\t{}\t{n}\t{streams}\t{}\t{}\t{}\t{backlog}\t{}\t{mhz}",
                        len >> 20,
                        (total + k * 500) / (k * 1000),
                        (bytes + total / 2) / total,
                        (tasks as u64 * 1000 + pieces / 2) / pieces,
                        (cpu + k * 500_000) / (k * 1_000_000),
                    );
                }
            }
        }
    }
    clocks::load::tick();
    println!("# load: {}", clocks::load::describe(&clocks::load::windows()));
}

/// One stream of `n` messages of `len` bytes: its ns.
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
