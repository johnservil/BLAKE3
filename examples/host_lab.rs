//! PROBE (probe/gpu-first): what limits one stream of long messages, and
//! whether the GPU taking its pieces first (before the CPU workers see
//! them) raises its throughput or lowers its time. Zooko's rule: kept
//! only for throughput or latency, never for energy alone.
//!
//! Part 1, the copy alone: the caller's copy of a 16 MiB source into a
//! 64 MiB buffer, round and round, alone and beside a thread keeping the
//! CPU pool busy hashing: one stream's ceiling, whoever hashes.
//! Part 2, streams of N messages of L bytes written in with submit_with, as
//! bench-hashes' stream cells write them, solo: the CPU alone (cpu); the
//! GPU claiming what waits unclaimed (claims, servil's design); the GPU
//! given whole batches first, the pool the overflow (first). Per cell: the
//! mean stream's time, GB/s, the process's CPU time a stream, and the
//! GPU's and the overflow's share of the pieces. Three rounds, the
//! configurations interleaved.
//!
//!     cargo run --release --example host_lab
use blake3_servil::{Hash, Mode};
use core::sync::atomic::{AtomicBool, Ordering};

fn main() {
    blake3_servil::initialize_multithreaded(None);
    let input: Vec<u8> = (0..16usize << 20).map(|i| (i % 251) as u8).collect();
    // Part 1.
    let mut dest = vec![0u8; 64 << 20];
    let copy = |dest: &mut [u8]| {
        let start = clocks::now();
        let mut at = 0;
        for _ in 0..64 {
            dest[at..at + input.len()].copy_from_slice(&input);
            at = (at + input.len()) % dest.len();
        }
        std::hint::black_box(&dest);
        let ns = clocks::since_ns(start);
        ((1u64 << 30) + ns / 2) / ns
    };
    copy(&mut dest);
    println!("# copy alone, GB/s: {} {} {}", copy(&mut dest), copy(&mut dest), copy(&mut dest));
    let busy = AtomicBool::new(true);
    let other = vec![blake3_servil::performance::Block::ZERO; (64 << 20) / 64];
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while busy.load(Ordering::Relaxed) {
                std::hint::black_box(blake3_servil::performance::hash_multithreaded(Mode::Plain, &other, 64 << 20));
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        println!("# copy beside the pool hashing, GB/s: {} {} {}", copy(&mut dest), copy(&mut dest), copy(&mut dest));
        busy.store(false, Ordering::Relaxed);
    });
    // Part 2.
    for _ in 0..100 {
        one(&input, 16 << 20, 4);
    }
    let started = std::time::Instant::now();
    while started.elapsed() < std::time::Duration::from_secs(2) {
        one(&input, 16 << 20, 4);
    }
    let configs: [(&str, bool, bool); 3] = [("cpu", true, false), ("claims", false, false), ("first", false, true)];
    println!("# round\tconfig\tlen_MiB\tmessages\tstreams\tstream_us\tGB/s\tcpu_ms_per_stream\tgpu_permille\toverflow_permille");
    for round in 0..3 {
        for len in [4usize << 20, 16 << 20] {
            for n in [1usize, 4, 16, 64] {
                for &(name, off, first) in &configs {
                    blake3_servil::__gpu_probe::OFF.store(off, Ordering::Relaxed);
                    blake3_servil::__gpu_probe::GPU_FIRST.store(first, Ordering::Relaxed);
                    let streams = ((1usize << 30) / (len * n)).max(4);
                    let warm = std::time::Instant::now();
                    while warm.elapsed() < std::time::Duration::from_millis(20) {
                        one(&input, len, n.min(4));
                    }
                    blake3_servil::__gpu_probe::GPU_PIECES.swap(0, Ordering::Relaxed);
                    blake3_servil::__gpu_probe::CPU_PIECES.swap(0, Ordering::Relaxed);
                    let cpu0 = clocks::process_cpu_ns();
                    let total: u64 = (0..streams).map(|_| one(&input, len, n)).sum();
                    let cpu = clocks::process_cpu_ns() - cpu0;
                    clocks::load::tick();
                    let gpu = blake3_servil::__gpu_probe::GPU_PIECES.swap(0, Ordering::Relaxed) as u64;
                    let overflow = blake3_servil::__gpu_probe::CPU_PIECES.swap(0, Ordering::Relaxed) as u64;
                    let k = streams as u64;
                    let bytes = (len * n) as u64 * k;
                    let pieces = bytes / 65536;
                    println!(
                        "{round}\t{name}\t{}\t{n}\t{streams}\t{}\t{}\t{}\t{}\t{}",
                        len >> 20,
                        (total + k * 500) / (k * 1000),
                        (bytes + total / 2) / total,
                        (cpu + k * 500_000) / (k * 1_000_000),
                        (gpu * 1000 + pieces / 2) / pieces,
                        (overflow * 1000 + pieces / 2) / pieces
                    );
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
