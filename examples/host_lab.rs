//! Probe (probe/gpu-short-streams-*): where a short stream of long messages
//! spends its time, the bench's way (BLAKE3-Servil #170: the GPU branch's
//! solo 16 MiB stream cell 1.38x slower, Mac check 1877, while 64 MiB read
//! level). Streams of N messages of L bytes, back to back, each message
//! written in with `submit_with` from one buffer, as bench-hashes' stream
//! cells write it; the same probe on the branch and on its base (no GPU).
//!
//! Per stream, run means over every measured stream: the submits (the
//! program writing its messages in), the finish (waiting for the digests),
//! the process's CPU time, and the calling thread's clock in MHz (its cycles over its time).
//!
//!     cargo run --release --example host_lab -- short [MIB_PER_CELL]
use blake3_servil::{Hash, Mode};

fn main() {
    let part = std::env::args().nth(1).expect("host_lab short [MIB_PER_CELL] | tasks");
    if part == "tasks" {
        return tasks();
    }
    assert_eq!(part, "short", "the parts: short, tasks");
    let budget: usize = std::env::args().nth(2).map_or(4096, |a| a.parse().expect("MIB_PER_CELL")) << 20;
    blake3_servil::initialize_multithreaded(None);
    let input: Vec<u8> = (0..64usize << 20).map(|i| (i % 251) as u8).collect();
    println!("# len\tmessages\tstreams\tsubmit_us\tfinish_us\tstream_us\tGB/s\tcpu_ms_per_stream\tcaller_MHz");
    // Warm: the pool, and the GPU's start and engaging (a run of 10 ms).
    for _ in 0..200 {
        one(&input, 16 << 20, 2);
    }
    for len in [4usize << 20, 16 << 20, 64 << 20] {
        for n in [1usize, 2, 4, 8] {
            let streams = (budget / (len * n)).max(20);
            for _ in 0..10 {
                one(&input, len, n);
            }
            let (mut submit, mut total) = (0u64, 0u64);
            #[cfg(target_os = "macos")]
            blake3_servil::__gpu_probe();
            let cpu0 = clocks::process_cpu_ns();
            let counts0 = clocks::Counts::read();
            for _ in 0..streams {
                let (s, t) = one(&input, len, n);
                submit += s;
                total += t;
                clocks::load::tick();
            }
            let cpu = clocks::process_cpu_ns() - cpu0;
            let mhz = match (counts0, clocks::Counts::read()) {
                (Some(a), Some(b)) => b.since(a).mhz().to_string(),
                _ => "-".into(),
            };
            let k = streams as u64;
            let bytes = (len * n * streams) as u64;
            // PROBE: what the stream's ends waited on (GPU commands, races).
            #[cfg(target_os = "macos")]
            let probe = {
                let [done, done_ns, races, won, race_ns, won_ns] = blake3_servil::__gpu_probe();
                format!(
                    "\tgpu {done} done, {} us each; races {races}, CPU won {won}; {} us a race, {} us a won one",
                    done_ns / done.max(1) / 1000,
                    race_ns / races.max(1) / 1000,
                    won_ns / won.max(1) / 1000
                )
            };
            #[cfg(not(target_os = "macos"))]
            let probe = String::new();
            println!(
                "{}\t{n}\t{streams}\t{}\t{}\t{}\t{}\t{}\t{mhz}{probe}",
                len >> 20,
                (submit + k * 500) / (k * 1000),
                (total - submit + k * 500) / (k * 1000),
                (total + k * 500) / (k * 1000),
                (bytes + total / 2) / total,
                (cpu + k * 500_000) / (k * 1_000_000),
            );
        }
    }
    clocks::load::tick();
    println!("# load: {}", clocks::load::describe(&clocks::load::windows()));
}

/// One stream of `n` messages of `len` bytes: the ns its submits took, and
/// the ns of the whole stream.
fn one(input: &[u8], len: usize, n: usize) -> (u64, u64) {
    let message = &input[..len];
    let mut places = vec![Hash::from([0u8; 32]); n];
    let start = clocks::now();
    let mut submitted = 0;
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
            submitted = clocks::since_ns(start);
            s.finish();
        },
    );
    let total = clocks::since_ns(start);
    assert_eq!(places[0], blake3_servil::convenience::hash(message));
    (submitted, total)
}

/// The GPU hashing a batch of 64 KiB tasks in one command (each checked
/// against the CPU first), beside the CPU pool hashing the same bytes, by
/// batch size: what a GPU worker taking tasks from a stream's ring would pay.
fn tasks() {
    blake3_servil::initialize_multithreaded(None);
    println!("# MiB\tgpu_first_after_idle_us\tgpu_back_to_back_us\tgpu_after_2ms_gap_us\tgpu_GB/s_back_to_back\tcpu_pool_us\tcpu_GB/s");
    for mib in [1usize, 2, 4, 8, 16, 32, 64] {
        #[cfg(target_os = "macos")]
        let [cold, steady, gapped] = blake3_servil::__gpu_probe_tasks(mib, 20);
        #[cfg(not(target_os = "macos"))]
        let [cold, steady, gapped] = [0u64; 3];
        let blocks = vec![blake3_servil::performance::Block::ZERO; (mib << 20) / 64];
        let len = mib << 20;
        for _ in 0..5 {
            std::hint::black_box(blake3_servil::performance::hash_multithreaded(blake3_servil::Mode::Plain, &blocks, len));
        }
        let reps = 20u64;
        let t0 = clocks::now();
        for _ in 0..reps {
            std::hint::black_box(blake3_servil::performance::hash_multithreaded(blake3_servil::Mode::Plain, &blocks, len));
        }
        let cpu = clocks::since_ns(t0) / reps;
        let bytes = (mib as u64) << 20;
        println!(
            "{mib}\t{}\t{}\t{}\t{}\t{}\t{}",
            (cold + 500) / 1000,
            (steady + 500) / 1000,
            (gapped + 500) / 1000,
            (bytes + steady.max(1) / 2) / steady.max(1),
            (cpu + 500) / 1000,
            (bytes + cpu / 2) / cpu
        );
    }
    clocks::load::tick();
    println!("# load: {}", clocks::load::describe(&clocks::load::windows()));
}
