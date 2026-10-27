//! Hash a file through the Metal stream, or measure streams of messages on
//! the CPU and on the GPU with the fork's clocks.
use blake3_servil::{Hash, Mode, convenience, initialize_multithreaded, metal, performance};
use std::hint::black_box;

/// Bytes a stream hashes in one timed call, at least 16 messages: enough
/// that opening and finishing the stream (the GPU's queue drains) costs
/// little of it.
const STREAM_CALL_BYTES: usize = 256 << 20;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    initialize_multithreaded(None);
    assert!(
        args.len() == 2 && ["--file", "--stream", "--tiny"].contains(&args[0].as_str()),
        "usage: metal --file FILE | metal --stream MiB | metal --tiny BYTES"
    );
    let mut gpu = metal::Hasher::new().expect("Metal initialization");
    if args[0] == "--file" {
        let input = std::fs::read(&args[1]).expect("read input");
        let mut digest = Hash::from_bytes([0; 32]);
        gpu.stream(
            Mode::Plain,
            |_: &mut [Hash]| {},
            |s| {
                s.submit_with(
                    input.len(),
                    |at, room| {
                        room.copy_from_slice(&input[at..][..room.len()]);
                        room.len()
                    },
                    &mut digest,
                );
                s.finish();
            },
        );
        println!("{digest}  {}", args[1]);
        return;
    }
    if args[0] == "--tiny" {
        // Short messages, copied in: the crate's stream against the Metal
        // stream, which sends them to the crate's: the routing's cost.
        let len: usize = args[1].parse().expect("bytes");
        let messages: Vec<Vec<u8>> = (0..4096u32).map(|i| i.to_le_bytes().repeat(len.div_ceil(4))[..len].to_vec()).collect();
        let mut places = vec![Hash::from_bytes([0; 32]); messages.len()];
        for (case, on_gpu) in [("crate stream", false), ("metal stream", true)] {
            let samples = clocks::measure(8, 250_000_000, || {
                let messages = black_box(&messages);
                let count = std::cell::Cell::new(0);
                let done = |place: &mut [Hash]| count.set(count.get() + place.len());
                fn copy(m: &[u8]) -> impl FnMut(usize, &mut [u8]) -> usize + '_ {
                    move |at, room| {
                        room.copy_from_slice(&m[at..][..room.len()]);
                        room.len()
                    }
                }
                if on_gpu {
                    gpu.stream(Mode::Plain, done, |s| {
                        for (m, out) in messages.iter().zip(&mut places) {
                            s.submit_with(m.len(), copy(m), out);
                        }
                        s.finish();
                    });
                } else {
                    performance::stream(Mode::Plain, done, |s| {
                        for (m, out) in messages.iter().zip(&mut places) {
                            s.submit_with(m.len(), copy(m), out);
                        }
                        s.finish();
                    });
                }
                assert_eq!(count.get(), messages.len());
            });
            let (ns, calls) = clocks::summary::mean_totals(samples.iter().map(|s| (s.wall_ns, s.calls)));
            eprintln!("{case}: {} ns a message", ns / (calls * messages.len() as u128));
        }
        return;
    }
    let mib: usize = args[1].parse().expect("MiB");
    assert!((1..=1024).contains(&mib), "MiB must be 1..=1024");
    let input: Vec<_> = (0..mib * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let expected = convenience::hash(&input);
    // Each call streams copies of the input, as the benchmark's continuous
    // messages do: each message copied into the stream's own memory
    // (submit_with), digests handed back in order. Prepared: the slots and
    // the CPU stream's messages already hold the message, so no copy is
    // made (every message is the same input). The GPU-only hasher takes
    // every message, whatever the device's threshold.
    let messages = (STREAM_CALL_BYTES / input.len()).max(16);
    let mut places = vec![Hash::from_bytes([0; 32]); messages];
    let bytes = (input.len() * messages) as u128;
    let mut gpu_only = metal::Hasher::with_gpu_from(0).expect("Metal initialization");
    let mut results = Vec::new();
    for (case, device, prepared) in [
        ("CPU stream", 0, false),
        ("Metal stream", 1, false),
        ("GPU only", 2, false),
        ("CPU stream", 0, true),
        ("Metal stream", 1, true),
        ("GPU only", 2, true),
    ] {
        let mut run = || {
            let message = black_box(&input);
            let handed_back = std::cell::Cell::new(0);
            let done = |place: &mut [Hash]| handed_back.set(handed_back.get() + place.len());
            let fill = |at: usize, room: &mut [u8]| {
                let message = &message[at..][..room.len()];
                if !prepared || room[..64] != message[..64] || room[room.len() - 64..] != message[message.len() - 64..] {
                    room.copy_from_slice(message)
                }
                room.len()
            };
            match device {
                0 => performance::stream(Mode::Plain, done, |s| {
                    for out in places.iter_mut() {
                        s.submit_with(message.len(), fill, out);
                    }
                    s.finish();
                }),
                1 => gpu.stream(Mode::Plain, done, |s| {
                    for out in places.iter_mut() {
                        s.submit_with(message.len(), fill, out);
                    }
                    s.finish();
                }),
                _ => gpu_only.stream(Mode::Plain, done, |s| {
                    for out in places.iter_mut() {
                        s.submit_with(message.len(), fill, out);
                    }
                    s.finish();
                }),
            }
            assert_eq!(handed_back.get(), messages);
            assert!(places.iter().all(|place| *place == expected));
        };
        run();
        let samples = clocks::measure(8, 250_000_000, &mut run);
        let how = if prepared { "prepared" } else { "copied" };
        results.push((format!("{case} {how}"), bytes, samples));
    }
    clocks::load::tick();
    let windows = clocks::load::windows();
    eprintln!("{}", clocks::load::describe(&windows));
    eprintln!("Diagnostic sweep; use bench-hashes check --metal for a speed verdict.");
    eprintln!("Cycles describe the calling thread; cpu_ns covers all process threads.");
    println!("case,bytes,calls,wall_ns,cpu_ns,p_cycles,e_cycles,started_ns");
    for (case, bytes, samples) in results {
        let (ns, calls) = clocks::summary::mean_totals(samples.iter().map(|s| (s.wall_ns, s.calls)));
        let mb_s = (bytes * calls * 1000 + ns / 2) / ns;
        let busy = samples.iter().any(|s| windows.iter().any(|w| w.holds(s.started_ns) && w.busy()));
        let covered = samples.iter().all(|s| windows.iter().any(|w| w.holds(s.started_ns)));
        let note = if busy {
            " (busy window; no speed conclusion)"
        } else if !covered {
            " (load window incomplete; no speed conclusion)"
        } else {
            ""
        };
        eprintln!("{case}: {mb_s} MB/s{note}");
        for s in samples {
            let cycles = s.counts.map(|c| format!("{},{}", c.p.cycles, c.e.cycles)).unwrap_or_else(|| "NA,NA".into());
            println!("{case},{bytes},{},{},{},{cycles},{}", s.calls, s.wall_ns, s.cpu_ns.map(|n| n.to_string()).unwrap_or_else(|| "NA".into()), s.started_ns);
        }
    }
}
