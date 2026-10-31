//! Probe (branch probe/metal-gpu): how the Metal GPU path behaves on this
//! Mac, beside the CPU. `host_lab PART`, PART one of:
//!
//! - `threads`: the process's threads before and after Metal starts, the
//!   first `Hasher::new` (the shader compiles) and the first call, and the
//!   GPU's digests against the CPU's at awkward lengths.
//! - `oneshot`: one message a call, back to back, by size: the CPU pool,
//!   the GPU from a copy, from the caller's pages (no copy), from its own
//!   memory (prepared), CPU/GPU splits, and the no-copy wrap alone.
//! - `gaps`: the same calls, each after the thread sleeps 1, 10, or 100 ms.
//! - `ramp`: the GPU's speed from idle under a queue of 64 MiB messages,
//!   and how fast one message runs after a burst and a pause.
//! - `stream`: streams of messages by size: the crate's stream, writing
//!   each message into its ring, against the GPU with four messages in
//!   flight, each written into its memory or already there.
//! - `power`: fixed phases of each kind of work, for powermetrics; each
//!   phase's start and end in Unix milliseconds.
//!
//! Every timing comes from `clocks`; rates are each case's mean (total
//! bytes over total time).
use blake3_servil::metal;
use blake3_servil::performance::{self, Block};
use blake3_servil::{Hash, Mode};
use std::collections::VecDeque;
use std::hint::black_box;

const KIB: usize = 1024;
const MIB: usize = 1024 * 1024;
const PAGE: usize = 16 * KIB;

/// The process's threads now.
fn threads() -> i32 {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // Sound: info is a proc_taskinfo of `size` bytes.
    let n = unsafe { libc::proc_pidinfo(libc::getpid(), libc::PROC_PIDTASKINFO, 0, (&mut info as *mut libc::proc_taskinfo).cast(), size) };
    assert_eq!(n, size, "proc_pidinfo");
    info.pti_threadnum
}

/// A page-aligned input of `len` bytes (whole pages kept, the pattern
/// `i % 251`), for the life of the process.
fn input(len: usize) -> &'static [Block] {
    let bytes = len.div_ceil(PAGE).max(1) * PAGE;
    let layout = std::alloc::Layout::from_size_align(bytes, PAGE).unwrap();
    // Sound: a nonzero size; the memory is leaked, kept for the process.
    let p = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!p.is_null());
    let blocks = unsafe { std::slice::from_raw_parts_mut(p.cast::<Block>(), bytes / 64) };
    for (i, b) in Block::bytes_mut(blocks).iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    blocks
}

fn mb_s(bytes: u128, ns: u128) -> u128 {
    (bytes * 1000 + ns / 2) / ns.max(1)
}

/// Measure `f`, which hashes `bytes` a call, back to back: one line.
fn nonstop(case: &str, bytes: usize, mut f: impl FnMut()) {
    f();
    let samples = clocks::measure(6, 50_000_000, &mut f);
    let (ns, calls) = clocks::summary::mean_totals(samples.iter().map(|s| (s.wall_ns, s.calls)));
    let cpu: u128 = samples.iter().map(|s| u128::from(s.cpu_ns.unwrap_or(0))).sum();
    println!(
        "nonstop,{case},{bytes},{},{},{},{}",
        calls,
        ns / calls,
        mb_s(bytes as u128 * calls, ns),
        cpu * 100 / ns.max(1)
    );
}

/// Measure `f` after idle gaps of `gap_ms`: one line.
fn after_gap(case: &str, bytes: usize, gap_ms: u64, mut f: impl FnMut()) {
    let calls = (300 / gap_ms).clamp(8, 40);
    let b = clocks::measure_after_gaps_prepared(calls, clocks::Gap::Idle, gap_ms * 1_000_000, &mut (), |_| {}, |_| f());
    let ns = u128::from(b.calls.wall_ns);
    println!("gap,{case},{bytes},{gap_ms},{},{},{}", calls, ns / u128::from(calls), mb_s(bytes as u128 * u128::from(calls), ns));
}

/// The GPU hashes `count` copies of `src` with up to four in flight in
/// its slots: written into a slot's memory first when `fill`, else read
/// from the slots as they are (each already holding `src`).
fn gpu_stream(gpu: &mut metal::Hasher, src: &[u8], count: usize, fill: bool, expected: Hash) {
    let mut in_flight = VecDeque::with_capacity(4);
    let mut free: Vec<usize> = (0..4).collect();
    for _ in 0..count {
        let slot = match free.pop() {
            Some(slot) => slot,
            None => {
                let slot = in_flight.pop_front().unwrap();
                assert_eq!(gpu.wait(slot), expected);
                slot
            }
        };
        if fill {
            gpu.slot(slot, src.len()).copy_from_slice(src);
        }
        gpu.submit(slot, src.len());
        in_flight.push_back(slot);
    }
    while let Some(slot) = in_flight.pop_front() {
        assert_eq!(gpu.wait(slot), expected);
    }
}

/// Every slot holding `src`.
fn prefill(gpu: &mut metal::Hasher, src: &[u8]) {
    for slot in 0..4 {
        gpu.slot(slot, src.len()).copy_from_slice(src);
    }
}

/// The crate's stream hashes a copy of `src` into each of `places`,
/// written into its ring piece by piece (`submit_with`).
fn cpu_stream(src: &[u8], places: &mut [Hash], expected: Hash) {
    let handed = std::cell::Cell::new(0usize);
    performance::stream(
        Mode::Plain,
        |p: &mut [Hash]| handed.set(handed.get() + p.len()),
        |s| {
            for out in places.iter_mut() {
                s.submit_with(
                    src.len(),
                    |at, room| {
                        room.copy_from_slice(&src[at..][..room.len()]);
                        room.len()
                    },
                    out,
                );
            }
            s.finish();
        },
    );
    assert_eq!(handed.get(), places.len());
    assert!(places.iter().all(|p| *p == expected));
}

fn unix_ms() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

fn main() {
    let part = std::env::args().nth(1).expect("host_lab PART");
    let t0 = threads();
    blake3_servil::initialize_multithreaded(None);
    let t1 = threads();
    let start = clocks::now();
    let mut gpu = metal::Hasher::new().expect("Metal");
    let new_ns = clocks::since_ns(start);
    let t2 = threads();
    println!("# part {part}; threads: {t0} at start, {t1} after initialize_multithreaded, {t2} after the first metal::Hasher::new ({new_ns} ns)");
    match part.as_str() {
        "threads" => part_threads(&mut gpu),
        "oneshot" => part_oneshot(&mut gpu),
        "gaps" => part_gaps(&mut gpu),
        "ramp" => part_ramp(&mut gpu),
        "stream" => part_stream(&mut gpu),
        "power" => part_power(&mut gpu),
        _ => panic!("unknown part {part}"),
    }
    clocks::load::tick();
    let windows = clocks::load::windows();
    println!("# load: {}", clocks::load::describe(&windows));
}

fn part_threads(gpu: &mut metal::Hasher) {
    let start = clocks::now();
    let second = metal::Hasher::new().expect("Metal");
    println!("# a second metal::Hasher::new: {} ns", clocks::since_ns(start));
    drop(second);
    let one = input(MIB);
    let bytes = &Block::bytes(one)[..MIB];
    let start = clocks::now();
    let digest = gpu.hash(bytes, 0);
    println!("# the first GPU call, 1 MiB from a copy: {} ns; threads now {}", clocks::since_ns(start), threads());
    assert_eq!(digest, blake3_servil::convenience::hash(bytes));
    let start = clocks::now();
    let digest = gpu.hash_nocopy(bytes);
    println!("# the first no-copy GPU call, 1 MiB: {} ns; threads now {}", clocks::since_ns(start), threads());
    assert_eq!(digest, blake3_servil::convenience::hash(bytes));
    std::thread::sleep(std::time::Duration::from_millis(300));
    println!("# threads after 300 ms idle: {}", threads());
    // Digests at awkward lengths: every path against the CPU's.
    let big = input(9 * MIB);
    let all = Block::bytes(big);
    let lens = [0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 2049, 4097, 16 * KIB + 1, 65537, MIB - 1, MIB + 1, 2 * MIB + 17, 3 * MIB + 17, 8 * MIB + 1023];
    for len in lens {
        let m = &all[..len];
        let expected = blake3_servil::convenience::hash(m);
        assert_eq!(gpu.hash(m, 0), expected, "copy, len {len}");
        if len >= 2 * MIB {
            for p in [1, 25, 50, 75, 99] {
                assert_eq!(gpu.hash(m, p), expected, "split {p}%, len {len}");
            }
        }
        if len > 0 {
            assert_eq!(gpu.hash_nocopy(m), expected, "no copy, len {len}");
        }
        gpu.slot(0, len.max(1))[..len].copy_from_slice(m);
        gpu.submit(0, len);
        assert_eq!(gpu.wait(0), expected, "slot, len {len}");
    }
    println!("# digests agree at {} lengths, every path", lens.len());
}

const SIZES: [usize; 7] = [64 * KIB, 256 * KIB, MIB, 4 * MIB, 16 * MIB, 64 * MIB, 128 * MIB];

fn part_oneshot(gpu: &mut metal::Hasher) {
    println!("kind,case,bytes,calls,ns_per_call,MB_s,cpu_percent_of_wall");
    for len in SIZES {
        let blocks = input(len);
        let bytes = &Block::bytes(blocks)[..len];
        let expected = blake3_servil::convenience::hash(bytes);
        assert_eq!(performance::hash_multithreaded(Mode::Plain, blocks, len), expected);
        nonstop("cpu pool", len, || assert_eq!(performance::hash_multithreaded(Mode::Plain, black_box(blocks), len), expected));
        nonstop("gpu from a copy", len, || assert_eq!(gpu.hash(black_box(bytes), 0), expected));
        nonstop("gpu no copy", len, || assert_eq!(gpu.hash_nocopy(black_box(bytes)), expected));
        nonstop("no-copy wrap alone", len, || gpu.wrap_only(black_box(bytes)));
        prefill(gpu, bytes);
        nonstop("gpu prepared", len, || {
            gpu.submit(0, len);
            assert_eq!(gpu.wait(0), expected);
        });
        if len >= 2 * MIB {
            for p in [25, 50, 75] {
                nonstop(&format!("split cpu {p}%"), len, || assert_eq!(gpu.hash(black_box(bytes), p), expected));
            }
        }
    }
}

fn part_gaps(gpu: &mut metal::Hasher) {
    println!("kind,case,bytes,gap_ms,calls,ns_per_call,MB_s");
    for len in [MIB, 16 * MIB, 128 * MIB] {
        let blocks = input(len);
        let bytes = &Block::bytes(blocks)[..len];
        let expected = blake3_servil::convenience::hash(bytes);
        prefill(gpu, bytes);
        for gap_ms in [1, 10, 100] {
            after_gap("cpu pool", len, gap_ms, || assert_eq!(performance::hash_multithreaded(Mode::Plain, black_box(blocks), len), expected));
            after_gap("gpu from a copy", len, gap_ms, || assert_eq!(gpu.hash(black_box(bytes), 0), expected));
            after_gap("gpu prepared", len, gap_ms, || {
                gpu.submit(0, len);
                assert_eq!(gpu.wait(0), expected);
            });
            if len >= 2 * MIB {
                after_gap("split cpu 50%", len, gap_ms, || assert_eq!(gpu.hash(black_box(bytes), 50), expected));
            }
        }
    }
}

fn part_ramp(gpu: &mut metal::Hasher) {
    let len = 64 * MIB;
    let blocks = input(len);
    let bytes = &Block::bytes(blocks)[..len];
    let expected = blake3_servil::convenience::hash(bytes);
    prefill(gpu, bytes);
    // From idle: each message's completion, four in flight.
    std::thread::sleep(std::time::Duration::from_secs(2));
    let start = clocks::now();
    let mut in_flight = VecDeque::new();
    let mut done_ns = Vec::new();
    for k in 0..160 {
        if in_flight.len() == 4 {
            let slot = in_flight.pop_front().unwrap();
            assert_eq!(gpu.wait(slot), expected);
            done_ns.push(clocks::since_ns(start));
        }
        let slot = k % 4;
        gpu.submit(slot, len);
        in_flight.push_back(slot);
    }
    while let Some(slot) = in_flight.pop_front() {
        assert_eq!(gpu.wait(slot), expected);
        done_ns.push(clocks::since_ns(start));
    }
    println!("ramp,message,done_ns,interval_ns,MB_s");
    let mut last = 0;
    for (k, t) in done_ns.iter().enumerate() {
        println!("ramp,{k},{t},{},{}", t - last, mb_s(len as u128, u128::from(t - last)));
        last = *t;
    }
    // After a warm burst and a pause: one message alone.
    println!("pause,pause_us,ns,MB_s");
    for pause_us in [0u64, 200, 1000, 5000, 20_000, 100_000, 500_000] {
        for _ in 0..3 {
            gpu_stream(gpu, bytes, 24, false, expected);
            std::thread::sleep(std::time::Duration::from_micros(pause_us));
            let start = clocks::now();
            gpu.submit(0, len);
            assert_eq!(gpu.wait(0), expected);
            let ns = clocks::since_ns(start);
            println!("pause,{pause_us},{ns},{}", mb_s(len as u128, u128::from(ns)));
        }
    }
}

fn part_stream(gpu: &mut metal::Hasher) {
    println!("kind,case,bytes,calls,ns_per_call,MB_s,cpu_percent_of_wall");
    for len in [MIB, 4 * MIB, 8 * MIB, 16 * MIB, 64 * MIB] {
        let count = (256 * MIB / len).max(16);
        let blocks = input(len);
        let src = &Block::bytes(blocks)[..len];
        let expected = blake3_servil::convenience::hash(src);
        let mut places = vec![Hash::from_bytes([0; 32]); count];
        let total = len * count;
        nonstop(&format!("cpu stream, {count} messages written in"), total, || cpu_stream(black_box(src), &mut places, expected));
        nonstop(&format!("gpu, {count} messages written in"), total, || gpu_stream(gpu, black_box(src), count, true, expected));
        prefill(gpu, src);
        nonstop(&format!("gpu, {count} messages in place"), total, || gpu_stream(gpu, black_box(src), count, false, expected));
    }
}

fn part_power(gpu: &mut metal::Hasher) {
    let len = 64 * MIB;
    let blocks = input(len);
    let src = &Block::bytes(blocks)[..len];
    let expected = blake3_servil::convenience::hash(src);
    let mut places = vec![Hash::from_bytes([0; 32]); 4];
    prefill(gpu, src);
    println!("phase,name,start_unix_ms,end_unix_ms,bytes,MB_s");
    let phase = |name: &str, f: &mut dyn FnMut() -> usize| {
        std::thread::sleep(std::time::Duration::from_secs(4));
        let (start_ms, start) = (unix_ms(), clocks::now());
        let mut bytes = 0usize;
        while clocks::since_ns(start) < 10_000_000_000 {
            bytes += f();
        }
        let ns = clocks::since_ns(start);
        println!("phase,{name},{start_ms},{},{bytes},{}", unix_ms(), mb_s(bytes as u128, u128::from(ns)));
    };
    for round in 0..2 {
        println!("# round {round}");
        phase("idle", &mut || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0
        });
        phase("cpu stream, written in", &mut || {
            cpu_stream(src, &mut places, expected);
            4 * len
        });
        phase("gpu, written in", &mut || {
            gpu_stream(gpu, src, 4, true, expected);
            4 * len
        });
        phase("gpu, in place", &mut || {
            gpu_stream(gpu, src, 4, false, expected);
            4 * len
        });
        phase("cpu pool one-shot", &mut || {
            assert_eq!(performance::hash_multithreaded(Mode::Plain, blocks, len), expected);
            len
        });
        phase("gpu no-copy one-shot", &mut || {
            assert_eq!(gpu.hash_nocopy(src), expected);
            len
        });
        prefill(gpu, src);
    }
}
