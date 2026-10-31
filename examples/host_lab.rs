//! Probe (branches probe/metal-sustained-old and -new, one source): streams
//! of long messages through the public stream API, written in
//! (`submit_with`), on whichever build this is (servil, CPU only; or the
//! GPU candidate). `host_lab PART`, PART one of:
//!
//! - `sustained`: one stream of about 2 GiB of messages of 4, 16, or 64 MiB
//!   a call, back to back: the rate, and the CPU cores busy.
//! - `bursts`: a stream of two such messages a call, each after the thread
//!   idles 100 ms, as the benchmark's samples come.
//! - `power`: 10 s phases of sustained streams, for powermetrics; each
//!   phase's start and end in Unix milliseconds.
//!
//! Every timing comes from `clocks`; rates are each case's mean.
use blake3_servil::{Hash, Mode, performance};
use std::hint::black_box;

const MIB: usize = 1024 * 1024;

fn threads() -> i32 {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // Sound: info is a proc_taskinfo of `size` bytes.
    let n = unsafe { libc::proc_pidinfo(libc::getpid(), libc::PROC_PIDTASKINFO, 0, (&mut info as *mut libc::proc_taskinfo).cast(), size) };
    assert_eq!(n, size, "proc_pidinfo");
    info.pti_threadnum
}

/// One stream: a copy of `src` written into it for each of `places`.
fn stream_all(src: &[u8], places: &mut [Hash], expected: Hash) {
    performance::stream(
        Mode::Plain,
        |_: &mut [Hash]| {},
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
    assert!(places.iter().all(|p| *p == expected));
}

fn source(len: usize) -> (Vec<u8>, Hash) {
    let src: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let expected = blake3_servil::convenience::hash(&src);
    (src, expected)
}

fn unix_ms() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

fn main() {
    let part = std::env::args().nth(1).expect("host_lab PART");
    blake3_servil::initialize_multithreaded(None);
    println!("# part {part}; threads after initialize_multithreaded: {}", threads());
    match part.as_str() {
        "sustained" => {
            println!("case,message_bytes,messages,calls,ns_per_call,MB_s,cores");
            for len in [4 * MIB, 16 * MIB, 64 * MIB] {
                let (src, expected) = source(len);
                let count = 2048 * MIB / len;
                let mut places = vec![Hash::from_bytes([0; 32]); count];
                stream_all(&src, &mut places, expected);
                let samples = clocks::measure(6, 300_000_000, || stream_all(black_box(&src), &mut places, expected));
                let (ns, calls) = clocks::summary::mean_totals(samples.iter().map(|s| (s.wall_ns, s.calls)));
                let cpu: u128 = samples.iter().map(|s| u128::from(s.cpu_ns.unwrap_or(0))).sum();
                let bytes = (len * count) as u128 * calls;
                println!("sustained,{len},{count},{calls},{},{},{}", ns / calls, (bytes * 1000 + ns / 2) / ns, cpu * 100 / ns);
            }
        }
        "bursts" => {
            println!("case,message_bytes,messages,calls,ns_per_call,MB_s");
            for len in [4 * MIB, 16 * MIB, 64 * MIB] {
                let (src, expected) = source(len);
                let mut places = vec![Hash::from_bytes([0; 32]); 2];
                let calls = 20;
                let b = clocks::measure_after_gaps_prepared(calls, clocks::Gap::Idle, 100_000_000, &mut (), |_| {}, |_| stream_all(black_box(&src), &mut places, expected));
                let ns = u128::from(b.calls.wall_ns);
                let bytes = (2 * len) as u128 * u128::from(calls);
                println!("bursts,{len},2,{calls},{},{}", ns / u128::from(calls), (bytes * 1000 + ns / 2) / ns);
            }
        }
        "power" => {
            println!("phase,name,start_unix_ms,end_unix_ms,bytes,MB_s");
            for round in 0..2 {
                for len in [16 * MIB, 64 * MIB] {
                    let (src, expected) = source(len);
                    let count = 1024 * MIB / len;
                    let mut places = vec![Hash::from_bytes([0; 32]); count];
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let (start_ms, start) = (unix_ms(), clocks::now());
                    let mut bytes = 0usize;
                    while clocks::since_ns(start) < 10_000_000_000 {
                        stream_all(&src, &mut places, expected);
                        bytes += len * count;
                    }
                    let ns = clocks::since_ns(start);
                    println!("phase,round {round} sustained {} MiB,{start_ms},{},{bytes},{}", len / MIB, unix_ms(), (bytes as u128 * 1000 + u128::from(ns) / 2) / u128::from(ns));
                }
            }
        }
        _ => panic!("unknown part {part}"),
    }
    println!("# threads at the end: {}", threads());
    #[cfg(blake3_gpu)]
    println!("# long messages: on the GPU, on the CPU from a GPU slot, races the CPU won: {:?}", blake3_servil::__gpu_counts());
}
