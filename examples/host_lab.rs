//! Probe (branch probe/neon-cold, never merged): the messages left over
//! below an SME2 group, four ways, by core kind and with or without work
//! between calls. V0 today (SME2 groups, then the NEON parent kernels);
//! V1 the NEON remainder first; V2 the remainder padded into one more
//! SME2 group from a scratch copy; V3 the remainder through hash() one
//! message at a time (the integer-only kernel, no NEON). ns per call and
//! cycles per call (thread_selfcounts), every variant checked against
//! hash() first.
use blake3_servil::platform::Platform;
use blake3_servil::{Hash, IncrementCounter};
use std::hint::black_box;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const FLAGS: u8 = 1 | 2 | 8;
const TABLE: usize = 128;

fn digests(p: Platform, blocks: &[&[u8; 64]], out: &mut [Hash]) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 32) };
    for (i, run) in blocks.chunks(TABLE).enumerate() {
        p.hash_many::<64>(run, &IV, 0, IncrementCounter::No, FLAGS, 0, 0, &mut bytes[i * TABLE * 32..]);
    }
}

fn variant(v: usize, msgs: &[&[u8]], blocks: &[&[u8; 64]], out: &mut [Hash]) {
    let n = blocks.len();
    let whole = n / 16 * 16;
    match v {
        0 => blake3_servil::hash_many(msgs, out),
        1 => {
            digests(Platform::NEON, &blocks[whole..], &mut out[whole..]);
            digests(Platform::SME2, &blocks[..whole], &mut out[..whole]);
        }
        2 => {
            digests(Platform::SME2, &blocks[..whole], &mut out[..whole]);
            if whole < n {
                let mut scratch = [0u8; 16 * 64];
                for (i, b) in blocks[whole..].iter().enumerate() {
                    scratch[i * 64..][..64].copy_from_slice(&b[..]);
                }
                let padded: Vec<&[u8; 64]> = scratch.chunks_exact(64).map(|c| c.try_into().unwrap()).collect();
                let mut cvs = [Hash::from_bytes([0; 32]); 16];
                digests(Platform::SME2, &padded, &mut cvs);
                out[whole..].copy_from_slice(&cvs[..n - whole]);
            }
        }
        _ => {
            digests(Platform::SME2, &blocks[..whole], &mut out[..whole]);
            for (o, m) in out[whole..].iter_mut().zip(&msgs[whole..]) {
                *o = blake3_servil::hash(m);
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
fn set_qos(class: u32) {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, relpri: i32) -> i32;
    }
    assert_eq!(unsafe { pthread_set_qos_class_self_np(class, 0) }, 0);
}
#[cfg(not(target_vendor = "apple"))]
fn set_qos(_class: u32) {}

#[cfg(target_vendor = "apple")]
fn cycles() -> u64 {
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Cpi { instructions: u64, cycles: u64, user: u64, system: u64 }
    unsafe extern "C" {
        fn thread_selfcounts(kind: u32, dst: *mut std::ffi::c_void, size: usize) -> i32;
    }
    let mut levels = [Cpi::default(); 2];
    assert_eq!(unsafe { thread_selfcounts(4, levels.as_mut_ptr().cast(), std::mem::size_of_val(&levels)) }, 0);
    levels[0].cycles + levels[1].cycles
}
#[cfg(not(target_vendor = "apple"))]
fn cycles() -> u64 { 0 }

/// Best of 9 batches of about 2 ms: (ns per call, cycles per call).
fn time(mut f: impl FnMut()) -> (f64, f64) {
    let started = Instant::now();
    let mut reps = 0u64;
    while started.elapsed().as_micros() < 2000 {
        f();
        reps += 1;
    }
    let mut best = (f64::MAX, f64::MAX);
    for _ in 0..9 {
        let c0 = cycles();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / reps as f64;
        let c = (cycles() - c0) as f64 / reps as f64;
        if ns < best.0 {
            best = (ns, c);
        }
    }
    best
}

const COUNTS: &[usize] = &[17, 20, 24, 28, 31, 40, 48, 100, 200, 500, 1000, 1009, 1016, 1023, 1024];

fn main() {
    assert!(matches!(Platform::detect(), Platform::SME2), "this probe needs SME2");
    let max = *COUNTS.iter().max().unwrap();
    let data: Vec<u8> = (0..max * 64).map(|i| (i * 7 + (i >> 6)) as u8).collect();
    let msgs: Vec<&[u8]> = data.chunks(64).collect();
    let blocks: Vec<&[u8; 64]> = data.chunks_exact(64).map(|c| c.try_into().unwrap()).collect();
    for &n in COUNTS {
        for v in 0..4 {
            let mut out = vec![Hash::from_bytes([0; 32]); n];
            variant(v, &msgs[..n], &blocks[..n], &mut out);
            for (i, o) in out.iter().enumerate() {
                assert_eq!(*o, blake3_servil::hash(msgs[i]), "variant {v}, {n} messages, message {i}");
            }
        }
    }
    println!("probe: SME2 remainders four ways; every variant agrees with hash(); ns per call, cycles in brackets");
    let classes = [("P (user-interactive)", 0x21u32), ("E (background)", 0x09u32)];
    for &(label, class) in &classes {
        set_qos(class);
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 20 {
            black_box(time(|| {}));
        }
        for read in [false, true] {
            println!("{label}, {}", if read { "digests read between calls" } else { "back to back" });
            println!("{:>6} {:>18} {:>18} {:>18} {:>18}", "n", "V0 today", "V1 NEON first", "V2 padded", "V3 hash()");
            for &n in COUNTS {
                let mut out = vec![Hash::from_bytes([0; 32]); n];
                let mut row = format!("{n:>6}");
                let mut results = [(0.0, 0.0); 4];
                for _round in 0..3 {
                    for v in 0..4 {
                        let r = time(|| {
                            variant(v, black_box(&msgs[..n]), &blocks[..n], &mut out);
                            if read {
                                for d in &out {
                                    black_box(d.as_bytes());
                                }
                            }
                        });
                        if results[v].0 == 0.0 || r.0 < results[v].0 {
                            results[v] = r;
                        }
                    }
                }
                for (ns, c) in results {
                    row += &format!(" {:>9.0} [{:>6.0}]", ns, c);
                }
                println!("{row}");
            }
        }
    }
}
