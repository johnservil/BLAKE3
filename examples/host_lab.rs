//! Probe (branches probe/overlap-old and probe/overlap-new, never merged):
//! the same program on servil and on candidate/overlap-group, run as
//! alternating jobs. hash_many ns and cycles per call at remainders 0, 1,
//! 8, 12, 13, 14, 15 after 1 to 127 SME2 groups, by core kind, back to
//! back and with the digests read between calls.
use blake3_servil::Hash;
use std::hint::black_box;
use std::time::Instant;

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

fn main() {
    let mut counts = vec![];
    for base in [16usize, 32, 96, 240, 496, 1008, 2032] {
        for r in [0usize, 1, 8, 12, 13, 14, 15] {
            counts.push(base + r);
        }
    }
    let max = *counts.iter().max().unwrap();
    let data: Vec<u8> = (0..max * 64).map(|i| (i * 7 + (i >> 6)) as u8).collect();
    let msgs: Vec<&[u8]> = data.chunks(64).collect();
    for &n in &counts {
        let mut out = vec![Hash::from_bytes([0; 32]); n];
        blake3_servil::hash_many(&msgs[..n], &mut out);
        for (i, o) in out.iter().enumerate() {
            assert_eq!(*o, blake3_servil::hash(msgs[i]), "{n} messages, message {i}");
        }
    }
    println!("probe: hash_many by remainder; every digest agrees with hash(); ns per call, cycles in brackets");
    for &(label, class) in &[("P (user-interactive)", 0x21u32), ("E (background)", 0x09u32)] {
        set_qos(class);
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 20 {
            black_box(time(|| {}));
        }
        for read in [false, true] {
            println!("{label}, {}", if read { "digests read between calls" } else { "back to back" });
            for &n in &counts {
                let mut out = vec![Hash::from_bytes([0; 32]); n];
                let mut best = (f64::MAX, 0.0);
                for _round in 0..3 {
                    let r = time(|| {
                        blake3_servil::hash_many(black_box(&msgs[..n]), &mut out);
                        if read {
                            for d in &out {
                                black_box(d.as_bytes());
                            }
                        }
                    });
                    if r.0 < best.0 {
                        best = r;
                    }
                }
                println!("{n:>6} {:>9.0} [{:>7.0}]", best.0, best.1);
            }
        }
    }
}
