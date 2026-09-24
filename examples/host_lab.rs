//! Probe (branch probe/ecore-kernels, never merged): every NEON chunk plan,
//! SME2, portable, and hash() by size, at user-interactive QoS (P-cores) and
//! background QoS (E-cores on macOS), with a fixed scalar reference loop
//! that shows which core kind a measurement ran on.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;
use std::hint::black_box;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

#[cfg(target_vendor = "apple")]
fn set_qos(class: u32) {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, relpri: i32) -> i32;
    }
    assert_eq!(unsafe { pthread_set_qos_class_self_np(class, 0) }, 0);
}
#[cfg(not(target_vendor = "apple"))]
fn set_qos(_class: u32) {}


/// (P cycles, E cycles) this thread has run, from thread_selfcounts.
#[cfg(target_vendor = "apple")]
fn cycles() -> (u64, u64) {
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Cpi { instructions: u64, cycles: u64, user: u64, system: u64 }
    unsafe extern "C" {
        fn thread_selfcounts(kind: u32, dst: *mut std::ffi::c_void, size: usize) -> i32;
    }
    let mut levels = [Cpi::default(); 2];
    assert_eq!(unsafe { thread_selfcounts(4, levels.as_mut_ptr().cast(), std::mem::size_of_val(&levels)) }, 0);
    (levels[0].cycles, levels[1].cycles)
}
#[cfg(not(target_vendor = "apple"))]
fn cycles() -> (u64, u64) { (0, 0) }

/// Best of 7 batches of `f`, each about 3 ms: (cycles per call, share of
/// the batch's cycles on E-cores, GHz). Off Apple: ns per call, 0, 0.
fn time(mut f: impl FnMut()) -> (f64, f64, f64) {
    let started = Instant::now();
    let mut reps = 0u64;
    while started.elapsed().as_micros() < 3000 {
        f();
        reps += 1;
    }
    let mut best = (f64::MAX, 0.0, 0.0);
    for _ in 0..7 {
        let (p0, e0) = cycles();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64;
        let (p1, e1) = cycles();
        let total = ((p1 - p0) + (e1 - e0)) as f64;
        let per_call = if total > 0.0 { total / reps as f64 } else { ns / reps as f64 };
        if per_call < best.0 {
            best = (per_call, if total > 0.0 { (e1 - e0) as f64 / total } else { 0.0 }, total / ns);
        }
    }
    best
}

/// A dependent scalar chain: cycles (or ns) per 1000 steps.
fn reference() -> (f64, f64, f64) {
    time(|| {
        let mut x = black_box(0x1234_5678_9abc_def0u64);
        for _ in 0..1000 {
            x = (x ^ (x >> 7)).rotate_left(13).wrapping_add(0x9e37_79b9_7f4a_7c15);
        }
        black_box(x);
    })
}

fn cell(label: impl std::fmt::Display, t: (f64, f64, f64), bytes: usize) -> String {
    format!(" {label}:{:.2}{}", t.0 / bytes as f64, if t.1 > 0.05 && t.1 < 0.95 { "~" } else { "" })
}

fn main() {
    let data: Vec<u8> = (0..(1 << 20)).map(|i| (i * 7 + (i >> 11)) as u8).collect();
    let chunks: Vec<&[u8; 1024]> = data.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
    let mut out = vec![0u8; 32 * chunks.len()];
    let neon = Platform::neon().expect("NEON");
    let detected = Platform::detect();
    let portable = Platform::portable();
    println!("probe: ecore kernels; detected platform chunks via {}", detected.hash_many_name());
    let classes = [("user-interactive", 0x21u32), ("utility", 0x11u32), ("background", 0x09u32)];
    let sizes = [64usize, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 65536];
    for round in 0..3 {
        for &(name, class) in &classes {
            set_qos(class);
            // Let the scheduler move the thread.
            let t = Instant::now();
            while t.elapsed().as_millis() < 30 {
                black_box(reference());
            }
            let r = reference();
            println!("\nround {round} · {name} · E share {:.2} · {:.2} GHz · reference {:.0} cycles / 1000 steps (cycles per byte below; ~ marks a mixed batch)", r.1, r.2, r.0);
            let mut line = String::from("  NEON chunks by count:");
            for n in 1..=16 {
                line += &cell(n, time(|| unsafe_many(&neon, &chunks[..n], &mut out)), n * 1024);
            }
            println!("{line}");
            let mut line = String::from("  detected platform chunks:");
            for n in [16, 32, 64, 128] {
                line += &cell(n, time(|| unsafe_many(&detected, &chunks[..n], &mut out)), n * 1024);
            }
            println!("{line}");
            let mut line = String::from("  portable chunks:");
            for n in [1, 4, 16] {
                line += &cell(n, time(|| unsafe_many(&portable, &chunks[..n], &mut out)), n * 1024);
            }
            println!("{line}");
            let mut line = String::from("  hash() by size:");
            for &len in &sizes {
                line += &cell(len, time(|| { black_box(blake3_servil::hash(black_box(&data[..len]))); }), len);
            }
            println!("{line}");
            let r = reference();
            println!("  after: E share {:.2} · {:.2} GHz · reference {:.0}", r.1, r.2, r.0);
        }
    }
}

fn unsafe_many(platform: &Platform, inputs: &[&[u8; 1024]], out: &mut [u8]) {
    platform.hash_many::<1024>(black_box(inputs), &IV, 0, IncrementCounter::Yes, 0, 1, 2, out);
    black_box(&out);
}
