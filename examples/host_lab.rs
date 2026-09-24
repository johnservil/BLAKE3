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

/// Best of 7 batches, each about 3 ms, of `f`; ns per call.
fn time(mut f: impl FnMut()) -> f64 {
    let started = Instant::now();
    let mut reps = 0u64;
    while started.elapsed().as_micros() < 3000 {
        f();
        reps += 1;
    }
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / reps as f64);
    }
    best
}

/// A dependent scalar chain: ns per 1000 steps. P and E differ by clock alone.
fn reference() -> f64 {
    time(|| {
        let mut x = black_box(0x1234_5678_9abc_def0u64);
        for _ in 0..1000 {
            x = (x ^ (x >> 7)).rotate_left(13).wrapping_add(0x9e37_79b9_7f4a_7c15);
        }
        black_box(x);
    })
}

fn main() {
    let data: Vec<u8> = (0..(1 << 20)).map(|i| (i * 7 + (i >> 11)) as u8).collect();
    let chunks: Vec<&[u8; 1024]> = data.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
    let mut out = vec![0u8; 32 * chunks.len()];
    let neon = Platform::neon().expect("NEON");
    let detected = Platform::detect();
    let portable = Platform::portable();
    println!("probe: ecore kernels; detected platform chunks via {}", detected.hash_many_name());
    let classes = [("P (user-interactive)", 0x21u32), ("E (background)", 0x09u32)];
    let sizes = [64usize, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 65536];
    for round in 0..3 {
        for &(name, class) in &classes {
            set_qos(class);
            // Let the scheduler move the thread.
            let t = Instant::now();
            while t.elapsed().as_millis() < 30 {
                black_box(reference());
            }
            println!("\nround {round} · {name} · reference {:.0} ns / 1000 steps", reference());
            let mut line = String::from("  NEON chunks, ns/B by count:");
            for n in 1..=16 {
                let ns = time(|| unsafe_many(&neon, &chunks[..n], &mut out));
                line += &format!(" {n}:{:.3}", ns / (n * 1024) as f64);
            }
            println!("{line}");
            let mut line = String::from("  detected platform chunks, ns/B:");
            for n in [16, 32, 64, 128] {
                let ns = time(|| unsafe_many(&detected, &chunks[..n], &mut out));
                line += &format!(" {n}:{:.3}", ns / (n * 1024) as f64);
            }
            println!("{line}");
            let mut line = String::from("  portable chunks, ns/B:");
            for n in [1, 4, 16] {
                let ns = time(|| unsafe_many(&portable, &chunks[..n], &mut out));
                line += &format!(" {n}:{:.3}", ns / (n * 1024) as f64);
            }
            println!("{line}");
            let mut line = String::from("  hash(), ns/B by size:");
            for &len in &sizes {
                let ns = time(|| {
                    black_box(blake3_servil::hash(black_box(&data[..len])));
                });
                line += &format!(" {len}:{:.3}", ns / len as f64);
            }
            println!("{line}");
            println!("  reference after: {:.0} ns / 1000 steps", reference());
        }
    }
}

fn unsafe_many(platform: &Platform, inputs: &[&[u8; 1024]], out: &mut [u8]) {
    platform.hash_many::<1024>(black_box(inputs), &IV, 0, IncrementCounter::Yes, 0, 1, 2, out);
    black_box(&out);
}
