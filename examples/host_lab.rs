//! Probe (branch probe/ecore-kernels, never merged): chunk-kernel plans by
//! core kind, in cycles per byte (thread_selfcounts), each plan's output
//! checked against the portable implementation first.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;
use std::hint::black_box;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
type Kernel = unsafe extern "C" fn(*const *const u8, u64, *const u32, u64, u64, *mut u8);

unsafe extern "C" {
    fn blake3_hybrid_k1(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k2(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k3(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k4(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k5(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k6(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k8(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k9(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k10(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k4q(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k4pp(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k5q(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k6qp(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_k7qp(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
}

fn kernel(name: &str) -> (Kernel, usize) {
    match name {
        "k1" => (blake3_hybrid_k1, 1),
        "k2" => (blake3_hybrid_k2, 2),
        "k3" => (blake3_hybrid_k3, 3),
        "k4" => (blake3_hybrid_k4, 4),
        "k5" => (blake3_hybrid_k5, 5),
        "k6" => (blake3_hybrid_k6, 6),
        "k8" => (blake3_hybrid_k8, 8),
        "k9" => (blake3_hybrid_k9, 9),
        "k10" => (blake3_hybrid_k10, 10),
        "k4q" => (blake3_hybrid_k4q, 4),
        "k4pp" => (blake3_hybrid_k4pp, 4),
        "k5q" => (blake3_hybrid_k5q, 5),
        "k6qp" => (blake3_hybrid_k6qp, 6),
        "k7qp" => (blake3_hybrid_k7qp, 7),
        _ => panic!("no kernel {name}"),
    }
}

const PLANS: &[&str] = &[
    "k1", "k2", "k3", "k2+k1",
    "k4", "k4q", "k4pp", "k2+k2", "k3+k1",
    "k5", "k5q", "k4q+k1", "k3+k2",
    "k6", "k6qp", "k3+k3", "k4q+k2", "k5+k1", "k4pp+k2",
    "k4+k3", "k7qp", "k4q+k3", "k5q+k2",
    "k8", "k4q+k4q", "k9", "k10", "k8+k2",
    "k8+k4", "k8+k4q", "k6qp+k6qp",
    "k10+k6", "k8+k8", "k9+k7qp", "k10+k6qp",
];

fn run(plan: &str, table: &[*const u8], out: &mut [u8]) {
    let flags = 1u64 << 8 | 2u64 << 16 | 64u64 << 24;
    let mut done = 0;
    for name in plan.split('+') {
        let (k, n) = kernel(name);
        unsafe { k(table[done..].as_ptr(), 16, IV.as_ptr(), done as u64, flags, out[32 * done..].as_mut_ptr()) };
        done += n;
    }
}

fn count(plan: &str) -> usize {
    plan.split('+').map(|k| kernel(k).1).sum()
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

/// Best of 7 batches (about 2 ms each): (cycles or ns per call, E share).
fn time(mut f: impl FnMut()) -> (f64, f64) {
    let started = Instant::now();
    let mut reps = 0u64;
    while started.elapsed().as_micros() < 2000 {
        f();
        reps += 1;
    }
    let mut best = (f64::MAX, 0.0);
    for _ in 0..7 {
        let (p0, e0) = cycles();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64;
        let (p1, e1) = cycles();
        let total = ((p1 - p0) + (e1 - e0)) as f64;
        let per = if total > 0.0 { total / reps as f64 } else { ns / reps as f64 };
        if per < best.0 {
            best = (per, if total > 0.0 { (e1 - e0) as f64 / total } else { 0.0 });
        }
    }
    best
}

fn main() {
    let data: Vec<u8> = (0..16 * 1024).map(|i| (i * 7 + (i >> 11)) as u8).collect();
    let chunks: Vec<&[u8; 1024]> = data.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
    let table: Vec<*const u8> = chunks.iter().map(|c| c.as_ptr()).collect();
    let mut expected = vec![0u8; 32 * 16];
    Platform::portable().hash_many::<1024>(&chunks, &IV, 0, IncrementCounter::Yes, 0, 1, 2, &mut expected);
    for plan in PLANS {
        let n = count(plan);
        let mut out = vec![0u8; 32 * 16];
        run(plan, &table, &mut out);
        assert_eq!(out[..32 * n], expected[..32 * n], "plan {plan} disagrees with portable");
    }
    println!("probe: chunk plans by core kind; every plan agrees with portable; cycles per byte (ns per byte off Apple)");
    let classes = [("user-interactive", 0x21u32), ("background", 0x09u32)];
    let mut results: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); PLANS.len()]; classes.len()];
    let mut mixed = 0;
    for _round in 0..3 {
        for (ci, &(_, class)) in classes.iter().enumerate() {
            set_qos(class);
            let t = Instant::now();
            while t.elapsed().as_millis() < 20 {
                black_box(time(|| {}));
            }
            let mut out = vec![0u8; 32 * 16];
            for (pi, plan) in PLANS.iter().enumerate() {
                let n = count(plan);
                let (c, e) = time(|| {
                    run(plan, black_box(&table), &mut out);
                    black_box(&out);
                });
                if e > 0.05 && e < 0.95 {
                    mixed += 1;
                }
                results[ci][pi].push(c / (n * 1024) as f64);
            }
        }
    }
    println!("mixed-kind batches: {mixed}");
    println!("{:>12} {:>3} {:>7} {:>7} {:>5}", "plan", "n", "P", "E", "E/P");
    for (pi, plan) in PLANS.iter().enumerate() {
        let med = |v: &Vec<f64>| {
            let mut v = v.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (p, e) = (med(&results[0][pi]), med(&results[1][pi]));
        println!("{:>12} {:>3} {:>7.3} {:>7.3} {:>5.2}", plan, count(plan), p, e, e / p);
    }
}
