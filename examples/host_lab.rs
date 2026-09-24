//! Probe (branch probe/mixed-parents, never merged): parent-kernel plans
//! for one-block messages, old (p2, p4, p8, k1 in turn) against new (a
//! scalar lane beside NEON: p3, p5, p7, p9), by core kind, in cycles per
//! call (thread_selfcounts), each plan checked against the portable
//! implementation first.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;
use std::hint::black_box;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
type Kernel = unsafe extern "C" fn(*const *const u8, u64, *const u32, u64, u64, *mut u8);

unsafe extern "C" {
    fn blake3_hybrid_k1(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p2(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p3(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p4(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p5(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p7(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p8(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
    fn blake3_hybrid_p9(i: *const *const u8, b: u64, k: *const u32, c: u64, f: u64, o: *mut u8);
}

fn kernel(name: &str) -> (Kernel, usize) {
    match name {
        "k1" => (blake3_hybrid_k1, 1),
        "p2" => (blake3_hybrid_p2, 2),
        "p3" => (blake3_hybrid_p3, 3),
        "p4" => (blake3_hybrid_p4, 4),
        "p5" => (blake3_hybrid_p5, 5),
        "p7" => (blake3_hybrid_p7, 7),
        "p8" => (blake3_hybrid_p8, 8),
        "p9" => (blake3_hybrid_p9, 9),
        _ => panic!("no kernel {name}"),
    }
}

/// (old plan, candidate) pairs: ten messages, three candidates, each
/// against the old plan (repeated so each pairing is timed side by side).
const PLANS: &[(&str, &str)] = &[
    ("p8+p2", "p9+k1"), ("p8+p2", "p5+p5"), ("p8+p2", "p7+p3"), ("p8+p2", "p8+p2"),
];

/// One-block messages: CHUNK_START | CHUNK_END | ROOT, block length 64, counter 0.
const FLAGS: u8 = 1 | 2 | 8;

fn run(plan: &str, table: &[*const u8], out: &mut [u8]) {
    let packed = FLAGS as u64 | 64u64 << 24;
    let mut done = 0;
    for name in plan.split('+') {
        let (k, n) = kernel(name);
        unsafe { k(table[done..].as_ptr(), 1, IV.as_ptr(), 0, packed, out[32 * done..].as_mut_ptr()) };
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
    let data: Vec<u8> = (0..16 * 64).map(|i| (i * 7 + (i >> 6)) as u8).collect();
    let blocks: Vec<&[u8; 64]> = data.chunks_exact(64).map(|c| c.try_into().unwrap()).collect();
    let table: Vec<*const u8> = blocks.iter().map(|c| c.as_ptr()).collect();
    let mut expected = vec![0u8; 32 * 16];
    Platform::portable().hash_many::<64>(&blocks, &IV, 0, IncrementCounter::No, FLAGS, 0, 0, &mut expected);
    for &(old, new) in PLANS {
        for plan in [old, new] {
            let n = count(plan);
            let mut out = vec![0u8; 32 * 16];
            run(plan, &table, &mut out);
            assert_eq!(out[..32 * n], expected[..32 * n], "plan {plan} disagrees with portable");
        }
    }
    println!("probe: one-block parent plans by core kind; every plan agrees with portable; cycles per call (ns off Apple)");
    let classes = [("user-interactive", 0x21u32), ("background", 0x09u32)];
    let mut results: Vec<Vec<Vec<(f64, f64)>>> = vec![vec![Vec::new(); PLANS.len()]; classes.len()];
    let mut mixed = 0;
    for _round in 0..5 {
        for (ci, &(_, class)) in classes.iter().enumerate() {
            set_qos(class);
            let t = Instant::now();
            while t.elapsed().as_millis() < 20 {
                black_box(time(|| {}));
            }
            let mut out = vec![0u8; 32 * 16];
            for (pi, &(old, new)) in PLANS.iter().enumerate() {
                let mut pair = [0.0; 2];
                for (slot, plan) in [old, new].into_iter().enumerate() {
                    let (c, e) = time(|| {
                        run(plan, black_box(&table), &mut out);
                        black_box(&out);
                    });
                    if e > 0.05 && e < 0.95 {
                        mixed += 1;
                    }
                    pair[slot] = c;
                }
                results[ci][pi].push((pair[0], pair[1]));
            }
        }
    }
    println!("mixed-kind batches: {mixed}");
    println!("{:>3} {:>12} {:>7} {:>7} {:>6} | {:>12} {:>7} {:>7} {:>6}", "n", "old", "P", "E", "", "new", "P", "E", "new/old P, E");
    for (pi, &(old, new)) in PLANS.iter().enumerate() {
        let med = |ci: usize, slot: usize| {
            let mut v: Vec<f64> = results[ci][pi].iter().map(|p| if slot == 0 { p.0 } else { p.1 }).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (op, oe, np, ne) = (med(0, 0), med(1, 0), med(0, 1), med(1, 1));
        println!("{:>3} {:>12} {:>7.0} {:>7.0} {:>6} | {:>12} {:>7.0} {:>7.0}  {:+.0}% {:+.0}%", count(old), old, op, oe, "", new, np, ne, (np / op - 1.0) * 100.0, (ne / oe - 1.0) * 100.0);
    }
}
