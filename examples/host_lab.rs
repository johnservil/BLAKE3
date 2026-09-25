//! Probe (branch probe/sme2-thread, never merged): one thread streaming
//! SME2 over a contiguous stretch of the input, in back-to-back 1 MiB
//! subtrees, while the pool hashes the rest on NEON, against the pool
//! alone at the same thread budget. The split is swept; the best is
//! reported. Then: two threads running the raw SME2 chunk kernel at once,
//! against one, to see whether a process reaches a second SME unit.
//! Wall time and the calling thread's cycles per ns.
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Arc;

unsafe extern "C" {
    fn blake3_sme2_hash16_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
}
const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

/// A persistent helper thread that spins for jobs: (kind, ptr, len, budget).
struct Helper {
    generation: AtomicU64,
    done: AtomicU64,
    ptr: AtomicUsize,
    len: AtomicUsize,
    budget: AtomicUsize, // 0: raw SME2 chunk kernel; n: hash_multithreaded_with_budget(n)
}

fn raw_sme2(region: &[u8]) {
    #[repr(C, align(128))]
    struct Out([u8; 32 * 128]);
    let mut out = Out([0; 32 * 128]);
    let mut table = [std::ptr::null::<u8>(); 128];
    for (g, block) in region.chunks(128 * 1024).enumerate() {
        let n = block.len() / 1024;
        for i in 0..n { table[i] = block[i * 1024..].as_ptr(); }
        unsafe { blake3_sme2_hash16_chunks_512(table.as_ptr(), IV.as_ptr(), (g * 128) as u64, 1 << 8 | 2 << 16, out.0.as_mut_ptr(), (n / 16) as u64) };
    }
    black_box(&out.0);
}

fn helper_main(h: Arc<Helper>, qos: u32) {
    clocks::set_qos(qos);
    let mut seen = 0;
    loop {
        let g = h.generation.load(Acquire);
        if g == seen { std::hint::spin_loop(); continue; }
        if g == u64::MAX { return; }
        seen = g;
        let region = unsafe { std::slice::from_raw_parts(h.ptr.load(Relaxed) as *const u8, h.len.load(Relaxed)) };
        match h.budget.load(Relaxed) {
            0 => raw_sme2(region),
            b => { black_box(blake3_servil::hash_multithreaded_with_budget(region, b)); }
        }
        h.done.store(g, Release);
    }
}

fn post(h: &Helper, region: &[u8], budget: usize) -> u64 {
    h.ptr.store(region.as_ptr() as usize, Relaxed);
    h.len.store(region.len(), Relaxed);
    h.budget.store(budget, Relaxed);
    h.generation.fetch_add(1, Release) + 1
}

fn wait(h: &Helper, g: u64) {
    while h.done.load(Acquire) != g { std::hint::spin_loop(); }
}

/// The SME2 thread (this one) hashes input[..cut] in 1 MiB pieces with
/// hash() (the flat walk, under the turn); the helper runs the pool with
/// budget - 1 threads on input[cut..].
fn split_call(h: &Helper, input: &[u8], cut: usize, budget: usize) {
    let g = if cut < input.len() { Some(post(h, &input[cut..], budget - 1)) } else { None };
    for piece in input[..cut].chunks(1 << 20) { black_box(blake3_servil::hash(black_box(piece))); }
    if let Some(g) = g { wait(h, g); }
}

fn main() {
    clocks::set_qos(clocks::USER_INTERACTIVE);
    blake3_servil::initialize();
    let cpus = std::thread::available_parallelism().unwrap().get();
    println!("probe/sme2-thread: platform {}, {cpus} CPUs", blake3_servil::kernel_report().platform);
    let h = Arc::new(Helper { generation: AtomicU64::new(0), done: AtomicU64::new(0), ptr: AtomicUsize::new(0), len: AtomicUsize::new(0), budget: AtomicUsize::new(1) });
    { let h = h.clone(); std::thread::spawn(move || helper_main(h, clocks::USER_INTERACTIVE)); }
    let input: Vec<u8> = (0..64usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    let mib = 1usize << 20;
    for round in 0..2 {
        println!("round {round}: ns/B (fastest of 7 batches) [caller's cycles/ns]");
        for &size in &[1 * mib, 8 * mib, 64 * mib] {
            let data = &input[..size];
            let st = clocks::measure(7, 20_000, || { black_box(blake3_servil::hash(black_box(data))); }).fastest();
            println!("  {:>3} MiB  hash (1 thread) {:.4} [{:.2}]", size / mib, st.ns / size as f64, st.per_ns());
            for budget in [2usize, 4, cpus] {
                let pool = clocks::measure(7, 20_000, || { black_box(blake3_servil::hash_multithreaded_with_budget(black_box(data), budget)); }).fastest();
                let mut best = (f64::MAX, 0.0, 0usize);
                let steps: Vec<usize> = (1..=16).map(|k| size / 16 * k).filter(|&c| c % mib == 0 || size <= mib).collect();
                for cut in steps.iter().copied().chain(if size <= mib { vec![size / 4, size / 2] } else { vec![] }) {
                    let s = clocks::measure(5, 20_000, || split_call(&h, data, cut, budget)).fastest();
                    if s.ns < best.0 { best = (s.ns, s.per_ns(), cut); }
                }
                println!("  {:>3} MiB  budget {budget:>2}: pool {:.4} [{:.2}]  SME2 thread + pool {:.4} [{:.2}] at {:.0}% to SME2  ({:+.1}%)",
                    size / mib, pool.ns / size as f64, pool.per_ns(), best.0 / size as f64, best.1, 100.0 * best.2 as f64 / size as f64, 100.0 * (best.0 / pool.ns - 1.0));
            }
        }
        // Two SME2 threads at once (raw kernel, no turn) against one.
        let data = &input[..16 * mib];
        let one = clocks::measure(7, 20_000, || raw_sme2(black_box(data))).fastest();
        let two = clocks::measure(7, 20_000, || {
            let g = post(&h, &data[8 * mib..], 0);
            raw_sme2(black_box(&data[..8 * mib]));
            wait(&h, g);
        }).fastest();
        println!("  raw SME2 chunk kernel, 16 MiB: one thread {:.4} ns/B [{:.2}]; two threads, 8 MiB each, {:.4} ns/B [{:.2}] ({:.2}x one's throughput)",
            one.ns / data.len() as f64, one.per_ns(), two.ns / data.len() as f64, two.per_ns(), one.ns / two.ns);
    }
    h.generation.store(u64::MAX, Release);
}
