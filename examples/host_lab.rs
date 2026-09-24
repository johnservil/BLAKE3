//! Probe (branch probe/ecore-trigger, never merged): what sends a thread to
//! E-cores after SME2? Samples of 1 ms of small NEON hashing (4 KiB), each
//! classified P or E by thread_selfcounts, in several arrangements.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

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
fn cycles() -> (u64, u64) { (1, 0) }

/// One sample: 1 ms of 4 KiB hashes; true when most of its cycles ran on E.
fn sample(small: &[u8]) -> bool {
    let (p0, e0) = cycles();
    let t = Instant::now();
    while t.elapsed().as_micros() < 1000 {
        black_box(blake3_servil::hash(black_box(small)));
    }
    let (p1, e1) = cycles();
    let (p, e) = (p1 - p0, e1 - e0);
    e > p
}

/// The share of `n` samples on E, with `between` run before each sample.
fn samples(n: usize, small: &[u8], mut between: impl FnMut()) -> (usize, usize) {
    let mut e = 0;
    for _ in 0..n {
        between();
        e += sample(small) as usize;
    }
    (e, n)
}

/// A background thread running `work` until stopped; returns its E share.
fn beside<T: Send + 'static>(work: impl Fn() + Send + 'static, body: impl FnOnce() -> T) -> (T, f64) {
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    let h = std::thread::spawn(move || {
        let (p0, e0) = cycles();
        while !s.load(Ordering::Relaxed) {
            work();
        }
        let (p1, e1) = cycles();
        (e1 - e0) as f64 / ((p1 - p0) + (e1 - e0)).max(1) as f64
    });
    let out = body();
    stop.store(true, Ordering::Relaxed);
    (out, h.join().unwrap())
}

/// Two persistent threads, like the benchmark's shared copies: `run` posts
/// a job, blocks on a condvar until both copies finish.
struct Duo {
    state: std::sync::Mutex<(u64, usize, u8)>, // (generation, done count, job kind)
    posted: std::sync::Condvar,
    done: std::sync::Condvar,
}
impl Duo {
    fn new(big: Arc<Vec<u8>>) -> &'static Duo {
        let duo: &'static Duo = Box::leak(Box::new(Duo { state: std::sync::Mutex::new((0, 0, 0)), posted: std::sync::Condvar::new(), done: std::sync::Condvar::new() }));
        for copy in 0..2u8 {
            // Each copy hashes its own buffer, as in the benchmark.
            let big: Arc<Vec<u8>> = Arc::new(big.iter().map(|b| b ^ (copy + 1)).collect());
            std::thread::spawn(move || {
                let neon = Platform::neon().unwrap();
                let mut seen = 0;
                loop {
                    let kind = {
                        let mut g = duo.state.lock().unwrap();
                        while g.0 == seen { g = duo.posted.wait(g).unwrap(); }
                        seen = g.0;
                        g.2
                    };
                    work(kind, &big, &neon);
                    let mut g = duo.state.lock().unwrap();
                    g.1 += 1;
                    if g.1 == 2 { duo.done.notify_one(); }
                }
            });
        }
        duo
    }
    fn run(&self, kind: u8) {
        let mut g = self.state.lock().unwrap();
        g.0 += 1; g.1 = 0; g.2 = kind;
        self.posted.notify_all();
        while g.1 < 2 { g = self.done.wait(g).unwrap(); }
    }
}

/// 0: 4 KiB hash; 1: 8 MiB hash (SME2); 2: 8 MiB on NEON.
fn work(kind: u8, big: &[u8], neon: &Platform) {
    match kind {
        0 => { black_box(blake3_servil::hash(&big[..4096])); }
        1 => { black_box(blake3_servil::hash(big)); }
        _ => {
            let chunks: Vec<&[u8; 1024]> = big.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
            let mut out = vec![0u8; 32 * 16];
            for g in chunks.chunks(16) {
                neon.hash_many::<1024>(g, &IV, 0, IncrementCounter::Yes, 0, 1, 2, &mut out);
            }
            black_box(&out);
        }
    }
}

fn main() {
    let big: Arc<Vec<u8>> = Arc::new((0..8 << 20).map(|i| (i * 7 + (i >> 11)) as u8).collect());
    let small: Vec<u8> = big[..4096].to_vec();
    let neon = Platform::neon().expect("NEON");
    println!("probe: E-core trigger; samples of 1 ms of 4 KiB hashes, count on E of n");
    for round in 0..2 {
        println!("\nround {round}");
        let (e, n) = samples(2000, &small, || {});
        println!("  1 alone                                   {e:4} / {n}");
        let b = big.clone();
        let ((e, n), a) = beside(move || { black_box(blake3_servil::hash(&b[..])); }, || samples(2000, &small, || {}));
        println!("  2 beside a thread hashing 8 MiB (SME2)    {e:4} / {n}   (that thread's E share {a:.3})");
        let b = big.clone();
        let ((e, n), a) = beside(
            move || {
                let chunks: Vec<&[u8; 1024]> = b.chunks_exact(1024).take(8192).map(|c| c.try_into().unwrap()).collect();
                let mut out = vec![0u8; 32 * 16];
                for g in chunks.chunks(16) {
                    neon.hash_many::<1024>(g, &IV, 0, IncrementCounter::Yes, 0, 1, 2, &mut out);
                }
                black_box(&out);
            },
            || samples(2000, &small, || {}),
        );
        println!("  4 beside a thread hashing 8 MiB on NEON   {e:4} / {n}   (that thread's E share {a:.3})");
        for len in [64usize << 10, 256 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20] {
            let (e, n) = samples(1000, &small, || { black_box(blake3_servil::hash(&big[..len])); });
            println!("  3/5 one thread, {:>5} KiB SME2 before each  {e:4} / {n}", len >> 10);
        }
        let (e, n) = samples(1000, &small, || {
            let chunks: Vec<&[u8; 1024]> = big.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
            let mut out = vec![0u8; 32 * 16];
            for g in chunks.chunks(16) {
                neon.hash_many::<1024>(g, &IV, 0, IncrementCounter::Yes, 0, 1, 2, &mut out);
            }
            black_box(&out);
        });
        println!("  3n one thread, 8 MiB NEON before each       {e:4} / {n}");
        let duo = Duo::new(big.clone());
        for (label, own, copies) in [("7 own 8 MiB SME2, then copies 4 KiB, block", 1u8, 0u8), ("8 own 8 MiB NEON, then copies 4 KiB, block", 2, 0), ("9 own 8 MiB SME2, then copies 8 MiB SME2, block", 1, 1), ("9n own 8 MiB NEON, then copies 8 MiB NEON, block", 2, 2), ("10 copies 8 MiB SME2 only, block", 3, 1)] {
            let (e, n) = samples(1000, &small, || {
                if own != 3 { work(own, &big, &neon); }
                duo.run(copies);
            });
            println!("  {label:48} {e:4} / {n}");
        }
        // Afterwards: does the effect linger once SME2 stops?
        let (e, n) = samples(2000, &small, || {});
        println!("  6 alone again, right after                {e:4} / {n}");
    }
}
