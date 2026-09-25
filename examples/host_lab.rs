//! Probe (branch probe/energy, never merged): energy per byte, from the
//! process's own energy counters (`proc_pid_rusage`, RUSAGE_INFO_V6:
//! `ri_energy_nj`, `ri_penergy_nj`, `ri_billed_energy`), beside wall time
//! and core cycles, at user-interactive QoS (P-cores) and background QoS
//! (E-cores). Run it once on the default build and once with `no_sme2`:
//! the same calls, SME2 against NEON alone.
//!
//! First the counters' reliability: sleeping (should read near zero), a
//! scalar spin loop (a known steady load: power in watts, round to
//! round), then each workload in rounds, interleaved, reporting the median
//! and the spread of pJ/B. macOS only; elsewhere it says so and stops.
#[path = "support/clocks.rs"]
mod clocks;

use std::hint::black_box;
use std::time::{Duration, Instant};

/// The process's energy so far, in nJ: all cores, P-cores alone, and the
/// scheduler's billed energy (units undocumented; reported raw), with the
/// process's cycles, all and P.
#[derive(Clone, Copy, Debug, Default)]
struct Energy {
    nj: u64,
    p_nj: u64,
    billed: u64,
    cycles: u64,
    #[allow(dead_code)]
    p_cycles: u64,
}

#[cfg(target_vendor = "apple")]
fn energy() -> Energy {
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
        fn getpid() -> i32;
    }
    // rusage_info_v6 as u64 words: ri_uuid (2), then ri_user_time at 2 ...
    // ri_cycles 32, ri_billed_energy 33, ri_pcycles 41, ri_energy_nj 42,
    // ri_penergy_nj 43 (<sys/resource.h>).
    let mut words = [0u64; 128];
    assert_eq!(unsafe { proc_pid_rusage(getpid(), 6, words.as_mut_ptr()) }, 0, "proc_pid_rusage V6");
    Energy { nj: words[42], p_nj: words[43], billed: words[33], cycles: words[32], p_cycles: words[41] }
}

#[cfg(not(target_vendor = "apple"))]
fn energy() -> Energy {
    Energy::default()
}

/// One stretch of `f`, repeated for about `ms` milliseconds: its wall
/// time, energy, and thread cycles, and how many calls it made.
struct Stretch {
    calls: u64,
    ns: f64,
    nj: f64,
    p_nj: f64,
    billed: f64,
    proc_cycles: f64,
    thread_cycles: (f64, f64),
}

fn stretch(ms: u64, mut f: impl FnMut()) -> Stretch {
    // The pool's workers spin up to 200 us after a multithreaded call and
    // then sleep; 5 ms of sleep keeps their energy out of the next stretch.
    std::thread::sleep(Duration::from_millis(5));
    let (tp0, te0) = clocks::cycles();
    let e0 = energy();
    let t = Instant::now();
    let limit = Duration::from_millis(ms);
    let mut calls = 0u64;
    while t.elapsed() < limit {
        f();
        calls += 1;
    }
    let ns = t.elapsed().as_nanos() as f64;
    let e1 = energy();
    let (tp1, te1) = clocks::cycles();
    Stretch {
        calls,
        ns,
        nj: (e1.nj - e0.nj) as f64,
        p_nj: (e1.p_nj - e0.p_nj) as f64,
        billed: (e1.billed - e0.billed) as f64,
        proc_cycles: (e1.cycles - e0.cycles) as f64,
        thread_cycles: ((tp1 - tp0) as f64, (te1 - te0) as f64),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn spread(v: &[f64]) -> (f64, f64) {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(0.0, f64::max);
    (lo, hi)
}

type Work<'a> = (&'a str, usize, Box<dyn FnMut() + 'a>);

/// How a thread hashes a 64 KiB piece: `hash` (the default path: SME2
/// where built and present) or NEON alone (the platform's hash_many on
/// the piece's 64 chunks; the chaining values are dropped, the energy is
/// the chunks').
#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    Hash,
    Neon,
}

const PIECE: usize = 64 * 1024;

fn hash_piece(kernel: Kernel, piece: &[u8]) {
    match kernel {
        Kernel::Hash => {
            black_box(blake3_servil::hash(black_box(piece)));
        }
        Kernel::Neon => {
            let platform = blake3_servil::platform::Platform::neon().expect("AArch64 has NEON");
            let chunks: Vec<&[u8; 1024]> = piece.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
            let mut out = [0u8; 32 * 64];
            const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
            platform.hash_many::<1024>(black_box(&chunks), &IV, 0, blake3_servil::IncrementCounter::Yes, 0, 1, 2, &mut out[..32 * chunks.len()]);
            black_box(&out);
        }
    }
}

/// One call's worth of a candidate efficient design: `helpers` scoped
/// threads at QoS `helper_qos` pull 64 KiB pieces of `input` from a shared
/// cursor with `helper_kernel`; the calling thread pulls too with `hash`
/// when `caller_works`, and otherwise waits in join (asleep). Thread
/// creation is inside the call, as a design with sleeping workers pays a
/// wake per worker instead.
/// `helpers` is a list of (QoS, kernel) per helper thread; helpers take
/// `helper_piece` bytes per pull, the caller PIECE.
fn pulled(input: &[u8], caller_works: bool, helpers: &[(u32, Kernel)], helper_piece: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let cursor = AtomicUsize::new(0);
    let pull = |kernel: Kernel, piece: usize| loop {
        let start = cursor.fetch_add(piece, Ordering::Relaxed);
        if start >= input.len() {
            break;
        }
        hash_piece(kernel, &input[start..(start + piece).min(input.len())]);
    };
    std::thread::scope(|scope| {
        for &(qos, kernel) in helpers {
            scope.spawn(move || {
                clocks::set_qos(qos);
                pull(kernel, helper_piece)
            });
        }
        if caller_works {
            pull(Kernel::Hash, PIECE);
        }
    });
}

fn main() {
    if !cfg!(target_vendor = "apple") {
        println!("probe/energy: macOS only (proc_pid_rusage energy counters)");
        return;
    }
    println!("probe/energy: platform {}", blake3_servil::kernel_report().platform);
    blake3_servil::initialize();

    for &(label, class) in &[("P (user-interactive)", clocks::USER_INTERACTIVE), ("E (background)", clocks::BACKGROUND)] {
        clocks::set_qos(class);
        println!("\n== {label}");

        // Reliability: sleep reads near zero; a spin loop reads a steady power.
        for round in 0..3 {
            let e0 = energy();
            let t = Instant::now();
            std::thread::sleep(Duration::from_millis(200));
            let ns = t.elapsed().as_nanos() as f64;
            let e1 = energy();
            let mut x = 1u64;
            let s = stretch(200, || {
                for _ in 0..1000 {
                    x = black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407).rotate_left(7));
                }
            });
            println!(
                "  round {round}: sleep 200 ms {:.3} W ({} nJ, billed {}); spin {:.3} W, P share {:.2}, billed {:.0} per ms, {:.2} thread cyc/ns, E cyc share {:.2}",
                (e1.nj - e0.nj) as f64 / ns,
                e1.nj - e0.nj,
                e1.billed - e0.billed,
                s.nj / s.ns,
                if s.nj > 0.0 { s.p_nj / s.nj } else { 0.0 },
                s.billed / (s.ns / 1e6),
                (s.thread_cycles.0 + s.thread_cycles.1) / s.ns,
                s.thread_cycles.1 / (s.thread_cycles.0 + s.thread_cycles.1).max(1.0),
            );
        }

        let sizes = [1024usize, 8 * 1024, 16 * 1024, 256 * 1024, 1024 * 1024, 8 * 1024 * 1024];
        let inputs: Vec<Vec<u8>> = sizes.iter().map(|&n| (0..n).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect()).collect();
        let messages: Vec<[u8; 64]> = (0..1024).map(|i| [i as u8; 64]).collect();
        let refs: Vec<&[u8]> = messages.iter().map(|m| &m[..]).collect();
        let mut outs = vec![blake3_servil::Hash::from([0; 32]); 1024];
        let big = &inputs[5];
        let mut works: Vec<Work> = Vec::new();
        for input in &inputs {
            let name: &'static str = match input.len() {
                1024 => "hash 1 KiB",
                8192 => "hash 8 KiB",
                16384 => "hash 16 KiB",
                262144 => "hash 256 KiB",
                1048576 => "hash 1 MiB",
                _ => "hash 8 MiB",
            };
            works.push((name, input.len(), Box::new(move || {
                black_box(blake3_servil::hash(black_box(input)));
            })));
        }
        works.push(("hash_many 1024 x 64 B", 1024 * 64, Box::new(|| {
            blake3_servil::hash_many(black_box(&refs), &mut outs);
            black_box(&outs);
        })));
        works.push(("Hasher 8 MiB in 64 KiB pieces", big.len(), Box::new(move || {
            let mut h = blake3_servil::Hasher::new();
            for piece in big.chunks(64 * 1024) {
                h.update(piece);
            }
            black_box(h.finalize());
        })));
        for budget in [2usize, 4, 0] {
            let name: &'static str = match budget {
                2 => "hash_multithreaded 8 MiB, 2 threads",
                4 => "hash_multithreaded 8 MiB, 4 threads",
                _ => "hash_multithreaded 8 MiB, all threads",
            };
            works.push((name, big.len(), Box::new(move || {
                let h = if budget == 0 {
                    blake3_servil::hash_multithreaded(black_box(big))
                } else {
                    blake3_servil::hash_multithreaded_with_budget(black_box(big), budget)
                };
                black_box(h);
            })));
        }

        // Candidate efficient designs, pieces pulled from a cursor.
        const E: u32 = clocks::BACKGROUND;
        const P: u32 = clocks::USER_INTERACTIVE;
        use Kernel::{Hash, Neon};
        let designs: Vec<(&'static str, usize, bool, Vec<(u32, Kernel)>, usize)> = vec![
            ("4 E threads NEON, caller waits", 5, false, vec![(E, Neon); 4], PIECE),
            ("1 E thread hash, caller waits", 5, false, vec![(E, Hash)], PIECE),
            ("caller hash + 4 E NEON", 5, true, vec![(E, Neon); 4], PIECE),
            ("caller hash + 4 E NEON, 16 KiB", 5, true, vec![(E, Neon); 4], 16 * 1024),
            ("caller hash + 2 E NEON, 16 KiB", 5, true, vec![(E, Neon); 2], 16 * 1024),
            ("caller hash + 1 E hash", 5, true, vec![(E, Hash)], PIECE),
            ("caller hash + 1 E hash + 3 E NEON, 16 KiB", 5, true, vec![(E, Hash), (E, Neon), (E, Neon), (E, Neon)], 16 * 1024),
            ("caller hash + 4 P NEON", 5, true, vec![(P, Neon); 4], PIECE),
            ("1 MiB: caller hash + 4 E NEON, 16 KiB", 4, true, vec![(E, Neon); 4], 16 * 1024),
            ("256 KiB: caller hash + 4 E NEON, 16 KiB", 3, true, vec![(E, Neon); 4], 16 * 1024),
            ("256 KiB: caller hash + 2 E NEON, 16 KiB", 3, true, vec![(E, Neon); 2], 16 * 1024),
        ];
        for (name, input, caller_works, helpers, piece) in designs {
            let input = &inputs[input];
            works.push((name, input.len(), Box::new(move || pulled(input, caller_works, &helpers, piece))));
        }

        // Rounds interleave the workloads, so drift reaches them alike.
        const ROUNDS: usize = 5;
        let mut rows: Vec<Vec<Stretch>> = (0..works.len()).map(|_| Vec::new()).collect();
        for _ in 0..ROUNDS {
            for (i, (_, _, f)) in works.iter_mut().enumerate() {
                rows[i].push(stretch(150, f));
            }
        }
        println!(
            "  {:<44} {:>9} {:>17} {:>8} {:>7} {:>9} {:>9} {:>6}",
            "workload (8 MiB unless named)", "ns/B", "pJ/B (min-max)", "W", "P nJ %", "billed/B", "proc cyc/B", "E cyc%"
        );
        for (i, (name, bytes, _)) in works.iter().enumerate() {
            let per_byte = |s: &Stretch, v: f64| v / (s.calls as f64 * *bytes as f64);
            let pj: Vec<f64> = rows[i].iter().map(|s| 1000.0 * per_byte(s, s.nj)).collect();
            let (lo, hi) = spread(&pj);
            let ns = median(rows[i].iter().map(|s| per_byte(s, s.ns)).collect());
            let watts = median(rows[i].iter().map(|s| s.nj / s.ns).collect());
            let p_share = median(rows[i].iter().map(|s| if s.nj > 0.0 { 100.0 * s.p_nj / s.nj } else { 0.0 }).collect());
            let billed = median(rows[i].iter().map(|s| per_byte(s, s.billed)).collect());
            let cyc = median(rows[i].iter().map(|s| per_byte(s, s.proc_cycles)).collect());
            let e_share = median(
                rows[i].iter().map(|s| 100.0 * s.thread_cycles.1 / (s.thread_cycles.0 + s.thread_cycles.1).max(1.0)).collect(),
            );
            println!(
                "  {:<44} {:>9.4} {:>7.1} ({:>4.0}-{:<4.0}) {:>8.3} {:>7.1} {:>9.4} {:>9.3} {:>6.1}",
                name, ns, median(pj.clone()), lo, hi, watts, p_share, billed, cyc, e_share
            );
        }
    }
}
