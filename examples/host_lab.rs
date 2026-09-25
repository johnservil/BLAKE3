//! Probe (branch probe/sme2-hybrid, never merged): SME2 chunk kernels with
//! an integer lane beside the sixteen streaming-vector lanes (16 + k chunks
//! per group, tools/gen_sme2_hybrid.py) against the plain SME2 kernel (16
//! per group). Every kernel's output is checked against the portable
//! implementation first; then time per chunk, wall and cycles, P and E.
#[path = "support/clocks.rs"]
mod clocks;
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
type Kernel = unsafe extern "C" fn(*const *const u8, *const u32, u64, u32, *mut u8, u64) -> u64;

unsafe extern "C" {
    fn blake3_sme2_hash16_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
    fn blake3_sme2x1_hash_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
    fn blake3_sme2x2_hash_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
    fn blake3_sme2x4_hash_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
}

const KERNELS: &[(&str, Kernel, usize)] = &[
    ("sme2 16", blake3_sme2_hash16_chunks_512, 16),
    ("sme2x1 17", blake3_sme2x1_hash_chunks_512, 17),
    ("sme2x2 18", blake3_sme2x2_hash_chunks_512, 18),
    ("sme2x4 20", blake3_sme2x4_hash_chunks_512, 20),
];

/// Up to 160 chunks' worth of whole groups per call, as DEGREE's eight
/// groups of sixteen do (the pointer table holds 160).
fn groups_for(lanes: usize) -> usize {
    160 / lanes
}

fn main() {
    assert!(matches!(Platform::detect(), Platform::SME2), "this probe needs SME2");
    let max_chunks = 8 * 20;
    let data: Vec<u8> = (0..max_chunks * 1024).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    let chunks: Vec<&[u8; 1024]> = data.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
    let table: Vec<*const u8> = chunks.iter().map(|c| c.as_ptr()).collect();
    let flags = 0u32 | 1 << 8 | 2 << 16; // CHUNK_START on block 0, CHUNK_END on block 15
    let counter = 5u64 << 32 | 0xffff_fff0; // low word wraps inside a group
    for &(name, kernel, lanes) in KERNELS {
        for groups in [1usize, 2, 8 * 20 / lanes] {
            let n = groups * lanes;
            let mut expected = vec![0u8; 32 * n];
            Platform::portable().hash_many::<1024>(&chunks[..n], &IV, counter, IncrementCounter::Yes, 0, 1, 2, &mut expected);
            let mut out = vec![0u8; 32 * n];
            let lanes_back = unsafe { kernel(table.as_ptr(), IV.as_ptr(), counter, flags, out.as_mut_ptr(), groups as u64) };
            assert_eq!(lanes_back, 16, "{name}: vector length");
            for c in 0..n {
                assert_eq!(out[32 * c..32 * c + 32], expected[32 * c..32 * c + 32], "{name}: {groups} groups, chunk {c}");
            }
        }
    }
    println!("probe: SME2 chunk kernels with an integer lane; every kernel agrees with portable; per chunk, best of 9 batches, 5 rounds");
    for &(label, class) in &[("P (user-interactive)", clocks::USER_INTERACTIVE), ("E (background)", clocks::BACKGROUND)] {
        clocks::set_qos(class);
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 20 {
            std::hint::black_box(clocks::measure(1, 100, || {}));
        }
        println!("{label}");
        let mut best: Vec<Option<clocks::Sample>> = vec![None; KERNELS.len()];
        let mut out = vec![0u8; 32 * max_chunks];
        for round in 0..5 {
            for i in 0..KERNELS.len() {
                let v = (i + round) % KERNELS.len();
                let (_, kernel, lanes) = KERNELS[v];
                let groups = groups_for(lanes);
                let s = clocks::measure(9, 2000, || unsafe {
                    kernel(std::hint::black_box(table.as_ptr()), IV.as_ptr(), 0, flags, out.as_mut_ptr(), groups as u64);
                })
                .fastest();
                let per = (groups * lanes) as f64;
                let s = clocks::Sample { ns: s.ns / per, p_cycles: s.p_cycles / per, e_cycles: s.e_cycles / per };
                if best[v].map_or(true, |b| s.ns < b.ns) {
                    best[v] = Some(s);
                }
            }
        }
        let base = best[0].unwrap();
        for (i, &(name, _, lanes)) in KERNELS.iter().enumerate() {
            let s = best[i].unwrap();
            println!("  {:<10} {:>3} chunks/group  {:>8.1} ns/chunk ({:.3} ns/B)  {:>7.0} cyc/chunk  {:+5.1}% wall {:+5.1}% cyc  [{}]",
                name, lanes, s.ns, s.ns / 1024.0, s.cycles(), (s.ns / base.ns - 1.0) * 100.0,
                if base.has_cycles() { (s.cycles() / base.cycles() - 1.0) * 100.0 } else { 0.0 }, s.show());
        }
    }
}
