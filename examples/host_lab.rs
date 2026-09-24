//! Probe (branch probe/sme2-gap, never merged): the time of an SME2 call
//! (hash_many of 128 one-block messages, one kernel call) and of the same
//! batch on NEON, after a gap of scalar work of a given length.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;
use std::hint::black_box;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

fn spin_ns(ns: u64) {
    if ns == 0 {
        return;
    }
    let t = Instant::now();
    let mut x = 1u64;
    while (t.elapsed().as_nanos() as u64) < ns {
        for _ in 0..8 {
            x = black_box(x.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(7));
        }
    }
    black_box(x);
}

fn main() {
    main_mt512();
    let data: Vec<u8> = (0..65536).map(|i| (i * 7 + (i >> 11)) as u8).collect();
    let blocks: Vec<&[u8; 64]> = data.chunks_exact(64).take(128).map(|c| c.try_into().unwrap()).collect();
    let mut out = vec![0u8; 32 * 128];
    let sme = Platform::detect();
    let neon = Platform::neon().unwrap();
    println!("probe: 128 one-block messages in one call after a gap of scalar work; median of 201 calls, ns ({})", sme.hash_many_name());
    for (name, p) in [("SME2", sme), ("NEON", neon)] {
        let mut line = format!("  {name}:");
        for gap in [0u64, 250, 500, 1000, 2000, 4000, 8000, 16000, 64000] {
            let mut v: Vec<u64> = (0..201)
                .map(|_| {
                    spin_ns(gap);
                    let t = Instant::now();
                    p.hash_many::<64>(black_box(&blocks), &IV, 0, IncrementCounter::No, 11, 0, 0, &mut out);
                    black_box(&out);
                    t.elapsed().as_nanos() as u64
                })
                .collect();
            v.sort();
            line += &format!(" {}us:{}", gap as f64 / 1000.0, v[100]);
        }
        println!("{line}");
    }
}

// The benchmark's servil_batch, st against mt and against st after a pass.
fn servil_batch(input: &[u8], messages: usize, iterations: usize, hash_many: impl Fn(&[&[u8]], &mut [blake3_servil::Hash])) {
    let batch: Vec<&[u8]> = input.chunks_exact(64).collect();
    assert_eq!(batch.len(), messages);
    let mut digests = vec![blake3_servil::Hash::from_bytes([0; 32]); messages];
    for _ in 0..iterations {
        hash_many(black_box(&batch), &mut digests);
        for digest in &digests { black_box(digest.as_bytes()); }
    }
}
fn main_mt512() {
    let data: Vec<u8> = (0..(1 << 20)).map(|i| (i * 13) as u8).collect();
    for n in [256usize, 512, 1024] {
        let input = &data[..n * 64];
        let iters = 200_000 / n;
        let mut r = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..30 {
            fn sum_st(b: &[&[u8]], d: &mut [blake3_servil::Hash]) { let t: usize = b.iter().map(|m| m.len()).sum(); black_box(t); blake3_servil::hash_many(b, d) }
            fn scalar_st(b: &[&[u8]], d: &mut [blake3_servil::Hash]) { let mut t = 0usize; for m in b { t += m.len(); if t >= 65536 { break; } } black_box(t); blake3_servil::hash_many(b, d) }
            fn touch_st(b: &[&[u8]], d: &mut [blake3_servil::Hash]) { let mut t = 0u8; for h in d.iter() { t ^= h.as_bytes()[0]; } black_box(t); blake3_servil::hash_many(b, d) }
            for (k, f) in [blake3_servil::hash_many as fn(&[&[u8]], &mut [blake3_servil::Hash]), blake3_servil::hash_many_multithreaded, sum_st, scalar_st, touch_st].iter().enumerate() {
                let t = Instant::now();
                servil_batch(input, n, iters, f);
                r[k].push(t.elapsed().as_nanos() as f64 / (iters * n) as f64);
            }
        }
        for v in &mut r { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); }
        println!("{n:5} msgs: st {:.2}  mt {:.2}  sum+st {:.2}  scalar-sum+st {:.2}  digest-pass+st {:.2} ns/msg", r[0][15], r[1][15], r[2][15], r[3][15], r[4][15]);
    }
}
