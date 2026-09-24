//! Probe (branch probe/transition, never merged): the cost of the first NEON
//! work after SME2, by how long SME2 ran and how long ago it stopped.
use blake3_servil::platform::Platform;
use blake3_servil::IncrementCounter;
use std::hint::black_box;
use std::time::Instant;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

fn spin_ns(ns: u64) {
    let t = Instant::now();
    let mut x = 1u64;
    while (t.elapsed().as_nanos() as u64) < ns {
        for _ in 0..16 {
            x = black_box(x.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(7));
        }
    }
    black_box(x);
}

fn main() {
    let data: Vec<u8> = (0..(1 << 20)).map(|i| (i * 7 + (i >> 11)) as u8).collect();
    let chunks: Vec<&[u8; 1024]> = data.chunks_exact(1024).map(|c| c.try_into().unwrap()).collect();
    let blocks: Vec<&[u8; 64]> = data.chunks_exact(64).map(|c| c.try_into().unwrap()).collect();
    let mut out = vec![0u8; 32 * 1024];
    let mut dst = vec![0u8; 4096];
    let sme = Platform::detect();
    let neon = Platform::neon().unwrap();
    println!("probe: first NEON work after SME2 (platform {}); median of 41, ns", sme.hash_many_name());
    // The NEON work: two chunks on the k2 pair kernel (about 0.8 us).
    let mut neon_work = || neon.hash_many::<1024>(&chunks[..2], &IV, 0, IncrementCounter::Yes, 0, 1, 2, &mut out[..64]);
    let t = |sme_work: &mut dyn FnMut(), gap: u64, after: &mut dyn FnMut()| -> f64 {
        let mut v = Vec::new();
        for _ in 0..41 {
            sme_work();
            spin_ns(gap);
            let s = Instant::now();
            after();
            v.push(s.elapsed().as_nanos() as f64);
            spin_ns(20_000);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[20]
    };
    let alone = t(&mut || {}, 0, &mut neon_work);
    println!("  NEON work alone: {alone:.0}");
    let works: Vec<(&str, Box<dyn FnMut()>)> = vec![
        ("1 group of 16 parents", Box::new(|| sme.hash_many::<64>(&blocks[..16], &IV, 0, IncrementCounter::No, 11, 0, 0, &mut vec![0u8; 512]))),
        ("8 groups of parents", Box::new(|| sme.hash_many::<64>(&blocks[..128], &IV, 0, IncrementCounter::No, 11, 0, 0, &mut vec![0u8; 4096]))),
        ("128 chunks (~19 us)", Box::new(|| sme.hash_many::<1024>(&chunks[..128], &IV, 0, IncrementCounter::Yes, 0, 1, 2, &mut vec![0u8; 4096]))),
        ("hash() 8 MiB", Box::new(|| { black_box(blake3_servil::hash(&data)); })),
    ];
    for (name, mut w) in works {
        let mut line = format!("  after {name:22} penalty by gap:");
        for gap in [0u64, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000] {
            let v = t(&mut *w, gap, &mut neon_work);
            line += &format!(" {}us:{:.0}", gap / 1000, v - alone);
        }
        println!("{line}");
    }
    let mut copy = || { dst.copy_from_slice(&data[..4096]); black_box(&dst); };
    let copy_alone = t(&mut || {}, 0, &mut copy);
    let mut w = || sme.hash_many::<64>(&blocks[..16], &IV, 0, IncrementCounter::No, 11, 0, 0, &mut vec![0u8; 512]);
    println!("  4 KiB memcpy alone {copy_alone:.0}; after 1 SME2 group, gap 0: {:.0}", t(&mut w, 0, &mut copy));
}
