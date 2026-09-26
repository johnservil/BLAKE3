//! probe/whir-merkle (never merged): WHIR's Merkle tree (worldfnd/whir
//! c03a4a5, src/protocols/merkle_tree.rs) built five ways, leaves of 256 B
//! and nodes of two 32-byte children, every node kept:
//!
//! - official 16: WHIR's BLAKE3 engine, the official crate's hidden
//!   `Platform::hash_many` sixteen messages per call, on one thread;
//! - official 16 + rayon: the same engine under WHIR's `parallel_hash`,
//!   which splits a layer with rayon::join down to pieces of at most
//!   `workload_size` bytes (128 KiB on macOS, 64 KiB on Linux AArch64);
//! - servil: `blake3_servil::hash_many`, one call per layer, one thread;
//! - servil + rayon: the same call as WHIR's engine, under `parallel_hash`;
//! - servil mt: `hash_many_multithreaded`, one call per layer, one thread
//!   making the calls.
//!
//! Every way's layers are checked against servil's before timing. Wall
//! time per tree (median of 11), and per leaf.
//!
//!     cargo run --release --example host_lab

use blake3::platform::{Platform, MAX_SIMD_DEGREE};
use blake3::IncrementCounter;
use std::time::Instant;

#[cfg(target_os = "macos")]
const WORKLOAD: usize = 1 << 17;
#[cfg(not(target_os = "macos"))]
const WORKLOAD: usize = 1 << 16;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

type Engine = fn(usize, &[u8], &mut [[u8; 32]]);

/// WHIR's blake3_engine.rs for sizes 64 and 256.
fn official(size: usize, input: &[u8], out: &mut [[u8; 32]]) {
    fn run<const N: usize>(input: &[u8], out: &mut [[u8; 32]]) {
        let platform = Platform::detect();
        let messages: Vec<&[u8; N]> = input.chunks_exact(N).map(|m| m.try_into().unwrap()).collect();
        for (group, digests) in messages.chunks(MAX_SIMD_DEGREE).zip(out.chunks_mut(MAX_SIMD_DEGREE)) {
            platform.hash_many::<N>(group, &IV, 0, IncrementCounter::No, 0, 1, 2 | 8, digests.as_flattened_mut());
        }
    }
    match size {
        64 => run::<64>(input, out),
        256 => run::<256>(input, out),
        _ => unreachable!(),
    }
}

fn servil(size: usize, input: &[u8], out: &mut [[u8; 32]]) {
    blake3_servil::hash_many(input, size, out);
}

fn servil_mt(size: usize, input: &[u8], out: &mut [[u8; 32]]) {
    blake3_servil::hash_many_multithreaded(input, size, out);
}

/// WHIR's parallel_hash.
fn split(engine: Engine, size: usize, input: &[u8], out: &mut [[u8; 32]]) {
    if input.len() > WORKLOAD && input.len() / size >= 2 {
        let (a, b) = input.split_at(input.len() / 2);
        let (oa, ob) = out.split_at_mut(out.len() / 2);
        rayon::join(|| split(engine, size, a, oa), || split(engine, size, b, ob));
    } else {
        engine(size, input, out);
    }
}

/// Every layer, leaves' digests first, the root last.
fn tree(leaves: &[u8], hash: &dyn Fn(usize, &[u8], &mut [[u8; 32]])) -> Vec<Vec<[u8; 32]>> {
    let mut layers = vec![vec![[0u8; 32]; leaves.len() / 256]];
    hash(256, leaves, &mut layers[0]);
    while layers.last().unwrap().len() > 1 {
        let below = layers.last().unwrap();
        let mut next = vec![[0u8; 32]; below.len() / 2];
        hash(64, below.as_flattened(), &mut next);
        layers.push(next);
    }
    layers
}

fn main() {
    blake3_servil::initialize();
    println!("WHIR Merkle tree, 256-byte leaves; rayon threads {}, WORKLOAD {} KiB", rayon::current_num_threads(), WORKLOAD / 1024);
    for log in [12, 14, 16, 18, 20] {
        let n = 1usize << log;
        let leaves: Vec<u8> = (0..n * 256).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]).collect();
        let ways: [(&str, Box<dyn Fn(usize, &[u8], &mut [[u8; 32]])>); 5] = [
            ("official 16", Box::new(|s, i, o| official(s, i, o))),
            ("official 16 + rayon", Box::new(|s, i, o| split(official, s, i, o))),
            ("servil", Box::new(|s, i, o| servil(s, i, o))),
            ("servil + rayon", Box::new(|s, i, o| split(servil, s, i, o))),
            ("servil mt", Box::new(|s, i, o| servil_mt(s, i, o))),
        ];
        let expected = tree(&leaves, &|s, i, o| servil(s, i, o));
        assert_eq!(*expected[0].last().unwrap(), *blake3::hash(&leaves[(n - 1) * 256..]).as_bytes());
        let mut line = format!("2^{log} leaves:");
        for (name, way) in &ways {
            assert!(tree(&leaves, way.as_ref()) == expected, "{name} disagrees");
            let mut times: Vec<u128> = (0..11)
                .map(|_| {
                    let start = Instant::now();
                    std::hint::black_box(tree(std::hint::black_box(&leaves), way.as_ref()));
                    start.elapsed().as_nanos()
                })
                .collect();
            times.sort();
            let t = times[times.len() / 2];
            line += &format!("  {name} {:.3} ms ({:.1} ns/leaf)", t as f64 / 1e6, t as f64 / n as f64);
        }
        println!("{line}");
    }
}
