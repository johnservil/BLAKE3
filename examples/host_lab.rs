//! binius64's multi-lane BLAKE3 against the servil fork's, one thread each.
//!
//! binius64 (https://github.com/binius-zk/binius64) hashes the leaves and inner
//! nodes of its Merkle trees with a multi-lane BLAKE3 kernel by Thomas Coratger:
//! N messages of one length, up to one 1024-byte chunk each, side by side, the
//! block compression in hand-written NEON on AArch64 (its `Blake3HashSuite`
//! takes N = 16). This probe times that kernel, vendored unchanged in
//! `examples/binius/`, against `blake3_servil::hash_many`, the fork's batch
//! call, on the same inputs:
//!
//! - batches of 64-byte messages (a Merkle inner node: two 32-byte digests),
//!   256-byte ones (a leaf of WHIR's trees) and 1024-byte ones (a whole chunk);
//! - a Merkle tree of 256-byte leaves, every layer to the root, unkeyed, as
//!   binius builds its trees: blake3(leaf), then blake3(left || right).
//!
//! Every digest of both is checked against `blake3_servil::hash` (tested
//! against BLAKE3's official vectors) before anything is timed. Timing is the
//! fork's `clocks` crate: wall time per batch of calls, and on macOS the
//! thread's cycles per core kind. A case runs `ROUNDS` rounds in a process,
//! the two sides alternating (A B B A ...), and the figure is the run's mean;
//! run the program several times to see how runs differ.
//!
//! Reproduce: check out this commit of github.com/johnservil/BLAKE3 and run
//!
//!     cargo run --release --example host_lab
//!
//! on an AArch64 machine (an Apple M4 has SME2, which blake3_servil uses for
//! batches of 16 or more; without it the fork runs its integer + NEON kernels).
//! It prints the table and writes it to binius-compare.txt.

#[path = "binius/blake3_portable.rs"]
#[allow(dead_code, clippy::all)]
mod binius;

use std::hint::black_box;
use std::mem::MaybeUninit;

const ROUNDS: usize = 4;
const BATCHES: usize = 8;
const BATCH_NS: u64 = 40_000_000;

/// `input`'s messages of `len` bytes through binius' kernel, `L` at a time;
/// a last group short of `L` runs all `L` lanes, its spare ones repeating
/// the last message, and keeps its own lanes' digests (as binius'
/// ParallelMultidigestImpl runs a short group, a whole pass).
fn binius_many<const L: usize>(input: &[u8], len: usize, out: &mut [[u8; 32]]) {
    let n = input.len() / len;
    assert!(n * len == input.len() && out.len() >= n && len <= 1024);
    let mut g = 0;
    while g < n {
        let data: [&[u8]; L] = core::array::from_fn(|i| {
            let k = (g + i).min(n - 1);
            &input[k * len..(k + 1) * len]
        });
        let mut digests = [MaybeUninit::uninit(); L];
        binius::PortableBlake3MultiDigest::<L>::digest(data, &mut digests);
        for (i, d) in digests.iter().take(n - g).enumerate() {
            // Sound: digest writes every lane.
            out[g + i] = unsafe { d.assume_init() };
        }
        g += L;
    }
}

type Many = fn(&[u8], usize, &mut [[u8; 32]]);

const SIDES: [(&str, Many); 3] = [
    ("servil hash_many", blake3_servil::hash_many),
    ("binius 16 lanes", binius_many::<16>),
    ("binius 12 lanes", binius_many::<12>),
];

/// A Merkle tree over `leaves` of `leaf_len` bytes, unkeyed, with `many`
/// for every layer; the root.
fn merkle(many: Many, leaves: &[u8], leaf_len: usize, layers: &mut [Vec<[u8; 32]>]) -> [u8; 32] {
    let mut from = leaves;
    let mut len = leaf_len;
    for layer in layers.iter_mut() {
        let n = from.len() / len;
        many(from, len, &mut layer[..n]);
        from = layer[..n].as_flattened();
        len = 64;
    }
    layers.last().unwrap()[0]
}

fn input(len: usize, count: usize) -> Vec<u8> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..len * count)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// The case's figure for one side: Q64.64 ns per call, and the batches.
fn time(f: &mut dyn FnMut()) -> (u128, Vec<clocks::Batch>) {
    let batches = clocks::measure(BATCHES, BATCH_NS, &mut *f);
    (clocks::summary::mean(batches.iter().map(|b| (b.wall_ns, b.calls))), batches)
}

/// Cycles per call over the batches, with the clock and core kind they ran
/// at, where the platform counts them.
fn cycles(batches: &[clocks::Batch]) -> String {
    let calls: u64 = batches.iter().map(|b| b.calls).sum();
    let mut total = clocks::Counts::default();
    for b in batches {
        match b.counts {
            Some(c) => {
                total.p.cycles += c.p.cycles;
                total.p.time_ns += c.p.time_ns;
                total.e.cycles += c.e.cycles;
                total.e.time_ns += c.e.time_ns;
            }
            None => return "no cycle counts on this platform".to_owned(),
        }
    }
    let cyc = total.p.cycles + total.e.cycles;
    format!("{} cycles/call at {} MHz, {}% E", (cyc + calls / 2) / calls, total.mhz(), total.e_percent())
}

fn main() {
    blake3_servil::initialize();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    };
    say(format!("binius64 BLAKE3 (vendored, 92a7e33c) against blake3_servil::hash_many, one thread; {ROUNDS} rounds of {BATCHES} batches of about {} ms a side", BATCH_NS / 1_000_000));
    say(format!("{}, {}", std::env::consts::OS, std::env::consts::ARCH));

    // (what, message length, messages); a Merkle case is ("merkle", leaf length, leaves).
    let cases: &[(&str, usize, usize)] = &[
        ("batch", 64, 16),
        ("batch", 64, 1024),
        ("batch", 64, 65536),
        ("batch", 256, 16),
        ("batch", 256, 4096),
        ("batch", 1024, 16),
        ("batch", 1024, 1024),
        ("merkle", 256, 1 << 10),
        ("merkle", 256, 1 << 16),
    ];
    for &(what, len, count) in cases {
        let data = input(len, count);
        let mut out = vec![[0u8; 32]; count];
        let mut layers: Vec<Vec<[u8; 32]>> = Vec::new();
        let mut n = count;
        while n >= 1 {
            layers.push(vec![[0u8; 32]; n]);
            if n == 1 {
                break;
            }
            n /= 2;
        }
        // Correctness first, every side.
        let want: Vec<[u8; 32]> = data.chunks_exact(len).map(|m| *blake3_servil::hash(m).as_bytes()).collect();
        let want_root = {
            let mut layer = want.clone();
            while layer.len() > 1 {
                layer = layer.chunks_exact(2).map(|p| *blake3_servil::hash(p.as_flattened()).as_bytes()).collect();
            }
            layer[0]
        };
        for &(name, many) in &SIDES {
            if what == "batch" {
                out.iter_mut().for_each(|d| *d = [0; 32]);
                many(&data, len, &mut out);
                assert!(out == want, "{name}: a wrong digest, {count} x {len} B");
            } else {
                assert_eq!(merkle(many, &data, len, &mut layers), want_root, "{name}: a wrong root, {count} leaves of {len} B");
            }
        }
        let units = count as u128; // messages a call (leaves, for a tree)
        let bytes = (len * count) as u128;
        let mut means: Vec<Vec<u128>> = vec![Vec::new(); SIDES.len()];
        let mut last: Vec<Vec<clocks::Batch>> = vec![Vec::new(); SIDES.len()];
        for round in 0..ROUNDS {
            let order: Vec<usize> = if round % 2 == 0 { (0..SIDES.len()).collect() } else { (0..SIDES.len()).rev().collect() };
            for s in order {
                let many = SIDES[s].1;
                let (mean, batches) = if what == "batch" {
                    time(&mut || many(black_box(&data), len, black_box(&mut out)))
                } else {
                    time(&mut || {
                        black_box(merkle(many, black_box(&data), len, &mut layers));
                    })
                };
                means[s].push(mean);
                last[s].extend(batches);
            }
        }
        say(String::new());
        say(if what == "batch" { format!("{count} messages of {len} B, one call") } else { format!("Merkle tree, {count} leaves of {len} B, every layer to the root") });
        let base = means[0].iter().sum::<u128>() / ROUNDS as u128;
        for (s, &(name, _)) in SIDES.iter().enumerate() {
            let mean = means[s].iter().sum::<u128>() / ROUNDS as u128; // Q64.64 ns a call
            let per_unit_ps = ((mean * 1000 / units) + (1 << 63)) >> 64;
            let gbps_milli = (bytes << 64) * 1000 / mean; // bytes per ns, x1000
            let ratio = (mean * 1000 + base / 2) / base;
            let rounds: Vec<String> = means[s].iter().map(|m| format!("{}", ((m * 1000 / units) + (1 << 63)) >> 64)).collect();
            say(format!(
                "  {name:17} {:>7}.{:03} ns/message  {:>3}.{:03} GB/s  x{}.{:03}  (rounds, ps/message: {})  {}",
                per_unit_ps / 1000,
                per_unit_ps % 1000,
                gbps_milli / 1000,
                gbps_milli % 1000,
                ratio / 1000,
                ratio % 1000,
                rounds.join(" "),
                cycles(&last[s])
            ));
        }
    }
    std::fs::write("binius-compare.txt", report).expect("write binius-compare.txt");
}
