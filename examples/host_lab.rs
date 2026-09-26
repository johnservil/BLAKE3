//! Probe (branch probe/commonware, never merged): commonwarexyz's BLAKE3
//! batch kernels (monorepo PR 4982, commit 25851f1: `Blake3::hash_many`
//! over a slice of messages, NEON four lanes on AArch64, and `hash_pair`,
//! two interleaved scalar lanes) against the servil fork's `hash_many`
//! and `hash_many_multithreaded` and a loop of the official crate's
//! `blake3::hash`, on the benchmark's batch shapes: 64- and 256-byte
//! messages, plus 4 KiB (a size of commonware's own table). Single
//! messages in commonware go to the official crate, so they are the
//! official contender's cells already.
//!
//! Every cell: digests checked against `blake3::hash` outside the timed
//! batches; then four rounds of the four contenders in turn (A B C D, D C
//! B A, ...), 5 batches of about 1 ms per contender per round; the
//! median batch's ns per message and its cycles per ns (clocks.rs). At
//! user-interactive QoS (P-cores) and at background QoS (E-cores).
#[path = "support/clocks.rs"]
mod clocks;

use commonware_cryptography::{Blake3 as Cw, Hasher};
use std::hint::black_box;

/// xorshift64, seed 0x9E3779B97F4A7C15: the probe's input bytes.
fn bytes(n: usize) -> Vec<u8> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

const NAMES: [&str; 4] = ["commonware", "official loop", "servil", "servil mt"];

fn cell(len: usize, n: usize) {
    let input = bytes(len * n);
    let slices: Vec<&[u8]> = input.chunks(len).collect();
    let mut out = vec![[0u8; 32]; n];
    let mut out_mt = vec![[0u8; 32]; n];
    let cw = Cw::hash_many(&slices);
    blake3_servil::hash_many(&input, len, &mut out);
    blake3_servil::hash_many_multithreaded(&input, len, &mut out_mt);
    for i in 0..n {
        let want = *blake3::hash(slices[i]).as_bytes();
        assert_eq!(cw[i].0, want, "commonware, {n} x {len} B, message {i}");
        assert_eq!(out[i], want, "servil, {n} x {len} B, message {i}");
        assert_eq!(out_mt[i], want, "servil mt, {n} x {len} B, message {i}");
    }
    let mut samples: [Vec<clocks::Sample>; 4] = Default::default();
    for round in 0..4 {
        for k in 0..4 {
            let c = if round % 2 == 0 { k } else { 3 - k };
            let m = match c {
                0 => clocks::measure(5, 1000, || {
                    black_box(Cw::hash_many(black_box(&slices)));
                }),
                1 => clocks::measure(5, 1000, || {
                    for s in &slices {
                        black_box(blake3::hash(black_box(s)));
                    }
                }),
                2 => clocks::measure(5, 1000, || {
                    blake3_servil::hash_many(black_box(&input), len, &mut out);
                    black_box(&out);
                }),
                _ => clocks::measure(5, 1000, || {
                    blake3_servil::hash_many_multithreaded(black_box(&input), len, &mut out_mt);
                    black_box(&out_mt);
                }),
            };
            samples[c].extend(m.batches);
        }
    }
    let mut line = format!("{len:>5} B x {n:>5}:");
    for (c, s) in samples.iter().enumerate() {
        let med = clocks::Measurement { batches: s.clone() }.median();
        line += &format!("  {}: {:.1} ns/msg ({})", NAMES[c], med.ns / n as f64, if med.has_cycles() { format!("{:.2}/ns{}", med.per_ns(), if med.e_share() > 0.5 { " E" } else { "" }) } else { "-".into() });
    }
    println!("{line}");
}

fn pair() {
    let l = bytes(64);
    let r: Vec<u8> = bytes(65)[1..].to_vec();
    let mut both = l.clone();
    both.extend_from_slice(&r);
    let mut out = [[0u8; 32]; 2];
    let (a, b) = Cw::hash_pair(&[&l[..32], &l[32..]], &[&r[..32], &r[32..]]);
    blake3_servil::hash_many(&both, 64, &mut out);
    assert_eq!((a.0, b.0), (*blake3::hash(&l).as_bytes(), *blake3::hash(&r).as_bytes()));
    assert_eq!(out, [a.0, b.0]);
    let cw = clocks::measure(21, 1000, || {
        black_box(Cw::hash_pair(black_box(&[&l[..32], &l[32..]]), black_box(&[&r[..32], &r[32..]])));
    })
    .median();
    let sv = clocks::measure(21, 1000, || {
        blake3_servil::hash_many(black_box(&both), 64, &mut out);
        black_box(&out);
    })
    .median();
    println!("pair of 64 B messages, ns per pair: commonware hash_pair ({}), servil hash_many of 2 ({})", cw.show(), sv.show());
}

/// servil's hash_many alone at every count to 40 and a few beyond: the
/// median of 21 batches of about 1 ms, ns per message (cycles per ns).
fn sweep(len: usize) {
    let mut line = format!("sweep {len:>4} B:");
    for n in (1..=40).chain([53, 100, 127, 128]) {
        let input = bytes(len * n);
        let mut out = vec![[0u8; 32]; n];
        let m = clocks::measure(21, 1000, || {
            blake3_servil::hash_many(black_box(&input), len, &mut out);
            black_box(&out);
        })
        .median();
        line += &format!(" {n}:{:.1}", m.ns / n as f64);
        if m.has_cycles() {
            line += &format!("({:.2})", m.per_ns());
        }
    }
    println!("{line}");
}

fn main() {
    blake3_servil::initialize();
    clocks::set_qos(clocks::USER_INTERACTIVE);
    for len in [64, 256] {
        sweep(len);
    }
    if std::env::var_os("SWEEP_ONLY").is_some() { return; }
    for (qos, name) in [(clocks::USER_INTERACTIVE, "user-interactive QoS")] {
        clocks::set_qos(qos);
        println!("== {name} ==");
        pair();
        for len in [64] {
            for n in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 64, 256, 1024, 16384] {
                cell(len, n);
            }
        }
    }
}
