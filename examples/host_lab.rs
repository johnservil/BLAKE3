//! probe/code-prefetch: does prefetching a kernel's code before a call
//! after other work recover the time its instruction lines take to come
//! from DRAM (bench-hashes NOTES, "The cause: where the hash's code is")?
//!
//! Each call follows the benchmark's own gap (clocks::Gap::Busy: a fixed
//! other program of about 1 MiB of code, a walk of 128 MiB, register work
//! to 1 ms), then the producer's copy, then the variant and `hash`, timed
//! together (the prefetch is part of the call's cost). Variants: nothing;
//! PRFM PLDL2KEEP over the kernels' code (64-byte steps); PRFM PLIL1KEEP;
//! PRFM PLIL2KEEP; a load of each line; PLDL2KEEP every 128 bytes. The ranges are the assembly
//! kernels each length's path runs (symbol to next symbol), Rust glue
//! excluded. Then the same variants nonstop (back to back, code warm):
//! the prefetch's cost when it finds the lines near.
//!
//! Round two: 64 B, 512 B, 1 KiB, the scalar kernel c1 (k1's code) alone.
//! Rounds interleave the variants.
use std::hint::black_box;
use std::io::Write;

unsafe extern "C" {
    fn blake3_hybrid_p2();
    fn blake3_hybrid_p4();
    fn blake3_hybrid_p8();
    fn blake3_hybrid_p3();
    fn blake3_hybrid_k1();
    fn blake3_hybrid_k2();
    fn blake3_hybrid_k3();
    fn blake3_hybrid_k4();
    fn blake3_hybrid_k5();
    fn blake3_hybrid_k8();
    fn blake3_hybrid_k9();
    fn blake3_sme2_hash16_chunks_512();
}

const SME2_TEXT: usize = 11256;

fn at(f: unsafe extern "C" fn()) -> usize {
    f as usize
}

/// (start, length) of each kernel range a length's path runs.
fn ranges(len: usize) -> Vec<(usize, usize)> {
    let span = |a: unsafe extern "C" fn(), b: unsafe extern "C" fn()| (at(a), at(b) - at(a));
    let p2 = span(blake3_hybrid_p2, blake3_hybrid_p4);
    let p4 = span(blake3_hybrid_p4, blake3_hybrid_p8);
    let p8 = span(blake3_hybrid_p8, blake3_hybrid_p3);
    let c1 = span(blake3_hybrid_k1, blake3_hybrid_k2);
    let sme2 = (at(blake3_sme2_hash16_chunks_512), SME2_TEXT);
    match len {
        64 | 512 | 1024 => vec![c1],
        2048 => vec![span(blake3_hybrid_k2, blake3_hybrid_k3), p2, c1],
        4096 => vec![span(blake3_hybrid_k4, blake3_hybrid_k5), p2, c1],
        8192 => vec![span(blake3_hybrid_k8, blake3_hybrid_k9), p4, p2, c1],
        _ => vec![sme2, p8, p4, p2, c1],
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    None,
    Pldl2,
    Plil1,
    Plil2,
    Load,
    Pldl2Wide,
}
const VARIANTS: [(Variant, &str); 6] = [(Variant::None, "nothing"), (Variant::Pldl2, "PLDL2KEEP"), (Variant::Pldl2Wide, "PLDL2/128"), (Variant::Plil1, "PLIL1KEEP"), (Variant::Plil2, "PLIL2KEEP"), (Variant::Load, "loads")];

#[inline(always)]
fn prefetch(variant: Variant, ranges: &[(usize, usize)]) {
    let mut sum = 0u64;
    for &(start, len) in ranges {
        let mut a = start & !63;
        let end = start + len;
        let step = if variant == Variant::Pldl2Wide { 128 } else { 64 };
        while a < end {
            // Sound: prefetches never fault; the loads read mapped text.
            unsafe {
                match variant {
                    Variant::None => {}
                    Variant::Pldl2 | Variant::Pldl2Wide => core::arch::asm!("prfm pldl2keep, [{0}]", in(reg) a, options(nostack, readonly)),
                    Variant::Plil1 => core::arch::asm!("prfm plil1keep, [{0}]", in(reg) a, options(nostack, readonly)),
                    Variant::Plil2 => core::arch::asm!("prfm plil2keep, [{0}]", in(reg) a, options(nostack, readonly)),
                    Variant::Load => sum = sum.wrapping_add(core::ptr::read_volatile(a as *const u64)),
                }
            }
            a += step;
        }
    }
    black_box(sum);
}

fn show(values: &[u128]) -> String {
    let mut values = values.to_vec();
    values.sort_unstable();
    let n = values.len();
    clocks::speeds::speeds(&values)
        .into_iter()
        .map(|speed| {
            let v = (speed.median + (1u128 << 63)) >> 64;
            let share = (speed.count * 1000 + n / 2) / n;
            format!("{v} ({share}/1000)")
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn main() {
    blake3_servil::initialize();
    let lengths = [64usize, 512, 1024];
    let rounds = 64;
    let calls = 4u64;
    let work = vec![1u8; 128 << 20];
    let mut report = String::new();
    let mut line = |s: String| {
        println!("{s}");
        report.push_str(&s);
        report.push('\n');
    };
    line(format!("probe/code-prefetch: {} rounds, {} calls a sample after clocks::Gap::Busy (128 MiB), ns per call", rounds, calls));
    for &len in &lengths {
        let r = ranges(len);
        let bytes: usize = r.iter().map(|x| x.1).sum();
        let source: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
        let mut input = vec![0u8; len];
        let mut cold: Vec<Vec<u128>> = vec![Vec::new(); VARIANTS.len()];
        let mut cycles: Vec<Vec<u128>> = vec![Vec::new(); VARIANTS.len()];
        for _ in 0..rounds {
            for (k, &(variant, _)) in VARIANTS.iter().enumerate() {
                let m = clocks::measure_after_gaps_prepared(
                    calls,
                    clocks::Gap::Busy(&work),
                    1_000_000,
                    &mut input,
                    |input| input.copy_from_slice(black_box(&source)),
                    |input| {
                        prefetch(variant, &r);
                        black_box(blake3_servil::hash(black_box(input)));
                    },
                );
                cold[k].push(clocks::speeds::per_unit(m.calls.wall_ns, calls));
                if let Some(c) = m.calls.counts {
                    cycles[k].push(clocks::speeds::per_unit(c.p.cycles + c.e.cycles, calls));
                }
            }
        }
        let mut warm: Vec<Vec<u128>> = vec![Vec::new(); VARIANTS.len()];
        for _ in 0..8 {
            for (k, &(variant, _)) in VARIANTS.iter().enumerate() {
                for b in clocks::measure(5, 2_000_000, || {
                    prefetch(variant, &r);
                    black_box(blake3_servil::hash(black_box(&input)));
                }) {
                    warm[k].push(clocks::speeds::per_unit(b.wall_ns, b.calls));
                }
            }
        }
        line(format!("{len} B, kernels' code {bytes} B ({} lines of 64 B)", bytes.div_ceil(64)));
        for (k, &(_, name)) in VARIANTS.iter().enumerate() {
            let cyc = if cycles[k].is_empty() { "no cycle counts".to_owned() } else { show(&cycles[k]) };
            line(format!("  {name:10} after other work {:28} cycles {:28} nonstop {}", show(&cold[k]), cyc, show(&warm[k])));
        }
    }
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
