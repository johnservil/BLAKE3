//! Probe (branch probe/piece-sizes, never merged): what the "for best
//! performance" advice rests on. 8 MiB through Hasher::update and
//! update_multithreaded in pieces of 16 KiB to 8 MiB, against hash and
//! hash_multithreaded on the whole; 1024 one-block messages through
//! hash_many against a loop of hash; the first hash_multithreaded of a
//! process with and without initialize() first. Wall time, and cycles
//! where the OS counts them (examples/support/clocks.rs).
#[path = "support/clocks.rs"]
mod clocks;
use std::hint::black_box;
use std::time::Instant;

unsafe extern "C" {
    fn blake3_sme2_hash16_chunks_512(i: *const *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
    fn blake3_sme2_hash16_parents_512(i: *const u8, k: *const u32, c: u64, f: u32, o: *mut u8, g: u64) -> u64;
}
const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

/// The raw SME2 kernels over `table` (one pointer per chunk) in pieces of
/// `piece_chunks`: the chunk kernel per piece, then with `parents` the
/// flat walk's parent levels down to 16 values (each its own streaming
/// session), and with `gap_ns` that much scalar work between pieces.
fn raw_pieces(table: &[*const u8], piece_chunks: usize, parents: bool, gap_ns: u64, cvs: &mut [u8], half: &mut [u8]) {
    let flags = 1u32 << 8 | 2 << 16;
    for (p, piece) in table.chunks(piece_chunks).enumerate() {
        unsafe {
            blake3_sme2_hash16_chunks_512(piece.as_ptr(), IV.as_ptr(), (p * piece_chunks) as u64, flags, cvs.as_mut_ptr(), (piece_chunks / 16) as u64);
            if parents {
                let mut count = piece_chunks;
                let (mut src, mut dst) = (cvs.as_mut_ptr(), half.as_mut_ptr());
                while count > 16 {
                    blake3_sme2_hash16_parents_512(src, IV.as_ptr(), 0, 4, dst, (count / 32) as u64);
                    count /= 2;
                    std::mem::swap(&mut src, &mut dst);
                }
            }
        }
        if gap_ns > 0 {
            let t = Instant::now();
            while (t.elapsed().as_nanos() as u64) < gap_ns {
                black_box(0);
            }
        }
    }
    black_box(&cvs);
}

fn main() {
    println!("probe/piece-sizes: platform {}", blake3_servil::kernel_report().platform);
    // The first multithreaded call, cold (before initialize), then warm.
    let input: Vec<u8> = (0..8usize << 20).map(|i| (i * 7 + (i >> 10) * 13) as u8).collect();
    let small = &input[..1 << 20];
    let t = Instant::now();
    black_box(blake3_servil::hash_multithreaded(black_box(small)));
    let cold = t.elapsed().as_nanos();
    let t = Instant::now();
    black_box(blake3_servil::hash_multithreaded(black_box(small)));
    let warm = t.elapsed().as_nanos();
    println!("first hash_multithreaded(1 MiB) of the process {cold} ns; the second {warm} ns");
    clocks::set_qos(clocks::USER_INTERACTIVE);
    let whole = blake3_servil::hash(&input);
    for round in 0..2 {
        println!("round {round}, 8 MiB, ns/B (fastest of 9 batches: wall, cycles)");
        let s = clocks::measure(9, 20_000, || { black_box(blake3_servil::hash(black_box(&input))); }).fastest();
        println!("  hash whole                          {:.4}  [{}]", s.ns / input.len() as f64, s.show());
        let s = clocks::measure(9, 20_000, || { black_box(blake3_servil::hash_multithreaded(black_box(&input))); }).fastest();
        println!("  hash_multithreaded whole            {:.4}  [{}]", s.ns / input.len() as f64, s.show());
        for piece in [16usize << 10, 64 << 10, 256 << 10, 1 << 20, 8 << 20] {
            let mut h = blake3_servil::Hasher::new();
            for p in input.chunks(piece) { h.update(p); }
            assert_eq!(h.finalize(), whole);
            let s = clocks::measure(9, 20_000, || {
                let mut h = blake3_servil::Hasher::new();
                for p in black_box(&input).chunks(piece) { h.update(p); }
                black_box(h.finalize());
            }).fastest();
            let m = clocks::measure(9, 20_000, || {
                let mut h = blake3_servil::Hasher::new();
                for p in black_box(&input).chunks(piece) { h.update_multithreaded(p); }
                black_box(h.finalize());
            }).fastest();
            println!("  pieces of {:>5} KiB: update {:.4} [{}]   update_multithreaded {:.4}", piece >> 10, s.ns / input.len() as f64, s.show(), m.ns / input.len() as f64);
        }
        if matches!(blake3_servil::platform::Platform::detect(), blake3_servil::platform::Platform::SME2) {
            let table: Vec<*const u8> = input.chunks(1024).map(|c| c.as_ptr()).collect();
            let (mut cvs, mut half) = (vec![0u8; 32 * 1024], vec![0u8; 32 * 512]);
            for (label, piece, parents, gap) in [
                ("raw 64 KiB: chunk kernel only", 64usize, false, 0u64),
                ("raw 64 KiB: chunks + parent levels", 64, true, 0),
                ("raw 64 KiB: chunks + parents + 0.1 us scalar", 64, true, 100),
                ("raw 64 KiB: chunks + parents + 0.25 us scalar", 64, true, 250),
                ("raw 64 KiB: chunks + parents + 0.5 us scalar", 64, true, 500),
                ("raw 64 KiB: chunks + parents + 1 us scalar", 64, true, 1000),
                ("raw 64 KiB: chunks + parents + 3 us scalar", 64, true, 3000),
                ("raw 256 KiB: chunks + parent levels", 256, true, 0),
                ("raw 1 MiB: chunks + parent levels", 1024, true, 0),
                ("raw 8 MiB: chunk kernel only", 8192, false, 0),
            ] {
                let big = if piece > 1024 { vec![0u8; 32 * piece] } else { Vec::new() };
                let s = clocks::measure(9, 20_000, || {
                    if piece > 1024 {
                        let mut big = big.clone();
                        raw_pieces(&table, piece, parents, gap, &mut big, &mut half);
                    } else {
                        raw_pieces(&table, piece, parents, gap, &mut cvs, &mut half);
                    }
                }).fastest();
                println!("  {label:<44} {:.4}  [{}]", s.ns / input.len() as f64, s.show());
            }
        }
        let messages: Vec<[u8; 64]> = (0..1024).map(|i| [i as u8; 64]).collect();
        let refs: Vec<&[u8]> = messages.iter().map(|m| &m[..]).collect();
        let mut outs = vec![blake3_servil::Hash::from([0; 32]); 1024];
        let s = clocks::measure(9, 20_000, || { blake3_servil::hash_many(black_box(&refs), &mut outs); black_box(&outs); }).fastest();
        let l = clocks::measure(9, 20_000, || { for (m, o) in refs.iter().zip(outs.iter_mut()) { *o = blake3_servil::hash(black_box(m)); } black_box(&outs); }).fastest();
        println!("  1024 x 64 B: hash_many {:.1} ns/msg, a loop of hash {:.1} ns/msg", s.ns / 1024.0, l.ns / 1024.0);
    }
}
