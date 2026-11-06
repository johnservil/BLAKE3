//! PROBE (probe/pair-warm): batches of 1 to 8 one-block messages on the
//! pair kernel alone against the parent plans (hash_many's kernels), warm
//! (back to back) and after other work (a 32 MiB write between calls):
//! whether the pair kernel, 2.9 KB of code against the plans' up to 17 KB,
//! can serve every call (AGENTS: a performance call is judged warm; the
//! smaller kernel wins on a tie). Means of each, five rounds interleaved.
//!
//!     cargo run --release --example host_lab
fn main() {
    let messages: Vec<[u8; 64]> = (0..16).map(|i| [i as u8 ^ 0x5a; 64]).collect();
    let table: Vec<&[u8; 64]> = messages.iter().collect();
    let mut out = vec![0u8; 16 * 32];
    let mut junk = vec![0u8; 32 << 20];
    for n in 1..=8 {
        let mut a = vec![0u8; n * 32];
        let mut b = vec![0u8; n * 32];
        blake3_servil::__probe_pairs(&table[..n], &mut a, true);
        blake3_servil::__probe_pairs(&table[..n], &mut b, false);
        assert_eq!(a, b, "{n} messages: the two kernels agree");
    }
    println!("# n\tkernel\twarm_ns\tafter_other_work_ns");
    for round in 0..5 {
        for n in 1..=8usize {
            for pairs in [true, false] {
                let name = if pairs { "pair" } else { "plans" };
                let reps = 20000u64;
                for _ in 0..1000 {
                    blake3_servil::__probe_pairs(&table[..n], &mut out, pairs);
                }
                let t = clocks::now();
                for _ in 0..reps {
                    blake3_servil::__probe_pairs(std::hint::black_box(&table[..n]), &mut out, pairs);
                }
                let warm = clocks::since_ns(t) / reps;
                let cold_reps = 200u64;
                let mut cold = 0u64;
                for k in 0..cold_reps {
                    for x in junk.iter_mut().step_by(64) {
                        *x = x.wrapping_add(k as u8);
                    }
                    std::hint::black_box(&junk);
                    let t = clocks::now();
                    blake3_servil::__probe_pairs(std::hint::black_box(&table[..n]), &mut out, pairs);
                    cold += clocks::since_ns(t);
                }
                println!("{round}\t{n}\t{name}\t{warm}\t{}", cold / cold_reps);
            }
        }
    }
    clocks::load::tick();
    println!("# load: {}", clocks::load::describe(&clocks::load::windows()));
}
