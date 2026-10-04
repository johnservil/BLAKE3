//! probe/hash-each: the collection cell's items (bench-hashes COLLECTIONS:
//! git/git's objects and Nix store files, 2048 items each in their shares by
//! octave of size, in the stride's order), hashed one `hash` call per item,
//! against one `hash_each_with` call for all of them. The items lie one after
//! another in one buffer, as the cell has them; each digest is written to the
//! item's slot.

const COLLECTIONS: [(&str, &[(usize, usize)]); 2] = [
    ("git objects", &[(1, 24), (13, 48), (12, 96), (42, 192), (182, 384), (237, 768), (135, 1536), (141, 3072), (173, 6144), (447, 12288), (321, 24576), (231, 49152), (84, 98304), (22, 196608), (4, 393216), (3, 786432)]),
    ("Nix files", &[(6, 1), (1, 6), (1, 12), (4, 24), (11, 48), (44, 96), (139, 192), (133, 384), (439, 768), (428, 1536), (264, 3072), (179, 6144), (130, 12288), (96, 24576), (70, 49152), (46, 98304), (31, 196608), (17, 393216), (9, 786432)]),
];

fn items(classes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let lengths: Vec<usize> = classes.iter().flat_map(|&(n, len)| std::iter::repeat_n(len, n)).collect();
    let count = lengths.len();
    let mut offset = 0;
    (0..count).map(|j| { let len = lengths[j * 389 % count]; offset += len; (offset - len, len) }).collect()
}

fn main() {
    blake3_servil::initialize();
    for (name, classes) in COLLECTIONS {
        let layout = items(classes);
        let total: usize = layout.iter().map(|&(_, l)| l).sum();
        let input: Vec<u8> = (0..total).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]).collect();
        let slices: Vec<&[u8]> = layout.iter().map(|&(o, l)| &input[o..o + l]).collect();
        let mut out = vec![[0u8; 32]; slices.len()];
        // The same digests both ways.
        blake3_servil::hash_each_with(blake3_servil::Mode::Hash, &slices, &mut out);
        for (s, d) in slices.iter().zip(&out) {
            assert_eq!(d, blake3_servil::hash(s).as_bytes());
        }
        // Small items alone (under 16 KiB), where the lanes are the question.
        let small: Vec<&[u8]> = slices.iter().copied().filter(|s| s.len() < 16 * 1024).collect();
        let small_bytes: usize = small.iter().map(|s| s.len()).sum();
        for (what, set, bytes) in [("all items", &slices, total), ("items under 16 KiB", &small, small_bytes)] {
            let mut out = vec![[0u8; 32]; set.len()];
            for round in 0..2 {
                for (how, each) in [("hash per item", false), ("hash_each_with", true)] {
                    let batches = clocks::measure(5, 200_000_000, || {
                        if each {
                            blake3_servil::hash_each_with(blake3_servil::Mode::Hash, set, &mut out);
                        } else {
                            for (s, d) in set.iter().zip(out.iter_mut()) {
                                *d = *blake3_servil::hash(s).as_bytes();
                            }
                        }
                        std::hint::black_box(&out);
                    });
                    let calls: u64 = batches.iter().map(|b| b.calls).sum();
                    let wall: u64 = batches.iter().map(|b| b.wall_ns).sum();
                    let ns_b = (wall as u128 * 10_000 / (calls as u128 * bytes as u128)) as u64;
                    let ns_item = (wall as u128 / (calls as u128 * set.len() as u128)) as u64;
                    println!("{name:12} {what:20} {} items {:>6} KiB  round {round}  {how:14}  {}.{:04} ns/B  {} ns/item   [{}]",
                        set.len(), bytes >> 10, ns_b / 10000, ns_b % 10000, ns_item, batches[0].show());
                }
            }
        }
    }
    eprintln!("host_lab: {}", clocks::load::describe(&clocks::load::windows()));
}
