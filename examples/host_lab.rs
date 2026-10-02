//! probe/willneed-mechanism: what madvise(WILLNEED) costs and saves when a
//! file already in the page cache is mapped and hashed, by file size and
//! by how many threads hash it.
//!
//! Each run maps the cached file afresh (as b3sum does), then optionally
//! calls madvise(WILLNEED) over the whole mapping, then hashes it, on the
//! pool (update_multithreaded) or on this thread alone (update). Four
//! modes, rotated each round: pool, pool+hint, one, one+hint. Per run: the
//! hint call's wall time and this thread's cycles in it; the hashing's wall
//! time; the process's minor faults and user/system CPU time over the run
//! (clocks::child::Usage). Prints each cell's medians (speeds per
//! clocks::speeds for wall times) and writes every sample to
//! willneed-samples.tsv in the working directory.

use clocks::child::Usage;
use std::io::{Read, Write};

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}
const PROT_READ: i32 = 1;
const MAP_SHARED: i32 = 1;
const MADV_WILLNEED: i32 = 3;

const SIZES: [usize; 7] = [64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 256 << 20, 1 << 30];
const MODES: [(&str, bool, bool); 4] = [("pool", true, false), ("pool+hint", true, true), ("one", false, false), ("one+hint", false, true)];
const ROUNDS: usize = 15;

struct Sample {
    hint_ns: u64,
    hint_cycles: Option<u64>,
    hash_ns: u64,
    usage: Usage,
}

fn main() {
    blake3_servil::initialize_multithreaded();
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let dir = home.join("willneed-files");
    std::fs::create_dir_all(&dir).unwrap();
    let page = page_size::get();
    println!("page size {page} B; {} CPUs; {ROUNDS} rounds", std::thread::available_parallelism().unwrap());
    let mut tsv = String::from("size\tmode\tround\thint_ns\thint_cycles\thash_ns\tuser_ns\tsystem_ns\tminor_faults\tmajor_faults\n");
    for &size in &SIZES {
        // The file, its bytes BLAKE3 output of its name, read once so the page cache holds it.
        let path = dir.join(format!("file-{size}"));
        if std::fs::metadata(&path).map(|m| m.len() as usize).ok() != Some(size) {
            let mut bytes = vec![0u8; size];
            blake3_servil::Hasher::new().update(format!("file-{size}").as_bytes()).finalize_xof().fill(&mut bytes);
            std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        }
        let mut sink = Vec::new();
        std::fs::File::open(&path).unwrap().read_to_end(&mut sink).unwrap();
        let expected = blake3_servil::hash(&sink);
        drop(sink);
        let file = std::fs::File::open(&path).unwrap();
        let mut cells: Vec<Vec<Sample>> = (0..MODES.len()).map(|_| Vec::new()).collect();
        for round in 0..=ROUNDS {
            for k in 0..MODES.len() {
                let m = (k + round) % MODES.len();
                let (_, pool, hint) = MODES[m];
                use std::os::fd::AsRawFd;
                // Sound: a fresh read-only shared mapping of an open file, unmapped below.
                let at = unsafe { mmap(std::ptr::null_mut(), size, PROT_READ, MAP_SHARED, file.as_raw_fd(), 0) };
                assert!(at as isize != -1, "mmap");
                // Sound: the mapping covers `size` bytes while it lives.
                let map = unsafe { std::slice::from_raw_parts(at, size) };
                clocks::load::tick();
                let before = Usage::own().unwrap();
                let (mut hint_ns, mut hint_cycles) = (0, None);
                if hint {
                    let counts = clocks::Counts::read();
                    let t = clocks::now();
                    // Sound: advice over the mapping.
                    assert_eq!(unsafe { madvise(at, size, MADV_WILLNEED) }, 0, "madvise");
                    hint_ns = clocks::since_ns(t);
                    hint_cycles = counts.zip(clocks::Counts::read()).map(|(a, b)| {
                        let c = b.since(a);
                        c.p.cycles + c.e.cycles
                    });
                }
                let t = clocks::now();
                let mut hasher = blake3_servil::Hasher::new();
                if pool {
                    hasher.update_multithreaded(map);
                } else {
                    hasher.update(map);
                }
                let digest = hasher.finalize();
                let hash_ns = clocks::since_ns(t);
                let usage = Usage::own().unwrap().since(before);
                assert_eq!(digest, expected, "the mapping hashes as the file reads");
                // Sound: the mapping made above, no longer borrowed.
                assert_eq!(unsafe { munmap(at, size) }, 0, "munmap");
                if round == 0 {
                    continue; // a warm-up round
                }
                tsv += &format!("{size}\t{}\t{round}\t{hint_ns}\t{}\t{hash_ns}\t{}\t{}\t{}\t{}\n", MODES[m].0,
                    hint_cycles.map_or("-".into(), |c| c.to_string()), usage.user_ns, usage.system_ns, usage.minor_faults, usage.major_faults);
                cells[m].push(Sample { hint_ns, hint_cycles, hash_ns, usage });
            }
        }
        println!("\n{} KiB ({} pages):", size >> 10, size.div_ceil(page));
        for (m, samples) in cells.iter().enumerate() {
            let median = |mut v: Vec<u64>| -> u64 { v.sort_unstable(); v[v.len() / 2] };
            let mut total: Vec<u128> = samples.iter().map(|s| clocks::speeds::per_unit(s.hint_ns + s.hash_ns, 1)).collect();
            total.sort_unstable();
            let speeds: Vec<String> = clocks::speeds::speeds(&total).iter().map(|sp| format!("{} ns ({}/{})", sp.median >> 64, sp.count, samples.len())).collect();
            println!(
                "  {:<10} total {:<30} hint {:>9} ns ({:>8} cycles)  hash {:>9} ns  minor faults {:>6}  user {:>9} ns  system {:>9} ns",
                MODES[m].0,
                speeds.join(" | "),
                median(samples.iter().map(|s| s.hint_ns).collect()),
                samples.iter().filter_map(|s| s.hint_cycles).max().map_or("-".into(), |_| median(samples.iter().filter_map(|s| s.hint_cycles).collect()).to_string()),
                median(samples.iter().map(|s| s.hash_ns).collect()),
                median(samples.iter().map(|s| s.usage.minor_faults).collect()),
                median(samples.iter().map(|s| s.usage.user_ns).collect()),
                median(samples.iter().map(|s| s.usage.system_ns).collect()),
            );
        }
    }
    println!("\nload: {}", clocks::load::describe(&clocks::load::windows()));
    std::fs::write("willneed-samples.tsv", tsv).unwrap();
}
