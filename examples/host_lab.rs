//! 0u.io/b3 (a Go BLAKE3 with NEON, SME, AVX2 and AVX-512 kernels) against
//! blake3-servil, on the machine it runs on.
//!
//! The Go package is built here as a C shared library from examples/b3go
//! (`go build -buildmode=c-shared`, Go 1.27 or later on PATH or in $GO),
//! loaded with dlopen, and timed with the fork's clocks crate, as the fork
//! is. Each Go function loops inside Go, and the fork's side loops the same
//! way, so one crossing from C into Go covers many hashes. Every digest of
//! both is checked against blake3_servil::hash first.
//!
//! Cases: one message, single-threaded (b3.Sum256 against
//! blake3_servil::hash), 64 B to 128 MiB; one message on every CPU (a Hasher
//! with WithParallelism(threads) against hash_multithreaded), back to back
//! and after 1 ms of sleep before each call (0u.io/b3's pool spins between
//! calls; the fork's sleeps); extended output, 64 KiB and 1 MiB of it.
//!
//!     cargo run --release --example host_lab
//!
//! prints the table and writes it to gob3-compare.txt.

use std::ffi::CString;
use std::hint::black_box;

type Sum = unsafe extern "C" fn(*const u8, usize, *mut u8, usize);
type Par = unsafe extern "C" fn(*const u8, usize, *mut u8, usize, i32);
type Xof = unsafe extern "C" fn(*const u8, usize, *mut u8, usize, usize);

/// The Go library, built from examples/b3go into the temporary directory.
fn build_go() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/b3go");
    let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
    let lib = std::env::temp_dir().join(format!("libb3go-{}.{ext}", std::process::id()));
    let go = std::env::var("GO").unwrap_or_else(|_| {
        ["go", "/opt/homebrew/bin/go", "/usr/local/go/bin/go"]
            .into_iter()
            .find(|g| std::process::Command::new(g).arg("version").output().is_ok())
            .expect("Go 1.27 or later is needed to build 0u.io/b3: install it (brew install go) or set GO")
            .to_owned()
    });
    let status = std::process::Command::new(&go)
        .args(["build", "-buildmode=c-shared", "-o"])
        .arg(&lib)
        .arg(".")
        .current_dir(&dir)
        .env("GOFLAGS", "-mod=mod")
        .status()
        .expect("run go build");
    assert!(status.success(), "go build of examples/b3go failed");
    lib
}

unsafe fn symbol<T>(handle: *mut libc::c_void, name: &str) -> T {
    let c = CString::new(name).unwrap();
    // Sound: the library exports `name` with type T (examples/b3go).
    unsafe {
        let p = libc::dlsym(handle, c.as_ptr());
        assert!(!p.is_null(), "{name} not in the Go library");
        std::mem::transmute_copy(&p)
    }
}

fn input(len: usize) -> Vec<u8> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Q64.64 ns per hash: `f()` hashes `per` times.
fn back_to_back(per: usize, mut f: impl FnMut()) -> (u128, String) {
    let b = clocks::measure(8, 50_000_000, &mut f);
    let mean = clocks::summary::mean(b.iter().map(|b| (b.wall_ns, b.calls * per as u64)));
    (mean, b[b.len() / 2].show())
}

/// Q64.64 ns per call, each after 1 ms asleep.
fn after_gaps(calls: u64, data: &[u8], f: impl Fn(&[u8])) -> u128 {
    let mut d = data.to_vec();
    let m = clocks::measure_after_gaps_prepared(calls, clocks::Gap::Idle, 1_000_000, &mut d[..], |_| {}, |d| f(d));
    clocks::summary::mean([(m.calls.wall_ns, m.calls.calls)])
}

fn ns(q: u128) -> String {
    let milli = (q * 1000 + (1 << 63)) >> 64;
    format!("{}.{:03}", milli / 1000, milli % 1000)
}

fn main() {
    let lib = build_go();
    let path = CString::new(lib.to_str().unwrap()).unwrap();
    // Sound: dlopen of the library just built.
    let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW) };
    assert!(!handle.is_null(), "dlopen {}", lib.display());
    let (go_sum, go_par, go_xof): (Sum, Par, Xof) = unsafe { (symbol(handle, "b3go_sum256"), symbol(handle, "b3go_parallel"), symbol(handle, "b3go_xof")) };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    blake3_servil::initialize_multithreaded(None);
    let mut report = String::new();
    let mut say = |s: String| {
        println!("{s}");
        report.push_str(&s);
        report.push('\n');
    };
    say(format!("0u.io/b3 v1.0.0 (Go) against blake3-servil, {} {}, {threads} CPUs", std::env::consts::OS, std::env::consts::ARCH));
    say("ns per message, lower is better; x: 0u.io/b3's time over blake3-servil's (above 1: blake3-servil faster)".to_owned());

    say(String::new());
    say("one message, one thread (b3.Sum256, blake3_servil::hash)".to_owned());
    for len in [64, 1024, 4096, 16 << 10, 64 << 10, 1 << 20, 16 << 20, 128 << 20] {
        let data = input(len);
        let per = ((1 << 20) / len).max(1);
        let mut out = [0u8; 32];
        unsafe { go_sum(data.as_ptr(), len, out.as_mut_ptr(), 1) };
        assert_eq!(out, *blake3_servil::hash(&data).as_bytes(), "0u.io/b3 Sum256, {len} B");
        let (theirs, t_show) = back_to_back(per, || unsafe { go_sum(black_box(data.as_ptr()), len, out.as_mut_ptr(), per) });
        let (ours, o_show) = back_to_back(per, || {
            for _ in 0..per {
                black_box(blake3_servil::hash(black_box(&data)));
            }
        });
        say(format!("  {len:>10} B   0u.io/b3 {:>12}   blake3-servil {:>12}   x{}   ({t_show} | {o_show})", ns(theirs), ns(ours), ratio(theirs, ours)));
    }

    say(String::new());
    say(format!("one message, every CPU ({threads} threads: WithParallelism({threads}), hash_multithreaded)"));
    for len in [1 << 20, 16 << 20, 128 << 20] {
        let data = input(len);
        let mut out = [0u8; 32];
        unsafe { go_par(data.as_ptr(), len, out.as_mut_ptr(), 1, threads as i32) };
        assert_eq!(out, *blake3_servil::hash(&data).as_bytes(), "0u.io/b3 WithParallelism, {len} B");
        let (theirs, _) = back_to_back(1, || unsafe { go_par(black_box(data.as_ptr()), len, out.as_mut_ptr(), 1, threads as i32) });
        let (ours, _) = back_to_back(1, || {
            black_box(blake3_servil::hash_multithreaded(black_box(&data)));
        });
        let calls = if len > 16 << 20 { 20 } else { 100 };
        let theirs_gap = after_gaps(calls, &data, |d| unsafe {
            let mut o = [0u8; 32];
            go_par(d.as_ptr(), d.len(), o.as_mut_ptr(), 1, threads as i32);
            black_box(o);
        });
        let ours_gap = after_gaps(calls, &data, |d| {
            black_box(blake3_servil::hash_multithreaded(d));
        });
        say(format!(
            "  {len:>10} B   back to back: 0u.io/b3 {:>12}  blake3-servil {:>12}  x{}   after 1 ms asleep: 0u.io/b3 {:>12}  blake3-servil {:>12}  x{}",
            ns(theirs),
            ns(ours),
            ratio(theirs, ours),
            ns(theirs_gap),
            ns(ours_gap),
            ratio(theirs_gap, ours_gap)
        ));
    }

    say(String::new());
    say("extended output, one thread, from a 64-byte message (h.XOF().Read, finalize_xof().fill)".to_owned());
    let data = input(64);
    for outlen in [64 << 10, 1 << 20] {
        let mut theirs_out = vec![0u8; outlen];
        let mut ours_out = vec![0u8; outlen];
        unsafe { go_xof(data.as_ptr(), 64, theirs_out.as_mut_ptr(), outlen, 1) };
        let mut h = blake3_servil::Hasher::new();
        h.update(&data);
        h.finalize_xof().fill(&mut ours_out);
        assert!(theirs_out == ours_out, "0u.io/b3 XOF, {outlen} B");
        let per = ((16 << 20) / outlen).max(1);
        let (theirs, _) = back_to_back(per, || unsafe { go_xof(black_box(data.as_ptr()), 64, theirs_out.as_mut_ptr(), outlen, per) });
        let (ours, _) = back_to_back(per, || {
            for _ in 0..per {
                let mut h = blake3_servil::Hasher::new();
                h.update(black_box(&data));
                h.finalize_xof().fill(black_box(&mut ours_out));
            }
        });
        say(format!("  {outlen:>10} B out   0u.io/b3 {:>12}   blake3-servil {:>12}   x{}", ns(theirs), ns(ours), ratio(theirs, ours)));
    }
    std::fs::write("gob3-compare.txt", report).expect("write gob3-compare.txt");
    let _ = std::fs::remove_file(&lib);
}

fn ratio(theirs: u128, ours: u128) -> String {
    let r = (theirs * 1000 + ours / 2) / ours;
    format!("{}.{:03}", r / 1000, r % 1000)
}
