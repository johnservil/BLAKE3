//! probe/self-test-time (never merged): the startup self-test's cost, cold,
//! in fresh processes: the time of a process's first hash() call, with
//! the self-test (the default) and without (NO_SELF_TEST, this probe's
//! build only), for a 1-byte and a 64 KiB input, 15 processes each; then
//! the warm self-test's time (it runs again, 200 times).
use std::time::Instant;

fn main() {
    if let Some(len) = std::env::args().nth(1) {
        let input = vec![3u8; len.parse().unwrap()];
        let t = Instant::now();
        std::hint::black_box(blake3_servil::hash(&input));
        println!("{}", t.elapsed().as_nanos());
        return;
    }
    let me = std::env::current_exe().unwrap();
    for (label, skip) in [("with self-test", false), ("without self-test", true)] {
        for len in [1, 65536] {
            let mut ns: Vec<u64> = (0..15)
                .map(|_| {
                    let mut command = std::process::Command::new(&me);
                    command.arg(len.to_string());
                    if skip {
                        command.env("NO_SELF_TEST", "1");
                    }
                    String::from_utf8(command.output().unwrap().stdout).unwrap().trim().parse().unwrap()
                })
                .collect();
            ns.sort();
            println!("{label}, first hash of {len} B: median {} us, min {} us, max {} us", ns[7] / 1000, ns[0] / 1000, ns[14] / 1000);
        }
    }
    blake3_servil::hash(b"x");
    let mut warm: Vec<u128> = (0..200).map(|_| { let t = Instant::now(); blake3_servil::__self_test_run_again(); t.elapsed().as_nanos() }).collect();
    warm.sort();
    println!("warm self-test: median {} us, min {} us", warm[100] / 1000, warm[0] / 1000);
}
