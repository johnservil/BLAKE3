//! Diagnostic-only: test a pinned benchmark on the quiet native Mac.
//! The existing runner's example job owns the process tree and deadline.
use std::path::PathBuf;
use std::process::Command;

fn run(bench: &std::path::Path, args: &[&str]) {
    let (program, args) = args.split_first().expect("a program");
    let status = Command::new(program).args(args).current_dir(bench).status().expect("command starts");
    assert!(status.success(), "{program} {args:?}: {status}");
}

fn main() {
    assert!(cfg!(target_os = "macos"), "this diagnostic is for the native Mac runner");
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2, "host_lab BENCH_COMMIT");
    let rev = &args[1];
    assert!(rev.len() == 40 && rev.bytes().all(|c| c.is_ascii_hexdigit()), "a full commit hash");
    let bench = PathBuf::from(std::env::var_os("HOME").expect("runner HOME")).join("checkouts/bench");
    assert!(bench.join(".git").exists(), "the runner's benchmark checkout");
    run(&bench, &["git", "fetch", "--quiet", "origin", rev]);
    run(&bench, &["git", "checkout", "--force", "--detach", rev]);
    run(&bench, &["cargo", "nextest", "run", "--release", "--no-run"]);
    clocks::load::quiet_within(120).expect("ten quiet seconds before the correctness suite");
    run(&bench, &["cargo", "nextest", "run", "--release"]);
    clocks::load::quiet_within(120).expect("ten quiet seconds before the machine suite");
    run(&bench, &["cargo", "nextest", "run", "--release", "-P", "machine"]);
    println!("benchmark correctness and complete machine suite passed at {rev}");
}
