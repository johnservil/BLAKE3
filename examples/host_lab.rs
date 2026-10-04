//! probe/b3sum-paths: `bench-hashes b3sum` on the Mac (the runner's example
//! job): official b3sum 1.8.2 against the fork's b3sum with each of its read
//! paths (docs/api-design.md, "The measurements that settle them", 1):
//! today's; today's never mapping; one thread; the queue; the stream; the
//! stream never mapping. The files in the runner's home, the record in the
//! job's results folder.

use std::process::Command;

fn run(command: &mut Command) -> String {
    eprintln!("host_lab: {command:?}");
    let output = command.output().unwrap_or_else(|e| panic!("{command:?}: {e}"));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "{command:?} failed");
    String::from_utf8(output.stdout).unwrap()
}

fn main() {
    let fork = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let bench = fork.join("record-bench");
    let _ = std::fs::remove_dir_all(&bench);
    run(Command::new("git").args(["clone", "-q", "--depth", "1", "--branch", "candidate/no-linger", "https://github.com/johnservil/bench-hashes"]).arg(&bench));
    let line = run(Command::new("sh").arg(bench.join("tools/b3sum-contenders.sh")).arg(fork).arg(home.join("b3sum-contenders")).args(["7b74d38", "93bcb15", "2b3bb8a", "8dc3ac5", "8b7875b", "d0f7b07"]));
    let pick = |prefix: &str| line.split_whitespace().find(|w| w.starts_with(prefix)).unwrap().split_once('=').unwrap().1.to_owned();
    let target = home.join("record-bench-target");
    run(Command::new("cargo").current_dir(&bench).args(["build", "--release", "--locked"]).env("CARGO_TARGET_DIR", &target));
    let status = Command::new(target.join("release/bench-hashes"))
        .arg("b3sum")
        .arg("--files")
        .arg(home.join("b3sum-bench-files"))
        .arg(format!("official={}", pick("official=")))
        .arg(format!("today={}", pick("fork-7b74d38=")))
        .arg(format!("today-nomap={}", pick("fork-93bcb15=")))
        .arg(format!("one-thread={}", pick("fork-2b3bb8a=")))
        .arg(format!("queue={}", pick("fork-8dc3ac5=")))
        .arg(format!("stream={}", pick("fork-8b7875b=")))
        .arg(format!("stream-nomap={}", pick("fork-d0f7b07=")))
        .status()
        .unwrap();
    assert!(status.success(), "bench-hashes b3sum failed");
}
