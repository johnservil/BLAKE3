//! probe/b3sum-record-0153: `bench-hashes b3sum` on the Mac (the runner's
//! example job) for bench-hashes 0.15.3's record: official b3sum 1.8.2
//! against the fork's b3sum at servil 6d9cac3, from bench-hashes 482ef3b; the
//! files in the runner's home, the record in the job's results folder.

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
    run(Command::new("git").args(["clone", "-q", "https://github.com/johnservil/bench-hashes"]).arg(&bench));
    run(Command::new("git").current_dir(&bench).args(["checkout", "-q", "482ef3b0e00316ddcd3afb99f3e6d7b4bb6d67a6"]));
    let line = run(Command::new("sh").arg(bench.join("tools/b3sum-contenders.sh")).arg(fork).arg(home.join("b3sum-contenders")).arg("6d9cac3"));
    let pick = |prefix: &str| line.split_whitespace().find(|w| w.starts_with(prefix)).unwrap().split_once('=').unwrap().1.to_owned();
    let target = home.join("record-bench-target");
    run(Command::new("cargo").current_dir(&bench).args(["build", "--release", "--locked"]).env("CARGO_TARGET_DIR", &target));
    let status = Command::new(target.join("release/bench-hashes"))
        .arg("b3sum")
        .arg("--files")
        .arg(home.join("b3sum-bench-files"))
        .arg(format!("official={}", pick("official=")))
        .arg(format!("servil={}", pick("fork-")))
        .status()
        .unwrap();
    assert!(status.success(), "bench-hashes b3sum failed");
}
