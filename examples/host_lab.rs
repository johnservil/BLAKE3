//! probe/b3sum-bench-mac: run tools/b3sum-bench on the Mac through the
//! runner's `example` job (host_lab). It builds the contenders from this
//! checkout (official b3sum 1.8.2; the fork's b3sum on Rayon, 779cd2d;
//! mapping every file on the pool, 0d99f9a; mapping cached files and
//! reading the rest, 71b087c) and the benchmark, then runs it with the files in
//! the runner's home and the report in the job's results folder (the
//! working directory).

use std::process::Command;

fn run(command: &mut Command) -> String {
    eprintln!("host_lab: {command:?}");
    let output = command.output().unwrap_or_else(|e| panic!("{command:?}: {e}"));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "{command:?} failed");
    String::from_utf8(output.stdout).unwrap()
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let out = std::env::current_dir().unwrap();
    let contenders = home.join("b3sum-contenders");
    let line = run(Command::new("sh").arg(root.join("tools/b3sum-bench/build-contenders.sh")).arg(&contenders).args(["779cd2d", "0d99f9a", "71b087c"]));
    let mut args: Vec<String> = Vec::new();
    // Rayon first: every other cell is compared with it.
    let words: Vec<&str> = line.split_whitespace().collect();
    let pick = |prefix: &str| words.iter().find(|w| w.starts_with(prefix)).unwrap().to_string();
    args.push(pick("fork-779cd2d=").replacen("fork-779cd2d=", "rayon=", 1));
    args.push(pick("fork-0d99f9a=").replacen("fork-0d99f9a=", "mmap-pool=", 1));
    args.push(pick("fork-71b087c=").replacen("fork-71b087c=", "candidate=", 1));
    args.push(pick("official="));
    let bench_target = home.join("b3sum-bench-target");
    run(Command::new("cargo").args(["build", "--release", "--manifest-path"]).arg(root.join("tools/b3sum-bench/Cargo.toml")).env("CARGO_TARGET_DIR", &bench_target));
    let status = Command::new(bench_target.join("release/b3sum-bench"))
        .arg("--files")
        .arg(home.join("b3sum-bench-files"))
        .arg("--out")
        .arg(&out)
        .args(&args)
        .status()
        .unwrap();
    assert!(status.success(), "b3sum-bench failed");
}
