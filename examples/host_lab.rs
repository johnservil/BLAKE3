//! probe/libra-bench: Libra's own benchmark on the Mac (the runner's example
//! job) over Libra as it is, with git-internal on this fork's kernels, and
//! with BLAKE3 IDs hashed in place (bench-hashes apps/libra-bench, 62241af87906ef1fa02711dc10808cf87e8f556b).

use std::process::Command;

fn main() {
    let fork = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let bench = home.join("libra-bench-kit");
    let _ = std::fs::remove_dir_all(&bench);
    let ok = |c: &mut Command| assert!(c.status().unwrap().success(), "{c:?} failed");
    ok(Command::new("git").args(["clone", "-q", "https://github.com/johnservil/bench-hashes"]).arg(&bench));
    ok(Command::new("git").current_dir(&bench).args(["checkout", "-q", "62241af87906ef1fa02711dc10808cf87e8f556b"]));
    let out = std::env::current_dir().unwrap().join("libra-bench-results");
    ok(Command::new("sh").arg(bench.join("apps/libra-bench/run.sh")).arg(fork).arg(home.join("libra-bench-work")).arg(&out).arg("5"));
}
