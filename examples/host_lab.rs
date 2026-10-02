//! probe/b3sum-read-overlap: b3sum-bench on the Mac with this commit's
//! b3sum, mapping against reading on a thread of its own (B3SUM_PROBE_OVERLAP).

use std::process::Command;

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let contenders = home.join("b3sum-contenders");
    let out = Command::new("sh").arg(root.join("tools/b3sum-bench/build-contenders.sh")).arg(&contenders).arg("1a127f3").output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let b = contenders.join("b3sum-1a127f3");
    let b = b.to_str().unwrap();
    let target = home.join("b3sum-bench-target");
    assert!(Command::new("cargo").args(["build", "--release", "--manifest-path"]).arg(root.join("tools/b3sum-bench/Cargo.toml")).env("CARGO_TARGET_DIR", &target).status().unwrap().success());
    let status = Command::new(target.join("release/b3sum-bench"))
        .arg("--files").arg(home.join("b3sum-bench-files"))
        .arg("--out").arg(std::env::current_dir().unwrap())
        .arg(format!("mmap=env {b}"))
        .arg(format!("ov4m=env B3SUM_PROBE_OVERLAP=4194304 {b}"))
        .arg(format!("ov16m=env B3SUM_PROBE_OVERLAP=16777216 {b}"))
        .arg(format!("read=env {b} --no-mmap"))
        .status().unwrap();
    assert!(status.success());
}
