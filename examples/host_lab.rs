//! probe/b3sum-mac-tests: b3sum's own tests on the Mac (the runner's
//! example job), where its residency test (mincore) is macOS's.

use std::process::Command;

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let status = Command::new("cargo")
        .args(["test", "--release", "--manifest-path"])
        .arg(root.join("b3sum/Cargo.toml"))
        .env("CARGO_TARGET_DIR", home.join("b3sum-test-target"))
        .status()
        .unwrap();
    assert!(status.success(), "b3sum's tests failed");
}
