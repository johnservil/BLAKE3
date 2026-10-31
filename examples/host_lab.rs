//! Probe (branch probe/fork-status): what makes the runner's fork checkout
//! read as dirty in the benchmark's provenance. Prints `git status` of the
//! checkout this was built in (CARGO_MANIFEST_DIR), the diff's stat, and
//! each untracked file outside nested checkouts with its size.
fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git").arg("-C").arg(dir).args(args).output().expect("git");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    println!("# checkout: {dir}");
    println!("# HEAD: {}", git(&["rev-parse", "HEAD"]).trim());
    println!("# status (porcelain, untracked files listed):\n{}", git(&["status", "--porcelain", "--untracked-files=all"]));
    println!("# diff --stat:\n{}", git(&["diff", "--stat"]));
    for path in git(&["ls-files", "--others", "--exclude-standard"]).lines() {
        let full = std::path::Path::new(dir).join(path);
        let len = std::fs::metadata(&full).map(|m| m.len()).unwrap_or(0);
        println!("untracked {path} ({len} bytes)");
    }
}
