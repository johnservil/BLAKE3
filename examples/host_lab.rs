//! Probe (branch probe/clean-build-against): removes target/build-against
//! from the checkout this was built in (CARGO_MANIFEST_DIR), where the
//! separated bench (before bench 40dc1872) and 0.18.3 shared one build
//! unit; 0.18.3 built after the separated bench ran the wrong build script
//! (job 1781). From 40dc1872 the separated bench builds elsewhere.
fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("build-against");
    println!("# removing {}: present {}", dir.display(), dir.exists());
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("removing target/build-against");
    }
    println!("# present after: {}", dir.exists());
}
