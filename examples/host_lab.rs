//! probe/gate-busy: which of the gate's runs finds other programs busy.
//! Runs the gate's own new binary (left by the last perf_regress job) 20
//! times with regress's arguments, each in a fresh directory, and prints
//! each run's load line from its samples file, with the run's wall time.

fn main() {
    let exe = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench-hashes/target/perf-new/release/bench-hashes");
    assert!(exe.exists(), "{} missing", exe.display());
    for run in 0..20 {
        let dir = std::env::temp_dir().join(format!("gate-busy-{}-{run}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let began = clocks::now();
        let status = std::process::Command::new(&exe)
            .args(["--contenders", "blake3-servil-st,blake3-servil-mt", "--points", "lent 64 B,lent 64 KiB,lent 1 MiB,lent batch 16,lent batch 4096", "--rounds", "24"])
            .current_dir(&dir).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap();
        let ms = clocks::since_ns(began) / 1_000_000;
        assert!(status.success());
        let machine = std::fs::read_dir(dir.join("benchmark-results")).unwrap().next().unwrap().unwrap().path();
        let samples = std::fs::read_to_string(machine.join("bench-hashes.samples.tsv")).unwrap();
        let load = samples.lines().find(|l| l.contains("load")).unwrap_or("no load line").to_owned();
        println!("run {run:2}  {ms:5} ms  {load}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
