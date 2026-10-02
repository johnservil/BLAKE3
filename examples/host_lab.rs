//! probe/queue-processes-mac: the queue's cells in many fresh processes on
//! the Mac (the runner's example job), at 1 ms and at 4 ms samples
//! (bench-hashes probe/queue-sample-length), each process with
//! --trace-clocks, so its samples' core kinds and clocks are known. Writes
//! each process's samples and trace under the working directory.

use std::process::Command;

const PROCESSES: usize = 20;

fn main() {
    let fork = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let out = std::env::current_dir().unwrap();
    let bench = fork.join("probe-bench");
    let _ = std::fs::remove_dir_all(&bench);
    assert!(Command::new("git").args(["clone", "-q", "--depth", "1", "--branch", "probe/queue-sample-length", "https://github.com/johnservil/bench-hashes"]).arg(&bench).status().unwrap().success());
    let target = home.join("probe-bench-target");
    assert!(Command::new("cargo").current_dir(&bench).args(["build", "--release", "--locked"]).env("CARGO_TARGET_DIR", &target).status().unwrap().success());
    let exe = target.join("release/bench-hashes");
    for i in 0..PROCESSES {
        for ms in [1, 4] {
            let dir = out.join(format!("ms{ms}-run{i:02}"));
            std::fs::create_dir_all(&dir).unwrap();
            let status = Command::new(&exe)
                .current_dir(&dir)
                .env("BENCH_QUEUE_SAMPLE_MS", ms.to_string())
                .args(["--contenders", "blake3-servil-mt,sha256", "--points", "continuous 64 B,continuous 1 KiB,continuous 16 KiB,continuous batch 16,continuous batch 4096", "--rounds", "24", "--trace-clocks", "trace.csv"])
                .stdout(std::process::Stdio::null())
                .status()
                .unwrap();
            assert!(status.success());
            eprintln!("host_lab: ms {ms} process {i} done");
        }
    }
}
