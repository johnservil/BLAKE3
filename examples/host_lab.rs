//! probe/summary-calibration: on the Mac (the runner's example job), the
//! regression check's two designs, calibrated: `bench-hashes regress` (the
//! fast speeds' ratio, 4 pairs and 4 to confirm) and `bench-hashes
//! regress-pairs` (each run's mean, 8 pairs, the median ratio and an exact
//! sign test; bench-hashes probe/summary-calibration), on four conditions:
//! one executable on both sides; the same code built with another layout
//! (bench-hashes' feature layout-perturb); and plants of +3% and +6%
//! (B3_PLANT, probe/plant-proportional: servil st's lent 64 B, 64 KiB,
//! 1 MiB). ALIGN builds every executable with LLVM's alignment of every
//! function and branch target, and the C kernels' functions.
//! Writes calibration.tsv and each check's output to the working directory.

use std::process::Command;

const ALIGN: bool = true;
const REPEATS: usize = 6;
const BENCH_BRANCH: &str = "probe/summary-calibration";

fn main() {
    let fork = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let out = std::env::current_dir().unwrap();
    // Inside the fork checkout: bench-hashes' build names the enclosing
    // checkout it is patched to (its build.rs).
    let bench = fork.join("calibration-bench");
    let _ = std::fs::remove_dir_all(&bench);
    run(Command::new("git").args(["clone", "-q", "--depth", "1", "--branch", BENCH_BRANCH, "https://github.com/johnservil/bench-hashes"]).arg(&bench));
    let tag = if ALIGN { "align" } else { "plain" };
    let exes: Vec<std::path::PathBuf> = [false, true]
        .iter()
        .map(|&perturb| {
            let target = home.join(format!("calibration-target-{tag}-{}", if perturb { "perturbed" } else { "as-built" }));
            let mut build = Command::new("cargo");
            build.current_dir(&bench).args(["build", "--release"]);
            build.arg("--config").arg(format!("patch.\"https://github.com/johnservil/BLAKE3\".blake3-servil.path=\"{}\"", fork.display()));
            build.arg("--config").arg(format!("patch.\"https://github.com/johnservil/BLAKE3\".clocks.path=\"{}\"", fork.join("clocks").display()));
            if perturb {
                build.args(["--features", "layout-perturb"]);
            }
            build.env("CARGO_TARGET_DIR", &target);
            if ALIGN {
                build.env("RUSTFLAGS", "-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=5");
                build.env("CFLAGS", "-falign-functions=64");
            }
            run(&mut build);
            let exe = out.join(format!("bench-{tag}-{}", if perturb { "perturbed" } else { "as-built" }));
            std::fs::copy(target.join("release/bench-hashes"), &exe).unwrap();
            exe
        })
        .collect();
    let (base, perturbed) = (&exes[0], &exes[1]);
    let plant = |p: u32| {
        let script = out.join(format!("plant-{p}.sh"));
        std::fs::write(&script, format!("#!/bin/sh\nB3_PLANT={p} exec '{}' \"$@\"\n", base.display())).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    };
    let conditions = [("same", base.clone()), ("layout", perturbed.clone()), ("plant3", plant(3)), ("plant6", plant(6))];
    let mut tsv = String::from("align\trepeat\tcondition\tdesign\texit\tseconds\toutput\n");
    for repeat in 0..REPEATS {
        for k in 0..conditions.len() {
            let (name, new) = &conditions[(k + repeat) % conditions.len()];
            for d in 0..2 {
                let design = if (d + repeat) % 2 == 0 { "regress" } else { "regress-pairs" };
                let mut check = Command::new(base);
                check.arg(design).arg(base).arg(new);
                if design == "regress-pairs" {
                    check.arg("8").arg(out.join(format!("ratios-{tag}-{repeat}-{name}.tsv")));
                }
                let started = std::time::Instant::now();
                let output = check.output().unwrap();
                let seconds = started.elapsed().as_secs();
                let text = String::from_utf8_lossy(&output.stdout).replace('\n', " / ");
                eprintln!("calibration: {tag} {repeat} {name} {design}: exit {:?}, {seconds} s: {text}", output.status.code());
                tsv += &format!("{tag}\t{repeat}\t{name}\t{design}\t{}\t{seconds}\t{text}\n", output.status.code().unwrap_or(-1));
                std::fs::write(out.join("calibration.tsv"), &tsv).unwrap();
            }
        }
    }
}

fn run(command: &mut Command) {
    eprintln!("host_lab: {command:?}");
    assert!(command.status().unwrap().success(), "{command:?} failed");
}
