//! probe/mac-debug-tests: the fork's suites in debug builds (overflow
//! checks, debug assertions) on the Mac, which the runner's test job runs
//! in release only. Runs cargo in this checkout; writes every result line.
use std::io::Write;
use std::process::Command;

fn main() {
    let fork = env!("CARGO_MANIFEST_DIR");
    let mut report = String::new();
    for args in [
        vec!["test", "--lib"],
        vec!["test", "--lib", "--features", "no_sme2"],
        vec!["test", "--lib", "--features", "pure"],
        vec!["test", "--test", "*"],
        vec!["test", "--doc"],
        vec!["test", "--manifest-path", "test_vectors/Cargo.toml"],
    ] {
        let output = Command::new("cargo").args(&args).current_dir(fork).env("CARGO_TARGET_DIR", format!("{fork}/target/debug-tests")).output().expect("cargo runs");
        let text = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
        report += &format!("== cargo {}: {}\n", args.join(" "), output.status);
        for line in text.lines().filter(|l| l.starts_with("test result") || l.contains("FAILED") || l.contains("panicked") || l.starts_with("error") || l.contains("warm queue") || l.contains("left:") || l.contains("right:")) {
            report += line;
            report.push('\n');
        }
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
