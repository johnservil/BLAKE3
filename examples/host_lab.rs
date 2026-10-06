//! probe/mac-debug-tests, round two: the suites under AddressSanitizer
//! and ThreadSanitizer on the Mac (nightly rustc, -Zsanitizer, no
//! build-std), the pool's and the queue's unsafe code on Apple's cores.
//! Round three: TSan alone, std uninstrumented (its reports inside std's
//! own synchronization may be false).
use std::io::Write;
use std::process::Command;

fn main() {
    let fork = env!("CARGO_MANIFEST_DIR");
    let mut report = String::new();
    let target = "aarch64-apple-darwin";
    for (sanitizer, args) in [
        ("thread", vec!["test", "--lib", "--target", target, "--", "lanes", "unsafe_paths", "many", "queue", "each"]),
        ("thread", vec!["test", "--test", "api_plan", "--target", target]),
    ] {
        let output = Command::new("cargo")
            .args(&args)
            .current_dir(fork)
            .env("CARGO_TARGET_DIR", format!("{fork}/target/san-{sanitizer}"))
            .env("RUSTFLAGS", format!("-Zsanitizer={sanitizer} -Cunsafe-allow-abi-mismatch=sanitizer"))
            .env("RUSTDOCFLAGS", format!("-Zsanitizer={sanitizer} -Cunsafe-allow-abi-mismatch=sanitizer"))
            .env("TSAN_OPTIONS", "halt_on_error=0")
            .output()
            .expect("cargo runs");
        report += &format!("== {sanitizer}: ");
        let text = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
        report += &format!("== cargo {}: {}\n", args.join(" "), output.status);
        for line in text.lines().filter(|l| l.starts_with("test result") || l.contains("FAILED") || l.contains("panicked") || l.starts_with("error") || l.contains("warm queue") || l.contains("left:") || l.contains("right:") || l.contains("WARNING: ThreadSanitizer") || l.contains("ERROR: AddressSanitizer") || l.contains("SUMMARY:")) {
            report += line;
            report.push('\n');
        }
    }
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
