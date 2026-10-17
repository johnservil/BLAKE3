//! The C API from C (tests/check.c): built with this machine's C compiler
//! (`cc`, C11, every warning an error) against the shared library, run,
//! and its digests checked against the Rust crate's; and b3.h from C++
//! (tests/check.cpp, C++11).

#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn from_c() {
    // The libraries, built here (`cargo test` builds only the Rust library
    // the tests link) by the same cargo and profile, in a target directory
    // of the test's own (Cargo's place for an integration test's files;
    // its own layout of target/ changes between versions).
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("c-libraries");
    let release = !cfg!(debug_assertions);
    let mut build = Command::new(env!("CARGO"));
    build.args(["build", "--lib", "--manifest-path"]).arg(here.join("Cargo.toml"));
    build.arg("--target-dir").arg(&target);
    if release {
        build.arg("--release");
    }
    assert!(build.status().unwrap().success(), "the C libraries build");
    let lib_dir = target.join(if release { "release" } else { "debug" });
    let program = lib_dir.join("b3-check");
    let status = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I"])
        .arg(&here)
        .arg(here.join("tests/check.c"))
        .arg("-L")
        .arg(&lib_dir)
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .args(["-lb3_servil", "-o"])
        .arg(&program)
        .status()
        .expect("a C compiler, cc");
    assert!(status.success(), "tests/check.c compiles and links");
    // And from C++ (C++11).
    let cpp = lib_dir.join("b3-check-cpp");
    let status = Command::new("c++")
        .args(["-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I"])
        .arg(&here)
        .arg(here.join("tests/check.cpp"))
        .arg("-L")
        .arg(&lib_dir)
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .args(["-lb3_servil", "-o"])
        .arg(&cpp)
        .status()
        .expect("a C++ compiler, c++");
    assert!(status.success(), "tests/check.cpp compiles and links");
    assert!(Command::new(&cpp).status().unwrap().success(), "tests/check.cpp runs");
    let out = Command::new(&program).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = String::from_utf8(out.stdout).unwrap();

    blake3_servil::initialize_multithreaded(None);
    let input: Vec<u8> = (0..100000).map(|i| (i % 251) as u8).collect();
    let key = *b"whats the Elvish word for friend";
    let context = "BLAKE3 2019-12-27 16:29:52 test vectors context";
    let modes = [blake3_servil::Mode::Hash, blake3_servil::Mode::Keyed(&key), blake3_servil::Mode::DeriveKey(context)];
    let mut expected = Vec::new();
    for mode in modes {
        for len in [0, 1025, 100000] {
            let d = blake3_servil::hash_with(mode, &input[..len]).to_hex();
            for _ in 0..3 {
                expected.push(d.to_string());
            }
        }
        for i in 0..4 {
            expected.push(blake3_servil::hash_with(mode, &input[i * 64..(i + 1) * 64]).to_hex().to_string());
        }
    }
    // b3.h's b3_hasher holds the crate's Hasher.
    let first = out.lines().next().unwrap();
    assert_eq!(first, format!("hasher {} 64", size_of::<blake3_servil::Hasher>()), "b3.h's B3_HASHER_SIZE");
    assert!(align_of::<blake3_servil::Hasher>() <= 64);
    let got: Vec<&str> = out.lines().skip(1).map(|l| l.split(' ').nth(1).unwrap()).collect();
    assert_eq!(got.len(), expected.len(), "{out}");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g, e, "line {i}: {}", out.lines().nth(i).unwrap());
    }
}
