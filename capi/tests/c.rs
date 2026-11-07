//! The installed C API: C11 and C++11 consumers use cargo-c's header,
//! libraries and pkg-config flags. Check both shared and static linking.

#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

/// Install just one library kind, so the linker must use that kind.
fn installed_consumer(kind: &str) -> String {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("c-libraries");
    let stage = target.join(kind);
    // A fresh install catches omitted artifacts instead of reusing an old one.
    if stage.exists() {
        std::fs::remove_dir_all(&stage).unwrap();
    }
    let installed = stage.join("b3-capi-test");
    let lib_dir = installed.join("lib");
    let mut build = Command::new(env!("CARGO"));
    build.args(["cinstall", "--manifest-path"]).arg(here.join("Cargo.toml"));
    build.arg("--target-dir").arg(target.join("build"));
    build.arg("--destdir").arg(&stage);
    build.args(["--prefix=/b3-capi-test", "--libdir=lib", "--library-type", kind]);
    if cfg!(debug_assertions) {
        build.arg("--debug");
    }
    assert!(build.status().expect("cargo-c installed on PATH").success(), "cargo cinstall {kind}");
    assert_eq!(std::fs::read(installed.join("include/b3.h")).unwrap(), std::fs::read(here.join("b3.h")).unwrap());

    let pkg_config = |args: &[&str]| {
        let out = Command::new("pkg-config")
            .env("PKG_CONFIG_PATH", "")
            .env("PKG_CONFIG_LIBDIR", lib_dir.join("pkgconfig"))
            .args(args)
            .arg("b3_servil")
            .output()
            .expect("pkg-config installed on PATH");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };
    assert_eq!(pkg_config(&["--modversion"]).trim(), env!("CARGO_PKG_VERSION"));
    assert_eq!(pkg_config(&["--variable=prefix"]).trim(), "/b3-capi-test");
    // Relocate the prefix for a consumer of this staged native installation.
    let flags = if kind == "staticlib" {
        assert!(!lib_dir.join("libb3_servil.so").exists() && !lib_dir.join("libb3_servil.dylib").exists());
        pkg_config(&["--define-prefix", "--static", "--cflags", "--libs"])
    } else {
        assert!(!lib_dir.join("libb3_servil.a").exists());
        pkg_config(&["--define-prefix", "--cflags", "--libs"])
    };
    // Compiler response files preserve pkg-config's quoted paths, including
    // spaces, without introducing a shell or a second argument parser.
    let response = stage.join("link-flags");
    std::fs::write(&response, flags).unwrap();
    let mut output = String::new();
    for (compiler, standard, source) in [("cc", "c11", "check.c"), ("c++", "c++11", "check.cpp")] {
        let program = stage.join(format!("b3-{source}"));
        let mut compile = Command::new(compiler);
        compile
            .arg(format!("-std={standard}"))
            .args(["-Wall", "-Wextra", "-Werror", "-pedantic"])
            .arg(here.join("tests").join(source))
            .arg(format!("@{}", response.display()));
        if kind == "cdylib" {
            compile.args(["-Xlinker", "-rpath", "-Xlinker"]).arg(&lib_dir);
        }
        let status = compile.arg("-o").arg(&program).status().expect("a C and C++ compiler");
        assert!(status.success(), "{source} compiles and links against {kind}");
        let mut run = Command::new(&program);
        if kind == "cdylib" {
            // The staged dylib's install name is the deployment prefix;
            // macOS falls back to this path by leaf name. Linux ignores it.
            run.env("DYLD_LIBRARY_PATH", &lib_dir);
        }
        let out = run.output().unwrap();
        assert!(out.status.success(), "{source}: {}", String::from_utf8_lossy(&out.stderr));
        if source == "check.c" {
            output = String::from_utf8(out.stdout).unwrap();
        }
    }
    output
}

#[test]
fn from_c() {
    for kind in ["cdylib", "staticlib"] {
        let out = installed_consumer(kind);
        blake3_servil::initialize_multithreaded(None);
        let input: Vec<u8> = (0..100000).map(|i| (i % 251) as u8).collect();
        let key = *b"whats the Elvish word for friend";
        let context = "BLAKE3 2019-12-27 16:29:52 test vectors context";
        let modes = [blake3_servil::Mode::Plain, blake3_servil::Mode::Keyed(&key), blake3_servil::Mode::DeriveKey(context)];
        let mut expected = Vec::new();
        for mode in modes {
            for len in [0, 1025, 100000] {
                let d = blake3_servil::convenience::hash_with(mode, &input[..len]).to_hex();
                // b3_hash, then the hasher fed in two pieces.
                for _ in 0..2 {
                    expected.push(d.to_string());
                }
            }
        }
        // b3.h's b3_hasher holds the crate's Hasher.
        let first = out.lines().next().unwrap();
        assert_eq!(first, format!("hasher {} 64", size_of::<blake3_servil::convenience::Hasher>()), "b3.h's B3_HASHER_SIZE");
        assert!(align_of::<blake3_servil::convenience::Hasher>() <= 64);
        let got: Vec<&str> = out.lines().skip(1).map(|l| l.split(' ').nth(1).unwrap()).collect();
        assert_eq!(got.len(), expected.len(), "{out}");
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "line {i}: {}", out.lines().nth(i).unwrap());
        }
    }
}
