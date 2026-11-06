#!/bin/sh
# The builds the Mac's and the VM's suites leave out, checked in minutes
# on this machine (part of a candidate's gate, CONTRIBUTING.md): the library
# without std, tested; the library on one CPU without SME2, tested;
# warnings as errors for the library and b3sum;
# the C API, tested from Rust and from C;
# compile-only checks for x86-64 (default features: its C and assembly
# with the clang here); and the stop for every other target. Each failure stops the
# script.
#
#     sh tools/check-builds.sh
set -eu
cd "$(git rev-parse --show-toplevel)"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/tmp/target/check-builds}
rustup target add x86_64-unknown-linux-gnu 2>/dev/null
step() { echo "== $*"; "$@"; }
step cargo nextest run --cargo-quiet --release --no-default-features
# nextest skips doc tests: the examples without std too, where Batch is absent.
step cargo test -q --release --doc --no-default-features
# One CPU, as in a one-CPU container: the pool has no worker thread, and
# the caller hashes everything (the suites run on many CPUs and never
# reach that path).
step taskset -c 0 cargo nextest run --cargo-quiet --release --lib
RUSTFLAGS="-D warnings" step cargo check -q --release --tests --examples
RUSTFLAGS="-D warnings" step cargo check -q --release --manifest-path b3sum/Cargo.toml
# The C API (capi/): its Rust tests, and b3.h compiled as C11 and run.
RUSTFLAGS="-D warnings" step cargo nextest run --cargo-quiet --release --manifest-path capi/Cargo.toml
RUSTFLAGS="-D warnings" step cargo check -q --tests --examples --target x86_64-unknown-linux-gnu
RUSTFLAGS="-D warnings" step cargo check -q --tests --target x86_64-unknown-linux-gnu --no-default-features
# A build for any other target stops in build.rs, naming official BLAKE3
# (#160, #180): Windows, and WebAssembly for the rest. The build script
# runs even where the target's standard library is absent; the Linux builds
# above are the control.
for target in x86_64-pc-windows-msvc wasm32-unknown-unknown; do
	step sh -c "cargo check -q --keep-going --target $target 2>&1 | grep -q 'for $target, use official BLAKE3'"
done
# A C compiler that cannot assemble the x86-64 kernels stops the build,
# naming official BLAKE3; the builds above are the control.
step sh -c "CC=/nonexistent/cc cargo check -q 2>&1 | grep -q 'use official BLAKE3 for a build without assembly'"
# Every generated kernel file is its generator's output.
step sh -c 'python3 tools/gen_neon_hybrid.py | cmp - c/blake3_neon_hybrid_aarch64.S && python3 tools/gen_sme2_hybrid.py | cmp - c/blake3_sme2_hybrid_aarch64.S'
step sh -c 'python3 tools/gen_x86.py | cmp - c/blake3_partial_x86-64_unix.S'
step sh -c 'python3 tools/gen_x86_chunk_root.py | cmp - c/blake3_chunk_root_x86-64_unix.S'
step sh -c 'python3 tools/gen_avx512_32.py | cmp - c/blake3_avx512_32.S'
step sh -c 'python3 tools/gen_output16.py | cmp - c/blake3_output16.S'
echo "check-builds: every build passes"
