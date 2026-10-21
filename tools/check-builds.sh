#!/bin/sh
# The builds the Mac's and the VM's suites leave out, checked in minutes
# on this machine (part of a candidate's gate, CONTRIBUTING.md): the library
# without std, tested; the library on one CPU without SME2, tested;
# warnings as errors for the default and pure builds and b3sum;
# the C API, tested from Rust and from C;
# and compile-only checks for x86-64 (default features: its C and assembly
# with the clang here) and x86-64. Each failure stops the script.
#
#     sh tools/check-builds.sh
set -eu
cd "$(git rev-parse --show-toplevel)"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/tmp/target/check-builds}
rustup target add x86_64-unknown-linux-gnu 2>/dev/null
step() { echo "== $*"; "$@"; }
step cargo nextest run --cargo-quiet --release --no-default-features
# One CPU, as in a one-CPU container: the pool has no worker thread, and
# the caller hashes everything (the suites run on many CPUs and never
# reach that path).
step taskset -c 0 cargo nextest run --cargo-quiet --release --lib
for features in "" "--features pure"; do
	# shellcheck disable=SC2086
	RUSTFLAGS="-D warnings" step cargo check -q --release --tests --examples $features
done
RUSTFLAGS="-D warnings" step cargo check -q --release --manifest-path b3sum/Cargo.toml
# The C API (capi/): its Rust tests, and b3.h compiled as C11 and run.
RUSTFLAGS="-D warnings" step cargo nextest run --cargo-quiet --release --manifest-path capi/Cargo.toml
RUSTFLAGS="-D warnings" step cargo check -q --tests --examples --target x86_64-unknown-linux-gnu
RUSTFLAGS="-D warnings" step cargo check -q --tests --target x86_64-unknown-linux-gnu --no-default-features
RUSTFLAGS="-D warnings" step cargo check -q --tests --target x86_64-unknown-linux-gnu --features pure
# The generated x86-64 kernels are their generator's output (tools/gen_x86.py).
step sh -c 'pypy3 tools/gen_x86.py unix | cmp - c/blake3_partial_x86-64_unix.S && pypy3 tools/gen_x86.py msvc | cmp - c/blake3_partial_x86-64_windows_msvc.asm'
echo "check-builds: every build passes"
