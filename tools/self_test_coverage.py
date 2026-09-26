"""Which assembly entries the startup self-test runs: a gdb script.

A program whose only call is one hash() runs exactly the self-test; gdb
counts a breakpoint on every AArch64 assembly entry (the integer + NEON
hybrids' k, q, p, and c kernels, and the SME2 kernels) and lists each
with its hit count, or NOT REACHED. From the fork's checkout, in the VM:

    mkdir -p /tmp/cov/src && printf '[package]\\nname = "cov"\\nversion = "0.1.0"\\nedition = "2021"\\n[dependencies]\\nblake3-servil = { path = "%s" }\\n' "$PWD" > /tmp/cov/Cargo.toml
    printf 'fn main() { std::hint::black_box(blake3_servil::hash(b"")); }\\n' > /tmp/cov/src/main.rs
    (cd /tmp/cov && cargo build --release)
    nm /tmp/cov/target/release/cov | grep -E ' [Tt] (blake3_hybrid_[kpqc][0-9]+|blake3_sme2[a-z0-9_]*_512)$' | awk '{print $3}' | sort -u > /tmp/syms.txt
    SHELL=/bin/sh gdb -q -batch -ex 'set startup-with-shell off' -x tools/self_test_coverage.py /tmp/cov/target/release/cov

(with the usual HOME, CARGO_TARGET_DIR, CC, and TMPDIR prefix for cargo;
CARGO_TARGET_DIR changes where the binary is). September 26, 2026: all 31
entries reached on SME2 (the VM).
"""
import collections

import gdb

symbols = [line.strip() for line in open("/tmp/syms.txt") if line.strip()]
hits = collections.Counter()


class Count(gdb.Breakpoint):
    def stop(self):
        hits[self.location] += 1
        return False


for symbol in symbols:
    Count(symbol)
gdb.execute("run")
for symbol in symbols:
    print(f"{symbol}: {hits[symbol] or 'NOT REACHED'}")
missing = [s for s in symbols if not hits[s]]
print(f"{len(symbols) - len(missing)} of {len(symbols)} assembly entries reached")
