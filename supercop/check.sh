#!/bin/sh
# Usage: sh supercop/check.sh [SUPERCOP_VERSION]
# Runs this directory's rust_servil through a fresh SUPERCOP tree as eBASH
# does (./do-part init, then ./do-part crypto_hash blake3) and reports its
# checksums and SUPERCOP's choice among crypto_hash_blake3's
# implementations. Run it after make-tarball.sh, before sending a tarball.
# It downloads SUPERCOP (about 50 MB), compiles it with every compiler it
# finds (a minute with gcc alone, longer per compiler); its cycles count
# only from an idle machine.
set -e
v=${1:-20261004}
here=$(cd "$(dirname "$0")" && pwd)
w=${TMPDIR:-/tmp}/supercop-check
rm -rf "$w"; mkdir -p "$w"; cd "$w"
curl -sSfL "https://bench.cr.yp.to/supercop/supercop-$v.tar.xz" | tar -xJ
cd "supercop-$v"
cp -r "$here/crypto_hash/blake3/rust_servil" crypto_hash/blake3/
unset CARGO_TARGET_DIR     # SUPERCOP's Rust build chooses its own
# do-part's exit status says nothing about blake3 (init exits 1 when it has
# no errors to show); the data says it all.
./do-part init > ../init.log 2>&1 || true
./do-part crypto_hash blake3 > ../blake3.log 2>&1 || true
data=$(ls bench/*/data)
# A try line: ... try CHECKSUMS ok|fails CYCLES_AT_1536 ... IMPLEMENTATION COMPILER
awk '$7 == "try" { sums[$8]++; if ($9 != "ok") bad++; if ($13 ~ /rust_servil/) { ours++; printf "rust_servil %s, %s cycles at 1536 bytes (%s)\n", $9, $10, $14 } }
     $7 == "implementation" { chosen = $8 }
     END {
       n = 0; for (s in sums) n++
       printf "checksums: %d failing, %d distinct value(s) among all builds\n", bad, n
       printf "SUPERCOP chose: %s\n", chosen
       if (!ours || bad || n != 1) { print "FAIL: rust_servil missing, a checksum failing, or the implementations disagreeing"; exit 1 }
     }' "$data"
echo "logs: $w/init.log, $w/blake3.log; data: $w/supercop-$v/$data"
