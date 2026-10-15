#!/bin/sh
# Instructions per hash() call, the fork's and upstream BLAKE3 1.8.7's, under
# qemu's instruction counter: exact, and the same whatever else the machine
# runs, so it compares two builds while the host is busy (a count is no
# timing: speed verdicts stay perf_regress's).
#
#   sh tools/insn_count/count.sh x86_64|aarch64 CPU LEN...
#
# CPU is a qemu -cpu model: name the machine's, since it picks the kernels.
# eBASH's Arm cores: cortex-a53, cortex-a72, cortex-a76, neoverse-n1 (none
# has SVE or SME); x86-64: max (AVX2; qemu 7.2 has no AVX-512) or, say,
# Haswell. `max` on aarch64 has SME2, which no eBASH machine has.
#
# Each count is (instructions at REPS + 1 calls - at 1 call) / REPS: the
# baseline makes one call, so both sides include the fork's one-time
# initialize(). Needs qemu-user, and for x86_64 the cross toolchain
# (gcc-x86-64-linux-gnu, libc6-dev-amd64-cross) and its Rust target; the
# plugin is built once from qemu's own source (libglib2.0-dev).
set -e
arch=$1 cpu=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
plugins=${TMPDIR:-/tmp}/qemu-plugins
if [ ! -f "$plugins/libinsn.so" ]; then
  mkdir -p "$plugins"
  v=$(qemu-x86_64 --version | sed -n 's/.*version \([0-9]*\.[0-9]*\).*/\1/p')
  for f in include/qemu/qemu-plugin.h tests/plugin/insn.c contrib/plugins/execlog.c; do
    curl -sSf -o "$plugins/$(basename $f)" "https://gitlab.com/qemu-project/qemu/-/raw/v$v.0/$f"
  done
  for p in insn execlog; do
    cc -shared -fPIC -O2 -I"$plugins" $(pkg-config --cflags glib-2.0) "$plugins/$p.c" -o "$plugins/lib$p.so" $(pkg-config --libs glib-2.0)
  done
fi
case $arch in
  x86_64)
    target=x86_64-unknown-linux-gnu
    export CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc
    run() { qemu-x86_64 -cpu $cpu -L /usr/x86_64-linux-gnu -plugin "$plugins/libinsn.so" -d plugin -D "$plugins/q.log" "$@"; } ;;
  aarch64)
    target=aarch64-unknown-linux-gnu
    run() { qemu-aarch64 -cpu $cpu -plugin "$plugins/libinsn.so" -d plugin -D "$plugins/q.log" "$@"; } ;;
  *) echo "usage: $0 x86_64|aarch64 CPU LEN..." >&2; exit 2 ;;
esac
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-${TMPDIR:-/tmp}/insn-count}
(cd "$here" && cargo build --release -q --target $target)
bin=$CARGO_TARGET_DIR/$target/release
count() { run "$bin/$1" "$2" "$3"; awk '/insns/{v=$NF} END{print v}' "$plugins/q.log"; }
printf '%8s %12s %12s\n' bytes upstream servil
for len in "$@"; do
  r=200; [ "$len" -gt 8192 ] && r=40
  u=$(( ($(count upstream $len $((r + 1))) - $(count upstream $len 1)) / r ))
  s=$(( ($(count servil $len $((r + 1))) - $(count servil $len 1)) / r ))
  printf '%8d %12d %12d\n' $len $u $s
done
