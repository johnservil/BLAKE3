#!/bin/sh
# Predicted cycles per pass of each LABEL's code in OBJECT (an AArch64
# object file), by LLVM 19's llvm-mca models of eBASH's Arm cores: a design
# guide, never a measurement (no machine here has these cores; llvm-mca
# models no caches). A loop's label gives one pass of its body (extract.py).
# The A72 runs the Cortex-A57 model, the closest LLVM 19 has; the A76 has
# none and the Neoverse N1's stands for it.
#
#   sh tools/mca/model.sh OBJECT LABEL...
#
# The rolled scalar kernel, a round of which runs seven times a compression:
#   clang-16 --target=aarch64-linux-gnu -c c/blake3_neon_hybrid_aarch64.S -o $TMPDIR/k.o
#   sh tools/mca/model.sh $TMPDIR/k.o blake3_hybrid_k1_round        # A53: 107 (x7 = 749)
# The portable C compression, for each core's compiler tuning:
#   gcc -O3 -mcpu=cortex-a53 -c c/blake3_portable.c -o $TMPDIR/p.o
#   sh tools/mca/model.sh $TMPDIR/p.o blake3_compress_in_place_portable
set -e
obj=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
llvm=${LLVM:-/usr/lib/llvm-19/bin}
out=$(mktemp -d)
"$llvm/llvm-objdump" -d --no-show-raw-insn "$obj" > "$out/dis"
${PYTHON:-pypy3} "$here/extract.py" "$out/dis" "$out" "$@" >/dev/null
printf '%-40s %12s %12s %12s\n' label cortex-a53 a72:a57 a76:n1
for label in "$@"; do
  printf '%-40s' "$label"
  for cpu in cortex-a53 cortex-a57 neoverse-n1; do
    "$llvm/llvm-mca" -mtriple=aarch64 -mcpu=$cpu -iterations=40 "$out/$label.s" 2>/dev/null |
      awk '/^Iterations:/{i=$2} /^Total Cycles:/{c=$3} END{printf " %12.1f", c/i}'
  done
  echo
done
rm -r "$out"
