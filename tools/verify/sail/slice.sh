#!/bin/sh
# Arm's Armv9.4 Sail model (github.com/rems-project/sail-arm), sliced to the
# decode functions of the instructions named in ROOTS, translated to Lean.
# Needs Sail built with sail-let-assignments.patch (SAIL), and sail-arm
# (SAIL_ARM: its arm-v9.4-a directory). The slice keeps what the roots use.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?output directory}
mkdir -p "$out"
cd "$SAIL_ARM"
srcs=$(sed -n 's/^SAIL_SRCS=//p' Makefile | sed 's/\$(DEVICES_PATH)/src\/devices.sail/')
printf ':slice_roots %s\n:slice\n:ast %s/sliced.sail\n:q\n' "$(cat "$here/ROOTS" | tr '\n' ' ')" "$out" > "$out/script.txt"
"$SAIL" --undefined-gen --is "$out/script.txt" $srcs src/termination_measures.sail > "$out/slice.log" 2>&1
# One file: the backend names its outputs by the $include markers.
grep -v '^\$include_\(start\|end\)' "$out/sliced.sail" > "$out/flat.sail"
"$SAIL" --no-core --lean --lean-output-dir "$out/lean" --lean-noncomputable -o armv9 "$out/flat.sail"
