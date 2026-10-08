#!/bin/sh
# Prove the safe-Rust tree walks equal to the C2SP specification's tree
# (README.md). Needs an Aeneas release directory (AENEAS: the aeneas and
# charon binaries and its built backends/lean), the Rust toolchain
# rust/*/rust-toolchain.toml names, and Lean (elan). Lake's packages
# (Mathlib) stay in TREE_WORK between runs.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
aeneas=${AENEAS:?set AENEAS to the Aeneas release directory}
work=${TREE_WORK:-$HOME/.cache/blake3-servil-tree}
mkdir -p "$work/lean"
# The committed translations are the Rust's.
for pair in treecore:Treecore widecore:Widecore; do
	crate=${pair%%:*}; name=${pair##*:}
	rm -rf "$work/$crate"; cp -r "$here/rust/$crate" "$work/$crate"
	(cd "$work/$crate" && PATH="$aeneas:$PATH" "$aeneas/charon" cargo --preset=aeneas >/dev/null)
	"$aeneas/aeneas" -backend lean "$work/$crate/$crate.llbc" -dest "$work/$crate/lean" >/dev/null
	cmp "$work/$crate/lean/$name.lean" "$here/lean/$name.lean" ||
		{ echo "$name.lean differs from the Rust's translation" >&2; exit 1; }
done
cp "$here"/lean/*.lean "$here/lean/lean-toolchain" "$work/lean/"
cp "$here/../../../c2sp/BLAKE3/Blake3.lean" "$work/lean/Blake3.lean"
ln -sfn "$aeneas" "$work/aeneas"
cd "$work/lean"
[ -f lake-manifest.json ] || { lake update; lake exe cache get; }
lake build
echo "proved: the safe-Rust tree walks compute the C2SP specification's tree"
