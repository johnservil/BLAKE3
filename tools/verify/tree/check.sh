#!/bin/sh
# Prove the library's tree walk (src/tree_core.rs) and the binary walk equal
# to the C2SP specification's tree (README.md), the library's walk at every
# buffer size it builds with (MAX_SIMD_DEGREE_OR_2: 128 SME2, 16, 8, 4, 2).
# Needs an Aeneas release directory (AENEAS: the aeneas and charon binaries
# and its built backends/lean), the Rust toolchain rust/*/rust-toolchain.toml
# names, and Lean (elan). Lake's packages (Mathlib) stay in TREE_WORK.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
aeneas=${AENEAS:?set AENEAS to the Aeneas release directory}
work=${TREE_WORK:-$HOME/.cache/blake3-servil-tree}
mkdir -p "$work/packages"
ln -sfn "$aeneas" "$work/aeneas"

# translate CRATE_DIR OUT_DIR: Charon and Aeneas on a crate, in place.
translate() {
	krate=$(basename "$1")
	(cd "$1" && CARGO_TARGET_DIR="$2/target" PATH="$aeneas:$PATH" \
		"$aeneas/charon" cargo --preset=aeneas >/dev/null && mv "$krate.llbc" "$2/")
	"$aeneas/aeneas" -backend lean "$2/$krate.llbc" -dest "$2/lean" >/dev/null
}

# The committed translations are the Rust's (widecore at MAX = 128).
for pair in treecore:Treecore widecore:Widecore; do
	crate=${pair%%:*}; name=${pair##*:}
	rm -rf "$work/$crate"; mkdir -p "$work/$crate"
	translate "$here/rust/$crate" "$work/$crate"
	cmp "$work/$crate/lean/$name.lean" "$here/lean/$name.lean" ||
		{ echo "$name.lean differs from the Rust's translation" >&2; exit 1; }
done

# Each MAX: the walk translated with it, the proof with its numbers.
for max in 128 16 8 4 2; do
	crate="$work/widecore-$max/widecore"; out="$work/widecore-$max"
	rm -rf "$out"; mkdir -p "$crate/src"
	cp "$here/rust/widecore/Cargo.toml" "$here/rust/widecore/rust-toolchain.toml" "$crate/"
	sed -e "s|pub const MAX: usize = 128;|pub const MAX: usize = $max;|" \
		-e "s|include!(\"../../../../../../src/tree_core.rs\");|include!(\"$here/../../../src/tree_core.rs\");|" \
		"$here/rust/widecore/src/lib.rs" > "$crate/src/lib.rs"
	translate "$crate" "$out"
	lean="$work/lean-$max"; mkdir -p "$lean"
	cp "$here"/lean/*.lean "$here/lean/lean-toolchain" "$here/lean/lakefile.lean" "$here/lean/lake-manifest.json" "$lean/"
	cp "$here/../../../c2sp/BLAKE3/Blake3.lean" "$lean/Blake3.lean"
	cp "$out/lean/Widecore.lean" "$lean/Widecore.lean"
	sed -i.orig -e "s/D ≤ 128/D ≤ $max/g" -e "s/256/$((2 * max))/g" "$lean/WideProofs.lean"
	rm -f "$lean/WideProofs.lean.orig"
	# Mathlib's prebuilt cache, once.
	[ -d "$work/packages/mathlib/.lake/build" ] || (cd "$lean" && lake exe cache get)
	(cd "$lean" && lake build)
	echo "MAX = $max: proved"
done
echo "proved: the library's tree walk and the binary walk compute the C2SP specification's tree"
