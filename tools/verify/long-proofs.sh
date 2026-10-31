#!/bin/sh
# The proofs for one commit, on this VM, their result posted to GitHub as
# the commit's status "long proofs" (beside CI's checks): the Lean
# specification against C2SP's text and the proofs' definition from it;
# the generated files against their generators; the helpers; every
# hybrid kernel; every SME2 kernel at every group count; every kind of
# x86-64 case prove_x86.py declares (compressions, hash_many, xof_many,
# the chunk root, the partial pass), plus hash_many at every input count
# (prove_x86_loop.py); the instruction models checked against the CPU;
# and the proofs' mutants.
# Run under a bounded process-group supervisor; nothing should measure
# meanwhile. The logs stay in tmp/long-proofs/COMMIT/, and
# the status names their SHA-256 and BLAKE3 hashes.
#
#     sh tools/verify/long-proofs.sh [COMMIT]     (default HEAD)
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
commit=$(git -C "$root" rev-parse "${1:-HEAD}")
token=$(cat "$root/ghtokenclassic.txt")
logs=$root/tmp/long-proofs/$commit
tree=/tmp/long-proofs-$commit
status() {
	python3 -c 'import json, sys; print(json.dumps({"state": sys.argv[1], "context": "long proofs", "description": sys.argv[2]}))' "$1" "$2" |
		curl -sSf --retry 5 --retry-all-errors --retry-delay 10 -o /dev/null -X POST -H "Authorization: token $token" -d @- \
			"https://api.github.com/repos/johnservil/BLAKE3/statuses/$commit"
}
# A status attaches to a commit GitHub has: an unpushed one fails at the
# first post (HTTP 422), after nothing; say so plainly.
curl -sf -o /dev/null -H "Authorization: token $token" "https://api.github.com/repos/johnservil/BLAKE3/commits/$commit" ||
	{ echo "long-proofs: GitHub has no commit $commit: push it first (the run posts its status there)" >&2; exit 1; }
b64() { printf %s "$1" | tr a-f A-F | basenc --base16 -d | basenc --base64url | tr -d '='; }
# The inputs that can change a proof result: the kernels, the proof
# tools (this script among them), the specification, and the hybrid
# generator. A green run for the same inputs is the same evidence, so
# a commit that changes none of them is not proved again; its status
# names the run that already passed.
inputs=$(git -C "$root" ls-tree "$commit" -- c c2sp tools/verify tools/gen_neon_hybrid.py | sha256sum | cut -d' ' -f1)
record=$root/tmp/long-proofs/inputs/$inputs.txt
if [ -f "$record" ]; then
	prior=$(sed -n '1p' "$record")
	psha=$(sed -n '2p' "$record")
	pb3=$(sed -n '3p' "$record")
	description="passed; inputs as $(printf %s "$prior" | cut -c1-12); SHA-256 $(b64 $psha) BLAKE3 $(b64 $pb3)"
	[ ${#description} -le 140 ] || { echo "long-proofs: the status description is ${#description} characters, over GitHub's 140: $description" >&2; exit 1; }
	status success "$description"
	echo "long proofs passed for $commit; same proof inputs as $prior, not rerun; logs in $root/tmp/long-proofs/$prior, SHA-256 $psha, BLAKE3 $pb3 (hex)"
	exit 0
fi
mkdir -p "$logs"
rm -rf "$tree"
git -C "$root" worktree prune
git -C "$root" worktree add -q --detach "$tree" "$commit"
status pending "running on the fork's VM"
cd "$tree"
export ELAN_HOME=${ELAN_HOME:-/root/.elan} PATH=/root/.elan/bin:$PATH
verdict=passed state=success
run() {  # name, command...: its log in $logs/name.txt; a failure ends the run
	name=$1; shift
	[ "$state" = success ] || return 0
	"$@" > "$logs/$name.txt" 2>&1 || { verdict="FAILED ($name)" state=failure; }
}
run lean_spec python3 c2sp/BLAKE3/check.py
run lean_emit sh -c 'python3 tools/verify/lean/emit.py && git diff --exit-code tools/verify/lean/compress.json tools/verify/lean/Generic.lean'
run generated sh -c 'python3 tools/gen_neon_hybrid.py | cmp - c/blake3_neon_hybrid_aarch64.S && python3 tools/gen_sme2_hybrid.py | cmp - c/blake3_sme2_hybrid_aarch64.S && python3 tools/gen_x86.py | cmp - c/blake3_partial_x86-64_unix.S'
run helpers python3 -m unittest discover -s tools/verify -p 'test_*.py'
run prove_hybrid python3 tools/verify/prove_hybrid.py
run prove_sme2 python3 tools/verify/prove_sme2.py chunks chunks_at messages parents xof chunks_x2 every parents_every xof_every messages_every x2_every
# Every kind prove_x86.py declares: the commit's own list, so a kernel a
# commit adds is proved by the script that checks it, whichever commit's
# copy of this script runs (October 25: servil's copy named four kinds and
# skipped a candidate's two new kernels).
run prove_x86 python3 tools/verify/prove_x86.py
run prove_x86_loop sh -c 'ulimit -v 6291456; exec python3 tools/verify/prove_x86_loop.py'
run cross_check python3 tools/verify/cross_check.py
run mutants python3 tools/verify/mutants.py
b3=$(cat "$logs"/*.txt | cargo run -q --release --manifest-path b3sum/Cargo.toml -- --no-names)
sha=$(cat "$logs"/*.txt | sha256sum | cut -d' ' -f1)
# The status's description holds 140 characters: the logs' two hashes,
# each whole in base64url (RFC 4648, 43 characters for 256 bits), fit
# beside the verdict.
description="$verdict; base64url SHA-256 $(b64 $sha) BLAKE3 $(b64 $b3)"
[ ${#description} -le 140 ] || { echo "long-proofs: the status description is ${#description} characters, over GitHub's 140: $description" >&2; exit 1; }
# The record first, then the status: a post that fails (a network down)
# loses no green run, and running this again posts it without proving
# again (the record's path above, #197).
if [ "$state" = success ]; then
	mkdir -p "$root/tmp/long-proofs/inputs"
	printf '%s\n%s\n%s\n' "$commit" "$sha" "$b3" > "$record"
fi
status "$state" "$description" ||
	{ echo "long-proofs: the status post failed; the run's record is kept, so running this again posts it without proving again" >&2; exit 1; }
echo "long proofs $verdict for $commit; logs in $logs, SHA-256 $sha, BLAKE3 $b3 (hex)"
[ "$state" = success ]
