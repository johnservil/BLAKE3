#!/bin/sh
# The proofs for one commit, on this VM, their result posted to GitHub as
# the commit's status "long proofs" (beside CI's checks): the Lean
# specification against C2SP's text and the proofs' definition from it;
# the generated hybrid file against its generator; the helpers; every
# hybrid kernel; every SME2 kernel at every group count (the chunk and
# parent kernels' inductions are what the SME2 contracts in the Rust and
# whole-call proofs rest on); the Rust paths and whole calls, with the
# instruction models they run checked against the CPU; the proofs' mutants;
# and the tests' (tools/mutants/check.py).
# About nine hours, in a worktree of the commit; nothing should measure
# meanwhile. The logs stay in tmp/long-proofs/COMMIT/, and the status
# names their BLAKE3 hash.
#
#     sh tools/verify/long-proofs.sh [COMMIT]     (default HEAD)
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
commit=$(git -C "$root" rev-parse "${1:-HEAD}")
token=$(cat "$root/ghtokenclassic.txt")
logs=$root/tmp/long-proofs/$commit
tree=/tmp/long-proofs-$commit
status() {
	pypy3 -c 'import json, sys; print(json.dumps({"state": sys.argv[1], "context": "long proofs", "description": sys.argv[2]}))' "$1" "$2" |
		curl -sSf -o /dev/null -X POST -H "Authorization: token $token" -d @- \
			"https://api.github.com/repos/johnservil/BLAKE3/statuses/$commit"
}
mkdir -p "$logs"
rm -rf "$tree"
git -C "$root" worktree prune
git -C "$root" worktree add -q --detach "$tree" "$commit"
status pending "running on the fork's VM"
cd "$tree"
# Temporary files on /tmp, the large tmpfs (the root filesystem is small).
export ELAN_HOME=${ELAN_HOME:-/root/.elan} PATH=/root/.elan/bin:$PATH TMPDIR=/tmp
verdict=passed state=success
run() {  # name, command...: its log in $logs/name.txt; a failure ends the run
	name=$1; shift
	[ "$state" = success ] || return 0
	"$@" > "$logs/$name.txt" 2>&1 || { verdict="FAILED ($name)" state=failure; }
}
# The steps are the commit's own, from its tree: a commit's proofs are
# what it names.
. ./tools/verify/long-proofs-steps.sh
hash=$(pypy3 -c 'import sys; sys.path.insert(0, "tools/verify"); import results
print(results.blake3(data="\n".join(results.blake3(sorted(sys.argv[1:]))).encode())[0][:16])' "$logs"/*.txt)
status "$state" "$verdict: Lean, hybrid, SME2, cross_check, mutants; logs' BLAKE3 $hash (VM)"
echo "long proofs $verdict for $commit; logs in $logs, BLAKE3 $hash"
