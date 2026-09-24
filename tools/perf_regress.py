#!/usr/bin/env python3
"""Performance-regression check for blake3-servil: the working tree against
a commit, measured side by side.

    python3 tools/perf_regress.py check                  # against HEAD
    python3 tools/perf_regress.py check --against v0.7.0 # against a release
    python3 tools/perf_regress.py compare OLD NEW        # two commits

Exit 0: no regression. 1: a confirmed regression. 2: no verdict (the
comparison itself was unreliable, see below).

`check` builds bench-hashes twice, once against the fork at REV (a git
worktree, cached per commit in tmp/perf-ab/) and once against this working
tree, the same benchmark source in both, and runs the two builds in
alternating pairs on this machine, one right after the other, each with
the contenders sha256 (the control), blake3-servil, and blake3-servil-mt.
Load, thermal state, and drift reach both sides of a pair alike, so there
is nothing to record, store, or keep current, and any machine can run it.

The rule, calibrated on the 16-vCPU VM with 32 runs of one commit taken back
to back while the host's load came and went (NOTES-servil.md):

* The statistic is a cell's 5th percentile per run: a low quantile moves
  when the code does, a median moves with the host.
* The runs go A B B A A B B A (A the old side): four pairs of neighbours
  in time, which share the machine's state, each side first in two of
  them, so a steady drift across the eight runs cancels. A cell is slower
  when, in every pair, the new side's 5th percentile exceeds the old
  side's by more than MARGIN.
* Any slower cell triggers another A B B A A B B A; a regression is a
  cell slower in both.
* The control is the same code on both sides. If the rule calls any of
  its cells slower or faster, the comparison is unreliable: no verdict.

Measured with four pairs: no false flag in 2400 cell comparisons of
unchanged code before confirmation (under 0.13% per cell at 95%
confidence); a cell 5% slower is caught 70% of the time, 10% slower 95%,
20% slower 100%. A check takes eight runs, about 37 s on the VM, plus the
builds.

The check measures the 25 points in POINTS, which cover the code paths
and boundaries of both use cases at the benchmark's points; the published graph's plateau sizes add
run time and no path.

Commits that predate the batch API (hash_many and friends) get a shim that
hashes a batch one message at a time, so the current benchmark builds; a
comparison involving such a commit judges the one-message cells alone.
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp/perf-ab"
CONTROL = "sha256"
SUBJECTS = ["blake3-servil", "blake3-servil-mt"]
CONTENDERS = [CONTROL] + SUBJECTS
# The code paths and boundaries of both use cases among the benchmark's
# points, and none of the plateau sizes the published graph needs (16-128
# MiB and batches past 16384 cost 60% of a full run and exercise no path
# that 8 MiB does not): one message on the scalar kernel (64 B, 1 KiB), the
# hybrids (2, 3, 4, 8 KiB; 5-7 KiB are no benchmark point), the first SME2
# groups (16, 32 KiB), the split threshold (64 KiB), bulk (256 KiB,
# 1 MiB), unequal subtrees (3 MiB), and the memory-resident plateau
# (8 MiB); batches of one, of the NEON parent plans (2, 3, 8), of a first
# and a partial SME2 group (16, 24), in bulk (64, 256), at the split
# (1024), and over the pool (2048, 4096, 16384).
POINTS = ["64 B", "1 KiB", "2 KiB", "3 KiB", "4 KiB", "8 KiB", "16 KiB", "32 KiB", "64 KiB",
          "256 KiB", "1 MiB", "3 MiB", "8 MiB",
          "1", "2", "3", "8", "16", "24", "64", "256", "1024", "2048", "4096", "16384"]
ROUNDS = 48
QUANTILE = 0.05
PAIRS = 4  # the runs go A B B A A B B A
MARGIN = 0.03

SHIM = r'''

// perf_regress.py shim: the batch API of later commits, one message at a
// time, so the current bench-hashes builds against this commit.
pub fn hash_many(inputs: &[&[u8]], outputs: &mut [Hash]) {
    assert_eq!(inputs.len(), outputs.len());
    for (input, output) in inputs.iter().zip(outputs) {
        *output = hash(input);
    }
}
#[cfg(feature = "std")]
pub fn hash_many_multithreaded(inputs: &[&[u8]], outputs: &mut [Hash]) {
    hash_many(inputs, outputs)
}
#[cfg(feature = "std")]
pub fn hash_many_multithreaded_with_budget(inputs: &[&[u8]], outputs: &mut [Hash], _max_threads: usize) {
    hash_many(inputs, outputs)
}
#[cfg(feature = "std")]
pub fn kernel_report_many() -> KernelReport {
    kernel_report()
}
#[cfg(feature = "std")]
pub fn kernel_report_many_multithreaded() -> KernelReport {
    kernel_report_multithreaded()
}
'''


# Every command runs in this environment: this one without git's repository
# variables. Inside a pre-commit hook git sets GIT_INDEX_FILE (for
# `git commit PATH...`, a temporary index that becomes the commit) and
# GIT_DIR; `git worktree add` would check out into that index, committing
# HEAD's tree, and bench-hashes' build script would run git against the
# fork's repository.
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, env=ENV, check=True, stdout=subprocess.PIPE,
                          text=True).stdout.strip()




def build_bench(bench):
    """Build bench-hashes from its own directory, where its .cargo/config.toml
    (target-cpu=native) applies, and return the executable's path."""
    out = subprocess.run(["cargo", "build", "--release", "--message-format=json-render-diagnostics"],
                         cwd=bench, env=ENV, check=True, stdout=subprocess.PIPE, text=True).stdout
    exes = [m["executable"] for m in map(json.loads, out.splitlines())
            if m.get("reason") == "compiler-artifact" and m.get("executable")
            and m["target"]["name"] == "bench-hashes"]
    assert len(exes) == 1, f"expected the bench-hashes executable, found {exes}"
    return exes[0]


def target_dir():
    meta = subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT / "bench-hashes",
                          env=ENV, check=True, stdout=subprocess.PIPE, text=True).stdout
    return Path(json.loads(meta)["target_directory"])


def bench_fingerprint():
    """The benchmark source that `commit_bench` copies into each build, as
    12 hex digits: every file it builds from, so a cached executable never
    outlives a change to the benchmark."""
    digest = hashlib.sha256()
    bench = ROOT / "bench-hashes"
    for path in sorted([bench / "Cargo.toml", bench / "Cargo.lock", bench / "build.rs", *(bench / "src").rglob("*.rs"),
                        *(bench / ".cargo").rglob("*")]):
        if path.is_file():
            digest.update(str(path.relative_to(bench)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:12]


def commit_bench(rev):
    """(bench-hashes executable built against the fork at `rev`, whether it
    needed the shim). Built once per fork commit and benchmark source, in a
    worktree under CACHE, and kept in the target directory."""
    commit = git("rev-parse", "--short=12", f"{rev}^{{commit}}")
    # Executables live in Cargo's target directory, which runs programs
    # everywhere (a VM's shared mount may not). The name carries the
    # benchmark's fingerprint as well as the fork's commit.
    exe = target_dir() / "perf-ab" / f"{commit}-{bench_fingerprint()}" / "bench-hashes"
    shimmed = "pub fn hash_many(" not in git("show", f"{commit}:src/lib.rs")
    if exe.exists():
        return str(exe), shimmed
    exe.parent.mkdir(parents=True, exist_ok=True)
    work = CACHE / commit
    work.mkdir(parents=True, exist_ok=True)
    tree = work / "src"
    if tree.exists():
        git("worktree", "remove", "--force", str(tree))
    git("worktree", "add", "--detach", str(tree), commit)
    try:
        if shimmed:
            lib = tree / "src/lib.rs"
            lib.write_text(lib.read_text() + SHIM)
        # The benchmark as it is now, so only the fork differs between sides.
        shutil.copytree(ROOT / "bench-hashes", tree / "bench-hashes",
                        ignore=shutil.ignore_patterns("target", "benchmark-results", "tmp", ".git"))
        shutil.copy2(build_bench(tree / "bench-hashes"), exe)
    finally:
        git("worktree", "remove", "--force", str(tree))
    return str(exe), shimmed


def run(exe):
    """One run in a scratch directory; {"contender|scenario|use_case|point": 5th percentile}."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, "--contenders", ",".join(CONTENDERS), "--points", ",".join(POINTS),
                        "--rounds", str(ROUNDS)], cwd=tmp, env=ENV, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        found = list(Path(tmp).glob("benchmark-results/*/bench-hashes.samples.tsv"))
        assert len(found) == 1, f"expected one samples file, found {found}"
        return parse(found[0].read_text())


def parse(text):
    cells, header = {}, None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if header is None:
            header = line.split("\t")
            assert header == ["contender", "scenario", "use_case", "point", "unit", "ps_per_unit"], header
        elif line:
            contender, scenario, use_case, point, _unit, values = line.split("\t")
            ordered = sorted(int(v) for v in values.split(","))
            cells[f"{contender}|{scenario}|{use_case}|{point}"] = ordered[int(QUANTILE * len(ordered))]
    return cells


def pairs(old, new, count, start):
    """`count` pairs of (old run, new run); the side that runs first
    alternates, beginning with old when `start` is even."""
    out = []
    for i in range(count):
        if (start + i) % 2 == 0:
            a = run(old)
            b = run(new)
        else:
            b = run(new)
            a = run(old)
        out.append((a, b))
        print(f"perf_regress: pair {start + i + 1} done", file=sys.stderr, flush=True)
    return out


def judge(measured, use_cases, contenders):
    """(slower, faster, ratio per cell): a cell is slower (faster) when every
    pair's new/old ratio exceeds 1 + MARGIN (falls below 1 - MARGIN)."""
    slower, faster, ratio = [], [], {}
    for key in measured[0][0]:
        contender, _scenario, use_case, _ = key.split("|")
        if contender not in contenders or use_case not in use_cases:
            continue
        ratios = [b[key] / a[key] for a, b in measured]
        ratio[key] = statistics.median(ratios)
        if min(ratios) > 1 + MARGIN:
            slower.append(key)
        elif max(ratios) < 1 - MARGIN:
            faster.append(key)
    return slower, faster, ratio


def compare(old_rev, new):
    """Compare the fork at `old_rev` with `new` (a commit, or None for the
    working tree). Returns the exit code."""
    old, old_shim = commit_bench(old_rev)
    if new is None:
        new_exe, new_shim, new_name = build_bench(ROOT / "bench-hashes"), False, "the working tree"
    else:
        (new_exe, new_shim), new_name = commit_bench(new), new
    use_cases = {"OneMessage"} if (old_shim or new_shim) else {"OneMessage", "ManyMessages"}
    print(f"perf_regress: {new_name} against {old_rev}, {PAIRS} alternating pairs, "
          f"use cases {', '.join(sorted(use_cases))}", file=sys.stderr, flush=True)
    measured = pairs(old, new_exe, PAIRS, 0)

    def unreliable(measured):
        control = judge(measured, use_cases, [CONTROL])
        if control[0] or control[1]:
            print(f"perf_regress: the control ({CONTROL}, the same code on both sides) moved in "
                  f"{len(control[0]) + len(control[1])} cells: the machine's state changed within pairs. "
                  "No verdict (exit 2); run again when nothing else runs on the machine.")
            return True
        return False

    if unreliable(measured):
        return 2
    slower, faster, ratio = judge(measured, use_cases, SUBJECTS)
    if slower:
        print(f"perf_regress: {len(slower)} cells slower in {PAIRS} pairs; {PAIRS} more pairs must agree",
              file=sys.stderr, flush=True)
        more = pairs(old, new_exe, PAIRS, PAIRS)
        if unreliable(more):
            return 2
        slower2, _, ratio2 = judge(more, use_cases, SUBJECTS)
        confirmed = sorted(set(slower) & set(slower2))
        if confirmed:
            print(f"perf_regress: REGRESSION: {new_name} is slower than {old_rev} in {len(confirmed)} cells "
                  f"(5th percentile, median of pair ratios):")
            for key in confirmed:
                print(f"  {key}: {ratio[key] - 1:+.1%}, then {ratio2[key] - 1:+.1%}")
            return 1
        print(f"perf_regress: the second {PAIRS} pairs did not confirm; no regression")
    for key in sorted(faster):
        print(f"  faster  {key}: {ratio[key] - 1:+.1%}")
    print(f"perf_regress: no regression against {old_rev}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="the working tree against a commit")
    p.add_argument("--against", default="HEAD")
    p = sub.add_parser("compare", help="two commits")
    p.add_argument("old")
    p.add_argument("new")
    args = parser.parse_args()
    if args.command == "check":
        return compare(args.against, None)
    return compare(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
