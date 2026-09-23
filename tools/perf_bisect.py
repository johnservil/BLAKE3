#!/usr/bin/env python3
"""Look for performance regressions already in git history.

    python3 tools/perf_bisect.py [--runs 6] [--out DIR] COMMIT...

Measures each commit with the current bench-hashes (the checkout at
./bench-hashes, copied beside each commit's worktree) and judges every
commit against the one before it with tools/perf_regress.py's rule, using
all of both commits' runs: a cell is slower when every run of the later
commit exceeds the earlier commit's slowest run by more than 5%. It also
judges the last commit against the first.

Commits older than the batch API (hash_many and friends) get a shim that
hashes a batch one message at a time, so the benchmark builds; their batch
cells measure the shim, and any comparison that involves such a commit
uses the one-message cells alone.

Runs are saved as DIR/<commit>/run<N>.tsv and reused, so an interrupted
bisect resumes where it stopped.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_regress as pr  # noqa: E402

ROOT = pr.ROOT
SHIM = r'''

// perf_bisect.py shim: the batch API of later commits, one message at a
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


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def prepare(commit, work):
    """A worktree of `commit` at work/src with the current bench-hashes
    inside; returns (worktree path, whether the shim was needed)."""
    tree = work / "src"
    if not tree.exists():
        git("worktree", "add", "--detach", str(tree), commit)
    lib = tree / "src/lib.rs"
    text = lib.read_text()
    shimmed = "pub fn hash_many(" not in text
    if shimmed and "perf_bisect.py shim" not in text:
        lib.write_text(text + SHIM)
    bench = tree / "bench-hashes"
    if not bench.exists():
        shutil.copytree(ROOT / "bench-hashes", bench,
                        ignore=shutil.ignore_patterns("target", "benchmark-results", "tmp", ".git"))
    return tree, shimmed


def measure(commit, runs, out):
    """`runs` saved runs of `commit`, measuring what is missing."""
    work = out / commit
    work.mkdir(parents=True, exist_ok=True)
    have = sorted(work.glob("run*.tsv"))
    if len(have) < runs:
        tree, _ = prepare(commit, work)
        exe = pr.build_bench(tree / "bench-hashes")
        for n in range(len(have), runs):
            (work / f"run{n}.tsv").write_text(pr.run_bench(exe))
            print(f"perf_bisect: {commit} run {n + 1}/{runs}", file=sys.stderr, flush=True)
    shimmed = "pub fn hash_many(" not in git("show", f"{commit}:src/lib.rs")
    return [pr.parse(p.read_text()) for p in sorted(work.glob("run*.tsv"))[:runs]], shimmed


def judge(label, a, b):
    (runs_a, shim_a), (runs_b, shim_b) = a, b
    use_cases = {"OneMessage"} if (shim_a or shim_b) else {"OneMessage", "ManyMessages"}
    base = pr.baseline_from(runs_a)
    # Every commit here was built on the same machine; a commit that changes
    # kernel_report()'s platform name is not a different machine.
    for meta, _ in runs_b:
        for key in [k for k in meta if k.startswith("kernel platform ")]:
            base["identity"][key] = meta[key]
    verdict, lines, slower = pr.judge(base, runs_b, use_cases)
    faster = [l for l in lines if l.startswith("  faster")]
    print(f"\n== {label} ({', '.join(sorted(use_cases))}): {verdict}"
          + (f", {len(slower)} cells slower" if slower else "") + (f", {len(faster)} faster" if faster else ""))
    for line in lines:
        if line.startswith("  slower") or line.startswith("  faster") or verdict == "incomparable":
            print(line)
    return verdict, slower


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("commits", nargs="+", help="commits, oldest first")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--out", default=str(ROOT / "tmp/bisect"))
    args = parser.parse_args()
    out = Path(args.out)
    commits = [git("rev-parse", "--short=7", c) for c in args.commits]
    measured = {c: measure(c, args.runs, out) for c in commits}
    found = []
    for a, b in zip(commits, commits[1:]):
        verdict, slower = judge(f"{b} against {a}", measured[a], measured[b])
        if slower:
            found.append((a, b, slower))
    judge(f"{commits[-1]} against {commits[0]} (first to last)", measured[commits[0]], measured[commits[-1]])
    # A regression that later commits keep reads "ok" between them, and
    # one that is fixed later leaves first-to-last clean: show every
    # flagged cell's 5th percentiles, per run, across all commits.
    for key in sorted({k for _, _, cells in found for k in cells}):
        print(f"\n{key}, 5th percentile per run (ps per unit):")
        for c in commits:
            runs, _ = measured[c]
            print(f"  {c}: " + " ".join(str(v) for v in sorted(cells[key] for _, cells in runs if key in cells)))
    print("\nperf_bisect: " + (f"regressions introduced by {', '.join(b for _, b, _ in found)}" if found
                                else "no commit is slower than the one before it"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
