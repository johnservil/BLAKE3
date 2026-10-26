"""Benchmark the mutants this machine's survivor list calls speed-only
(Zooko, October 26; BLAKE3 #183).

    python3 tools/mutants/bench.py [--since REV] [--pairs N] [--re PATTERN]

A row whose reason starts "speed" or "equally correct" says its mutant is
correct either way and faster one way: a claim a benchmark can test. Each
surviving place of such a mutant is applied alone to a clean worktree of
HEAD, and `bench-hashes check --against HEAD` measures the cells that run
its changed lines. Its verdict, one line a place:

    slower           the path earns its place (now with a figure)
    no cell slower   a candidate to delete: correct, and as fast either way
    faster           the mutant is an improvement
    no cell runs it  code no benchmark measures: a cell to write, or code to delete
    no verdict       the machine was busy: run again

A "faster" or "no cell slower" is a reading at N pairs (default 8):
rerun it at 23 (--pairs 23) before acting on it. Places come from
check.py's last full run (tmp/mutants-last/run/tests.tsv): a place any
test failed, crashed or timed out on is caught, and is skipped (a caught
place can crash the benchmark). --since REV takes only the rows the lists
gained or changed since REV, as check.py --since does for survivors.
--re keeps the rows whose mutant matches PATTERN.

One mutant at a time, about 10 minutes each on an i7-12700K: one heavy
job (run it under a bounded supervisor, nothing else measuring). The
bench is the checkout's bench-hashes/, as for check.
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check

SPEED = re.compile(r"(\[cpu [^\]]*\] )?(speed|equally correct)")
VERDICTS = {0: "no cell slower", 1: "slower", 2: "no verdict"}


def speed_rows(lines):
    """(file, mutant) of every row whose reason claims speed alone."""
    out = []
    for line in lines:
        f = line.rstrip("\n").split("\t")
        if len(f) == 4 and not line.startswith("#") and SPEED.match(f[3]):
            out.append((f[0], f[1]))
    return out


def caught_places(tests_tsv_lines):
    """{(file, mutant): {place}} for every place a test did not pass."""
    caught = {}
    for line in tests_tsv_lines:
        f = line.rstrip("\n").split("\t")
        if len(f) == 7 and not line.startswith("#") and f[5] != "PASS":
            caught.setdefault((f[0], f[1]), set()).add(f[2])
    return caught


def surviving(mutants, row, caught):
    """The listed mutants (cargo mutants --list --json) for `row`, at the
    places no test caught."""
    file, change = row
    places = [m for m in mutants if m["file"] == file and m["name"].split(": ", 1)[1] == change]
    return [m for m in places if m["name"].split(": ", 1)[0].split(":", 1)[1] not in caught.get(row, set())]


def verdict(returncode, stdout):
    """check's exit and output as a verdict, and its faster cells."""
    faster = [l.strip() for l in stdout.splitlines() if l.lstrip().startswith("faster ")]
    if "nothing to measure" in stdout:
        return "no cell runs it", faster
    v = VERDICTS.get(returncode, f"check failed ({returncode})")
    return ("faster" if v == "no cell slower" and faster else v), faster


def largest(stdout, shown=3):
    """The `shown` cells check moved furthest each way, and how many more."""
    out = []
    for kind in ("SLOWER", "faster"):
        cells = [l.strip() for l in stdout.splitlines() if l.lstrip().startswith(kind + " ")]
        size = lambda l: abs(float(re.search(r": ([+-]?\d+\.\d+)%", l).group(1)))
        cells.sort(key=size, reverse=True)
        out += cells[:shown] + ([f"and {len(cells) - shown} more cells {kind.lower()}"] if len(cells) > shown else [])
    return out


def git(*args, **kw):
    return subprocess.run(["git", "-C", check.ROOT, *args], capture_output=True, text=True, check=True, **kw).stdout


def main():
    args = sys.argv[1:]
    take = lambda flag: args[args.index(flag) + 1] if flag in args else None
    pairs, since, pattern = take("--pairs") or "8", take("--since"), take("--re")
    lines = open(check.KNOWN).readlines()
    rows = speed_rows(lines)
    if since is not None:
        before = set(git("show", f"{since}:{os.path.relpath(check.KNOWN, check.ROOT)}").splitlines(keepends=True))
        rows = speed_rows([l for l in lines if l not in before])
    if pattern is not None:
        rows = [r for r in rows if re.search(pattern, r[1])]
    tests_tsv = os.path.join(check.ROOT, "tmp", "mutants-last", "run", "tests.tsv")
    assert os.path.exists(tests_tsv), "check.py's last full run names the caught places: run check.py first"
    caught = caught_places(open(tests_tsv))
    listing = ["cargo", "mutants", "--list", "--json", "--diff", "--profile", "mutants", "--features", "serde,zeroize"]
    for f in sorted({f for f, _ in rows}):
        listing += ["--file", f]
    mutants = json.loads(subprocess.run(listing, cwd=check.ROOT, capture_output=True, text=True, check=True).stdout)
    head = git("rev-parse", "HEAD").strip()
    tree = os.path.join(check.scratch_dir(), "mutants-bench")
    git("worktree", "prune")  # a worktree deleted with tmp/ stays registered until pruned
    if not os.path.exists(tree):
        git("worktree", "add", "-q", "--detach", tree, head)
        subprocess.run(["git", "clone", "-q", "--shared", os.path.join(check.ROOT, "bench-hashes"), os.path.join(tree, "bench-hashes")], check=True)
        bench_head = subprocess.run(["git", "-C", os.path.join(check.ROOT, "bench-hashes"), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["git", "-C", os.path.join(tree, "bench-hashes"), "checkout", "-q", "--detach", bench_head], check=True)
    git("-C", tree, "checkout", "-q", "--detach", head)
    print(f"bench: {len(rows)} rows claim speed alone, at {head[:12]}; {pairs} pairs a place", flush=True)
    for row in rows:
        for m in surviving(mutants, row, caught):
            place = m["name"].split(": ", 1)[0].split(":", 1)[1]
            subprocess.run(["patch", "-s", os.path.join(tree, row[0])], input=m["diff"], text=True, check=True)
            try:
                run = subprocess.run(["cargo", "run", "-q", "--release", "--", "check", "--against", head, "--pairs", pairs],
                                     cwd=os.path.join(tree, "bench-hashes"), capture_output=True, text=True)
            finally:
                git("-C", tree, "checkout", "-q", "--", row[0])
            v, faster = verdict(run.returncode, run.stdout)
            print(f"{v}\t{row[0]}\t{row[1]}\t{place}" + "".join(f"\n    {l}" for l in largest(run.stdout)), flush=True)


if __name__ == "__main__":
    main()
