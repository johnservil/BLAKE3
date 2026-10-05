#!/usr/bin/env python3
"""Performance-regression check for blake3-servil: the working tree against
a commit, measured side by side.

    pypy3 tools/perf_regress.py check                  # against HEAD
    pypy3 tools/perf_regress.py check --against REV    # against a commit
    pypy3 tools/perf_regress.py compare OLD NEW        # two commits
    pypy3 tools/perf_regress.py build                  # bench-hashes against the working tree

It builds bench-hashes twice, once against the fork at the old commit and
once against the new one (for `check`, a commit object made from the
working tree), the same benchmark source in both, and runs
`bench-hashes regress OLD NEW`, whose exit it returns: 0, no cell slower;
1, a cell slower; 2, no verdict (a run's load was busy or unobserved). The
rule, its points, and its margins are bench-hashes' (`regress` in its
src/main.rs, by clocks::summary), and its calibration is in
NOTES-servil.md ("The regression check, calibrated").

Each side is a directory under tmp/perf-ab/ that it owns (a fork worktree,
a copy of bench-hashes with the side's own Cargo.lock, a target
directory), changed only where its sources differ, so Cargo's freshness
rebuilds only what changed and nothing tracked is ever written. `build`
builds the new side alone, for runs by hand, and prints the executable's
path.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The fork checkout the tool works in (its own, or --root's), and the
# sides' directories in it.
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp/perf-ab"

ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(*args, cwd=None, env=ENV, check=True):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, env=env, check=check, stdout=subprocess.PIPE,
                          text=True).stdout.strip()


def working_tree_commit():
    """A commit of the working tree as `git add -A` would stage it, with HEAD
    as its parent: its tree is written through a temporary copy of the
    index, and the commit object takes fixed dates, so one working tree
    always gives one commit. No ref or real index changes."""
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index"
        real = Path(git("rev-parse", "--git-path", "index"))
        shutil.copy2(real if real.is_absolute() else ROOT / real, index)
        env = {**ENV, "GIT_INDEX_FILE": str(index)}
        git("add", "-A", env=env)
        tree = git("write-tree", env=env)
    fixed = {**ENV, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"}
    return git("commit-tree", tree, "-p", "HEAD", "-m", "perf_regress: the working tree", env=fixed)


# bench-hashes depends on the fork's git repository at a pinned commit (what
# users measure). A side builds it against a fork checkout instead, with
# this patch (`..` from the side's copy of bench-hashes is the side's fork
# worktree). The patch changes the lock, so each side's copy owns its own
# Cargo.lock, derived from the committed one; the committed lock is never
# written.
PATCH = 'patch."https://github.com/johnservil/BLAKE3".blake3-servil.path=".."'
# The instrument is the working tree's on both sides, as the benchmark is:
# the clocks crate in this checkout.
# (a function: --root moves ROOT after this module loads).
def clocks_patch():
    return f'patch."https://github.com/johnservil/BLAKE3".clocks.path="{ROOT / "clocks"}"'


def write_if_different(path, content):
    """Write `content` (bytes) to `path` unless it holds them already, so
    an unchanged file keeps its mtime and Cargo sees it unchanged."""
    if not path.exists() or path.read_bytes() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def target_root():
    """Cargo's target directory for this checkout (CARGO_TARGET_DIR, or
    bench-hashes/target). Executables live there, where programs run (a
    VM's shared mount may not run them)."""
    return Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "bench-hashes/target"))


def side_bench(side, commit):
    """The bench-hashes executable built against the fork at `commit`, in
    the side `side` ("old", "new", or the runner's "bench"): a fork worktree under CACHE/side with a copy of bench-hashes
    inside, and a target directory of its own. Each step changes only what
    differs (the worktree moves only when its commit does, and then git
    rewrites only the files that differ; the copy is written file by file
    where it differs), so Cargo's own freshness decides what to rebuild,
    and a side whose code is unchanged builds nothing. A commit whose API
    the benchmark no longer calls fails to build: compare it with the tools
    of its own time (AGENTS.md, "Contracts change everywhere at once")."""
    commit = git("rev-parse", f"{commit}^{{commit}}")
    checkout = CACHE / side / "src"
    if not checkout.exists():
        if checkout.parent.exists():
            git("worktree", "prune")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(checkout), commit)
    elif git("rev-parse", "HEAD", cwd=checkout) != commit:
        git("checkout", "--quiet", "--force", "--detach", commit, cwd=checkout)
    # The benchmark as it is now, so only the fork differs between sides;
    # everything but the lock, which the side derives.
    bench, copy = ROOT / "bench-hashes", checkout / "bench-hashes"
    skip = {"target", "benchmark-results", "tmp", ".git"}
    wanted = set()
    for path in sorted(bench.rglob("*")):
        relative = path.relative_to(bench)
        if relative.parts[0] in skip or relative == Path("Cargo.lock") or not path.is_file():
            continue
        wanted.add(relative)
        write_if_different(copy / relative, path.read_bytes())
    for path in sorted(copy.rglob("*")):
        relative = path.relative_to(copy)
        if relative.parts[0] not in skip and relative != Path("Cargo.lock") and path.is_file() and relative not in wanted:
            path.unlink()
    # The side's lock: the committed one, patched by Cargo, derived again
    # only when the committed one changes. Cargo applies a patch only at the
    # locked version, so when the side's fork has another version,
    # `cargo update -p blake3-servil` takes it into the lock.
    committed = (bench / "Cargo.lock").read_bytes()
    derived_from = CACHE / side / "Cargo.lock.committed"
    if not derived_from.exists() or derived_from.read_bytes() != committed:
        (copy / "Cargo.lock").write_bytes(committed)
        derived_from.write_bytes(committed)
    # The copy's provenance is the checkout it was copied from (bench-hashes'
    # build.rs would otherwise ask git in the fork worktree around the copy).
    env = {**ENV, "CARGO_TARGET_DIR": str(target_root() / f"perf-{side}"), "BENCH_HASHES_CHECKOUT": str(bench)}
    fork_version = re.search(r'(?m)^version = "([^"]+)"', (checkout / "Cargo.toml").read_text()).group(1)
    locked = re.search(r'name = "blake3-servil"\nversion = "([^"]+)"', (copy / "Cargo.lock").read_text()).group(1)
    if locked != fork_version:
        subprocess.run(["cargo", "--config", PATCH, "--config", clocks_patch(), "update", "--quiet", "-p", "blake3-servil"], cwd=copy, env=env,
                       check=True)
    out = subprocess.run(["cargo", "--config", PATCH, "--config", clocks_patch(), "build", "--release", "--message-format=json-render-diagnostics"],
                         cwd=copy, env=env, check=True, stdout=subprocess.PIPE, text=True).stdout
    messages = [json.loads(line) for line in out.splitlines()]
    # The fork must come from the side's checkout, never the locked commit.
    sources = {m["package_id"] for m in messages
               if m.get("reason") == "compiler-artifact" and m["target"]["name"] == "blake3_servil"}
    fork = f"path+file://{checkout.resolve()}"
    assert sources and all(s.startswith(fork + "#") for s in sources), \
        f"bench-hashes built blake3-servil from {sources}, not the checkout {fork}"
    exes = [m["executable"] for m in messages
            if m.get("reason") == "compiler-artifact" and m.get("executable")
            and m["target"]["name"] == "bench-hashes"]
    assert len(exes) == 1, f"expected the bench-hashes executable, found {exes}"
    require_sme2_kernel(exes[0])
    return exes[0]


def machine_has_sme2():
    """Whether this CPU reports SME2 (Linux: /proc/cpuinfo; macOS: sysctl)."""
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_SME2"], env=ENV, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True).stdout.strip()
        return out == "1"
    cpuinfo = Path("/proc/cpuinfo")
    return cpuinfo.exists() and " sme2" in cpuinfo.read_text()


def require_sme2_kernel(exe):
    """Fail stop when this machine has SME2 and the build left the kernel
    out (the fork builds without it when the C compiler cannot assemble
    SME2): the check would measure NEON alone and miss every SME2 change."""
    if not machine_has_sme2():
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([exe, "--contenders", "blake3-servil-st,sha256", "--points", "16 KiB", "--rounds", "1"],
                             cwd=tmp, env=ENV, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True).stdout
    assert "SME2" in out, ("this CPU has SME2, and the fork was built without its SME2 kernel: "
                           "point CC at a compiler that assembles SME2 (CC=clang-19 in the VM)")


def compare(old_rev, new):
    """Compare the fork at `old_rev` with `new` (a commit, or None for the
    working tree): bench-hashes regress. Returns its exit code."""
    old = side_bench("old", old_rev)
    new_exe = side_bench("new", working_tree_commit() if new is None else new)
    name = "the working tree" if new is None else new
    print(f"perf_regress: {name} against {old_rev}", file=sys.stderr, flush=True)
    return subprocess.run([new_exe, "regress", old, new_exe], env=ENV).returncode


def main():
    global ROOT, CACHE
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, help="the fork checkout to work in (default: this tool's)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="the working tree against a commit")
    p.add_argument("--against", default="HEAD")
    p = sub.add_parser("compare", help="two commits")
    p.add_argument("old")
    p.add_argument("new")
    p = sub.add_parser("build", help="build bench-hashes against the working tree, or a commit, as it is; "
                                     "print the executable's path")
    p.add_argument("--commit", help="a fork commit (default: the working tree)")
    p.add_argument("--side", default="new", help="the side directory to build in (default: new)")
    args = parser.parse_args()
    if args.root:
        ROOT = args.root.resolve()
        CACHE = ROOT / "tmp/perf-ab"
    # The sides are one checkout's: one invocation at a time uses them.
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"perf_regress: another perf_regress holds {CACHE / 'lock'}; run one at a time")
        if args.command == "build":
            print(side_bench(args.side, args.commit or working_tree_commit()))
            return 0
        return compare(args.against, None) if args.command == "check" else compare(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
