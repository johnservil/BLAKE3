#!/usr/bin/env python3
"""Performance-regression check for blake3-servil: the working tree against
a commit, measured side by side.

    python3 tools/perf_regress.py check                  # against HEAD
    python3 tools/perf_regress.py check --against v0.7.0 # against a release
    python3 tools/perf_regress.py compare OLD NEW        # two commits
    python3 tools/perf_regress.py build                  # bench-hashes against the working tree

Exit 0: no solo regression (shared cells slower are listed). 1: a
confirmed solo regression. 2: no verdict (the comparison itself was
unreliable, see below).

`check` builds bench-hashes twice, once against the fork at REV (the old
side) and once against this working tree (the new side, as a commit object
made from it), the same benchmark source in both, and runs the two builds
in alternating pairs on this machine, one right after the other, each with
the contenders sha256 (the control), blake3-servil-st, and blake3-servil-mt.
Each side is a directory under tmp/perf-ab/ that it owns (a fork worktree,
a copy of bench-hashes with the side's own Cargo.lock, a target
directory), changed only where its sources differ, so Cargo's freshness
rebuilds only what changed and nothing tracked is ever written. `build`
builds the new side alone, for runs by hand.
Load, thermal state, and drift reach both sides of a pair alike, so there
is nothing to record, store, or keep current, and any machine can run it.

The rule, calibrated on the 16-vCPU VM with runs of one commit taken back
to back (NOTES-servil.md, "perf_regress"):

* The statistic is a cell's 5th percentile per run: a low quantile moves
  when the code does, a median moves with the host.
* The runs go A B B A A B B A (A the old side): four pairs of neighbours
  in time, which share the machine's state, each side first in two of
  them, so a steady drift across the eight runs cancels. A cell (one
  contender, scenario, use case, and point) is slower when, in every
  pair, the new side's 5th percentile exceeds the old side's by more than
  the scenario's margin: 3% solo, 10% shared (Zooko, September 25, 2026:
  the recommended usage first, the shared scenario measured and held to
  a looser line; AGENTS.md).
* A regression holds the change (exit 1) when a solo cell is slower; a
  shared cell slower is reported beside an exit 0, and the commit names
  it, its numbers, and the reason the change is worth it (Zooko,
  September 26, 2026).
* A pair's runs measure only the points where some cell is still open:
  one whose every pair so far exceeds its margin in one direction, so it
  could still be called slower or faster. The first pair measures every
  point; later pairs, fewer. This decides exactly as measuring every
  point in every pair would, from the pairs a cell's verdict can use.
* Any slower cell triggers another A B B A A B B A over the points of the
  slower cells alone; a regression is a cell slower in both.
* The control is the same code on both sides. If the rule calls any of
  its cells slower or faster, the comparison is unreliable: no verdict.
* Each listed cell also shows the median pair ratio of 90th percentiles,
  which a two-speed cell's slow speed reaches; it informs, and the
  verdict ignores it (the rule is calibrated for the 5th percentile).

Measured on unchanged code (VM, September 26, 2026; checks simulated
over consecutive runs of 24 rounds): false flags before confirmation in
0.07% of cells (2 of 2900), no false no-verdict in 25 checks; a solo cell
5% slower is caught 81% of the time, 10% slower 92%, 20% slower 95% (the
misses are cells whose speed differs from process to process: servil mt
at 64 KiB and 1024 messages, servil st at 32 KiB). A check takes about
16 s on the VM plus the builds, against 73 s when every pair ran every
point at 48 rounds.

The check measures the 29 points in POINTS, which cover the code paths
and boundaries of the one-message and batch use cases at the benchmark's points; the published graph's plateau sizes add
run time and no path.

The benchmark calls the batch API that takes one buffer of equal messages
(hash_many(input, message_len, out)). Older commits get a shim so it
builds: one with that API under the name hash_many_equal forwards to it,
and its batch cells are judged; one whose hash_many takes a slice of
slices has it renamed hash_many_slices and a copying shim over it, and
one that predates batches gets a shim hashing one message at a time. A
comparison involving either of those last two measures and judges the
one-message cells alone. Batch kernel reports that take no message length are
renamed and called through a shim that takes it.
Commits that predate Stream get a shim over a Hasher on the calling
thread (the check measures no streamed cells).
"""
import argparse
import fcntl
import hashlib
from fractions import Fraction
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp/perf-ab"
CONTROL = "sha256"
SUBJECTS = ["blake3-servil-st", "blake3-servil-mt"]
CONTENDERS = [CONTROL] + SUBJECTS
# The code paths and boundaries of both use cases among the benchmark's
# points, and none of the plateau sizes the published graph needs (16-128
# MiB and batches past 16384 cost 60% of a full run and exercise no path
# that 8 MiB does not): one message on the scalar kernel (64 B, 1 KiB), the
# hybrids (2, 3, 4, 8 KiB; a partial final chunk beside whole ones at
# 2304, 3839, 4470, 7935 B, the q kernels), the first SME2
# groups (16, 32 KiB), the split threshold (64 KiB), bulk (256 KiB,
# 1 MiB), unequal subtrees (3 MiB), and the memory-resident plateau
# (8 MiB); batches of one, of the NEON parent plans (2, 3, 8), of a first
# and a partial SME2 group (16, 24), in bulk (64, 256), at the split
# (1024), and over the pool (2048, 4096, 16384).
ONE_MESSAGE_POINTS = ["64 B", "1 KiB", "2 KiB", "2304 B", "3 KiB", "3839 B", "4 KiB", "4470 B", "7935 B", "8 KiB", "16 KiB",
                      "32 KiB", "64 KiB", "256 KiB", "1 MiB", "3 MiB", "8 MiB"]
POINTS = ONE_MESSAGE_POINTS + [
          "1", "2", "3", "8", "16", "24", "64", "256", "1024", "2048", "4096", "16384"]
# Rounds per run: the variance between processes exceeds a run's sampling
# noise, so short runs lose little (5% slower caught 81% at 24 rounds, 84%
# at 48, 61% at 12).
ROUNDS = 24
QUANTILE = 0.05
PAIRS = 4  # the runs go A B B A A B B A
# A cell is slower (faster) past this ratio, by scenario.
MARGIN = {"solo": Fraction(3, 100), "shared": Fraction(10, 100)}

# A commit's lib.rs contains one of these, newest first; each names the
# shim that makes the benchmark build against it, and whether its batch
# cells are judged.
BATCH_ONE_BUFFER = "pub fn hash_many(input: &[u8]"
BATCH_EQUAL = "pub fn hash_many_equal("
BATCH_SLICES = "pub fn hash_many(inputs: &[&[u8]]"

# The slice API's public names, renamed out of the way (hash_many_slices,
# ...) in every src/*.rs, definitions and in-crate calls alike.
SLICES_RENAME = (r"(pub fn |crate::)(hash_many(?:_multithreaded(?:_with_budget)?)?)\(", r"\1\2_slices(")

# The batch kernel reports before they took the message length, renamed
# out of the way (kernel_report_many_one_block, ...) and called through a
# shim that takes it (the check reads no kernels).
KERNELS_ONE_BLOCK = "pub fn kernel_report_many() ->"
KERNELS_RENAME = (r"\bkernel_report_many(_multithreaded)?\(\)", r"kernel_report_many\1_one_block()")

SHIM_KERNELS = r'''

// perf_regress.py shim: the batch kernel reports of later commits, which
// take the message length; this commit's describe one-block messages.
#[cfg(feature = "std")]
pub fn kernel_report_many(_message_len: usize) -> KernelReport {
    kernel_report_many_one_block()
}
#[cfg(feature = "std")]
pub fn kernel_report_many_multithreaded(_message_len: usize) -> KernelReport {
    kernel_report_many_multithreaded_one_block()
}
'''

SHIM_FORWARD = r'''

// perf_regress.py shim: the one-buffer batch API under its later names.
pub fn hash_many(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    hash_many_equal(input, message_len, out)
}
#[cfg(feature = "std")]
pub fn hash_many_multithreaded(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    hash_many_equal_multithreaded(input, message_len, out)
}
'''

SHIM_SLICES = r'''

// perf_regress.py shim: the one-buffer batch API over the slice API,
// copying (its batch cells are not judged).
fn one_buffer_over_slices(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]], f: fn(&[&[u8]], &mut [Hash])) {
    let inputs: Vec<&[u8]> = (0..out.len()).map(|i| &input[i * message_len..][..message_len]).collect();
    let mut hashes = vec![Hash::from_bytes([0; OUT_LEN]); out.len()];
    f(&inputs, &mut hashes);
    for (o, h) in out.iter_mut().zip(&hashes) {
        *o = *h.as_bytes();
    }
}
pub fn hash_many(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    one_buffer_over_slices(input, message_len, out, hash_many_slices)
}
#[cfg(feature = "std")]
pub fn hash_many_multithreaded(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    one_buffer_over_slices(input, message_len, out, hash_many_multithreaded_slices)
}
'''

SHIM = r'''

// perf_regress.py shim: the batch API of later commits, one message at a
// time, so the current bench-hashes builds against this commit.
pub fn hash_many(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(input.len(), message_len * out.len());
    for (i, o) in out.iter_mut().enumerate() {
        *o = *hash(&input[i * message_len..][..message_len]).as_bytes();
    }
}
#[cfg(feature = "std")]
pub fn hash_many_multithreaded(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    hash_many(input, message_len, out)
}
#[cfg(feature = "std")]
pub fn kernel_report_many(_message_len: usize) -> KernelReport {
    kernel_report()
}
#[cfg(feature = "std")]
pub fn kernel_report_many_multithreaded(_message_len: usize) -> KernelReport {
    kernel_report_multithreaded()
}
'''


SHIM_STREAM = r'''

// perf_regress.py shim: the Stream of later commits, over a Hasher on the
// calling thread, so the current bench-hashes builds against this commit
// (the check measures no streamed cells).
#[cfg(feature = "std")]
pub struct Stream { hasher: Hasher, buffer: Vec<u8> }
#[cfg(feature = "std")]
impl Stream {
    pub fn new() -> Self { Stream { hasher: Hasher::new(), buffer: vec![0; 1 << 16] } }
    pub fn new_multithreaded() -> Self { Self::new() }
    pub fn buffer(&mut self) -> &mut [u8] { &mut self.buffer }
    pub fn filled(&mut self, n: usize) { let b = std::mem::take(&mut self.buffer); self.hasher.update(&b[..n]); self.buffer = b; }
    pub fn finalize(self) -> Hash { self.hasher.finalize() }
}
'''


# Every command runs in this environment: this one without git's repository
# variables. Inside a pre-commit hook git sets GIT_INDEX_FILE (for
# `git commit PATH...`, a temporary index that becomes the commit) and
# GIT_DIR; `git worktree add` would check out into that index, committing
# HEAD's tree, and bench-hashes' build script would run git against the
# fork's repository.
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(*args, cwd=ROOT, env=ENV, check=True):
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=check, stdout=subprocess.PIPE,
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


def write_if_different(path, content):
    """Write `content` (bytes) to `path` unless it holds them already, so
    an unchanged file keeps its mtime and Cargo sees it unchanged."""
    if not path.exists() or path.read_bytes() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def shimmed_sources(commit):
    """({path in the fork: its source with the shims this commit needs},
    whether its batch cells cannot be judged). Each source is derived from
    the commit's own, so applying the shims again writes nothing."""
    lib = git("show", f"{commit}:src/lib.rs")
    out, shimmed = {}, False
    if BATCH_SLICES in lib:
        for name in git("ls-tree", "--name-only", f"{commit}:src").split():
            if name.endswith(".rs"):
                text = git("show", f"{commit}:src/{name}")
                renamed = re.sub(*SLICES_RENAME, text)
                if renamed != text:
                    out[f"src/{name}"] = renamed
        shimmed = True
    lib = out.get("src/lib.rs", lib)
    if BATCH_ONE_BUFFER in lib:
        pass
    elif BATCH_EQUAL in lib:
        lib += SHIM_FORWARD
    elif BATCH_SLICES in git("show", f"{commit}:src/lib.rs"):
        lib += SHIM_SLICES
    else:
        lib, shimmed = lib + SHIM, True
    if KERNELS_ONE_BLOCK in lib:
        lib = re.sub(*KERNELS_RENAME, lib) + SHIM_KERNELS
    if "pub use stream::Stream" not in lib:
        lib += SHIM_STREAM
    if lib != git("show", f"{commit}:src/lib.rs"):
        out["src/lib.rs"] = lib
    return out, shimmed


def target_root():
    """Cargo's target directory for this checkout (CARGO_TARGET_DIR, or
    bench-hashes/target). Executables live there, where programs run (a
    VM's shared mount may not run them)."""
    return Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "bench-hashes/target"))


def side_bench(side, commit):
    """(bench-hashes executable built against the fork at `commit`, whether
    its batch cells cannot be judged), built in the side `side` ("old" or
    "new"): a fork worktree under CACHE/side with a copy of bench-hashes
    inside, and a target directory of its own. Each step changes only what
    differs (the worktree moves only when its commit does, and then git
    rewrites only the files that differ; the copy is written file by file
    where it differs), so Cargo's own freshness decides what to rebuild,
    and a side whose code is unchanged builds nothing."""
    commit = git("rev-parse", f"{commit}^{{commit}}")
    checkout = CACHE / side / "src"
    if not checkout.exists():
        if checkout.parent.exists():
            git("worktree", "prune")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(checkout), commit)
    elif git("rev-parse", "HEAD", cwd=checkout) != commit:
        git("checkout", "--quiet", "--force", "--detach", commit, cwd=checkout)
    sources, shimmed = shimmed_sources(commit)
    # The shims, written where they differ (a moved checkout has none: the
    # forced checkout reset the files they change).
    for path, text in sources.items():
        write_if_different(checkout / path, text.encode())
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
    env = {**ENV, "CARGO_TARGET_DIR": str(target_root() / f"perf-{side}")}
    fork_version = re.search(r'(?m)^version = "([^"]+)"', (checkout / "Cargo.toml").read_text()).group(1)
    locked = re.search(r'name = "blake3-servil"\nversion = "([^"]+)"', (copy / "Cargo.lock").read_text()).group(1)
    if locked != fork_version:
        subprocess.run(["cargo", "--config", PATCH, "update", "--quiet", "-p", "blake3-servil"], cwd=copy, env=env,
                       check=True)
    out = subprocess.run(["cargo", "--config", PATCH, "build", "--release", "--message-format=json-render-diagnostics"],
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
    return exes[0], shimmed


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


def run(exe, points):
    """One run of `points` in a scratch directory; {"contender|scenario|use_case|point": 5th percentile}."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, "--contenders", ",".join(CONTENDERS), "--points", ",".join(points),
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
            assert header == ["contender", "scenario", "use_case", "point", "unit", "ns/units"], header
        elif line:
            contender, scenario, use_case, point, _unit, values = line.split("\t")
            # Each sample as measured, ns/units: exact until a ratio is printed.
            ordered = sorted(Fraction(*map(int, v.split("/"))) for v in values.split(","))
            # The statistic, and the 90th percentile, which a two-speed
            # cell's slow speed reaches (reported beside verdicts, never
            # judged: the rule's calibration is for the 5th percentile).
            cells[f"{contender}|{scenario}|{use_case}|{point}"] = (
                ordered[int(QUANTILE * len(ordered))], ordered[int(0.9 * len(ordered))])
    return cells


def ratios_of(measured, key):
    """The key's new/old ratios of 5th percentiles, one per pair that
    measured it (a pair measures a cell only while it is open, so these are
    the first pairs)."""
    return [b[key][0] / a[key][0] for a, b in measured if key in a]


def is_open(ratios, scenario):
    """Whether a cell with these pair ratios could still be called slower
    or faster: every ratio so far beyond its margin on one side."""
    margin = MARGIN[scenario]
    return all(r > 1 + margin for r in ratios) or all(r < 1 - margin for r in ratios)


def pairs(old, new, start, points, use_cases):
    """PAIRS pairs of (old run, new run) over `points`, the side that runs
    first alternating, beginning with old when `start` is even. After each
    pair, the next measures only the points with an open cell (is_open);
    when none is left, the pairs stop."""
    out = []
    for i in range(PAIRS):
        began = time.monotonic_ns()
        if (start + i) % 2 == 0:
            a = run(old, points)
            b = run(new, points)
        else:
            b = run(new, points)
            a = run(old, points)
        out.append((a, b))
        measured_points = len(points)
        still = {key.split("|")[3] for key in out[0][0]
                 if key.split("|")[2] in use_cases and is_open(ratios_of(out, key), key.split("|")[1])}
        points = [p for p in points if p in still]
        tenths = (time.monotonic_ns() - began + 50_000_000) // 100_000_000
        print(f"perf_regress: pair {start + i + 1} done in {tenths // 10}.{tenths % 10} s "
              f"({measured_points} points; {len(points)} still open)",
              file=sys.stderr, flush=True)
        if not points:
            break
    return out


def judge(measured, use_cases, contenders):
    """(slower, faster, ratio per cell, 90th-percentile ratio per cell): a
    cell is slower (faster) when every one of PAIRS pairs measured it and
    each pair's new/old ratio of 5th percentiles exceeds 1 + its scenario's
    margin (falls below 1 - it)."""
    slower, faster, ratio, slow = [], [], {}, {}
    for key in measured[0][0]:
        contender, scenario, use_case, _ = key.split("|")
        if contender not in contenders or use_case not in use_cases:
            continue
        ratios = ratios_of(measured, key)
        ratio[key] = statistics.median(ratios)
        slow[key] = statistics.median(b[key][1] / a[key][1] for a, b in measured if key in a)
        if len(ratios) == PAIRS and is_open(ratios, scenario):
            (slower if ratios[0] > 1 else faster).append(key)
    return slower, faster, ratio, slow


def compare(old_rev, new):
    """Compare the fork at `old_rev` with `new` (a commit, or None for the
    working tree). Returns the exit code."""
    old, old_shim = side_bench("old", old_rev)
    new_exe, new_shim = side_bench("new", working_tree_commit() if new is None else new)
    new_name = "the working tree" if new is None else new
    # A shimmed side's batch cells are not judged, so they are not run:
    # run, they changed the control's next cells (SHA-256 at 64 B 3-6%
    # slower beside servil f70c758's shimmed 256-byte batches, VM).
    shimmed = old_shim or new_shim
    use_cases = {"OneMessage"} if shimmed else {"OneMessage", "ManyMessages"}
    points = ONE_MESSAGE_POINTS if shimmed else POINTS
    print(f"perf_regress: {new_name} against {old_rev}, {PAIRS} alternating pairs, "
          f"use cases {', '.join(sorted(use_cases))}", file=sys.stderr, flush=True)
    measured = pairs(old, new_exe, 0, points, use_cases)

    def unreliable(measured):
        control = judge(measured, use_cases, [CONTROL])
        if control[0] or control[1]:
            print(f"perf_regress: the control ({CONTROL}, the same code on both sides) moved in "
                  f"{len(control[0]) + len(control[1])} cells: the machine's state changed within pairs. "
                  "No verdict (exit 2); run again when nothing else runs on the machine. A control that "
                  "moves on the new side run after run points at the change itself (it alters what its "
                  "cells leave behind for the next).")
            for key in sorted(control[0] + control[1]):
                print(f"  control {key}: {float(control[2][key] - 1):+.1%}")
            return True
        return False

    if unreliable(measured):
        return 2
    slower, faster, ratio, slow = judge(measured, use_cases, SUBJECTS)
    if slower:
        # The confirmation measures the slower cells' points alone.
        again = [p for p in points if p in {key.split("|")[3] for key in slower}]
        print(f"perf_regress: {len(slower)} cells slower in {PAIRS} pairs; {PAIRS} more pairs over their "
              f"{len(again)} points must agree", file=sys.stderr, flush=True)
        more = pairs(old, new_exe, PAIRS, again, use_cases)
        if unreliable(more):
            return 2
        slower2, _, ratio2, _ = judge(more, use_cases, SUBJECTS)
        confirmed = sorted(set(slower) & set(slower2))
        # Solo cells hold the change; shared cells are reported (Zooko,
        # September 26, 2026: a change that slows them has a reason worth
        # more, which its commit message names beside the cells).
        held = [key for key in confirmed if key.split("|")[1] == "solo"]
        reported = [key for key in confirmed if key not in held]

        def show(keys):
            for key in keys:
                print(f"  {key}: {float(ratio[key] - 1):+.1%}, then {float(ratio2[key] - 1):+.1%} "
                      f"(90th percentile {float(slow[key] - 1):+.1%})")

        if held:
            print(f"perf_regress: REGRESSION: {new_name} is slower than {old_rev} in {len(held)} solo cells "
                  f"(5th percentile, median of pair ratios):")
            show(held)
        if reported:
            print(f"perf_regress: {new_name} is slower than {old_rev} in {len(reported)} shared cells; "
                  "they do not hold the change: name them, their numbers, and the change's reason "
                  "in its commit message:")
            show(reported)
        if held:
            return 1
        if reported:
            print(f"perf_regress: no solo regression against {old_rev}")
            return 0
        print(f"perf_regress: the second {PAIRS} pairs did not confirm; no regression")
    for key in sorted(faster):
        print(f"  faster  {key}: {float(ratio[key] - 1):+.1%} (90th percentile {float(slow[key] - 1):+.1%})")
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
    sub.add_parser("build", help="build bench-hashes against the working tree; print the executable's path")
    args = parser.parse_args()
    # The sides are one checkout's: one invocation at a time uses them.
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"perf_regress: another perf_regress holds {CACHE / 'lock'}; run one at a time")
        if args.command == "build":
            print(side_bench("new", working_tree_commit())[0])
            return 0
        if args.command == "check":
            return compare(args.against, None)
        return compare(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
