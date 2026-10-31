#!/usr/bin/env python3
"""Benchmark runner for the servil fork, for the Mac's `benchrunner` account.

    sudo -u benchrunner -H python3 runner.py EXCHANGE

EXCHANGE holds `jobs/` (JSON job files, read here) and `results/` (written
here, one folder per job). Code comes from GitHub alone, at the commits a
job names. Done jobs are listed in ~/.benchrunner_done (the runner's home).
Ctrl-C stops the runner after the current job.

Jobs (commits as 7-40 hex digits; time_limit_seconds optional, default 1800,
at most 7200):

    {"type": "benchmark", "fork_commit": "...", "bench_commit": "...",
     "flags": ["--all", "--quick"], "contenders": [...], "points": [...],
     "rounds": N, "repeat": N}

(bench-hashes before its git dependency on the fork took `--thorough` for
a full run; later ones run in full by default and take `--quick`.)
    {"type": "check", "old_commit": "...", "new_commit": "...",
     "bench_commit": "...", "old_bench_commit": "...", "cells": "all", "pairs": N}

(bench-hashes check measures the cells whose code the change touches, or
every cell with "cells": "all"; "pairs" is the pairs of runs, its default
check's. "old_bench_commit" builds the old side with the bench at that
commit (check --old-bench), for an old side that predates the bench's
calls: 0.8 against the main line before the performance API. Every run's
samples file lands in the job's samples/.)

(Any job may add "powermetrics": true: the chip's CPU, GPU, and ANE power
every 500 ms beside it, in its results' powermetrics.txt.)
    {"type": "b3sum", "fork_commit": "...", "bench_commit": "...", "quick": true}

(A b3sum job builds the contenders, bench-hashes' tools/b3sum-contenders.py
at the fork commit: official BLAKE3's b3sum, the fork's as b3sum and
b3sum-1-thread, the system's sum programs; and runs bench-hashes b3sum on
the files in ~/b3sum-bench-files, made there once. The report, samples
and map land in the job's benchmark-results/. "quick" runs the small set.)
    {"type": "example", "example": "scaling" | "host_lab" | "energy_probe" | "job_trace",
     "fork_commit": "...", "args": ["message", "1048576"]}
    {"type": "test", "fork_commit": "..."}

(A test job runs the fork's suites natively, under cargo-nextest: the
library tests, the doc tests, the
integration tests in tests/ (the planned API's contract), and the
official vectors.)

(A benchmark job may add "train": [W_MS, S] and "nonstop_visits": N, the
continuous-load proposal's tests: bench-hashes' --train and
--nonstop-visits.)

(A benchmark job with "repeat": N runs the benchmark N times back to
back, run i's files in run-i/, for calibrations.)

Each result folder holds runner.log (every command and its output),
verdict.json, and the job's own files: the benchmark's report, graph,
and samples; host_lab's report.

The checkouts persist between jobs (CHECKOUTS: the fork, bench-hashes
inside it), moved to each job's commits with a forced checkout that
rewrites only the files that differ, so Cargo rebuilds only what changed:
the fork's target directory serves test and example jobs, check's
sides (target/check/ in the fork) its jobs, and bench-hashes' own build
against the fork as it is (`bench-hashes build-against`) benchmark jobs.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FORK_URL = "https://github.com/johnservil/BLAKE3.git"
BENCH_URL = "https://github.com/johnservil/bench-hashes.git"
HOME = Path.home()
CHECKOUTS = HOME / "checkouts"
DONE_FILE = HOME / ".benchrunner_done"
CARGO_BIN = HOME / ".cargo" / "bin"

BENCHMARK_FLAGS = {"--all", "--quick", "--thorough"}
# bench-hashes depends on the fork's git repository at a pinned commit; this
# patch builds it against the fork checkout enclosing it (`..`), the one the
# job names. Older bench-hashes, with a path dependency on `..`, ignore it.
PATCH = 'patch."https://github.com/johnservil/BLAKE3".blake3-servil.path=".."'
# blake3, blake3-mt, blake3-servil: bench-hashes' keys before September 25, 2026.
CONTENDERS = {"blake3-official", "blake3-official-mt", "blake3", "blake3-servil-st", "blake3-servil", "blake3-servil-mt", "blake3-mt",
              "sha256", "sha256-ring", "sha256-cc", "sha1dc", "sha3-256", "blake3-commonware"}
EXAMPLES = {"host_lab", "energy_probe", "job_trace", "sme_units"}
FEATURES = {"job-trace"}
DEFAULT_LIMIT = 1800
MAX_LIMIT = 7200

stopping = False


def on_signal(signum, _frame):
    global stopping
    stopping = True
    print("runner: stopping after the current job", file=sys.stderr, flush=True)


def hex_commit(job, key):
    value = job[key]
    assert isinstance(value, str) and 7 <= len(value) <= 40 and all(c in "0123456789abcdef" for c in value), \
        f"{key} must be 7-40 lowercase hex digits, got {value!r}"
    return value


def environment():
    env = {"PATH": f"{CARGO_BIN}:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(HOME),
           "LANG": "en_US.UTF-8", "TMPDIR": str(HOME / "tmp") + "/"}
    return env


class TimedOut(Exception):
    """The job's time limit ended it: its own verdict, "timed out", never
    "error" (a crash) or a handler's answer. A limit reached says nothing
    about what the job would have found."""


class Job:
    """One job's log and deadline; every command runs through `run`. The
    deadline is the job's one clock (wait_for.py keeps none over it)."""

    def __init__(self, log, limit):
        self.log = log
        self.limit = limit
        self.deadline = time.monotonic() + limit

    def run(self, cmd, cwd, capture=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimedOut(f"the job's {self.limit} s limit passed before: {' '.join(cmd)}")
        print(f"$ (cd {cwd} && {' '.join(cmd)})", file=self.log, flush=True)
        # A new session, so Ctrl-C in the runner's terminal reaches the
        # runner alone and the current job finishes.
        proc = subprocess.Popen(cmd, cwd=cwd, env=environment(), start_new_session=True,
                                stdout=subprocess.PIPE if capture else self.log, stderr=self.log, text=True)
        try:
            stdout, _ = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            # The whole session, not the command alone: cargo's children (a
            # benchmark it runs) would otherwise go on into the next job.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise TimedOut(f"the job's {self.limit} s limit ended: {' '.join(cmd)}")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return stdout

    def sync(self, url, dest, commit):
        """`dest`, a clone of `url` kept between jobs, at `commit`."""
        if not (dest / ".git").exists():
            self.run(["git", "clone", "--quiet", url, str(dest)], cwd=dest.parent)
        else:
            self.run(["git", "fetch", "--quiet", "--prune", "origin"], cwd=dest)
        self.run(["git", "checkout", "--quiet", "--force", "--detach", commit], cwd=dest)
        full = self.run(["git", "rev-parse", "HEAD"], cwd=dest, capture=True).strip()
        assert full.startswith(commit), f"{url}: asked for {commit}, got {full}"

    def checkouts(self, fork_commit, bench_commit):
        """The fork with bench-hashes nested inside it, where bench-hashes'
        path dependency (`..`) expects the fork; both kept between jobs.
        The fork's status leaves out bench-hashes and check's target/,
        as the maintainers' checkout does, and anything else untracked is
        removed before each job."""
        CHECKOUTS.mkdir(exist_ok=True)
        fork = CHECKOUTS / "fork"
        self.sync(FORK_URL, fork, fork_commit)
        (fork / ".git/info").mkdir(exist_ok=True)
        # The nested bench, the scratch folder, and build-against's copy of
        # the bench beside it (check.rs: a nested bench builds from
        # .bench-hashes-build) are the runner's, not the fork's.
        (fork / ".git/info/exclude").write_text("/bench-hashes/\n/tmp/\n/.bench-hashes-build/\n")
        # Anything else untracked goes, so the fork reads as the commit the
        # job names: three nested bench clones that probe examples left in
        # September and October made every build's provenance read dirty
        # (job 1778). Ignored and excluded paths stay (target/, the above).
        self.run(["git", "clean", "-ffd", "--quiet"], cwd=fork)
        if bench_commit is not None:
            self.sync(BENCH_URL, fork / "bench-hashes", bench_commit)
        return fork


def benchmark(job, run, work, out):
    flags = job.get("flags", [])
    assert set(flags) <= BENCHMARK_FLAGS, f"flags must come from {sorted(BENCHMARK_FLAGS)}"
    args = list(flags)
    if "contenders" in job:
        assert job["contenders"] and set(job["contenders"]) <= CONTENDERS, \
            f"contenders must come from {sorted(CONTENDERS)}"
        args += ["--contenders", ",".join(job["contenders"])]
    if "points" in job:
        points = job["points"]
        assert points and all(isinstance(p, str) and p.replace(" ", "").isalnum() for p in points), \
            "points are labels such as \"64 B\" or \"1024\""
        args += ["--points", ",".join(points)]
    if "rounds" in job:
        assert isinstance(job["rounds"], int) and job["rounds"] > 0, "rounds must be a positive integer"
        args += ["--rounds", str(job["rounds"])]
    # The continuous-load proposal's tests (bench-hashes FROZEN.md, October
    # 31, 2026): a nonstop visit's warm-up and samples, and a cell's visits.
    if "train" in job:
        train = job["train"]
        assert isinstance(train, list) and len(train) == 2 and all(isinstance(x, int) and x >= 0 for x in train) and train[1] > 0, \
            "train is [warm-up ms, samples a visit]"
        args += ["--train", f"{train[0]},{train[1]}"]
    if "nonstop_visits" in job:
        assert isinstance(job["nonstop_visits"], int) and job["nonstop_visits"] > 0, "nonstop_visits must be a positive integer"
        args += ["--nonstop-visits", str(job["nonstop_visits"])]
    repeat = job.get("repeat", 1)
    assert isinstance(repeat, int) and 1 <= repeat <= 100, "repeat must be 1-100"
    fork_commit = hex_commit(job, "fork_commit")
    fork = run.checkouts(fork_commit, hex_commit(job, "bench_commit"))
    # bench-hashes built against the fork checkout as it is, in a side
    # directory that owns its lock and target directory (the committed lock
    # is never written); Cargo rebuilds only what changed. The one build
    # mechanism, check's (bench-hashes #54).
    exe = run.run(["cargo", "run", "--release", "--quiet", "--", "build-against", str(fork)],
                  cwd=fork / "bench-hashes", capture=True).strip().splitlines()[-1]
    # Each run from its result folder: the benchmark writes
    # benchmark-results/ relative to its working directory.
    for folder in [out] if repeat == 1 else [out / f"run-{i}" for i in range(1, repeat + 1)]:
        folder.mkdir(exist_ok=True)
        run.run([exe, *args], cwd=folder)
    return "ok"


def b3sum(job, run, work, out):
    fork_commit = hex_commit(job, "fork_commit")
    fork = run.checkouts(fork_commit, hex_commit(job, "bench_commit"))
    # One contender a line, each one argument (a command may hold flags).
    contenders = [line for line in run.run([sys.executable, "tools/b3sum-contenders.py", str(fork), str(HOME / "b3sum-contenders"), fork_commit],
                                           cwd=fork / "bench-hashes", capture=True).splitlines() if "=" in line]
    exe = run.run(["cargo", "run", "--release", "--quiet", "--", "build-against", str(fork)],
                  cwd=fork / "bench-hashes", capture=True).strip().splitlines()[-1]
    quick = ["--quick"] if job.get("quick") else []
    run.run([exe, "b3sum", "--files", str(HOME / "b3sum-bench-files"), *quick, *contenders], cwd=out)
    return "ok"


def check(job, run, work, out):
    """bench-hashes check: the fork at new_commit, with bench-hashes at
    bench_commit nested inside it, against old_commit. check builds both
    sides itself (under the fork's target/check/), relinks each per pair,
    and measures the cells whose code the change touches ("cells": "all"
    measures every one). Its exit is the verdict: 0 no cell slower, 1 a
    regression, 2 no verdict (other programs kept the machine busy)."""
    old, new = hex_commit(job, "old_commit"), hex_commit(job, "new_commit")
    fork = run.checkouts(new, hex_commit(job, "bench_commit"))
    cmd = ["cargo", "run", "--release", "--quiet", "--", "check", "--against", old, "--keep", str(out / "samples")]
    if job.get("cells") == "all":
        cmd += ["--cells", "all"]
    elif "cells" in job:
        raise AssertionError("cells: \"all\", or left out (the cells the change touches)")
    if "pairs" in job:
        assert isinstance(job["pairs"], int) and job["pairs"] >= 3, "pairs: 3 or more"
        cmd += ["--pairs", str(job["pairs"])]
    if "old_bench_commit" in job:
        cmd += ["--old-bench", hex_commit(job, "old_bench_commit")]
    try:
        run.run(cmd, cwd=fork / "bench-hashes")
        return "no regression"
    except subprocess.CalledProcessError as e:
        if e.returncode in (1, 2):
            return {1: "regression", 2: "no verdict"}[e.returncode]
        raise


def example(job, run, work, out):
    assert job["example"] in EXAMPLES, f"example must be one of {sorted(EXAMPLES)}"
    features = job.get("features", [])
    assert set(features) <= FEATURES, f"features must come from {sorted(FEATURES)}"
    fork = run.checkouts(hex_commit(job, "fork_commit"), None)
    cmd = ["cargo", "build", "--release", "--example", job["example"]]
    if features:
        cmd += ["--features", ",".join(features)]
    run.run(cmd, cwd=fork)
    # An example's arguments: a few short words or numbers, nothing a shell
    # would read.
    args = job.get("args", [])
    assert isinstance(args, list) and len(args) <= 8 and all(isinstance(a, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", a) for a in args), \
        "args: at most 8 words of letters, digits, _ . -"
    # host_lab writes its report to the working directory.
    run.run([str(fork / "target" / "release" / "examples" / job["example"]), *args], cwd=out)
    return "ok"


def test(job, run, work, out):
    fork = run.checkouts(hex_commit(job, "fork_commit"), None)
    # Under cargo-nextest, each test in its own process; the doc examples
    # under cargo test --doc.
    run.run(["cargo", "nextest", "run", "--release", "--lib"], cwd=fork)
    run.run(["cargo", "test", "--release", "--doc"], cwd=fork)
    run.run(["cargo", "nextest", "run", "--release", "--tests"], cwd=fork)
    run.run(["cargo", "nextest", "run", "--release", "--manifest-path", "test_vectors/Cargo.toml"], cwd=fork)
    return "passed"


HANDLERS = {"benchmark": benchmark, "b3sum": b3sum, "check": check, "example": example, "test": test}


def power():
    """The Mac's power state in pmset's terms: the source ("AC Power",
    "Battery Power") and its charge, then the power mode setting in use
    (lowpowermode, or powermode 0 automatic, 1 low, 2 high). On battery a
    thread that sleeps between calls moved to the efficiency cores at
    about 1 GHz (job 351), so every job records it."""
    def pmset(*args):
        return subprocess.run(["pmset", *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
    batt = pmset("-g", "batt").splitlines()
    source = batt[0].removeprefix("Now drawing from ").strip("'") if batt else "unreported"
    charge = next((word.rstrip(";") for line in batt[1:] for word in line.split() if word.rstrip(";").endswith("%")), None)
    modes = [" ".join(line.split()) for line in pmset("-g").splitlines() if line.split()[:1] in (["powermode"], ["lowpowermode"])]
    return ", ".join([source] + ([charge] if charge else []) + modes)


# The chip's power beside a job ("powermetrics": true), from the one command
# line /etc/sudoers.d/benchrunner-powermetrics lets this user run as root:
# CPU, GPU, and ANE power every 500 ms, stopping by itself after MAX_LIMIT,
# so a runner that dies leaves no sampler running.
POWERMETRICS = ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power", "-i", "500", "-n", "14400"]
assert 14400 * 0.5 >= MAX_LIMIT


def process(path, results):
    out = results / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir()
    print(f"runner: {path.name}: started, log in {out / 'runner.log'}", file=sys.stderr, flush=True)
    verdict = {"job": path.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "power at start": power()}
    with open(out / "runner.log", "w") as log:
        try:
            job = json.loads(path.read_text())
            verdict["type"] = job.get("type")
            assert job.get("type") in HANDLERS, f"type must be one of {sorted(HANDLERS)}"
            limit = job.get("time_limit_seconds", DEFAULT_LIMIT)
            assert isinstance(limit, int) and 0 < limit <= MAX_LIMIT, f"time_limit_seconds must be 1-{MAX_LIMIT}"
            assert isinstance(job.get("powermetrics", False), bool), "powermetrics: true or false"
            meter = subprocess.Popen(POWERMETRICS, stdout=open(out / "powermetrics.txt", "w"), stderr=log) \
                if job.get("powermetrics") else None
            try:
                with tempfile.TemporaryDirectory(dir=HOME / "tmp") as work:
                    verdict["status"] = HANDLERS[job["type"]](job, Job(log, limit), Path(work), out)
            finally:
                if meter is not None:
                    # sudo passes the signal on to powermetrics, which ends its
                    # last sample and exits. A meter that outlasts 10 s is
                    # killed and noted beside the verdict: it is no reason to
                    # replace the job's own status (it once could, with
                    # "error", from wait's TimeoutExpired).
                    meter.send_signal(signal.SIGINT)
                    try:
                        meter.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        meter.kill()
                        meter.wait()
                        verdict["powermetrics"] = "did not exit within 10 s of SIGINT; killed (its last samples may be missing)"
        except TimedOut as e:
            verdict["status"] = "timed out"
            verdict["timed out"] = str(e)
            print(f"runner: timed out: {e}", file=log, flush=True)
        except Exception as e:
            verdict["status"] = "error"
            verdict["error"] = f"{type(e).__name__}: {e}"
            print(f"runner: {verdict['error']}", file=log, flush=True)
    verdict["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    verdict["power at end"] = power()
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"runner: {path.name}: {verdict['status']} -> {out}", file=sys.stderr, flush=True)


def main():
    assert len(sys.argv) == 2, "usage: runner.py EXCHANGE (the folder holding jobs/ and results/)"
    exchange = Path(sys.argv[1])
    jobs, results = exchange / "jobs", exchange / "results"
    assert jobs.is_dir() and results.is_dir(), f"{exchange} must hold jobs/ and results/"
    assert (CARGO_BIN / "cargo").exists(), f"no cargo at {CARGO_BIN}: install rustup for this account"
    (HOME / "tmp").mkdir(exist_ok=True)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    print(f"runner: watching {jobs}; Ctrl-C stops after the current job", file=sys.stderr, flush=True)
    while not stopping:
        pending = sorted(p for p in jobs.glob("*.json") if p.name not in done)
        if not pending:
            time.sleep(2)
            continue
        process(pending[0], results)
        done.add(pending[0].name)
        DONE_FILE.write_text("".join(f"{name}\n" for name in sorted(done)))


if __name__ == "__main__":
    main()
