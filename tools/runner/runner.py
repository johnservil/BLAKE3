#!/usr/bin/env python3
"""Benchmark runner for BLAKE3-Servil, for the Mac's `benchrunner` account.

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

(bench-hashes before its git dependency on BLAKE3-Servil took `--thorough` for
a full run; later ones run in full by default and take `--quick`.)
    {"type": "check", "old_commit": "...", "new_commit": "...",
     "bench_commit": "...", "old_bench_commit": "...", "points": "3 KiB,idle 3 KiB",
     "layouts": N}

(The check's scope: "points" names the cells, as bench-hashes check --points
takes them; "probe" the same for its three-pair survey, with no verdict;
"cells": "all" every cell. From bench-hashes 0.21 the check needs one; an
older bench, given none, measures the cells whose code the change touches.
"layouts" is the links a side, a pair
each, its default check's. "old_bench_commit" builds the old side with the bench at that
commit (check --old-bench), for an old side that predates the bench's
calls: 0.8 against the main line before the throughput API. Every run's
samples file lands in the job's samples/.)

(Any job may add "powermetrics": true: the chip's CPU, GPU, and ANE power
every 500 ms beside it, in its results' powermetrics.txt. And "top": true:
the twelve busiest processes every second beside it, each sample with its
time, in its results' top.txt, to name the other programs a timing job's
busy windows count.)
    {"type": "b3sum", "fork_commit": "...", "bench_commit": "...", "quick": true}

(A b3sum job builds the contenders, bench-hashes' tools/b3sum-contenders.py
at the BLAKE3-Servil commit: official BLAKE3's b3sum, BLAKE3-Servil's as b3sum and
b3sum-1-thread, the system's sum programs; and runs bench-hashes b3sum on
the files in ~/b3sum-bench-files, made there once. The report, samples
and map land in the job's benchmark-results/. "quick" runs the small set.)
    {"type": "crowd", "fork_commit": "...", "bench_commit": "..."}

(A crowd job runs `bench-hashes crowd` with its defaults, built against the
fork commit: many programs at once, one a CPU, each with its own pool. Its
samples and report land in the job's benchmark-results/.)
    {"type": "mutants", "fork_commit": "...", "since": "..."}

(A mutants job runs BLAKE3-Servil's tools/mutants/check.py --since SINCE at the
fork commit: the mutants in the lines the change since SINCE makes, judged
against the list for this machine. Its exit is the verdict, and each run's
outcomes land in the job's mutants-last/. No timing: other programs on the
Mac can only make some mutants time out, an outcome of their own.)
    {"type": "example", "example": "scaling" | "host_lab" | "energy_probe" | "job_trace",
     "fork_commit": "...", "args": ["message", "1048576"]}
    {"type": "test", "fork_commit": "..."}

(A test job runs BLAKE3-Servil's suites natively, under cargo-nextest: the
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

The checkouts persist between jobs (CHECKOUTS: BLAKE3-Servil, and bench-hashes
beside it), moved to each job's commits with a forced checkout that
rewrites only the files that differ, so Cargo rebuilds only what changed:
BLAKE3-Servil's target directory serves test and example jobs, check's
sides (target/check/ in BLAKE3-Servil) its jobs, and bench-hashes' own build
against BLAKE3-Servil as it is (`bench-hashes build-against`) benchmark jobs.
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

FORK_URL = "https://github.com/servil-software-factory/BLAKE3.git"
BENCH_URL = "https://github.com/servil-software-factory/bench-hashes.git"
HOME = Path.home()
CHECKOUTS = HOME / "checkouts"
DONE_FILE = HOME / ".benchrunner_done"
CARGO_BIN = HOME / ".cargo" / "bin"

BENCHMARK_FLAGS = {"--all", "--quick", "--thorough"}
# blake3, blake3-mt, blake3-servil: bench-hashes' keys before September 25, 2026.
CONTENDERS = {"blake3-official", "blake3-official-mt", "blake3", "blake3-servil-st", "blake3-servil", "blake3-servil-mt", "blake3-mt",
              "sha256", "sha256-ring", "sha256-cc", "sha1dc", "sha3-256", "blake3-commonware"}
EXAMPLES = {"host_lab", "energy_probe", "job_trace", "sme_units"}
FEATURES = {"job-trace"}
DEFAULT_LIMIT = 1800
MAX_LIMIT = 7200

stopping = False

# The session of the command a job is running: its PID and start time. A
# runner stopped some other way than Ctrl-C (its terminal closed, a kill)
# leaves that session running, and the next runner would measure beside
# it: two checks of job 1939 ran at once, each the other's load
# (bench-hashes #135, November 6, 2026). The next runner ends it first.
SESSION_FILE = Path.home() / "runner-session.txt"


def started(pid):
    """A process's start time as ps prints it, or None when it is gone."""
    out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
    return out.stdout.strip() or None


def end_leftover_session():
    """End the session an earlier runner left running, if it is the one it
    recorded (the same PID and start time; this account's processes only)."""
    if not SESSION_FILE.exists():
        return
    pid, recorded = SESSION_FILE.read_text().split("\t", 1)
    if started(int(pid)) == recorded.strip():
        print(f"runner: ending the session an earlier runner left running (PID {pid})", file=sys.stderr, flush=True)
        os.killpg(int(pid), signal.SIGKILL)
    SESSION_FILE.unlink()


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
        SESSION_FILE.write_text(f"{proc.pid}\t{started(proc.pid)}\n")
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
        """BLAKE3-Servil's checkout and bench-hashes', each a clone of its own
        beside the other (neither inside the other: bench-hashes names the
        fork's checkout, `build-against DIR`, `check --fork DIR`), kept
        between jobs: (fork, bench). Anything untracked goes before each
        job, so each reads as the commit the job names (nested bench clones
        that probe examples left in BLAKE3-Servil made every build's provenance
        read dirty, job 1778); ignored paths (target/) stay."""
        CHECKOUTS.mkdir(exist_ok=True)
        fork, bench = CHECKOUTS / "fork", CHECKOUTS / "bench"
        self.sync(FORK_URL, fork, fork_commit)
        (fork / ".git/info").mkdir(exist_ok=True)
        (fork / ".git/info/exclude").write_text("/tmp/\n")
        self.run(["git", "clean", "-ffd", "--quiet"], cwd=fork)
        if bench_commit is not None:
            self.sync(BENCH_URL, bench, bench_commit)
            self.run(["git", "clean", "-ffd", "--quiet"], cwd=bench)
        return fork, bench


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
    fork, bench = run.checkouts(fork_commit, hex_commit(job, "bench_commit"))
    # bench-hashes built against the BLAKE3-Servil checkout as it is, in a side
    # directory that owns its lock and target directory (the committed lock
    # is never written); Cargo rebuilds only what changed. The one build
    # mechanism, check's (bench-hashes #54).
    exe = run.run(["cargo", "run", "--release", "--quiet", "--", "build-against", str(fork)],
                  cwd=bench, capture=True).strip().splitlines()[-1]
    # Each run from its result folder: the benchmark writes
    # benchmark-results/ relative to its working directory.
    for folder in [out] if repeat == 1 else [out / f"run-{i}" for i in range(1, repeat + 1)]:
        folder.mkdir(exist_ok=True)
        run.run([exe, *args], cwd=folder)
    return "ok"


def b3sum(job, run, work, out):
    fork_commit = hex_commit(job, "fork_commit")
    fork, bench = run.checkouts(fork_commit, hex_commit(job, "bench_commit"))
    # One contender a line, each one argument (a command may hold flags).
    contenders = [line for line in run.run([sys.executable, "tools/b3sum-contenders.py", str(fork), str(HOME / "b3sum-contenders"), fork_commit],
                                           cwd=bench, capture=True).splitlines() if "=" in line]
    exe = run.run(["cargo", "run", "--release", "--quiet", "--", "build-against", str(fork)],
                  cwd=bench, capture=True).strip().splitlines()[-1]
    quick = ["--quick"] if job.get("quick") else []
    run.run([exe, "b3sum", "--files", str(HOME / "b3sum-bench-files"), *quick, *contenders], cwd=out)
    return "ok"


def check(job, run, work, out):
    """bench-hashes check: BLAKE3-Servil at new_commit, with bench-hashes at
    bench_commit beside it (a bench with `check --fork`, October 31, 2026),
    against old_commit. check builds both
    sides itself (under BLAKE3-Servil's target/check/), relinks each per pair,
    and measures the job's scope (the module docstring). Its exit is the
    verdict: 0 no cell slower, 1 a regression, 2 no verdict (other programs
    kept the machine busy, or no scope given to a bench that needs one)."""
    old, new = hex_commit(job, "old_commit"), hex_commit(job, "new_commit")
    fork, bench = run.checkouts(new, hex_commit(job, "bench_commit"))
    cmd = ["cargo", "run", "--release", "--quiet", "--", "check", "--fork", str(fork), "--against", old, "--keep", str(out / "samples")]
    scope = [k for k in ("cells", "points", "probe") if k in job]
    assert len(scope) <= 1, "one scope: cells, points, or probe"
    assert job.get("cells", "all") == "all", "cells: \"all\""
    for k in scope:
        cmd += [f"--{k}", job[k]]
    if "layouts" in job:
        assert isinstance(job["layouts"], int) and job["layouts"] >= 2, "layouts: 2 or more"
        cmd += ["--layouts", str(job["layouts"])]
    if "old_bench_commit" in job:
        cmd += ["--old-bench", hex_commit(job, "old_bench_commit")]
    try:
        run.run(cmd, cwd=bench)
        return "probe done" if "probe" in job else "no regression"
    except subprocess.CalledProcessError as e:
        if e.returncode in (1, 2):
            return {1: "regression", 2: "no verdict"}[e.returncode]
        raise


def example(job, run, work, out):
    assert job["example"] in EXAMPLES, f"example must be one of {sorted(EXAMPLES)}"
    features = job.get("features", [])
    assert set(features) <= FEATURES, f"features must come from {sorted(FEATURES)}"
    fork, _ = run.checkouts(hex_commit(job, "fork_commit"), None)
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
    fork, _ = run.checkouts(hex_commit(job, "fork_commit"), None)
    # Under cargo-nextest, each test in its own process; the doc examples
    # under cargo test --doc.
    run.run(["cargo", "nextest", "run", "--release", "--lib"], cwd=fork)
    run.run(["cargo", "test", "--release", "--doc"], cwd=fork)
    run.run(["cargo", "nextest", "run", "--release", "--tests"], cwd=fork)
    run.run(["cargo", "nextest", "run", "--release", "--manifest-path", "test_vectors/Cargo.toml"], cwd=fork)
    return "passed"


def mutants(job, run, work, out):
    """tools/mutants/check.py --since SINCE at fork_commit. Its exit is the
    verdict: 0 no new survivor, 1 a survivor differs from its list, 2 only
    the time-outs differ (no verdict on those)."""
    fork, _ = run.checkouts(hex_commit(job, "fork_commit"), None)
    try:
        run.run([sys.executable, "tools/mutants/check.py", "--since", hex_commit(job, "since")], cwd=fork)
        verdict = "no new survivor"
    except subprocess.CalledProcessError as e:
        if e.returncode not in (1, 2):
            raise
        # A traceback exits 1 too: a crash, never a verdict (job 1865's
        # preexec_fn failure read as "a survivor differs from its list").
        run.log.flush()
        text = Path(run.log.name).read_text()
        if "Traceback (most recent call last):" in text[text.rfind("$ (cd"):]:
            raise RuntimeError("tools/mutants/check.py stopped with a traceback (its log)") from e
        verdict = {1: "a survivor differs from its list", 2: "only time-outs differ"}[e.returncode]
    last = Path(fork) / "tmp" / "mutants-last"
    if last.is_dir():
        shutil.copytree(last, out / "mutants-last", dirs_exist_ok=True)
    return verdict


def crowd(job, run, work, out):
    fork, bench = run.checkouts(hex_commit(job, "fork_commit"), hex_commit(job, "bench_commit"))
    exe = run.run(["cargo", "run", "--release", "--quiet", "--", "build-against", str(fork)],
                  cwd=bench, capture=True).strip().splitlines()[-1]
    run.run([exe, "crowd"], cwd=out)
    return "ok"


# The jobs that time (README, "Running a job"): on mains power alone. On
# battery the Mac runs another machine's speeds (more calls on the
# efficiency cores, job 351), so such a job is refused before it runs, and
# one whose power leaves mains on the way says so in its status.
TIMED = {"benchmark", "b3sum", "crowd", "example", "check"}

HANDLERS = {"benchmark": benchmark, "b3sum": b3sum, "check": check, "crowd": crowd, "mutants": mutants, "example": example, "test": test}


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

# The busiest processes beside a job ("top": true), every second, stopping
# by itself after MAX_LIMIT; top's own share shows in its samples.
TOP = ["/usr/bin/top", "-l", "7200", "-s", "1", "-o", "cpu", "-n", "12", "-stats", "pid,command,cpu,time,user"]
assert 7200 * 1 >= MAX_LIMIT


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
            on_mains = lambda state: state.startswith("AC Power")
            assert job["type"] not in TIMED or on_mains(verdict["power at start"]), \
                f"a {job['type']} job times, so it runs on mains power alone; the Mac is on {verdict['power at start']}: plug it in and queue the job again"
            limit = job.get("time_limit_seconds", DEFAULT_LIMIT)
            assert isinstance(limit, int) and 0 < limit <= MAX_LIMIT, f"time_limit_seconds must be 1-{MAX_LIMIT}"
            meters = {}
            for name, command in [("powermetrics", POWERMETRICS), ("top", TOP)]:
                assert isinstance(job.get(name, False), bool), f"{name}: true or false"
                if job.get(name):
                    meters[name] = subprocess.Popen(command, stdout=open(out / f"{name}.txt", "w"), stderr=log)
            try:
                with tempfile.TemporaryDirectory(dir=HOME / "tmp") as work:
                    verdict["status"] = HANDLERS[job["type"]](job, Job(log, limit), Path(work), out)
            finally:
                for name, meter in meters.items():
                    # sudo passes the signal on to powermetrics, which ends its
                    # last sample and exits, as top does. A meter that outlasts
                    # 10 s is killed and noted beside the verdict: it is no
                    # reason to replace the job's own status (it once could,
                    # with "error", from wait's TimeoutExpired).
                    meter.send_signal(signal.SIGINT)
                    try:
                        meter.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        meter.kill()
                        meter.wait()
                        verdict[name] = "did not exit within 10 s of SIGINT; killed (its last samples may be missing)"
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
    if verdict.get("type") in TIMED and verdict.get("status") not in (None, "error") and not verdict["power at end"].startswith("AC Power"):
        verdict["status"] = f"power left mains during the job ({verdict['status']}): its results are no evidence of speed"
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
    end_leftover_session()
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
