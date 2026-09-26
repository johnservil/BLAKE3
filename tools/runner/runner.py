#!/usr/bin/env python3
"""Benchmark runner for the servil fork, for the Mac's `benchrunner` account.

    sudo -u benchrunner -H pypy3 runner.py EXCHANGE

EXCHANGE holds `jobs/` (JSON job files, read here) and `results/` (written
here, one folder per job). Code comes from GitHub alone, at the commits a
job names. Done jobs are listed in ~/.benchrunner_done (the runner's home).
Ctrl-C stops the runner after the current job.

Jobs (commits as 7-40 hex digits; time_limit_seconds optional, default 1800,
at most 7200):

    {"type": "benchmark", "fork_commit": "...", "bench_commit": "...",
     "flags": ["--all", "--quick"], "contenders": [...], "points": [...],
     "rounds": N, "trace_clocks": true}

(bench-hashes before its git dependency on the fork took `--thorough` for
a full run; later ones run in full by default and take `--quick`.)
    {"type": "perf_regress", "old_commit": "...", "new_commit": "...",
     "bench_commit": "..."}
    {"type": "example", "example": "scaling" | "host_lab", "fork_commit": "...",
     "features": ["no_sme2"]}

Each result folder holds runner.log (every command and its output),
verdict.json, and the job's own files: the benchmark's report, graph,
samples, and trace; host_lab's report.
"""
import json
import os
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
DONE_FILE = HOME / ".benchrunner_done"
CARGO_BIN = HOME / ".cargo" / "bin"

BENCHMARK_FLAGS = {"--all", "--quick", "--thorough"}
# bench-hashes depends on the fork's git repository at a pinned commit; this
# patch builds it against the fork checkout enclosing it (`..`), the one the
# job names. Older bench-hashes, with a path dependency on `..`, ignore it.
PATCH = 'patch."https://github.com/johnservil/BLAKE3".blake3-servil.path=".."'
# blake3, blake3-mt, blake3-servil: bench-hashes' keys before September 25, 2026.
CONTENDERS = {"blake3-official", "blake3-official-mt", "blake3", "ab-blake3", "blake3-commonware", "blake3-servil-st", "blake3-servil", "blake3-servil-mt", "blake3-mt",
              "sha256", "sha256-ring", "sha256-cc", "sha1dc"}
EXAMPLES = {"scaling", "host_lab"}
FEATURES = {"no_sme2", "pure"}
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


class Job:
    """One job's log and deadline; every command runs through `run`."""

    def __init__(self, log, limit):
        self.log = log
        self.deadline = time.monotonic() + limit

    def run(self, cmd, cwd, capture=False):
        remaining = self.deadline - time.monotonic()
        assert remaining > 0, "time limit reached"
        print(f"$ (cd {cwd} && {' '.join(cmd)})", file=self.log, flush=True)
        # A new session, so Ctrl-C in the runner's terminal reaches the
        # runner alone and the current job finishes.
        proc = subprocess.run(cmd, cwd=cwd, env=environment(), timeout=remaining, start_new_session=True,
                              stdout=subprocess.PIPE if capture else self.log, stderr=self.log, text=True)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return proc.stdout

    def clone(self, url, dest, commit):
        self.run(["git", "clone", "--quiet", url, str(dest)], cwd=dest.parent)
        self.run(["git", "checkout", "--quiet", "--detach", commit], cwd=dest)
        full = self.run(["git", "rev-parse", "HEAD"], cwd=dest, capture=True).strip()
        assert full.startswith(commit), f"{url}: asked for {commit}, got {full}"

    def checkouts(self, work, fork_commit, bench_commit):
        """The fork with bench-hashes nested inside it, where bench-hashes'
        path dependency (`..`) expects the fork."""
        fork = work / "fork"
        self.clone(FORK_URL, fork, fork_commit)
        if bench_commit is not None:
            self.clone(BENCH_URL, fork / "bench-hashes", bench_commit)
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
    if job.get("trace_clocks"):
        args += ["--trace-clocks", str(out / "trace.csv")]
    fork = run.checkouts(work, hex_commit(job, "fork_commit"), hex_commit(job, "bench_commit"))
    bench = fork / "bench-hashes"
    messages = run.run(["cargo", "--config", PATCH, "build", "--release", "--message-format=json-render-diagnostics"],
                       cwd=bench, capture=True)
    exes = [m["executable"] for m in map(json.loads, messages.splitlines())
            if m.get("reason") == "compiler-artifact" and m.get("executable")
            and m["target"]["name"] == "bench-hashes"]
    assert len(exes) == 1, f"expected the bench-hashes executable, found {exes}"
    # Run from the result folder: the benchmark writes benchmark-results/
    # relative to its working directory.
    run.run([exes[0], *args], cwd=out)
    return "ok"


def perf_regress(job, run, work, out):
    old, new = hex_commit(job, "old_commit"), hex_commit(job, "new_commit")
    fork = run.checkouts(work, new, hex_commit(job, "bench_commit"))
    try:
        run.run([sys.executable, "tools/perf_regress.py", "compare", old, new], cwd=fork)
        return "no regression"
    except subprocess.CalledProcessError as e:
        if e.returncode in (1, 2):
            return {1: "regression", 2: "no verdict"}[e.returncode]
        raise


def example(job, run, work, out):
    assert job["example"] in EXAMPLES, f"example must be one of {sorted(EXAMPLES)}"
    features = job.get("features", [])
    assert set(features) <= FEATURES, f"features must come from {sorted(FEATURES)}"
    fork = run.checkouts(work, hex_commit(job, "fork_commit"), None)
    cmd = ["cargo", "build", "--release", "--example", job["example"]]
    if features:
        cmd += ["--features", ",".join(features)]
    run.run(cmd, cwd=fork)
    # host_lab writes its report to the working directory.
    run.run([str(fork / "target" / "release" / "examples" / job["example"])], cwd=out)
    return "ok"


HANDLERS = {"benchmark": benchmark, "perf_regress": perf_regress, "example": example}


def process(path, results):
    out = results / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir()
    print(f"runner: {path.name}: started, log in {out / 'runner.log'}", file=sys.stderr, flush=True)
    verdict = {"job": path.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(out / "runner.log", "w") as log:
        try:
            job = json.loads(path.read_text())
            verdict["type"] = job.get("type")
            assert job.get("type") in HANDLERS, f"type must be one of {sorted(HANDLERS)}"
            limit = job.get("time_limit_seconds", DEFAULT_LIMIT)
            assert isinstance(limit, int) and 0 < limit <= MAX_LIMIT, f"time_limit_seconds must be 1-{MAX_LIMIT}"
            with tempfile.TemporaryDirectory(dir=HOME / "tmp") as work:
                verdict["status"] = HANDLERS[job["type"]](job, Job(log, limit), Path(work), out)
        except Exception as e:
            verdict["status"] = "error"
            verdict["error"] = f"{type(e).__name__}: {e}"
            print(f"runner: {verdict['error']}", file=log, flush=True)
    verdict["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
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
