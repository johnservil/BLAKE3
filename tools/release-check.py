#!/usr/bin/env python3
"""The release checklist's mechanical steps, read from where each gate
left its result, for one commit (CONTRIBUTING.md, "Releases").

    python3 tools/release-check.py [COMMIT]        # default: servil's tip

It runs nothing heavy. For each step it reads the evidence the gate
wrote and prints one line: PASS, FAIL, OPEN (no evidence yet, with the
command that makes it), or, each its own outcome, TIMED OUT (the gate, or
GitHub's answer here, ran out of time) and NO VERDICT (the gate could not
judge: a check whose machine was busy). Exit status 0 only when every step
passes; 1 when a step failed or is open; 2 when every other step passes
and the rest timed out or have no verdict.

- CI: a workflow_dispatch run on the commit (every job succeeded).
  Pull-request runs check a synthetic merge by default, not the head SHA
  GitHub associates with the run, so they are not release evidence.
- Long proofs: the commit's GitHub status "long proofs" (long-proofs.sh).
- Mac check: runner/results/*/verdict.json whose job names this commit as
  new_commit and the previous release as old_commit (a "check" job with
  "cells": "all").
- The logs, each named for the commit it ran on (C: the commit's first 12
  hex digits), so an earlier commit's log cannot stand in for it:
  - VM check: tmp/night/check-release-vm-C.log, from
    `cd ../bench-hashes && cargo run --release -- check --fork <this checkout> --against <previous tag> --cells all`,
    with `--old-bench <bench commit>` when the old side predates the
    bench's calls (0.8: 5be5f236, the old API's bench), as in the Mac
    job's "old_bench_commit".
  - Full mutation run: tmp/night/mutants-C.log with both final summaries
    matching their survivor and timeout lists, from
    `python3 tools/mutants/check.py` with the commit checked out.
  - check-builds: tmp/night/check-builds-C.log ending "every build passes".
  - Publish dry run: tmp/night/publish-dry-run-C.log, from the paired
    package dry-run in CONTRIBUTING.md, followed by "publish-dry-run: ok"
    only on success.
- The pages: README.md, QUALITY.md, CHANGELOG.md changed since the
  previous release (a release rewrites them for its readers).

The previous release is the nearest reachable v* tag, excluding tags on
this candidate itself. Creating the release's tag must not make the check
compare the release against itself; tags on other branches are irrelevant.
The first release has none: its checks run against the main line before
the release's changes (the commit the release's tracking issue names),
and its pages are read fresh rather than diffed.
"""

import glob
import json
import socket
import urllib.error
from pathlib import Path
import os
import re
import subprocess
import sys
import urllib.request

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
# The main checkout holds the token, the runner's jobs and results, and the
# night's logs, for a worktree too.
MAIN = subprocess.run(["git", "-C", ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, check=True).stdout.strip()
MAIN = os.path.dirname(MAIN) if MAIN.endswith("/.git") else MAIN


# A step's two outcomes besides PASS, FAIL and OPEN.
TIMED_OUT, NO_VERDICT = "timed out", "no verdict"


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout.strip()


class GitHubTimedOut(Exception):
    """GitHub's API did not answer within its 60 s: the step timed out, a
    different outcome from a failed or an open one."""


def github(path):
    token = open(os.path.join(MAIN, "ghtokenclassic.txt")).read().strip()
    request = urllib.request.Request("https://api.github.com" + path, headers={"Authorization": f"token {token}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except (TimeoutError, socket.timeout) as e:
        raise GitHubTimedOut(f"GitHub's API did not answer {path} within 60 s") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise GitHubTimedOut(f"GitHub's API did not answer {path} within 60 s") from e
        raise


def log_tail(path):
    full = os.path.join(MAIN, path)
    if not os.path.exists(full):
        return None
    return Path(full).read_text(errors="replace")[-4000:]


def log_ends_with(path, text):
    tail = log_tail(path)
    return None if tail is None else text in tail


def mutation_result(path):
    """check.py exits zero only when both survivor and timeout lists agree."""
    tail = log_tail(path)
    lines = [] if tail is None else tail.strip().splitlines()
    if len(lines) < 2:
        return None
    survivors, timeouts = lines[-2:]
    if survivors == "mutants: the list and this run differ, or a listed mutant has no reason":
        return False
    if not re.fullmatch(r"mutants: \d+ survive, each known and explained", survivors):
        return None
    if re.fullmatch(r"mutants: \d+ time out, each known \([^)]+\)", timeouts):
        return True
    if re.fullmatch(r"mutants: the time-outs differ from .+ \(no verdict on those mutants\)", timeouts):
        return NO_VERDICT
    return None


def ci_result(run):
    """Release CI tests the actual candidate, with every job executed."""
    if run["event"] != "workflow_dispatch":
        return None
    if run["status"] != "completed":
        return None
    if run["conclusion"] != "success":
        return TIMED_OUT if run["conclusion"] == "timed_out" else False
    jobs, page = [], 1
    while True:
        reply = github(f"/repos/johnservil/BLAKE3/actions/runs/{run['id']}/jobs?per_page=100&page={page}")
        jobs.extend(reply["jobs"])
        if len(jobs) == reply["total_count"]:
            break
        assert reply["jobs"] and len(jobs) < reply["total_count"], "CI jobs response is incomplete or inconsistent"
        page += 1
    if not jobs:
        return None
    if any(job["conclusion"] == "timed_out" for job in jobs):
        return TIMED_OUT
    if any(job["status"] != "completed" or job["conclusion"] in (None, "skipped") for job in jobs):
        return None
    return all(job["conclusion"] == "success" for job in jobs)


def previous_release(commit):
    """Nearest reachable release tag, excluding tags on the candidate itself."""
    tags = git("tag", "--list", "v*", "--merged", commit).split()
    tags = [tag for tag in tags if git("rev-parse", tag + "^{commit}") != commit]
    if not tags:
        return None
    matches = [arg for tag in tags for arg in ("--match", tag)]
    return git("describe", "--tags", "--abbrev=0", *matches, commit)


def main():
    commit = git("rev-parse", sys.argv[1] if len(sys.argv) > 1 else "servil")
    previous = previous_release(commit)
    results = []

    def step(name, ok, how):
        """ok: True (PASS), False (FAIL), None (OPEN), or TIMED_OUT or
        NO_VERDICT, each printed and counted as itself."""
        results.append(ok)
        label = {True: "PASS", False: "FAIL", None: "OPEN", TIMED_OUT: "TIMED OUT", NO_VERDICT: "NO VERDICT"}[ok]
        print(f"{label}  {name}" + ("" if ok is True else f": {how}"))

    try:
        runs = github(f"/repos/johnservil/BLAKE3/actions/runs?head_sha={commit}&event=workflow_dispatch&per_page=20").get("workflow_runs", [])
        ci = [r for r in runs if r["name"] == "tests"]
        result = ci_result(ci[0]) if ci else None
        step("CI, every job", result,
             "dispatch the tests workflow on the commit; every job must run and succeed" if result != TIMED_OUT else "a job reached its timeout-minutes")
    except GitHubTimedOut as e:
        step("CI, every job", TIMED_OUT, str(e))

    try:
        statuses = github(f"/repos/johnservil/BLAKE3/commits/{commit}/statuses")
        proofs = [s for s in statuses if s["context"] == "long proofs"]
        step("long proofs", (proofs[0]["state"] == "success") if proofs else None, "sh tools/verify/long-proofs.sh <commit> (overnight)")
    except GitHubTimedOut as e:
        step("long proofs", TIMED_OUT, str(e))

    mac = None
    for verdict in sorted(glob.glob(os.path.join(MAIN, "runner/results/*/verdict.json"))):
        v = json.loads(Path(verdict).read_text())
        job = os.path.join(MAIN, "runner/jobs", v["job"])
        if not os.path.exists(job):
            continue
        j = json.loads(Path(job).read_text())
        if (j.get("type") == "check" and j.get("cells") == "all"
                and j.get("new_commit") == commit
                and (previous is None or j.get("old_commit") == git("rev-parse", previous))):
            mac = v["status"]
    # The runner's verdict, each status as itself: "timed out" (its limit
    # ended the job) and "no verdict" (the machine was busy) are neither a
    # pass nor a regression.
    step("Mac check against the previous release",
         None if mac is None else TIMED_OUT if mac == "timed out" else NO_VERDICT if mac == "no verdict" else mac == "no regression",
         f"a check job (cells all) with old_commit {previous or '<the main line before this release>'} and new_commit {commit[:10]}, and old_bench_commit when the old side predates the bench's calls"
         + (f" (the last one: {mac})" if mac in ("timed out", "no verdict") else ""))

    step("VM check against the previous release", log_ends_with(f"tmp/night/check-release-vm-{commit[:12]}.log", "no cell slower"),
         f"(cd ../bench-hashes && cargo run --release -- check --fork {ROOT} --against {previous or '<the main line before this release>'} --cells all [--old-bench <bench commit>]) > tmp/night/check-release-vm-{commit[:12]}.log")
    step("full mutation run", mutation_result(f"tmp/night/mutants-{commit[:12]}.log"),
         f"python3 tools/mutants/check.py > tmp/night/mutants-{commit[:12]}.log, the commit checked out (overnight)")
    step("check-builds", log_ends_with(f"tmp/night/check-builds-{commit[:12]}.log", "every build passes"),
         f"sh tools/check-builds.sh > tmp/night/check-builds-{commit[:12]}.log")
    step("cargo publish --dry-run", log_ends_with(f"tmp/night/publish-dry-run-{commit[:12]}.log", "publish-dry-run: ok"),
         f"paired package dry-run in CONTRIBUTING.md; save its output to tmp/night/publish-dry-run-{commit[:12]}.log and append 'publish-dry-run: ok' only on success")

    if previous:
        changed = set(git("diff", "--name-only", previous, commit).split())
        for page in ["README.md", "QUALITY.md", "CHANGELOG.md"]:
            step(f"{page} rewritten since {previous}", page in changed, "a release rewrites it for its readers")
    else:
        step("pages read fresh for the first release", None, "no previous release: read README.md, QUALITY.md and CHANGELOG.md as a newcomer would, and record it in the release issue")

    failing = sum(1 for r in results if r is False or r is None)
    undecided = sum(1 for r in results if r in (TIMED_OUT, NO_VERDICT))
    if not failing and not undecided:
        print("release-check: every step passes")
    else:
        print(f"release-check: {failing} steps open or failing, {undecided} timed out or without a verdict")
    sys.exit(0 if not failing and not undecided else 1 if failing else 2)


if __name__ == "__main__":
    main()
