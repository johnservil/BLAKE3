#!/usr/bin/env python3
"""Performance-regression check for blake3-servil, built on bench-hashes.

    python3 tools/perf_regress.py check     # exit 0 pass, 1 regression, 2 no verdict
    python3 tools/perf_regress.py record    # this machine's baseline, 12 runs
    python3 tools/perf_regress.py record --from A.tsv ...   # ... from saved runs
    python3 tools/perf_regress.py compare --base A.tsv ... --new B.tsv ...

`check` runs bench-hashes (the checkout at ./bench-hashes) with the
contenders blake3 (crates.io, untouched by this fork: the control),
blake3-servil, and blake3-servil-mt, and compares every cell of both use
cases with perf-baselines/<machine>.json. `record` writes that file; review
and commit it. Recording is an explicit act, as for golden vectors: nothing
here replaces a baseline on its own. `compare` judges saved samples files
the same way (tools/perf_bisect.py uses it).

How the rule was chosen (September 2026, 16-vCPU VM on an M4 Max, 16 runs
of one commit; NOTES-sme2-bench.md has the numbers):

* The statistic is a cell's 5th percentile per run. A cell's median can
  move 1.8x between runs of one commit when the host puts two vCPUs on one
  SME unit; a low quantile moves when the code does.
* Runs are the unit of variation. Whole runs sit in a slow state for some
  cells (a quarter of runs, 20-45% slower in the SME2 batch cells), which
  no interval computed inside one run can see. So a baseline is RECORD_RUNS
  separate runs, and it keeps each cell's slowest run.
* A cell is slower when every new run's 5th percentile exceeds the
  baseline's slowest by more than SLOWER_BY. CHECK_RUNS runs decide; when
  any cell is slower, one more run must agree.
* The control gates comparability and rescales nothing: its speed does not
  track the fork's (it uses neither SME2 nor the pool). A control more
  than CONTROL_TOLERANCE away from the baseline's means another machine or
  state: no verdict.
* A different machine or toolchain (CPU brand and identity, core count,
  OS, compiler, build target and features, kernel platform per contender)
  also gives no verdict.

Measured with these settings: 0.13% of checks on unchanged code report a
regression; a cell 10% slower is caught 66% of the time, 20% slower 88%,
50% slower 97%. Exit 2 (no verdict) is not a failure: the pre-commit hook
and CI pass it with a warning that says what to record.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "perf-baselines"
CONTROL = "blake3"
SUBJECTS = ["blake3-servil", "blake3-servil-mt"]
CONTENDERS = [CONTROL] + SUBJECTS
QUANTILE = 0.05
RECORD_RUNS = 12
CHECK_RUNS = 2
SLOWER_BY = 0.05
CONTROL_TOLERANCE = 0.07
IDENTITY_KEYS = ["cpu type", "cpu count", "os type", "cpu identity", "rust compiler",
                 "build target", "target features"]


def build_bench(bench):
    """Build bench-hashes from its own directory, where its .cargo/config.toml
    (target-cpu=native) applies, and return the executable's path."""
    out = subprocess.run(["cargo", "build", "--release", "--message-format=json-render-diagnostics"],
                         cwd=bench, check=True, stdout=subprocess.PIPE, text=True).stdout
    exes = [m["executable"] for m in map(json.loads, out.splitlines())
            if m.get("reason") == "compiler-artifact" and m.get("executable")
            and m["target"]["name"] == "bench-hashes"]
    assert len(exes) == 1, f"expected the bench-hashes executable, found {exes}"
    return exes[0]


def run_bench(exe):
    """One run of CONTENDERS in a scratch directory (the machine's committed
    records in bench-hashes/benchmark-results stay as they are); returns
    the samples file's text."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [exe, "--contenders", ",".join(CONTENDERS)]
        print("perf_regress: " + " ".join(cmd), file=sys.stderr, flush=True)
        subprocess.run(cmd, cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        found = list(Path(tmp).glob("benchmark-results/*/bench-hashes.duo.samples.tsv"))
        assert len(found) == 1, f"expected one samples file, found {found}"
        return found[0].read_text()


def parse(text):
    """One run: (metadata, {"contender|use_case|point": 5th percentile})."""
    meta, cells, header = {}, {}, None
    for line in text.splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            meta[key] = value
        elif header is None:
            header = line.split("\t")
            assert header == ["contender", "use_case", "point", "unit", "ps_per_unit"], header
        elif line:
            contender, use_case, point, _unit, values = line.split("\t")
            ordered = sorted(int(v) for v in values.split(","))
            cells[f"{contender}|{use_case}|{point}"] = ordered[int(QUANTILE * len(ordered))]
    assert header is not None, "not a bench-hashes samples file"
    return meta, cells


def identity(meta):
    ident = {key: meta.get(key, "") for key in IDENTITY_KEYS}
    ident.update({k: v for k, v in meta.items() if k.startswith("kernel platform ")})
    return ident


def machine_name(meta):
    import hashlib
    digest = hashlib.sha256(json.dumps(identity(meta), sort_keys=True).encode()).hexdigest()[:12]
    brand = re.sub(r"[^A-Za-z0-9]+", "", meta.get("cpu type", "cpu"))
    return f"{brand}.{meta.get('cpu count', '')}cpu.{digest}"


def baseline_from(runs):
    """A baseline from parsed runs: identity, provenance, and per cell the
    5th percentile of every run."""
    metas = [m for m, _ in runs]
    ids = [identity(m) for m in metas]
    assert all(i == ids[0] for i in ids), "baseline runs from different machines or builds"
    return {"identity": ids[0], "runs": len(runs),
            "fork": metas[0].get("blake3-servil source", ""),
            "recorded": [m.get("timestamp", "") for m in metas],
            "cells": {k: [c[k] for _, c in runs] for k in runs[0][1]}}


def judge(base, runs, use_cases):
    """(verdict, report lines, slower cells) for parsed new runs against a
    baseline. verdict: 'ok', 'slower', or 'incomparable'."""
    lines = []
    for meta, _ in runs:
        diff = sorted(k for k in set(base["identity"]) | set(identity(meta))
                      if base["identity"].get(k) != identity(meta).get(k))
        if diff:
            lines.append("not the baseline's machine or build:")
            lines += [f"  {k}: baseline {base['identity'].get(k)!r}, now {identity(meta).get(k)!r}" for k in diff]
            return "incomparable", lines, []
    chosen = lambda key: key.split("|")[1] in use_cases
    for i, (_, cells) in enumerate(runs):
        ratios = [cells[k] / statistics.median(v) for k, v in base["cells"].items()
                  if k.startswith(CONTROL + "|") and chosen(k) and k in cells]
        factor = statistics.median(ratios)
        lines.append(f"run {i + 1}: control ({CONTROL}) at x{factor:.3f} of the baseline's speed")
        if abs(factor - 1) > CONTROL_TOLERANCE:
            lines.append(f"  outside 1 +/- {CONTROL_TOLERANCE}: another machine, a busy one, or another thermal state")
            return "incomparable", lines, []
    slower, faster = [], []
    for key, history in sorted(base["cells"].items()):
        if key.split("|")[0] not in SUBJECTS or not chosen(key) or any(key not in c for _, c in runs):
            continue
        now = [c[key] for _, c in runs]
        if min(now) > max(history) * (1 + SLOWER_BY):
            slower.append(key)
            lines.append(f"  slower  {key}: best new run {min(now)} ps against the baseline's slowest {max(history)} "
                         f"({min(now) / max(history) - 1:+.1%})")
        elif max(now) < min(history) * (1 - SLOWER_BY):
            faster.append(key)
            lines.append(f"  faster  {key}: {max(now) / min(history) - 1:+.1%}")
    if faster:
        lines.append(f"{len(faster)} cells faster than every baseline run: after review, `record` a new baseline")
    return ("slower" if slower else "ok"), lines, slower


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["check", "record"]:
        p = sub.add_parser(name)
        p.add_argument("--bench", default=str(ROOT / "bench-hashes"), help="bench-hashes checkout")
    sub.choices["record"].add_argument("--runs", type=int, default=RECORD_RUNS)
    sub.choices["record"].add_argument("--from", dest="saved", nargs="+",
                                       help="record from these saved samples files instead of new runs")
    p = sub.add_parser("compare")
    p.add_argument("--base", nargs="+", required=True, help="samples files of the baseline runs")
    p.add_argument("--new", nargs="+", required=True, help="samples files of the new runs")
    for p in sub.choices.values():
        p.add_argument("--use-cases", default="OneMessage,ManyMessages",
                       help="comma-separated: OneMessage, ManyMessages")
    args = parser.parse_args()
    use_cases = set(args.use_cases.split(","))

    if args.command == "compare":
        base = baseline_from([parse(Path(f).read_text()) for f in args.base])
        verdict, lines, slower = judge(base, [parse(Path(f).read_text()) for f in args.new], use_cases)
        print("\n".join(lines))
        print(f"perf_regress: {verdict}" + (f" in {len(slower)} cells" if slower else ""))
        return {"ok": 0, "slower": 1, "incomparable": 2}[verdict]

    if args.command == "record" and args.saved:
        runs = [parse(Path(f).read_text()) for f in args.saved]
    else:
        exe = build_bench(args.bench)
    if args.command == "record":
        if not args.saved:
            runs = [parse(run_bench(exe)) for _ in range(args.runs)]
        base = baseline_from(runs)
        BASELINES.mkdir(exist_ok=True)
        path = BASELINES / f"{machine_name(runs[0][0])}.json"
        path.write_text(json.dumps(base, indent=1, sort_keys=True) + "\n")
        print(f"perf_regress: baseline of {len(runs)} runs written to {path.relative_to(ROOT)}; review and commit it")
        return 0

    runs = [parse(run_bench(exe))]
    path = BASELINES / f"{machine_name(runs[0][0])}.json"
    if not path.exists():
        print(f"perf_regress: no baseline for this machine ({path.relative_to(ROOT)}); "
              "run `python3 tools/perf_regress.py record` and commit the file. No verdict (exit 2).")
        return 2
    base = json.loads(path.read_text())
    runs += [parse(run_bench(exe)) for _ in range(CHECK_RUNS - 1)]
    verdict, lines, slower = judge(base, runs, use_cases)
    if verdict == "slower":
        print("\n".join(lines))
        print(f"perf_regress: {len(slower)} cells slower in {len(runs)} runs; one more run must agree")
        runs.append(parse(run_bench(exe)))
        verdict, lines, slower = judge(base, runs, use_cases)
    print("\n".join(lines))
    if verdict == "incomparable":
        print("perf_regress: no verdict (exit 2)")
        return 2
    if verdict == "ok":
        print(f"perf_regress: no regression ({len(runs)} runs against a baseline of {base['runs']})")
        return 0
    print(f"perf_regress: REGRESSION in {len(slower)} cells, in every one of {len(runs)} runs (exit 1)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
