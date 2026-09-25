#!/usr/bin/env python3
"""Where another contender beats servil: the minimax to-do list.

    pypy3 tools/losses.py SAMPLES.tsv [--margin 0.03]

Reads a bench-hashes samples file (v2) and lists, for servil and servil mt
in each scenario (solo, shared) and use case, every point where some other
contender's median time is lower than servil's, with how much lower. A
cell counts as lost when the best competitor is faster by more than
MARGIN (default 3%, below which run-to-run differences decide); losses
under the margin are listed as "close". The last line is the score: lost
cells, the number to drive down.

As in the benchmark's CHECKS, servil meets the single-threaded contenders
and servil mt meets every contender, servil included (a multithreaded call
slower than the single-threaded one is a defect, AGENTS.md).

Cells are compared by median, the figure the graph draws. Where servil's
samples split into two speeds (the benchmark's rule: a gap of 4% of the
median between neighbouring samples, 10% of samples or more on each side,
medians 1.25x apart or more), the line says so and gives the slow speed's
share and ratio to the median: on the Mac these are mostly samples that
ran on E-cores.
"""
import argparse
import statistics
from collections import defaultdict

SUBJECTS = ["blake3-servil-st", "blake3-servil-mt"]
MULTITHREADED = {"blake3-official-mt", "blake3-servil-mt"}
# Records before September 25, 2026 name the contenders by their old keys.
RENAMED = {"blake3-servil": "blake3-servil-st", "blake3": "blake3-official", "blake3-mt": "blake3-official-mt"}


def speeds(values):
    """(median, slow speed's median or None, slow speed's share)."""
    v = sorted(values)
    n, median = len(v), statistics.median(v)
    side = max(1, -(-n * 100 // 1000))
    best = None
    for split in range(side, n - side + 1):
        gap = v[split] - v[split - 1]
        if gap >= median * 0.04 and (best is None or gap > best[1]):
            best = (split, gap)
    if best:
        fast, slow = statistics.median(v[:best[0]]), statistics.median(v[best[0]:])
        if slow >= fast * 1.25:
            return median, slow, (n - best[0]) / n
    return median, None, 0.0


def load(path):
    cells, header = {}, None
    for line in open(path):
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if header is None:
            header = fields
            assert header == ["contender", "scenario", "use_case", "point", "unit", "ps_per_unit"], header
            continue
        contender, scenario, use_case, point, unit, values = fields
        contender = RENAMED.get(contender, contender)
        cells[(contender, scenario, use_case, point)] = speeds([int(v) for v in values.split(",")])
    return cells


def order(point):
    """Sort key for point labels: sizes by bytes, batches by count."""
    number, _, unit = point.partition(" ")
    return float(number) * {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20}[unit]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("samples")
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()
    cells = load(args.samples)
    contenders = sorted({k[0] for k in cells})
    lost = 0
    for subject in SUBJECTS:
        rivals = [c for c in contenders if c != subject and (subject == "blake3-servil-mt" or c not in MULTITHREADED)]
        for scenario in ("solo", "shared"):
            for use_case in ("OneMessage", "ManyMessages"):
                points = sorted({k[3] for k in cells if k[0] == subject and k[1] == scenario and k[2] == use_case}, key=order)
                rows = []
                for point in points:
                    mine = cells[(subject, scenario, use_case, point)]
                    beaten = []
                    for rival in rivals:
                        theirs = cells.get((rival, scenario, use_case, point))
                        if theirs and theirs[0] < mine[0]:
                            beaten.append((mine[0] / theirs[0] - 1, rival))
                    if beaten:
                        beaten.sort(reverse=True)
                        worst = beaten[0][0]
                        kind = "LOST " if worst > args.margin else "close"
                        lost += worst > args.margin
                        two = f"   [two speeds: {mine[2]:.0%} of samples at {mine[1] / mine[0]:.2f}x]" if mine[1] else ""
                        rows.append(f"    {kind} {point:>9}  " + ", ".join(f"{r} {g:+.0%}" for g, r in beaten) + two)
                if rows:
                    print(f"{subject} · {scenario} · {use_case} (servil slower by):")
                    print("\n".join(rows))
    print(f"\nscore: {lost} lost cells (a competitor faster by more than {args.margin:.0%})")


if __name__ == "__main__":
    main()
