#!/usr/bin/env python3
"""Look for performance regressions already in git history.

    python3 tools/perf_bisect.py COMMIT...     # oldest first

Compares every commit with the one before it, then the last with the
first, each comparison an A B B A A B B A run of tools/perf_regress.py
compare on this machine. The last-with-first comparison catches slowdowns
that each step kept under the margin. Each commit's benchmark build is
cached, so a longer list costs one build per commit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_regress as pr  # noqa: E402


def main():
    commits = sys.argv[1:]
    if len(commits) < 2:
        sys.exit(__doc__)
    verdicts = []
    for old, new in list(zip(commits, commits[1:])) + [(commits[0], commits[-1])]:
        print(f"\n== {new} against {old}", flush=True)
        verdicts.append((old, new, pr.compare(old, new)))
    print("\nperf_bisect:")
    for old, new, code in verdicts:
        print(f"  {new} against {old}: {['no regression', 'REGRESSION', 'no verdict'][code]}")
    return 1 if any(code == 1 for _, _, code in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
