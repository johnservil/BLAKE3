#!/usr/bin/env python3
"""C bad-call guards under Miri: a diagnostic/abort, never a reference UB.
Nextest uses 100 for a caught test panic; an aborting extern C call may
make the interpreter return 1. The exact diagnostic, not the exit alone,
identifies the intended guard.

Run after Miri setup, with the chosen nightly selected. Cases come from
its compiled Rust test, not a second list or a parser of Rust source.
"""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
COMMAND = ["cargo", "miri", "nextest", "run", "--manifest-path", "capi/Cargo.toml",
           "--config-file", ".config/nextest.toml", "--profile", "miri",
           "--run-ignored", "only", "--success-output", "immediate",
           "-E", "test(bad_calls_stop_at_the_guard)"]
FLAGS = "-Zmiri-num-cpus=1 -Zmiri-ignore-leaks -Zmiri-strict-provenance -Zmiri-env-forward=B3_CAPI_GUARD_CASE"


def run(case):
    result = subprocess.run(COMMAND, cwd=ROOT,
                            env=dict(os.environ, MIRIFLAGS=FLAGS, B3_CAPI_GUARD_CASE=case),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.returncode, result.stdout


def main():
    code, output = run("list")
    prefix = "B3_CAPI_GUARD_CASES="
    inventories = [line.strip()[len(prefix):] for line in output.splitlines() if line.strip().startswith(prefix)]
    assert code == 0 and len(inventories) == 1, output
    cases = json.loads(inventories[0])
    assert cases and len({case for case, _ in cases}) == len(cases), cases
    for case, message in cases:
        code, output = run(case)
        assert code in (1, 100) and message in output and "panicked at" in output and "Undefined Behavior" not in output, output
        print(f"{case}: guard diagnosed the call without Miri UB", flush=True)
    print(f"C boundary guards: {len(cases)} interpreted cases checked", flush=True)


if __name__ == "__main__":
    main()
