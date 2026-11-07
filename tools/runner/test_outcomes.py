"""The runner's completed jobs use statuses wait_for already understands."""
import subprocess
import unittest
from pathlib import Path

import runner


class FakeRun:
    def __init__(self, code=0):
        self.code = code
        self.command = None

    def checkouts(self, *_):
        return Path("/code/library"), Path("/code/benchmark")

    def run(self, command, **_):
        self.command = command
        if self.code:
            raise subprocess.CalledProcessError(self.code, command)


class CheckOutcomes(unittest.TestCase):
    def job(self, probe=False):
        job = {"old_commit": "1" * 40, "new_commit": "2" * 40, "bench_commit": "3" * 40}
        job["probe" if probe else "points"] = "lent 64 KiB"
        return job

    def test_probe_completes_without_claiming_a_speed_verdict(self):
        run = FakeRun()
        self.assertEqual(runner.check(self.job(True), run, None, Path("/results")), "ok")
        self.assertIn("--probe", run.command)

    def test_gate_and_failure_outcomes_keep_their_meaning(self):
        for code, outcome in [(0, "no regression"), (1, "regression"), (2, "no verdict")]:
            self.assertEqual(runner.check(self.job(), FakeRun(code), None, Path("/results")), outcome)


if __name__ == "__main__":
    unittest.main()
