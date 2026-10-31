"""Release-gate controls. All GitHub answers and release logs are fixtures."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("release_check", Path(__file__).with_name("release-check.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

HEAD, OLD = "a" * 40, "b" * 40
RUN = {"id": 7, "name": "tests", "event": "workflow_dispatch", "status": "completed", "conclusion": "success"}
MUTATION_OK = "mutants: 12 survive, each known and explained\nmutants: 3 time out, each known (timeouts-x86_64.txt)\n"


def job(conclusion="success", status="completed"):
    return {"name": "test job", "status": status, "conclusion": conclusion}


class CI(unittest.TestCase):
    def result(self, jobs):
        with patch.object(check, "github", return_value={"total_count": len(jobs), "jobs": jobs}):
            return check.ci_result(RUN)

    def test_pr_merge_run_is_not_release_evidence(self):
        with patch.object(check, "github") as github:
            self.assertIsNone(check.ci_result(dict(RUN, event="pull_request")))
            github.assert_not_called()

    def test_success_needs_successful_jobs(self):
        self.assertIs(self.result([job(), job()]), True)
        self.assertIsNone(self.result([]))

    def test_skipped_or_pending_is_open(self):
        for other in [job("skipped"), job(None, "in_progress")]:
            with self.subTest(other=other):
                self.assertIsNone(self.result([job(), other]))

    def test_failed_or_cancelled_does_not_pass(self):
        for conclusion in ["failure", "cancelled", "action_required"]:
            with self.subTest(conclusion=conclusion):
                self.assertIs(self.result([job(conclusion)]), False)

    def test_job_timeout_stays_distinct(self):
        self.assertEqual(self.result([job("timed_out")]), check.TIMED_OUT)

    def test_unfinished_or_failed_workflow_does_not_pass(self):
        with patch.object(check, "github") as github:
            self.assertIsNone(check.ci_result(dict(RUN, status="in_progress", conclusion=None)))
            self.assertIs(check.ci_result(dict(RUN, conclusion="failure")), False)
            self.assertEqual(check.ci_result(dict(RUN, conclusion="timed_out")), check.TIMED_OUT)
            github.assert_not_called()

    def test_all_pages_are_checked(self):
        replies = [{"total_count": 101, "jobs": [job()] * 100}, {"total_count": 101, "jobs": [job("skipped")]}]
        with patch.object(check, "github", side_effect=replies) as github:
            self.assertIsNone(check.ci_result(RUN))
            self.assertIn("page=2", github.call_args.args[0])

    def test_missing_page_stops(self):
        with patch.object(check, "github", return_value={"total_count": 1, "jobs": []}):
            with self.assertRaises(AssertionError):
                check.ci_result(RUN)


class ReleaseEvidence(unittest.TestCase):
    def run_gate(self, cells="all", kind="check", ci_jobs=None, new=HEAD, old=OLD, verdict="no regression", git_override=None, ci_event="workflow_dispatch", mutation_log=MUTATION_OK, power=("AC Power, 100%", "AC Power, 100%")):
        jobs = [job(), job()] if ci_jobs is None else ci_jobs
        def git(*args):
            if args[0] == "rev-parse":
                return HEAD if args[1] == "candidate" else OLD
            if args[0] == "tag":
                return "v0.3.0+fixture"
            if args[0] == "describe":
                return "v0.3.0+fixture"
            if args[0] == "diff":
                return "README.md\nQUALITY.md\nCHANGELOG.md"
            raise AssertionError(args)

        def github(path):
            if "/actions/runs?" in path:
                return {"workflow_runs": [dict(RUN, event=ci_event)]}
            if "/actions/runs/7/jobs?" in path:
                return {"total_count": len(jobs), "jobs": jobs}
            if path.endswith("/statuses"):
                return [{"context": "long proofs", "state": "success"}]
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runner/jobs").mkdir(parents=True)
            (root / "runner/results/1").mkdir(parents=True)
            task = {"type": kind, "old_commit": old, "new_commit": new}
            if cells is not None:
                task["cells"] = cells
            (root / "runner/jobs/job.json").write_text(json.dumps(task))
            readings = {k: v for k, v in zip(("power at start", "power at end"), power) if v is not None}
            (root / "runner/results/1/verdict.json").write_text(json.dumps({"job": "job.json", "status": verdict, **readings}))
            logs = root / "tmp/night"
            logs.mkdir(parents=True)
            candidate = (git_override or git)("rev-parse", "candidate")[:12]
            for name, text in [("check-release-vm", "no cell slower"), ("check-builds", "every build passes"), ("publish-dry-run", "publish-dry-run: ok")]:
                (logs / f"{name}-{candidate}.log").write_text(text + "\n")
            if mutation_log is not None:
                (logs / f"mutants-{candidate}.log").write_text(mutation_log)
            output = io.StringIO()
            with patch.object(check, "MAIN", tmp), patch.object(check, "git", git_override or git), \
                 patch.object(check, "github", github), \
                 patch("sys.argv", ["release-check.py", "candidate"]), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as caught:
                    check.main()
            return caught.exception.code, output.getvalue()

    def test_complete_fixture_passes(self):
        self.assertEqual(self.run_gate()[0], 0)

    def test_pr_merge_ci_not_pass(self):
        code, output = self.run_gate(ci_event="pull_request")
        self.assertEqual(code, 1)
        self.assertIn("OPEN  CI, every job", output)

    def test_skipped_ci_not_pass(self):
        code, output = self.run_gate(ci_jobs=[job(), job("skipped")])
        self.assertEqual(code, 1)
        self.assertIn("OPEN  CI, every job", output)

    def test_mac_off_ac_power_is_no_verdict(self):
        # A check on battery measures the efficiency cores (#202): no
        # evidence, whether it started, ended or went unrecorded so.
        for power in [("Battery Power, 20%", "Battery Power, 12%"), ("AC Power, 100%", "Battery Power, 98%"), ("AC Power, 100%", None)]:
            code, output = self.run_gate(power=power)
            self.assertEqual(code, 2, power)
            self.assertIn("NO VERDICT  Mac check", output)
            self.assertIn("run it again plugged in", output)

    def test_partial_mac_not_pass(self):
        code, output = self.run_gate(cells=None)
        self.assertEqual(code, 1)
        self.assertIn("OPEN  Mac check", output)

    def test_non_check_mac_not_pass(self):
        self.assertEqual(self.run_gate(kind="benchmark")[0], 1)

    def test_wrong_commits_not_pass(self):
        self.assertEqual(self.run_gate(new="c" * 40)[0], 1)
        self.assertEqual(self.run_gate(old="c" * 40)[0], 1)

    def test_mutation_timeout_mismatch_is_no_verdict(self):
        text = "mutants: 12 survive, each known and explained\nmutants: the time-outs differ from timeouts-x86_64.txt (no verdict on those mutants)\n"
        code, output = self.run_gate(mutation_log=text)
        self.assertEqual(code, 2, output)
        self.assertIn("NO VERDICT  full mutation run", output)

    def test_mutation_survivor_failure_is_failed(self):
        text = "mutants: the list and this run differ, or a listed mutant has no reason\nmutants: 3 time out, each known (timeouts-x86_64.txt)\n"
        code, output = self.run_gate(mutation_log=text)
        self.assertEqual(code, 1, output)
        self.assertIn("FAIL  full mutation run", output)

    def test_incomplete_mutation_log_is_open(self):
        for text in [None, "", MUTATION_OK.splitlines()[0], MUTATION_OK + "interrupted\n"]:
            with self.subTest(text=text):
                code, output = self.run_gate(mutation_log=text)
                self.assertEqual(code, 1, output)
                self.assertIn("OPEN  full mutation run", output)

    def test_mac_undecided_outcomes_remain_distinct(self):
        for verdict, label in [("timed out", "TIMED OUT"), ("no verdict", "NO VERDICT")]:
            with self.subTest(verdict=verdict):
                code, output = self.run_gate(verdict=verdict)
                self.assertEqual(code, 2)
                self.assertIn(label + "  Mac check", output)


class PreviousRelease(unittest.TestCase):
    """Real Git histories, with synthetic CI/Mac/log evidence for main()."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.date = "2020-01-01T00:00:00+00:00"
        self.git("init", "-q", "--initial-branch=main")
        self.git("config", "user.name", "Sam Shackleton")
        self.git("config", "user.email", "samshackleton-tech@users.noreply.github.com")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", str(self.repo / "no-hooks"))
        for page in ["README.md", "QUALITY.md", "CHANGELOG.md"]:
            (self.repo / page).write_text("old\n")
        self.git("add", ".")
        self.git("commit", "-qm", "old release")
        self.old = self.git("rev-parse", "HEAD")
        self.git("tag", "v0.3.0+fixture")
        self.date = "2020-01-02T00:00:00+00:00"
        for page in ["README.md", "QUALITY.md", "CHANGELOG.md"]:
            (self.repo / page).write_text("new\n")
        self.git("commit", "-qam", "candidate")
        self.new = self.git("rev-parse", "HEAD")
        self.git("branch", "candidate")

    def git(self, *args):
        env = dict(os.environ, GIT_AUTHOR_DATE=self.date, GIT_COMMITTER_DATE=self.date)
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True, env=env, stderr=subprocess.PIPE).strip()

    def test_tagged_candidate_uses_previous_release(self):
        self.git("tag", "-a", "v0.8.0+fixture", "-m", "rehearsal tag")
        code, output = ReleaseEvidence.run_gate(self, new=self.new, old=self.old, git_override=self.git)
        self.assertEqual(code, 0, output)
        self.assertIn("rewritten since v0.3.0+fixture", output)

    def test_unrelated_recent_tag_cannot_set_the_baseline(self):
        self.git("checkout", "-qb", "unrelated", self.old)
        self.date = "2020-01-03T00:00:00+00:00"
        (self.repo / "side.txt").write_text("another branch\n")
        self.git("add", ".")
        self.git("commit", "-qm", "unrelated work")
        self.git("tag", "v99.0.0+fixture")
        self.git("checkout", "-q", "candidate")
        code, output = ReleaseEvidence.run_gate(self, new=self.new, old=self.old, git_override=self.git)
        self.assertEqual(code, 0, output)

    def test_no_reachable_release_is_open(self):
        self.git("tag", "-d", "v0.3.0+fixture")
        self.git("tag", "v0.8.0+fixture")
        code, output = ReleaseEvidence.run_gate(self, new=self.new, old=self.old, git_override=self.git)
        self.assertEqual(code, 1)
        self.assertIn("no previous release: read README.md", output)

    def test_release_on_second_parent_is_reachable(self):
        self.git("checkout", "-qb", "release-side", self.old)
        self.date = "2020-01-03T00:00:00+00:00"
        (self.repo / "side.txt").write_text("previous release\n")
        self.git("add", ".")
        self.git("commit", "-qm", "previous release on another parent")
        self.git("tag", "v0.4.0+fixture")
        self.git("checkout", "-q", "candidate")
        self.git("merge", "--no-ff", "-qm", "merge release", "release-side")
        head = self.git("rev-parse", "HEAD")
        with patch.object(check, "git", self.git):
            self.assertEqual(check.previous_release(head), "v0.4.0+fixture")


if __name__ == "__main__":
    unittest.main()
