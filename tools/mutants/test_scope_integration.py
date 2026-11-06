"""The merged #250/#245/#256 entry point: scope, empty records, hang rules.

Invoke main with Git and mutation execution substituted, rather than
starting a real mutation run to test what work would be selected.
"""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import check


class Output(io.StringIO):
    def reconfigure(self, **kwargs):
        self.buffering = kwargs


class Planned(Exception):
    pass


class ScopeIntegration(unittest.TestCase):
    def plan(self, args=(), *, origin=True, altered=False):
        output, runs, cleanup, row_checks = Output(), [], [], []
        commit, base = "a"*40, "b"*40
        hanging = "src/a.rs\treplace wait with ()"
        listed = ["src/a.rs:9:5: replace wait with ()"]

        def command(argv, **kwargs):
            if "merge-base" in argv:
                return subprocess.CompletedProcess(argv, 0 if origin else 1, stdout=base if origin else "")
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, stdout=commit)
            if argv[:4] == ["cargo", "mutants", "--list", "--json"]:
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([{"name": "one planned mutant"}]))
            if "status" in argv or "diff" in argv:
                return subprocess.CompletedProcess(argv, 0, stdout="")
            raise AssertionError(f"unexpected command in scope planning: {argv}")

        def mutation_run(*args, **kwargs):
            runs.append((args, kwargs))
            raise Planned()

        def rows(remove=False):
            row_checks.append(remove)
            return listed

        def known(path=None, inactive=None):
            if path == check.TIMEOUTS:
                return [], {hanging: (1, "known loop that never ends")}
            return ["src/a.rs"], {}

        with contextlib.ExitStack() as stack:
            for target, value in [
                ("sys.argv", ["check.py", *args]), ("sys.stdout", output),
                ("check.subprocess.run", command), ("check.remove_leftovers", lambda: cleanup.append(True)),
                ("check.check_rows_exist", rows), ("check.read_known", known), ("check.run", mutation_run),
                ("check.LOOM_FILES", []),
                ("check.changed_tests", lambda _: ({"test_prefetch"} if altered else set(), True, {"src/test.rs"} if altered else set())),
                ("check.caught_by", lambda _: None),
                ("check.test_code_index", lambda _: [("test", "test_prefetch", "prefetch_kernels(input);", "src/test.rs")]),
                ("check.library_functions", lambda _: [("src/a.rs", "prefetch_kernels")]),
            ]:
                stack.enter_context(patch(target, value))
            error = None
            try:
                check.main()
            except Planned:
                pass
            except SystemExit as exc:
                error = exc.code
        return output.getvalue(), runs, cleanup, row_checks, error

    def test_default_is_a_change_and_no_change_runs_no_mutants(self):
        text, runs, cleanup, rows, error = self.plan()
        self.assertIn("mutants: the change since " + "b"*40, text)
        self.assertNotIn("mutants: the whole run", text)
        self.assertEqual(runs, [])
        self.assertEqual(rows, [True])
        self.assertIsNone(error)

    def test_missing_origin_stops_before_any_work(self):
        text, runs, cleanup, rows, error = self.plan(origin=False)
        self.assertIn("no origin/servil", error)
        self.assertEqual((runs, cleanup, rows), ([], [], []))
        self.assertNotIn("mutants: the whole run", text)

    def test_explicit_since_needs_no_origin(self):
        text, runs, cleanup, rows, error = self.plan(["--since", "chosen"], origin=False)
        self.assertIn("mutants: the change since chosen", text)
        self.assertEqual(runs, [])
        self.assertIsNone(error)

    def test_no_record_keeps_the_plan_a_change(self):
        text, runs, cleanup, rows, error = self.plan(altered=True)
        self.assertIn("finds no record", text)
        self.assertIn("the plan: 1 mutants of 1 functions", text)
        self.assertNotIn("mutants: the whole run", text)
        self.assertEqual(len(runs), 1)
        args, kwargs = runs[0]
        self.assertEqual(args[0], ["src/a.rs"])
        self.assertIn("--re", args[1])
        self.assertTrue(any("prefetch_kernels" in a for a in args[1]))
        self.assertIsNone(error)

    def test_explicit_whole_skips_known_hangs(self):
        text, runs, cleanup, rows, error = self.plan(["--whole"])
        self.assertIn("mutants: the whole run", text)
        self.assertIn("1 known hangs", text)
        self.assertEqual(rows, [False])
        self.assertIn("--exclude-re", runs[0][0][1])
        self.assertIsNone(error)

    def test_write_runs_known_hangs_and_labels_the_whole_scope(self):
        text, runs, cleanup, rows, error = self.plan(["--write"])
        self.assertIn("mutants: the whole run", text)
        self.assertEqual(runs[0][0][1], [])
        self.assertIsNone(error)

    def test_all_hangs_runs_them_in_the_explicit_whole_run(self):
        text, runs, cleanup, rows, error = self.plan(["--whole", "--all-hangs"])
        self.assertIn("mutants: the whole run", text)
        self.assertEqual(runs[0][0][1], [])
        self.assertIsNone(error)

    def test_contradictory_scopes_stop_before_work(self):
        for flag in ["--whole", "--write"]:
            text, runs, cleanup, rows, error = self.plan(["--since", "chosen", flag])
            self.assertIn("not both", error)
            self.assertEqual((runs, cleanup, rows), ([], [], []))

    def test_help_does_not_start_a_check(self):
        text, runs, cleanup, rows, error = self.plan(["--help"])
        self.assertIn("Mutation testing", text)
        self.assertEqual((runs, cleanup, rows), ([], [], []))
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
