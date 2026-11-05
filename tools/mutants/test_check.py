"""CPU-conditioned survivor-list controls: python3 -m unittest discover -s tools/mutants."""
import os
import subprocess
import tempfile
import unittest
from unittest.mock import mock_open, patch

import check


class Conditions(unittest.TestCase):
    def test_has_lacks_are_complements(self):
        for flags in [set(), {"avx512f"}, {"avx512vl"}, {"avx512f", "avx512vl"}]:
            with self.subTest(flags=flags), patch.object(check, "FLAGS", flags):
                self.assertTrue(check.applies("unconditional reason"))
                self.assertEqual(check.applies("[cpu has avx512f] reason"), "avx512f" in flags)
                self.assertEqual(check.applies("[cpu lacks avx512f] reason"), "avx512f" not in flags)
                self.assertEqual(check.applies("[cpu has avx512f avx512vl] reason"), len(flags) == 2)
                self.assertEqual(check.applies("[cpu lacks avx512f avx512vl] reason"), len(flags) != 2)

    def test_malformed_conditions_stop(self):
        for reason in ["[cpu have avx2] reason", "[cpu has ] reason", "[cpu has avx2 reason",
                       "[cpu has avx2]reason", "[cpu has avx2] ", " [cpu has avx2] reason",
                       "[CPU has avx2] reason", "[cpu has avx2, avx512f] reason"]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "malformed CPU condition"):
                check.applies(reason)

    def test_select_and_preserve_inactive_rows(self):
        rows = "files: src/platform.rs\n" + "".join(
            f"src/platform.rs\tmutant\t1\t[cpu {condition} avx2] reason\n"
            for condition in ["has", "lacks"])
        for flags, active, inactive in [({"avx2"}, "has", "lacks"), (set(), "lacks", "has")]:
            with patch.object(check, "FLAGS", flags), patch.object(check, "INACTIVE", []), \
                 patch("builtins.open", mock_open(read_data=rows)):
                files, known = check.read_known()
                self.assertEqual(files, ["src/platform.rs"])
                self.assertEqual(known, {"src/platform.rs\tmutant": (1, f"[cpu {active} avx2] reason")})
                self.assertEqual(check.INACTIVE, [f"src/platform.rs\tmutant\t1\t[cpu {inactive} avx2] reason"])

    def test_duplicate_active_rows_stop(self):
        rows = "src/platform.rs\tmutant\t1\treason\nsrc/platform.rs\tmutant\t2\t[cpu has avx2] reason\n"
        with patch.object(check, "FLAGS", {"avx2"}), patch("builtins.open", mock_open(read_data=rows)), \
             self.assertRaisesRegex(AssertionError, "two rows apply"):
            check.read_known()


class Readers(unittest.TestCase):
    def test_linux_fields(self):
        for field in ["flags", "Features"]:
            with patch.object(check.sys, "platform", "linux"), \
                 patch("builtins.open", mock_open(read_data=f"processor: 0\n{field}\t: avx2 avx512f\n")):
                self.assertEqual(check.cpu_flags(), {"avx2", "avx512f"})

    def test_linux_reader_error_propagates(self):
        with patch.object(check.sys, "platform", "linux"), patch("builtins.open", side_effect=PermissionError("denied")), \
             self.assertRaises(PermissionError):
            check.cpu_flags()

    def test_linux_missing_field_stops(self):
        with patch.object(check.sys, "platform", "linux"), patch("builtins.open", mock_open(read_data="processor: 0\n")), \
             self.assertRaisesRegex(ValueError, "no flags or Features"):
            check.cpu_flags()

    def test_macos_selection(self):
        out = subprocess.CompletedProcess([], 0, "hw.optional.avx2_0: 1\nhw.optional.avx512f: 0\n")
        with patch.object(check.sys, "platform", "darwin"), patch.object(check.subprocess, "run", return_value=out) as run:
            self.assertEqual(check.cpu_flags(), {"avx2_0"})
            run.assert_called_once_with(["sysctl", "hw.optional"], capture_output=True, text=True, check=True)

    def test_macos_reader_errors_propagate(self):
        for error in [FileNotFoundError("sysctl"), subprocess.CalledProcessError(1, ["sysctl", "hw.optional"])]:
            with patch.object(check.sys, "platform", "darwin"), patch.object(check.subprocess, "run", side_effect=error), \
                 self.assertRaises(type(error)):
                check.cpu_flags()

    def test_macos_missing_or_malformed_output_stops(self):
        for text in ["", "hw.optional.avx2_0", "hw.optional.avx2_0:", "unexpected: 1"]:
            out = subprocess.CompletedProcess([], 0, text)
            with patch.object(check.sys, "platform", "darwin"), patch.object(check.subprocess, "run", return_value=out), \
                 self.assertRaisesRegex(ValueError, "missing or malformed"):
                check.cpu_flags()

class Placeholders(unittest.TestCase):
    def test_a_new_time_outs_placeholder_names_the_stopped_tests(self):
        name = "src/lanes.rs\treplace spin_for_work -> bool with true"
        with patch.object(check, "STOPPED", {name: {"blake3-servil b_test", "blake3-servil a_test"}}):
            self.assertEqual(check.unclassified(name),
                             "UNCLASSIFIED (the clock stopped: blake3-servil a_test, blake3-servil b_test)")
            self.assertEqual(check.unclassified("src/lib.rs\tother"), "UNCLASSIFIED")

class Model(unittest.TestCase):
    def test_a_name_counts_place_by_place_with_the_model(self):
        # gate20 (October 24): Finish::wait's deleted ! at two places; the
        # suite's clock stopped one, which the model catches, and the
        # other survives both. Counted by name, the caught place stood as
        # a time-out.
        name = "src/finish.rs\tdelete ! in Finish::wait"
        outcomes = {"run": {(name, "91:16"): "MissedMutant", (name, "97:75"): "TimedOut"},
                    "loom": {(name, "91:16"): "MissedMutant", (name, "97:75"): "CaughtMutant"}}
        with patch.object(check, "PLACE_OUTCOMES", outcomes), patch.object(check, "LOOM_FILES", ["src/finish.rs"]):
            found, timed = check.with_model("run", {name: 1}, {name: 1})
        self.assertEqual((found.get(name, 0), timed.get(name, 0)), (1, 0))

class Limits(unittest.TestCase):
    def test_tests_share_the_cpus_and_fit_in_memory(self):
        gib = 1 << 30
        with patch.object(check, "memory_cap", lambda shards: min(8 * gib, (15 * gib - 4 * gib) // shards)):
            threads, cap = check.test_limits(6, 20, 15 * gib)
        self.assertEqual(threads, 3)
        self.assertLessEqual(6 * threads * cap, 11 * gib, "every shard's tests at their cap fit")
        self.assertGreater(cap, 10 * 41 << 20, "ten times the heaviest honest test")
        with patch.object(check, "memory_cap", lambda shards: 8 * gib):
            self.assertEqual(check.test_limits(6, 4, 15 * gib)[0], 1, "at least one test thread")

class Rows(unittest.TestCase):
    def test_a_row_naming_a_mutant_the_source_lacks_is_stale(self):
        listed = ["src/park.rs:97:9: replace Park::wake_one -> bool with false",
                  "src/lanes.rs:130:60: replace * with +"]
        rows = [("known-aarch64.txt", "src/park.rs", "replace Park::wake_one -> bool with false"),
                ("known-x86_64.txt", "src/lanes.rs", "replace * with +"),
                # The control: a mutant no line names.
                ("timeouts-aarch64.txt", "src/lanes.rs", "replace Pool::any_parked -> bool with false")]
        self.assertEqual(check.stale_rows(rows, listed), rows[2:])
        self.assertEqual(check.stale_rows(rows[:2], listed), [])

    def test_rows_are_read_from_every_list_but_headers(self):
        text = ("# a comment\nfiles: src/park.rs\nmodel files: src/park.rs\n"
                "src/park.rs\treplace Park::wake_one -> bool with false\t1\tspeed: a reason\n\n")
        with patch("builtins.open", mock_open(read_data=text)):
            self.assertEqual(check.list_rows(["/x/known-x86_64.txt"]),
                             [("known-x86_64.txt", "src/park.rs", "replace Park::wake_one -> bool with false")])


class Leftovers(unittest.TestCase):
    def test_a_new_row_takes_every_other_machines_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            known = os.path.join(d, "known-otherarch.txt")
            timeouts = os.path.join(d, "timeouts-otherarch.txt")
            with open(known, "w") as f:
                f.write("# header\nfiles: src/a.rs\nsrc/a.rs\treplace a with b\t1\tspeed: equally correct paths\n")
            with open(timeouts, "w") as f:
                f.write("# header\nsrc/a.rs\treplace c with d\t1\ta hang test_x showed\n"
                        "src/a.rs\treplace e with f\t1\tUNCLASSIFIED\n")
            lists = [known, timeouts]
            # The same outcome there: its reason, named by machine.
            self.assertEqual(check.unclassified("src/a.rs\treplace a with b", "survives", lists),
                             "as on otherarch: survives (known-otherarch.txt): speed: equally correct paths")
            # A different outcome: an OPEN row naming both.
            self.assertEqual(check.unclassified("src/a.rs\treplace c with d", "survives", lists),
                             f"OPEN: the machines differ: on {check.platform.machine()}: survives; "
                             "on otherarch: times out (timeouts-otherarch.txt): a hang test_x showed")
            # Nobody knows (or the other row is unexplained too): UNCLASSIFIED.
            self.assertTrue(check.unclassified("src/a.rs\treplace e with f", "times out", lists).startswith("UNCLASSIFIED"))
            self.assertTrue(check.unclassified("src/a.rs\treplace g with h", "survives", lists).startswith("UNCLASSIFIED"))

    def test_only_the_shards_worktrees_go(self):
        listed = "\n".join([
            "worktree /repo", "HEAD a", "",
            "worktree /tmpdir/blake3-servil-mutants/mutants-abc123", "HEAD b", "",
            "worktree /elsewhere/mutants-xyz", "HEAD c", "",
            "worktree /repo/tmp/tests-map-worktree", "HEAD d", "",
        ])
        calls = []
        def run(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=listed if "list" in args else "")
        with patch.object(check, "ROOT", "/repo"), patch.object(check, "WORKTREES", "/tmpdir/blake3-servil-mutants"), \
             patch.object(check, "TARGET_DIR", "/tmpdir/target"), patch.object(check.os.path, "realpath", lambda p: p), \
             patch.object(check.subprocess, "run", run), patch.object(check.shutil, "rmtree", lambda *a, **k: None), \
             patch.object(check.os, "makedirs", lambda *a, **k: None):
            check.remove_leftovers()
        removed = [a[-1] for a in calls if "remove" in a]
        self.assertEqual(removed, ["/tmpdir/blake3-servil-mutants/mutants-abc123"], "every other worktree stays")


if __name__ == "__main__":
    unittest.main()


class Hangs(unittest.TestCase):
    """BLAKE3 #244: the known hangs stay out of a whole run, and a test
    nextest ends after another failed is no outcome."""

    listed = ["src/stream.rs:600:15: replace + with - in room",
              "src/stream.rs:601:9: replace + with - in room",
              "src/stream.rs:700:3: replace < with <= in Stream<'env, D>::finish",
              "src/lanes.rs:10:5: replace splits -> bool with true",
              "src/lanes.rs:90:5: replace splits -> bool with true",
              "src/park.rs:40:9: replace Park::park_unless with ()",
              "src/lanes.rs:20:7: replace help with ()"]

    def test_only_names_whose_every_place_hangs_with_a_reason_are_skipped(self):
        tknown = {"src/stream.rs\treplace + with - in room": (2, "a loop variant broken"),
                  "src/stream.rs\treplace < with <= in Stream<'env, D>::finish": (1, "a loop variant broken"),
                  # One of two places hangs: the other's outcome is the run's to find.
                  "src/lanes.rs\treplace splits -> bool with true": (1, "a hang"),
                  # A model file's row: the suite and the model decide it together.
                  "src/park.rs\treplace Park::park_unless with ()": (1, "a hang"),
                  "src/lanes.rs\treplace help with ()": (1, "OPEN: the machines differ")}
        with patch.object(check, "LOOM_FILES", ["src/park.rs"]):
            self.assertEqual(check.known_hangs(tknown, self.listed),
                             ["src/stream.rs\treplace + with - in room", "src/stream.rs\treplace < with <= in Stream<'env, D>::finish"])
            unclassified = {"src/stream.rs\treplace + with - in room": (2, "UNCLASSIFIED")}
            self.assertEqual(check.known_hangs(unclassified, self.listed), [])

    def test_an_excluded_name_matches_its_places_alone(self):
        import re
        args = check.exclude_args(["src/stream.rs\treplace + with - in room"])
        self.assertEqual(args[0], "--exclude-re")
        pattern = re.compile(args[1])
        self.assertEqual([l for l in self.listed if pattern.search(l)], self.listed[:2])
        # Not the same change in another function, nor in another file.
        for other in ["src/stream.rs:600:15: replace + with - in room_for", "src/lanes.rs:600:15: replace + with - in room",
                      "src/stream.rs:600:15: replace + with - in Stream::room"]:
            self.assertIsNone(pattern.search(other), other)

    def test_a_test_ended_after_another_failed_is_no_outcome(self):
        caught = [("a", "FAIL", 70), ("b", "SIGTERM", 71), ("c", "PASS", 5)]
        clock = [("a", "TIMEOUT", 30000), ("b", "SIGTERM", 30001), ("c", "PASS", 5)]
        both = [("a", "TIMEOUT", 30000), ("b", "FAIL", 400)]
        self.assertFalse(check.clock_only(caught), "a test failed: caught")
        self.assertTrue(check.clock_only(clock), "the clock stopped one; the rest were ended: timed out")
        self.assertFalse(check.clock_only(both), "a test failed beside the clock: caught")
        self.assertFalse(check.clock_only([("a", "PASS", 1)]))
