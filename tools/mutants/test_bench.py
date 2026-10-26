"""bench.py's choices: python3 -m unittest discover -s tools/mutants."""
import unittest

import bench


class Rows(unittest.TestCase):
    def test_only_rows_claiming_speed_alone(self):
        lines = ["# a comment\tspeed\n", "files: src/a.rs\n",
                 "src/a.rs\treplace f with ()\t1\tspeed: a threshold\n",
                 "src/a.rs\tdelete ! in g\t1\tequally correct paths: either way\n",
                 "src/a.rs\treplace h -> bool with true\t1\t[cpu has avx2] speed: a CPU's own\n",
                 # The controls: reasons that claim no speed.
                 "src/a.rs\treplace | with ^ in k\t1\tequivalent: flag bits never overlap\n",
                 "src/a.rs\treplace m with ()\t1\tAArch64 only: compiled out here\n"]
        self.assertEqual(bench.speed_rows(lines),
                         [("src/a.rs", "replace f with ()"), ("src/a.rs", "delete ! in g"), ("src/a.rs", "replace h -> bool with true")])


class Places(unittest.TestCase):
    def test_a_place_any_test_did_not_pass_is_caught(self):
        tsv = ["# header\n",
               "src/a.rs\tdelete ! in g\t10:5\tsrc/a.rs g\tt1\tPASS\t3\n",
               "src/a.rs\tdelete ! in g\t12:7\tsrc/a.rs g\tt1\tPASS\t3\n",
               "src/a.rs\tdelete ! in g\t12:7\tsrc/a.rs g\tt2\tFAIL\t3\n",
               "src/a.rs\tdelete ! in g\t14:1\tsrc/a.rs g\tt2\tTIMEOUT\t30000\n",
               "(unmutated)\t\t\t\tt1\tPASS\t3\n"]
        caught = bench.caught_places(tsv)
        self.assertEqual(caught, {("src/a.rs", "delete ! in g"): {"12:7", "14:1"}})
        mutants = [{"file": "src/a.rs", "name": f"src/a.rs:{p}: delete ! in g", "diff": p} for p in ("10:5", "12:7", "14:1")]
        mutants.append({"file": "src/b.rs", "name": "src/b.rs:1:1: delete ! in g", "diff": "other file"})
        self.assertEqual([m["diff"] for m in bench.surviving(mutants, ("src/a.rs", "delete ! in g"), caught)], ["10:5"])
        # Without a record of catches, every place of the name.
        self.assertEqual(len(bench.surviving(mutants, ("src/a.rs", "delete ! in g"), {})), 3)


class Verdicts(unittest.TestCase):
    def test_check_exits_and_lines_become_verdicts(self):
        self.assertEqual(bench.verdict(0, "regress: no cell slower\n"), ("no cell slower", []))
        self.assertEqual(bench.verdict(1, "  SLOWER x: +9%\nregress: REGRESSION: 1 cells slower\n"), ("slower", []))
        self.assertEqual(bench.verdict(2, "regress: other programs kept the machine busy\n"), ("no verdict", []))
        self.assertEqual(bench.verdict(0, "check: the change touches no line a cell executes; nothing to measure\n"), ("no cell runs it", []))
        faster = "  faster x|solo|A|1: -6.0%\nregress: no cell slower\n"
        self.assertEqual(bench.verdict(0, faster), ("faster", ["faster x|solo|A|1: -6.0%"]))
        # A slower cell beside a faster one is slower.
        self.assertEqual(bench.verdict(1, "  SLOWER y: +5%\n" + faster)[0], "slower")
        self.assertEqual(bench.verdict(101, "")[0], "check failed (101)")


class Shown(unittest.TestCase):
    def test_the_largest_moves_each_way_and_a_count(self):
        out = "".join(f"  SLOWER c{i}: +{i}.0% (pairs)\n" for i in range(1, 6)) + "  faster f: -6.0% (pairs)\nregress: REGRESSION\n"
        self.assertEqual(bench.largest(out), ["SLOWER c5: +5.0% (pairs)", "SLOWER c4: +4.0% (pairs)", "SLOWER c3: +3.0% (pairs)",
                                              "and 2 more cells slower", "faster f: -6.0% (pairs)"])


if __name__ == "__main__":
    unittest.main()
