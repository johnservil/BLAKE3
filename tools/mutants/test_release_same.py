"""release_same.py's contract: python3 -m unittest discover -s tools/mutants.

A mutant that only deletes lines of code a build leaves out moves the
panic locations after it, so its tested build hashes otherwise; its
release build with the lines left blank hashes alike, and counts as same
code. A mutant that changes code does not, nor one whose diff fails."""

import difflib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import release_same  # noqa: E402


class KeepLines(unittest.TestCase):
    def test_deleted_lines_come_back_blank_after_the_first(self):
        original = "a\nfn f() {\n    x;\n    y;\n}\nz\n"
        mutated = "a\nfn f() {\n    () /* ~ changed by cargo-mutants ~ */\n}\nz\n"
        kept = release_same.keep_lines(original, mutated, 3)
        self.assertEqual(kept, "a\nfn f() {\n    () /* ~ changed by cargo-mutants ~ */\n\n}\nz\n")
        self.assertEqual(kept.count("\n"), original.count("\n"))
        self.assertEqual(kept.splitlines()[-1], "z")

    def test_a_mutant_that_keeps_or_adds_lines_is_unchanged(self):
        self.assertEqual(release_same.keep_lines("a\nb\n", "a\nc\n", 2), "a\nc\n")
        self.assertEqual(release_same.keep_lines("a\nb\n", "a\nc\nd\n", 2), "a\nc\nd\n")

    def test_a_start_outside_the_file_stops(self):
        with self.assertRaises(AssertionError):
            release_same.keep_lines("a\nb\nc\n", "a\n", 9)


LIB = """#[cfg(any())]
fn foreign(x: u32) -> u32 {
    let y = x + 1;
    let z = y * 2;
    z - 3
}

pub fn pick(v: &[u8], i: usize) -> u8 {
    v[i]
}

#[cfg(test)]
mod tests {
    #[test]
    fn picks() {
        assert_eq!(super::pick(&[1, 2], 1), 2);
    }
}
"""
CARGO = """[package]
name = "release-same-fixture"
version = "0.0.0"
edition = "2021"

[features]
serde = []
zeroize = []

[profile.mutants]
inherits = "release"
incremental = true
"""


def mutant_diff(new, name):
    """cargo-mutants' form: the file, then the mutant's name."""
    return "".join(difflib.unified_diff(LIB.splitlines(True), new.splitlines(True), "src/lib.rs", name))


@unittest.skipUnless(sys.platform == "linux" and shutil.which("cargo") and shutil.which("patch") and os.environ.get("MUTANTS_B3SUM"),
                     "Linux, cargo, patch, and MUTANTS_B3SUM: official b3sum 1.8.7 (check.py's official_b3sum installs it)")
class Confirm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.tree = os.path.join(cls.dir, "tree")
        os.makedirs(os.path.join(cls.tree, "src"))
        open(os.path.join(cls.tree, "Cargo.toml"), "w").write(CARGO)
        open(os.path.join(cls.tree, "src", "lib.rs"), "w").write(LIB)
        git = ["git", "-C", cls.tree, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null"]
        subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=cls.tree, check=True, capture_output=True)
        subprocess.run(["git", "init", "-q", cls.tree], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "fixture"], check=True)
        cls.target = os.path.join(cls.dir, "target")
        # MUTANTS_B3SUM in the shard's environment alone, as check.py sets it
        # (its own has none: a run on the EPYC stopped at a KeyError, BLAKE3 #218).
        cls.env = dict(os.environ)
        cls.b3sum = os.environ.pop("MUTANTS_B3SUM")
        foreign = LIB.replace("    let y = x + 1;\n    let z = y * 2;\n    z - 3\n", "    0 /* ~ changed by cargo-mutants ~ */\n")
        changed = LIB.replace("    v[i]\n", "    v[0] /* ~ changed by cargo-mutants ~ */\n")
        cls.places = [
            ("foreign", "src/lib.rs", mutant_diff(foreign, "replace foreign -> u32 with 0"), 3),
            ("changed", "src/lib.rs", mutant_diff(changed, "replace v[i] with v[0]"), 9),
            ("broken", "src/lib.rs", "--- src/lib.rs\n+++ nothing\n@@ -1,1 +1,1 @@\n-no such line\n+other\n", 1),
        ]
        try:
            cls.same = release_same.confirm(cls.tree, cls.target, cls.places, release_same_build_args(), cls.env)
        finally:
            os.environ["MUTANTS_B3SUM"] = cls.b3sum

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_compiled_out_lines_deleted_are_same_code(self):
        self.assertIn("foreign", self.same)

    def test_its_build_as_written_hashes_otherwise(self):
        # The premise: the deleted lines move pick's bounds-check location.
        path = os.path.join(self.tree, "src", "lib.rs")
        base = release_same.build(self.tree, self.target, release_same_build_args(), self.env)
        try:
            subprocess.run(["patch", "-s", "-p0", path], input=self.places[0][2], text=True, cwd=self.tree, check=True)
            written = release_same.build(self.tree, self.target, release_same_build_args(), self.env)
        finally:
            open(path, "w").write(LIB)
        self.assertNotEqual(written, base)

    def test_a_change_to_code_is_not(self):
        self.assertNotIn("changed", self.same)

    def test_a_diff_that_fails_is_not(self):
        self.assertNotIn("broken", self.same)

    def test_the_worktree_is_left_clean(self):
        self.assertTrue(release_same.clean(self.tree))
        with open(os.path.join(self.tree, "src", "lib.rs")) as f:
            self.assertEqual(f.read(), LIB)
        untracked = subprocess.run(["git", "-C", self.tree, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        self.assertEqual(untracked, "", "nothing left behind, a failed diff's rejects included")


def release_same_build_args():
    return ["--profile", "mutants", "--features", "serde,zeroize"]


if __name__ == "__main__":
    unittest.main()
