"""A survivor's second look for "same code" (check.py, on Linux): each
surviving place built again as a release build, with the mutant's line
count kept, every test binary hashed against the unmutated release build's
(code_hash.py). Where all of them hash alike, the mutant changes nothing a
user's build runs but line numbers, and check.py counts it as same code
(SAME_RELEASE_REASON), where its tested build had hashed otherwise.

Two things make a mutant's tested build hash otherwise when it changes no
code a release build has (BLAKE3 #219, November 3, 2026; x86-64):
- A mutant that deletes lines moves the location of every later panic in
  its file, and the locations sit in the read-only data the hash covers.
  cargo-mutants deletes lines when it replaces a function's body with a
  value: AArch64's prefetch-after-pause functions are compiled out on
  x86-64, yet each such mutant of theirs hashed otherwise, and stood as a
  "speed" survivor. Built with its deleted lines left blank, it hashes
  alike. The difference between the two builds is the line numbers alone,
  which no result depends on.
- The tested builds use the mutants profile, whose incremental build
  optimizes less (more codegen units, less inlining across them): a
  comparison a release build drops can stay in its code.
  SME2_SIZED_LEN's `16 * CHUNK_LEN` made `16 + CHUNK_LEN` changes the
  api_plan test binary with incremental on, and none with it off.

The release build is the mutants profile with incremental off
(CARGO_PROFILE_MUTANTS_INCREMENTAL=false): release's settings, the
features the tests build with. It builds in its own target directory, in
the shard's worktree, after cargo-mutants has left it; each mutant is
applied, built and taken back in turn, the worktree checked clean before
and after. A build that fails counts as other code.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_hash  # noqa: E402


def keep_lines(original, mutated, start_line):
    """`mutated` with as many lines as `original`: the lines the mutant
    deleted left blank, after the line it starts on (1-based). A mutant
    that adds or keeps lines comes back as it is."""
    lost = original.count("\n") - mutated.count("\n")
    if lost <= 0:
        return mutated
    lines = mutated.split("\n")
    assert 1 <= start_line <= len(lines), f"line {start_line} outside a {len(lines)}-line file"
    lines[start_line - 1] += "\n" * lost
    out = "\n".join(lines)
    assert out.count("\n") == original.count("\n")
    return out


def build(tree, target, build_args, env):
    """Each test binary's code hash, by its name without cargo's hash
    suffix, from a release build of `tree`; None if the build fails."""
    env = dict(env, CARGO_TARGET_DIR=target, CARGO_PROFILE_MUTANTS_INCREMENTAL="false")
    r = subprocess.run(["cargo", "test", *build_args, "--lib", "--tests", "--no-run", "--message-format=json"],
                       cwd=tree, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    exes = []
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            m = json.loads(line)
            if m.get("reason") == "compiler-artifact" and m.get("executable"):
                exes.append(m["executable"])
    assert exes, "a release build with no test binary"
    # The b3sum of the shard's environment: check.py sets it there, not in its own.
    return {os.path.basename(e).rsplit("-", 1)[0]: code_hash.code_hash(e, env["MUTANTS_B3SUM"]) for e in exes}


def clean(tree):
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=tree,
                          capture_output=True, text=True, check=True).stdout == ""


def confirm(tree, target, places, build_args, env):
    """The places of `places` whose release build, line count kept, hashes
    as the unmutated one's in every test binary. A place: (key, file, diff,
    start_line), diff being cargo-mutants' diff of the mutant."""
    if not places:
        return []
    assert clean(tree), f"{tree}: the worktree must be clean"
    base = build(tree, target, build_args, env)
    assert base, f"{tree}: the unmutated release build fails"
    same = []
    for key, file, diff, start_line in places:
        path = os.path.join(tree, file)
        original = open(path).read()
        try:
            r = subprocess.run(["patch", "-s", "-p0", "--no-backup-if-mismatch", "-r", os.devnull, path], input=diff, cwd=tree,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"release_same: {key}: its diff does not apply ({r.stdout.strip()} {r.stderr.strip()}); other code", flush=True)
                continue
            kept = keep_lines(original, open(path).read(), start_line)  # read before the write truncates
            open(path, "w").write(kept)
            if build(tree, target, build_args, env) == base:
                same.append(key)
        finally:
            open(path, "w").write(original)
    assert clean(tree), f"{tree}: a mutant left behind"
    return same
