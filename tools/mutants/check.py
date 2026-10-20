"""Mutation testing against a list of the mutants known to survive.

    pypy3 tools/mutants/check.py [--since REV] [--write]

cargo-mutants plants likely bugs in the files tools/mutants/known-<arch>.txt
names (its `files:` line), one at a time, and runs the library's and the
integration tests on each. Every mutant that survives them must be in the
list with its reason, and every listed mutant must still survive. So the
check fails on a new survivor (a missing test, or a new equivalent mutant
to list with its reason), on a listed one now caught (the list is
stale), and on a listed one whose reason is still UNCLASSIFIED. A mutant
is named by its file, function, and change, without its line: the same
change at several places in one function counts once per place. It runs
in git worktrees of HEAD of its own (SHARDS at once), so the checkout is
never mutated. With --write, the list is rewritten from this run, each
new mutant's reason UNCLASSIFIED and the others' kept.

With --since REV, a change's check: only the mutants in the lines the
change makes (git diff REV...HEAD, cargo-mutants --in-diff), and it fails
on a new survivor among them; a listed one now caught waits for a whole
run. Either way it fails while the list names a missing test ("OPEN: a
missing test"): that is work to do, not a survivor to keep. A change that removes or alters test code (a line of src/test.rs or
tests/, or one with an assertion or a test's name) can weaken the tests of
code it leaves alone, which only a whole run sees: then the whole run
happens instead, by itself.
"""

import collections
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
KNOWN = os.path.join(ROOT, "tools", "mutants", f"known-{platform.machine()}.txt")
NAME = re.compile(r"^(\S+?):\d+:\d+: (.*)$")


def read_known():
    files, known = [], {}
    for line in open(KNOWN):
        line = line.rstrip("\n")
        if line.startswith("files:"):
            files = line.split()[1:]
        elif line and not line.startswith("#"):
            file, mutant, count, reason = line.split("\t")
            known[f"{file}\t{mutant}"] = (int(count), reason)
    return files, known


SHARDS = 3  # runs at once, each in its own worktree and build
# The seconds a mutant's tests may take: the unmutated tests take 1.5 s on
# the VM and a caught mutant's at most 2.5 s (October 16), so a mutant
# still running at 10 s hangs.
TIMEOUT = 20
# Each process's address space, builds' and tests' alike: a mutant that
# allocates without end fails its test here, and leaves the machine's
# memory to the rest (one of 57 GB took the long proofs' memory).
MEMORY = 8 << 30


def cap_memory():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))


def run(files, extra=(), shards=SHARDS, label="run"):
    """The mutants that survive the library's tests, by name, counted, and
    every mutant's outcome by function. Each shard's results are kept in
    tmp/mutants-last/<label>/ for a look afterwards."""
    # The public features on, so the code behind them is tested too (the
    # build variants, pure and the like, are their own builds).
    # A mutant whose tests outlast TIMEOUT is a hang, which the check counts
    # as caught.
    args = ["cargo", "mutants", "--in-place", "--profile", "mutants", "--features", "serde,zeroize", "--timeout", str(TIMEOUT),
            *extra]
    for f in files:
        args += ["--file", f]
    runs = []
    for k in range(shards):
        tree = tempfile.mkdtemp(prefix="mutants-", dir="/tmp")
        subprocess.run(["git", "-C", ROOT, "worktree", "add", "-q", "--detach", tree, "HEAD"], check=True)
        # The library's and the integration tests; the doc tests stay out
        # (each compiles a program of its own: minutes per mutant).
        # Its own build directory, named outright (a cargo config's
        # target-dir would otherwise give the shards one, and each would
        # test the others' builds).
        env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(tree, "target"))
        shard = args + ["--shard", f"{k}/{shards}", "--output", os.path.join(tree, "mutants-result"), "--", "--lib", "--tests"]
        runs.append((tree, subprocess.Popen(shard, cwd=tree, env=env, preexec_fn=cap_memory)))
    names = collections.Counter()
    functions = collections.defaultdict(list)  # (file, function) -> [(outcome, name)]
    kept = os.path.join(ROOT, "tmp", "mutants-last", label)
    shutil.rmtree(kept, ignore_errors=True)
    for k, (tree, process) in enumerate(runs):
        # 0: every mutant caught; 2: some missed; 3: some timed out. Any
        # other status (a failing unmutated baseline) is no result.
        assert process.wait() in (0, 2, 3), f"cargo mutants failed ({process.returncode}) in {tree}"
        out = os.path.join(tree, "mutants-result", "mutants.out")
        # No mutants.out when there was nothing to test (a diff of no
        # mutants), so none of its files.
        missed = os.path.join(out, "missed.txt")
        for line in (open(missed).read().splitlines() if os.path.exists(missed) else []):
            m = NAME.match(line)
            assert m, f"a missed mutant cargo-mutants names unexpectedly: {line}"
            names[f"{m.group(1)}\t{m.group(2)}"] += 1
        outcomes = os.path.join(out, "outcomes.json")
        for outcome in (json.load(open(outcomes))["outcomes"] if os.path.exists(outcomes) else []):
            mutant = outcome["scenario"]
            if not isinstance(mutant, dict):
                continue  # the unmutated baseline
            mutant = mutant["Mutant"]
            function = (mutant.get("function") or {}).get("function_name", "")
            m = NAME.match(mutant["name"])
            functions[(mutant["file"], function)].append((outcome["summary"], f"{m.group(1)}\t{m.group(2)}"))
        if os.path.exists(out):
            shutil.copytree(out, os.path.join(kept, f"shard{k}"))
        subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", tree], check=True)
        shutil.rmtree(tree, ignore_errors=True)
    return names, functions


def confirm(found, known, differ=None):
    """`found` with every name whose count differs from the list's run
    again, alone: a test that fails by chance (under load, say) neither
    hides a survivor nor fails the check."""
    if differ is None:
        differ = sorted(k for k in set(found) | set(known) if found.get(k, 0) != known.get(k, (0, ""))[0])
    if not differ:
        return found
    patterns = [": " + re.escape(k.split("\t", 1)[1]) + "$" for k in differ]
    again, _ = run(sorted({k.split("\t", 1)[0] for k in differ}), [arg for p in patterns for arg in ("--re", p)], shards=1, label="confirm")
    confirmed = collections.Counter(found)
    for k in differ:
        confirmed[k] = again.get(k, 0)
    print(f"confirmed alone: {len(differ)} names whose count differed from the list")
    return +confirmed


def weakens_tests(diff):
    """Whether a diff removes or alters test code: a line it takes away (a
    changed line is one taken away and one added) in src/test.rs or
    tests/, or one holding an assertion or a test."""
    file = ""
    for line in diff.splitlines():
        if line.startswith("--- "):
            file = line[6:] if line.startswith("--- a/") else ""
        elif line.startswith("-") and not line.startswith("---"):
            if file == "src/test.rs" or file.startswith("tests/") or re.search(r"assert|#\[test\]|fn test_", line):
                return True
    return False


def figure(functions, known):
    """How much of the library mutation testing covers, by function: every
    mutant caught, every survivor explained, or one still open."""
    caught = explained = 0
    open_ = []
    for (file, function), outcomes in sorted(functions.items()):
        reasons = [known.get(name, (0, "UNCLASSIFIED"))[1] for summary, name in outcomes if summary == "MissedMutant"]
        if not reasons:
            caught += 1
        elif all(not r.startswith(("OPEN", "UNCLASSIFIED")) for r in reasons):
            explained += 1
        else:
            open_.append(f"{file} {function}")
    print(f"functions: {len(functions)}: {caught} every mutant caught, {explained} every survivor explained, {len(open_)} open; "
          "the assembly kernels are the proofs'")
    return open_


def main():
    files, known = read_known()
    # A test known to be missing is work to do now, not a survivor to keep:
    # the check fails while one is listed, whichever run this is.
    missing = sorted(k for k, (_, reason) in known.items() if reason.startswith("OPEN: a missing test"))
    for k in missing:
        print(f"MISSING TEST: {k}")
    if missing:
        sys.exit("mutants: the list names missing tests; write them first")
    since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else None
    if since is not None:
        diff = subprocess.run(["git", "-C", ROOT, "diff", f"{since}...HEAD", "--", *files], capture_output=True, text=True, check=True).stdout
        if weakens_tests(diff):
            print(f"mutants: the change since {since} removes or alters test code; the whole run")
            since = None
    if since is not None:
        if not diff:
            print(f"mutants: the change since {since} touches no mutated file")
            return
        path = os.path.join(tempfile.mkdtemp(prefix="mutants-diff-", dir="/tmp"), "change.diff")
        with open(path, "w") as f:
            f.write(diff)
        # One shard: a change's mutants are few, and each shard builds anew.
        found, functions = run(files, ["--in-diff", path], shards=1)
        # Only the names with more survivors than the list: the run holds
        # the change's lines alone, so a listed name's other places are
        # missing from it (a whole run sees those, and stale entries).
        found = confirm(found, known, sorted(k for k in found if found[k] > known.get(k, (0, ""))[0]))
        new = {k: n for k, n in found.items() if n > known.get(k, (0, ""))[0]}
        for k, n in sorted(new.items()):
            print(f"NEW SURVIVOR ({n - known.get(k, (0, ''))[0]}): {k}")
        print(f"mutants: the change since {since}: {sum(1 for f in functions.values() for _ in f)} mutants, "
              + ("no new survivor" if not new else f"{len(new)} names with a new survivor"))
        sys.exit(1 if new else 0)
    found, functions = run(files)
    found = confirm(found, known)
    figure(functions, {k: known.get(k, (n, "UNCLASSIFIED")) for k, n in found.items()})
    if "--write" in sys.argv:
        header = [l.rstrip("\n") for l in open(KNOWN) if l.startswith("#") or l.startswith("files:")]
        rows = [f"{k}\t{n}\t{known.get(k, (0, 'UNCLASSIFIED'))[1]}" for k, n in sorted(found.items())]
        open(KNOWN, "w").write("\n".join(header + rows) + "\n")
        print(f"{KNOWN}: {sum(found.values())} survivors, {sum(1 for k in found if k not in known)} names new")
        return
    unexplained = sorted(k for k, (_, reason) in known.items() if reason == "UNCLASSIFIED")
    for k in unexplained:
        print(f"NO REASON: {k}")
    new = {k: n for k, n in found.items() if n > known.get(k, (0, ""))[0]}
    # An OPEN pool entry is caught or not as the run's scheduling falls
    # (the list says why): its count is the most that survive, and a run
    # that catches more of them is no change to the list.
    stale = {k: c for k, (c, reason) in known.items() if found.get(k, 0) < c and not (k.startswith("src/lanes.rs") and reason.startswith("OPEN: the pool's scheduling"))}
    for k, n in sorted(new.items()):
        print(f"NEW SURVIVOR ({n - known.get(k, (0, ''))[0]}): {k}")
    for k, c in sorted(stale.items()):
        print(f"NOW CAUGHT ({c - found.get(k, 0)}): {k}")
    ok = not new and not stale and not unexplained
    print(f"mutants: {sum(found.values())} survive, each known and explained" if ok else "mutants: the list and this run differ, or a listed mutant has no reason")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
