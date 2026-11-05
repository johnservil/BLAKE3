"""Mutation testing against a list of the mutants known to survive.

    python3 tools/mutants/check.py [--since REV] [--write] [--map] [--whole-suite] [--all-hangs]

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
never mutated. A `model files:` line names the files whose mutants the
model checker judges beside the suite (tests/shuttle_*.rs: Shuttle, over
the pool's handshakes, thousands of schedules in seconds): a wait that
never ends is a deadlock it reports, not a time-out. With --write, the list is rewritten from this run, each
new mutant's reason UNCLASSIFIED and the others' kept.

A mutant whose build hashes as the unmutated build's, every test binary
alike (code_hash.py: every byte the loader maps), changes nothing this
machine runs: code for another architecture, or one the optimizer erases.
On Linux the test runner (test-runner) sees it while nextest lists the
binaries, and its first test stops at once, so it costs a build and no
test. It is a survivor of its own kind, "same code": --write gives its
row the reason SAME_REASON, and a run fails (NO LONGER SAME CODE) on a
row with that reason whose mutant now builds to other code. A survivor
whose tested build hashes otherwise gets a second look (release_same.py):
built as a release build with the lines it deletes left blank, it may hash
alike, since a deleted line moves every later panic's location and the
tested builds' incremental optimization keeps code a release build drops.
It is then same code too (SAME_RELEASE_REASON), at about a release
build's cost a surviving place.

A mutant whose tests no test fails, but the clock stops (nextest's, or
cargo-mutants' backstop), has an outcome of its own: it timed out. That
is neither caught nor survived: the tests did not finish, which says
nothing of what they would have found (a loop that never ends, a wait
that never returns; or a test the load made slow). Time-outs have their
own list, tools/mutants/timeouts-<arch>.txt, in KNOWN's form, and are
checked against it the same way (NEW TIME-OUT, NO LONGER TIMES OUT).
A whole run skips the known hangs (known_hangs: every place of the name a
time-out, listed for this CPU with its reason, outside the model files):
each would only wait for the clock again to say what its row says, and they
were most of a run's test time (64% on an EPYC 9255, BLAKE3 #244). Their
rows stand as listed; --since runs them on the lines a change touches, and
--write (which must confirm every row it keeps) and --all-hangs run them.
Before anything runs, every row of every architecture's lists must name a
mutant the source holds now (cargo mutants --list, which any machine reads
for every target): a row a run on older code left stops the check, named.
Exit status: 0 both lists match; 1 a survivor differs from its list (or a
reason is missing, or a row is stale); 2 only the time-outs differ (no
verdict on those).

The tests run under cargo-nextest (.config/nextest.toml), each in its own
process, a test killed at 30 s as a hang; a mutant's tests stop at the
first that fails, the others ended at once (SIGTERM, no outcome of theirs). Every run records each test's outcome and time on every
mutant, by its place (one change at several places is a mutant at each),
and on the unmutated tree (tmp/mutants-last/<run>/tests.tsv). With
--map, every test runs on every mutant (the `mutants-map` profile, for when
the tests change), and the run weighs the tests by cost: the cheapest set
of tests that still catches every caught mutant, the tests outside it
(each a big test whose catches cheaper ones make), and the mutants only an
expensive test catches (each a small test worth writing).

With --since REV, a change's check: the mutants in the lines the change
makes (git diff REV...HEAD, cargo-mutants --in-diff), and it fails on a new
survivor among them; a listed one now caught waits for a whole run. Either
way it fails while the list names a missing test ("OPEN: a missing test"):
that is work to do, not a survivor to keep. A change that alters a test (a
line taken away inside a #[test] function, comments aside) can lose the
catches of code it leaves alone: the mutants the latest map (a --map run of
an ancestor) records that test catching, the startup self-test's aside, run
again too. A change to other test code (a helper, a test module's or
tests/'s own items), or one with no such map, runs the whole check.
"""

import collections
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import release_same

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
KNOWN = os.path.join(ROOT, "tools", "mutants", f"known-{platform.machine()}.txt")
# The mutants (by name, counted in places, as KNOWN) that no test catches but
# the clock stops: a loop that never ends, a wait that never returns. Their
# own outcome, never folded into caught or survived: a time-out says the
# tests did not finish, not what they would have found. Rows as KNOWN's, CPU
# conditions too.
TIMEOUTS = os.path.join(ROOT, "tools", "mutants", f"timeouts-{platform.machine()}.txt")
NAME = re.compile(r"^(\S+?):\d+:\d+: (.*)$")


def cpu_flags():
    """The CPU's feature flags as the OS names them (Linux: /proc/cpuinfo's
    flags or Features line; macOS: sysctl's hw.optional names set to 1)."""
    if sys.platform == "linux":
        with open("/proc/cpuinfo") as info:
            for line in info:
                key, sep, value = line.partition(":")
                if sep and key.strip() in ("flags", "Features"):
                    flags = set(value.split())
                    break
            else:
                raise ValueError("/proc/cpuinfo has no flags or Features line")
        # One fact cpuinfo has as fields, not a flag: an AMD CPU of family
        # 0x1A (Zen 5) or later, where the wide AVX-512 kernels run
        # (platform::wide_cpu); the rows for their dispatch depend on it.
        with open("/proc/cpuinfo") as info:
            fields = dict(line.partition(":")[::2] for line in info if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields.get("vendor_id") == "AuthenticAMD" and fields.get("cpu family", "").isdigit() and int(fields["cpu family"]) >= 0x1A:
            flags.add("amd_family_ge_1a")
        return flags
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "hw.optional"], capture_output=True, text=True, check=True).stdout
        rows = [line.partition(":") for line in out.splitlines()]
        if not rows or any(not sep or not key.startswith("hw.optional.") or not value.strip()
                           for key, sep, value in rows):
            raise ValueError("sysctl hw.optional returned missing or malformed feature data")
        return {key.rsplit(".", 1)[-1] for key, _, value in rows if value.strip() == "1"}
    raise RuntimeError(f"CPU feature reader unavailable on {sys.platform}")


# A row whose survival depends on the CPU, not only the architecture: its
# reason begins "[cpu has FLAG ...] " or "[cpu lacks FLAG ...] " (flags as
# cpu_flags() names them), and the row applies only where that holds (an
# EPYC 9255, October 22: AVX-512's detection mutants survive where the CPU
# has it, and are caught on the i7, where it is absent, and the reverse).
CONDITION = re.compile(r"^\[cpu (has|lacks) ([A-Za-z0-9_.]+(?: [A-Za-z0-9_.]+)*)\] \S")
FLAGS = cpu_flags()


def applies(reason):
    m = CONDITION.match(reason)
    if reason.lstrip().startswith("[") and m is None:
        raise ValueError(f"malformed CPU condition: {reason!r}")
    return m is None or all(f in FLAGS for f in m.group(2).split()) == (m.group(1) == "has")


def read_known(path=None, inactive=None):
    """A list's files and the rows that apply on this CPU; the rest are
    kept in `inactive` (INACTIVE for KNOWN), for --write. A list that does
    not exist yet (TIMEOUTS, on a new machine) has no rows."""
    path, inactive = (KNOWN, INACTIVE) if path is None else (path, inactive)
    files, known = [], {}
    if path == KNOWN:
        LOOM_FILES.clear()
    if not os.path.exists(path):
        return files, known
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("files:"):
            files = line.split()[1:]
        elif line.startswith("model files: "):
            # The files whose mutants the model checker judges beside the
            # suite (tests/shuttle_*.rs), in place of the suite's clock.
            LOOM_FILES.extend(line.split()[2:])
        elif line and not line.startswith("#"):
            file, mutant, count, reason = line.split("\t")
            if applies(reason):
                name = f"{file}\t{mutant}"
                assert name not in known, f"{name}: two rows apply on this CPU"
                known[name] = (int(count), reason)
            else:
                inactive.append(line)
    return files, known


# The rows for other CPUs, as written (a name may have one row a CPU).
INACTIVE = []
LOOM_FILES = []
INACTIVE_TIMEOUTS = []


MAP = "--map" in sys.argv
# Each mutant's tests are chosen by the lines it changes (the map of
# tools/tests/reach.py for HEAD's tree, made by `reach.py map` at HEAD, as
# the pre-commit hook keeps it); the whole suite where no test ran them,
# the rule says so, or there is no map. The verdicts are the suite's (one
# run of 1,070 mutants agreed on every row but the two whose whole-suite
# outcome is a hang that narrowing avoids, October 23); the shards run
# in two thirds of the time. --whole-suite runs every test on every
# mutant, as before.
REACH = "--whole-suite" not in sys.argv


# Runs at once, each in its own worktree and build: on the VM (16 CPUs),
# a whole run took 47 minutes at 6 and about 60 at 3 (October 20).
SHARDS = 6
# The host's target triple, for its target runner's variable.
HOST = next(l.split(": ", 1)[1] for l in subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True).stdout.splitlines() if l.startswith("host: "))
# The target directory cargo is set to use (a cargo config's target-dir, or
# the checkout's target/): the shards build inside it.
TARGET_DIR = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout)["target_directory"]
# cargo-mutants needs a limit on a mutant's tests (with --in-place it would
# otherwise take 300 s). The run's one clock is nextest's, a test's 30 s
# (.config/nextest.toml, profile mutants), which ends a hanging test first;
# this is a backstop far beyond it, and its firing stops the check as an
# error, never a verdict (it overlapped the 5 s clock, so a mutant's
# outcome hung on which ran out first: an EPYC 9255, October 22).
BACKSTOP = 3600
# How every mutant builds (and release_same's release build, with the
# profile's incremental off): the public features on, so the code behind
# them is tested too (the build variants, pure and the like, are their own builds).
BUILD = ["--profile", "mutants", "--features", "serde,zeroize"]
# Each process's memory, builds' and tests' alike: a mutant that allocates
# without end fails its test here, and leaves the machine's memory to the
# rest (one of 57 GB took the long proofs' memory). The limit is on data
# (heap and anonymous mappings, RLIMIT_DATA), which is what a runaway test
# takes; address space would count the linker's reservations too, and a
# 2.2 GiB address-space cap crashed rust-lld (i7, October 23). At most
# 8 GiB, and at most the machine's memory less 4 GiB shared among the
# shards, since SHARDS such mutants at once must fit: six at 8 GiB killed
# a 14 GiB machine twice (October 23), once the 30 s clock let a runaway
# test grow that far. A fresh build and the suite pass under 2.25 GB.
def physical_memory():
    """The machine's memory in bytes: sysconf's pages on Linux; macOS has no
    SC_PHYS_PAGES, and names it hw.memsize (the Mac's mutants job stopped
    in each shard's preexec_fn without it, runner job 1865)."""
    if sys.platform == "darwin":
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def memory_cap(shards):
    return min(8 << 30, (physical_memory() - (4 << 30)) // shards)


def test_limits(shards, cpus, total):
    """(test threads a shard, each test process's data cap in bytes). A
    shard runs cpus / shards tests at once, so the run keeps every CPU
    busy as one nextest per CPU did, and shards x threads test processes
    at their cap fit in the machine's memory less 4 GiB. One per CPU in
    each shard was 120 test processes on the i7, each allowed the builds'
    1.8 GiB: a runaway mutant's tests, several at once in each shard, took
    the machine to 1.4 GiB free (gate22, October 24), where no honest test
    holds more than 41 MiB (the suite, measured the same day)."""
    threads = max(1, cpus // shards)
    return threads, min(memory_cap(shards), (total - (4 << 30)) // (shards * threads))


def capped(shards):
    """The shard's preexec_fn, which caps its data on Linux; None on macOS,
    which enforces no RLIMIT_DATA on memory a program maps (its malloc's),
    and where setting it stopped every shard (runner jobs 1865-1867)."""
    if sys.platform != "linux":
        return None
    cap = memory_cap(shards)
    return lambda: cap_memory(cap)


def cap_memory(cap):
    """In a shard's child before it runs (preexec_fn): its data limit, `cap`
    bytes, computed in the parent (memory_cap): a child forked from a
    threaded parent runs nothing else safely, a subprocess least of all
    (macOS's sysctl there stopped every shard, runner job 1866)."""
    import resource
    resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))


def with_model(label, found, timed):
    """`found` and `timed` with the model files' names counted again place
    by place, from `label`'s run and the model's ("loom"): a place is
    caught when either catches it (the model runs the handshake's
    schedules; the suite has the standard library's tests of it, such as
    a spurious wake). A name counts per place, never by its totals: two
    places of one name may fall each to a different side (Finish::wait's
    deleted !: the model catches one, the other survives both; counted
    by name, the caught one stood as a time-out, gate20, October 24)."""
    found, timed = collections.Counter(found), collections.Counter(timed)
    outcomes, model = PLACE_OUTCOMES.get(label, {}), PLACE_OUTCOMES.get("loom", {})
    names = {n for n, _ in outcomes if n.split("\t", 1)[0] in LOOM_FILES}
    for n in names:
        found[n] = timed[n] = 0
    for (n, place), summary in outcomes.items():
        if n not in names or summary == "CaughtMutant" or model.get((n, place)) == "CaughtMutant":
            continue
        if summary == "MissedMutant":
            found[n] += 1
        else:
            timed[n] += 1
    return +found, +timed


def with_loom(found, timed, functions, loom_functions):
    """The suite's outcomes with the model's for the model files' mutants
    (with_model), and the model's functions beside the suite's."""
    found, timed = with_model("run", found, timed)
    return found, timed, {**functions, **loom_functions}


def official_b3sum():
    """Official BLAKE3's b3sum 1.8.7, the hash code_hash.py identifies a
    test binary's code by: installed once into this machine's target
    directory, so the identity of the code under test never rests on that
    code's own hashing."""
    root = os.path.join(TARGET_DIR, "b3sum-official-1.8.7")
    path = os.path.join(root, "bin", "b3sum")
    if not os.path.exists(path):
        subprocess.run(["cargo", "install", "b3sum", "--version", "=1.8.7", "--locked", "--root", root], check=True)
    version = subprocess.run([path, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    assert version == "b3sum 1.8.7", f"{path} is {version!r}, not official b3sum 1.8.7"
    return path


def scratch_dir():
    """tmp/ in the repository (git ignores it), made if absent."""
    path = os.path.join(ROOT, "tmp")
    os.makedirs(path, exist_ok=True)
    return path


# The shards' worktrees: sources alone (their builds go under TARGET_DIR),
# a few MB each, in the system's temporary directory. In the repository's
# tmp/ they sat on the VM's shared mount, which runs nothing and fails to
# delete a large tree ("Directory not empty"): a run lost its verdict so
# on October 24.
WORKTREES = os.path.join(tempfile.gettempdir(), "blake3-servil-mutants")


def remove_leftovers():
    """The worktrees and builds an earlier run left (a crash, a stop by
    hand: a run removes none of its own, crash-only), removed before this
    run starts, so a failure to remove them costs no run. One run at a
    time: none of them is live."""
    listed = subprocess.run(["git", "-C", ROOT, "worktree", "list", "--porcelain"], capture_output=True, text=True, check=True).stdout
    home = os.path.realpath(WORKTREES) + os.sep
    for line in listed.splitlines():
        tree = line.removeprefix("worktree ")
        if line.startswith("worktree ") and os.path.realpath(tree).startswith(home):
            subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", tree], check=True)
    subprocess.run(["git", "-C", ROOT, "worktree", "prune"], check=True)
    shutil.rmtree(WORKTREES, ignore_errors=True)
    shutil.rmtree(os.path.join(TARGET_DIR, "mutants"), ignore_errors=True)
    os.makedirs(WORKTREES)


def run(files, extra=(), shards=SHARDS, label="run", loom=False):
    """The mutants that survive the library's tests, by name, counted, and
    every mutant's outcome by function. Each shard's results are kept in
    tmp/mutants-last/<label>/ for a look afterwards."""
    # A hang is nextest's to end (a test's 30 s); BACKSTOP only guards the run.
    args = ["cargo", "mutants", "--in-place", *BUILD, "--timeout", str(BACKSTOP),
            "--test-tool", "nextest", *extra]
    for f in files:
        args += ["--file", f]
    runs = []
    envs = {}  # each shard's environment, by worktree
    survivors = {}  # each shard's surviving places, by worktree: release_same's
    for k in range(shards):
        # Its sources in WORKTREES; its build under TARGET_DIR, on disk
        # where cargo builds (six worktrees' release builds in a tmpfs
        # /tmp filled a 14 GiB machine: i7, three times, October 22-23).
        tree = tempfile.mkdtemp(prefix="mutants-", dir=WORKTREES)
        subprocess.run(["git", "-C", ROOT, "worktree", "add", "-q", "--detach", tree, "HEAD"], check=True)
        # The checkout's lock (git ignores it): the shard builds what the
        # checkout builds, from the crates already downloaded, without the
        # network (a slow crates.io failed a run's baseline, October 24).
        if os.path.exists(os.path.join(ROOT, "Cargo.lock")):
            shutil.copy(os.path.join(ROOT, "Cargo.lock"), tree)
        # The library's and the integration tests; the doc tests stay out
        # (each compiles a program of its own: minutes per mutant).
        # Its own build directory, named outright (a cargo config's
        # target-dir would otherwise give the shards one, and each would
        # test the others' builds), inside the target directory cargo is
        # set to use: where this machine runs what it builds (the VM's
        # shared mount runs nothing, so its cargo builds in /tmp).
        target = os.path.join(TARGET_DIR, "mutants", os.path.basename(tree))
        env = dict(os.environ, CARGO_TARGET_DIR=target, NEXTEST_PROFILE="mutants-map" if MAP else "mutants")
        # The tests' own threads and memory cap (test_limits), the cap set
        # by a target runner for test binaries alone; builds keep theirs.
        threads, test_cap = test_limits(shards, os.cpu_count(), physical_memory())
        env["NEXTEST_TEST_THREADS"] = str(threads)
        env["MUTANTS_TEST_DATA_KIB"] = str(test_cap >> 10)
        # Through sh: a checkout's files may sit where nothing executes.
        env[f"CARGO_TARGET_{HOST.upper().replace('-', '_')}_RUNNER"] = "sh " + os.path.join(ROOT, "tools", "mutants", "test-runner")
        if sys.platform == "linux":  # code_hash.py reads ELF
            env["MUTANTS_SAME_CODE"] = os.path.join(target, "same-code")
            os.makedirs(env["MUTANTS_SAME_CODE"])
            env["MUTANTS_CODE_HASH"] = os.path.join(ROOT, "tools", "mutants", "code_hash.py")
            env["MUTANTS_B3SUM"] = official_b3sum()
            env["MUTANTS_SAME_CODE_MARKER"] = SAME_CODE_MARKER
        if loom:
            # The model checker's build of the same files (tests/shuttle_*.rs
            # compile them beside Shuttle's primitives): every schedule it
            # tries, in place of the suite's one schedule and its clock.
            env["RUSTFLAGS"] = os.environ.get("RUSTFLAGS", "") + " --cfg shuttle"
        elif REACH:
            # Each mutant's tests alone: tools/mutants/cargo-nextest, first
            # on PATH, narrows cargo-mutants' nextest run to the tests that
            # ran the mutated lines (the map tools/tests/reach.py keeps for
            # HEAD's tree), the whole suite where the rule says so.
            env["PATH"] = os.path.join(tree, "tools", "mutants") + os.pathsep + env.get("PATH", "")
            env["MUTANTS_REACH"] = "1"
            env["TESTS_MAP_DIR"] = os.path.join(ROOT, "tmp", "tests-map")
        # Consecutive slices (cargo-mutants' default): a shard's mutants in one
        # file one after another rebuild less (round-robin evened the shards
        # and made each build 10% slower, the run no faster, October 21).
        tests = ["--test", "shuttle_finish", "--test", "shuttle_park"] if loom else ["--lib", "--tests"]
        shard = args + ["--shard", f"{k}/{shards}", "--output", os.path.join(tree, "mutants-result"), "--", *tests]
        runs.append((tree, subprocess.Popen(shard, cwd=tree, env=env, preexec_fn=capped(shards))))
        envs[tree] = env
    names = collections.Counter()
    timed = collections.Counter()  # the time-outs, by name: their own outcome
    functions = collections.defaultdict(list)  # (file, function) -> [(outcome, name)]
    rows = []  # (mutant name, function, test, outcome, ms); the unmutated tree's mutant name is UNMUTATED
    kept = os.path.join(ROOT, "tmp", "mutants-last", "map" if MAP and label == "run" else label)
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
            log = os.path.join(out, outcome["log_path"]) if outcome.get("log_path") else None
            if not isinstance(mutant, dict):
                rows += [(UNMUTATED, "", *t) for t in read_tests(log)]  # the unmutated baseline
                continue
            mutant = mutant["Mutant"]
            function = (mutant.get("function") or {}).get("function_name", "")
            m = NAME.match(mutant["name"])
            name = f"{m.group(1)}\t{m.group(2)}"
            summary = outcome["summary"]
            if summary == "Timeout":
                # cargo-mutants' own limit: nextest's clock should have come
                # first. Still a time-out, never an error or a catch.
                print(f"BACKSTOP: the {BACKSTOP} s limit ended {mutant['name']}'s tests, not nextest's clock (.config/nextest.toml)")
                timed[name] += 1
            place = mutant["name"].split(": ", 1)[0].split(":", 1)[1]
            if summary == "CaughtMutant" and SAME_CODE_MARKER in open(log, errors="replace").read():
                # Its build hashes as the unmutated one: it changes nothing
                # here, a survivor no test ran on (test-runner).
                names[name] += 1
                SAME[label][name] += 1
                PLACE_OUTCOMES.setdefault(label, {})[(name, place)] = "MissedMutant"
                functions[(mutant["file"], function)].append(("MissedMutant", name))
                continue
            if summary in ("CaughtMutant", "MissedMutant", "Timeout"):
                # The mutant's place too: one name can be several places,
                # each caught (or not) by tests of its own.
                tests = read_tests(log)
                rows += [(f"{name}\t{place}", f"{mutant['file']} {function}", *t) for t in tests]
                if summary == "CaughtMutant" and clock_only(tests):
                    # No test failed: the clock stopped one. Timed out, not caught.
                    summary = "TimedOut"
                    timed[name] += 1
                    CLOCK_ONLY.add(f"{name}\t{place}")
                    STOPPED.setdefault(name, set()).update(t for t, o, _ in tests if o == "TIMEOUT")
                PLACE_OUTCOMES.setdefault(label, {})[(name, place)] = summary
            functions[(mutant["file"], function)].append((summary, name))
            if summary == "MissedMutant" and sys.platform == "linux" and not loom:
                # A survivor whose tested build hashed otherwise: its
                # release build's turn (release_same, after the shards).
                survivors.setdefault(tree, []).append(((name, place), mutant["file"], open(os.path.join(out, outcome["diff_path"])).read(),
                                                       mutant["span"]["start"]["line"]))
        if os.path.exists(out):
            shutil.copytree(out, os.path.join(kept, f"shard{k}"))
    if survivors:
        # Each shard's survivors in its own worktree and release target, the
        # shards at once (their cargo-mutants runs are over).
        def look(tree):
            target = os.path.join(TARGET_DIR, "mutants", os.path.basename(tree) + "-release")
            return release_same.confirm(tree, target, survivors[tree], BUILD, envs[tree])
        with ThreadPoolExecutor(len(survivors)) as pool:
            same = [key for keys in pool.map(look, sorted(survivors)) for key in keys]
        for name, _ in same:
            SAME[label][name] += 1
            RELEASE_SAME[label][name] += 1
        print(f"mutants: {len(same)} of {sum(map(len, survivors.values()))} surviving places build to the same code in a release build, "
              "their line count kept (release_same.py)", flush=True)
    if rows:
        os.makedirs(kept, exist_ok=True)
        with open(os.path.join(kept, "tests.tsv"), "w") as f:
            f.write(HEADER)
            f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
        if MAP and label == "run":  # the confirming pass reruns a few mutants: its weights would be noise
            weigh(rows, os.path.join(kept, "tests.tsv"))
            head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
            open(os.path.join(kept, "commit"), "w").write(head)
    return names, functions, timed


# Same code: a mutant whose every test binary hashes as the unmutated
# build's (test-runner) exits its first test at once with the marker.
SAME_CODE_MARKER = "mutants: same code: every test binary hashes as the unmutated build's"
SAME_REASON = "same code: every test binary hashes as the unmutated build's here (tools/mutants/code_hash.py)"
# The places a name had of it, by run label; confirm() replaces a
# rerun name's "run" count with its "confirm" one, as it does survivors'.
SAME = collections.defaultdict(collections.Counter)
# Of those, the places only a release build with the line count kept shows
# as same code (release_same.py): a row all of whose places are same code,
# some of them so, has the reason SAME_RELEASE_REASON.
RELEASE_SAME = collections.defaultdict(collections.Counter)
SAME_RELEASE_REASON = ("same code: every test binary of a release build, the mutant's line count kept, "
                       "hashes as the unmutated build's here (tools/mutants/release_same.py)")
SAME_REASONS = (SAME_REASON, SAME_RELEASE_REASON)
# The places that no test failed, but the clock stopped (listed by name in
# TIMEOUTS): a hang, or a test made slow by the mutant, or by the machine's
# load (a mutant compiled out of the build was stopped so once, at 5 s).
CLOCK_ONLY = set()
# Each run's outcome a place, by label: {(name, place): summary}, for
# with_model's count place by place.
PLACE_OUTCOMES = {}
# The tests the clock stopped, a timed-out name's: a new row's placeholder
# names them, so its reason starts from what stopped (October 24).
STOPPED = {}


def open_pool(name, reason):
    """A row whose outcome the pool's schedules decide, run by run (its
    reason begins OPEN: the pool's scheduling): a run that catches it is
    one draw, so the row stays."""
    return name.startswith("src/lanes.rs") and reason.startswith("OPEN: the pool's scheduling")


def elsewhere(name, lists=None):
    """What the other machines' lists say of mutant `name`: one entry per
    list naming it, "on ARCH: survives|times out (FILE): REASON", in a fixed
    order. Which mutants exist is the source's alone, so a mutant named on
    another machine's list is this one; its outcome there is a fact about
    the code, kept beside this machine's, never in place of it."""
    here = platform.machine()
    if lists is None:
        lists = sorted(glob.glob(os.path.join(ROOT, "tools", "mutants", "known-*.txt")) +
                       glob.glob(os.path.join(ROOT, "tools", "mutants", "timeouts-*.txt")))
    out = []
    for path in lists:
        base = os.path.basename(path)
        kind, arch = base[:-4].split("-", 1)
        if arch == here:
            continue
        for line in open(path):
            if line.startswith("#") or line.startswith(("files:", "model files:")) or not line.strip():
                continue
            file, mutant, _, reason = line.rstrip("\n").split("\t", 3)
            if f"{file}\t{mutant}" == name and not reason.startswith("UNCLASSIFIED"):
                out.append(f"on {arch}: {'survives' if kind == 'known' else 'times out'} ({base}): {reason}")
    return out


def unclassified(name, outcome=None, lists=None):
    """A new row's reason. Where another machine's list explains the same
    mutant, its outcome and reason, each named by machine: the same
    outcome there makes it this row's reason ("as on ARCH"); a different
    one makes it an OPEN row naming both, a finding the run reports
    (MACHINES DIFFER). Where no machine knows, UNCLASSIFIED, and for a
    time-out the tests the clock stopped, until someone writes one."""
    known = elsewhere(name, lists)
    if known and outcome:
        mine = f"on {platform.machine()}: {outcome}"
        if all(f": {outcome} (" in k for k in known):
            return f"as {'; '.join(known)}"
        return f"OPEN: the machines differ: {mine}; " + "; ".join(known)
    stopped = STOPPED.get(name)
    return "UNCLASSIFIED" + (f" (the clock stopped: {', '.join(sorted(stopped))})" if stopped else "")

UNMUTATED = "(unmutated)\t\t"
HEADER = "# mutant (file, change, line:column)\tfunction\ttest\toutcome\tms\n"  # the record's format
SELF_TEST = "(the startup self-test)"
STATUS = re.compile(r"^\s+(PASS|FAIL|TIMEOUT|SIG[A-Z]+|ABORT)\s+\[\s*([\d.]+)s\]\s+(?:\(\s*\d+/\d+\)\s+)?(\S+)\s+(\S+)")


def clock_only(tests):
    """Whether the clock alone stopped a mutant's tests: some test did not
    pass, and each such ran to the clock (TIMEOUT). SIGTERM is a test
    nextest ended because another had failed or timed out (the mutants
    profile's fail-fast ends the rest at once): no outcome of its own, since
    nothing in a mutant sends it."""
    failed = [o for _, o, _ in tests if o not in ("PASS", "SIGTERM")]
    return bool(failed) and all(o == "TIMEOUT" for o in failed)


def read_tests(log):
    """Each test a cargo-nextest log ran: (test, outcome, ms). A failure
    whose output names the startup self-test is the self-test's catch,
    whichever test it ran in (a broken build fails it in every test)."""
    tests, last = [], None
    for line in open(log, errors="replace"):
        m = STATUS.match(line)
        if m:
            last = [f"{m.group(3)} {m.group(4)}", m.group(1), round(float(m.group(2)) * 1000)]
            tests.append(last)
        elif last and last[1] != "PASS" and "startup self-test failed" in line:
            last[0] = SELF_TEST
    return [tuple(t) for t in tests]


def weigh(rows, path):
    """The tests weighed by cost, from a run's record: the cheapest set of
    tests that catches every caught mutant (greedy, each step the test
    with the most uncaught mutants per millisecond), the tests outside it,
    and the mutants only an expensive test catches. Small tests are cheap,
    so they come first; a test outside the set is a reading (a test's
    worth goes beyond the mutants it catches: a contract, a shape no
    mutant models), and so is a mutant to give a small test."""
    cost = {}
    for name, _, test, outcome, ms in rows:
        if name == UNMUTATED and outcome == "PASS":
            cost[test] = min(cost.get(test, ms), ms)
    floor = max(1, min(cost.values(), default=1))
    cost[SELF_TEST] = floor  # it runs as every test process starts
    catches, function_of = collections.defaultdict(set), {}
    for name, function, test, outcome, _ in rows:
        if name != UNMUTATED and outcome != "PASS":
            catches[test].add(name)
            function_of[name] = function
    price = lambda t: max(floor, cost.get(t, 5000))
    uncovered, cover = set().union(*catches.values()) if catches else set(), []
    while uncovered:
        best = max(catches, key=lambda t: (len(catches[t] & uncovered) / price(t), -price(t), t))
        if not catches[best] & uncovered:
            break
        cover.append(best)
        uncovered -= catches[best]
    print(f"tests: {len(cost) - 1} take {sum(v for t, v in cost.items() if t != SELF_TEST)} ms unmutated; "
          f"{len(cover)} of them, {sum(price(t) for t in cover)} ms, catch every caught mutant "
          f"({len(function_of)} places; the startup self-test {len(catches[SELF_TEST])}) ({path})")
    for t in sorted(set(catches) - set(cover), key=lambda t: -price(t))[:10]:
        print(f"  outside the cheapest set: {price(t)} ms {t} ({len(catches[t])} catches, each made by cheaper tests)")
    cheapest = {n: min(price(t) for t in catches if n in catches[t]) for n in function_of}
    dear = collections.Counter(function_of[n] for n, ms in cheapest.items() if ms >= 100)
    for function, n in dear.most_common(10):
        print(f"  caught only by tests of 100 ms or more: {n} mutants in {function}")


def list_rows(paths):
    """Every row of the lists at `paths`, as (list, file, mutant)."""
    rows = []
    for path in paths:
        for line in open(path):
            if line.strip() and not line.startswith(("#", "files:", "model files:")):
                file, mutant = line.split("\t")[:2]
                rows.append((os.path.basename(path), file, mutant))
    return rows


def stale_rows(rows, listed):
    """The rows naming a mutant that `listed` (cargo mutants --list's
    lines, "file:line:column: mutant") lacks."""
    exist = {(m.group(1), m.group(2)) for m in (re.match(r"(\S+?):\d+:\d+: (.*)$", l) for l in listed) if m}
    return [r for r in rows if (r[1], r[2]) not in exist]


def check_rows_exist():
    """Every row of every architecture's lists names a mutant the source
    holds now. Which mutants exist is the source's alone: cargo mutants
    --list reads it unbuilt, every target's files alike, so any machine
    checks every list. The outcomes and reasons are the runs' and people's.
    A list a run on other code wrote stops the check here, by row (26 of
    the AArch64 list's rows named mutants gone, unnoticed: October 26)."""
    paths = sorted(glob.glob(os.path.join(ROOT, "tools", "mutants", "known-*.txt"))
                   + glob.glob(os.path.join(ROOT, "tools", "mutants", "timeouts-*.txt")))
    rows = list_rows(paths)
    files = sorted({f for _, f, _ in rows})
    args = ["cargo", "mutants", "--list", *BUILD]
    for f in files:
        args += ["--file", f]
    listed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    stale = stale_rows(rows, listed)
    for name, file, mutant in stale:
        print(f"STALE ROW: {name}: {file}\t{mutant}")
    if stale:
        sys.exit(f"mutants: {len(stale)} rows name mutants the source no longer has; remove them")
    return listed


def known_hangs(tknown, listed):
    """The time-out names a whole run need not run again (BLAKE3 #244):
    each listed for this CPU with a reason (not OPEN, not UNCLASSIFIED),
    outside the model files (whose rows the suite and the model decide
    together, with_model), and with every place of the name a time-out,
    so skipping it skips no place whose outcome is unknown. Each costs a
    hang to nextest's clock (30 s) and a build a run, and says what the
    list already says. A --since run still runs them, as the lines a change
    touches deserve; --write and --all-hangs run them in a whole run too."""
    places = collections.Counter(f"{m.group(1)}\t{m.group(2)}" for m in (NAME.match(l) for l in listed) if m)
    return sorted(k for k, (count, reason) in tknown.items()
                  if not reason.startswith(("OPEN", "UNCLASSIFIED"))
                  and k.split("\t", 1)[0] not in LOOM_FILES
                  and count == places[k])


def exclude_args(names):
    """cargo-mutants' --exclude-re for each name, every place of it: a
    name is "file<TAB>mutant", cargo-mutants' "file:line:column: mutant"."""
    return [arg for k in names for arg in ("--exclude-re", "^" + re.escape(k.split("\t", 1)[0]) + r":\d+:\d+: " + re.escape(k.split("\t", 1)[1]) + "$")]


def confirm(found, known, timed, tknown, differ=None):
    """`found` (survivors) and `timed` (time-outs) with every name whose
    count of either differs from its list run again, alone: a test that
    fails by chance (under load, say) neither hides a survivor nor fails
    the check, and a test the load made slow is not taken for a hang."""
    if differ is None:
        differ = sorted(k for k in set(found) | set(known) | set(timed) | set(tknown)
                        if found.get(k, 0) != known.get(k, (0, ""))[0] or timed.get(k, 0) != tknown.get(k, (0, ""))[0])
    if not differ:
        return found, timed
    patterns = [": " + re.escape(k.split("\t", 1)[1]) + "$" for k in differ]
    again, _, again_timed = run(sorted({k.split("\t", 1)[0] for k in differ}), [arg for p in patterns for arg in ("--re", p)], shards=1, label="confirm")
    if LOOM_FILES:
        again, again_timed = with_model("confirm", again, again_timed)
    confirmed, confirmed_timed = collections.Counter(found), collections.Counter(timed)
    for k in differ:
        confirmed[k], confirmed_timed[k] = again.get(k, 0), again_timed.get(k, 0)
        SAME["run"][k] = SAME["confirm"][k]
        RELEASE_SAME["run"][k] = RELEASE_SAME["confirm"][k]
    print(f"confirmed alone: {len(differ)} names whose count differed from a list")
    return +confirmed, +confirmed_timed


def changed_tests(since, head="HEAD"):
    """The tests a change alters, from the code before it: (names, sure).
    A line the change takes away (a changed line is one taken away and one
    added) inside a #[test] function names that test; one in other test code
    (a helper, a test module's or tests/'s own items) can weaken any test,
    so `sure` is False. Lines of the library's code are the change's own
    mutants' business."""
    diff = subprocess.run(["git", "-C", ROOT, "diff", "-U0", f"{since}...{head}", "--", "src", "tests"], capture_output=True, text=True, check=True).stdout
    names, sure, file, old = set(), True, None, []
    for line in diff.splitlines():
        if line.startswith("--- "):
            file = line[6:] if line.startswith("--- a/") else None
            old = subprocess.run(["git", "-C", ROOT, "show", f"{since}:{file}"], capture_output=True, text=True).stdout.splitlines() if file else []
        elif line.startswith("@@") and file:
            start, count = re.match(r"@@ -(\d+)(?:,(\d+))?", line).groups()
            for n in range(int(start), int(start) + int(count if count is not None else 1)):
                if not old[n - 1].strip() or old[n - 1].lstrip().startswith("//"):
                    continue  # a comment or a blank line weakens no test
                # src/test.rs is the library's test module (lib.rs: #[cfg(test)] mod test).
                kind, name = enclosing(old, n - 1, file.startswith("tests/") or file == "src/test.rs")
                if kind == "test":
                    names.add(name)
                elif kind == "helper":
                    sure = False
    return names, sure


def enclosing(lines, i, test_file):
    """What line `i` of a source file (rustfmt's layout) belongs to: ("test",
    its name) inside a #[test] function, ("helper", None) inside other test
    code (a #[cfg(test)] module, or anywhere in tests/), ("code", None)
    elsewhere."""
    depth = len(lines[i]) - len(lines[i].lstrip())
    fn = None
    for k in range(i, -1, -1):
        line = lines[k]
        indent = len(line) - len(line.lstrip())
        if not line.strip() or indent > depth or (indent == depth and k != i and not line.lstrip().startswith(("fn ", "pub fn ", "mod ", "pub(crate) mod ", "#["))):
            continue
        m = re.match(r"\s*(?:pub(?:\([a-z]+\))? )?(?:async )?fn (\w+)", line)
        if m and fn is None:
            attrs = []
            for a in range(k - 1, -1, -1):
                if lines[a].lstrip().startswith(("#[", "///", "//")):
                    attrs.append(lines[a])
                else:
                    break
            if any(re.search(r"#\[test\]", a) for a in attrs):
                return "test", m.group(1)
            fn = m.group(1)
        if re.match(r"\s*(?:pub(?:\([a-z]+\))? )?mod \w+ \{", line):
            above = lines[k - 1] if k else ""
            if "cfg(test" in above or "cfg(all(test" in above:
                return "helper", None
        depth = min(depth, indent)
    return ("helper" if test_file else "code"), None


MAP_DIR = os.path.join(ROOT, "tmp", "mutants-last", "map")


def caught_by(tests):
    """The mutants the latest map (a --map run of an ancestor of HEAD) records
    as caught by any of `tests` (by function name) at a place the startup
    self-test does not catch, by name (rerun at all its places); None when
    there is no such map."""
    record, commit = os.path.join(MAP_DIR, "tests.tsv"), os.path.join(MAP_DIR, "commit")
    if not (os.path.exists(record) and os.path.exists(commit)):
        return None
    if subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor", open(commit).read().strip(), "HEAD"]).returncode != 0:
        return None
    by, self_test = collections.defaultdict(set), set()
    lines = open(record)
    assert next(lines) == HEADER, f"{record} is in an older format: make a new map with --map"
    for line in lines:
        file, mutant, place, _function, test, outcome, _ms = line.rstrip("\n").split("\t")
        if file == "(unmutated)" or outcome == "PASS":
            continue
        name = (f"{file}\t{mutant}", place)
        if test == SELF_TEST:
            self_test.add(name)
        elif re.split(r"[ :]", test)[-1] in tests:
            by[name].add(test)
    return sorted({name for name, _ in set(by) - self_test})


def figure(functions, known):
    """How much of the library mutation testing covers, by function: every
    mutant caught, every survivor explained, or one still open."""
    caught = explained = 0
    open_ = []
    for (file, function), outcomes in sorted(functions.items()):
        reasons = [known.get(name, (0, "UNCLASSIFIED"))[1] for summary, name in outcomes if summary in ("MissedMutant", "TimedOut")]
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
    # The run's first line names what it measured, so a release gate can
    # hold its log to the release's commit (John Servil, BLAKE3 #125).
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    changed = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"mutants: BLAKE3-Servil at {commit}{', with uncommitted changes' if changed else ''}", flush=True)
    remove_leftovers()
    listed = check_rows_exist()
    files, known = read_known()
    _, tknown = read_known(TIMEOUTS, INACTIVE_TIMEOUTS)
    # A test known to be missing is work to do now, not a survivor to keep:
    # the check fails while one is listed, whichever run this is.
    missing = sorted(k for k, (_, reason) in known.items() if reason.startswith("OPEN: a missing test"))
    for k in missing:
        print(f"MISSING TEST: {k}")
    if missing:
        sys.exit("mutants: the list names missing tests; write them first")
    since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else None
    extra = []  # the mutants the change's altered tests caught (the map's)
    if since is not None:
        diff = subprocess.run(["git", "-C", ROOT, "diff", f"{since}...HEAD", "--", *files, *LOOM_FILES], capture_output=True, text=True, check=True).stdout
        tests, sure = changed_tests(since)
        caught = caught_by(tests) if tests and sure else []
        if not sure:
            print(f"mutants: the change since {since} alters test code outside a test (a helper); the whole run")
            since = None
        elif caught is None:
            print(f"mutants: the change since {since} alters {len(tests)} test{'s' if len(tests) > 1 else ''} and no map of an ancestor holds their catches (check.py --map); the whole run")
            since = None
        else:
            extra = caught
            if tests:
                print(f"mutants: the change alters {len(tests)} test{'s' if len(tests) > 1 else ''}; the map's {len(extra)} mutant names they catch run again")
    if since is not None:
        if not diff and not extra:
            print(f"mutants: the change since {since} touches no mutated file and alters no test")
            return
        path = os.path.join(tempfile.mkdtemp(prefix="mutants-diff-", dir=scratch_dir()), "change.diff")
        with open(path, "w") as f:
            f.write(diff)
        # One shard: a change's mutants are few, and each shard builds anew.
        found, functions, timed = run(files, ["--in-diff", path], shards=1) if diff else (collections.Counter(), {}, collections.Counter())
        if any(f in diff for f in LOOM_FILES):
            found, timed, functions = with_loom(found, timed, functions, run(LOOM_FILES, ["--in-diff", path], shards=1, label="loom", loom=True)[1])
        if extra:
            # The altered tests' catches, wherever they are: by name.
            again, more, again_timed = run(sorted({k.split("\t", 1)[0] for k in extra}), [arg for k in extra for arg in ("--re", ": " + re.escape(k.split("\t", 1)[1]) + "$")], label="tests")
            found, timed = found + again, timed + again_timed
            functions = {**more, **functions}
        # Only the names with more survivors than the list: the run holds
        # the change's lines alone, so a listed name's other places are
        # missing from it (a whole run sees those, and stale entries).
        found, timed = confirm(found, known, timed, tknown, sorted({k for k in found if found[k] > known.get(k, (0, ""))[0]}
                                                                   | {k for k in timed if timed[k] > tknown.get(k, (0, ""))[0]}))
        new = {k: n for k, n in found.items() if n > known.get(k, (0, ""))[0]}
        new_timed = {k: n for k, n in timed.items() if n > tknown.get(k, (0, ""))[0]}
        for k, n in sorted(new.items()):
            print(f"NEW SURVIVOR ({n - known.get(k, (0, ''))[0]}): {k}")
        for k, n in sorted(new_timed.items()):
            print(f"NEW TIME-OUT ({n - tknown.get(k, (0, ''))[0]}): {k}")
        print(f"mutants: the change since {since}: {sum(1 for f in functions.values() for _ in f)} mutants, "
              + ("no new survivor" if not new else f"{len(new)} names with a new survivor")
              + ("" if not new_timed else f", {len(new_timed)} names with a new time-out (no verdict on them)"))
        sys.exit(1 if new else 2 if new_timed else 0)
    # The known hangs stay out of a whole run (known_hangs); their rows stand
    # as listed. A run that rewrites the lists (--write) runs them, so a row
    # that no longer hangs leaves its list: a skipped one would stay listed
    # (two sources::find rows, caught at servil 06f6e26d, did in the trial).
    hangs = [] if "--all-hangs" in sys.argv or "--write" in sys.argv else known_hangs(tknown, listed)
    if hangs:
        print(f"mutants: {len(hangs)} known hangs ({sum(tknown[k][0] for k in hangs)} places) not run, each listed with its reason "
              f"in {os.path.basename(TIMEOUTS)} (--all-hangs runs them)", flush=True)
    found, functions, timed = run(files, exclude_args(hangs))
    timed.update({k: tknown[k][0] for k in hangs})
    if LOOM_FILES:
        found, timed, functions = with_loom(found, timed, functions, run(LOOM_FILES, shards=1, label="loom", loom=True)[1])
    found, timed = confirm(found, known, timed, tknown)
    figure(functions, {**{k: tknown.get(k, (n, "UNCLASSIFIED")) for k, n in timed.items()},
                       **{k: known.get(k, (n, "UNCLASSIFIED")) for k, n in found.items()}})
    if "--write" in sys.argv:
        header = [l.rstrip("\n") for l in open(KNOWN) if l.startswith("#") or l.startswith(("files:", "model files:"))]
        # The rows for other CPUs stay as they were: this run cannot see them.
        # An OPEN pool row (its outcome left to the schedules) stays at its
        # most places when this run's schedules caught it: the run is one
        # draw, and the row is what the draws have shown (October 26).
        found = {**{k: c for k, (c, r) in known.items() if open_pool(k, r)}, **found}
        found = {k: max(n, known[k][0]) if k in known and open_pool(k, known[k][1]) else n for k, n in found.items()}
        def reason(k, n):
            if SAME["run"][k] == n:
                return SAME_RELEASE_REASON if RELEASE_SAME["run"][k] else SAME_REASON
            return known[k][1] if k in known and known[k][1] not in SAME_REASONS else unclassified(k, 'survives')
        rows = sorted([f"{k}\t{n}\t{reason(k, n)}" for k, n in found.items()] + INACTIVE)
        open(KNOWN, "w").write("\n".join(header + rows) + "\n")
        print(f"{KNOWN}: {sum(found.values())} survivors, {sum(1 for k in found if k not in known)} names new")
        theader = [l.rstrip("\n") for l in open(TIMEOUTS) if l.startswith("#")] if os.path.exists(TIMEOUTS) else \
            [f"# The mutants that time out on {platform.machine()}: no test fails, the clock stops one (tools/mutants/check.py, TIMEOUTS).",
             "# A line: file <TAB> mutant (function and change) <TAB> places <TAB> reason."]
        timed = {**{k: c for k, (c, r) in tknown.items() if open_pool(k, r)}, **timed}
        timed = {k: max(n, tknown[k][0]) if k in tknown and open_pool(k, tknown[k][1]) else n for k, n in timed.items()}
        trows = sorted([f"{k}\t{n}\t{tknown[k][1] if k in tknown else unclassified(k, 'times out')}" for k, n in timed.items()] + INACTIVE_TIMEOUTS)
        open(TIMEOUTS, "w").write("\n".join(theader + trows) + "\n")
        print(f"{TIMEOUTS}: {sum(timed.values())} time-outs, {sum(1 for k in timed if k not in tknown)} names new")
        return
    unexplained = sorted(k for k, (_, reason) in {**tknown, **known}.items() if reason.startswith("UNCLASSIFIED"))
    for k in unexplained:
        print(f"NO REASON: {k}")
    new = {k: n for k, n in found.items() if n > known.get(k, (0, ""))[0]}
    # An OPEN pool entry is caught or not as the run's scheduling falls
    # (the list says why): its count is the most that survive, and a run
    # that catches more of them is no change to the list.
    stale = {k: c for k, (c, reason) in known.items() if found.get(k, 0) < c and not open_pool(k, reason)}
    for k, n in sorted(new.items()):
        print(f"NEW SURVIVOR ({n - known.get(k, (0, ''))[0]}): {k}")
    # A row the tool explained as same code holds only while it is.
    same = SAME["run"]
    drifted = sorted(k for k, (c, reason) in known.items() if reason in SAME_REASONS and same[k] != found.get(k, 0))
    for k in drifted:
        print(f"NO LONGER SAME CODE ({found.get(k, 0) - same[k]} of {found.get(k, 0)} places build to other code): {k}")
    # Same code on this machine and on every other whose list has the
    # mutant: not code for another machine, but a change every build
    # erases, or code that does nothing. A finding to act on (an
    # equivalent mutant to note, or code to remove), not a failure.
    for k in sorted(k for k, n in found.items() if same[k] == n):
        others = elsewhere(k)
        if others and all(any(r in o for r in SAME_REASONS) for o in others):
            print(f"SAME CODE EVERYWHERE: {k} ({'; '.join(o.split(':', 1)[0] for o in others)} too)")
    for k, c in sorted(stale.items()):
        print(f"{'NOW TIMES OUT' if timed.get(k) else 'NOW CAUGHT'} ({c - found.get(k, 0)}): {k}")
    # The time-outs against their list, alike (an OPEN pool row's count is
    # the most that time out, as for survivors).
    new_timed = {k: n for k, n in timed.items() if n > tknown.get(k, (0, ""))[0]}
    stale_timed = {k: c for k, (c, reason) in tknown.items() if timed.get(k, 0) < c and not open_pool(k, reason)}
    for k, n in sorted(new_timed.items()):
        print(f"NEW TIME-OUT ({n - tknown.get(k, (0, ''))[0]}): {k}")
    for k, c in sorted(stale_timed.items()):
        print(f"NO LONGER TIMES OUT ({c - timed.get(k, 0)}): {k}")
    # The places the suite's clock alone stopped (a model file's place the
    # model catches is caught, and is not listed: with_model).
    model = PLACE_OUTCOMES.get("loom", {})
    # Where another machine's list judges a mutant otherwise than this run
    # (survives there, times out here, or the reverse): both facts are the
    # code's; the run names the difference (a mutant caught here and listed
    # elsewhere is the other list's to explain, and its run's to report).
    for k in sorted(set(found) | set(timed)):
        mine = "survives" if k in found else "times out"
        for other in elsewhere(k):
            if f": {mine} (" not in other:
                print(f"MACHINES DIFFER: {k}: here {mine}; {other}")
    for k in sorted(CLOCK_ONLY):
        n, place = k.rsplit("\t", 1)
        if model.get((n, place)) != "CaughtMutant":
            print(f"TIMED OUT (place): {k}")
    ok = not new and not stale and not unexplained and not drifted
    ok_timed = not new_timed and not stale_timed
    print(f"mutants: {sum(found.values())} survive, each known and explained" if ok else "mutants: the list and this run differ, or a listed mutant has no reason")
    print(f"mutants: {sum(timed.values())} time out, each known ({os.path.basename(TIMEOUTS)})" if ok_timed
          else f"mutants: the time-outs differ from {os.path.basename(TIMEOUTS)} (no verdict on those mutants)")
    # One status an outcome: 1 a survivor or list wrong, 2 only time-outs differ, 0 both as listed.
    sys.exit(0 if ok and ok_timed else 1 if not ok else 2)


if __name__ == "__main__":
    main()
