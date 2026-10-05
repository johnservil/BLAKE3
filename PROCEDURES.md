# Procedures and environment

For the servil team: how things are done in this repository and its guest. The principles they serve are in `AGENTS.md`; bench-hashes' own procedures are in its `PROCEDURES.md`.

# The performance-regression check: every code commit

Speed is this fork's purpose, so no commit that makes it slower may enter git unnoticed. **Every commit that touches `src/`, `c/`, `build.rs`, `Cargo.toml`, or `Cargo.lock` must pass `tools/perf_regress.py check` first.** The check builds bench-hashes against `HEAD` and against the working tree and runs `bench-hashes regress` on the two builds: eight pairs of runs on this machine, one of each side, back to back in alternating order, each over the lent cells (about 30 s in the VM, builds aside), so load and drift fall on both sides alike; there are no stored numbers and nothing to keep current, and any machine can run it. A solo cell is slower when the median of its pairs' ratios exceeds 3% and an exact sign test agrees (`clocks::summary::verdict`). A change that only moves the code's layout is held about one time in eight, at the lent 64 B cell by about the margin (the Mac, NOTES "The regression check, calibrated"): a lone 64 B hold near 3% on a change that does not touch that path is likely layout, and lands with `--no-verify`, its numbers and that reading in the commit message. The rule, its points, and its margins are bench-hashes' (`regress`).

**Install the pre-commit hook once per checkout**, and the check runs by itself on every code commit:

- macOS host: `sh tools/install-git-hooks.sh`
- the Debian VM: `sh /workspace/vm/setup.sh` (after every VM restart; the mount drops executable bits, so the guest's hook lives in `/tmp/git-hooks`)

**Run it by hand** when the hook is not installed or before pushing: `pypy3 tools/perf_regress.py check` (`python3` where PyPy is absent). It measures the working tree, so commit with everything the commit contains in the tree (`git commit -a`, or stash unrelated edits first). Nothing else should run on the machine meanwhile, and on the VM nothing heavy on the host either.

**What each result obliges you to do:**

- *Exit 0, no regression:* commit.
- *Exit 1, a regression:* the hook aborts the commit. Do not commit around it. Find the cause and fix it, or, when the slowdown is the deliberate price of something worth more (correctness, simplicity with a measured cost), stop and ask the user; if they accept it, commit with `git commit --no-verify` and state the regressed cells, their numbers, and the user's decision in the commit message.
- *Exit 2, no verdict:* `clocks` found other programs keeping the machine busy during a run, or saw no load window. Stop whatever else is running and check again.

**Releases:** `python3 tools/gen-ver.py X.Y.Z` from a clean tree (Zooko's technique, copied from bench-hashes: a commit setting X.Y.Z, a second setting X.Y.Z+<first commit>, a lightweight tag vX.Y.Z+<first commit>; push `servil`, then the tag by name). Before a release, run `check --against <previous release tag>`. Each commit is checked only against its parent, so slowdowns too small to flag one at a time could add up; the release check sees their sum.

**Commits that skipped the check** (`--no-verify`, or made where the hook was absent) must be checked before they are pushed: `pypy3 tools/perf_regress.py compare <parent> <commit>` for one, `pypy3 tools/perf_bisect.py <commit> <commit> ...` for a run of them (each against the one before, then the last against the first).

`NOTES-servil.md` ("The regression check, calibrated") gives the rule's measured false holds and sensitivity on the Mac, and why it judges the lent cells alone (the queue's move 6-60% between processes of identical code).

# Branches: candidates, then servil

`servil` is this fork's main line. Every commit on it has passed the whole gate below, on the VM and natively on the Mac.

Work happens on `candidate/<topic>` branches (`candidate/p-e-classification`, `candidate/workgroup-placement`). Their commits pass the pre-commit check on the machine where they are made, as every code commit does, and may be pushed before native measurement: that is how the Mac's benchmark runner, which builds from GitHub alone, gets them.

A candidate reaches `servil` only when all of these hold, and never without the performance check:

1. It is a fast-forward of `servil`'s current tip, so what was measured is exactly what lands. When `servil` has moved, rebase and check again.
2. Every test suite passes, judged by each suite's exit status (a failing test binary prints no `test result` line, so a list of result lines can hide it): the fork's default, `no_sme2`, and `pure` builds, the integration tests, the official vectors, b3sum's, and the benchmark's own tests. The fork's GitHub CI passes on the candidate's tip: it builds every target with warnings as errors, which the local suites and the Mac's test job do not (an unused import reached `servil` 1352c21 that way, October 5, 2026).
3. `pypy3 tools/perf_regress.py compare servil candidate/<topic>` reports no regression on the VM.
4. The same comparison reports no regression natively on the Mac.

A regression the user accepts, as the section above describes, lands with the user's decision, the regressed cells, and their numbers in the merge's message (for a fast-forward, the promoted commit's message; amend the message only, leaving the measured tree unchanged). The user accepts such trades under this rule (September 25, 2026; revised September 27): every slowed cell stays ahead of every competitor, the gains outweigh the losses, and the user decides; a streaming API's cells may not slow to speed up another API (AGENTS.md, "The streaming APIs first"). A candidate waiting on the Mac waits on its branch; the VM's verdict alone does not promote it.

A promotion fast-forwards `servil` to the candidate, records the gate's evidence (suites, both verdicts, their job numbers) as a note on the tip (`git notes --ref=perf add`; push `servil` with `refs/notes/perf`), and deletes the candidate branch here and on GitHub. After a promotion, pin bench-hashes to the new tip (`cargo update -p blake3-servil` in bench-hashes, commit the `Cargo.lock`, push): that pin is what users measure, and records are made on it.

# Before a question goes to Zooko

Before a change or a question goes on Zooko's decision list, ask whether it adds to or rescues a second mechanism for a problem the design already solves (AGENTS.md, "Revisit complexity as you learn"). If it does, settle first which mechanism stays: make the first serve, or remove the second.

# The Mac

`tools/runner/README.md` describes the runner. Zooko starts it with `sh ~/piplayground/blake3-servil/tools/runner/setup-mac.sh` (again after a change to `runner.py` or `perf_regress.py`, which it installs). Write `runner/jobs/NNN-name.json` naming pushed commits; wait with `pypy3 tools/runner/wait_for.py NNN-name`; results land in `runner/results/`. A job runs once per file name, so a changed job takes a new number. Keep the VM idle while a Mac job runs, Activity Monitor closed (its polling keeps half a CPU busy), and the Mac on mains power: on battery it runs more calls that follow a pause on the efficiency cores (233 of 400 against 32; jobs 351, 357), and every job's `verdict.json` records the power state at its start and end (`power at start`, `power at end`); read them before trusting a job. A direct A/B is four `benchmark` jobs, old new new old, back to back, read with `bench-hashes compare` (the old jobs' samples files, `--`, the new jobs'; and each side against itself, for what repetition alone moves); a calibration is one `benchmark` job with `"repeat": N`. Mac-only measurements (cycles by core kind, QoS) go in a `probe/<topic>` branch that adds its own `examples/host_lab.rs` (NOTES-servil.md, "Probes on the Mac"); the `probe/*` branches on origin are those probes, each cited in the NOTES where its finding is.

# Probes

Every clock read, in the fork, in bench-hashes, and in any scratch probe, goes through the `clocks/` crate in this repository (AGENTS.md, "Measuring"): `clocks::measure(batches, batch_ns, || ...)` for a probe's batches (each batch's wall time and the thread's counts per core kind, shown as ns per call and the clock it ran at), `clocks::measure_after_gaps_prepared(calls, gap, gap_ns, input, prepare, call)` for calls that must each come after a gap (`clocks::Gap::Idle`, a sleep, or `clocks::Gap::Busy(work)`, a fixed other program and a walk of `work`; each call timed alone and summed, since a call of a few clock ticks cannot be timed alone: the crate's "Resolution"), `clocks::now()` and `clocks::since_ns()` for wall time alone, `clocks::Counts::read()` for the counts. The fork's examples have it as a dev-dependency; a scratch crate adds `clocks = { path = "/workspace/clocks" }`. Its documentation says which clocks and why (github.com/johnservil/measure-clocks3, `CPU-TIME-CLOCKS-AND-FREQUENCY.md`, has the experiments). The pool's pacing in `src/lanes.rs` reads `Instant` directly: control, not measurement, and the library depends on nothing.

# `perf_regress` and older commits

The benchmark calls the current fork API; a commit that predates it fails to build, and is compared with the tools of its own time (AGENTS.md, "Contracts change everywhere at once"). A release check is against the previous promotion of `servil` that the benchmark builds against. `pypy3 tools/perf_regress.py build` builds bench-hashes against the working tree for runs by hand and prints the executable's path.

# Environment

## Where things are

- `/workspace` is the host checkout of this fork (github.com/johnservil/BLAKE3, main branch `servil`, work on `candidate/<topic>` branches), mounted through sandboxfs. It persists across VM restarts.
- `/workspace/bench-hashes` is the benchmark's own repository (github.com/johnservil/bench-hashes, branch `main`), nested inside the fork. It depends on the fork by git (`servil` branch) at the commit its `Cargo.lock` pins, so a user's `cargo run --release` measures that commit. To build it against this checkout's working tree, run `pypy3 tools/perf_regress.py build`: it builds in a directory of its own under `tmp/perf-ab/new/` (a fork worktree holding a copy of bench-hashes with its own `Cargo.lock`, where the patch `--config 'patch."https://github.com/johnservil/BLAKE3".blake3-servil.path=".."'` applies) and prints the executable's path; nothing tracked changes, and Cargo rebuilds only what changed. `/workspace/.git/info/exclude` keeps it, `benchmark-results/`, `tmp/`, `vm/`, and the token out of the fork's status.
- `/workspace/vm/` holds everything the guest needs that a restart would otherwise remove:
  - `vm/home/` is `HOME` for `git` and `cargo`: `.gitconfig` with `safe.directory = *`, John Servil's `user.name`/`user.email`, and the credential helper.
  - `vm/home/bin/gh-cred.sh` speaks the git credential protocol and reads the johnservil classic token from `/workspace/ghtokenclassic.txt` (never print that file). Both repos have `credential.helper = !sh /workspace/vm/home/bin/gh-cred.sh` (the mount drops executable bits, hence `!sh`).
  - `vm/setup.sh` installs `clang-19` from apt.llvm.org, and `pypy3` and `librsvg2-bin` from Debian, when they are absent; creates `/tmp/target`; configures git and cargo for every shell of the boot (below); re-points both repos' credential helpers; and installs the guest's pre-commit hook in `/tmp/git-hooks`. Run `sh /workspace/vm/setup.sh` first after a VM restart.
- Guest disk (`/tmp`, `/usr`, apt packages) vanishes with the VM. Only `/workspace` persists.

## Building and running

- The VM is Debian 12 on AArch64 with 16 vCPUs (inspect `nproc` after a restart). Its CPU exposes SME2 with 512-bit streaming vectors (`/proc/cpuinfo` lists `sme2`), so the fork's kernels run here. Absolute timings differ from Apple hardware; relative comparisons hold.
- The fork's SME2 kernel is `c/blake3_sme2_aarch64.S`, compiled by the `cc` crate with `-march=armv9-a+sme2`. The system `cc` (GCC 12) and `as` (binutils 2.40) predate SME2, so under them the fork builds without the SME2 kernel and warns (the user's decision, September 25, 2026, for Debian 12 and Raspberry Pi OS users); every VM build takes `CC=clang-19`, which assembles SME2, and `perf_regress` fails stop when a build on an SME2 machine lacks the kernel; `TMPDIR` gives clang a temporary directory that exists in the guest.
- `vm/setup.sh` configures the guest system for the whole boot, so every `git` and `cargo` command runs as it is: `/etc/gitconfig` includes `vm/home/.gitconfig` (whose `safe.directory` covers the mount's files, which show as uid 501 while the guest runs as uid 0), `$CARGO_HOME/config.toml` sets the target directory and `CC=clang-19`, and the Mac's `HOME` and `TMPDIR`, which the guest's shells inherit, are created in the guest.
- Build the fork: `cargo build --release`
- Test the fork: `cargo test --release` (add `--features no_sme2` or `--features pure` for the other platform paths), and the official published vectors with `--manifest-path /workspace/test_vectors/Cargo.toml`, all with the same environment prefix.
- Run the benchmark from `/workspace/bench-hashes`, since it writes `benchmark-results/` relative to the current directory: `cd /workspace/bench-hashes && cargo run --release -- --quick --contenders blake3-official,blake3-servil-st` measures the pinned fork commit; `$(pypy3 /workspace/tools/perf_regress.py build) --quick ...`, run from a scratch directory, measures the working tree. A full run is the default; `--quick` takes seconds.
- `CARGO_TARGET_DIR=/tmp/target` is a tmpfs build cache (rebuilt after a restart); `CARGO_HOME=/usr/local/cargo`. The toolchain is rustc 1.98.1 without the `rustfmt` component, so there is no formatting check in the guest.
- Commands for the user go on one line, with no `\` continuations.
- Never `sleep` in commands.
- Run long commands (builds, benchmark runs, package installs) without a timeout and let their output stream, so the user can watch progress and interrupt when they choose.
