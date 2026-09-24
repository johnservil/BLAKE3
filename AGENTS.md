# Style Guides

## Communication

- Phrase positively or neutrally; avoid negations and "not this, but that" contrasts.
- Frame positively: show the promising, successful aspects of the recommended path. Mention an alternative only when its trade-offs deserve our attention.
- The reader has limited working memory and limited ability to search back through recent text. Include only what the current focus needs.

Sixteen actions that improve writing:
1. Sand off filler words
2. Find the real actors
3. Restore actions to verbs
4. Delete empty verbs
5. Prefer characters as subjects
6. Put subjects and verbs together
7. Put verbs and objects together
8. Make the opening familiar
9. Put new and important information last
10. Repair topic flow
11. Repair stress flow
12. Establish a clear topic sentence
13. Make subjects consistent across a passage
14. Control passive voice deliberately
15. Name responsibility
16. Trim metadiscourse

## Presentation: every item costs the reader

Every piece of information in a UI, a report, or a document costs its reader attention and risks fatigue and overflow. Review each one with two questions: who is this designed for, and which information pays that reader much more than it costs them? Keep what passes; cut the rest. Information for maintainers (diagnostics, spreads, provenance details, internal names) stays out of what users read; it belongs in maintainer notes, logs, and diagnostic flags.

## Simplicity

Prefer the simplest design that meets the contract and performs well.
Simplicity has several dimensions:

- **Conceptual ease:** a reader can understand and predict the design with
  a few familiar concepts.
- **Less code and information:** fewer moving parts, less state, and fewer
  facts to carry in working memory.
- **Fewer runtime cases:** fewer branches, special cases, tuning knobs,
  and distinct operating regimes.

Use one clear mechanism wherever it serves. Added complexity earns its
place through a concrete need and a demonstrated benefit. When approaches
perform similarly, choose the simpler one. Apply this standard to code,
interfaces, documentation, and performance optimizations.

## Strategy: minimax

Judge a design by its worst plausible case first. Performing well in every situation (or as many as possible) beats excelling in some while falling behind in others: a user meets whatever situation their own program creates. Plausible situations include several threads of one program hashing at once, other programs busy on the machine, a quiet machine, a VM, and every input size and batch size. For each candidate, find the situation where it does worst and compare those worst cases; a best case decides only between designs whose worst cases are level. Consequences here:

- A resource that can be shared (an SME unit, a cluster, memory bandwidth) is judged at its shared speed, since some program will share it.
- A multithreaded call that runs slower than the single-threaded call on the same task is a defect: it could have run single-threaded.
- A task that takes more than proportionally longer than a smaller one is a defect: it could have done the smaller task twice.
- Effort goes first to the cells where we lead by the least or trail.

## Strategy: we own every slowdown a user could meet

Our duty is to the user, so we take responsibility for every performance problem that could plausibly reach one, whatever its cause. A slowdown that comes from the operating system's scheduler, the hardware, thread placement, clocks, or an interaction we find hard to reproduce, understand, or control is still ours. We never set such a slowdown aside with "it is probably the environment, not our code". Each one gets one of three outcomes, in this order of preference:

1. Control it: change the design so the slowdown cannot happen, or cannot happen from anything we did.
2. Understand it well enough to tell users how to control it, and document that.
3. At the least, understand it well enough to predict when it happens and how large it is, and state that in the user-facing results or docs.

Until a problem reaches one of these outcomes it stays open: it goes on the next-steps list, the record or commit that shows it says so, and it blocks the claim that a change has no regression.

## Correctness tests

Use reproducible inputs and fixed, independently established expected
outputs. A deterministic RNG is a compact specification of test bytes;
record its algorithm, seed, and length, and check in the expected digests.
Published test vectors and independent reference implementations establish
the answers. Regenerating golden outputs is an explicit, reviewed action.
Tests never silently regenerate their own expected answers.

The same fixed vectors can exercise different kernels, thread budgets,
concurrent calls, and scheduling interleavings. Input generation and
execution scheduling are separate concerns. Differential tests supplement
these anchors. Keep benchmark correctness checks outside timed intervals,
and share the implementation dispatch between checking and timing.

## Coding: Design By Contract

We document and `assert` every precondition our code relies on (`debug_assert` only on hot paths). Contracts are **expansive** (the caller carries the responsibility), **conceptually simple** (a few sentences of English; simplicity beats familiarity), and **structurally simple** to enforce (few lines, types, data elements, conditionals).

We never write "defensive code" — code that complicates a contract to ease the caller's life. When running code detects that a caller misunderstood the contract, it **fails stop**: panic with a clear message. Stopping is safer than proceeding, and it lets people fix the caller or loosen the contract. Defensive codebases grow buggier over time; DBC codebases stay predictable.

## Interfaces: fewest new concepts

A highly desirable property of an interface and its contract: the user learns the fewest new concepts. Zero new concepts earns a perfect score. Each new term (a resource unit, a sharing rule, a tuning knob) taxes working memory and needs a place in prediction and control. Prefer familiar concepts the caller already holds (threads, inputs, budgets), keep implementation units unnamed in public docs, and express observable behavior (speed, thread count, fairness beside concurrent calls) in those familiar terms.

# Where to start

Read `/workspace/bench-hashes/NEXT-STEPS.md` first: it says what the work is now (optimising this fork against the benchmark) and where the last session left both repositories. `NOTES-servil.md` in this directory holds the fork's design notes and the measurements behind each change.

# Targets

**Virtual machines are first-class optimization targets.** People run BLAKE3 inside VMs like the Debian guest this repository is developed in, and its speed there matters as much as on the native Mac. A change is good when it helps both, or helps one and leaves the other level; a change that wins natively and loses in a VM (or the reverse) needs a decision on the record, not a default. VMs behave differently in ways that matter here: an idle vCPU that spins or calls `sched_yield` steals host time from the vCPUs that hash, `WFE` returns at once instead of idling, the first NEON instruction after an SME2 kernel costs about 4 µs (about 1 µs natively), and a 16-vCPU guest may sit on fewer fast host cores than it has vCPUs. `examples/host_lab.rs` measures each of these; run it on both and keep both reports.

# Performance regressions: check every code commit

Speed is this fork's purpose, so no commit that makes it slower may enter git unnoticed. **Every commit that touches `src/`, `c/`, `build.rs`, `Cargo.toml`, or `Cargo.lock` must pass `tools/perf_regress.py check` first.** The check builds bench-hashes against `HEAD` and against the working tree and runs the two builds alternately on this machine (A B B A A B B A, about 40 seconds on the VM plus builds), so load and drift fall on both sides alike; there are no stored numbers and nothing to keep current, and any machine can run it.

**Install the pre-commit hook once per checkout**, and the check runs by itself on every code commit:

- macOS host: `sh tools/install-git-hooks.sh`
- the Debian VM: `sh /workspace/vm/setup.sh` (after every VM restart; the mount drops executable bits, so the guest's hook lives in `/tmp/git-hooks`)

**Run it by hand** when the hook is not installed or before pushing: `pypy3 tools/perf_regress.py check` (`python3` where PyPy is absent; in the VM with the usual `HOME=/workspace/vm/home CARGO_TARGET_DIR=/tmp/target CC=clang-19 TMPDIR=/tmp` prefix). It measures the working tree, so commit with everything the commit contains in the tree (`git commit -a`, or stash unrelated edits first). Nothing else should run on the machine meanwhile, and on the VM nothing heavy on the host either.

**What each result obliges you to do:**

- *Exit 0, no regression:* commit.
- *Exit 1, a confirmed regression:* the hook aborts the commit. Do not commit around it. Find the cause and fix it, or, when the slowdown is the deliberate price of something worth more (correctness, simplicity with a measured cost), stop and ask the user; if they accept it, commit with `git commit --no-verify` and state the regressed cells, their numbers, and the user's decision in the commit message.
- *Exit 2, no verdict:* the control (SHA-256, the same code on both sides) moved, so the machine's state changed during the check. Stop whatever else is running and check again.

**Releases:** before tagging a release, run `check --against <previous release tag>`. Each commit is checked only against its parent, so slowdowns too small to flag one at a time could add up; the release check sees their sum.

**Commits that skipped the check** (`--no-verify`, or made where the hook was absent) must be checked before they are pushed: `pypy3 tools/perf_regress.py compare <parent> <commit>` for one, `pypy3 tools/perf_bisect.py <commit> <commit> ...` for a run of them (each against the one before, then the last against the first).

`NOTES-servil.md` ("Performance-regression check") explains the rule and its measured false-alarm rate and sensitivity.

# Branches: candidates, then servil

`servil` is this fork's main line. Every commit on it has passed the whole gate below, on the VM and natively on the Mac.

Work happens on `candidate/<topic>` branches (`candidate/p-e-classification`, `candidate/workgroup-placement`). Their commits pass the pre-commit check on the machine where they are made, as every code commit does, and may be pushed before native measurement: that is how the Mac's benchmark runner, which builds from GitHub alone, gets them.

A candidate reaches `servil` only when all of these hold, and never without the performance check:

1. It is a fast-forward of `servil`'s current tip, so what was measured is exactly what lands. When `servil` has moved, rebase and check again.
2. Every test suite passes: the fork's default, `no_sme2`, and `pure` builds, the official vectors, and the benchmark's own tests.
3. `pypy3 tools/perf_regress.py compare servil candidate/<topic>` reports no regression on the VM.
4. The same comparison reports no regression natively on the Mac.

A regression the user accepts, as the section above describes, lands with the user's decision, the regressed cells, and their numbers in the merge's message. A candidate waiting on the Mac waits on its branch; the VM's verdict alone does not promote it.

# Environment

## Where things are

- `/workspace` is the host checkout of this fork (github.com/johnservil/BLAKE3, main branch `servil`, work on `candidate/<topic>` branches), mounted through sandboxfs. It persists across VM restarts.
- `/workspace/bench-hashes` is the benchmark's own repository (github.com/johnservil/bench-hashes, branch `main`), nested inside the fork. Its `Cargo.toml` and `build.rs` point the `blake3-servil` path dependency at `..`, so edits to the fork take effect on the benchmark's next build. `/workspace/.git/info/exclude` keeps it, `benchmark-results/`, `tmp/`, `vm/`, and the token out of the fork's status.
- `/workspace/vm/` holds everything the guest needs that a restart would otherwise remove:
  - `vm/home/` is `HOME` for `git` and `cargo`: `.gitconfig` with `safe.directory = *`, John Servil's `user.name`/`user.email`, and the credential helper.
  - `vm/home/bin/gh-cred.sh` speaks the git credential protocol and reads the johnservil classic token from `/workspace/ghtokenclassic.txt` (never print that file). Both repos have `credential.helper = !sh /workspace/vm/home/bin/gh-cred.sh` (the mount drops executable bits, hence `!sh`).
  - `vm/setup.sh` installs `clang-19` from apt.llvm.org, and `pypy3` and `librsvg2-bin` from Debian, when they are absent; creates `/tmp/target`; re-points both repos' credential helpers; and installs the guest's pre-commit hook in `/tmp/git-hooks`. Run `sh /workspace/vm/setup.sh` first after a VM restart.
- Guest disk (`/tmp`, `/usr`, apt packages) vanishes with the VM. Only `/workspace` persists.

## Building and running

- The VM is Debian 12 on AArch64 with 16 vCPUs (inspect `nproc` after a restart). Its CPU exposes SME2 with 512-bit streaming vectors (`/proc/cpuinfo` lists `sme2`), so the fork's kernels run here. Absolute timings differ from Apple hardware; relative comparisons hold.
- The fork's SME2 kernel is `c/blake3_sme2_aarch64.S`, compiled by the `cc` crate with `-march=armv9-a+sme2`. The system `cc` (GCC 12) and `as` (binutils 2.40) predate SME2, so the fork's build script fails under them with a message naming the fix. `clang-19` assembles SME2; `TMPDIR` gives clang a temporary directory that exists in the guest.
- Every `git` and `cargo` command takes `HOME=/workspace/vm/home`. Files on the mount show as uid 501 while the guest runs as uid 0, which is what `safe.directory` covers.
- Build the fork: `HOME=/workspace/vm/home CARGO_TARGET_DIR=/tmp/target CC=clang-19 TMPDIR=/tmp cargo build --release`
- Test the fork: `cargo test --release` (add `--features no_sme2` or `--features pure` for the other platform paths), and the official published vectors with `--manifest-path /workspace/test_vectors/Cargo.toml`, all with the same environment prefix.
- Run the benchmark from `/workspace/bench-hashes`, since it writes `benchmark-results/` relative to the current directory: `cd /workspace/bench-hashes && HOME=/workspace/vm/home CARGO_TARGET_DIR=/tmp/target CC=clang-19 TMPDIR=/tmp cargo run --release -- --contenders blake3,blake3-servil`
- `CARGO_TARGET_DIR=/tmp/target` is a tmpfs build cache (rebuilt after a restart); `CARGO_HOME=/usr/local/cargo`. The toolchain is rustc 1.98.1 without the `rustfmt` component, so there is no formatting check in the guest.
- Commands for the user go on one line, with no `\` continuations.
- Never `sleep` in commands.
- Run long commands (builds, benchmark runs, package installs) without a timeout and let their output stream, so the user can watch progress and interrupt when they choose.
