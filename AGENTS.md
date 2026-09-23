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

Read `/workspace/bench-hashes/NEXT-STEPS.md` first: it says what the work is now (optimising this fork against the benchmark) and where the last session left both repositories. `NOTES-sme2-bench.md` in this directory holds the fork's design notes and the measurements behind each change.

# Targets

**Virtual machines are first-class optimization targets.** People run BLAKE3 inside VMs like the Debian guest this repository is developed in, and its speed there matters as much as on the native Mac. A change is good when it helps both, or helps one and leaves the other level; a change that wins natively and loses in a VM (or the reverse) needs a decision on the record, not a default. VMs behave differently in ways that matter here: an idle vCPU that spins or calls `sched_yield` steals host time from the vCPUs that hash, `WFE` returns at once instead of idling, the first NEON instruction after an SME2 kernel costs about 4 µs (about 1 µs natively), and a 16-vCPU guest may sit on fewer fast host cores than it has vCPUs. `examples/host_lab.rs` measures each of these; run it on both and keep both reports.

# Performance regressions: check every code commit

Speed is this fork's purpose, so no commit that makes it slower may enter git unnoticed. **Every commit that touches `src/`, `c/`, `build.rs`, `Cargo.toml`, or `Cargo.lock` must pass the performance-regression check on the machine where it is made.** The check is `tools/perf_regress.py`; it measures `blake3` (the control), `blake3-servil`, and `blake3-servil-mt` with bench-hashes and compares every cell of both use cases with the committed baseline for this machine in `perf-baselines/`.

**Install the pre-commit hook once per checkout**, and the check runs by itself on every code commit:

- macOS host: `sh tools/install-git-hooks.sh`
- the Debian VM: `sh /workspace/vm/setup.sh` (after every VM restart; the mount drops executable bits, so the guest's hook lives in `/tmp/git-hooks`)

**Run it by hand** when the hook is not installed, before pushing, or to see where you stand: `python3 tools/perf_regress.py check` (in the VM with the usual `HOME=/workspace/vm/home CARGO_TARGET_DIR=/tmp/target CC=clang-19 TMPDIR=/tmp` prefix). It takes about 75 seconds on the VM. It measures the working tree, so commit with everything the commit contains in the tree (`git commit -a`, or stash unrelated edits first).

**What each result obliges you to do:**

- *Exit 0, no regression:* commit.
- *Exit 1, a confirmed regression:* the hook aborts the commit. Do not commit around it. Find the cause and fix it, or, when the slowdown is the deliberate price of something worth more (correctness, simplicity with a measured cost), stop and ask the user; if they accept it, commit with `git commit --no-verify` and state the regressed cells, their numbers, and the user's decision in the commit message.
- *Exit 2, no verdict:* there is no baseline for this machine, or this machine, toolchain, OS, or state differs from the baseline's (the output says which). Record a baseline on a quiet machine, `python3 tools/perf_regress.py record` (about 7 minutes on the VM; nothing else running), and commit `perf-baselines/<machine>.json` **in a commit of its own, made before the code change**, so the baseline measures the code as it was. Then run the check again. A new rustc, OS kernel, or VM host means a new baseline file.
- *Cells reported faster:* after the commit that made them faster, record a new baseline and commit it on its own with the before and after numbers, so the next regression is measured against the new speed. Replacing a baseline is a reviewed change, like regenerating golden vectors; never do it to make a failing check pass.

**Commits that skipped the check** (made with `--no-verify`, on a machine without a baseline, or in another checkout) must be checked before they are pushed: `python3 tools/perf_bisect.py <last checked commit> <each later code commit> ...` measures each commit and judges it against the one before (see its docstring). Its runs are kept in `tmp/bisect/`.

Keep baselines current on both targets, the Mac and the VM: VMs are first-class (above). `NOTES-sme2-bench.md` ("Performance-regression check") explains the rule and its measured false-alarm rate (0.13% of checks on unchanged code).

# Environment

## Where things are

- `/workspace` is the host checkout of this fork (github.com/johnservil/BLAKE3, branch `sme2-bench`), mounted through sandboxfs. It persists across VM restarts.
- `/workspace/bench-hashes` is the benchmark's own repository (github.com/johnservil/bench-hashes, branch `main`), nested inside the fork. Its `Cargo.toml` and `build.rs` point the `blake3-servil` path dependency at `..`, so edits to the fork take effect on the benchmark's next build. `/workspace/.git/info/exclude` keeps it, `benchmark-results/`, `tmp/`, `vm/`, and the token out of the fork's status.
- `/workspace/vm/` holds everything the guest needs that a restart would otherwise remove:
  - `vm/home/` is `HOME` for `git` and `cargo`: `.gitconfig` with `safe.directory = *`, John Servil's `user.name`/`user.email`, and the credential helper.
  - `vm/home/bin/gh-cred.sh` speaks the git credential protocol and reads the johnservil classic token from `/workspace/ghtokenclassic.txt` (never print that file). Both repos have `credential.helper = !sh /workspace/vm/home/bin/gh-cred.sh` (the mount drops executable bits, hence `!sh`).
  - `vm/setup.sh` installs `clang-19` from apt.llvm.org when it is absent, creates `/tmp/target`, and re-points both repos' credential helpers. Run `sh /workspace/vm/setup.sh` first after a VM restart.
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
- Never `sleep` in commands. When a network call fails, report it and stop; the user decides about retries.
- Run long commands (builds, benchmark runs, package installs) without a timeout and let their output stream, so the user can watch progress and interrupt when they choose.
