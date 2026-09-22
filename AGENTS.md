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

## Coding: Design By Contract

We document and `assert` every precondition our code relies on (`debug_assert` only on hot paths). Contracts are **expansive** (the caller carries the responsibility), **conceptually simple** (a few sentences of English; simplicity beats familiarity), and **structurally simple** to enforce (few lines, types, data elements, conditionals).

We never write "defensive code" — code that complicates a contract to ease the caller's life. When running code detects that a caller misunderstood the contract, it **fails stop**: panic with a clear message. Stopping is safer than proceeding, and it lets people fix the caller or loosen the contract. Defensive codebases grow buggier over time; DBC codebases stay predictable.

## Interfaces: fewest new concepts

A highly desirable property of an interface and its contract: the user learns the fewest new concepts. Zero new concepts earns a perfect score. Each new term (a resource unit, a sharing rule, a tuning knob) taxes working memory and needs a place in prediction and control. Prefer familiar concepts the caller already holds (threads, inputs, budgets), keep implementation units unnamed in public docs, and express observable behavior (speed, thread count, fairness beside concurrent calls) in those familiar terms.

# Environment

- The VM is Debian 12 on AArch64 with two cores. Its CPU exposes SME2 with 512-bit streaming vectors (`/proc/cpuinfo` lists `sme2`), so the fork's kernels run here. Absolute timings differ from Apple hardware; relative comparisons hold.
- The fork's SME2 kernel is `c/blake3_sme2_aarch64.S`, compiled by the `cc` crate with `-march=armv9-a+sme2`. The system `cc` (GCC 12) and `as` (binutils 2.40) predate SME2, so the fork's build script fails under them with a message naming the fix. `clang-19` is installed and assembles SME2. Build with `CC=clang-19 TMPDIR=/tmp cargo run --release`; `TMPDIR` gives clang a temporary directory that exists in the guest.
- The johnservil classic token is in `ghtokenclassic.txt` (gitignored; never print it). Both this repository and the fork checkout at `/upstream/BLAKE3` (a working clone of `sme2-bench` for editing the fork) have `credential.helper` set to `/tmp/home/bin/gh-cred.sh`, which reads that file; the fork checkout also has `user.name`/`user.email` set to John Servil. The checkout and the helper live on the guest disk, so recreate them after a VM restart before pushing.
- `/workspace` is the host checkout mounted through sandboxfs. Files show as uid 501 while the guest runs as uid 0, so git needs `safe.directory`. `HOME` points at an absent host path; use `HOME=/tmp/home` (which holds a `.gitconfig` with `safe.directory = /workspace`) for both `git` and `cargo` commands. `/tmp/home/.gitconfig` also sets `safe.directory = *`. `/upstream/BLAKE3` and `/tmp/home` live on the guest disk and vanish with the VM; `/workspace` persists.
- `CARGO_TARGET_DIR=/tmp/target` on a tmpfs; `CARGO_HOME=/usr/local/cargo`. The toolchain is rustc 1.98.1 without the `rustfmt` component, so there is no formatting check available in the guest.
- Commands for the user go on one line, with no `\` continuations.
- Never `sleep` in commands. When a network call fails, report it and stop; the user decides about retries.
- Run long commands (builds, benchmark runs, package installs) without a timeout and let their output stream, so the user can watch progress and interrupt when they choose.
