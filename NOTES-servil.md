# Notes for the servil fork's maintainers

`servil`, the main branch of github.com/johnservil/BLAKE3, is a fork of
BLAKE3 1.8.7 published as the crate `blake3-servil` (library
`blake3_servil`), so it links beside crates.io `blake3`. Its purpose: the
fastest BLAKE3 on Apple M4-class hardware, natively and in virtual
machines, in every situation a program meets (alone, beside other work,
beside other hashing threads), judged by the worst case first (AGENTS.md).

This file holds what the code alone does not say: what runs where, the
hardware facts the design rests on, what was tried and rejected, and the
tools' pitfalls. Commit messages carry the full numbers behind each change;
`git notes --ref perf show <commit>` has each promotion's gate verdicts.
Runner job numbers below refer to `runner/results/` (outside git, on the
mount).

Treat the benchmark (`bench-hashes/`, its own repository) as a customer:
tune to the hardware, never to its harness.

## Layout

| Piece | Where | Role |
|---|---|---|
| SME2 kernels | `c/blake3_sme2_aarch64.S`, `src/ffi_sme2.rs` | 16 chunks or 16 parent blocks per group on 512-bit streaming vectors; `DEGREE = 128` (eight groups per entry into streaming mode) |
| NEON hybrid kernels | `c/blake3_neon_hybrid_aarch64.S`, **generated** by `tools/gen_neon_hybrid.py`; `src/ffi_neon_hybrid.rs` | k1-k10 (whole chunks), q1-q9 (whole chunks plus a partial one), p2-p9 (parents, one-block messages; p3/p5/p7/p9 with a scalar lane); scalar chunks run on the integer units beside NEON work |
| One SME2 call at a time | `Sme2Turn` in `src/platform.rs` | Calls large enough for SME2 take a process-wide turn; a call finding it taken runs NEON |
| Batches | `src/many.rs` | One digest per message; runs of 64-byte messages go to the parent kernels many lanes at a time |
| Multithreaded | `src/lanes.rs` | One pool over every CPU, NEON only; public contract speaks of threads alone |
| Regression check | `tools/perf_regress.py`, `tools/perf_bisect.py`, `tools/git-hooks/` | Working tree against HEAD (or two commits), A B B A A B B A, through bench-hashes; mandatory for code commits |
| Mac runner | `tools/runner/` (README) | Jobs from the VM run natively on the Mac as the `benchrunner` account, code from GitHub |
| The minimax list | `tools/losses.py SAMPLES.tsv` | Every cell where another contender beats servil or servil mt by more than 3% |
| Platform facts | `examples/host_lab.rs`, `examples/scaling.rs`, `host-lab-reports/` | Primitives, scaling, idle waiters, SME2 callers, WFE on the machine it runs on |

The generated assembly is committed. Edit the generator and run
`python3 tools/gen_neon_hybrid.py > c/blake3_neon_hybrid_aarch64.S`;
`test_assembly_matches_generator` fails on drift. When changing the
generator, check that existing kernels stay byte-identical unless meant.

## What runs where (M4 and the VM; elsewhere NEON and scalar alone)

- `hash()` and friends: up to 1 KiB the scalar kernel c1 (one call, root
  included); 2-15 chunks the NEON hybrids by the plans in
  `ffi_neon_hybrid.rs` (a trailing partial chunk beside the whole ones, q
  kernels); 16 chunks and up SME2 for full groups, NEON for the rest.
  Parents: SME2 for groups of 16, NEON below. Root: scalar.
- `hash_many()`: runs of 16+ one-block messages on SME2, fewer on NEON,
  other lengths through `hash()`.
- `hash_multithreaded()`: below 64 KiB, `hash()`'s path; from 64 KiB the
  pool on NEON only. Batches: under 64 KiB of messages (and, without a
  length pass, fewer than 1024 messages of a block or less) the serial
  path; else the pool, NEON only.
- Every SME2-sized call takes the turn first (inputs of 16 chunks, batches
  of 16 messages).

Mac, solo, ns/B (record e16e836, fork 2a82c8c): servil 64 B .704, 1 KiB
.680, 2 KiB .458, 3 KiB .333, 4 KiB .318, 8 KiB .255, 16 KiB .223, 1 MiB
.174, 128 MiB .182; upstream blake3 .746 / .716 / .729 / .746 / .403 / .392
/ .387 / .387 / .404; SHA-256 ring about .30 from 1 KiB. servil mt 1 MiB
.032, 128 MiB .022. Batches ns/msg: 1 message 46.6, 16-512 about 10.

## Hardware facts the design rests on

**A single chunk is a dependency chain with a floor.** G's half is six
single-cycle steps (add, eor, ror, add, eor, ror); ADD takes no rotated
operand, and an EOR with a rotated operand takes two cycles (folding the
rotations into the xors was 7-13% slower). One 64-byte block needs 168
cycles, about 2.6 cycles/B, against hardware SHA-256's 1.4-1.6. Inputs up
to 2 KiB cannot catch SHA-256 on this hardware: understood and predicted.

**E-cores have fewer integer units.** Cycles per byte, P / E (M4 Max,
`thread_selfcounts` per perf level; user-interactive QoS runs on P-cores,
background QoS on E-cores; counts steady to 1% where wall times at
background QoS swing 2x with the E clock):

    k1 scalar          2.89 / 4.07     k6 2 scalars + 2 pairs  1.00 / 2.07
    k2 pair            1.84 / 2.08     k7 scalar + quad + pair 1.00 / 1.51
    k3 scalar + pair   1.24 / 1.64     k9 scalar + 2 quads     0.95 / 1.48
    k4 2 scalars+pair  1.16 / 2.41     k10 2 scalars + 2 quads 0.86 / 1.63
    k8 2 scalars + quad + pair: 8 chunks 7180 / 14170 cycles (two quads 8718 / 13310)
    SME2, 16+ chunks   0.57 / 0.57     (the E cluster's SME unit, same cycles)

A second scalar chunk costs E-cores what it saves P-cores. The user chose
P-core speed where they trade (k4, k8); `probe/ecore-kernels` has the
probe.

**Two SME2 threads of one process share one SME unit** (macOS keeps a
thread group on one P-cluster): each runs at half speed or less, slower
than NEON (shared 1 MiB 0.18 or about 0.35 ns/B; NEON 0.25). Which state a
run draws varies (28-70% of rounds). n threads each hashing 1 MiB at once,
ns/B per thread: SME2 .170 / .176 / .322 / .47 / .87-.94 for n = 1, 2, 4,
8, 16; NEON .252 / .252 / .258 / .259 / .31-.33.

**Concurrent SME2 sends a process's threads to E-cores.** A thread that
ran a long SME2 call itself, then sleeps while two other threads of its
process run long SME2 calls together (started at once), takes 4-10% of
its following work on an E-core; with no SME2 of its own, one copy in
SME2, or a staggered start, about none; it stops as soon as they stop
(`probe/ecore-trigger`, jobs 042-043). In the benchmark this put 4-5% of
every contender's small solo samples on E-cores, and the turn removes it
(5.1% -> 0.0%). Mechanism unknown: three concurrent SME2 copies read 0-1
in 1000, so "the scheduler seeks a free SME unit" does not fit.

**SME2 batch calls want to run back to back.** On the Mac, 128 one-block
messages in one SME2 call take 1292 ns back to back and 1500 after any
0.25-16 µs gap of pure ALU work (NEON: level). A pass over memory before
the call costs more: 512 messages 9.7 -> 12.3-12.6 ns each after a length
sum, a scalar pass, or a read of the digests, on the Mac and the VM alike.
Acquire/release atomics beside the SME2 kernels cost about 1 µs per call
on the VM; an atomic swap cost a batch of 24 messages 3-9%. Realistic SME2
batch rates, with a program's own work between calls, are nearer 12 than
the benchmark's 9.5-10 ns/msg. Hypothesis (untested): the SME unit reads at
L2, so lines the core just touched cost coherence traffic.

**NEON goes cold.** On the Mac, NEON work after 20 µs of scalar-only code
takes 1667 ns against about 900 warm; after long SME2 work it is no slower
than that. So the old "first NEON after SME2 costs 4 µs" is NEON going cold
during any stretch without vector work, not a mode-switch cost. It shows
in tight loops: 1000 one-block messages 11.67 ns each, 1024 only 9.45 (the
last eight run on NEON after seven SME2 calls). Padding the remainder onto
SME2 lost everywhere at 24 messages; open (`probe/transition`).

**SME2 remainders: a penalty that follows machine state, not the
remainder** (probe/neon-cold, jobs 125-126, September 25, 2026). With the
same pieces for every variant, the NEON remainder after the SME2 groups
and before them cost the same on P- and E-cores. Some sizes pay 20-28%
on P-cores (31, 111, 215, 500-511, 1023 messages) and others with the same
remainder do not (24, 100, 104, 200, 1016); 500 paid nothing in job 125
and 25% in job 126. Padding the remainder into one more SME2 group
removes the penalty where it strikes (-18 to -20%) and costs 20-70% where
it does not, and E-core cycles: a trade. The first probe's "NEON first
wins 19-27%" came from its harness (hash_many rebuilds its pointer table
per call; the variants did not): compare variants built from the same
pieces.

The slow state is visible in the cycle counter (jobs 127-133, the Mac
quiet): 3.93 core cycles per ns when the SME unit runs fast, 3.20-3.30
when it runs slow; the difference is time the core waits on the unit.
Each measurement can be classified by that ratio, which beats comparing
wall times across runs (unchanged code paths moved up to +-35% between
jobs as the state flipped). What puts the unit in the slow state, as far
as measured: the digests read between calls when the pass takes about
0.25 us or more (1008 and more digests slow, 512 fast); NEON leftovers
of about 210 ns or more after the groups (p8 + p7 always; p9 + p3 after 96
messages); leftovers of 4-8 after about 500 messages. Not explained:
after two or more groups the overlap group below also runs slow though it
leaves no NEON work.

Tried: 13-15 one-block leftovers as one more, overlapping, SME2 group
read in place (candidate/overlap-group, not promoted). Against servil
through the API (jobs 130-133, old/new/new/old): 29-31 messages -20 to
-22% on P-cores in both modes (old slow, new fast), 45-47 -18 to -20%
back to back but +4 to +5% with reads, 109-111 +8 to +11% (both slow; the
second streaming session costs), 253-511 -3 to +4%, 1021 and up level. A
trade. A Pareto version needs the extra group inside the same streaming
session (a kernel entry taking a separate last group), whose cost (about
150 ns per group) is below the NEON leftovers' (250-280 ns) in either
state.

**Idle threads cost the busy ones.** Beside eight hashing threads, eight
idle ones: asleep, free; spinning on loads +18% (VM and Mac); `sched_yield`
in a loop +36% on the VM, +2% on the Mac. `WFE` returns every 0.1-1.3 µs
on both, so it is a spin. A wake costs the waker 12-20 µs on the VM and the
sleeper arrives 40-60 µs later.

**`hash()` has no fixed cost to shave.** 1 to 16 blocks on the VM fit
42.8-43.5 ns per block and -2 to -3 ns fixed (September 25, 2026): up to
1 KiB the time is the compression chain alone.

**One scalar lane beside NEON pays on both core kinds; a second does
not.** Parent plans p3/p5/p7/p9 (a scalar lane beside the NEON parents)
beat running p2/p4/p8 and k1 in turn by 8-38% on P- and E-cores alike
(probe/mixed-parents, jobs 121-122); the second scalar lane of k4, k6, k8,
k10 is what E-cores pay for.

**Frames and zeroing matter at small sizes.** `compress_subtree_to_parent_node`
is `inline(never)` (its arrays in `hash()`'s frame made every call probe
and zero them); arrays sized for 16 chaining values at 2-16 KiB in place
of 6 KiB made 2-4 KiB 4-5% faster. Watch frame sizes whenever
`MAX_SIMD_DEGREE` or a stack buffer changes.

## The design, and why

**Single chunk (to 1 KiB)**: c1 does the whole chunk and root in one call,
saves only the registers it names, keeps flags and counter in registers,
reads a full final block in place (a short one from a padded copy).

**NEON plans** (`CHUNK_PLANS`): each count's plan was the one with the
smallest worst ratio to the best plan across a P-core, an E-core, and the
VM, then the user's choice of P-core speed for 4 and 8 chunks. q kernels:
the last scalar lane hashes a partial chunk (block count and final length
in a seventh argument, bytes from a zero-padded copy) beside the whole
chunks for its own blocks, then a second loop finishes the whole ones. Ten
whole chunks keep one k10 call and the partial chunk after it (every split
costs more); one whole chunk uses q1 (a NEON pair with a duplicate lane)
from five partial blocks (below that P-cores lost up to 14%).

**SME2**: eight groups per entry into streaming mode (the entry costs about
half a microsecond). Serial calls use SME2 under the turn; the pool never
does (pacing SME2 against NEON, permits per SME unit, and SME-only workers
were all measured; see Rejected).

**The turn** (30c599b, the user's decision): a flag on its own cache line,
plain load and store (two calls taking it at one instant can both get it,
which is what running SME2 at once always cost). Accepted costs: shared
cells lose the mode where both copies hold an SME unit, shared small
batches +6-16%. `perf_regress` gives no verdict on it (it moves the
control, SHA-256 in the same process).

**The pool** (`lanes.rs`): `cpus - 1` workers plus callers; jobs in a
lock-free slot table; pieces pulled through a cursor, shrinking toward the
end (`next_piece_len`, 8-128 KiB, so the slowest thread's last piece is
small); one `active` count limits reservations and publishes results;
workers ranked (rank r takes from a job only 20 ns x r after registration,
so small calls wake few); polls spin with `spin_loop` and yield every
20 µs; a worker sleeps after 200 µs without a piece; a call arriving when
callers fill the CPUs hashes its input whole. Every piece runs NEON (on
the Mac the NEON pool beat the SME2 pool at every mt point from 128 KiB).
Polls are loads only; `cursor` and `active` sit on their own lines
(fifteen pollers' RMWs had stalled a caller's register by 10-150 µs).
`initialize()` spawns the workers; the first multithreaded call does if the
program has not (documented: up to tens of milliseconds, once).

**Batches over the pool**: messages gathered into ranges by
`next_piece_len`; servil mt below the split threshold hashes on the
calling thread, and fewer than 1024 messages of a block or less go there
with no length pass (a pass before SME2 kernels cost 25%).

**TABLE = 128** one-block messages per platform call (VM, 1024 messages:
128 -> 9.7 ns, 256 -> 10.2, 512 and up 12.5); likely the same
back-to-back effect; not measured natively.

## Rejected (with the reason; do not retry without new evidence)

- Lanes (one worker per SME unit, fair-share admission): three threads on
  an M4 Max, lost to single-threaded from 128 KiB under two callers.
- Rayon-style pools per caller: two halve each other.
- **Straggler backoff / latency-based stand-down**: cannot tell another
  process from an unrelated thread of its own. Do not reintroduce.
- `spin_loop` polls without ever yielding: hold a scheduler quantum.
- Longer spins before sleeping (10 ms): fifteen spinning vCPUs starve the
  callers of host time.
- Wake cascades (a woken worker waking the next): the wake sits on the
  hashing path.
- Splitting from 32 KiB; smaller minimum pieces; equal-size pieces; longest
  piece 64 or 256 KiB: slower or level.
- WFE in place of spinning: returns every 1.3 µs natively too.
- SME2 permits per unit in the pool; pacing SME2 against NEON (locked both
  shared copies into a mode slower than NEON, 112-135 switches a run);
  SME-only workers (tag `experiment/sme-only-workers`, 5-14% faster for mt
  on the Mac, level on the VM; superseded by the NEON pool and the turn).
- Folding G's rotations into its xors: 7-13% slower.
- Two scalar chunks for 2 KiB (0.587 against k2's 0.455 ns/B, VM); two
  scalar lanes for one chunk plus a partial (E-cores 14-27% slower).
- A padded SME2 group for a batch's remainder: 80-290% slower at 24
  messages on the VM, every variant.
- k4 as two NEON pairs and the "minimax" plans (no second scalar chunk):
  E-cores 16-24% faster, P-cores up to 17% slower; the user rejected them.
- A direct small-tree path (1-2% at 4 KiB) and parents plus root folded
  into k4 (about 3.6%): estimated, not built; complexity for one size.
- q4 as one quad for the four whole chunks (September 25, 2026): 15%
  slower at 4470 B on the VM than two pairs; the quad transposes its
  messages through the stack every block.
- Whole chunks, then the partial chunk alone, for 4, 6, or 8 whole chunks
  and a short partial one (k4/k6/k8 + c1 in place of q4/q6/q8): VM up to
  8% faster at 8 KiB + 1 block, 4-5% to three blocks; a trade, since it
  keeps two scalar lanes, which E-cores pay for (k8 14170 E cycles against
  two quads' 13310).

## Tooling and its pitfalls

**perf_regress** (`check` = working tree against HEAD, `compare OLD NEW`):
eight runs A B B A A B B A of sha256 (the control), servil, and servil mt
at 29 points, 48 rounds; a cell is slower when all four pairs' 5th
percentiles are more than 3% above; a second eight must agree; the control
moving means no verdict. Calibrated on 32 runs of one commit: no false flag
in 2400 cell comparisons; 5% slower caught 70%, 10% 95%, 20% always. Each
listed cell also shows its 90th-percentile ratio, which the verdict ignores
(a two-speed cell's 5th percentile sees only the fast speed). It measures
only its points; a kernel no point exercises can change unseen (4 KiB was
added for that). `check --against <last release>` before a release.

**bench-hashes depends on the fork by git** at the commit its
`Cargo.lock` pins (what users measure). `perf_regress` and the Mac runner
build it against a local checkout with `cargo --config
'patch."https://github.com/johnservil/BLAKE3".blake3-servil.path=".."'`
(bench-hashes nested in that checkout); `perf_regress` restores the
`Cargo.lock` the patch rewrites. Records of fork commit X: pin X in
bench-hashes (`cargo update -p blake3-servil --precise X`), run
unpatched, commit the lock with the records.

Pitfalls met (keep them in mind):
- Records made through the runner before September 25, 2026 show the fork
  as `dirty-…`: the fresh clone's nested bench-hashes was an untracked
  directory in the fork's status. bench-hashes' `build.rs` now leaves it
  out.
- The hook runs `perf_regress` inside `git commit`, where git's GIT_*
  variables once made `git worktree add` overwrite the index being
  committed: commit a227f6d is empty. Fixed (every subprocess drops GIT_*;
  the hook refuses a check that changes the index).
- In the VM every commit that touches code needs the build environment
  (`CC=clang-19 ...`) on the `git commit` itself. Without it the fork
  builds without SME2 (a warning; since September 25, 2026, the user's
  choice for Linux systems with older compilers), and `perf_regress` fails
  stop on an SME2 machine, which aborts the commit and silently leaves the
  branch where it was: check `git log` before pushing or naming a commit
  in a job.
- `git reset --soft X && git commit` records the index; add `-a`.
- Rebases and `--no-verify` skip the hook: compare such commits by hand.

**Mac runner** (`tools/runner/README.md`): the user starts it with
`sh ~/piplayground/blake3-servil/tools/runner/setup-mac.sh`; jobs are JSON
in `runner/jobs/NNN-name.json` (full hex commits, pushed first; a rerun
needs a new name); `pypy3 tools/runner/wait_for.py NNN-name` waits idle.
Nothing heavy may run in the VM during a Mac job (they share cores). Its
results matched the user's own Terminal runs (704 cells, median ratio
1.001). Results belong to the runner's account: they cannot be moved.

**Probe timing**: `examples/support/clocks.rs` (included with `#[path]`)
measures wall time and cycles per core kind together, per batch, and
shows their ratio; use it in every probe (AGENTS.md, "Measuring"). The
record of switching between the two: cycle normalization in the benchmark
until September 2026 (removed: it hid SME2 waits), cycles for the E-core
kernel probes (wall time swings 2x with the E clock), wall time for the
SME2 remainder probes, where cycles per ns then exposed the slow state.

**Probes on the Mac**: the runner runs only allow-listed examples, so a
probe replaces `examples/host_lab.rs` on a `probe/<topic>` branch (never
merged; examples only, so the hook skips it) and runs as a `host_lab`
job. Measure with `thread_selfcounts` cycles per perf level at
user-interactive and background QoS; A/B as old / new / new / old jobs,
each probe on its own base. Kept probes: `probe/ecore-kernels`,
`probe/ecore-trigger`, `probe/transition`, `probe/sme2-gap`; bench-hashes
`probe/fork-no-sme2` builds the fork with `no_sme2`.

**The minimax list**: `pypy3 tools/losses.py <samples.tsv>`; compares
medians (servil against single-threaded contenders, servil mt against
all); marks two-speed cells.

## Testing

    cargo test --release --lib                      # 67 tests
    cargo test --release --features no_sme2 --lib   # 66
    cargo test --release --features pure --lib      # 56
    cargo test --release --doc                      # 19
    cargo test --release --manifest-path test_vectors/Cargo.toml   # 2
    cargo test --release --manifest-path bench-hashes/Cargo.toml   # 7

Among them: every chunk count and every q kernel at every partial length
against the portable compressor; every length from one chunk and a byte to
seventeen chunks against the reference implementation; the pool's cuts,
merges, caps, and 32 concurrent callers. In the VM add the usual
`HOME=/workspace/vm/home CARGO_TARGET_DIR=/tmp/target CC=clang-19
TMPDIR=/tmp` prefix.

## Future work

- A GPU kernel (Metal) for large inputs; the VM has no GPU.
