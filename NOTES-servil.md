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
| Batches | `src/many.rs` | One digest per message: one-block messages on the parent kernels, 2-16 blocks on the NEON parent plans or padded SME2 groups, 2-15 chunks side by side on SME2 |
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
  Parents: SME2 for groups of 16, NEON below. Root: scalar. Whole
  power-of-two subtrees of 32 KiB to 1 MiB: the flat walk, SME2 alone,
  with two integer chunks beside each group of 16 from 256 KiB.
- `hash_many(input, message_len, out)`: equal messages back to back in one
  buffer (the only batch API; the slice-of-slices one went, Zooko,
  September 26). `many::hash_many_on` picks by length:
  - The padded batch contract (7cd10ec, Zooko's decision): message i
    sits at i x `slot_len(len)` (len rounded up to 64; 64 for empty), zero
    past its end; kernels record the last block's length (SME2 message
    kernel: flags bits 40-47; NEON plans: packed last_len). Multiples of 64
    are unchanged; 64-byte messages keep their own entry (a shared one cost
    64 B x 3 7%).
  - One block (64 B): the platform's `hash_many` TABLE (128) at a time,
    SME2 parent-kernel groups of 16 from 16 messages, the NEON parent
    plans (p2-p9) below and for remainders; on SME2 from 11 messages, and
    from 13 left over past groups, every group on the message kernel with
    a padded last group (ffcef50; `ONE_BLOCK_PAD_MIN`). Short blocks
    (1-63 B) on SME2 always take the message kernel from 11.
  - Two chunks (1025-2048 B) below the SME2 threshold, and on NEON-only
    CPUs: side by side on the NEON parent plans (first chunks, second
    chunks, roots; 0869c79): Mac 2000 B x 4 1050 -> 703 ns/msg.
  - 2-16 whole blocks (`hash_blocks`, 9fd0ac9, 666e550): below ten
    messages the integer + NEON parent plans (p2-p9 take any block count
    at one counter); from ten (`SME2_TAIL_MIN`), and from five past whole
    groups (`SME2_TAIL_MIN_AFTER_GROUPS`), whole SME2 groups with the last
    one padded, its spare lanes pointing at the last message again and
    the message kernel storing only the real lanes (bits 24-31 of its
    packed flags); fewer left over past the groups on the plans, before
    the groups. CPUs without SHA-3 keep the C four-lane kernel (a spare
    fourth lane for three, c1 for one or two). 256 B, ns/msg, 37f1247 /
    666e550, VM: 2 180/104, 3 181/76, 5 105/70, 9 99/60, 12 91/51, 15
    106/41, 24 74/51; Mac P-core (jobs 256-266): 2 184/116, 3 183/79,
    12 99/50, 15 115/40, 24 73/50. SHA-256 ring takes about 93 (VM).
  - 2-15 chunks (`hash_chunked`, 0bed4e7): on SME2, from
    `SME2_CHUNKED_MIN` messages (7-12 by chunk count), sixteen side by
    side: chunk k of every lane on the message kernel at counter k (the
    whole chunks in one call, the counter stepping per group through bits
    32-63 of the flags word; the last chunk in a second), then each
    parent level with the pairs gathered into contiguous blocks, then the
    root; the last group padded; fewer on `hash()` one at a time. Mac
    P-core, 16 messages, ns/msg before / after: 2 KiB 987/331, 4 KiB
    1310/661, 8 KiB 2047/1305, 15 KiB 3958/2751 (jobs 268-271). Its
    scratch (15 KiB) stays uninitialised: zeroed, it cost a third of the
    time (VM 16 x 4 KiB 869 against 665 ns/msg).
  - Other lengths: `hash()` one at a time.
  `many::hash_run` stays `#[inline(never)]`: inlined for all sixteen
  lengths, two-message batches took 30% longer. Open: tails of one to
  four messages past the SME2 groups still run in the SME unit's slow
  state (256 B x 18-20: 62 ns/msg against 38 at 16; 34-36 and 100 alike),
  the SME2 remainder problem again.
- `hash_multithreaded()`: below 64 KiB, `hash()`'s path; from 64 KiB the
  pool on NEON only. Batches: under 64 KiB in all the serial path; else
  the pool in ranges of messages, NEON only.
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

**The core's integer units run beside the SME unit** (probe/sme-scalar,
job 142, M4 Max, cycles per loop iteration of 24 SME2 vector ops and
24-192 integer ops of BLAKE3's add/xor/rotate pattern): P-core, SME2 alone
24.0, integer alone 16.5 (96 ops) or 33.2 (192), both 24.0 and 31.9;
E-core 24.2, 24.4, 48.5, both 24.4 and 48.4. Both together cost the
longer, never the sum: the unit takes one vector op per core cycle, and
the core does about 4 integer ops per cycle beside it. So an SME2
kernel can carry integer BLAKE3 lanes for free: one latency-bound chain
(168 cycles a block) fits about 4 blocks beside each 16-lane group
compression (about 680 cycles), a group of 16 SME2 chunks plus 4 integer
chunks, +25% throughput. **Streaming mode lowers the P-core clock**: the
same integer loop, same cycles, 3.93 cycles per ns inside streaming mode
against 4.51 outside (13% more wall time); on E-cores 2.5 either way.
The "fast state" of the SME2 remainders below is the streaming clock.

**An SME2 chunk kernel with an integer lane** (tools/gen_sme2_hybrid.py on
probe/sme2-hybrid, job 143): 16 SME2 chunks plus k chunks on the integer
lane, k blocks beside each SME2 block step. Per chunk against the plain
kernel, M4 Max P-core: k = 1 -5.2%, k = 2 -9.5% (0.145 -> 0.131 ns/B), k = 4
+1% (integer-bound); rerun (job 144, the Mac quiet): -6.0%, -10.1%, +0.5%.
E-core cycles, which the E clock does not move: +3.3%, -2.7%, +44% (job
143) and +2.2%, -2.2%, +44% (job 144); E wall time swung with the clock
(143: k = 2 +4.0% at 2.42 cycles per ns against the baseline's 2.58; 144:
-1.5% at equal clocks). So k = 2 is faster on both core kinds, not a
trade; k = 4 is integer-bound on E-cores. A case for judging wall time
within one state: the first reading of E wall time called it a trade.
Earlier two-SME2-unit evidence (github.io/blake3-sme2, an earlier project
of ours): two SME2 threads at 83.7 ps/B against one's 163.3 over 1 GiB,
and SME2 threads on both P clusters and the E cluster plus NEON threads
the fastest all-core plan. Integration: groups of
18 fit no power-of-two chunk count, which the tree walk hands hash_many
(128 = 7 x 18 + 2; exact mixes of 18- and 16-chunk groups start at 256 = 8 x
18 + 7 x 16, 6.25% of the chunks on the lane; 512 = 24 x 18 + 5 x 16, 9.4%).

**The flat walk** (`sme2::compress_subtree_flat`, a3aa406 and 23b8b42):
each whole power-of-two subtree of 32 to 1024 chunks goes bottom up on
SME2 alone, chunks in groups of 18 (from 256 chunks) and 16, then every
parent level on the SME2 parent kernel; nothing runs on NEON between SME2
kernels. From 256 KiB (a3aa406, perf_regress against 8a66cf5): Mac solo
256 KiB -5.0%, 1 MiB -12.0%, 3 MiB -15.3%, 8 MiB -11.3%, shared 1-8 MiB
-13 to -15% (job 145); VM 256 KiB -5.2%, 1 MiB -11.5%, 3 MiB -10.7%.
Widened to 32 KiB (23b8b42, groups of 16 alone below 256 chunks): single-
threaded level on both machines (32-128 KiB, and streamed, whose 64 KiB
pieces take it: Mac jobs 149-152, VM A/B -0.7 to -1.2% streamed solo);
servil mt 64 KiB faster on the Mac, solo -6.7% / -11.1%, shared -19.4% /
-5.6%. Its CV arrays (56 KiB of stack, uninitialised) cost nothing
measurable at 32 KiB. Open: the Hasher's piece sizes other than powers of
two from 32 KiB, and inputs below 32 KiB (15 or fewer parents per level).

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

Taken: 13-15 one-block leftovers as one more, overlapping, SME2 group
read in place (4d0751f; `OVERLAP_MIN`; Zooko accepted the trade on
September 25, under the rule that a slowed cell stays ahead of every
competitor). Against the NEON leftovers
through the API (jobs 130-133, old/new/new/old): 29-31 messages -20 to
-22% on P-cores in both modes (old slow, new fast), 45-47 -18 to -20%
back to back but +4 to +5% with reads, 109-111 +8 to +11% (both slow; the
second streaming session costs), 253-511 -3 to +4%, 1021 and up level. A
trade. A Pareto version needs the extra group inside the same streaming
session (a kernel entry taking a separate last group), whose cost (about
150 ns per group) is below the NEON leftovers' (250-280 ns) in either
state.

**The slow state, measured directly** (probe/piece-sizes and
probe/stack-map, jobs 162-173, September 25, 2026, M4 Max P-core; the VM
alike). The raw kernels over 8 MiB in 64 KiB pieces (chunks, then parent
levels): 0.152 ns/B at 3.94 cycles per ns back to back and with 0.1 us of
scalar work between pieces; 0.179 (3.41/ns) at 0.25 us; 0.195-0.20
(3.20/ns) from 0.5 us. So about 0.2 us of other work between SME2 kernels
puts the unit in its slow state for the next ones, about 2.5 us per 64
KiB piece. Streaming sessions back to back cost no state (three per piece
ran at 3.94). Three placements also put it there, each at some addresses
and not others:

- 32-byte chaining-value stores off a 32-byte boundary (VM, heap
  buffers: aligned 0.159 ns/B, odd multiples of 16 bytes up to 0.193).
- The walk's buffers on the stack: a 64 KiB frame is probed with a store
  into each 4 KiB page on every call (`str xzr, [sp]`), and those stores
  land among the buffers the SME unit is about to use. hash(32 KiB) by
  stack depth, Mac, deterministic per address: 0.178-0.245 ns/B with the
  buffers on the stack, 0.177-0.181 at every depth with them on the heap
  (jobs 169-173).
- Buffers at the same address mod 4 KiB (the stack arrays of 8, 32, and 16
  KiB all started at one residue); with distinct residues the heap scratch
  ran fast at all 64 offsets from the input (job 170).

The flat walk now runs down to the two children on SME2 (padded groups
below sixteen parents), in a per-thread 64 KiB scratch on 128-byte lines
with its buffers at 0, 1, and 3 KiB mod 4 KiB (483162d). Mac, hash() in a
loop, ns/B before / after: 32 KiB 0.213 / 0.181, 64 KiB 0.205 / 0.166, 128
KiB 0.201 / 0.160, 256 KiB 0.172 / 0.1695, 1 MiB 0.152 / 0.151. Open:

- 256 KiB takes 2.12x as long as 128 KiB (0.1695 against 0.160 ns/B;
  3.48-3.50 cycles per ns). Not the integer lane: without it 256 KiB runs
  0.1768 at 3.50/ns and 1 MiB 0.164 against 0.151 with it (probe/no-lane,
  jobs 181-184), so the lane stays. The Hasher's 256 KiB pieces run the
  same kernels at 3.70/ns and 0.1596; what hash(256 KiB) adds is open.
- The Hasher in 64 KiB pieces stays at 0.204 ns/B (3.21-3.25/ns). Past
  the first input it now pushes one chaining value per subtree (the walk
  runs down to it, 0220e69): 256 KiB pieces 0.1676 -> 0.1596 (-4.8%, jobs
  176-179), 64 KiB level. What else falls between its SME2 kernels is
  open; the next idea is folding the stack's merges into the walk's
  padded levels.
- The VM stays two-speed per process at 32-64 KiB (0.172 or 0.210), with
  the scratch on or off the stack: guest pages land at host addresses the
  guest cannot see, so a placement effect above 4 KiB would show this way.

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

**The SME unit writing lines the core has just written costs** (VM,
September 26, 2026). A scratch output array for a padded SME2 group,
zeroed by the core and then written by the kernel, cost 16 x 256 B 46
ns/msg against 38 written straight to the caller's buffer (uninitialised
scratch: 41; the core's copy out, a fraction of it). The side-by-side
chunk path's zeroed 15 KiB scratch cost 16 x 4 KiB 869 ns/msg against 665
uninitialised. Hence kernels that store only the lanes asked for, and
scratch left uninitialised for the kernels to write first.

**Energy per byte** (probe/energy, jobs 154-160, September 25, 2026, M4
Max, the Mac quiet). The process's `proc_pid_rusage` RUSAGE_INFO_V6
counters: `ri_energy_nj` and `ri_penergy_nj` read sleep as 0.000-0.009 W
and a scalar spin loop as a steady 2.7-3.4 W on a P-core, 0.046-0.057 W
on an E-core; `ri_billed_energy` reads 0 always (unused). They are the
kernel's estimate; whether it includes the SME unit's own power is
unknown (powermetrics, which needs root, would tell). pJ/B, median of 5
stretches, spread under 5% unless noted:

    kernel (single thread)        P-core QoS          E-core QoS
    SME2, 1-8 MiB                 689  (4.5 W)        85-89 (0.15-0.18 W)
    SME2, 16 KiB-256 KiB          620-703             67-81
    SME2 batches (1024 x 64 B)    522-571             67-70
    NEON (no_sme2), 16 KiB-8 MiB  1850-1990 (7.5-7.9 W) 222-228
    NEON hybrids, 8 KiB           1891-2023           210-239
    scalar c1, 1 KiB              2585-2741           318-345

So SME2 costs 2.5-3x less energy per byte than NEON on both core kinds,
and E-cores 8-9x less than P-cores for the same kernel (3x slower). The
multithreaded calls, 8 MiB: all threads 1816-1891 pJ/B (72 W); with a
budget of 2 threads 2743 (20 W) against 1226 for the caller and one
scoped thread each running `hash`: the pool's idle workers, polling on
P-cores through the call, cost about half of it. At background QoS the
pool's workers stay on P-cores (95-97% of the energy is P-core energy).

Candidate efficient designs, 8 MiB in 64 KiB pieces pulled from a cursor
(threads spawned per call; job 158, then 160), caller at P QoS: the
caller on `hash` beside 4 E-core helpers on NEON (background QoS) 0.120 /
0.138 ns/B and 464 / 454 pJ/B, against `hash` alone 0.152 and 689:
faster and a third less energy. At 1 MiB level (0.157 against 0.152); at
256 KiB 1.7x slower (spawn and the helpers' first pieces). The caller
beside one E-core thread on `hash` (both SME units) 0.177: slower than the
caller alone, rejected. The caller beside 4 P-core NEON threads 0.060
ns/B at 1450 pJ/B, against the pool with 4 threads, 0.074 at 2300: the
caller's own pieces on SME2 and no idle pollers beat the pool on both.
At background QoS the helpers make the call 2.2x faster at about twice
the energy (NEON on E-cores 225 pJ/B against SME2's 85).

**A user's Apple M3 Ultra** (20 P + 8 E cores, two dies, no SME2; fork
b74b59e, bench d28326e, quiet; kept in the fork's tmp/AppleM3Ultra.darwin25/).
What it tells us about the machines without SME2 (every M1-M3):

- servil st from 8 KiB runs the NEON hybrids at 0.283-0.290 ns/B, a lead
  of only 12-14% over SHA-256 ring's 0.32-0.33; on the M4 SME2 doubles it.
  On pre-M4 Apple chips this is a minimax cell: the NEON path's bulk rate.
- servil mt, 28 CPUs: 128 MiB 0.016 ns/B; 64-128 KiB only 2x servil st
  (0.143, 0.137), as on the M4 Max. There, and only there, two copies ran
  faster than one (shared 0.128 and 0.123): the M4 Max and the VM show the
  opposite. Unexplained; a guess is a solo call's workers straddling the
  two dies. Nothing to test it on.
- Batches: 18.8-19.2 ns per message from 8 messages (SHA-256 34), servil
  mt 2.3 at 262144.
- Streamed (the old Hasher): 2304-7935 B took up to 1.8x one call of the
  same size (3 KiB 0.604 against 0.358 ns/B); Stream hashes such inputs in
  one call (M4 record: 0.362 against 0.350).

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

**The SME2 thread** (0366e48, September 25, 2026): a multithreaded call
that gets the SME2 turn hashes a prefix of its input itself, in whole
subtrees of up to 1 MiB back to back, sized to 1.65 NEON threads' worth
(none below 128 KiB), then helps on NEON. The split came from a sweep
(probe/sme2-thread, job 187: 62% beside one NEON thread, 38% beside three,
12% beside fifteen, on both machines). Budgets, against the NEON pool
(jobs 188-191): Mac 2 threads -20 to -25%, 4 -8 to -14%, 8 -3 to -8%, all
level to -7%; VM 2 -24 to -29%, 4 -13 to -19%, 8 -3 to -11%, all level. Two
threads now beat one at every size (they did not: 1 MiB 0.160 against
0.146 on the Mac). A 24 KiB prefix at 256 KiB over 16 threads cost 18-20%
(VM), hence the floor. **Two SME units are reachable**: two threads running
the raw chunk kernel at once, 2.0-2.4x one's throughput on the Mac, 1.7-2.0x
on the VM (job 187); a second SME2 thread in the pool is the next idea.

**Stream** (59ad5c1, c242b2e): hashing behind the caller. Four 1 MiB
buffers (whole subtrees, the flat walk's largest) per stream; the caller
fills one (`buffer()`/`filled(n)`, or `update_reader`, which reads into
them) while
a hashing thread runs `Hasher::update` (or the crate-internal
`update_multithreaded`) on full
ones; `buffer()` blocks when all four are out. The hashing thread and the
buffers are the calling thread's, kept asleep for its next stream (a
thread-local); a stream shorter than a buffer is hashed in place at
finalize. Mac, streamed 64 KiB pieces with the producer's copy (record
ec5de66): servil 1 MiB 0.161 ns/B, 8 MiB 0.157, 128 MiB 0.150; servil mt 1
MiB 0.046, 8 MiB 0.041. Open: short streams' fill and drain, the per-stream
wake, 64 B overhead (0.91 against hash()'s 0.70 ns/B).

**Streaming scope** (Zooko, September 25, 2026): input in memory goes to
one call (`hash`, `hash_multithreaded`); input that arrives goes to a
`Stream`, each read landing in its buffers. `Hasher::update_multithreaded`
(caller and hashing taking turns) left the public API, and `Stream`'s
copying `update` and `io::Write` went with it; a caller whose pieces arrive
in buffers it does not own copies them into `buffer()` itself. The
benchmark's streamed use case measures that one pattern, each piece's
read standing as a memory copy for every contender.

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
- The pool's caller hashing its own pieces on SME2 under the turn, the
  workers on NEON (probe/sme2-caller, September 25, 2026): slower on both
  machines and faster nowhere. VM: servil mt 256 KiB +18%, batches
  2048-16384 +5 to +18%; Mac (job 161): 256 KiB +8 to +10%, batches
  4096-16384 +8 to +11%, shared 1024 messages +35 to +46%. The scoped
  probe's caller-on-SME2 gain (probe/energy) compared unequal thread
  counts. Hypotheses, untested: a streaming entry per piece (8-128 KiB)
  and the SME unit's slow state between pieces; the cluster's clock
  lowered for the NEON workers beside a streaming core.
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
percentiles are more than 3% above (solo cells) or 10% above (shared
cells, since September 25, 2026); a second eight must agree; the control
moving means no verdict. Confirmed solo cells hold the change (exit 1);
confirmed shared cells are listed beside exit 0, and the commit message
names them and the reason (since September 26, 2026). Calibrated on 32 runs of one commit: no false flag
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

    cargo test --release --lib                      # 86 tests, 1 ignored
    cargo test --release --features no_sme2 --lib   # 82
    cargo test --release --features pure --lib      # 71
    cargo test --release --doc                      # 21
    cargo test --release --manifest-path test_vectors/Cargo.toml   # 2
    cargo test --release --manifest-path bench-hashes/Cargo.toml   # 7

Among them: every chunk count and every q kernel at every partial length
against the portable compressor; every length from one chunk and a byte to
seventeen chunks against the reference implementation; the pool's cuts,
merges, caps, and 32 concurrent callers; every platform's `hash_many` at
every input shape against the portable one (`test::platform_hash_many`);
every kernel against guard pages (`test::guard_pages`: inputs and outputs
flush against an inaccessible page, so one byte read or written outside
them faults, which the sanitizers cannot see in assembly); Miri-sized
unsafe paths (`test::unsafe_paths`). In the VM add the usual
`HOME=/workspace/vm/home CARGO_TARGET_DIR=/tmp/target CC=clang-19
TMPDIR=/tmp` prefix.

A long differential run against the reference implementation, off by
default: `BLAKE3_DIFF_SECONDS=1200 BLAKE3_DIFF_SEED=2 cargo test --release
--lib -- --ignored differential --nocapture` (every entry point, lengths
skewed toward block and chunk boundaries up to 4 MiB, one to four threads
at once, so the turn, the pool, and streams meet).

**Checks beyond the suites** (September 26, 2026; nightly Rust with
`rustup toolchain install nightly --component miri,rust-src,llvm-tools`,
logs in `tmp/quality/`, outside git):

- Coverage: `RUSTFLAGS="-C instrument-coverage"` and the toolchain's
  `llvm-profdata` / `llvm-cov` (`rustup component add llvm-tools`) over
  the library tests: 93% of lines. What it showed unexercised became the
  platform, guard-page, and C-kernel tests (d395f9e); the rest is
  platform-absent code (x86, no SHA-3) and fail-stop branches.
- AddressSanitizer: `RUSTFLAGS="-Zsanitizer=address" cargo +nightly test
  -Zbuild-std --target aarch64-unknown-linux-gnu --release --lib`: clean.
  A deliberate heap overflow in a control program was caught.
- ThreadSanitizer: the same with `-Zsanitizer=thread`, the pool, stream,
  turn, and unsafe-path tests, 31 runs: clean. A deliberate data race in a
  control program was caught.
- Miri: `MIRIFLAGS=-Zmiri-num-cpus=4 cargo +nightly miri test --features
  pure --lib -- unsafe_paths test_miri_smoketest` (41 minutes; three
  workers): batches, the pool from two callers, a stream past one buffer:
  no undefined behaviour.
- Kani (`cargo kani`, `#[cfg(kani)] mod proofs` in lanes.rs and many.rs;
  QUALITY.md has the install): next_piece_len's bounds, the pool cut's
  loop step (whole subtrees, by induction), slot_len, fill_table to 20
  lanes; 2 s in all but fill_table (45 s). The guest's glibc 2.36 takes
  Kani 0.64.0 at most (0.65 on need 2.39). Lessons: a symbolic divisor
  over all of usize ran 3.5 h without an answer (bound it); the whole cut
  loop ran CBMC out of memory at every bound down to 96 KiB, so prove a
  loop step and argue by induction; always run it under `timeout`.
  fill_table at all 144 lanes: symbolic execution alone took 7 minutes;
  stopped there.
- Found and fixed: hash_blocks' padded-group choice (a latent assert on a
  CPU combination that does not exist), Platform::hash_many silently
  dropping the tail of an input that is not whole blocks (now a compile
  error), a dead Stream branch that would have hashed the last buffer
  alone, and the message kernel's one-block case (a block run twice and
  flags_start missed; no caller used it until the side-by-side path).

## Future work

- A GPU kernel (Metal) for large inputs; the VM has no GPU.
