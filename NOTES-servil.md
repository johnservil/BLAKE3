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
| Platform facts | `examples/scaling.rs`, `host-lab-reports/` (the host lab: `examples/host_lab.rs` at 9868745) | Primitives, scaling, idle waiters, SME2 callers, WFE on the machine it runs on |

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
- Extended output (`OutputReader::fill`): whole groups of sixteen blocks
  on SME2 (`blake3_sme2_xof16_512`: one block broadcast, counters per
  lane, both halves of the state, ZA1 transposing each lane's 64 bytes),
  under the turn; then groups of eight and four on NEON (src/neon_xof.rs:
  two independent four-lane states a step, 0.29 ns/B where one ran 0.38),
  the rest block by block on the portable compressor. Mac (jobs
  1068-1080, A B B A, mains): SME2 0.69-0.71 -> 0.154-0.155 ns/B from 4
  KiB; NEON-only builds (M1-M3's path) 0.70-0.72 -> 0.30-0.31, 1 KiB
  0.80-0.85 -> 0.37-0.38, 256 B 1.07-1.16 -> 0.73-0.74 (October 2).
- Every SME2-sized call takes the turn first (inputs of 16 chunks, batches
  of 16 messages, extended output of 16 blocks).

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
- The VM's two speeds at 32-64 KiB (0.18 or 0.22 ns/B at 32 KiB) came
  from the scratch block's own placement: a `Box` on 128-byte lines puts
  its start wherever malloc chooses, so the layout's residues were off by
  it. hash(32 KiB) by the block's start mod 4 KiB (a 4 KiB-aligned block
  shifted by R, six processes): every multiple of 1 KiB fast (0.18; 64
  KiB 0.166), 0x680, 0x700, 0xe80, 0xf00 always slow (0.22; 0.205), most
  other residues either. glibc gives each thread's block the same residue
  run after run: the first spawned thread 0xd00 (slow in almost every
  process; the benchmark's shared copies and a program's first worker),
  later threads 0xe00 (fast), the main thread 0x180 (by process). Not the
  cause: the binary, ASLR (off: the same), the vCPU (pinned to each of
  16), the input's offset mod 4 KiB (ten offsets, all fast with the
  block aligned). The block now sits on a 4 KiB boundary: five of six
  processes fast on every thread. Open: the sixth ran every spawned
  thread slow at the same virtual residues, which fits host placement the
  guest cannot see; one run in a benchmark switched from fast to slow
  mid-run.

**A call after a 1 ms sleep** (September 28, night; probe/after-gap and
probe/first-call, jobs 478-479 and 533, M4 Max). The thread wakes at about
1.3 GHz and spends a quarter to half of its time on E-cores, SHA-256's
calls alike. hash(4 KiB) takes 5300 cycles back to back and 8900 after
the gap; SHA-256's stay level (6260 and 6390): the NEON hybrids lose more
on E-cores than SHA-256's instructions. 20 us of integer work first
halves it (the clock's ramp). A first call's cost over a second one right
after it: servil 238-244 ns at 64 B, SHA-256 168-204; at 1 KiB 181-300
and 168-301: no cold-start cost of servil's own to shave. The second
call still takes servil 2.6-2.8 us at 1 KiB against SHA-256's 1.15-1.2:
one chunk's sixteen dependent compressions against SHA-256's hardware.

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

**The energy counter's repeatability** (September 28, night,
probe/energy-repeat, job 698; `clocks::process_energy_nj`, M4 Max, each
after 50 ms asleep, two sets of ten): a 100 ms integer spin on a P-core
1.67-3.09 W, medians 2.90 and 2.97 W; hash over 64 MiB 192-422 pJ/B,
medians 262 and 267; hash_multithreaded over 64 MiB 281-956 pJ/B, medians
651 and 685. Single readings spread 1.5-3.4x (the low ones suggest the
kernel attributes energy late, in lumps); medians of ten agree within
about 5%. The lag, measured (job 699, hash over 64 MiB, eight
each): read at once 147-326 pJ/B; the same work read again after 1 ms
asleep 313-441, after 10 ms 384-412 (7% spread), after 100 ms 313-378.
The kernel credits a thread's energy late, at its next block or switch:
a reading taken after the measured threads have slept (10 ms) is complete
and repeatable, one taken at once undercounts by a third and spreads.
The energy probes of jobs 560-567 and 617-624 slept 5 ms before their
last reading; job 698 read at once. For stage 3: sleep, then read.

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
so a call's pieces go to few); polls spin with `spin_loop` and yield every
20 µs, only while a job is registered or a `Hold` lives; a worker that
finds neither sleeps at once (1046c10, Zooko: nothing kept awake between
calls; before, workers spun 200 µs after their last piece, which the
benchmark's back-to-back loops rewarded and few programs meet); a call
arriving when callers fill the CPUs hashes its input whole. Every piece runs NEON (on
the Mac the NEON pool beat the SME2 pool at every mt point from 128 KiB).
Polls are loads only; `cursor` and `active` sit on their own lines
(fifteen pollers' RMWs had stalled a caller's register by 10-150 µs).
`initialize()` spawns the workers; the first multithreaded call does if the
program has not (documented: up to tens of milliseconds, once).

**Waking** (1046c10, September 27, 2026): every call meets sleeping
workers, so a call wakes only those it can use (pieces - 1, within its
budget, less those awake or on their way). The caller wakes one; the
first to wake takes the rest as owed and wakes them one by one before
its first piece. Costs measured (median of 51, after 1 ms): the waker
pays about 3 µs per `notify_one` on the Mac, 4-12 µs in the VM;
`notify_all` of fifteen costs the waker 25-56 µs; a woken worker arrives
15-20 µs after its wake on the Mac, 20-45 µs in the VM, on a core the
idle time slowed. A 1 MiB call on the Mac: the first worker at 10-18 µs,
the last at 45-70 µs, the first woken's own first piece about 35 µs in
(it spends about 2.6 µs per wake). **MIN_SPLIT_LEN = 768 KiB**: with 1 ms
of sleep before each call the split came back sooner from 512 KiB on the
Mac (136 µs against 158; 256-384 KiB level; jobs 364-367) and from
768 KiB in the VM (146 against 181; 512 KiB 154 against 130); Zooko took
the length where it pays on both (hash_multithreaded is built for a low
worst case). **Now 512 KiB** (Zooko, September 28, 2026, native first;
c46c57c): Mac after the gap 512 KiB about 40% faster, the VM's 20-30%
slower. The split once sat at 32 KiB, when the workers polled between
calls; with sleeping workers a wake costs 15-70 µs, more than a 32 KiB
input takes one core. After idle, VM: mt 64 KiB 1.68 -> 0.59 ns/B (st 0.55),
256 KiB 0.50 -> 0.28 (st 0.29), 4096 x 64 B 32.2 -> 17.1 (st 18.1); back to
back, Mac solo 1 MiB mt 0.067 ns/B against st 0.152.

**Two aims, stated in the API docs** (Zooko, September 27, 2026): each
hashing interface is built for top speed (hash, hash_many, Stream: the
fastest way to do the task for a caller that keeps it fed) or for a low
worst case (hash_multithreaded, hash_many_multithreaded: other cores only
where waking them pays on every machine measured, never slower than the
single-threaded form, nothing running between calls). A judgment call
between speed and everyone's worst case goes to the aim the interface is
built for; users who want top speed on a stream of inputs are steered to
the interfaces that keep the engine fed. Zooko on the name of the second
aim: still open ("the what else is a little unclear").

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

**Stream, deleted** (September 27, 2026, Zooko: nothing to distract from
the API plan until the benchmark is frozen). It hashed behind the caller
in four 1 MiB buffers the caller filled (a copy, unless the data arrived
by read), on a hashing thread per calling thread (59ad5c1, c242b2e), and
held the pool while its next buffer waited (a `lanes::Hold`, 3d7102e:
streamed mt 32 MiB 0.065 -> 0.043 ns/B on the Mac). The planned queue
(docs/api-design.md) passes the caller's buffers by ownership instead;
the hold comes back with it. The code is in git history.

**The planned API** (September 28, 2026; docs/api-design.md,
tests/api_plan.rs). `Mode` (plain, keyed, derive-key) and `Threads` (One,
All, Budget(n)) are arguments of `hash_with` and `hash_many_with`; the
`_with_budget` functions went. Every batch path takes the mode's key words
and flags (`many::hash_many_on(input, len, key, flags, ..)`, down to the
SME2 chunked-message kernel and the NEON `hash_messages_raw`, which gained
a flags argument): the mode's flags ride in the low byte of each kernel's
packed flags, beside CHUNK_START / CHUNK_END / PARENT / ROOT, so every mode
costs the same (perf_regress on the VM: level). `initialize()` runs the
self-test alone; `initialize_multithreaded()` also starts the pool.

**The queue, as rebuilt from scratch** (September 28, 2026; branch
candidate/queue-simple; `src/queue.rs` module docs have the mechanism).
The streaming API maximises throughput (Zooko): latency is spent to buy
it, and the hashing threads must never wait on a handover.
- `submit` (the caller's thread) plans tasks: the whole subtrees
  `Hasher::update` would hash, in parts of at most `lanes::TASK_LEN` (64
  KiB; 32 level, 16 KiB 50% slower on streams, jobs 468-473); a message is
  a stream of one piece and is finalized at delivery; `Queue::fixed`
  batches are ranges of slots through `hash_many`'s kernels; messages
  under 16 KiB (`TASK_MIN`) go up to 64 to a task (`lanes::Member`), side
  by side on the multi-lane kernels when each is one block, and so do
  `Queue::fixed` batches under 64 KiB (up to 64 KiB of them). An open
  task goes to the pool when full or when the delivery thread waits on
  it, outside the queue's lock. No one-shot thresholds (the 768 KiB split
  is the one-shot calls' business).
- One task list (`lanes::TASKS`, a Mutex<VecDeque> polled with try_lock).
  An SME2 thread hashes tasks only on SME2 under the turn; the workers
  only on NEON (Zooko's suggestion). Pushes wake a thread per task in
  flight, the SME2 thread first. A task costs the list about a
  microsecond of throughput on the Mac (batches of 16 64-byte messages
  as a task each: 68 ns per message, twice hashing them at delivery),
  hence members.
- One delivery thread holds the pool (`lanes::Hold`) while anything is in
  flight and sleeps with nothing in flight. The submitters and it share
  no lock on their common paths (September 28, night): entries are
  chained in submission order through each slot's `next` (a submitter
  links after `State::tail`; the delivery thread follows from the last
  slot it delivered, which it keeps as the chain's head until the next
  is delivered); each delivered slot goes back through `returned` before
  the handler runs (so the slots follow the program's in-flight count,
  not the threads' timing; handing back a round's slots at its end grew
  the queue by blocks under `tests/queue_no_alloc.rs`); the queue's hold
  is an atomic flag with a store-then-recheck handshake (the submitter
  links then sets it; the delivery thread clears it then looks for a
  link). The delivery thread waits on the entry it stopped at through
  that entry's own count, without the lock.

**Where a 64-byte message's time went** (September 28, night; VM probes
under `perf`, Mac jobs 480-497). The benchmark's program thread is the
bottleneck for short messages: it never waited for a buffer with 1024 in
flight. Its `submit` cost 258 ns on the VM: 50000 futex calls per million
messages (a std Mutex parking the loser of the queue lock, which the
delivery thread held while each front slot's count missed in its cache),
the `Arc` clone per call, the delivery thread's polling taking the
lock's line. After the chain, `lock_polling` (try_lock a while before
parking), and the borrowed state: 3000 futex calls per million, submit
about 110 ns (VM) and 63 ns (Mac, probe/queue-submit, job 492: taking a
returned buffer from the harness's channel another 25 ns). Mac, solo,
old -> new: 64 B messages 2.6 -> 1.15 ns/B, 256 B 0.63 -> 0.29, 16 KiB
0.165 -> 0.072; batches of 16 34 -> 8-19 ns/msg, of 64 21 -> 6.5, of 256
11 -> 4.2; shared batches of 16 84 -> 12-19. SHA-256 does a 64-byte
message in 36 ns, about what a handover's cache lines cost; `Queue::fixed`
is the API that beats it. The delivery thread closing a part-filled
task only after 16 polls (`CLOSE_AFTER_POLLS`): first tried on the VM
alone (level), then on the Mac after a traced run (job 680) showed
batches of 16 at two speeds, 5.5-8.8 and 30-32 ns/msg, the program's
thread busy throughout: handing each part-filled task over at once took
the submitters' lock and made tasks of a few, a loop. Taken (jobs
701-704): batches of 16 at one speed, 5.2-5.4 ns/msg; of 64, the slow
speed 15-16 -> 5.4-5.9; 1 KiB messages 0.18-0.22 -> 0.13-0.14 ns/B; VM
batches of 16 27% faster. Tried and left out: a 64 KiB byte cap
on message members (4 KiB messages 10-30% slower, jobs 494-497); each
slot on a 128-byte line of its own (jobs 504-507), the delivery thread
backing off over idle rounds (jobs 534-537), and a short message's digest
in its slot beside `left` (jobs 544-547): all level on the Mac. A task's
short messages as one chain entry (a group: items in a reused array,
one count, sealed when the task closes or another entry follows): the VM
probe slower (175-190 against 150-166 ns per message; the delivery
thread polls the open group's count while the submitter writes the same
slot), and which slots become groups follows the threads' timing, so
their first reserve came after warm-up (tests/queue_no_alloc.rs: 6
allocations); dropped unbuilt on the Mac. So the
64-byte cell sits at the handover's floor in this harness: the program's
thread, the bottleneck, spends 25 ns taking a returned buffer from its
channel and about 40-60 ns in `submit`, mostly lines another core wrote.

**The 64-byte cell at full clock** (September 28, morning, Mac, the
fixed benchmark; probe/submit-64, jobs 742, 755, 763; traced run, job
762). The program's thread stays the bottleneck (busy 99-100%), at 3.76-
3.90 GHz where SHA-256's lone thread runs at 4.45 (its cluster's other
cores are busy). Per message it retires 432-484 instructions in every
sample, and takes 129-579 cycles (median 158): the spread between
samples, 0.54-2.3 ns/B, is stalls, not clock or core kind (all on P).
Its fastest samples match SHA-256 (0.54 against 0.55 ns/B). `submit`
alone, 1024 pre-filled buffers in a burst: 28-34 ns, 100-127 cycles,
about 320 instructions per message; taking a buffer back from the
program's channel 10 ns and 122 instructions; the copy 55. So both
halves have work: `submit`'s instructions (a lock, the slot, the open
task's member, the link, the `Any` downcast through a call) and its
stalls. A write prefetch of the slot four submissions ahead
(probe/slot-prefetch, jobs 758-761): level (old against old moves the
servil cells up to 23%; SHA-256 in the same runs within 1%).

**A ceiling near one 16 KiB task a microsecond** (September 28, night,
Mac, jobs 572-587, open): two programs through their own queues move 16
KiB messages no faster in all than one alone (solo 0.07 ns/B, shared
0.14 each: about 0.87 million tasks a second), where 4 KiB messages
(gathered 64 to a task) and 64 KiB ones (a task each, at the hashing
threads' capacity) scale. Not the task's size (MEMBERS 16, tasks of
about 600 B instead of 2 KiB: level, probe/members-16) nor the list's
lock (pollers leaving it to pushers: level, probe/push-first). Gathering
messages under 32 KiB into tasks of up to 256 KiB (probe/members-32k):
shared 16 KiB 0.142 -> 0.089 ns/B, solo 0.068 -> 0.072 (+6%, fewer tasks
to spread): a trade for Zooko. The SME2 thread is not it either
(level without it). What it is (probe/submit-16k, jobs 608-611, 625):
`submit` itself, 780 ns per 16 KiB message alone and 2.6 us with two
programs: the task list's lock (acquired in 90-170 ns alone, 630 ns with
two pushers and the pollers; held 230-510 ns a push, while the pusher
copied the 2 KiB task into ring slots other cores last read and
incremented the in-flight count every finishing thread writes). The
count moved outside the lock (taken: two programs 2.0 -> 1.65 us per
message in the probe, the benchmark's shared 16 KiB -4%); pollers pausing
after a failed try_lock helped two programs (2.9 -> 2.0 us in the probe;
in the benchmark, probe/poll-pause, jobs 630-633: shared 16 KiB -9%, solo
16 KiB +3%) and was left out; 600-byte tasks (MEMBERS 16) held the
lock 137 ns instead of 227. Linking a task's members through their
entries instead of carrying them (candidate/member-links, tasks of about
100 bytes; jobs 626-629): 16 KiB messages 6% faster, batches of 4096 3%,
64-byte messages 25-33% slower (a line per member entry, 64 a task,
against the carried array's 17): left out. Keeping both would take two
task lists, small subtree tasks and gathered tasks with their arrays;
past that, a lock-free list.

**The SME2 thread and gathered tasks** (September 28, night, Mac, jobs
588-603). With no SME2 thread (probe/no-sme2-thread, NEON workers
hashing every task): 64-byte messages 15-20% faster, 16 KiB 10-15%,
batches of 16 about 10% and of 65536 7%, 1 KiB messages 15-20% slower
(or noise: 0.12-0.19 across runs), the rest level. Taken: the SME2
thread runs gathered tasks (short messages, small batches) on NEON
(64-byte messages 14-15% faster solo and shared, batches of 16 and 64
10-15%, the rest level; VM no regression). The SME2 thread running every
task on NEON (probe/sme2-thread-all-neon, jobs 604-607): level on 16 KiB
to 64 MiB messages and batches of 4096 and 65536, so subtree tasks keep
SME2, at the same speed for less energy per byte.

**The 64-byte continuous cell's two speeds** (job 680, a full run with
`--trace-clocks`): all twelve solo samples on P-cores; the benchmark
thread's clock 2.6-3.2 GHz against the core's 4.4 (other busy cores of
its cluster lower it); 0.74 ns/B at 3.06 GHz, 0.78-1.14 at 2.61. Neither
core kind nor clock explains the spread; open. Job 705 (after the
delivery thread's later close): servil 0.83-1.21 ns/B, SHA-256 in the same
run 0.84-1.07, both at 2.3-3.0 GHz (the machine's clock through a full
run, the same for both): at the clock they share, the two are close.

**The queue's ceiling for 64 KiB messages** (probe/submit-64k, job 714;
open): with the benchmark's 1 MiB in flight (16 buffers), a program
spends 3.9 us per 64 KiB message: 1.4 us in `submit`, 1.7 us waiting for
a buffer to come back, the rest copying. So the engine is the limit,
about 16 GB/s, where hash_multithreaded reaches 45: a round trip near 64
us for tasks that hash in 10 (SME2) to 30 us (a P-core's NEON). E-cores
holding up in-order delivery would explain it (80 us a task), but the
pool's threads at user-interactive QoS measured level
(probe/workers-qos, jobs 715-718). 1 MiB messages alike (10 us in
submit, 32 waiting, 57 in all). Timed by stage (probe/task-times, job
719, 64 KiB messages): NEON workers took 88% of the tasks at 33 us of
hashing each, the SME2 thread 12% at 20 us; a task waited 4-6 us in the
list; a message's round trip averaged 51 us. Tasks of about 100 bytes,
their members in a queue's fixed set of member blocks
(candidate/member-blocks, jobs 720-723): 2% faster overall, mixed by
cell (batches of 4096 7% slower solo, shared 16 KiB 11%), 64 KiB level:
the task's size is not this ceiling; left out. Tasks of 32 KiB
(probe/task-32k, jobs 724-727): large messages 10% faster solo, batches
of 1024 18% slower (two tasks now); subtrees of 32 KiB with batch tasks
kept at 64 KiB (probe/task-32k-subtrees, jobs 728-733, three runs a
side): 256 KiB-64 MiB messages 4-14% faster solo, 64 KiB 5% slower,
shared level: a trade, left out.

**Idle workers sleep** (September 28, night): a worker, and the SME2
thread, that finds nothing for 50 us (`lanes::WORKER_IDLE`, about a
wake's cost) sleeps even while a job or a queue holds the pool; pushes
and jobs wake as many as they want awake. Before, every worker a stream
woke polled until the stream drained (the queue's first burst of 1024
64-byte messages pushes about 16 tasks and woke all fifteen). VM probe,
a million 64-byte messages: user CPU 0.77 -> 0.67-0.72 s at the same
speed; after hash_multithreaded 8 MiB the process spent 145 us of CPU
in the next 200 ms, against 748; Mac continuous cells level or better
(jobs 548-551); perf_regress on the VM: no regression.

**Wakes as a tree** (probe/wake-tree, jobs 634-637): the caller waking two
sleepers itself and each woken worker up to three of those owed, instead
of one and then all the rest: level from 1 to 8 MiB and on batches of
16384 and 65536 after the gap; left out.

**Hot words on lines of their own** (September 28, night, Mac): the task
list's lock, its `queued` count, and the in-flight count apart (jobs
655-664: 16 KiB messages 14% faster solo, 9% shared, 64 B level), the
queue's submitter lock, returned-slots lock, delivery lock, and `active`
apart (jobs 665-670: 256 B 5% solo and 11% shared, 1 KiB 6%, 16 KiB 3%),
free slots taken oldest-returned first (jobs 651-654: 64 B 9%, 256 B
5-10%): taken. The pool's slots, `callers`, `linger_until`, `registered`,
and `sleepers` apart (probe/pool-lines, jobs 671-676): level, left out.

**WORKER_IDLE's length** (jobs 682-693): 200 us level with 50; 15 us
breaks lingering (workers without a piece sleep between updates: streams
of 8 MiB 0.115 -> 0.27 ns/B) and read the continuous cells 5-10% faster,
a gain that vanished when lingering was exempt from it (probe/idle-15-
linger: level): likely the lingering stream's aftereffect on the cells
after it in each round (eight cores hashing NEON at high power, then
polling), not the queue's own. Left at 50 us; lingering's effect on its
neighbours is one more cost for Zooko's bound.

**The caller's own pieces on SME2** (September 28, night): run_job's
caller hashing its later pieces on its own platform (SME2 under the
turn) instead of NEON: level on the Mac from 1 to 128 MiB (jobs
529-532); left out. **The split at 512 KiB** (`probe/split-512`): Mac
after the gap 512 KiB 0.35 -> 0.19-0.24 ns/B and 8192 messages 22.5 ->
13.3 ns/msg (jobs 538-541), VM 512 KiB 20-30% slower; taken (Zooko, September 28). **4 KiB pieces** (`MIN_PIECE_LEN`,
`probe/piece-4k`, jobs 552-559): streams through a lingering Hasher
25-70% slower (a 64 KiB job's cut ends in 4 KiB pieces), one message
level once lingering kept eight workers; left at 8 KiB.
- Storage is recycled, io_uring style (Zooko: no malloc per submission):
  slots in blocks that never move, results keeping their capacity, lists
  growing only to the program's in-flight high-water mark;
  `tests/queue_no_alloc.rs` counts every allocation (none after warm-up).
  The pool's task list is shared by every queue: each makes room in it
  for its slots times the most tasks a submission has had, and gives the
  room back when dropped (September 28, night). The room "every task in
  flight" it had before counted tasks finished but not yet counted down,
  so the list grew after warm-up once in 60-100 runs of the test (half
  the time under TSan); now 0 in 150, and 0 in 8 under TSan.
- Measured (Mac): see bench-hashes NEXT-STEPS, "Resume here". Per-message
  delivery is serial by contract, so a program's own per-message costs
  bound `Queue::messages` for tiny messages; `Queue::fixed` is the API
  for them.

**Lingering between multithreaded updates** (September 28, night; Zooko's
decision of that day in docs/api-design.md, the bound his open question).
`Hasher::update_multithreaded` past a message's first 128 KiB
(`LINGER_AFTER`), for an update of 64 KiB or more, sets the pool's
`linger_until` 50 us ahead (`lanes::LINGER`) and wakes sleepers without
waiting; workers poll while a job is registered or the deadline is
ahead. The next update hashes its whole subtrees of 64 KiB and more over
the pool (`subtree_children` from `LINGER_SPLIT_LEN`; eight NEON pieces
of 8 KiB, the caller's SME2 share zero at that size) when the eight
workers it wants are awake, else on its own thread. Mac, 64 KiB pieces
after the gap, old -> new (jobs 508-515): 128 MiB 0.24 -> 0.098 ns/B, 32
MiB 0.25 -> 0.108, 4 MiB 0.27 -> 0.146, 1 MiB 0.26 -> 0.19; shared 128
MiB 0.26 -> 0.11; 64-512 KiB level within the after-gap noise (jobs
520-523). Using the pool as soon as the deadline was set, before the
woken workers arrived, left the caller alone on NEON (slower than its
SME2): 256 KiB 10% slower. The bound's reasoning: waiting as long as a
wake costs before sleeping spends at most twice what knowing the future
would; a program that stops updating leaves workers polling at most
50 us.

Its energy (probe/energy-new against probe/energy-old, jobs 560-567;
`proc_pid_rusage` V6 `ri_energy_nj`, the kernel's estimate; Mac, 32 MiB
in 64 KiB pieces with a copy per piece): `update` 450-480 pJ/B at 0.28-
0.30 ns/B; `update_multithreaded` before lingering 405 pJ/B at 0.33,
lingering 1.7-2.6 nJ/B at 0.13-0.23 ns/B: 4-6x the energy for 1.5-2.6x
the speed. About eight cores stay busy (CPU time 1.1-1.8 ns per byte):
each 64 KiB job wakes the workers its eight pieces want, and every woken
worker polls between updates, hashing on NEON (more energy per byte than
SME2) when it has a piece. Four workers kept ready instead of eight:
level. A lingering update's 64 KiB job on four threads instead of eight
(probe/linger-4 against probe/energy-final, jobs 617-624): 18% less
energy (1.55-1.60 against 1.89-1.98 nJ/B) and half the CPU time; in the
benchmark long streams 7-12% slower solo (128 MiB 0.102 -> 0.114 ns/B)
and 18-27% shared, 1 MiB 20% faster solo and 256 KiB 20% shared: a
time-for-energy choice left to Zooko with the bound. The same probe, the queue: 1 KiB messages 5.5 -> 4.1-4.8 nJ/B at
4x the speed; 64-byte messages 450-470 -> 520-580 nJ per message at 1.4x
the speed (more threads poll: CPU per message 165 -> 290-325 ns).

**The feed design, superseded** (candidate/queue-speed, September 28): a
64-slot ring per queue beside a pool job, an engine thread doing intake,
helping, and delivery. Lessons kept: the engine as both hasher and
deliverer delayed deliveries (help-any) or starved short streams
(help-front); helping after a wait, waking after a burst, 16 KiB tasks,
and an SME2 turn per worker task all lost (probe/queue-timeline, jobs
422-449). Pieces waiting in the engine's pending list while it hashed
were the bubble the rebuilt design removed.

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

**The trades, measured again on the fixed benchmark** (September 28,
morning; bench-hashes 92b21b2, whose continuous cells run in a phase of
their own at full clock; each trade cherry-picked onto c46c57c as
`trade/<name>`, Mac old/new/new/old full runs, new/old medians of the
pooled samples, servil mt; `tmp/ab.py`). The calls after the gap move
by up to 1.9x either way between identical code (their clock states), so
only the continuous cells and the long streams read here:
- `trade/members-32k` (messages under 32 KiB gathered, a gathered task
  up to 256 KiB; jobs 743-746): continuous 16 KiB messages 26% slower
  solo, 30% faster shared; the other continuous cells level (geometric
  means solo 1.01, shared 0.98).
- `trade/subtrees-32k` (subtree tasks of 32 KiB, batch tasks of 64 KiB;
  jobs 747-750): continuous messages 3% faster (geometric mean, solo and
  shared), 256 KiB-64 MiB 6-7% faster solo; continuous batches of 64
  messages 11% slower solo, of 4096 14% slower shared.
- `trade/linger-4` (a lingering update's job on four threads, 18% less
  energy in probe/linger-4; jobs 751-754): streams of 4-128 MiB in 64
  KiB pieces 2-7% slower solo and 15-41% slower shared, 1-2 MiB 13-26%
  faster; continuous cells level.

## Rejected (with the reason; do not retry without new evidence)

- **A woken worker prefetching a piece's kernels** (probe/wake-prefetch,
  October 2, 2026, jobs 1032-1035): after its sleep, before its first
  piece, PRFM over k10, k6, and the parents. hash_multithreaded after
  other work and after idling, 256 KiB-8 MiB, batches of 8192-65536:
  level within each side's own spread (512 KiB and 1 MiB move as much
  old against old). The wake's own 15-70 us dwarfs the code's fetch.

- **The task list back to its ring's start when it empties**
  (probe/task-ring-clear, October 2, 2026, jobs 1009-1012). The list's
  room follows every slot (1024 short messages in flight: about 1040
  tasks of 2.1 KiB, 2.2 MiB), and a VecDeque's head walks all of it;
  `clear()` on the pop that empties it keeps the live window at the
  start. Solo level (64 B-64 KiB messages, batches of 16-4096, within
  each side's own 5-26% spread); shared 64 B and 256 B messages twice as
  slow (0.637|1.127 -> 1.292|2.304 ns/B): two queues' pushes and the
  workers' pops then meet on the same few lines, where the walking head
  spreads them.

- **p4 as one NEON quad** (probe/p4-quad, September 27, 2026): four
  64-byte messages 14% slower on the VM than two pairs.
- **p4 as two scalar blocks beside a pair: taken after all** (642757f,
  reverted in 9850a21, restored by Zooko's decision, September 27, 2026:
  hash_many is built for top speed, no alternative kinder to E-cores has
  been found, it was the last batch where BLAKE3 official led, and P-cores
  do most of the work): four 64-byte messages 13-14% faster on P-cores and
  the VM (Mac 23.2 -> 20.3 ns/msg, official 21.6), 30% more cycles on
  E-cores (720 against 555 per call; 347 against 392 on P; jobs 396-399).
  Other parent plans keep the rule: two scalar lanes stay out.
  **Measuring E-cores on this Mac**: in High Power mode threads at
  background QoS stay on P-cores; keep 14 threads spinning at
  user-interactive QoS and the background thread spends about a fifth of
  its time on E; per-kind cycles per call = that kind's cycles per
  instruction x instructions per call (probe/p4-base-ecore2).
- **Wake fan-out as a tree** (probe/wake-half, September 27, 2026): each
  woken worker wakes the larger half of the owed sleepers instead of the
  first woken waking them all. Level on both machines (Mac jobs 369-372:
  solo mt 1 MiB 0.066 against 0.067 ns/B, 8 MiB 0.027 against 0.028; VM
  within noise). The simpler scheme stays.
- **All woken workers draining the owed wakes one at a time** (September
  27, VM): every early worker spent its time waking; the first piece taken
  at about 120 µs into a 1 MiB call.

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
  Measured again after the gap (September 28, night, probe/plans-pairs,
  jobs 568-571; 4 KiB as [2, 2], 8 KiB as [4, 4]): 4 KiB 20% slower at
  the fast speed and 27-45% faster at the slow one; 8 KiB slower at both
  (fast 1.13 -> 1.35 ns/B, slow 3.4 -> 4.4). No change.
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

**Clocks tick at 24 MHz** (41.67 ns) on the M4 Max, for every Darwin
wall clock and the CPU-time clocks alike (measure-clocks3, results of
December 7, 2025: timings of a 1.3 us call read 1,208, 1,250, 1,292,
... ns), and in the VM (`arch_timer` at 24 MHz). A call of a few ticks
is timed either as a batch or, where each call must come after a gap, as
a sum of single readings (`clocks::measure_after_gaps`): each reading
starts at a phase nothing correlates with the call, so the sum's rounding
averages out; a median or minimum of single readings would keep it.
After a 1 ms gap the core's clock is anywhere from its lowest to its
highest, so an after-gap sample's spread is the clock states', far above
the ticks'. First measurement (VM, quick run, September 28): after the
gap a 64 B call costs about 500 ns for every hash alike, against about
40 ns back to back; unexplained (the vCPU's wake, cold caches, the
clock?), to be measured on the Mac.

**perf_regress as it is (October 1, 2026, late).** It builds the two sides
and runs `bench-hashes regress`: alternating pairs, every run over all 14
nonstop points, each pair's verdict on a cell the fast speeds' ratio
(clocks::speeds::compare), margins 3% solo and 10% shared, the control,
no verdict on busy or unobserved load, four confirming pairs (Zooko: one
statistic for every comparison). Calibration, a slowdown planted in
`hash()` (probe/plant-32, probe/plant-25; lent 64 B servil st):

| | VM | Mac (jobs 935-952, mains) |
|---|---|---|
| identical code | 8 of 8 no hold; 2 listed a cell (a shared +20%, two faster) | 7 of 7 no hold, nothing listed (1 busy) |
| +9.5% (Mac +9.4-10.2%) | 5 of 5 held, the right cell | 3 of 3 held (2 busy) |
| +3.8% (Mac +4.0-4.6%) | 0 of 5: the control (sha256 lent 64 B) moved +8.3-8.9% every time | 3 of 4 held (1 busy) |

On the VM the planted code moved SHA-256's code, and with it a small
nonstop cell, by about 8%: the control turns that into no verdict, never
a false hold. Decided (Zooko, October 1, 2026): such a VM no-verdict is
answered by the Mac's verdict; the VM's listed lines on identical code
(each side built from its own path, laid out apart) are left as they are.

**Busy windows on the Mac with nothing running** (October 1, 2026): 4 of
18 calibration checks (jobs 935-952) met a window where other programs kept
1.1-3.0 CPUs busy, with every app closed but Terminal; the check gave no
verdict each time, as it should. A `top` record over four more checks (jobs
955-958, none busy) saw macOS's own work: XprotectService about a quarter
of a CPU just after each build (it scans a newly built program), pi's node
process about 13% on average, Spotlight, WindowServer, short bursts of
others, the VM 0.5%. Expect about one Mac check in five to give no verdict
this way, and run it again (Zooko, October 1, 2026: not ours to chase; it
slows no user's hashing, and the tools catch it). Later the same night
five records in six were busy; a `top` over record job 967 found the
watching itself: with Activity Monitor open, sysmond takes about 46% of a
CPU every 2 s, and with pi's node (11-28%) and Wi-Fi's airportd bursts a
second crossed one CPU. With Activity Monitor closed, job 968 was quiet
(0.25 CPUs, 0.66 at most). Keep Activity Monitor closed while the Mac
measures.

**perf_regress before it** (`check` = working tree against HEAD, `compare OLD NEW`,
`build` = the working tree's bench-hashes for runs by hand): runs A B B
A A B B A of sha256 (the control), servil, and servil mt at 14 points
over the benchmark's five nonstop use cases, 24 rounds each; a cell is
slower when all four pairs' 5th percentiles are more than its margin
above: 3% solo, 10% shared; the control moving means no verdict. Since
October 1, 2026 it measures no calls after a gap and has no slow-speed
rule: both gave verdicts on identical code ("perf_regress on the Mac:
layout luck per side"); the paragraphs below about them are history. A
first calibration on these use cases (VM, September 28,
night): four checks of unchanged code, no regression called (one first
flag the confirmation dropped; one cell called faster, the continuous
batches of 16, which run at two speeds); a planted slowdown in
`Queue::submit` of +100-300% on the continuous messages caught and
confirmed, one of about 1% not; about 20% planted in `hash()` for 64 B
not confirmed, where the after-gap medians of unchanged code vary about
twofold between VM runs at 64 B (servil st 7.5 and 14.1 ns/B). The Mac
awaits the runner's restart (its installed perf_regress predates this). Confirmed solo
cells hold the change (exit 1); confirmed shared cells are listed beside
exit 0, and the commit message names them and the reason (since
September 26, 2026). Each listed cell also shows its 90th-percentile
ratio, which the verdict ignores (a two-speed cell's 5th percentile sees
only the fast speed). It measures only its points; a kernel no point
exercises can change unseen (4 KiB was added for that). `check --against
<last release>` before a release.

Since September 26, 2026 a check takes about 17 s on the VM when neither
side's code changed (95 s before), for three reasons:
- *Curtailment.* A pair's runs measure only the points with a cell still
  open (every pair so far beyond its margin one way); the first pair
  measures all 29, then typically 5-12, then 0-3. The verdict is the one
  every pair measuring every point would give from the same pairs.
  Confirmation (four more pairs) measures only the slower cells' points.
- *Shorter runs.* The variance between processes exceeds a run's
  sampling noise: a cell's 5th percentile across runs varies 1.4% at 24
  samples, 1.5% at 12, 1.8% at 6 (VM, median cell), so rounds went 48 ->
  24. Calibrated on unchanged code (VM, checks simulated over consecutive
  runs): false flags before confirmation 0.07% of cells, no false
  no-verdict in 25 checks; a solo cell 5% slower caught 81% (84% at 48
  rounds, 61% at 12), 10% 92%, 20% 95%. The misses are the cells whose
  speed differs between processes (servil mt at 64 KiB and 1024
  messages, servil st at 32 KiB). The Mac (job 329: 48 runs from one
  build, checks over consecutive runs): false flags 1 of 4756 cells, no
  false no-verdict in 41 checks, 5% slower caught 77%, 10% 96%, 20% 99%;
  open points after each pair 8.4, 1.7, 0.3, 0.
- *Cargo's freshness.* Each side is a directory it owns (fork worktree,
  bench-hashes copy with its own lock, target directory), changed only
  where its sources differ, so an unchanged side builds nothing (1.2 s
  of git and copying). Two bugs had made every build rebuild
  bench-hashes: the committed lock patched and written back on every
  check, and bench-hashes' build.rs watching `<git dir>/refs/tags`,
  which a worktree's git directory lacks (a missing watched path is
  always stale; tags live in the common directory).
A synthetic slowdown (+33% at 64 B) was held on 64 B and one-message
batches, confirmed over 6 then 4 points, in about 35 s.

After idle (since September 26, 2026): bench-hashes' third scenario, calls
after the thread slept 1 ms, judged at a 20% margin and holding a change
like solo cells (a program hashing now and then is the recommended usage
too). VM calibration (7 checks over consecutive runs): no false flag or
no-verdict; 50% slower caught 88%, 70% 99%, twice as slow always; Mac
(job 338, 41 checks): the same at 20% (50% slower caught 82%, twice as
slow 99%), after-idle 5th percentiles varying 9% between runs there. A
planted 50 us spin on waking the pool's sleepers was held on servil mt
after idle at 256 KiB (+58%, then +43%) and 4096 messages (+50%, +45%),
no solo cell moved. The after-idle cells add about 2.5 s a run.

Found while calibrating (September 26, 2026), both open:
- *The VM warms up.* Under sustained load the guest slows over its first
  3-4 minutes, then holds: SHA-256 +5-11%, servil mt about +6% (settled
  in about 2 minutes), servil st about +2%. Within a 9 s run there is no
  drift (first half against second half 1.00%, odd against even
  rounds 1.03%); the Mac's record shows none over minutes (0.31% against
  0.29%). A B B A with the all-pairs rule cancels monotone drift (it
  pushes alternate pairs opposite ways), so verdicts stay unbiased; a
  record made cold compares SHA-256 with BLAKE3 up to 5-9% differently
  than one made warm.
- *A trivial change moves shared 32-64 KiB by a fifth.* 64 dependent
  `black_box` steps added to `hash()` (40 ns at 32 KiB, 0.7%) made shared
  servil st 32 KiB +24% and 64 KiB +20% in all eight pairs: code layout,
  or a shift in when the two copies take the SME2 lock. Ours to explain.

**bench-hashes depends on the fork by git** at the commit its
`Cargo.lock` pins (what users measure). A build against a local checkout
takes `cargo --config
'patch."https://github.com/johnservil/BLAKE3".blake3-servil.path=".."'`
(bench-hashes nested in that checkout), and the patch changes the lock,
so it happens where the lock belongs to the build: `perf_regress`'s sides
(`tmp/perf-ab/old/`, `new/`: a fork worktree, a copy of bench-hashes
with its own lock derived from the committed one, a target directory;
`perf_regress.py build` for runs by hand), and the Mac runner's
throwaway clones. Until September 26, 2026 perf_regress patched the
committed lock and wrote it back, a change and a change back on every
check, which made Cargo rebuild bench-hashes every time. Records of fork commit X: pin X in
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

**Probe timing**: the `clocks/` crate (fork and bench-hashes alike; until
September 26, 2026 `examples/support/clocks.rs`, and bench-hashes' own
bindings) measures wall time and cycles per core kind together, per
batch, and shows the clock they ran at; use it in every probe (AGENTS.md,
"Measuring"). The
record of switching between the two: cycle normalization in the benchmark
until September 2026 (removed: it hid SME2 waits), cycles for the E-core
kernel probes (wall time swings 2x with the E clock), wall time for the
SME2 remainder probes, where cycles per ns then exposed the slow state.

**Probes on the Mac**: the runner runs only allow-listed examples, so a
probe adds its own `examples/host_lab.rs` on a `probe/<topic>` branch (never
merged; examples only, so the hook skips it) and runs as a `host_lab`
job. Measure with `thread_selfcounts` cycles per perf level at
user-interactive and background QoS; A/B as old / new / new / old jobs,
each probe on its own base. Kept probes: `probe/ecore-kernels`,
`probe/ecore-trigger`, `probe/transition`, `probe/sme2-gap`; bench-hashes
`probe/fork-no-sme2` builds the fork with `no_sme2`.

**Pauses slow the core's clock** (September 27, 2026, jobs 351 on
battery and 357 on mains, M4 Max, `powermode 2`): hash(64 KiB) back to
back 13.5 µs on P at 3.2 GHz; with 1 ms of sleep before each call
39 µs, the P-core at about 1.2 GHz (battery 41 µs); back to back again
afterwards the clock stays low (mains 29 µs at 1.5 GHz on P; battery
20 µs on E). Battery power moves more of the calls after a pause onto
E-cores (233 of 400 against 32) and changes the times little; the
clock's fall after a pause happens on either. So the after-idle cells
measure the machine's clock ramp as much as our wakes, and records made
after a stretch of pauses read slow. A 200 ms spin ran on P at 4.4 GHz
at background QoS as at user-interactive (job 350, battery). Since then
bench-hashes records the power state (report, samples, graph; the
graph's header names a run on battery or in Low Power Mode),
perf_regress prints it beside its verdict, and the runner writes it into
every `verdict.json`. Open, the machine's but ours to predict and tell
users: the clock's fall after a pause (how long a pause, how long the
recovery) and what it costs st and mt. The per-core-kind split
of `clocks::Counts` over calls of 20-400 µs is unchecked (it read
"100% E" at 3.5 GHz, beyond an E-core's clock).

**The clock after a pause, measured** (September 27, 2026, jobs 358 and
359 alike, `probe/clock-pauses`, M4 Max on mains, user-interactive QoS;
each trial 20 ms of back-to-back calls, a pause, then calls one by one;
median of 15). hash(64 KiB), warm 13.5 µs at 3.2 GHz:
- Sleeps of 20-200 µs cost nothing.
- Sleeps of 500 µs to 20 ms: the first two calls run at about 1.06 GHz
  (41-42 µs each), then the clock returns by the third to fifth call:
  about 85 µs of work at a third of the speed. Predictable, and the
  machine's: macOS lowers an idle core's clock after about half a
  millisecond and raises it again after about 85 µs of work.
- A sleep of 100 ms: the calls stay at about 1.9 GHz (22 µs) through all
  40 calls.
- A 1 ms spin (`std::hint::spin_loop`) instead of a sleep: the calls
  after it run at 2.5 GHz (17 µs) through all 40. The pool's pollers spin
  this way, so a worker that polled may hash at a lower clock; its own
  clock is unmeasured so far (the probe reads the caller's).
The first call after the pause, st against mt (µs): pauses of at most
100 µs, 64 KiB 13.5 against 4.5-5.5, 1 MiB 153 against 31-36; 200 µs
(the workers just asleep, the caller still fast), 64 KiB 13.5 against
25-30, mt's worst ratio and entirely ours; 1-5 ms, 64 KiB 41 against
48-76, 1 MiB 213 against 130-145; 20-100 ms, 64 KiB 41-43 against
100-117, 1 MiB 234-542 against 146-514. The pool pays from 1 MiB after
any pause and never at 64 KiB once its workers sleep.

**The minimax list**: `pypy3 tools/losses.py <samples.tsv>`; compares
medians (servil against single-threaded contenders, servil mt against
all); marks two-speed cells.

## Testing

**The suites run in debug builds too, on CI's targets** (October 2,
2026): the fork's CI (upstream's ci.yml, every branch since 69d90b9)
runs them in debug on Linux x86-64 and arm64, macOS, Windows, wasm32,
and cross targets, where overflow checks and slow, loaded runners find
what release runs on the VM do not: a test's wrapping length overflowed
(every debug suite failed), and `tests/queue_no_alloc.rs` counted two
allocations after warm-up on slow builds. Those were libtest's own: its
notice that a test has run over 60 seconds allocates on the main thread
(found with a backtrace per counted allocation, the VM's pure debug
build on four loaded CPUs: 3 of 3 failed before, passes after). The test
now runs without the harness. A queue held twice for a moment (a
submitter handing it over again while the delivery thread lets it go)
could also have grown the delivery thread's lists once; their room is now
two per queue alive (ad20a52), which closes that case, though it was not
this failure's cause.

    cargo test --release --lib                      # 86 tests
    cargo test --release --features no_sme2 --lib   # 82
    cargo test --release --features pure --lib      # 71
    cargo test --release --doc                      # 21
    cargo test --release --test api_plan            # 12, the planned API's contract
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

## The startup self-test (September 26, 2026)

`src/self_test.rs`: 39 chained cases before a process's first hash
(through Platform::detect, a relaxed load on the fast path; std only, not
under Miri). Zooko asked for about 100 µs; measured, every AArch64
assembly entry (31: hybrid k1-k10, q1-q9, p2-p5, p7-p9, c1; SME2 chunks,
messages, parents, sme2x2) costs warm 93 µs (Mac) / 95 (VM) and in a
fresh process 140-200 / 125: the hybrids are 361 KB of unrolled code and
the SME2 kernels 32 KB, and fetching them cold is most of the difference.
Zooko chose this budget over leaving kernels out (option 4). Rejected
for now: direct one-block kernel calls (warm about 40 µs, cold floor
unchanged), per-kernel checks on first use.

What reaches what (gdb, `tools/self_test_coverage.py`): k_n only from
whole inputs of n chunks (rising counters; 11-16 chunks run k8-k10 with
k3-k6); batches of one-chunk messages, all at counter 0, take the parent
plans; q_n from n whole chunks and a partial one of two blocks or more
(n > 9 leads with k(n-9), q9 after; one chunk takes q1 only past four
blocks); sme2x2 only in the flat walk from 256 KiB, so the self-test
calls it once directly on one group of 18 chunks, keyed, against the
portable kernel's values. Chaining carries the first 32 output bytes into
the start of the next input (every message's, in a batch); carrying the
last digest of a batch missed a fault (the unit test caught it).

Measurement pitfalls: gdb in the guest needs `SHELL=/bin/sh` and `set
startup-with-shell off` ($SHELL names a zsh the guest lacks); the
fresh-process time needs a process per sample (probe/self-test-time,
job 307).

## Future work

Every open item of the fork and the benchmark, in one list; the older
blocks of bench-hashes' NEXT-STEPS.md hold each item's evidence. Take an
item out when it lands or is rejected.

### Speed

- **The queue's long messages** (October 5, 2026): from 1 MiB the queue
  is slower than waiting for each call (64 MiB: 0.058 ns/B against
  hash_multithreaded's 0.046, Mac job 1213), where the stream prototype
  reached 0.026: larger tasks for long messages (fewer handovers, the
  SME2 flat walk), or a reader thread's double buffering, are the first
  things to try.
- **b3sum's read paths on native Linux** (needs a Linux computer, not a VM
  guest; Zooko, October 3, 2026): repeat measurement 1 of
  docs/api-design.md (the probe/b3sum-* branches, `bench-hashes b3sum`),
  and add io_uring to both today's double-buffered reader and the stream.
  The stream stays until then (on the Mac, jobs 1197-1198, today's reader
  thread beat it). If today's double buffering still wins on both, move it
  into the crate so other programs get it without writing it:
  `Hasher::update_reader`, which reads a reader on one thread today, is
  its natural home.
- The queue redesigned (October 3, 2026, under discussion with Zooko):
  the calling thread hashes what no worker has taken when it needs room,
  workers woken by backlog and sleeping when none waits, no delivery
  thread; possibly one ring of input bytes the queue owns, read into in
  place. Removing all lingering (candidate/no-linger 0ba5a45) slowed the
  current queue's cells (Mac jobs 1186-1189: solo 16 KiB x2.7, 64 B x1.4,
  batches of 64 x1.3; shared x1.3-2.1) because the calling thread pays a
  wake per push; b3sum level, less CPU time.
- In that design, test the wake threshold (wake a worker when half the
  ring waits) against waking on every push into an empty ring, and
  remove the threshold if they measure level (Zooko, October 3, 2026).
- Once the BLAKE3-owned buffer works and is measured: measure serving
  data already in the program's own buffers by a copy into it, and
  revisit whether that one design can replace the owned-buffers queue
  (simpler, perhaps a little slower; Zooko, October 3, 2026).
- Benchmark many short messages of different lengths, one after another
  (the benchmark's message cells repeat one length): a use case the
  BLAKE3-owned buffer must serve or steer elsewhere (Zooko, October 3,
  2026, "make sure we have benchmarks of this use case").
- The queue's cells stable enough for the regression check: on identical
  code their per-process means move 6-60% ("The regression check,
  calibrated"); the streaming APIs come first, so this is first.
- Try putting b3sum on io_uring on Linux.
- `hash_range(file, offset, len)`: BLAKE3 doing the reads, for files and
  sockets (api-design.md's third concurrency model).
- The queue: hand the first message into an empty queue over at once
  (today it waits about 1 us for company, then a wake of 15-45 us).
- The queue: delivery on the worker that completes the oldest entry (one
  handover fewer, no delivery thread polling while entries are in flight).
- A lock-free task list: shared 16 KiB messages meet a ceiling at its lock
  ("A ceiling near one 16 KiB task").
- A second SME2 thread in the pool (two SME units reachable, job 187).
- Shared lent batches of 64-256: 1.6-1.7x a batch of 16 per message (the
  SME unit shared).
- A p4 kinder to E-cores; the E-core cells (2-chunk messages at 4, 1000 B
  x 4; tails of 1-4 multi-block messages past SME2 groups); 12 messages
  shared behind BLAKE3 official (25.8 against 21.7 ns/msg).
- 2-3 KiB against SHA-256: servil ties ring at 4 KiB warm and leads it at
  16 KiB after other work; below 4 KiB SHA-256 leads. Ideas estimated:
  parents and root inside k4 (about 3.6%), a direct small-tree path
  (1-2%), a faster pair chain.
- SME2 batches from 16 messages when calls have work between them (about
  12 ns/msg there against the benchmark's 10).
- A GPU kernel (Metal) for large inputs; the VM has no GPU.

### Interfaces

- Revisit a possible energy-efficiency option (removed October 2, 2026:
  its complexity outweighed its likely use; the September 25
  measurements, "Energy per byte" above, are where to start).
- update_rayon on the fork's pool, one mechanism for multithreading
  (changes its contract: today it runs on the caller's Rayon pool).
- A Merkle tree API (`servil::merkle`, for users like Remco's WHIR);
  write its trade-offs up for Zooko before building.

### For Zooko

- Lingering's 50 us bound: about eight cores poll between updates for
  2.4x the speed of `update` (long messages in 64 KiB pieces).
- The three trades: members-32k, subtrees-32k, linger-4.
- Whether the docs keep the promise that hash_multithreaded runs no
  slower than hash (untested since the benchmark's servil-only checks
  left).

### Slowdowns to explain (open, AGENTS: "we own every slowdown")

- The queue's shares swing per run (256 B messages, batches of 16, 1 KiB):
  set at process start, perhaps thread placement.
- The queue's cells slow the next cell (about 2% on the VM); a lingering
  stream slows the cells after it 5-10%.
- The run-order effect (back-to-back short runs).
- servil mt's shared queue 64 B and 64 KiB cells read apart on identical
  code in the VM.
- The first job of a series ran 5-7% fast on the cores (confirm with
  cycles).
- servil's lent 64-256 KiB cells pay about 4x ring's read copy.
- VM: ring's nonstop batches 30% slower solo in one process of four;
  SHA-256 slowed beside a second copy at 64-256 B.
- Which part of the cycles-to-wall-time ratio is ours (SME2 waits, our
  power draw) and which the machine's (heat, a host, a scheduler).
- NEON going cold after stretches without vector work; what else enters
  the SME unit's slow state.
- The E-core trigger's mechanism.
- The Mac's serial 128 MiB rise.

### Tooling

- P/E classification of every Mac sample; perf_regress P against P.
- Host load in the VM (no steal time): a reference loop timed beside the
  samples would show it.
- `tools/promote.py` (check the gate, write the note, fast-forward) and a
  pre-push hook refusing a `servil` tip without both verdicts.

### Docs and others' work

- A fresh reading as the three readers of README, METHODOLOGY, the graph,
  the guide, and CONTRIBUTING.
- Devon's BLAKE3#1 (x86 two-chunk batching), once he trims it; judge it
  by the frozen benchmark.

## The optimisation pass behind bench-hashes 0.15.0 (October 5, 2026)

Zooko froze the benchmark (0.15.0) so that the maps before and after the
pass come from one benchmark; the API the pass works behind went in first,
each call in its simplest form (`Verifier`, `update_each`,
`finalize_each`, `hash_each_multithreaded_with`). Mac numbers are A/Bs of
the benchmark's cells, old new new old; bench-hashes NEXT-STEPS has the
round-by-round report.

- **The Hasher's stage** (Zooko's choice, the Hasher's job): `Hasher` is a
  `HasherCore` (the tree's state) and a 16 KiB stage. An update of 16 KiB
  or more with nothing staged, and a fresh hasher's first chunk, go to the
  core; shorter ones gather until the message's next 16 KiB boundary.
  Many messages at once 0.519 -> 0.306 ns/B (jobs 1246-1249). Cost: a
  whole message in one update, 64 B +7% (4 ns), 1.5-15 KiB +4-5% (the
  copy; probe/one-update, jobs 1242-1245); the docs say so.
- **The Verifier's batches**: a piece's whole nodes are checked together,
  their groups hashed in one batch. Alone over a 64 MiB encoding (VM):
  64 KiB pieces 0.25 -> 0.237 ns/B, 1 MiB pieces 0.178 (building: 0.174).
  What stays at 64 KiB is SME2's cost per small batch between other work:
  batches of 16 groups cost building itself 0.202 against 64 groups'
  0.174; the NEON hybrids instead cost more (0.263).
- **Lanes across messages without a new kernel loop**: `update_each`
  holds each stage that fills a whole group and hashes the turn's groups
  together; parents carry no counter, so groups of different messages
  share each level's call. Then one SME2 entry for all the groups: the
  chunk kernel's entry `hash16_chunks_at` takes a counter per group (flags
  bit 48, the table's cursor in the body's stack frame). 0.306 -> 0.286
  -> 0.250 ns/B on the Mac (jobs 1255-1258, 1276-1279).
- **Collections over the pool**: `Work::Each`, ranges of items of about a
  thread's share of the bytes through `hash_each_with`'s code; items of
  512 KiB or more through `hash_multithreaded`. Mac: 0.229 -> 0.029 ns/B
  (git objects), 0.218 -> 0.039 (Nix files). Through `&dyn Fn` the lone
  items' call cost servil st's shared cells 4-7%; generic, level.
- **Queue::messages shares tasks below 64 KiB**: a 16 KiB message was a
  task of its own (the pieces' 16 KiB threshold); now messages share the
  batches' threshold, a task's bytes. Pipelined 16 KiB 0.148 -> 0.111.
- **Tried and dropped**: a single 16 KiB group on the NEON hybrids instead
  of SME2 (no change: `Sme2Turn::take(_, false)` keeps the platform); a
  10 s wait between the gate's builds and its runs (the gate's busy window
  stayed, jobs 1264).
- **Open**: the Mac gate has given no verdict since jobs 1260 (one busy
  half-second window per check, 1-4 CPUs of other programs, the cause
  outside the builds, the VM, and the runner); CI builds with warnings as
  errors and the Mac's test job does not, so an unused import reached
  servil 1352c21 (fixed on the candidate, a9ae1d5).

## Benchmark/API alignment and the memory-working gap (September 30, 2026)

Zooko's September 28 evening table is now encoded in the benchmark's
candidate: continuous synchronous calls on lent whole messages, pieces,
and batches, alongside the queue's owned-buffer cells. After-gap servil
mt pieces call `update`; continuous lent pieces call
`update_multithreaded`. Hashing kernels and the pool are unchanged.

`clocks::measure_after_gaps_prepared` owns the measurement order: sweep
all of a caller-owned buffer at 64-byte intervals, spend any remaining
part of 1 ms on integer arithmetic, write the input, then call. The
caller chooses and keeps the work buffer; bench-hashes uses 128 MiB per
measuring thread, with written physical pages. A complete sweep is the
minimum gap even when it takes longer than 1 ms. Preparation and hashing
have separate wall intervals and per-core-kind counts. Seven clocks
tests pass. The benchmark's trace carries both intervals separately;
Linux supplies wall time and carries the explicit absence of cycle counts.

`perf_regress` includes eight points on the three lent axes, at the
continuous margins. The VM check against HEAD passed (four alternating
pairs, a fifth confirmation pair rejected the remaining noise finding,
`tmp/benchmark-alignment/perf-check-2.log`). Its first attempt exposed
an existing graph assumption: a narrowed run selecting only a queue cell
still selected servil st, which has no cell there, and graph provenance
panicked. Benchmark graph eligibility now requires two points per shown
axis and a cell for each selected contender; sparse runs still write
samples and reports. Two dedicated tests hold those cases.

Mac jobs 776-779 failed before measurements: installed perf_regress.py
imports speeds.py, which setup-mac.sh had left out. The setup now installs
that dependency beside it. Zooko can restart the runner in the morning;
tonight's native measurements use a GitHub-sourced host_lab diagnostic
that calls the checkout's tool rather than its installed copy. Candidate
work stays separate from promotion. Bench-hashes NEXT-STEPS holds the
native and historical validation evidence when it arrives.

### The Python rule's fixed-point boundary (September 30)

The historical report check exposed a rare disagreement between the
Python and Rust two-speed rules at a ratio boundary. Python carried
exact Fractions; Rust carried Q64.64 samples and a half-up Q64 midpoint.
The Python twin now uses the same representation and midpoint. Two
explicit boundary vectors extend the shared file (13 total). For
2499/2000, the independently calculated Q64 representation is
23049206720100084744, 3/288230376151711744000 below the exact ratio;
against 1 its permille ratio rounds to 1249, so it is one speed. A value
above that boundary remains two. Seven clocks tests and all 13 Python
vectors agree. The Rust rule and measured samples are unchanged.


## Context-reset audit (September 30, 2026): timing evidence still open

Current focus: Zooko asks whether large benchmark swings occur in users'
typical operation. That is unestablished. He rejected aggregation across
more processes as a substitute for diagnosis. Hold hashing fixed; compare
representative direct callers with the harness and vary work between
calls one factor at a time. Measure wall and per-kind counts through
clocks and use the shared speed/share rule. Cache/TLB/ASLR/code layout
remain hypotheses. No hashing implementation source changed this session.

**Clocks experiment confound found in the handover audit:**
perf_regress.py::clocks_patch patches to ROOT/clocks for every build.
probe/benchmark-alignment descends from 9cea065 (unwarmed helper), so
job 788's requested 1820efb and d005716 sides both used its unwarmed
clocks. This job did not test the d005716 warm-up. Production hashing
provenance alone does not identify that patched helper. The code remains
committed on the candidate; its native effect is open. Preserve the
regression tool's intentional common-measurement basis, while building
actual clocks variants explicitly for any timing-helper A/B.

The benchmark's newest NEXT-STEPS block is the authoritative handover.
It corrects earlier overclaims of per-process causality, bounds on
variation, and archaeology conclusively refuting a regression. Those
checks pass at their thresholds and sampled cells; real-caller behavior
and smaller differences need direct evidence. Native Mac first, VM after;
promotion and publication still await validated evidence.

## Load in clocks, and one reader of samples (September 30, 2026)

Zooko asked for load detection as code every measurement runs, after
the day's Mac jobs ran beside his browser (bench-hashes NOTES, "Load moved
into clocks", has what the old detector had flagged). `clocks::load`
reads machine CPU time (Linux `/proc/stat` with steal; macOS
`host_statistics`, each 32-bit counter differenced on its own) and this
process's CPU time at most once a second, from `tick()`, which every
`clocks::measure*` calls outside its timed intervals; `Batch.started_ns`
places each sample in a window. VM costs: a reading 8.6 us, a tick
between readings 18 ns (`reading_cost`, an ignored test); the Mac's is
unmeasured. `tools/samples.py` reads samples v4 only (AGENTS.md,
"Contracts change everywhere at once"), and perf_regress, ab.py,
losses.py, and bench-hashes' compare-runs.py and check-report.py all use
it. perf_regress exits 2 when a run was busy. The VM check on this tree
passed; both of its sides share the working tree's clocks, so it
measures no effect of the ticks themselves (outside every interval, one
reading a second).

## Cold calls pay for servil's code size (September 30, 2026, jobs 792-798)

After the benchmark's 128 MiB sweep, a call whose code has left the
core's L1 instruction cache fetches it from DRAM. For servil's `hash`
that doubles 4 and 16 KiB (1.6 -> 3.1 us, 4.2 -> 8.7 us in the benchmark;
3.6 and 7.3 us in the probe with the icache invalidated), where SHA-256
ring's small code moves about 7%. Warm code: servil ties ring at 4 KiB
and wins at 16 KiB; cold, it loses both (bench-hashes NOTES, "The cause:
where the hash's code is", has the experiments). The unrolled hybrids
(k1-k10, q1-q9, 361 KB) and the SME2 kernels are the candidates: a
leaner code path for single small messages would serve callers who hash
now and then. Unmeasured: how many bytes of code each length's path
fetches.

## One-shot calls prefetch their code after a pause (October 2, 2026, jobs 971-985)

The remedy for the section above, without making the code smaller:
`hash_serial` (hash, keyed_hash, derive_key, hash_multithreaded below the
split), for inputs of 1-64 KiB, issues PRFM PLDL2KEEP over the code its
path runs, one per 128-byte line, before it hashes. The misses that a
call after other work met one after another (the next line fetched when
the core reached it) then overlap. Which code: `neon_hybrid::
prefetch_tree_code` walks the same plans the hashing takes (chunk
kernels, each parent level's plan, the root's c1), from 16 chunks the
SME2 file's 11 KB (and the hybrids' parents below the flat walk). The
generator puts an end label after each kernel; the SME2 file has one.

It runs only when the thread's previous call in the range came over 100
us before (a thread-local stamp of CNTVCT_EL0, 3.5 ns): unconditional,
it cost nonstop calls 2-4% (lent 4 KiB x1.035-1.046, jobs 972-975),
whose code is near.

Probe (probe/code-prefetch, job 971; each call after the benchmark's
own busy gap, the prefetch inside the timed call), ns/call: 2 KiB 1479
-> 1208, 4 KiB 2552|3521 -> 1740, 8 KiB 4802 -> 2833, 16 KiB 6042 ->
4583, 64 KiB level; cycles alike (8 KiB 21902 -> 13219: stalls gone, the
clock the same). PLDL2KEEP, PLIL1KEEP, and PLIL2KEEP did the same; plain
loads of each line recovered a third as much (each load waits).
Benchmark A/B, f60f9bb against 91c4a77 (jobs 981-984, mains, quiet,
two runs a side), servil st after other work: 2304 B x0.64-0.86, 3839 B
x0.47-0.59, 4 KiB x0.59, 7935 B x0.50, 8 KiB x0.49-0.64, 16 KiB x0.61
(0.282 ns/B, ahead of SHA-256 ring's 0.305 there), 32 KiB x0.98; every
cell from 2304 B to 16 KiB at one speed (several ran at two before);
mt alike; SHA-256 ring level. Nonstop lent: 4 KiB x1.023 (st and mt),
16 KiB x1.057 (st) and x0.969 (mt) on identical code, 1 and 64 KiB
level. perf_regress on the Mac (job 985): no regression. The +2.3% at
lent 4 KiB was the check itself: traced (jobs 1013-1014), 38
instructions and 85 cycles a call more, of which the read of CNTFRQ_EL0
cost about 10 cycles on the Mac (probe/stamp-cost, job 1015). Read once
(a7177c6): lent 4 KiB x0.999 (st) and x0.983 (mt) against 91c4a77,
1-64 KiB level, the gains after other work unchanged (jobs 1019-1022).

**The NEON-only path too** (M1-M3's; probe/neon-only-cold against
-before, built no_sme2 on the Mac, jobs 1123-1127, two runs a side, the
after side first): hash() after other work 2 KiB 1286-1292 -> 1182-1224
ns, 4 KiB 2386-2458 -> 1761-1823, 8 KiB 4172-4281 -> 2688-2740, 16 KiB
8026-8255 -> 5297-5302, 32 KiB 11833-11896 -> 9099-9141; a Hasher per
message alike (16 KiB 8094-8386 -> 5391-5427); nonstop level (4 KiB
1313 -> 1315-1316).

**The short path's layout** (2f46995, f1aafd3, October 2). With the pause
check inlined into hash_serial, a call of 1 KiB or less (which never
prefetches) read slower after other work: four runs a side against
b132f8c (jobs 1049-1056) 64 B and 128 B x1.21-1.31 (SHA-256 ring
x1.00-1.08), about one DRAM miss at 64 B. Out of line (2f46995) the check
cost the short path three instructions, but the compiler made the 1-64
KiB branch the fall-through, so a short call jumped to a line it never
touched before; #[cold] on prefetch_after_pause (f1aafd3) puts the call
after the short path. Four runs a side again (jobs 1057-1064): st 64 B
x1.08 fast / x0.85 slow, 128-512 B x0.76-0.94, 1 KiB one speed between
the old two; mt 64-512 B x0.76-0.92, 1 KiB x1.06; ring x0.71-1.17 in the
same runs; 4 KiB x0.63-0.67 and 8 KiB x0.55 kept; nonstop level. The
lesson: a branch added to a cold path costs it a line unless the short
case stays the fall-through.

**A fresh Hasher's first update too** (7510d44, October 2): a Hasher per
message (and the digest traits) prefetches the kernels of the subtrees
its update loop cuts (prefetch_update_kernels: next_subtree_len from
counter 0, the last chunk left to the chunk state). Mac, probe/hasher-
prefetch against probe/hasher-prefetch-before (jobs 1025-1028, A B B A,
mains), after other work, ns/call: 2 KiB 1500-1521 -> 1333-1344, 4 KiB
2646-2734 -> 1953, 5000 B 3338-3448 -> 2615-2651, 8 KiB 4536-4557 ->
2963-2968, 16 KiB 5552-5562 -> 4417-4484; nonstop level within 1%.
perf_regress on the Mac (job 1029): no regression.

Not prefetched, and so still cold after other work: 1 KiB and below
(c1, 3.9 KB, ran x0.84-1.10 with two speeds), batches (`hash_many`'s
plans: the batch of 4 after other work still costs more per message
than the batch of 2), and the Rust code around the kernels.

What remains, measured (October 2; probe/code-prefetch round two, job
995; probe/glue-prefetch, VM): c1 prefetched at 64 B-1 KiB gains 2-10%
(1 KiB 865 -> 844 ns, mostly a minority slow speed gone), less than its
stamp check costs calls of 50-700 ns: left out. The bound, a whole call
on another buffer after the gap: 4 KiB 1469 ns against the library's
2297, 8 KiB 2240 against 3500; a 64-byte call first recovers about 300
ns of that (c1 and the shared entry code), 2 us of NEON work nothing
(the vector unit is awake: the producer's copy uses it). The rest, about
550 ns at 4 KiB, is the multi-chunk path's Rust code (9 KB around
`hash`, in one place) and frames. Prefetching that 16 KiB of text as
well made every call 400-500 ns slower: prefetches beyond some number in
flight appear to be dropped, the kernels' among them. More prefetch
volume costs; the order and amount matter.

NEON-only builds (M1-M3) lose little as the input leaves the caches
(VM, no_sme2: 1 MiB 0.236 ns/B, 128 MiB 0.256 repeated, 0.255 from DRAM):
the core's prefetchers serve NEON loads, so the flat walk's input
prefetch is SME2's alone.

## The flat walk prefetches the next subtree (October 2, 2026, jobs 990-994)

servil st's one-shot rate fell as the input left the caches (record
968: 8 MiB 0.155 ns/B, 64 MiB 0.171, 128 MiB 0.176; lent 64 MiB 0.190):
the SME unit's loads do not draw the next lines in as the core's do. The
tree walk now carries how many bytes of the caller's input follow each
subtree (`ahead` in compress_subtree_wide and the flat walks; Hasher::
update passes the rest of its input), and each flat walk first issues
PRFM PLDL2KEEP over as many of them as it hashes: the next subtree's
bytes arrive while this one hashes. Nothing past the caller's input
(prefetching the bytes after a lone 1 MiB input cost it 10% in the VM).

Mac A/B, fe50659 against f75e6a6 (jobs 990-993, mains; 992 met a burst
of other load, and the quiet runs alone agree): servil st from 1 to 128
MiB at 0.152-0.156 ns/B; 32 MiB x0.96, 64 MiB x0.91, 128 MiB x0.89;
lent 16 MiB x0.97, 64 MiB x0.90 (shared alike at the fast speed); in-cache
sizes, mt, Hasher in 64 KiB pieces, and SHA-256 ring level.
perf_regress on the Mac (job 994): no regression. Not carried yet: the
pool's SME2 prefix (lanes, `ahead` 0) and the queue's tasks.

## servil 3a8327f against b132f8c, whole (October 2, 2026, jobs 1004-1008)

A full Mac record of the promoted servil (job 1004, `--all`, mains,
quiet), compared cell by cell with job 968 (b132f8c): the one-shot gains
above; every cell slower by more than 5% at its fast speed is a batch,
after other work or after idling (16 messages after other work 28.6 ->
37.3 ns/msg), or the queue's owned batches (1024 x1.11, 16384 x1.21), on
a path no change touched. An A/B of those cells, two runs a side in
mirrored order (jobs 1005-1008): after other work both ways, x0.63 to
x1.28, st and mt (the same code below the split) apart, which is layout
and process luck (NOTES above, "perf_regress on the Mac"); owned batches
x0.96-1.06 and lent x0.99-1.00, each side against itself 5-9% apart
(owned 1024 x1.055 old/old, x1.083 new/new). No regression.

The VM, the same pair (servil b132f8c against 1d2b652, the default
contenders' run, two a side, A B B A, quiet): batches after other work
and after idling mostly faster (x0.55-0.90), the queue's solo batches
x1.11-1.21 slower and its shared x0.73-0.89 faster, each side against
itself up to 22% apart in the same cells (old/old 4096 x1.219, new/new
x1.235): the VM's own spread, no regression shown.

## update_reader through a 1 MiB buffer (October 2, 2026, jobs 1036-1040)

On SME2, 100 ns of other work before a 64 KiB call costs it 2.3 us (VM:
hash(64 KiB) back to back 11.5 us, after 100-1000 ns of integer work
13.9-14.5 us; the Hasher in 64 KiB pieces 13.8 us a piece): the slow
state again, which every read between updates meets. update_reader (and
update_mmap's fallback) updated per 64 KiB read; now a reader that fills
the first 64 KiB goes on into a 1 MiB heap buffer (those 64 KiB at its
start, so every update is a whole, aligned 1 MiB subtree), filled by as
many reads as it takes. A shorter reader keeps the stack buffer and one
update. Mac, files in the page cache (probe/reader-buffer against
probe/reader-buffer-before, A B B A, mains), ns/B: 64 MiB 0.264-0.266 ->
0.178-0.180, 8 MiB 0.265-0.273 -> 0.204-0.207, 1 MiB 0.347-0.368 ->
0.270-0.272, 100 KB level (0.37-0.40). VM, 64 MiB: 0.314 -> 0.233.
As first built (a39fb7c, on servil 9dda4be for about 20 minutes) it
returned a reader's error before hashing the bytes already in its
buffer, which a caller retrying after WouldBlock would have lost from the
digest; 5739af6 hashes them first, and a test with a failing reader and a
retrying caller holds it (it fails on a39fb7c).

## Lent batches of 1024-4096: the SME unit's state, not the copy (October 2)

Lent batches of 1024 and 4096 64-byte messages run at 12.0-12.2 ns/msg
on the Mac (256: 10.0, 16384: 10.6; back to back 9.7). The VM, hash_many
after each of: the producer's copy, integer work as long (no memory), a
read-only pass, nothing (ns/msg, 4096 messages): 12.87, 12.78, 12.83,
9.69. Any other work between calls costs the same: the SME unit's slow
state after work off it (NOTES above, "The slow state, measured
directly"), held for several microseconds of SME2 work (the excess per
batch: 256 about 1 us, 4096 about 8.6, 16384 about 5). Cleaning the
copied lines first (DC CVAU or CVAC) made it slower; tables of 256 or
512 messages per kernel call (TABLE, 128) level. Left as it is.

## servil ff8f203 against b132f8c, whole, and an open lean (October 2, jobs 1084-1096)

The full Mac record of ff8f203 (job 1084; two busy windows of 204)
against job 968: 75 of 138 servil cells within 5%, 32 faster, 31 slower,
the slower ones batches after other work or after idling (layout and
wake luck, answered by A/Bs above) and the queue's shared batches of 16
(x2.07, a share swing: the A/B of jobs 1085-1088 reads it x1.01). Open:
hash_many_multithreaded after other work leans slower, four runs a side
(jobs 1089-1096): 8192 x1.03, 16384 x1.06, 32768 x1.03, 65536 level at
its main speed, each side against itself up to x1.065 apart. No change
of the night touches the pool's batch pieces; a layout or wake-timing
effect is likely and unshown. Bisected, four runs a side each (jobs
1097-1120): b132f8c -> 3a8327f level (16384 x0.96); 3a8327f -> ff8f203
16384 x1.053; 3a8327f -> 5739af6 16384 x1.23 fast / x0.94 slow, the new
side at one speed between the old's two. The same commit 3a8327f ran
16384 at 6.11 (90%) | 7.74 in one session and 5.10 (29%) | 6.67 (71%) in
the next: the cell's speeds and shares move by session as much as the
lean, and no commit shifts it consistently. No regression shown.
Explained (job 1121, traced, four runs): each run's 12 samples spread
continuously, 4.8-9.0 ns/msg, at a steady caller clock (3.6-3.7 GHz, the
streaming clock of its SME2 prefix; all on P-cores): a broad spread, not
two speeds, so the split rule finds one or two by where 12 samples
fall, and a cell's medians and shares move about 10% from session to
session. The caller waits on its workers' wakes, which the spread likely
is. Such a cell needs many more samples than 12, or comparisons within
one session.

## Batch tails, measured again at real gaps (October 2, 2026, job 1001)

Leftovers of 1-4 past whole groups run on NEON and cost far more than
their work (VM, back to back: 32 -> 33 x 64 B 317 -> 782 ns; x 1 KiB
4.9 -> 6.9 us). probe/tail-pad set the threshold from which the last
group is padded (both ONE_BLOCK_PAD_AFTER_GROUPS and
SME2_TAIL_MIN_AFTER_GROUPS) to 5 (servil), 3, and 1, on the Mac, each in
a process of its own, at 64 B, 256 B, and 1 KiB, 16-129 messages, back
to back and with the program's work between calls (a read of every digest
and 2 us of integer work). With work between, 5 is best or level at
nearly every count (256 B x 19: 184.7 ns/msg at 5, 206.8 at 3, 206.1 at
1; 1 KiB x 19: 345 / 430 / 431; 64 B level within 5%); 1 and 3 win only
some back-to-back cells (64 B x 20: 29.6 at 5, 16.1 at 3), which real
programs rarely make. The thresholds stay.

## The regression check, calibrated (October 2, 2026, Mac jobs 1164-1166)

Zooko asked for a summary that is simpler and more reliable than the two
speeds. Two designs, on four conditions, six repeats each, on the Mac
(probe/plant-proportional's examples/host_lab.rs drives them;
bench-hashes probe/summary-calibration holds the second design):
- **fast**: `regress` as it was, each pair's fast speeds' ratio, four pairs
  all beyond the margin, four more to confirm, a SHA-256 control;
- **mean**: eight pairs, each run's mean per cell (total ns over total
  units), the median of the pairs' ratios beyond the margin and an exact
  sign test (7 of 8 pairs).
Conditions: one executable on both sides; the same code with another
layout (bench-hashes' feature layout-perturb, about 2 KiB of code); and
+3% and +6% planted (B3_PLANT: P of every 100 hash() calls hash twice, in
one executable, so servil st's lent 64 B, 64 KiB, and 1 MiB move by P%).
Held / passed / no verdict (busy):

| | fast | mean | fast, aligned | mean, aligned | fast, primed | mean, primed |
|---|---|---|---|---|---|---|
| one executable | 0/6/0 | 0/4/2 | 0/5/1 | 0/5/1 | 0/6/0 | 0/5/1 |
| another layout | 0/6/0 | 1/5/0 | 0/5/1 | 3/2/1 | 0/6/0 | 0/5/1 |
| +3% | 2/3/1 | 2/3/1 | 2/4/0 | 2/3/1 | 3/3/0 | 4/2/0 |
| +6% | 5/0/1 | 4/0/2 | 4/0/2 | 5/0/1 | 6/0/0 | 5/0/1 |

- **The fast speeds missed bulk.** At +6% the fast design held 64 B and
  64 KiB and never 1 MiB (0 of 18); the mean held all three nearly every
  time.
- **The queue's cells cannot be judged at 3%.** Every false hold of the
  mean was a queue cell (continuous messages of 64 B and 1 KiB, batches of
  4096, lent batches of 16); on identical code their per-process means
  move 6-60% even as medians of eight pairs. The fast speeds avoided false
  holds there by ignoring the slow speed's share, which is a real cost to
  callers. 25 of 34 solo cells kept their null median within 3%.
- **The shared lent cells** kept their null medians within 8% in all 36
  checks: at 10% none would have held.
- **Alignment** (LLVM's -align-all-functions=6, -align-all-nofallthru-
  blocks=5, the C kernels' -falign-functions=64): no narrower, the binary
  6% larger. Dropped.
- **Priming** (Devon Jonte's finding, bench-hashes#4: one untimed batch
  before each cell's calibration): the queue's 64 B samples grow to their
  intended length (Mac 176 us before); the queue cells' null spread
  narrows little.
Taken (Zooko, October 2): the mean everywhere; the gate as the mean design
over the lent cells only (the queue's cells out of it, their stability the
next work), solo at 3% and shared at 10%, with no control and no
confirmation stage; priming in place of the single-call retiming. A check
takes about 31 s in the VM (was about 60).

**The gate as built** (bench-hashes 1a2d01f, Mac job 1167, eight repeats,
probe/plant-final with bench-hashes probe/final-calibration), held /
passed / no verdict (busy): one executable 0/5/3; another layout 1/7/0;
+3% 5/2/1; +6% 6/0/2; 28 s a check. Two findings:
- The layout hold: servil st's solo lent 64 B at +3.5% (pairs +4.1 +17.1
  +1.4 -4.4 +2.8 +1.5 +5.0 +11.4%): that build ran the cell slower, which
  no count of pairs separates from code. A 3% margin on the 64 B call
  holds about one layout-only change in eight; PROCEDURES says how such a
  hold lands.
- The shared lent 64 B cells switch between states 20-30% apart per
  process (layout checks: the other layout about 20% faster in 6 of 8; one
  +3% check held by that cell alone at +19.5%). So the gate judges solo
  cells alone (bench-hashes 1a2d01f's successor): from the same runs,
  one executable 0 of 5 held, another layout 1 of 8, +3% 4 of 7, +6% 6 of
  6.

The VM (the same launcher run in the guest, bench-hashes 5b3c5cb, solo
alone, eight repeats, quiet): one executable 0 of 8 held, another layout
0 of 8, +3% 5 of 8 (at 64 B or 1 MiB), +6% 8 of 8; 31 s a check.

## The queue's cells between processes (October 2, 2026, VM)

Why the regression check cannot judge the queue's cells (the section
below). Twenty fresh processes of servil mt's queue cells, 24 rounds
each (bench-hashes probe/queue-sample-length, BENCH_QUEUE_SAMPLE_MS for
the queue's sample length), the spread of the processes' means beside
what their samples' own noise predicts (SD of the mean: within-process
variance over 24, pooled):

| cell | 1 ms samples: SD, noise, ratio | 4 ms samples: SD, noise, ratio | 4 ms against 1 ms |
|---|---|---|---|
| messages 64 B | 7.9%, 4.3%, 1.9 | 7.3%, 4.6%, 1.6 | 0.911 |
| messages 1 KiB | 2.0%, 1.6%, 1.2 | 1.1%, 1.0%, 1.1 | 0.866 |
| messages 16 KiB | 2.3%, 2.6%, 0.9 | 3.5%, 1.9%, 1.9 | 0.945 |
| batches of 16 | 2.9%, 1.3%, 2.2 | 2.1%, 1.4%, 1.5 | 0.925 |
| batches of 4096 | 3.6%, 2.8%, 1.3 | 2.0%, 1.7%, 1.2 | 0.927 |

- Most of a queue cell's spread is its samples' own noise: a 1 ms sample
  fills the queue, wakes its workers (15-45 us each), and drains it.
- 4 ms samples read 5-13% faster (filling and draining weigh less): the
  1 ms samples understate the queue's steady speed.
- The 64 B cell (and 16 KiB, and batches of 16 at 1 ms) differ between
  processes beyond their noise: a per-process state, for the Mac to
  locate (where each thread ran, per core kind), since a VM's host places
  its vCPUs.
- None gets near what a 3% check needs: a median of eight pairs moves by
  about 0.6 of a process's spread (64 B: about 4.4%).

**The Mac** (probe/queue-processes-mac, job 1176, 20 processes, 1 ms and
4 ms samples, traced): the queue's cells settle into a per-process state.
Messages of 1 KiB: 13 processes near the median, 7 at 1.30-1.51x (4 ms:
11 and 9); batches of 16: 10 at 0.75-0.80x, 10 at 1.20-1.55x (spread over
noise 4.6; 4 ms alike); 16 KiB over noise 3.9-5.5; 64 B and batches of
4096 near their noise. Longer samples do not merge the states. The
producer (the calling thread) runs the same instructions per input in
both states (1 KiB: 671-686; batches of 16: 643-667), on P-cores, at the
same clock (3.7-3.9 GHz), yet takes about 1.3x the time per input in a
slow process, busier (92-95% of wall against 84-88% for 1 KiB): more
stall cycles, likely its handovers' lines crossing between the M4 Max's
two P-clusters when macOS places the pool's or the delivery thread on
the other. Next: which CPU each queue thread and the producer ran on
(pthread_cpu_number_np), fast processes beside slow.

## Memory: what each call allocates (a survey, October 2, 2026)

For the memory guarantees Zooko asked to document (bench-hashes
NEXT-STEPS, the current to-do list). Read from the code, not yet measured
or tested beyond tests/queue_no_alloc.rs:
- **Nothing on the heap**: hash, keyed_hash, derive_key, hash_with,
  hash_many, hash_many_with, Hasher::new/update/finalize, OutputReader
  (the Hasher's chaining-value stack is an ArrayVec inside it).
- **Per call, freed at return**: hash_multithreaded and
  hash_many_multithreaded above the split (512 KiB): a Vec of pieces
  (capacity 64, 16 bytes each) and, for one message, a Vec of chaining
  values (32 bytes a piece): about 3 KiB. update_multithreaded alike per
  subtree it sends to the pool. Hasher::update_reader past 64 KiB: a
  1 MiB buffer (src/io.rs), freed at return; update_mmap*: the mapping.
- **Once per process**: the pool (its first multithreaded call, or
  initialize_multithreaded): one thread per CPU beyond the first, and the
  SME2 thread where SME2 is, each with std's default stack (2 MiB of
  address space, touched as used); the task list, a VecDeque that grows
  to the most tasks ever queued at once (each queue makes room for what
  its entries can have waiting); the queue's delivery thread.
- **Per queue**: its slots, in blocks that grow to the most submissions
  in flight at once and then stay (bounded by the program's own buffers),
  each slot's results Vec sized for its largest submission.
To settle before documenting: whether the transient Vecs earn their
place (a fixed array of 64 pieces on the stack would make the
multithreaded calls allocation-free too), and a test per claim (the
counting allocator of tests/queue_no_alloc.rs).

## b3sum, measured: tools/b3sum-bench (October 2, 2026, jobs 1136-1137)

`tools/b3sum-bench` (its README says how it measures) runs b3sum builds as
a user runs them, one process per run, start to exit, on files of 4 KiB
to 1 GiB and a tree of 1000 x 16 KiB, warm and cold (evicted without
root). Its counts come from `clocks::child`; on the Mac the per-core-kind
times from `proc_pid_rusage` add up to `wait4`'s CPU time within 0.1 ms
(job 1136), and every cold run read its whole input from storage (msync
MS_INVALIDATE evicts on APFS; posix_fadvise on ext4). Two Mac runs of the
same code (1136, 1137) agree within 1-3% in nearly every cell (rayon 16
MiB warm moved a share). The Mac runs through probe/b3sum-bench-mac,
whose `examples/host_lab.rs` builds the contenders and the tool.

Contenders: "rayon" the fork's b3sum before the move (779cd2d), "pool"
on the fork's pool with madvise WILLNEED (candidate/b3sum-pool), "read"
the pool's b3sum with --no-mmap (update_reader, one thread), "official"
b3sum 1.8.2 from crates.io. Fast speeds, time per run (the Mac's from
job 1137, quiet; the VM's from three runs of 15 rounds on its ext4 disk,
the cells in each against Rayon alike):

    warm                 rayon     pool      read      official
    Mac 4 KiB            1.85 ms   1.58      1.51      1.74
    Mac 16 MiB           3.31 ms   2.50      4.57      2.65
    Mac 1 GiB            68.4 ms   45.2      201       45.2   (23.7 GB/s pool)
    Mac 1000 x 16 KiB    20.9 ms   20.5      19.8      58.4
    VM 4 KiB             1.02 ms   0.52      -         0.96
    VM 1 GiB             64.2 ms   28.3      208       41.9   (37.9 GB/s pool)
    cold
    Mac 16 MiB           8.13 ms   7.81      6.07      7.99
    Mac 1 GiB            365 ms    401       224       357
    VM 16 MiB            4.49 ms   9.05      4.18      4.63
    VM 1 GiB             89.3 ms   66.0      207       62.4

What it shows:
- **The pool wins warm.** Large files take two thirds of Rayon's time on
  the Mac and under half in the VM (Rayon's threads each take the SME2
  turn, one SME unit between them: probe/rayon-vs-pool, job 1002), level with
  official on the Mac and ahead of it in the VM; small files gain a
  quarter to a half (no Rayon pool to start: the fork's Rayon b3sum kept
  two CPUs busy for a 4 KiB file).
- **Cold, reading beats mapping.** On the Mac every mapping contender
  reads 1 GiB at 2.7-3.0 GB/s, the plain reads of `--no-mmap` at 4.8 on
  one thread; in the VM reads win from 64 KiB to 16 MiB. Page faults
  bring a mapping's pages in a few at a time; a read of 1 MiB asks the
  storage for 1 MiB.
- **Open** (ours, AGENTS "we own every slowdown"): the pool cold, 16 MiB
  in the VM 2x Rayon's time (about twice its major faults; WILLNEED took
  2.5x to 2.0x), 1 GiB on the Mac 1.10x. The pool's workers poll while
  others wait on faults (12-13 CPUs busy in the VM's 16 MiB cell).
- **The design these point to**: b3sum reads each file into its own
  buffers, 1 MiB at a time, and hands each to the pool while it reads the
  next (`Queue::pieces`, or update_multithreaded per piece): reads' cold
  speed with the pool's warm one. io_uring is the Linux form of the same
  overlap. Each is a contender for b3sum-bench.
- **A mixed tree** (1000 files, 1 KiB-16 MiB, 74 MiB, in one run, as
  `find | xargs b3sum`; Mac job 1146, quiet): warm Rayon 34.8 ms, the
  pool 26.3 (x0.755), official 52.3; cold Rayon 209, the pool 123
  (x0.589), plain reads 121, official 207. (VM: warm x0.875, cold level.)
- **The design taken** (Mac jobs 1148-1160, VM runs r5-r10): a file of
  512 KiB or more (where the pool takes over) whose first page is in the
  page cache (mincore) is mapped and hashed in place over the pool; every
  other input, stdin included, is read in 4 MiB pieces, the first on the
  calling thread (a file of one piece starts no thread), the rest on a
  scoped reader thread into the second of two buffers kept for the run,
  while the calling thread hashes the last piece with update_multithreaded.
  Mac, fast speeds (job 1158): warm 1 GiB Rayon 68.6 ms, this 37.3 (28.8
  GB/s; official 1.8.2 45.8); cold 1 GiB 367, 159 (6.7 GB/s); warm mixed
  tree 34.3, 24.4; cold mixed tree 191, 99.
  - Reads alone (5bd0577) lost warm in the VM (1 GiB 79 ms against 29
    mapped: its page-cache copy, one thread, at 13.6 GB/s).
  - Mapping files below 512 KiB lost on the Mac: hashed on one thread, a
    mapping faults once a 16 KiB page, and each fault puts the SME unit in
    its slow state (probe/willneed-mechanism, job 1156: 1 GiB on one
    thread 480 ms mapped, 185 prefaulted; the faults' kernel time alike,
    the difference user time, 4.5 us a page). Reading them instead: warm
    mixed tree 29.8 -> 24.4 ms on the Mac, 19.4 -> 20.2 in the VM, where
    Linux maps many pages a fault.
  - madvise(WILLNEED) prefaults, at about 260 ns a page on one thread:
    1 GiB on the pool 32.1 ms without, 39.5 with (the workers fault in
    parallel, at six times the system time but less wall time).
  - Residency by one page, not sixteen: at warm 256 MiB about 2% more wall
    time (mean 11.36-11.56 ms against 11.23-11.33, jobs 1158-1160; CPU
    time, cycles, and core kinds alike; back with the 15 calls' answers
    ignored), nothing at other sizes: taken for the simpler code.
  - The per-file reader thread of the first experiment (probe/b3sum-read-
    overlap) cost a tree of small files 3.5x: a thread and two buffers
    per file.
- Official b3sum's tree of 1000 small files takes 2.8x the fork's time on
  the Mac (13.6x in the VM), at 3-6 CPUs busy: its Rayon threads spin
  between files.

## The split below 512 KiB, measured again (October 1, 2026; jobs 841-851)

Zooko asked whether `hash_multithreaded` should use threads below 512
KiB again, now that minimax is gone. History, from the notes above: the
split sat at 32 KiB while workers polled between calls; once they slept
between calls (Serve real programs), a wake cost 15-70 us, more than a
32 KiB hash, and the split moved to 768 KiB (the worst case of both
machines, minimax) and then 512 KiB (native first).

Probe branches `probe/split-{32,64,128,256}k` change `MIN_SPLIT_LEN`
alone. Mac, mains, quiet, two runs a side in mirrored order (base
846/847, 32 KiB 845/848, 64 KiB 844/849, 128 KiB 843/850, 256 KiB
842/851; 837-840 ran on battery and are left out), servil st and mt,
16 KiB-1 MiB and batches of 256-8192, after other work, after idling,
and nonstop lent. servil st, the control, is level within 1-2% in every
cell (after idling's two-speed cells within their noise).

- After other work (workers asleep): the length at the split is slower
  than on the caller's thread: 32 KiB x2.34, 64 KiB x1.92, 128 KiB
  x1.25-1.29 (a slow speed x1.6-2.6), 256 KiB x1.08 (slow x1.76);
  batches likewise (512 x2.61 at a 32 KiB split, 1024 x2.14 at 64 KiB,
  2048 x1.33 at 128 KiB). This breaks "never slower than hash".
- After idling: as noisy as ever (two speeds), mostly slower below the
  split (64 KiB x1.8 at 32-64 KiB splits).
- Nonstop, lent (workers awake from the previous call): 256 KiB x0.81 at
  every split up to 256 KiB, batches of 4096 x0.81-0.84 (shared
  x0.41-0.70, shared 256 KiB x0.80-0.86); 64 KiB x1.12-1.14 slower at 32
  and 64 KiB splits; 1 MiB level.
- 1 MiB after other work read x0.72-0.79 at 64-256 KiB splits, but only
  as a new fast speed in 17-42% of samples (0.074-0.093 ns/B); the main
  speed matched the base (0.109-0.123 against 0.110-0.112). The cut of a
  1 MiB input does not depend on MIN_SPLIT_LEN: some calls found workers
  that a neighbouring cell's split left awake. A share, not a speed-up.

Verdict: 512 KiB stays; no constant wins everywhere. The gain is real
only where workers are already awake (nonstop calls back to back): a
rule "split from 256 KiB while workers poll, else from 512 KiB" would
take 256 KiB nonstop x0.81 and its batches, for one branch on state the
pool already has (`sleepers`). A design decision for Zooko; the queue is
the API built for that pattern.

**Built, measured, dropped** (Zooko, October 1, 2026: every piece earns
its place). The rule as built (split from 256 KiB when `workers_ready`)
changed nothing in the VM (old/new, two runs a side: lent 256 KiB 0.2005
-> 0.2008 ns/B, lent batches of 4096 12.48 -> 12.52 ns/msg, each within
the same code's spread): a stream of 256 KiB calls never wakes the
workers, since none of its calls split, so the rule never triggers. The
measured gain needs every such call to split (the probes fixed the split
at 256 KiB), or a call that wakes the workers without using them, which
spends their CPU time for a call that may come (AGENTS.md, "Serve real
programs"). As built it would help only when other work had just woken
the workers, and make a 256 KiB cell's speed depend on the cell before it.

## The queue hung without SME2; hash_multithreaded on one CPU (October 1, 2026)

**The hang** (fix 7270b21). Every Linux CI run since September 30 hung
in its quick run (x86-64 and arm64 GitHub runners) until cancelled.
Reproduced in the VM with a NEON-only build (`no_sme2`): 15 of 15 runs
of the nonstop cells hung, owned 256 KiB messages and batches of 4096
most often; the SME2 build never did. gdb at the hang: four tasks queued
and in flight, all fifteen workers asleep, none notified, `registered`
1 (the delivery thread's hold). Cause: a worker slept when it took no
piece and either nothing was registered or no task was queued. A
queue's tasks are pushed (waking workers) before the delivery thread
takes its hold, on its own thread; a worker woken in between saw nothing
registered, slept again on the push's wake, and the hold wakes no one.
On SME2 machines the SME2 thread's own task loop took the tasks. Now a
worker sleeps only with no piece taken and no task queued, and polls
while tasks are queued. After: 0 of 15 and 0 of 18 such runs hang;
every suite passes (both builds). The fork's tests never met it: no
test runs the queue's large tasks NEON-only under load. Wanted: such a
test (no_sme2, many 256 KiB submissions), and CI for the fork.

**One CPU** (802b6a5). With every CPU already holding a caller (or one
CPU: a taskset, or a container's limit, which available_parallelism
reads), hash_multithreaded hashed on the workers' platform (NEON) where
hash() takes SME2: 0.252 against 0.166 ns/B at 512 KiB pinned to one
CPU in the VM. It now calls hash_serial, as the batch path did; 0.165
against 0.165 after.

**The Mac's verdict on both** (job 852, 7270b21 against fa1ec7b, mains):
held, one cell: servil st, one 64-byte message after other work, fast
speed x1.34 and slow x1.17, shares level (4.12 (53%) | 8.14 (47%) ->
5.53 (56%) | 9.55 (44%) ns/B). That call never enters lanes.rs (hash()
on 64 bytes runs the one-chunk kernel), and cells under about 1 us after
a gap move 20% with code layout alone (job 783, ring as servil; "Cold
calls pay for servil's code size"): layout is the likely cause,
unproven. Open: a control (the same code laid out differently) before
any claim that these commits slow nothing.

## The queue on one CPU (October 1, 2026)

**The hang.** Devon Jonte's audit of bench-hashes (his branch
`candidate/devon-benchmark-audit`, AUDIT.md; x86-64 Linux) found
api_plan's bursts and resubmission tests hanging under `taskset -c 0`.
Reproduced in the VM with `no_sme2` (the SME2 build passes): the pool
spawns `available_parallelism() - 1` workers, which counts the process's
affinity and quota, so one CPU gives none, and without SME2 no SME2
thread either. A `Time` queue still pushed tasks, which nobody took, and
its delivery thread waited on them forever. Every one-CPU container on
x86 met it with the first queue entry of 16 KiB or more, or the first
gathered short message.

**The fix.** A queue hashes as tasks only when the pool has a thread
to take them (`lanes::takes_tasks`: more than one CPU, or SME2);
otherwise its delivery thread hashes every entry, as with `Energy`. One
boolean, `Inner::tasks`, replaces `max_threads` (which only ever answered
that question). `tests/one_cpu.rs` pins its process to one CPU and runs
each queue shape against the reference implementation: hangs (60 s
timeout) before, passes after, in the default, no_sme2, and pure builds.
VM perf_regress: no regression on the third check; the first two gave
no verdict, the control (sha256 lent 64 B, code unchanged) +6.4% and
+6.1% on the new side, a finding for the layout question below.

## perf_regress on the Mac: layout luck per side (October 1, 2026, jobs 852-901)

**Finding.** perf_regress's verdicts on the Mac's cold cells (after a
gap, under about 1 us) and on the nonstop two-speed cells were wrong in
about one check in five tonight, on identical code as often as on real
changes. The pair 7270b21 against fa1ec7b: held in 852 (64 B after other
work, +34%) and 865 (a batch of 4, +44%), passed in 853, 855, 864, 866.
Self-compares (one commit against itself): held in 880 (lent batches of
16, via the slow-speed rule: fast x1.31, slow x2.26), 881 (servil st 64 B
after other work, +54% then +47%, every pair one way, servil mt and
sha256 the same way), and 883 (lent batches of 16, a share swing); 869
called the new side 47-49% faster at batches of 16 after other work.

**Cause, for the cold cells: each side's own layout.** The two sides of
a self-compare are built from different source paths (`tmp/perf-ab/old`,
`new`), which enter the crates' hashes: in the VM 559 of 2578 functions
sit at other addresses in one side than the other (0x68-0xa8 apart).
A cold call fetches its code from DRAM (bench-hashes NOTES, "The cause:
where the hash's code is"), so its time depends on where that code lies,
by up to about +-60% at 64 B on the Mac, and that luck stays with a side
for the whole check: alternating the sides cannot cancel it. Controls:
- each run from a fresh copy of its executable (probe
  `perf-regress-fresh-copy`): 881 and 883 still held. Not the file's
  pages.
- both sides running one executable (probe `perf-regress-same-exe`,
  jobs 892-901, 8 valid checks): no cold-cell series one way through
  all pairs, no hold; with two executables 6 of 16 checks had one.
- 64-byte function alignment (bench probe `align-functions`): nonstop
  lent 64 B in the VM level again (sha256 x1.083 -> x0.996), but the
  Mac's cold cells unchanged (865 held with it).

**A second effect, run order.** In the short later pairs the second
process of a pair runs 64 B after other work about 1.5x slower than the
first (one executable, ratios 1.35/0.66, 1.21/0.65, 1.53/0.69 by pair):
alternation cancels it in the median, but it pushes single pairs past
the 20% margin, and with layout luck the pairs line up.

**The slow-speed rule fires on identical code.** `slow_speed_moved`
compares two-speed nonstop cells' slow medians at the solo margin (3%);
with one executable on both sides it called lent batches of 16 or shared
64 B messages faster in 5 of 8 checks (slow x0.70-0.94).

**Decided (Zooko, October 1, 2026; AGENTS.md, "Every piece earns its
place"):** the slow-speed rule is deleted, and perf_regress measures the
nonstop use cases alone: telling a 20% change from layout luck of about
+-60% would take a dozen layouts per side or more, a dozen builds, for a
number that predicts little about a user's own layout. What that gives
up: perf_regress no longer sees a cold call slow down; a cold-path
regression shows in the benchmark's cells after a gap (with the same
luck) and in review (the code a call fetches). A check on the VM: 27 s
with builds, pair 1 8.2 s (21 s before). Open: the run-order effect's
cause.

**The nonstop-only check on identical code** (October 1, 2026): servil
6afda66 against itself, 8 checks on the Mac (jobs 916-923, mains) and 8
in the VM: no verdict of any kind, no cell listed slower or faster, no
control move, in all 16. (One "faster" on servil's lent batches of 16, a
two-speed cell, came in a VM check of a real change the same day: watch
for it.)

**Function alignment, measured and left out** (Zooko, October 1, 2026:
not worth its complexity). Every function on a 64-byte boundary
(`-C llvm-args=-align-all-functions=6`), five pairs of consecutive
code-changing commits, each built both ways, judged by the control
(sha256, whose code never changes): on the VM, unaligned, the control
moved one way through every pair in 1 of 5 (fa1ec7b -> 2cc0c00, +-5%, a
no-verdict that repeats on every rerun), aligned in none; on the Mac
(jobs 902-911, bench probe `align-functions-2`, on battery), no control
cell moved past 3% either way. The Mac's nonstop cells meet no layout
luck to remove, so the only benefit is fewer stuck checks on the VM.

**The Mac's held 64 B cell (job 852) is layout luck**: the A/Bs of
7270b21 against fa1ec7b read servil st 64 B after other work x0.93
(unaligned, jobs 856-859) and x0.88 (aligned, 860-863). No regression.
