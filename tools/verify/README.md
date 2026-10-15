# Proofs of the compression code

These tools prove every assembly kernel on AArch64 (the SME2 hybrid,
`c/blake3_sme2_hybrid_aarch64.S`, at one and two groups so far), and the library's Rust
compression code, equal to the compression function of the Lean
specification of BLAKE3 (`c2sp/BLAKE3/`), for every input, key, counter,
and flags value:

    pypy3 tools/verify/prove_hybrid.py [KERNEL...]   # c/blake3_neon_hybrid_aarch64.S: c1, k2-k10, p2-p9, q1-q9 (327 cases, 2 min on 16 cores)
    pypy3 tools/verify/prove_sme2.py [KERNEL...]     # c/blake3_sme2_aarch64.S: chunks, chunks_at, messages, parents, xof, and the chunk and parent kernels at every group count (36 cases, hours); c/blake3_sme2_hybrid_aarch64.S: chunks_x2 (2 cases, 9 min)
    pypy3 tools/verify/prove_sme2.py xof_every messages_every  # the other kernels at every group count (hours)
    pypy3 tools/verify/lean/emit.py                   # the definition and schedules, from the Lean specification (needs Lean)
    pypy3 tools/verify/prove_rust.py                  # the Rust paths, compiled: every platform's (363 cases)
    pypy3 tools/verify/prove_timing.py                # whole calls return the standard's result on one path (2316 cases)
    pypy3 tools/verify/prove_flat.py                  # the SME2 flat walk at every subtree it takes: the standard's tree over its kernels (30 cases)
    pypy3 tools/verify/cross_check.py                 # the NEON and integer models against the CPU (and every form the Rust and whole-call proofs run)
    pypy3 tools/verify/cross_check_sme.py             # the streaming SVE and SME2 models against the CPU (needs SME2)
    pypy3 tools/verify/mutants.py                     # wrong kernels are rejected
    pypy3 -m unittest discover -s tools/verify -p test_helpers.py  # the normalizer, the induction's generalization, the leak detector
    pypy3 tools/verify/isla_check.py                  # the NEON and integer models against Arm's specification (needs Isla)
    sh tools/verify/tree/check.sh                       # the tree walk in safe Rust against the specification's tree (needs Aeneas)

They need Python 3 with Z3 (`python3-z3`), an AArch64 machine with the
SHA-3 extension, and Clang and llvm-objdump 17 or later (for SME2). CI
runs all but the SME2 cross-check, which needs an SME2 CPU (Apple M4, or
the VM on it).

**Coverage.** The hybrid kernels at every block count they take (1 to
16; the q kernels' partial chunk at every block count). The SME2 kernels
at the 512-bit streaming vector length (`cntw` = 16, which they check
themselves), for one and two groups, the last group storing 1, 15, or
all 16 values; the SME2 kernel with an integer lane
(`c/blake3_sme2_hybrid_aarch64.S`: 16 SME2 lanes and 2 integer ones a
group) for one and two groups (not yet at every group count, where the
library calls it with 8 or more), and the message kernel at every message length of 2 to 16
blocks with the last block of any length. The chunk kernel at every group
count up to 2^40 groups, its last group storing all 16 values, by
induction over its group loop (`prove_sme2.py every`, 85 minutes): every
read of a chunk is checked to lie in the current group's own chunk. The
parent kernel the same way (`parents_every`, under a minute); it preloads
the next group's pairs, so each group's words differ instead (word w of
group G is pair_w + G): hashing another group's pairs gives other terms.
The extended-output kernel (`xof_every`, under a minute), and the message
kernel at each message length of 2 to 16 blocks (`messages_every`, 2 to
61 minutes each). CI runs them as a matrix of jobs.

## What a proof shows

For each kernel and block count, `prove_hybrid.py` assembles the file,
takes the kernel's instructions from objdump's disassembly of the object
(what the CPU runs, macros expanded), and executes them symbolically
(`aarch64.py`, `vector.py`): registers and memory hold Z3 terms over the
inputs, pointers are (region, offset) pairs. The run checks:

- **The result**: each output word equals the specification's
  (`lean_spec.py`, from the Lean specification), compared in a normal form
  (`canon.py`) that sorts sums and xors, folds each one's constants and
  parameters into one term wherever its operands came from, keeps
  rotations, and undoes the byte and lane shuffles of loads, stores,
  spills, and transposes; small terms without rounds in them (the
  counter's halves, the flag words) go to Z3's solver. Equal normal forms mean equal terms; a mismatch is a
  failed proof, never a false one.
- **Memory**: every access stays inside the buffer the kernel's contract
  gives it (inputs, key, out, the pointer table) or inside the stack frame
  it allocated; nothing is read from the stack before it is written; read-
  only buffers are never written.
- **Control flow depends only on the block count**: a branch, a select,
  an address, or a table index that depends on data stops the proof. A
  proved kernel is therefore also free of data-dependent branches and
  addresses (constant time, in that sense).
- **The calling convention**: callee-saved registers (x19-x29, d8-d15)
  and sp come back unchanged, and the kernel returns through x30.

## What it rests on

- **The definition**: the compression function of the Lean specification,
  `c2sp/BLAKE3/Blake3.lean`, which is generated from C2SP's `BLAKE3.md` and
  checked against all of it (its README). `lean/Generic.lean` holds the
  same generated text over any word type; `lean/Bridge.lean` proves, in
  Lean's kernel, that at 32-bit words it is the specification's
  `BLAKE3_COMPRESS`; `lean/Sound.lean` proves that the symbolic terms it
  computes (`symbolic`) evaluate, under each node's meaning on 32-bit words
  (`Term.eval`), to the specification's `BLAKE3_COMPRESS` of the inputs.
  `lean/Emit.lean` writes those terms as a graph, `lean/compress.json`, and
  `lean_spec.py` reads each node as Z3's operation of the same meaning; the
  printer and the reader (some 40 lines) are checked together against the
  specification's own outputs at sample inputs (`lean/emit.py`).

- **The schedules** of whole calls: `lean/Schedule.lean` lists, for an
  input of `n` bytes, the compressions BLAKE3 performs in order, each word
  named by its source (the key, an input word, an earlier compression's
  chaining value); `lean/ScheduleProofs.lean`'s `run_eq` proves that
  performing them gives the specification's `blake3` output, for every
  key, mode, and input. `lean/Emit.lean` writes the schedules of the
  whole-call lengths to `lean/schedules.json` (`lean/emit.py`), and
  `whole.py` performs them with `lean_spec.py`'s compression.

- **The instruction models** (`aarch64.py`, `vector.py`, `sme.py`), each
  a few lines following the Arm architecture reference manual.
  `cross_check.py` and `cross_check_sme.py` run every register form the
  kernels use (83 NEON and integer, 73 streaming, ZA moves, slice loads
  and stores, and predicates among them), with their own immediates, on
  the CPU and in the model over random states. The SME2 check found one
  model error: a four-register ZA move rounds its slice base down to a
  multiple of four (the kernels use aligned bases, so no proof was
  affected); the model now does too. `isla_check.py` proves the models
  equal to Arm's own specification (Sail's Armv9.4, run by Isla) for every
  input on every path that completes: the general registers, the flags,
  and a NEON register's 128 bits. It checks 322 of the 325 forms the
  proofs use, the flag-setting and conditional ones and the multiplies
  among them; Isla's symbolic run stops inside Arm's specification for
  `tbl` with one table register, `addv` and `cnt`, which stay checked
  against the CPU. It then plants
  model errors and requires each rejected: a carry computed as `>`, `hi`
  ignoring Z, `xar` (NEON and streaming) and a shifted `eor` rotated one
  place off, `dup` reading the wrong lane, a ZA move into the wrong tile.
  The snapshot runs with Sail's later fix of `unsigned_subrange`
  (rems-project/sail 1f8f173), which the check applies itself. Without
  it, the snapshot reads `dup`'s element index as `imm5[4]`
  ([Isla #107](https://github.com/rems-project/isla/issues/107)). It
  needs Isla's source (`ISLA`), `isla-footprint` (`ISLA_FOOTPRINT`),
  the Armv9.4 snapshot (`ISLA_SNAPSHOT`), and LLVM's assembler (`LLVM_MC`):
  Isla `e9b5d94`, isla-snapshots `d8b3101`, whose snapshot archive's
  SHA-256 is `d8c547eefd125a8bd01a827d2733fc9eddbc415839e8d278310e527c7721a3db`.
  The SME2 kernels' streaming register forms run the same way in
  streaming mode, all 16 lanes and ZA cell by cell, each ZA move at the
  slice base its register holds in the kernels: 19 of 23 agree. The four
  moves into tile 1's vertical slices exhaust Isla's memory (98 GB, as do
  single-register writes into ZA); they, the predicated `add`, `cmphi`,
  and the ZA loads and stores stay checked against the CPU. Memory effects
  are outside it.
- **objdump's disassembly** of the assembled object, and the assembler.
- **The contracts as the harnesses state them**: the input buffers apart
  from each other and from `out` (c1 also with `out` equal to `key`, as
  `compress_in_place` calls it), `packed` with zero upper bits, and the
  q kernels' table layout (`prove_partial`). A caller that breaks them
  is outside the proof; the Rust callers are covered by tests and Kani.

Whole calls (`prove_timing.py`, through `rust/src/lib.rs`'s `verify_hash`)
run the compiled library's `hash`, built by the compiler
`rust/rust-toolchain.toml` names, with its default features as
users build it, on symbolic message bytes: a thread's first call, then its
second on new bytes: `hash`, `keyed_hash` on a symbolic key, `derive_key`
with the test vectors' context string, and a `Hasher` given the message in
two updates cut at its middle. Each returns the standard's 32-byte result
for its message (`whole.py`: the compressions `lean/schedules.json` lists
for the length, which `lean/ScheduleProofs.lean`'s `run_eq` proves are
the standard's, performed with `lean_spec.py`'s compression;
`test_helpers.py` checks it on the official test vectors), and runs one
path. Stack bytes not yet written read as
indeterminate values (compiled code copies structures with their
padding): one that steered a branch or an address would stop the run, and
one in the result would fail the comparison. The process state is the library's file bytes and
relocated pointers, the platform `initialize()` recorded (NEON or SME2),
std's feature cache holding every feature (the library
asks it only for SHA-3, which Apple's cores have), and a new thread's TLS
block. The clock is fixed: both calls after a pause, or both back to back,
the one choice besides the length that steers a call (whether it prefetches
its code first). The C library calls run by their POSIX contracts (memcpy,
memset, malloc, posix_memalign, free, pthread keys). The executor stops at
any branch, select, address or table index that depends on the message, so
each length that runs to the end runs one path for every message
(`test_helpers.py` plants a branch on a message byte and a table index taken
from one, and checks the run stops at each).

**Forms kept between runs.** A passing run of `prove_rust.py` or
`prove_timing.py` records the instruction forms it ran (`results.py`,
under `/tmp/target/verify-results`), named by the hash of the built
library and of the verify tools; `cross_check.py` and `isla_check.py`
check those forms' models from that record, and run the proofs
themselves when no record matches.

**The kernels by their statements.** Inside the Rust paths and whole calls,
each assembly kernel the hybrid plans and the SME2 wrapper call (k2 to k10,
the p kernels, SME2's chunk and parent kernels) runs by the statement its
own proof establishes (`table_cvs` in `prove_hybrid.py`, `chunk_cvs` and
`parent_cvs` in `prove_sme2.py`: one function each, used by the proof and
the contract alike), only where the call is one the proof covers (block
counts, flag words) and only where the library's bytes of the kernel are
the bytes of the object its proof ran (`proved_code`). The scalar kernels
run instruction by instruction. The cases run in parallel worker
processes, which return the instruction forms they ran.

The Rust paths (`prove_rust.py`, through `rust/src/lib.rs`, one exported
call of the library's `Platform` methods each, the library built as users
build it): the portable compression in place and its extended-output block;
the NEON and SME2 platforms' compression in place (the Rust around the
scalar kernel); every platform's compression of 1 to 16 whole blocks; the
`hash_many` of chunks and of parents at every count to each platform's
degree (portable 1, NEON 16, SME2 128: the Rust around the hybrid kernels'
plans, and SME2's wrapper with its kernels run by the statements
`prove_sme2.py` proves of them, `chunk_cvs` and `parent_cvs`, one function
each for both; parents one after another in one buffer, as every caller
lays them out); the NEON platform's extended output at every
count (Rust NEON intrinsics, eight blocks an iteration, then four, then the
portable code): 1 to 20 blocks one by one, and 16 or more by induction over
the loop (`induction.py`).

**Induction** (`induction.py`): two successive arrivals at a loop's head are
generalized into one state at a symbolic iteration K (positions equal in
both kept, those that differ by a constant written start + K x difference,
the rest unknown); every path from that state is run, the run splitting
where a branch on the lengths is open, and each must return to the head in
the state for K + 1, its iteration's output right and nothing else
written, or leave the loop with all its output right. Only symbols declared
as lengths may steer a branch or an address, so a branch on data still
stops a proof; equalities of lengths are proved under the path's
assumptions. The compiled library runs as a whole, its calls of memcpy and
memmove by their contracts.

Not proved here: the C NEON kernel, the x86 kernels, and the library's
Rust that arranges the tree (chunks, parents, the threads), which the
tests cover. `tree/` proves a safe-Rust version of the library's
single-thread walk (`tree/README.md`).

**Code changed for the proofs.** The SME2 message kernel took the last
block's length with a branch (0 for 64), so a proof had to fix the
length; it now computes `((length - 1) & 63) + 1`, three instructions a
group, and one proof covers every length.

## Extending

An instruction the models lack stops a proof with "instruction not
modelled"; add it to `aarch64.py` (scalar) or `vector.py` (NEON), then to
`cross_check.py`'s coverage by running it (it finds every form itself). A
new kernel needs a harness stating its contract (`prove_table`,
`prove_partial`, `prove_c1` are the patterns) and a line in `cases`.
