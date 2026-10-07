# Proofs of the assembly kernels

These tools prove every assembly kernel on AArch64 equal to BLAKE3's
compression function, for every input, key, counter, and flags value:

    python3 tools/verify/prove_hybrid.py [KERNEL...]   # c/blake3_neon_hybrid_aarch64.S: c1, k2-k10, p2-p9, q1-q9 (327 cases, 2 min on 16 cores)
    python3 tools/verify/prove_sme2.py [KERNEL...]     # c/blake3_sme2_aarch64.S: chunks, chunks_at, messages, parents, xof (34 cases, 3 min)
    python3 tools/verify/check_spec.py                  # the definition (spec.py) against BLAKE3's official test vectors
    python3 tools/verify/cross_check.py                 # the NEON and integer models against the CPU
    python3 tools/verify/cross_check_sme.py             # the streaming SVE and SME2 models against the CPU (needs SME2)
    python3 tools/verify/mutants.py                     # wrong kernels are rejected

They need Python 3 with Z3 (`python3-z3`), an AArch64 machine with the
SHA-3 extension, and Clang and llvm-objdump 17 or later (for SME2). CI
runs all but the SME2 cross-check, which needs an SME2 CPU (Apple M4, or
the VM on it).

**Coverage.** The hybrid kernels at every block count they take (1 to
16; the q kernels' partial chunk at every block count). The SME2 kernels
at the 512-bit streaming vector length (`cntw` = 16, which they check
themselves), for one and two groups, the last group storing 1, 15, or
all 16 values, and the message kernel at every message length of 2 to 16
blocks with the last block of any length. More groups repeat the loop
these cases run twice; the proofs do not yet show the loop's state the
same at each iteration.

## What a proof shows

For each kernel and block count, `prove_hybrid.py` assembles the file,
takes the kernel's instructions from objdump's disassembly of the object
(what the CPU runs, macros expanded), and executes them symbolically
(`aarch64.py`, `vector.py`): registers and memory hold Z3 terms over the
inputs, pointers are (region, offset) pairs. The run checks:

- **The result**: each output word equals the specification's
  (`spec.py`, written from the BLAKE3 paper), compared in a normal form
  (`canon.py`) that sorts sums and xors, keeps rotations, and undoes the
  byte and lane shuffles of loads, stores, spills, and transposes; small
  terms without rounds in them (the counter's halves, the flag words) go
  to Z3's solver. Equal normal forms mean equal terms; a mismatch is a
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

- **The definition**, `spec.py`: BLAKE3's compression function in 40
  lines, written from the BLAKE3 paper (section 2.2). `check_spec.py`
  builds a whole BLAKE3 on it (chunks, tree, keyed hashing, key
  derivation, extended output, from the paper's sections 2.1-2.6) and
  reproduces all 35 official test vectors in all three modes, 131 bytes
  each; a swapped permutation entry or a rotation off by one fails it.

- **The instruction models** (`aarch64.py`, `vector.py`, `sme.py`), each
  a few lines following the Arm architecture reference manual.
  `cross_check.py` and `cross_check_sme.py` run every register form the
  kernels use (83 NEON and integer, 73 streaming, ZA moves, slice loads
  and stores, and predicates among them), with their own immediates, on
  the CPU and in the model over random states. The SME2 check found one
  model error: a four-register ZA move rounds its slice base down to a
  multiple of four (the kernels use aligned bases, so no proof was
  affected); the model now does too. A stronger base would replace the models with Arm's
  own machine-readable specification (Sail, Isla).
- **objdump's disassembly** of the assembled object, and the assembler.
- **The contracts as the harnesses state them**: the input buffers apart
  from each other and from `out` (c1 also with `out` equal to `key`, as
  `compress_in_place` calls it), `packed` with zero upper bits, and the
  q kernels' table layout (`prove_partial`). A caller that breaks them
  is outside the proof; the Rust callers are covered by tests and Kani.

Not proved here: the C NEON kernel, the x86 kernels, the portable Rust,
and the Rust around the kernels.

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
