# Proofs of the assembly kernels

These tools prove the hybrid kernels in `c/blake3_neon_hybrid_aarch64.S`
equal to BLAKE3's compression function, for every input, key, counter,
and flags value, at every block count each kernel takes:

    python3 tools/verify/prove_hybrid.py [KERNEL...]   # c1, k2-k10, p2-p9, q1-q9 (327 cases, 2 min on 16 cores)
    python3 tools/verify/cross_check.py                 # the instruction models against the CPU
    python3 tools/verify/mutants.py                     # wrong kernels are rejected

They need Python 3 with Z3 (`python3-z3`) and an AArch64 machine with the
SHA-3 extension; CI runs all three (`kernel_proofs`).

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

- **The instruction models**, about 40 forms, each a few lines following
  the Arm architecture reference manual. `cross_check.py` runs every
  register-only form the kernels use, with its own immediates, on the CPU
  and in the model over random states; loads and stores are covered by
  the region checks. A stronger base would replace the models with Arm's
  own machine-readable specification (Sail, Isla).
- **objdump's disassembly** of the assembled object, and the assembler.
- **The contracts as the harnesses state them**: the input buffers apart
  from each other and from `out` (c1 also with `out` equal to `key`, as
  `compress_in_place` calls it), `packed` with zero upper bits, and the
  q kernels' table layout (`prove_partial`). A caller that breaks them
  is outside the proof; the Rust callers are covered by tests and Kani.

Not proved here: the SME2 kernels (`c/blake3_sme2_aarch64.S`), the C NEON
kernel, the portable Rust, and the Rust around the kernels.

## Extending

An instruction the models lack stops a proof with "instruction not
modelled"; add it to `aarch64.py` (scalar) or `vector.py` (NEON), then to
`cross_check.py`'s coverage by running it (it finds every form itself). A
new kernel needs a harness stating its contract (`prove_table`,
`prove_partial`, `prove_c1` are the patterns) and a line in `cases`.
