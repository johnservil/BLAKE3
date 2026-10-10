# Probe: the machine-code proofs in Lean, on Arm's own specification

Can Lean prove our kernels against Arm's specification of the architecture
(Sail's Armv9.4 model, translated to Lean by Sail), so that one checker
covers everything and our own instruction models, Z3, and Isla go?

**Findings so far** (October 10-11, 2026):

- The whole model does not translate: Sail ran out of memory at 81 GB, two
  hours in. Sliced to the instructions a proof needs (`slice.sh`, `ROOTS`),
  it translates in seconds and builds in Lean in seconds (three
  instructions: 948 lines of Sail, 24 Lean definitions).
- Sail's Lean backend (0.20.3 and main) mistranslated an assignment to a
  tuple of local variables, `(result, nzcv) = AddWithCarry(...)`: the
  values were dropped, so `add` wrote an undefined value.
  `sail-let-assignments.patch` lets `rewrite_var_updates` thread variables
  updated in a let's body, as it does for an if's branches; the Lean
  backend omits the letbind_effects pass that made this unnecessary
  elsewhere.
- With it, `Probe.lean` proves `add w0, w1, w2` as Arm's specification
  decodes and runs it: from any state, register 0 receives the 32-bit sum
  of registers 1 and 2. Lean's standard axioms only.
- The generated code carries widths as unevaluated expressions inside
  instance arguments (`if sf == 1 then 64 else 32`), which `simp` does not
  rewrite; `erw` does.

Next: decoding from the opcode word, loads and stores, and a sequence of
instructions (one G function), to measure how proofs scale.
