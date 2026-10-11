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

- Upstream: rems-project/sail#1772 proposes the fix, with an exec test
  that fails without it and passes with it; Sail's Lean suite passes
  (247 passes, 0 failures).
- Decoding from the opcode word: slicing to the integer data-processing
  decoder (`__DecodeA64_DataProcReg`) keeps 928 functions (17,518 lines
  of Sail, 17,841 of Lean; it brings in PMU events and implementation-
  defined choices). Sail's main branch needs the `vector_update` overload
  declared in the slice (its rewrites call it; the slice drops overloads).
  Lean's elaboration of the result takes over 40 minutes; the decoder is
  one 2,542-line definition.

- Arm's model maps `MOD` (`emod_nat`, prelude.sail) to Lean's `Nat.div`,
  so every rotation (`ROR`) in the Lean model computed a quotient; mapped
  to `Nat.mod` here. Unreported upstream (sail-arm) as of this note.
- `Gpr.lean` reads and writes general registers as plain values (`gpr`,
  `setGpr`), proved equal to Arm's `read_gpr` and `write_gpr`;
  `Insns.lean` proves `add`, `eor`, and `ror` (`extr`) on 32-bit
  registers for any registers and any state; `G.lean` proves 14
  instructions, BLAKE3's G as the compiled portable code computes it,
  compute G on any inputs from any state. Lean's standard axioms; the
  whole sequence checks in under a second.
- The translation, tested: compiled (without `--lean-noncomputable`), the
  Lean model agrees with the CPU on 8,000 random cases of `add`, `eor`,
  `eor` with a rotated operand, and `extr` (`cpu_check.py`). With `MOD`
  as `Nat.div`, 1,931 of 2,000 `eor ..., ror #k` results differ: the
  check finds translation bugs of this kind. (`extr` computes without
  `ROR`, so it is unaffected.)
- Choices: the generated model resolves Arm's UNKNOWN values with Sail's
  trivial choice source (fixed values). A proof under it holds for one
  resolution; a faithful proof quantifies over every choice source.
- `bv_decide` checks its certificate with compiled code by default and
  adds an axiom for it; the proofs here avoid it.

- Loads: slicing to `ldr` (unsigned offset) keeps 1,461 functions (26,837
  lines of Sail, 25,410 of Lean): `Mem_read`, `AArch64_MemSingle_read`, and
  address translation (`AArch64_TranslateAddress`) with its checks; Lean was
  still building it after 30 minutes, when it was stopped. Where
  the theorem states that virtual memory behaves as memory (at `Mem_read`,
  as a hypothesis on the state; or translation proved the identity under
  conditions on the system registers) is a decision for Zooko.

Next: the opcode word's decoding (whether the decoder's build time is a
one-time cost we accept, or a smaller route to it), loads and stores, NEON,
any choice source, and a whole kernel against `Blake3.lean`'s compression.
