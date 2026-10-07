**Title:** BLAKE3: add a Lean specification

---

This adds `BLAKE3/`, an executable Lean specification of BLAKE3 transcribed from `BLAKE3.md`, in
the layout of `kopis/` (#368), and one paragraph in the appendix that points at it.

**How the Lean follows the document.** Sections 2.2 to 3.3 (the IV, the permutation, the flags,
`G`, and `BLAKE3_COMPRESS`) are generated from `BLAKE3.md` by `generate.py`, which translates each
pseudocode construct by one rule and refuses any form outside those rules. Section 4 (the tree and
the three modes) is prose, and is transcribed by hand, each definition under the text it
implements.

**How it is checked.** `python3 BLAKE3/check.py` checks that:

1. the generated section equals `generate.py`'s translation of `BLAKE3.md`;
2. the Lean reproduces every value of the appendix's two traces (34 compressions, the state after
   each round, both hash values), after the trace is checked against its own text;
3. the Lean reproduces the 35 test vectors the appendix links, in all three modes;
4. `Theorems.lean` proves the tree divides its chunks as section 4.3.2's rules say;
5. every quotation of `BLAKE3.md` in the Lean's comments is the document's text.

Lean itself is the only dependency (v4.34.1, pinned in `lean-toolchain`), and the check takes a few
seconds. The Markdown lint passes.

**For the maintainers to decide:**

- The version: the document gains one informative paragraph.
- Whether the Lean should settle any disagreement with the prose, as `kopis.md` says of its Lean.
  This change keeps `BLAKE3.md` the specification and the Lean its transcription.
- The first trace labels its one compression `CHUNK 1, BLOCK 0`; the second trace numbers chunks
  from 0. A one-character change, for a separate commit.
