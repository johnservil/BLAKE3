**Title:** BLAKE3: add a Lean specification, definitive where the two differ

---

This adds `BLAKE3/`, an executable Lean specification of BLAKE3 transcribed from `BLAKE3.md`, in
the layout of `kopis/` (#368). A paragraph in the introduction points at it and makes it
definitive where the two differ, as `kopis.md` does for its Lean.

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

The change leaves every BLAKE3 output as it was and makes the Lean definitive, so v1.1.0 seems the
right version. #384 corrects the first trace's chunk label; the two changes are independent.
