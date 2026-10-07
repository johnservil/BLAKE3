# BLAKE3 in Lean

`Blake3.lean` is a running Lean specification of BLAKE3, transcribed from the Markdown
specification, [`BLAKE3.md`](../BLAKE3.md). Sections 2.2 to 3.3 (the constants, `G`, and
`BLAKE3_COMPRESS`) are generated from the Markdown by `generate.py`, one rule per pseudocode
construct; section 4 (the tree and the hashing modes) is prose there and is transcribed by hand,
each definition under the text it implements.

`check.py` asserts the Lean matches the Markdown:

1. the generated section equals `generate.py`'s translation of `BLAKE3.md`, byte for byte;
2. every value of the appendix's traces (each compression's inputs and output, the state after
   each round, the hash values) is extracted, checked for consistency with the text, and
   reproduced by the Lean;
3. the official test vectors the appendix links are fetched at the commit it names, checked
   against their SHA-256, and reproduced by the Lean in all three modes;
4. `Theorems.lean` proves the tree's split is the one section 4.3.2's rules allow, and that the
   permutation is one, on Lean's standard axioms alone.

## How to run

Install Lean (see [here](https://lean-lang.org/install/manual/)); it needs no other package.

    python3 check.py

Or the parts: `lake build` checks the specification and the theorems, and
`lake env lean --run Tests.lean traces.json test_vectors.json` runs the known-answer tests
(`check.py` writes `traces.json` and fetches `test_vectors.json`).
