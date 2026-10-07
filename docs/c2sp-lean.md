# A Lean specification for C2SP's BLAKE3 (for Zooko)

`c2sp/BLAKE3/` holds a Lean specification of BLAKE3 laid out as a C2SP spec's companion
directory, beside `c2sp/BLAKE3.md`, a copy of C2SP's `BLAKE3.md` (C2SP/C2SP at c559ede, its
latest commit, July 28, 2026; v1.0.0 is cd7fc6e, September 14, 2024). Its README says what it is
and how to run it.

## Have C2SP's maintainers discussed Lean specifications?

Yes, once, and favourably, this month. No spec in the C2SP repository carries Lean yet.

- **BLAKE3's own proposal**, C2SP/C2SP#101 (by veorq, 2024): "One of our motivations with this
  submission is to get BLAKE3 added to OpenSSL (which seems to require some formal validation)."
  A formal specification serves that stated aim.

- **Kopis** (a KEM by Michael Rosenberg, `rozbb`): its proposal, C2SP/C2SP#375 (approved October 6,
  2026, by FiloSottile, dconnolly, and str4d), points at a Rust implementation with "a Lean
  specification and proofs that the Rust in fact implements the Lean spec". In its draft pull
  request, C2SP/C2SP#368, rozbb planned to move the Lean into the test vectors' repository and
  wrote "IMO if the Lean or Python ever disagree (they really really shouldn't), then the Lean
  should win." FiloSottile answered (October 1, 2026): "It might even make sense to put the Lean
  in the C2SP repo, under the kopis/ directory." The pull request now adds `kopis/Kopis.lean`,
  `kopis/Tests.lean`, `kopis/README.md`, `kopis/lakefile.toml`, `kopis/lean-toolchain`, and
  `kopis/lake-manifest.json`.
- **The manual** (`.github/MANUAL.md`, "GitHub permissions"): a spec's maintainers may approve
  and merge changes to its `.md` file "and to anything under a directory named like the spec",
  so a `BLAKE3/` directory is the BLAKE3 maintainers' own (Zooko is one of four). Commit messages
  take a `BLAKE3:` prefix. Test vectors go in C2SP/CCTV, which has no BLAKE3 directory; C2SP's
  BLAKE3 links the BLAKE3 team's `test_vectors.json` at a pinned commit instead.
- **Companion code elsewhere in C2SP**: Go (XAES-256-GCM, chacha8rand), C (XAES-256-GCM's
  OpenSSL), and Python (det-keygen, jq255) reference implementations, each in a directory named
  for its spec. Issue #200 (CCP-SIV) asked for a Python reference implementation; it is open.
- Searches of C2SP's issues and pull requests for Cryptol, hacspec, "machine-readable", and
  "executable" find nothing.

## The style the precedent sets, and what `c2sp/BLAKE3/` does

Kopis's Lean (the only precedent) is "a running Lean specification [...] transcribed function by
function from the Markdown specification": each function under the spec's name, its spec text
above it, a list of how the pseudocode's notation maps to Lean at the top, Lake with a pinned
toolchain, and known-answer tests run by `lake exe`. It depends on Mathlib and a library of the
author's (for TurboSHAKE).

`c2sp/BLAKE3/` follows the same layout and conventions, and differs where BLAKE3 allows more:

- **No dependencies**: core Lean alone (v4.34.1, the current stable release), so a reviewer
  builds it in seconds.
- **The formal part is generated, not transcribed**: sections 2.2 to 3.3 are fixed forms in
  `BLAKE3.md` (code blocks of numbers, a list of flags, pseudocode), and `generate.py` translates
  them deterministically, one rule per construct; it refuses anything else, so a change to the
  document regenerates cleanly or is refused. Section 4 is prose and is transcribed by hand.
- **A checker ties the Lean to every other part of the document** (`check.py`): the generated
  text, every value in the appendix's traces (34 compressions, the state after each of 7 rounds,
  two hash values), and the 35 official test vectors in all three modes; and it checks the
  appendix against its own text (each block starts from the previous output; the messages are
  the ones described).
- **Theorems for the prose**: section 4.3.2 describes the tree by rules ("left subtrees are full
  [...] left subtrees are big"); `Theorems.lean` proves the definition's split satisfies them and
  is the only one that does, in Lean's kernel.

Four deliberate errors were each caught: a rotation changed in the document (the generated text
differs), the same regenerated (the appendix's first compression differs), a wrong tree split (the
theorems fail), and parents without the mode flag (the keyed example's hash differs).

## Two things the document could gain

- **The first example's label**: its one compression is labelled `CHUNK 1, BLOCK 0`; it is
  chunk 0 (the second example numbers from 0). A one-character fix.
- **Section 3.3's last PERMUTE**: the pseudocode permutes `m` after the seventh round too. That is
  harmless (`m` is not used again), and the Lean keeps it as written.

## The pull request, ready for review

`c2sp/pr/` holds it: `PULL_REQUEST.md` (its title and description) and
`0001-BLAKE3-add-a-Lean-specification.patch` (one commit on C2SP/C2SP main at a293183, with the
`BLAKE3:` prefix the manual asks for; a clone with the branch `BLAKE3-lean` is in
`/workspace/tmp/C2SP`). It adds `BLAKE3/` (the files of `c2sp/BLAKE3/`, identical) and one
paragraph in the appendix of `BLAKE3.md` pointing at it, in the form `kopis.md` uses. Nothing is
pushed or submitted.

Checked before saving it:

- C2SP's own Markdown lint (`.github/lint`, Go 1.27) passes on the changed `BLAKE3.md`.
- `check.py` passes inside the C2SP layout: the generated section, the appendix's 34 compressions
  and their round states, the 35 vectors in three modes, 7 theorems on the standard axioms, and all
  17 quotations of the document in the Lean's comments.
- Beyond the vectors: the Lean agrees with the official b3sum 1.8.2 on 63 more inputs (every chunk
  and block boundary up to 64 KiB and random lengths, all three modes, random keys and contexts,
  outputs of 1 to 300 bytes).
- Four deliberate errors are each caught (above).
- The English, read as each of the three readers: positive or neutral phrasing (the remaining
  negatives are the specification's own words, quoted), each term introduced before use, nothing a
  reader of the directory does not use.

The description asks the co-maintainers three things: the version for the change, whether the
Lean should settle disagreements with the prose (as `kopis.md` says of its Lean; the change keeps
`BLAKE3.md` the specification), and the first trace's chunk label.

## Submitting it

Push the branch to a fork of C2SP/C2SP and open the pull request with `PULL_REQUEST.md`'s title and
description. As a maintainer of BLAKE3 you can merge it once your co-maintainers agree.
