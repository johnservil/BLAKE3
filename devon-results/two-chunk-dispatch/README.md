# Short-dispatch isolation and additional assurance

Follow-up to [the first two-chunk record](../two-chunk-x86/README.md).
Measured hashing source: **d772be9**, baseline **c1a4e8b**. Additional
fixed-vector/Miri test: **82de0d3**, a test-only change. Intel i7-12700K,
Linux, Rust 1.98.1, generic builds; instrument and clocks unchanged from
the first record (`bd3acd0`, clocks source `8825450`).

## A structural improvement, with the original costs preserved

Disassembly of the first optimization showed the short batch dispatcher's
stack reservation grew from 448 to 512 bytes. Moving long-message hashing
out of line alone left 448 bytes: the empty-message hash still brought
its tree state into the common frame. A single helper now handles empty,
single and long-message cases on x86. The short dispatcher reserves **72
bytes**, and those other cases tail-jump to the helper. Its compiled
assembly is retained. This isolates the batch mechanism instead of padding
code to seek a favorable executable layout. ARM compiled branches retain
their original mechanisms; their performance remains untested here.

The earlier hypothesis that the four-at-a-time wrapper limits AVX2 bulk
batches to four lanes was ruled out by source inspection: it forwards all
complete groups to the platform, which already uses eight-wide groups.
The supported gain comes from filling SIMD lanes across two-chunk messages.

## Committed-source target repeats

Each CPU affinity block runs old/new/new/old, 32 samples of roughly 2 ms
per cell, with producer work outside timing, raw v4 samples and traces,
shared speed/share rules and same-build comparisons. All eight probe runs
were quiet (0.12-0.13 other CPUs average, 0.13-0.15 max window).

| Sixteen 2048-byte messages | Old ns/message | New ns/message | New/old time |
| --- | ---: | ---: | ---: |
| CPU 0, P core | 759.55 | 357.17 | 0.470 |
| CPU 16, E core | 1515.28 | 861.38 | 0.568 |

Both target cells have one speed, 100% share. Their same-build repeats
are within 0.1%. Batches of eight or more retain approximately **2.1x
throughput on the P core** and **1.76x on the E core**. Four-message gains
are smaller (time ratios 0.650/0.884). A one-message 64-byte batch pays
about 1.6% in a separate pinned exploratory block; larger 64-byte batches
are about level. Every measured cell's medians, splits and shares remain
in the comparison files. This target stays outside the frozen 64-byte axis.

## Frozen workload follow-up: costs remain under review

A clean 35-point, **48-round** old/new/new/old block retains all raw samples,
reports and comparisons. It completed in 42.4-43.3 seconds per run, quiet;
all **131 report cells in each run** match the samples. Its instrument
provenance and measured commits are clean. The previous record stays intact.

Selected solo fast-speed time ratios:

| Workload | New/old | Old/old repeat | New/new repeat |
| --- | ---: | ---: | ---: |
| Owned 1 KiB messages | 1.016 | 0.994 | 1.019 |
| Owned 64 KiB messages | 1.062 | 1.023 | 1.030 |
| Owned batches of 4096 | 1.126 | 1.024 | 0.834 |
| Lent 1 MiB MT | 1.121 | 1.196 | 0.968 |

The 1 KiB gap is smaller than in the first candidate's runs. Batch 4096
and MT cells still show substantial run-to-run variation and slower
candidate values in this broad workload. Small after-idle calls and speed
shares also vary. These findings remain open; neither the smaller frame
nor the diagnostic tool's passing verdict proves regression-free behavior.
Two x86 checks of the structural change pass after confirmation dismisses
initially slower cells; the test-only addition's check also passes.

Additional matched-capacity controls restrict the processes before pool
initialization to sixteen logical CPUs: P-only `0-15`, or mixed P/E
`0-11,16-19`. Each block uses the same three owned queue cells and 192
rounds, old/new/new/old. The 1 KiB ratios are 0.998 and 1.003, and batch
16 ratios 1.006 and 1.001. Batch 4096 reads 0.892/0.945 but its repeats
and speed shares swing (P-only new/new fast 1.161, mixed old/old 1.086),
so those cells supply sensitivity evidence rather than a new gain claim.
These narrower/affinity-controlled workloads differ from the broad block;
they leave the broad block's observed costs open. We can describe where
the findings occur, while their causal mechanism still needs isolation.

## Assurance

The final tree passes **79 library tests**, **18 API tests** (the parent
also runs the other 17 in a bounded one-CPU subprocess), one isolated
one-CPU test, one allocation test, and **22 doctests**. The structural
hashing source passes default/pure/no_sme2 and ASan suites, plus the two
published-vector tests. Logs are retained in `quality/`.

A new fixed anchor executes the batched two-chunk tree directly on the
portable platform, at four and seventeen messages, for hash/keyed/
derive-key modes. It uses the published 2048-byte input (`byte i = i % 251`)
and fixed first-32-byte digests copied from the published JSON. **Miri
passes this case** in a bounded pure build; SIMD/FFI are outside that Miri
verdict and remain covered by native backend, reference and guard-page
checks from the first record. Initial extra half-bytes in the literal
transcription failed decoding; the corrected literals were checked
against the published file before committing the test. No test silently
regenerates expected digests.

ASan uses `detect_leaks=0` for the ptrace restriction; its earlier positive
control caught a planted overflow. TSan's memory-layout startup limitation
remains. No Mac/ARM, thread-cycle, frequency or energy verdict is supplied.
Shared bootstrap bands remain excluded from inferential comparisons.

## Reproduce

Use the resolved lock retained in the first record for throwaway checkouts.
Build the probe at c1a4e8b and d772be9 separately, then run old/new/new/old
with `taskset -c 0` and `taskset -c 16`. The probe writes `clocks.csv` and
stdout v4 samples. Apply an external 30-second process-group deadline.
Use `tools/compare-runs.py` from the audited instrument with the fork's
`--rules tools/speeds.py`, as in the first record.

The frozen block uses the same 35 `POINTS_BY_USE_CASE` points as that record,
contenders `sha256,blake3-servil-st,blake3-servil-mt`, `--rounds 48` and
`--trace-clocks clocks.csv`, with 120-second external deadlines. Affinity
controls use `--rounds 192 --points 'continuous 1 KiB,continuous batch 16,continuous batch 4096'` and 60-second deadlines.

```sh
cargo test --release --locked --lib --tests
cargo test --release --locked --features pure --lib --tests
cargo test --release --locked --features no_sme2 --lib --tests
cargo test --release --locked --doc
cargo +nightly miri test --features pure --lib portable_two_chunk_batch_matches_published_vectors
```

Use an external deadline for each potentially hanging command. The existing
host_lab example's compile failure is reported as upstream issue #2; targeted
suites above avoid it. This candidate remains a draft for review, with a
supported workload-specific gain and explicit whole-instrument limitations.
