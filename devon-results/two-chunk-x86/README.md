# Two-chunk x86 batch experiment

Devon Jonte, October 1, 2026. Intel i7-12700K, Linux 7.0.0-34-generic,
Rust 1.98.1, generic target. Baseline `c1a4e8b`; optimized hashing source
`83e7e74`. The optimization fills x86 SIMD lanes across independent
2048-byte messages, batching their first chunks, second chunks (counter 1),
and separate parent roots. It adds no unsafe code and retains the algorithm,
mode flags and public outputs. ARM and other input lengths retain their paths.

## Supported gain: solo 2048-byte batches

`examples/x86_batch_probe.rs` uses shared clocks for 32 samples of roughly
2 ms each, with input production outside timing and every output observed.
Old/new/new/old runs retain raw v4 samples, starts, load and clock traces.
Analysis uses the shared samples/speeds rules, with old/old and new/new beside
old/new. All twelve published probe runs were quiet (0.11-0.13 other CPUs
average, maximum windows 0.13-0.15). Linux supplies no thread cycle counts.

| Affinity, 16 messages | Old ns/message | New ns/message | New/old time |
| --- | ---: | ---: | ---: |
| Default (CPUs 0-19 allowed) | 755.5 | 374.2 | 0.495 |
| CPU 0 (P core) | 759.5 | 357.1 | 0.470 |
| CPU 16 (E core) | 1519.0 | 862.3 | 0.568 |

These target cells have one speed (100% share) under the shared rule.
Eight to 128 messages show similar ratios: roughly 2.0-2.1x throughput
on the P core/default and 1.75-1.78x on the E core. CPU identity comes
from OS topology, not the absent cycle counter. No energy claim follows.
Four/six-message batches gain less: CPU 0 time ratios 0.650/0.758,
CPU 16 0.889/0.911. One/three-message batches retain their old path.

Pinned target same-build repeats are close (the 16-message cell within
0.1%); unpinned new/new target medians move about 5-7%. The roughly 50%
time reduction comfortably exceeds that observed spread. A separate earlier
four-run exploratory block also supported the gain; it remains local with
its source patch. Published numbers above come from committed source.
This gain is specific to 2048-byte batches, outside bench-hashes' frozen
64-byte batch axis. Adjacent 2047/2049-byte messages remain controls, and
every cell's speeds and shares are in the comparisons.

## Regression diagnostics and remaining slowdowns

Two x86 `perf_regress.py check` executions pass (four alternating pairs
apiece, no held cell remaining). Their logs are in `quality/`. The tool
curtails undecided points and deletes temporary raw files. A separately
retained 35-point, 24-round old/new/new/old benchmark block therefore
provides inspectable raw samples, clean provenance and exact report checks
(all 131 cells on each side/run agree). Instrument: audited bench-hashes
`bd3acd0`; clocks: unchanged from `8825450`, shared across sides. Both
versions of the hashing library include John's one-CPU fix.

**A passing diagnostic check does not establish regression-free behavior.**
The retained block and a reverse-order 48-round six-point focus show slower
queue cells on the candidate binaries, even though those calls hash 64-byte
messages or individual 1 KiB messages and never enter the new two-chunk path:

- Owned 1 KiB messages, solo: about +7.4-7.5% fast-speed time in both blocks.
- Owned batches of 4096, solo: +11.1% fast speed in the broad block; +16.2%
  in the focus. Focus new/new differs by 13.1%, old/old by 3.7%.
- Lent 1 MiB MT: +10.6% broad, with new/new differing by 14.7%; +6.5% focus.
- Small after-idle calls move substantially; solo ST 64 B fast speed is
  +71.3% broad and +12.8% focus, with changes in speeds and their shares.

These slowdowns remain open. Treat this as a correctness-assured optimization
candidate for review, with its target gain established and whole-instrument
costs requiring explanation before promotion. The streaming-first policy
keeps these cells visible. Their cause could involve code placement or
process scheduling; the measurements alone do not establish it.

A same-code fresh-copy control (`two-chunk-layout-control`) builds baseline
`c1a4e8b` twice in different tool side paths, same compiler and instrument.
Their executable hashes and Rust symbol hashes/addresses differ (for
`hash_many_on`, 0xc6fd0 versus 0xc7050). Its queue 1 KiB original/original
repeat moves +8.1%; copy/copy +7.3%. Batch 4096 slow speed changes +22.4%
between builds. These controls demonstrate sensitivity in unchanged code;
they leave the candidate's observed slowdowns open rather than erase them.
Shared bootstrap bands are excluded from inferential verdicts throughout.

## Correctness evidence

- Default and pure: 78 library tests, 18 API tests (including a fresh
  one-CPU subprocess running the other 17), one isolated one-CPU test,
  one warmed allocation test; all pass on final source.
- no_sme2: 78 library and then-current 16 API tests pass; added published
  two-chunk and eight-caller tests subsequently pass in default/pure/ASan.
- 22 doctests and both published-vector tests pass.
- New independent-reference test covers every byte alignment 0-63,
  portable/SSE2/SSE4.1/AVX2 and AVX-512 when available, all three modes,
  counts around lane/group/table boundaries, and output sentinels.
- New API tests exercise the published 2048-byte vectors in batches,
  eight concurrent callers, all thread budgets, and queue delivery at
  2048-byte batch sizes. Existing guard-page tests include these lengths,
  inputs and outputs flush against inaccessible pages, and MT forms.
- A deliberately wrong second-chunk counter fails the new reference test.
  The mutation was restored before the final tests and source commit.
- AddressSanitizer with rebuilt standard library passes final lib/API/
  queue suites. A planted heap overflow is detected. Leak checking is
  disabled because LeakSanitizer requires ptrace unavailable here.
  Assembly is not instrumented; guard pages complement these checks.
- ThreadSanitizer builds but cannot start: incompatible process memory
  layout in this environment. This supplies no race-sanitizer verdict.

Plain `cargo test` also tries an existing broken `host_lab` example which
imports the removed private `lanes::probe`; targeted suites above bypass
that pre-existing compile failure. Mac and ARM64/SME2 remain untested.

## Reproduction

Obtain the fork's candidate branch and copy this record's resolved lock
into each throwaway checkout as `Cargo.lock` before using `--locked`.
Build the exact old and new commits in separate directories and preserve
the executables:

```sh
cargo build --release --locked --example x86_batch_probe
./target/release/examples/x86_batch_probe COMMIT > samples.tsv
```

Run old/new/new/old from isolated directories (the probe writes clocks.csv).
For topology controls prefix the probe with `taskset -c 0` or `taskset -c 16`.
Use an external 30-second process-group deadline for each probe. Our binary
hashes are retained for artifact identification; builds need not reproduce
identical bytes across checkout paths, as the fresh-copy control illustrates.

```sh
python3 /path/to/bench-hashes/tools/compare-runs.py old-1/samples.tsv new-1/samples.tsv new-2/samples.tsv old-2/samples.tsv --rules /path/to/BLAKE3/tools/speeds.py
cargo test --release --locked --lib --tests
cargo test --release --locked --features pure --lib --tests
cargo test --release --locked --features no_sme2 --lib --tests
cargo test --release --locked --doc
cargo test --release --manifest-path test_vectors/Cargo.toml
```

The frozen diagnostic block uses the 35 points in
`tools/perf_regress.py::POINTS_BY_USE_CASE`, selected with the tool's
`argument()` names, contenders `sha256,blake3-servil-st,blake3-servil-mt`,
`--rounds 24 --trace-clocks clocks.csv`. The focus/control use 48 rounds
and points `continuous 1 KiB,continuous batch 16,continuous batch 4096,lent 1 MiB,64 B,idle 64 B`.
Original full graphs, guides, supervisor logs and frozen clock traces remain
in the local evidence archive; this publication retains their raw samples,
reports and comparisons. The earlier benchmark full record remains separate.
