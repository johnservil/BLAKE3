# How the speed chart was made

The README's speed chart comes from one run of [bench-hashes](https://github.com/johnservil/bench-hashes) on 2026-09-26, on an Apple M4 Max with 16 cores (12 performance, 4 efficiency). `tools/speed_chart.py` draws it, and this page, from the run's report, [`benchmark-results/AppleM4Max.darwin25/bench-hashes.result.txt`](https://github.com/johnservil/bench-hashes/blob/main/benchmark-results/AppleM4Max.darwin25/bench-hashes.result.txt), whose methodology is in bench-hashes' [METHODOLOGY.md](https://github.com/johnservil/bench-hashes/blob/main/METHODOLOGY.md). Each bar is the median time for one input of that size, taken over the run's rounds, as a speed. For SHA-256 the chart shows the fastest of three implementations in the run (sha2, ring, and Apple's CommonCrypto).

| bar | implementation | GB/s |
|---|---|---:|
| BLAKE3, every core | blake3-servil (this repository) at [01bc76e](https://github.com/johnservil/BLAKE3/commit/01bc76e757e64bc383114f5e37fd470b4d30b751), `hash_multithreaded` | 32 |
| BLAKE3, one core | blake3-servil (this repository) at [01bc76e](https://github.com/johnservil/BLAKE3/commit/01bc76e757e64bc383114f5e37fd470b4d30b751), `hash` | 6.76 |
| SHA-256 | ring 0.17.14 | 3.13 |
| SHA3-256 | sha3 0.11.0 | 1.03 |
| SHA-1 | sha1-checked 0.10.0 (SHA-1 with the collision detection git uses) | 0.77 |

- bench-hashes: commit [7cf12f3](https://github.com/johnservil/bench-hashes/commit/7cf12f3c8f09991593f2e161df45ebd3a9882b22)
- compiler: rustc 1.98.0-nightly (f428d123a 2026-06-19), target aarch64-apple-darwin
- load during the run: busy: other programs kept 0.74 CPUs busy on average, 3.14 in the busiest 5 s; some results may read slower than this machine can run

To draw the chart again from a newer record, with bench-hashes cloned inside this repository:

```sh
python3 tools/speed_chart.py bench-hashes/benchmark-results/AppleM4Max.darwin25 media
```
