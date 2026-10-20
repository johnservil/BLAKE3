# BLAKE3, the servil fork

A faster [BLAKE3](https://blake3.io) for 64-bit Arm and x86-64. It computes the same
digests as official BLAKE3.

## How fast

<img src="https://devonjonte.github.io/bench-hashes/benchmark-results/12thGenIntelRCoreTMi712700K.linux70038generic/bench-hashes.chart.svg" alt="Hashing one 128 MiB message after a pause on an i7-12700K: BLAKE3 multi-core and on one core beside SHA-256, SHA3-256 and SHA-1, in GB/s" width="720">

<img src="https://devonjonte.github.io/bench-hashes/benchmark-results/12thGenIntelRCoreTMi712700K.linux70038generic/b3sum.chart.svg" alt="Hashing a tree of 1000 files on an i7-12700K: b3sum, multi-core and on one thread, beside sha256sum, sha1sum and md5sum, in GB/s" width="720">

Speed depends on your machine and on how your program hashes.
[bench-hashes](https://github.com/johnservil/bench-hashes) measures both on your own
computer; its published maps cover an [Intel i7-12700K](https://devonjonte.github.io/bench-hashes/)
and an [Apple M4 Max](https://johnservil.github.io/bench-hashes/benchmark-results/AppleM4Max.darwin25/bench-hashes.map.html).

## How it is checked

**It is new.** An AI, John Servil, wrote it under Zooko's direction, and no human has
reviewed it line by line or audited it. You would be among its first users.

Its checks go further than most:

- The Arm assembly kernels are **proved correct** for all inputs, as the CPU runs them.
- **The tests are tested:** about 2,500 likely bugs are planted in the Rust code, one at a
  time, and a test must catch each one.
- Digests are checked against the official test vectors and an independent
  implementation, and each process checks its kernels before its first hash.
- Memory use is checked with sanitizers, Miri, and guard pages that catch a kernel reading
  one byte too far.

Details: [`QUALITY.md`](QUALITY.md).

## Using it

```toml
[dependencies]
blake3-servil = { git = "https://github.com/johnservil/BLAKE3", branch = "servil" }
```

```rust
blake3_servil::initialize(); // once, before any other call
let hash = blake3_servil::hash(b"foobarbaz");
```

The [crate documentation](https://johnservil.github.io/BLAKE3/blake3_servil/) covers the
faster calls for programs that hash a lot, and the C library in [`capi/`](capi/).
[`b3sum/README.md`](b3sum/README.md) covers the command-line tool.

It runs on 64-bit Arm (Linux, macOS) and x86-64 (Linux, Windows) with the latest stable
Rust; [official BLAKE3](https://github.com/BLAKE3-team/BLAKE3) serves other platforms.

## About BLAKE3

BLAKE3 is a cryptographic hash function: fast, secure, and parallel, serving as a hash,
MAC, KDF and XOF. It was designed by Jack O'Connor, Samuel Neves, Jean-Philippe Aumasson
and Zooko; the [paper](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf)
gives the design and [C2SP](https://c2sp.org/BLAKE3) the specification. It is not a
password hash; for passwords, use [Argon2](https://github.com/P-H-C/phc-winner-argon2).

Other implementations: [official BLAKE3](https://github.com/BLAKE3-team/BLAKE3) in Rust
and C, and this repository's [reference implementation](reference_impl/reference_impl.rs),
the clearest place to see how BLAKE3 works.

## Contributing and licenses

[CONTRIBUTING.md](CONTRIBUTING.md) says how to work on the fork. Released into the public
domain with [CC0 1.0](./LICENSE_CC0), or under [Apache 2.0](./LICENSE_A2) or [Apache 2.0
with LLVM exceptions](./LICENSE_A2LLVM).
