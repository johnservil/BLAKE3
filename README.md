# BLAKE3, the servil fork

## Warning

This code is new. An AI wrote it rapidly under Zooko's direction (his guidance is in
`AGENTS.md`), and it awaits its first line-by-line review by a person and its first audit.
It is checked with tests against the reference implementation and fixed digests,
sanitizers, Miri, formal proofs of its AArch64 assembly kernels, mutation testing, and a self-test that every process runs
before its first hash; [`QUALITY.md`](QUALITY.md) lists every check, the bugs they found,
and how to repeat them. Its speed is measured on an Apple M4 Max (and a Linux VM on it) and
on an Intel i7-12700K; its tests also run on other 64-bit Arm CPUs (down to a Raspberry Pi
4's kind), on x86-64 Linux and Windows, and on WebAssembly. You would be its first user.
Use it at your own risk.

## What it is

This fork of BLAKE3, written by John Servil (an AI) under Zooko's direction, hashes faster
on 64-bit Arm, above all on Apple M4-class chips, and on x86-64. It computes the same digests as
official BLAKE3, and adds calls for the ways programs hash: one large message over every
core (`hash_multithreaded`), batches and collections of messages (`hash_many`,
`hash_each_with`), many messages arriving at once (`Hasher::update_each`), a stream of
buffers hashed on other threads while your program reads the next (`Queue`), and verified
streaming in the format [iroh-blobs](https://github.com/n0-computer/iroh-blobs) stores
(`outboard_with`, `Verifier`).

It supports 64-bit Arm (Linux and macOS), x86-64 (Linux, macOS, and Windows with MSVC), and
WebAssembly, with the latest stable Rust. [Official
BLAKE3](https://github.com/BLAKE3-team/BLAKE3) serves other targets (32-bit x86 or Arm,
big-endian CPUs, Windows with MinGW) and older compilers.

[bench-hashes](https://github.com/johnservil/bench-hashes) measures it against other hashes
on your own machine; here are its [full results on an Apple M4
Max](https://johnservil.github.io/bench-hashes/benchmark-results/AppleM4Max.darwin25/bench-hashes.map.html).

## Using it

From Rust, add

```toml
[dependencies]
blake3-servil = { git = "https://github.com/johnservil/BLAKE3", branch = "servil" }
```

and call it as you would official BLAKE3's `blake3` crate, under its own name, after one call
of `initialize` (or `initialize_multithreaded`, for the multithreaded calls):

```rust
blake3_servil::initialize(); // once, before any other call (it runs the self-test)
let hash = blake3_servil::hash(b"foobarbaz");
let mut hasher = blake3_servil::Hasher::new();
hasher.update(b"foo");
hasher.update(b"barbaz");
assert_eq!(hasher.finalize(), hash);
```

The [crate documentation](https://johnservil.github.io/BLAKE3/blake3_servil/) opens with a
table of which call fits your data. `keyed_hash` (a MAC) and `derive_key` (a KDF) work as in
official BLAKE3, below.

From C, build the library in `capi/` (`cargo build --release` there gives `libb3.a` and
`libb3.so` or `libb3.dylib`), include [`capi/b3.h`](capi/b3.h), and link with `-lb3`:

```c
#include "b3.h"

b3_mode plain = b3_plain();
b3_digest digest = b3_hash(&plain, "foobarbaz", 9);
```

A mode (`b3_plain`, `b3_keyed`, `b3_derive_key`) says which of BLAKE3's three functions a
call computes. Beside `b3_hash` are `b3_hash_many` for batches of messages, a
`_multithreaded` form of each for other cores, and `b3_hasher` for a message in pieces,
held in your memory. `b3.h` documents each call.

The command-line `b3sum` prints the BLAKE3 hashes of files or of standard input:

```bash
cargo install --git https://github.com/johnservil/BLAKE3 b3sum
```

`CONTRIBUTING.md` says how to work on the fork.

## The BLAKE3 algorithm

BLAKE3 is a cryptographic hash function that is:

- **Much faster** than MD5, SHA-1, SHA-2, SHA-3, and BLAKE2.
- **Secure**, unlike MD5 and SHA-1. And secure against length extension,
  unlike SHA-2.
- **Highly parallelizable** across any number of threads and SIMD lanes,
  because it's a Merkle tree on the inside.
- Capable of **verified streaming** and **incremental updates**, again
  because it's a Merkle tree.
- A **PRF**, **MAC**, **KDF**, and **XOF**, as well as a regular hash.
- **One algorithm with no variants**, which is fast on x86-64 and also
  on smaller architectures.

For the original benchmarks and the design, see the
[BLAKE3 paper](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf).

<p align="center">
<img src="https://johnservil.github.io/bench-hashes/benchmark-results/AppleM4Max.darwin25/bench-hashes.chart.svg" alt="Hashing one 1 MiB message on an Apple M4 Max: BLAKE3 on every core and on one core beside SHA-256, SHA3-256, and SHA-1, in GB/s">
</p>

<p align="center">From <a href="https://github.com/johnservil/bench-hashes">bench-hashes</a>' latest record on an Apple M4 Max (<a href="https://johnservil.github.io/bench-hashes/benchmark-results/AppleM4Max.darwin25/bench-hashes.map.html">every size</a>; <a href="https://github.com/johnservil/bench-hashes/blob/main/METHODOLOGY.md">how it measures</a>)</p>

BLAKE3 is based on an optimized instance of the established hash
function [BLAKE2](https://blake2.net) and on the [original Bao tree
mode](https://github.com/oconnor663/bao/blob/master/docs/spec_0.9.1.md).
The specifications and design rationale are available in the [BLAKE3
paper](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf).
The default output size is 256 bits. The current version of
[Bao](https://github.com/oconnor663/bao) implements verified streaming
with BLAKE3.

Official BLAKE3 ([BLAKE3-team/BLAKE3](https://github.com/BLAKE3-team/BLAKE3)) includes the
`blake3` Rust crate, `b3sum`, a C implementation, the [Rust reference
implementation](reference_impl/reference_impl.rs) (also here: start there to see how BLAKE3
works), and [test vectors](test_vectors/test_vectors.json) covering extended outputs, all
three modes, and many input lengths (also here).

BLAKE3 was designed by:

* [@oconnor663] (Jack O'Connor)
* [@sneves] (Samuel Neves)
* [@veorq] (Jean-Philippe Aumasson)
* [@zookozcash] (Zooko)

The development of BLAKE3 was sponsored by [Electric Coin Company](https://electriccoin.co).

BLAKE3 is also [specified](https://c2sp.org/BLAKE3) in the [Community
Cryptography Specification Project (C2SP)](https://c2sp.org).

*NOTE: BLAKE3 is not a password hashing algorithm, because it's
designed to be fast, whereas password hashing should not be fast. If you
hash passwords to store the hashes or if you derive keys from passwords,
we recommend [Argon2](https://github.com/P-H-C/phc-winner-argon2).*

## The three modes

Besides `hash`, BLAKE3 provides two other modes, `keyed_hash` and
`derive_key`. The `keyed_hash` mode takes a 256-bit key:

```rust
// MAC an input all at once.
let example_key = [42u8; 32];
let mac1 = blake3_servil::keyed_hash(&example_key, b"example input");

// MAC incrementally.
let mut hasher = blake3_servil::Hasher::new_keyed(&example_key);
hasher.update(b"example input");
let mac2 = hasher.finalize();
assert_eq!(mac1, mac2);
```

The `derive_key` mode takes a context string and some key material (not a
password). The context string should be hardcoded, globally unique, and
application-specific. A good default format for the context string is
`"[application] [commit timestamp] [purpose]"`:

```rust
// Derive a couple of subkeys for different purposes.
const EMAIL_CONTEXT: &str = "BLAKE3 example 2020-01-07 17:10:44 email key";
const API_CONTEXT: &str = "BLAKE3 example 2020-01-07 17:11:21 API key";
let input_key_material = b"usually at least 32 random bytes, not a password";
let email_key = blake3_servil::derive_key(EMAIL_CONTEXT, input_key_material);
let api_key = blake3_servil::derive_key(API_CONTEXT, input_key_material);
assert_ne!(email_key, api_key);
```

## Other implementations

There are too many implementations out there for us to keep track of,
but some highlights include [an optimized Go
implementation](https://github.com/zeebo/blake3), [Wasm bindings for
Node.js and browsers](https://github.com/connor4312/blake3), [binary
wheels for Python](https://github.com/oconnor663/blake3-py), [.NET
bindings](https://github.com/xoofx/Blake3.NET), and [a pure Java
implementation](https://commons.apache.org/proper/commons-codec/apidocs/org/apache/commons/codec/digest/Blake3.html).

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Licenses

This work is released into the public domain with [CC0 1.0](./LICENSE_CC0).
Alternatively, it is licensed under any of the following:

* [Apache 2.0](./LICENSE_A2)
* [Apache 2.0 with LLVM exceptions](./LICENSE_A2LLVM)


## BLAKE3's adoption and deployment

* [Bazel](https://github.com/bazelbuild/bazel/releases/tag/6.4.0)
* [Cargo](https://github.com/rust-lang/cargo/pull/14137)
* [Ccache](https://ccache.dev/releasenotes.html#_ccache_4_0)
* [Chia](https://github.com/Chia-Network/chia-blockchain/blob/main/CHANGELOG.md#10beta8-aka-beta-18---2020-07-16)
* [Clickhouse](https://github.com/ClickHouse/ClickHouse/blob/master/rust/chcache/Cargo.toml#L7)
* [Farcaster](https://www.farcaster.xyz/)
* [IPFS](https://github.com/ipfs/go-verifcid/issues/13)
* [Iroh](https://www.iroh.computer/blog/blake3-hazmat-api)
* [LLVM](https://reviews.llvm.org/D121510)
* [Mold](https://arxiv.org/html/2608.23228v1)
* [Nix](https://github.com/NixOS/nix/pull/12379)
* [Nym](https://github.com/nymtech/nym/blob/59056a22c5e6b01a38da2124662bd1fa3c8abef2/common/nymsphinx/params/src/lib.rs#L5)
* [OpenZFS](https://github.com/openzfs/zfs/)
* [Redox](https://www.redox-os.org/news/pkgar-introduction/)
* [Solana](https://docs.rs/solana-program/1.9.5/solana_program/blake3/index.html)
* [Tekken 8](https://x.com/rodarmor/status/1751567502050771189)
* [Wasmer](https://github.com/wasmerio/wasmer/blob/4f935a8c162bf604df223003e434e4f7ca253688/lib/cache/src/hash.rs#L21)


## Miscellany

- [@veorq] and [@oconnor663] did [an interview with Cryptography FM](https://cryptography.fireside.fm/3).
- [@oconnor663] did [an interview with Saito](https://www.youtube.com/watch?v=cJkmIt7yN_E).

[@oconnor663]: https://github.com/oconnor663
[@sneves]: https://github.com/sneves
[@veorq]: https://github.com/veorq
[@zookozcash]: https://github.com/zookozcash
