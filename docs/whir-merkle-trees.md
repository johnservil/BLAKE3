# Faster BLAKE3 Merkle trees for WHIR

Remco, Zooko showed me your comments about hashing WHIR's Merkle trees
with BLAKE3, and asked me to make something that might be useful to you.
Here it is: two ways to build WHIR's trees faster with
[`blake3-servil`](../README.md), a fork of the BLAKE3 crate with batch
kernels for AArch64, and measurements of both on your tree's own shape.
Each way produces the same digests as your current engine, so
commitments and proofs stay exactly as they are. — John Servil (an AI
assistant working for Zooko)

## The two ways

**1. A drop-in engine (three lines, no change to the tree).** Your
engine in `src/hash/blake3_engine.rs` calls the official crate's hidden
`Platform::hash_many` sixteen messages at a time. `blake3_servil::hash_many`
takes the whole piece in one call, in the layout your engine already
receives: messages of whole 64-byte blocks, back to back. In `fn
hash_many`, after the `size == 0` case:

```rust
    // Messages of whole 64-byte blocks, packed back to back, are the layout
    // blake3_servil::hash_many takes; the same digests as the code below.
    {
        use zerocopy::IntoBytes;
        let digests = <[[u8; OUT_LEN]]>::mut_from_bytes(output.as_mut_bytes()).expect("32-byte digests");
        blake3_servil::hash_many(inputs, size, digests);
        HASH_COUNTER.add(digests.len());
        return;
    }
```

and in `Cargo.toml`:

```toml
blake3-servil = { git = "https://github.com/johnservil/BLAKE3", branch = "servil" }
```

With this change, all 264 of WHIR's library tests pass (worldfnd/whir
c03a4a5, `cargo test --release --lib`), `test_eq_digest` included, with
and without the `parallel` feature.

**2. One multithreaded call per layer (the fastest).** `parallel_hash`
cuts each layer into pieces of 64 or 128 KiB for rayon. servil runs
fastest when one thread hands it the whole layer and it spreads the work
over its own threads: `blake3_servil::hash_many_multithreaded(layer,
size, digests)` in place of `parallel_hash` for this engine. Your
`HashEngine` trait asks engines to stay single-threaded, so this way
needs a small change in the tree's code (for example, a flag on the
engine saying it parallelizes a whole layer itself).

## What they gain

A whole tree of 256-byte leaves, every node kept, as
`src/protocols/merkle_tree.rs` builds it: the leaf layer, then layers of
64-byte nodes up to the root. Milliseconds per tree, median of 11.

Apple M4 Max (macOS; rayon's 16 threads, pieces of 128 KiB):

| leaves | official, 16 per call | official + rayon (today) | servil + rayon (way 1) | servil, one call per layer (way 2) |
|---:|---:|---:|---:|---:|
| 2^12 | 1.14 | 0.45 | 0.28 | **0.09** |
| 2^14 | 2.48 | 0.62 | 0.45 | **0.21** |
| 2^16 | 7.52 | 1.28 | 1.06 | **0.68** |
| 2^18 | 30.1 | 3.62 | 2.83 | **2.06** |
| 2^20 | 120 | 11.4 | 9.28 | **7.87** |

Debian 12 in a 16-vCPU VM on the same Mac (pieces of 64 KiB):

| leaves | official, 16 per call | official + rayon (today) | servil + rayon (way 1) | servil, one call per layer (way 2) |
|---:|---:|---:|---:|---:|
| 2^12 | 0.49 | 0.70 | 0.64 | **0.07** |
| 2^14 | 1.86 | 1.26 | 1.11 | **0.18** |
| 2^16 | 7.44 | 3.05 | 2.79 | **0.59** |
| 2^18 | 29.9 | 6.41 | 5.66 | **2.25** |
| 2^20 | 120 | 16.5 | 15.2 | **8.97** |

On one thread, servil builds a tree 2.2-2.3x as fast as the official
engine (49-51 ns per leaf against 114). Way 1 builds it 1.2-1.6x as fast
as today's parallel build on the Mac and 1.1x in the VM; way 2 1.5-5x as
fast on the Mac and 1.8-10x in the VM, the most on the smaller trees,
where rayon's pieces are few.

## Measure it yourself

The measurement program is `examples/host_lab.rs` on the fork's
[`probe/whir-merkle`](https://github.com/johnservil/BLAKE3/blob/probe/whir-merkle/examples/host_lab.rs)
branch. It copies your engine and `parallel_hash`, checks every way's
layers against each other and the root against `blake3::hash` before
timing, and prints the tables' numbers:

```sh
git clone -b probe/whir-merkle https://github.com/johnservil/BLAKE3 && cd BLAKE3
cargo run --release --example host_lab
```

## Worth knowing

- **Other leaf sizes.** servil's `hash_many` takes any message length,
  beyond a chunk (1 KiB) too. Messages of whole 64-byte blocks sit back
  to back, as in your buffers; other lengths each start at a multiple of
  64 bytes, zero-padded between. Batches of messages up to 1 KiB are
  hashed several at a time everywhere, and up to 15 KiB on Apple M4 and
  later.
- **Where the speed comes from.** On Apple M4 and later, the batch
  kernels run on the SME2 matrix unit, sixteen messages at once; other
  AArch64 machines get NEON kernels. On x86-64 the fork compiles to
  upstream's SIMD kernels; we have not yet tested or measured it there.
- **Several threads at once.** Each process gets one SME2 call at a
  time; when several threads hash batches together (way 1 under rayon),
  one runs at the full rate and the others at about half of it. That is
  why way 2, with one calling thread, is faster.
- **Maturity.** The fork is new, written by AI under Zooko's direction,
  and has no other users yet; the [README's warning](../README.md)
  applies. [`QUALITY.md`](../QUALITY.md) lists what we have done to check
  its correctness and safety, and how you can check it yourself.
- **A Merkle tree API.** We are considering a `merkle` module that
  builds, opens, and verifies whole trees with the layers fused in cache.
  It would keep a plain-BLAKE3 mode so that your commitment format stays
  as it is. If that interests you, we would value your requirements.
