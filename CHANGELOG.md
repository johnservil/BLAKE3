# Changelog

## Unreleased

- `hash_each_with(mode, items, out)` hashes a collection of messages of any
  lengths in one call, short ones side by side in the SIMD lanes: on an
  Apple M4 Max a repository's objects or a store's files under 16 KiB take
  about a quarter less time than one `hash` call each.
- Verified streaming and range reads: `outboard_with` and
  `outboard_multithreaded_with` return a message's hash and its outboard,
  the parent nodes above its 16 KiB groups (the layout iroh-blobs stores),
  and `verify_range_with` checks any range of whole groups against the
  hash as it arrives. Building an outboard costs about what hashing costs.
- `Verifier` checks a whole message as it arrives in iroh-blobs' wire
  format (bao-tree's pre-order encoding with 16 KiB blocks), in pieces of
  any length, and gives back each group once it is checked.
- `b3sum` hashes on this crate's own worker threads instead of Rayon's.
  A file of 512 KiB or more already in the page cache is mapped and
  hashed in place; any other input, standard input included, is read
  4 MiB at a time while the threads hash the last piece. On an Apple M4 Max (`bench-hashes b3sum`) a
  1 GiB file in the page cache takes 37 ms (was 69; official b3sum 1.8.2,
  46), and one read from storage 159 ms (was 367). `--no-mmap` has no
  effect.
- The crate docs say what each call allocates ("Memory"): single-threaded
  calls nothing, multithreaded calls a list of their pieces (48 bytes for
  each 128 KiB of input), a queue up to the buffers you keep in flight, and
  once per process the self-test and the threads, which `initialize()`
  and `initialize_multithreaded()` move to start-up.
  `initialize_multithreaded()` now starts the queue's delivery thread too.
- Simpler: no thread budgets and no time-or-energy choice. A call's
  threading is in its name, and its one option is the mode. Each one-shot
  call has a full form that takes a `Mode`: `hash_with(mode, input)`,
  `hash_multithreaded_with`, `hash_many_with(mode, input, message_len,
  out)`, `hash_many_multithreaded_with`. `Threads` and `Efficiency` are
  gone; the queue's constructors take a mode and a handler
  (`Queue::messages(mode, handler)`, `Queue::pieces(mode, handler)`,
  `Queue::fixed(message_len, mode, handler)`), and a queue always hashes
  on every thread that pays. `b3sum --num-threads` is accepted and
  ignored, with a warning.
- One-shot calls of 1-64 KiB (`hash`, `keyed_hash`, `derive_key`, and
  `hash_multithreaded` below its split) that follow a pause fetch their
  code from memory in parallel instead of line by line: on an Apple M4 Max,
  a call after other work takes about 40% less time at 4 KiB, 45% at
  8 KiB, and 40% at 16 KiB (16 KiB now ahead of hardware SHA-256 there).
  Calls back to back are unchanged. M1-M3 Macs gain alike (4 KiB 26%
  less time, 8-16 KiB 35%, 32 KiB 23%). A `Hasher` used once per message
  (and the RustCrypto digest traits, which use it) gains the same: 4 KiB
  27% less time, 8 KiB 35%.
- Hashing an input much larger than the caches on one thread (`hash`,
  `Hasher::update`) keeps its speed: on an Apple M4 Max, 64-128 MiB take
  about 10% less time, at the rate of an 8 MiB input.
- `initialize_multithreaded` returns once every worker thread has
  started, and a warm `Queue` allocates nothing on macOS too (the standard
  library there allocates a lock at its first use, which the worker
  threads met after `initialize_multithreaded` had returned).
- Extended output (`OutputReader::fill`, `Hasher::finalize_xof`) runs
  sixteen blocks at a time on SME2 and eight on NEON: about 4.5x as fast
  on Apple M4 and later from 1 KiB (0.70 -> 0.155 ns per byte), and 2.3x
  on M1-M3 and other AArch64 CPUs (0.70 -> 0.30).
- `Hasher::update_reader` reads in 1 MiB pieces once a reader has more
  than 64 KiB: on an Apple M4 Max, hashing a file of 8-64 MiB in the page
  cache takes 23-33% less time.
- Builds with an assembler too old for the SHA-3 extension (GNU as before
  2.30) succeed, without the integer + NEON and SME2 kernels, with a
  warning; builds for `wasm32-wasip1` pass their tests (`Queue` needs
  threads, which that target lacks).

- `Hasher::update_multithreaded` is public: `Hasher::update` over several
  threads, with the same result, for updates of 512 KiB or more; shorter
  ones run on the calling thread. A long message read in shorter pieces
  hashes fastest through `Queue::pieces`.
- `Queue` is faster for a stream of inputs: on an Apple M4 Max about 2x for
  messages one after another and 2.5-4x for batches of 64-byte messages,
  with the program keeping enough in flight. Worker threads sleep as soon
  as they find nothing to take; nothing keeps running between calls.
- `hash_multithreaded` and `hash_many_multithreaded` leave the calling
  thread from 512 KiB (was 768 KiB): on an Apple M4 Max a 512 KiB input
  after a pause hashes about 40% faster.

## 0.3.0

- `Queue`: a stream of inputs hashed behind your program. You hand it your
  buffers and move on; each comes back hashed through a handler you write.
  `Queue::messages` takes separate inputs, `Queue::pieces` one long input
  in pieces, `Queue::fixed` messages of one length many per buffer. With
  `Efficiency::Time` the inputs in flight are hashed on several threads at
  once; with `Efficiency::Energy` on one.
- `Mode` (plain, keyed, key derivation) and `Threads` (one, all, a
  budget): `hash_with(mode, threads, input)` and
  `hash_many_with(mode, threads, input, message_len, out)` take both.
  Batches now hash in every mode at the same speed.
- `initialize()` runs the startup self-test alone (under 200 µs on an
  Apple M4 Max); `initialize_multithreaded()` also starts the worker
  threads (under 1 ms). A program that called `initialize()` to start the
  workers calls `initialize_multithreaded()` instead.
- Removed: `hash_multithreaded_with_budget` and
  `hash_many_multithreaded_with_budget`; use `hash_with` and
  `hash_many_with` with `Threads::Budget(n)`.
- The worker threads sleep whenever no call has work for them, and a
  multithreaded call leaves the calling thread from 768 KiB. A program
  that makes multithreaded calls back to back with no work between them
  runs them slower than 0.2.0 did (which kept the workers spinning between
  calls); a program that pauses between calls runs them faster, and none
  keeps a CPU busy while it waits.
