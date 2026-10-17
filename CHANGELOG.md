# Changelog

## Unreleased

- **Breaking: `initialize_multithreaded` takes the number of threads**
  the multithreaded calls hash on, `Option<NonZeroUsize>`: `None` for one
  a CPU, as before; `Some(1)` hashes on the calling thread alone, with no
  workers. Call it as `initialize_multithreaded(None)` to keep the old
  behavior.
- **Breaking: `Queue` is gone, and `stream()` takes its place** (Zooko,
  October 16, 2026). A stream borrows the program's messages (or has them
  written or read into its ring), writes each digest into the place the
  program names, and hands the places back in order to one handler.
  `Queue::messages` becomes `submit` or `submit_with`; `Queue::pieces`,
  `submit_reader` or `submit_with`; `Queue::fixed`, `submit_batch` with a
  place of one digest per message. The queue's delivery thread is gone.
- **One scheduler**: the pool's workers take every multithreaded call's
  work, and every stream's, from sources each call or stream publishes
  (a ring and two counters, no lock), and park each on its own token. On
  an i7-12700K against the stream on the earlier pool: no cell slower,
  messages 5-49% faster, collections 12-20%, `hash_multithreaded` on
  1 MiB 7%.
- **A C API**, `capi/b3.h`: a mode as a value (`b3_plain`, `b3_keyed`,
  `b3_derive_key`, the context hashed once), digests as values, the
  one-shot and batch calls on one thread or many, and a hasher in the
  caller's memory. It builds as `libb3_servil`, and compiles as C11 and C++11.
- **Breaking: the supported targets are 64-bit Arm (Linux, macOS),
  x86-64 (Linux, macOS, Windows with MSVC), and WebAssembly, with the
  latest stable Rust.** 32-bit x86 and Arm, big-endian CPUs, Windows with
  MinGW, the SSE2 kernel (x86 CPUs without SSE4.1), and older compilers
  go to official BLAKE3; with them go the `neon`, `prefer_intrinsics`,
  `no_sse2`, `no_sse41`, and `digest` features, and upstream's C library.
  GCC 12 is the oldest C compiler supported.
- **Breaking: the deprecated `guts` module goes** (`hazmat` serves it).
- `Mode::DeriveKeyMaterial` takes a derived key's context already hashed
  (`hazmat::hash_derive_key_context`), for many calls with one context.
- Each kernel report (`kernel_report` and its forms) is checked against
  the dispatch by a test, which found four wrong: the batch split counts
  the messages' slots (512 messages of 1000 bytes, not 524); with one CPU
  nothing splits; without the SHA-3 extension NEON runs batches of 64-byte
  messages from two and other whole blocks from three.
- The startup self-test checks, in a test on every run, that it reaches
  every assembly kernel this machine has; it gained two cases on AArch64
  (42), for the SME2 kernel at a counter per group and, on CPUs without
  SME2, the three-chunk hybrid kernel.
- The startup self-test checks the build and the machine, leaving what
  the code computes to the tests and proofs: on x86-64 and the other
  non-AArch64 targets it runs each kernel once in 9 small cases (from 40),
  28 us in a fresh process on an i7-12700K's performance core, from 109.
  On x86 it first stops a binary built for CPUs with LZCNT or BMI1 on a
  CPU without them, which runs those instructions as others and would
  compute wrong digests (seen under QEMU's Ivy Bridge).
- On x86-64 with SSE4.1 or AVX2, whole chunks and a partial one after
  them hash in one pass, the partial chunk in a lane the whole ones leave
  spare: three whole chunks in four lanes, five to seven in eight (AVX2).
  On an i7-12700K, hash() of 3839 bytes takes 1.55 us from 2.11 (x0.74),
  8191 bytes 1.76 from 3.31 (x0.53); after other work 3839 and 7935 bytes
  x0.90 and x0.85; other lengths level.
- `Verifier::update_multithreaded` checks a piece's groups over this
  crate's threads when the piece holds 512 KiB of them or more: a 64 MiB
  encoding verifies at 0.105-0.112 ns/B in 1-4 MiB pieces, from
  0.274-0.287 with `update` (i7-12700K); pieces under 512 KiB run as
  `update`.
- The scalar one-chunk kernel loads each step's four message words before
  its arithmetic and interleaves the step's four G functions, so in-order
  cores (Cortex-A53, as in the Raspberry Pi 3) no longer wait on each load:
  llvm-mca predicts a chunk-or-less message and every single compression
  about 3x faster there (unmeasured on the hardware); the same size of code,
  level on the M4 Max VM.
- The crate builds with any C compiler and CFLAGS: one whose assembler
  predates SME2 (GCC 12 with binutils 2.40, as on Debian 12) leaves out the
  SME2 kernel with a warning even when CFLAGS sets its own -march, such as
  -march=native, under which the build used to fail.
- **Breaking: every program calls `initialize()` (or
  `initialize_multithreaded()`) once, before any other call.** It runs the
  startup self-test and detects the CPU; a call before it panics, in every
  build, saying so. Each call then loads the detected platform instead of
  checking the self-test and detecting the CPU again, so the check costs
  nothing.
- **Breaking: a program that makes multithreaded calls or uses a `Queue`
  calls `initialize_multithreaded()` first**, in place of `initialize()`;
  such a call before it panics (one load per call). It starts the first
  worker and the queue's delivery thread and returns; the first worker
  starts the others, and calls run on the workers started so far. The
  first multithreaded call no longer starts the threads, nor waits for 19
  of them one after another: b3sum hashes a 1 MiB file in the page cache
  in 1.06 ms, from 1.41 (official 1.22; i7-12700K, Linux).
- On x86-64 with AVX2 or AVX-512, a message of 1089 to 2047 bytes (a whole
  chunk and a partial one, as a 1500-byte packet) hashes the two chunks'
  first blocks side by side, on the kernels' existing two-input path: the
  chain of dependent blocks at 1536 bytes falls from 25 to 18, at 2047 from
  33 to 18 (instructions per call under qemu: 1536 bytes 11577 -> 8281;
  unmeasured on x86 hardware).
- The API documentation is rewritten for choosing a call and predicting
  its speed: a table of which call fits which data, and each call's speed
  on an Apple M4 Max.
- b3sum hashes several files at once on several threads, printing their
  lines in order: a tree of 1000 files hashes 1.6-3.4x as fast on an
  Apple M4 Max (read from storage: 16 KiB files 5.90 -> 1.72 ns/B).
- The startup self-test runs in every build, `no_std` ones included
  (they skipped it): a build or CPU that computes wrong digests stops
  with a panic naming the code path, and threads hashing meanwhile stop
  too. It allocates nothing now (about 45 KiB of stack while it runs),
  and `initialize` is available without `std`.
- Calls made now and then, after other work or a pause, run faster: a
  message of a chunk or less hashes on a one-chunk kernel a quarter of
  the size (64 bytes about a third faster after other work on an Apple M4
  Max, 2-4% slower nonstop), and a batch of up to 8 one-block messages on
  the smallest batch kernel (4 messages after a pause 340 -> 161 ns each).
- A `Queue` keeps fewer workers awake and feeds each of them more: its
  long messages, 64-byte messages, and batches hash 15-45% faster (1 KiB
  messages 20% slower).
- A `Hasher` gathers updates shorter than 16 KiB until the message's next
  16 KiB boundary, and hashes them 16 chunks side by side: a message that
  arrives in network-sized pieces hashes nearly as fast as one in whole
  buffers. A `Hasher` is now about 18 KiB.
- `Hasher::update_each` takes a turn's pieces for many hashers in one call;
  the whole groups it gathers from different messages hash together. On
  an Apple M4 Max a server's 256 uploads in 1448-byte pieces hash at 0.25
  ns/B, against 0.51 one `update` per piece and SHA-256's 0.31.
- `hash_each_multithreaded_with` hashes a collection over this crate's
  threads: a repository's objects 7.8 times faster than on one thread.
- `Queue::messages`: messages shorter than 64 KiB share tasks, as batches
  do (pipelined 16 KiB messages a quarter faster).
- `hash_each_with(mode, items, out)` hashes a collection of messages of any
  lengths in one call, short ones side by side in the SIMD lanes: on an
  Apple M4 Max a repository's objects or a store's files under 16 KiB take
  about a quarter less time than one `hash` call each.
- Verified streaming and range reads: `outboard_with` and
  `outboard_multithreaded_with` return a message's hash and its outboard,
  the parent nodes above its 16 KiB groups (the layout iroh-blobs stores),
  and `verify_range_with` checks any range of whole groups against the
  hash as it arrives. Building an outboard costs about what hashing costs.
- Many messages in progress at once: `Hasher::update_each` and
  `Hasher::update_each_multithreaded` feed a turn's pieces to their
  hashers in one call, and `Hasher::finalize_each` gives the hashes of
  those that ended. `hash_each_multithreaded_with` is `hash_each_with`
  with this crate's threads allowed.
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
