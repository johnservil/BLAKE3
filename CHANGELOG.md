# blake3-servil 0.8, against official BLAKE3 1.8.7

The same digests. Faster on 64-bit Arm and x86-64. Every kernel is
assembly; every Arm kernel and the fork's own x86-64 kernels are proved
correct for every input, and upstream's x86-64 kernels are being proved.
Everything else is tested, and the tests are tested
([QUALITY.md](QUALITY.md)).

## Before you switch

1. Targets: 64-bit Arm on Linux and macOS, x86-64 on Linux and Windows
   with MSVC. Latest stable Rust and a C compiler, GCC 12 or Clang.
   Official BLAKE3 keeps 32-bit CPUs, big-endian CPUs, WebAssembly, MinGW
   and older compilers.
2. Call `blake3_servil::initialize()` once before any other call. A
   program that makes multithreaded calls or uses a stream calls
   `initialize_multithreaded(None)` instead. A call before it panics.
3. `keyed_hash(key, input)` is now `hash_with(Mode::Keyed(key), input)`.
   `derive_key(context, material)` is now
   `hash_with(Mode::DeriveKey(context), material)`. `Hasher::new_keyed`
   and `Hasher::new_derive_key` are now `Hasher::new_with(mode)`. `hash`
   and `Hasher::new` are unchanged.
4. Threads are the crate's own. `Hasher::update_rayon` is now
   `update_multithreaded`. `hash_multithreaded(input)` hashes one message
   over the threads. There is no `rayon` feature.
5. Gone: the `mmap` feature and `Hasher::update_mmap` (map the file
   yourself, or use `update_reader`). Gone: the `traits-preview` feature.
   Gone: the `guts`, `hazmat` and `platform` modules. Gone: the `no_avx2`,
   `no_avx512`, `no_neon` and `no_sme2` features. `pure` stays, and means
   portable Rust alone: no assembly, and no SIMD on x86-64 (official
   BLAKE3's `pure` keeps its Rust SSE4.1 and AVX2 kernels).
6. b3sum installs from the package `b3sums` (`cargo install b3sums`);
   crates.io's `b3sum` is official BLAKE3's. The program is still `b3sum`.
   `--num-threads N` means at most N threads in all, and 1 means one
   thread. `--no-mmap` does nothing. Every file is read to its end. A
   broken pipe ends b3sum quietly.

## New

- `hash_many` and `hash_many_multithreaded`: messages of one length in
  one buffer.
- `hash_each_with` and `hash_each_multithreaded_with`: a collection of any
  lengths.
- `Hasher::update_each`: many messages in progress at once.
- `stream()`: inputs one after another, hashed behind the caller.
- `outboard_with`, `verify_range_with` and `Verifier`: verified streaming
  in iroh-blobs' format.
- A C API in `capi/b3.h`. The library is `libb3_servil`. C11 and C++11.
- `Mode::DeriveKeyMaterial` with `hash_derive_key_context`: one context
  hashed once for many derivations.

## Different

- A startup self-test checks the build and the CPU. A wrong digest stops
  the program and names the code path.
- A panic on one of the crate's threads, or in a caller during a
  multithreaded call, ends the process.
- A `Hasher` is about 18 KiB. It gathers short updates and hashes them
  side by side, so a message arriving in small pieces hashes as fast as
  one in a buffer.
- The AVX2 and AVX-512 batch kernels read nothing past an input's end.
- The crate leaves Arm's data-independent timing (DIT) as the calling
  thread has it. A program that wants Arm's promise turns DIT on in its
  own threads before it hashes (the quality manual, "Timing and secrets":
  [docs/quality-manual.md](docs/quality-manual.md)).
- Building the Arm kernels needs an assembler that knows armv8.2-a+sha3,
  GNU as 2.30 or LLVM 7. Without one, the build warns and leaves them out.
