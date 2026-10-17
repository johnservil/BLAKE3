# b3sum

A command line utility for calculating
[BLAKE3](https://github.com/BLAKE3-team/BLAKE3) hashes, similar to
Coreutils tools like `b2sum` or `md5sum`: the servil fork's, on its
crate (the repository's [README](../README.md) says what the fork is).

```
Usage: b3sum [OPTIONS] [FILE]...

Arguments:
  [FILE]...  Files to hash, or checkfiles to check

Options:
      --keyed                 Use the keyed mode, reading the 32-byte key from stdin
      --derive-key <CONTEXT>  Use the key derivation mode, with the given context string
  -l, --length <LEN>          The number of output bytes, before hex encoding [default: 32]
      --seek <SEEK>           The starting output byte offset, before hex encoding [default: 0]
      --num-threads <NUM>     How many threads to hash on [default: one per CPU]
      --no-mmap               No effect: b3sum chooses between reading and mapping each file
      --no-names              Omit filenames in the output
      --raw                   Write raw output bytes to stdout, rather than hex
      --tag                   Output BSD-style checksums: BLAKE3 ([FILE]) = [HASH]
  -c, --check                 Read BLAKE3 sums from the [FILE]s and check them
      --quiet                 Skip printing OK for each checked file
  -h, --help                  Print help (see more with '--help')
  -V, --version               Print version
```

`--num-threads 1` runs b3sum on one thread, reading and hashing each file
in turn: for one b3sum per core, or to leave the other cores to other
work. With a larger N, up to N files hash at once, each spread over up to
N threads, so more than N threads can run together. To limit b3sum to
chosen CPUs, start it under `taskset -c LIST b3sum ...` (Linux).

See also [this document about how the `--check` flag
works](what_does_check_do.md).

# Example

Hash the file `foo.txt`:

```bash
b3sum foo.txt
```

Time hashing a gigabyte of data, to see how fast it is:

```bash
# Create a 1 GB file.
head -c 1000000000 /dev/zero > /tmp/bigfile
# Hash it with SHA-256.
time openssl sha256 /tmp/bigfile
# Hash it with BLAKE3.
time b3sum /tmp/bigfile
```


# Installation

With [Rust and
Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
installed, build this `b3sum` with:

```
cargo install --git https://github.com/johnservil/BLAKE3 b3sum
```

(`cargo install b3sum`, and the prebuilt binaries on [official BLAKE3's
releases page](https://github.com/BLAKE3-team/BLAKE3/releases), give
official BLAKE3's `b3sum`.)

On Linux for example, Cargo will put the compiled binary in
`~/.cargo/bin`. You might want to add that directory to your `$PATH`, or
`rustup` might have done it for you when you installed Cargo.

If you want to install directly from this directory, you can run `cargo
install --path .`. Or you can just build with `cargo build --release`,
which puts the binary at `./target/release/b3sum`.

On Linux, a statically linked `b3sum` starts sooner, skipping the dynamic
loader: about 100-160 µs less per process on an i7-12700K (a 4 KiB file
in 0.67-0.73 ms, from 0.77-0.89), which counts where `b3sum` runs once
per file. Build it so with your machine's target:

```
RUSTFLAGS="-C target-feature=+crt-static" cargo install --path . --target x86_64-unknown-linux-gnu
```

The `--target` keeps the flag off the build's procedural macros, which
cannot link statically.
