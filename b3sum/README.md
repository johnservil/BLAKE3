# b3sum

A command-line tool that prints [BLAKE3](https://blake3.io) hashes of files or standard
input, like `sha256sum` or `md5sum`. This is the servil fork's `b3sum`, built on its crate
(the repository's [README](../README.md) says what the fork is).

<img src="https://devonjonte.github.io/bench-hashes/benchmark-results/12thGenIntelRCoreTMi712700K.linux70038generic/b3sum.chart.svg" alt="Hashing a tree of 1000 files on an i7-12700K: b3sum, multi-core and on one thread, beside sha256sum, sha1sum and md5sum, in GB/s" width="720">

# Installation

With [Rust and Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
installed:

```
cargo install --git https://github.com/johnservil/BLAKE3 b3sum
```

On Linux, build it statically linked instead: each run then starts in about two thirds
of the time, which matters where `b3sum` runs once per file. Give your machine's target
(`aarch64-unknown-linux-gnu` on 64-bit Arm):

```
RUSTFLAGS="-C target-feature=+crt-static" cargo install --git https://github.com/johnservil/BLAKE3 b3sum --target x86_64-unknown-linux-gnu
```

Cargo puts the binary in `~/.cargo/bin`. (`cargo install b3sum` and the prebuilt
binaries on [official BLAKE3's releases page](https://github.com/BLAKE3-team/BLAKE3/releases)
give official BLAKE3's `b3sum`.)

# Example

```bash
b3sum foo.txt
```

Time hashing a gigabyte, to see how fast it is on your machine:

```bash
head -c 1000000000 /dev/zero > /tmp/bigfile
time sha256sum /tmp/bigfile
time b3sum /tmp/bigfile
```

# Usage

```
Usage: b3sum [OPTIONS] [FILE]...

Arguments:
  [FILE]...  Files to hash, or checkfiles to check

Options:
      --keyed                 Use the keyed mode, reading the 32-byte key from stdin
      --derive-key <CONTEXT>  Use the key derivation mode, with the given context string
  -l, --length <LEN>          The number of output bytes, before hex encoding [default: 32]
      --seek <SEEK>           The starting output byte offset, before hex encoding [default: 0]
      --num-threads <NUM>     The most threads hashing at once [default: as fast as b3sum can]
      --no-mmap               No effect: b3sum chooses between reading and mapping each file
      --no-names              Omit filenames in the output
      --raw                   Write raw output bytes to stdout, rather than hex
      --tag                   Output BSD-style checksums: BLAKE3 ([FILE]) = [HASH]
  -c, --check                 Read BLAKE3 sums from the [FILE]s and check them
      --quiet                 Skip printing OK for each checked file
  -h, --help                  Print help (see more with '--help')
  -V, --version               Print version
```

By default b3sum uses every CPU: several files at once, and a large file over several
threads. `--num-threads 1` hashes on one thread, file by file, leaving the other cores to
other work (or to other b3sums). On Linux, `taskset -c LIST b3sum ...` runs it on chosen
CPUs.

[What `--check` does](what_does_check_do.md) explains checkfiles.
