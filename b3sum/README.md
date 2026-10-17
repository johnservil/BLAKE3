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

`--num-threads N` hashes on at most N threads in all: several files at
once, or one file over several threads. By default b3sum hashes as many
files as there are CPUs at once, and a large one over every CPU besides:
more threads than CPUs, which keeps every core busy to the end (the bound
costs a tree with a few large files up to a third more time). `--num-threads 1` runs b3sum on
one thread, reading and hashing each file in turn: for one b3sum per
core, or to leave the other cores to other work. With more, a large file
read from storage, or stdin, has one more thread reading ahead, which
hashes nothing. To run b3sum on chosen CPUs, start it under
`taskset -c LIST b3sum ...` (Linux).

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

**On Linux, build it statically linked** (we recommend it, and bench-hashes
measures this build): it starts without the dynamic loader, so each run
costs about as little as starting a process at all. On an i7-12700K, a
64-byte file takes 0.49 ms a run against 0.74 ms dynamically linked, a
third less where `b3sum` runs once per file (`for f in *; do b3sum "$f";
done`). Give your machine's target (`aarch64-unknown-linux-gnu` on 64-bit
Arm):

```
RUSTFLAGS="-C target-feature=+crt-static" cargo install --git https://github.com/johnservil/BLAKE3 b3sum --target x86_64-unknown-linux-gnu
```

The `--target` keeps the flag off the build's procedural macros, which
cannot link statically. (macOS links its C library dynamically always;
its shared-library cache keeps start-up short there.)

(`cargo install b3sum`, and the prebuilt binaries on [official BLAKE3's
releases page](https://github.com/BLAKE3-team/BLAKE3/releases), give
official BLAKE3's `b3sum`.)

On Linux for example, Cargo will put the compiled binary in
`~/.cargo/bin`. You might want to add that directory to your `$PATH`, or
`rustup` might have done it for you when you installed Cargo.

If you want to install directly from this directory, you can run `cargo
install --path .` (with the flag and `--target` above, on Linux). Or you
can just build with `cargo build --release`, which puts the binary at
`./target/release/b3sum`.
