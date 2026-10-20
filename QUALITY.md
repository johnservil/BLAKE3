# How blake3-servil is checked

blake3-servil is a modified BLAKE3 implementation that aims to produce
exactly the official digests, faster.

Most of the extra speed comes from code written for specific processors
and from splitting work across processor cores. That is also where
mistakes are most likely, so the project checks the code in several
different ways.

## Proof of the fastest Arm code

The fastest parts are small routines called kernels, some of them
written in assembly.

For the Arm kernels, the project uses formal verification. Instead of testing selected inputs, a computer-checked
proof compares the actual machine instructions with a written
specification of BLAKE3 and shows that the result is correct for every
input. The proof also shows that a kernel stays within the memory it is
given and that the instructions it chooses depend on the input's length,
not on its contents.

This proof has a limited scope. It covers the Arm assembly kernels. It
does not cover the assembly used on Intel and AMD processors, or the
Rust code that prepares work and combines results.

## Tests, and tests of the tests

Everything else is tested.

The test suites compare outputs with published BLAKE3 examples, called
test vectors, and with a separate, simpler reference implementation.
They run these checks on the operating systems and processor types the
project supports.

The project also uses mutation testing to check the tests themselves.
It deliberately introduces one small mistake at a time — for example,
changing a comparison or skipping a step — and requires at least one
test to fail. If no test notices, the developers add a test or record why
that mistake is harmless.

Each program also runs a startup self-test. Before doing real work, it
hashes a few inputs with known answers. This can catch a broken build or
a processor that does not execute the code as expected.

## Memory and thread safety

Some performance-critical Rust code uses `unsafe`, which means the
programmer, rather than the compiler, takes responsibility for following
Rust's memory rules. Assembly code needs separate checks.

- **AddressSanitizer** rebuilds the Rust and C code with extra memory
  checks. It catches reads and writes outside allocated memory and use
  of memory after it has been released.
- **ThreadSanitizer** watches access from multiple threads. It catches
  data races, where threads access the same memory without proper
  coordination and at least one of them writes.
- **Miri** runs Rust code in an interpreter that enforces Rust's memory
  rules. It is slow, but it can catch mistakes in `unsafe` code that a
  normal test run may not expose.
- **Guard pages** cover assembly. A test places an inaccessible page of
  memory immediately beside an input or output buffer. If a kernel reads
  or writes even one byte outside that buffer, the processor faults and
  the test crashes.

These tools see different layers. None of the first three can inspect
assembly, which is why guard pages matter.

## What the checks found

The checks found eight bugs. All are fixed, and none is known to have
reached a user. Examples:

- an Arm kernel produced the wrong digest for very short messages;
- a function silently ignored a partial final block of input;
- a reading helper could lose bytes it had already read when a later
  operation failed;
- Miri found an unsafe pattern in code shared between threads;
- batch assembly on Intel and AMD processors read beyond the end of an
  input, which could crash a program. This bug came from upstream
  BLAKE3 and has been reported there.

The proofs have not found a bug in the Arm kernels. They did find
mistakes in some of the tools and models used to construct the proofs.

## What remains uncertain

No human has carefully reviewed every line, and nobody outside the
project has audited the code. The Intel and AMD assembly relies on tests
rather than proof, as does the Rust code around the proved Arm kernels.
These checks reduce risk, but they cannot guarantee that no bug remains.

## Run the checks

```sh
git clone -b servil https://github.com/johnservil/BLAKE3
cd BLAKE3
cargo test --release --lib --tests
```

The [quality manual](docs/quality-manual.md) gives details for experts
and maintainers, including the limits of each check and how to run the
proofs.
