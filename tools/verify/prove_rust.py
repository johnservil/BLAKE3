"""Prove blake3-servil's Rust compression paths equal to the Lean
specification's compression (tools/verify/README.md): each path as the
compiler built it for AArch64, the Rust and any assembly it calls run
together by the symbolic executor.

    python3 tools/verify/prove_rust.py

The paths (tools/verify/rust/src/lib.rs, each a call of the library's own
`Platform` method): the portable compression in place and its extended
output, the NEON platform's compression in place (Rust around the scalar
kernel), and the NEON platform's extended output of 1 to 20 blocks
(Rust NEON intrinsics, four blocks at a time, and the portable code for
the rest). Each runs on symbolic chaining values, blocks, lengths,
counters, and flags.
"""

import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, Concat, ZeroExt
import aarch64
from aarch64 import Machine, Region, Ptr, Unproved
import lean_spec as spec
from prove_hybrid import same, byte_symbols

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "/tmp/target/verify-rust"
STACK = 1 << 20


def build():
    subprocess.run(["cargo", "build", "-q", "--release", "--manifest-path", os.path.join(HERE, "rust", "Cargo.toml")],
                   env={**os.environ, "CARGO_TARGET_DIR": TARGET}, check=True)
    return os.path.join(TARGET, "release", "libverify_entries.so")


class Library:
    """The built library: its code, its image, and its PLT's entries."""
    def __init__(self, so):
        self.so = so
        self.syms = aarch64.symbols(so)
        self.code = aarch64.disassemble(so, 0, 1 << 30)
        self.image = aarch64.image_bytes(so)
        listing = subprocess.run(["objdump", "-d", so], capture_output=True, text=True, check=True).stdout
        self.plt = {int(a, 16): n for a, n in re.findall(r"^([0-9a-f]+) <(\w+)@plt>:", listing, re.M)}


def machine(lib, regions):
    image = lib.image
    regions = dict(regions)
    for name in ("image", "text"):
        regions[name] = Region(name, max(image) + 1, False, lambda o: BitVecVal(image.get(o, 0), 8))
    regions["stack"] = Region("stack", STACK, True, None)
    m = Machine(lib.code, regions)
    m.plt = lib.plt
    for i in range(31):
        m.x[i] = BitVec(f"x{i}_in", 64)
    for i in range(32):
        m.v[i] = [BitVec(f"v{i}_{l}_in", 32) for l in range(16)]
    m.x["sp"], m.x[30] = Ptr("stack", STACK), Ptr("return", 0)
    return m


def run(lib, m, entry):
    aarch64.run(m, lib.syms[entry])
    for i in range(19, 30):
        if isinstance(m.x[i], Ptr) or not same(m.x[i], BitVec(f"x{i}_in", 64)):
            raise Unproved(f"callee-saved x{i} changed")
    if not (isinstance(m.x["sp"], Ptr) and m.x["sp"].offset == STACK):
        raise Unproved("sp not restored")


def inputs():
    cvb, block = byte_symbols("cv", 32), byte_symbols("block", 64)
    return cvb, block, BitVec("last_len", 8), BitVec("counter", 64), BitVec("flags", 8)


def want(cvb, block, length, counter, flags):
    return spec.compress(spec.words(cvb), spec.words(block), counter, ZeroExt(24, length), ZeroExt(24, flags))


def compare(region, start, words):
    if any(start + j not in region.bytes for j in range(4 * len(words))):
        raise Unproved("the output is not wholly written")
    got = spec.words([region.bytes[start + j] for j in range(4 * len(words))])
    for w, (g, x) in enumerate(zip(got, words)):
        if not same(g, x):
            raise Unproved(f"output word {start // 4 + w} differs from the specification")


def prove_in_place(lib, entry):
    cvb, block, length, counter, flags = inputs()
    m = machine(lib, {"cv": Region("cv", 32, True, lambda o: cvb[o]), "block": Region("block", 64, False, lambda o: block[o])})
    m.x[0], m.x[1], m.x[2], m.x[3], m.x[4] = Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, length), counter, ZeroExt(56, flags)
    run(lib, m, entry)
    compare(m.regions["cv"], 0, want(cvb, block, length, counter, flags)[:8])
    return m.steps


def prove_xof(lib, entry, blocks):
    cvb, block, length, counter, flags = inputs()
    regions = {"cv": Region("cv", 32, False, lambda o: cvb[o]), "block": Region("block", 64, False, lambda o: block[o]),
               "out": Region("out", 64 * max(blocks, 1), True, None)}
    m = machine(lib, regions)
    m.x[0], m.x[1], m.x[2], m.x[3], m.x[4], m.x[5] = Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, length), counter, ZeroExt(56, flags), Ptr("out", 0)
    if blocks:
        m.x[6] = BitVecVal(blocks, 64)
    run(lib, m, entry)
    for i in range(max(blocks, 1)):
        compare(m.regions["out"], 64 * i, want(cvb, block, length, counter + i, flags))
    return m.steps


def main(exit=True):
    lib = Library(build())
    cases = [("portable compression in place", lambda: prove_in_place(lib, "verify_portable_compress_in_place")),
             ("portable extended output block", lambda: prove_xof(lib, "verify_portable_compress_xof", 0)),
             ("NEON platform's compression in place", lambda: prove_in_place(lib, "verify_neon_compress_in_place"))]
    cases += [(f"NEON platform's extended output, {n} blocks", (lambda n: lambda: prove_xof(lib, "verify_neon_xof_many", n))(n))
              for n in range(1, 21)]
    failed = 0
    for what, prove in cases:
        t = time.time()
        try:
            steps = prove()
            print(f"{what}: proved ({steps} instructions, {time.time() - t:.1f} s)", flush=True)
        except Unproved as e:
            print(f"{what}: NOT PROVED: {e}", flush=True)
            failed += 1
    print(f"{len(cases) - failed} of {len(cases)} cases proved")
    if exit:
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
