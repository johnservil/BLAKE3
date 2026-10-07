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


def loop_head(lib, entry, function):
    """The head of `function`'s loop: the target of its one backward jump that
    a run of 24 blocks takes twice."""
    import collections
    start = lib.syms[function]
    later = [a for a in lib.syms.values() if a > start]
    end = min(later) if later else start + (1 << 20)
    jumps = collections.Counter()
    step = aarch64.step
    last = [None]
    def trace(m, pc, mnem, ops):
        if last[0] is not None and pc < last[0] and start <= pc < end:
            jumps[pc] += 1
        last[0] = pc
        return step(m, pc, mnem, ops)
    aarch64.step = trace
    try:
        prove_xof(lib, entry, 24)
    finally:
        aarch64.step = step
    heads = [pc for pc, n in jumps.items() if n == 2]
    if len(heads) != 1:
        raise Unproved(f"{function}: expected one loop taken twice, found {sorted(map(hex, heads))}")
    return heads[0]


def prove_xof_every(lib, entry, function, per_iteration, from_count):
    """The extended output at every count from `from_count` blocks, by
    induction over the loop of `function`, which writes `per_iteration`
    blocks an iteration (induction.py). Counts below `from_count` are the
    concrete cases."""
    import induction
    from z3 import ULE, UGE, BitVecVal
    head = loop_head(lib, entry, function)
    cvb, block, length, counter, flags = inputs()
    n, k = BitVec("len_n", 64), BitVec("len_k", 64)
    regions = {"cv": Region("cv", 32, False, lambda o: cvb[o]), "block": Region("block", 64, False, lambda o: block[o]),
               "out": Region("out", n * 64, True, None)}
    m = machine(lib, regions)
    m.lengths = {"len_n", "len_k"}
    m.assumptions = [UGE(n, from_count), ULE(n, 1 << 57)]
    m.x[0], m.x[1], m.x[2], m.x[3], m.x[4], m.x[5], m.x[6] = (Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, length), counter,
                                                              ZeroExt(56, flags), Ptr("out", 0), n)
    s0 = induction.run_to(m, lib.syms[entry], head)
    if s0.regions["out"].bytes:
        raise Unproved("output written before the loop")
    s1 = induction.run_to(induction.snapshot(s0), head, head)
    check_blocks(s1, 0, per_iteration, cvb, block, length, counter, flags, BitVecVal(0, 64))
    g = induction.Generalized(s0, s1, k)
    mk = g.machine(s0, k)
    mk.assumptions = list(s0.assumptions) + [ULE(k * per_iteration + per_iteration, n), ULE(k, 1 << 54)]
    ends = induction.paths(mk, head, {head})
    kinds = {"head": 0, "return": 0}
    import canon
    for end, how in ends:
        kinds[how] += 1
        canon.set_context(end.assumptions, end.lengths)
        if how == "head":
            expect_state(end, g, k + 1)
            check_blocks(end, 0, per_iteration, cvb, block, length, counter, flags, k * per_iteration, exact=True)
        else:
            rest = next((r for r in range(per_iteration) if aarch64.holds(end, n == k * per_iteration + per_iteration + r)), None)
            if rest is None:
                raise Unproved("a path leaves the loop with a count the proof does not determine")
            check_blocks(end, 0, per_iteration + rest, cvb, block, length, counter, flags, k * per_iteration, exact=True)
            for i in range(19, 30):
                if isinstance(end.x[i], Ptr) or not same(end.x[i], BitVec(f"x{i}_in", 64)):
                    raise Unproved(f"callee-saved x{i} changed")
    canon.set_context([], set())
    return f"{kinds['head']} paths back to the loop's head, {kinds['return']} out of it"


def check_blocks(m, first, count, cvb, block, length, counter, flags, base_block, exact=False):
    """Output blocks base_block + first .. + count - 1 are the specification's
    at counter + their index; with `exact`, the only bytes written."""
    out = m.regions["out"]
    want_keys = set()
    for i in range(first, first + count):
        idx = base_block + i
        words = want(cvb, block, length, counter + idx, flags)
        offs = [aarch64.key(aarch64.as_bv(idx * 64 if not isinstance(idx, int) else BitVecVal(idx * 64, 64)) + j) for j in range(64)]
        want_keys.update(offs)
        if any(o not in out.bytes for o in offs):
            raise Unproved(f"output block {i} of the iteration is not wholly written")
        got = spec.words([out.bytes[o] for o in offs])
        for w, (gw, x) in enumerate(zip(got, words)):
            if not same(gw, x):
                raise Unproved(f"output block {i} of the iteration, word {w}, differs from the specification")
    if exact and set(out.bytes) != want_keys:
        raise Unproved("the iteration writes output bytes outside its blocks")


def expect_state(m, g, kval):
    """The machine is the generalized state at iteration `kval`."""
    inst = g.at(kval)
    def agree(a, b):
        if b is None:
            return True
        if isinstance(a, tuple) or isinstance(b, tuple):
            # A byte of a stored pointer.
            return (isinstance(a, tuple) and isinstance(b, tuple) and a[1] == b[1]
                    and agree(a[0], b[0]))
        if isinstance(b, Ptr) or isinstance(a, Ptr):
            return (isinstance(a, Ptr) and isinstance(b, Ptr) and a.region == b.region
                    and aarch64.holds(m, aarch64.as_bv(a.offset) == aarch64.as_bv(b.offset)))
        if same(a, b):
            return True
        return aarch64.lengths_only(m, a - b) and aarch64.holds(m, a == b)
    for r, gv in g.x.items():
        if not agree(m.x[r], inst(gv)):
            raise Unproved(f"x{r} at the loop's head differs from the next iteration's")
    for r, lanes in g.v.items():
        for l, gv in enumerate(lanes):
            if not agree(m.v[r][l], inst(gv)):
                raise Unproved(f"v{r} lane {l} at the loop's head differs from the next iteration's")
    for name, mem in g.mem.items():
        if name == "out":
            continue
        region = m.regions[name]
        for off, b in mem.items():
            if b is None:
                continue
            if isinstance(b, tuple) and b[0] == "word":
                continue
            if off not in region.bytes or not agree(region.bytes[off], b):
                raise Unproved(f"{name}+{off} at the loop's head differs from the next iteration's")
        for off in {o - b[2] for o, b in mem.items() if isinstance(b, tuple) and b[0] == "word"}:
            w = induction.word(region, off)
            if w is None or not agree(w, inst(mem[off][1])):
                raise Unproved(f"{name}+{off} (a word) at the loop's head differs from the next iteration's")


def main(exit=True):
    lib = Library(build())
    cases = [("portable compression in place", lambda: prove_in_place(lib, "verify_portable_compress_in_place")),
             ("portable extended output block", lambda: prove_xof(lib, "verify_portable_compress_xof", 0)),
             ("NEON platform's compression in place", lambda: prove_in_place(lib, "verify_neon_compress_in_place"))]
    cases += [(f"NEON platform's extended output, {n} blocks", (lambda n: lambda: prove_xof(lib, "verify_neon_xof_many", n))(n))
              for n in range(1, 21)]
    cases.append(("NEON platform's extended output, every count from 16 blocks (induction over its loop)",
                  lambda: prove_xof_every(lib, "verify_neon_xof_many",
                                          "_RNvNtCs5y7Y5DyUngN_13blake3_servil8neon_xof8xof_many", 8, 16)))
    failed = 0
    for what, prove in cases:
        t = time.time()
        try:
            result = prove()
            detail = f"{result} instructions" if isinstance(result, int) else result
            print(f"{what}: proved ({detail}, {time.time() - t:.1f} s)", flush=True)
        except Unproved as e:
            print(f"{what}: NOT PROVED: {e}", flush=True)
            failed += 1
    print(f"{len(cases) - failed} of {len(cases)} cases proved")
    if exit:
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
