"""Prove the hybrid kernels in c/blake3_neon_hybrid_aarch64.S equal to
BLAKE3's compression function, for every input, key, counter, and flags,
at every block count they take (tools/verify/README.md).

    python3 tools/verify/prove_hybrid.py [KERNEL...]

Assembles the file, disassembles each kernel, and runs it symbolically
(aarch64.py). For each block count it checks that the output equals the
specification's (spec.py), that every access stays inside the buffers the
kernel's contract gives it and inside its own stack frame, that the path
depends only on the block count, and that the callee-saved registers come
back unchanged. Exits nonzero on the first failure.
"""

import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, Concat, Extract, ZeroExt, simplify, eq, Solver, unsat
import aarch64
from aarch64 import Machine, Region, Ptr, Unproved
import spec
import canon

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STACK = 1 << 16


def assemble():
    obj = os.path.join(tempfile.mkdtemp(), "hybrid.o")
    subprocess.run(["cc", "-c", os.path.join(ROOT_DIR, "c/blake3_neon_hybrid_aarch64.S"), "-o", obj], check=True)
    return obj


def same(a, b):
    """Whether two terms are equal for every value of their symbols: by
    normal form (canon.py); two small terms with no rounds in them, by the
    solver."""
    if canon.canon(a) == canon.canon(b):
        return True
    if canon.has_arith(a) or canon.has_arith(b):
        return False     # the normal forms differ: no search over the rounds
    s = Solver()
    s.set("timeout", 120_000)
    s.add(a != b)
    return s.check() == unsat


class Setup:
    """A machine at a kernel's entry: symbolic junk in every register, the
    stack, the return address, and the regions the caller gives."""
    def __init__(self, obj, entry_name, stop_name, regions):
        syms = aarch64.symbols(obj)
        text = aarch64.text_bytes(obj)
        self.entry = syms[entry_name]
        code = aarch64.disassemble(obj, syms[entry_name], syms[stop_name])
        regions = dict(regions)
        regions["text"] = Region("text", max(text) + 1, False, lambda o: BitVecVal(text[o], 8))
        regions["stack"] = Region("stack", STACK, True, None)
        self.m = Machine(code, regions)
        for i in range(31):
            self.m.x[i] = BitVec(f"x{i}_in", 64)
        for i in range(32):
            self.m.v[i] = BitVec(f"v{i}_in", 128)
        self.m.x["sp"] = Ptr("stack", STACK)
        self.m.x[30] = Ptr("return", 0)

    def run_and_check_abi(self):
        m = aarch64.run(self.m, self.entry)
        if not (isinstance(m.x[30], Ptr) and m.x[30].region == "return"):
            raise Unproved("returned through a changed x30")
        if not (isinstance(m.x["sp"], Ptr) and m.x["sp"].offset == STACK):
            raise Unproved(f"sp not restored: {m.x['sp']}")
        for i in range(19, 30):
            if not same(m.x[i], BitVec(f"x{i}_in", 64)) if not isinstance(m.x[i], Ptr) else True:
                raise Unproved(f"callee-saved x{i} changed")
        for i in range(8, 16):
            if not same(Extract(63, 0, m.v[i]), Extract(63, 0, BitVec(f"v{i}_in", 128))):
                raise Unproved(f"callee-saved d{i} changed")
        return m


def byte_symbols(name, n):
    return [BitVec(f"{name}{i}", 8) for i in range(n)]


def prove_c1(obj, blocks, alias):
    """c1(input, blocks, key, counter, packed, out, last): the chaining
    value of `blocks` blocks, the first blocks-1 from `input`, the last
    from `last`, every block at `counter`, the first with flags | start,
    the last with flags | end and block length last_len (packed =
    flags | start << 8 | end << 16 | last_len << 24); out may be key."""
    inp = byte_symbols("in", 64 * (blocks - 1))
    last = byte_symbols("last", 64)
    key = byte_symbols("key", 32)
    regions = {
        "input": Region("input", len(inp), False, lambda o: inp[o]),
        "last": Region("last", 64, False, lambda o: last[o]),
        "key": Region("key", 32, alias, lambda o: key[o]),
    }
    if not alias:
        regions["out"] = Region("out", 32, True, None)
    s = Setup(obj, "blake3_hybrid_c1", "blake3_hybrid_k1_end", regions)
    counter = BitVec("counter", 64)
    flags, start, end, last_len = (BitVec(n, 8) for n in ("flags", "start", "end", "last_len"))
    m = s.m
    m.x[0], m.x[1], m.x[2], m.x[3] = Ptr("input", 0), BitVecVal(blocks, 64), Ptr("key", 0), counter
    m.x[4] = ZeroExt(32, Concat(last_len, end, start, flags))
    m.x[5] = Ptr("key" if alias else "out", 0)
    m.x[6] = Ptr("last", 0)
    s.run_and_check_abi()
    cv = spec.words(key)
    for i in range(blocks):
        block = spec.words(inp[64 * i:64 * i + 64] if i < blocks - 1 else last)
        f = ZeroExt(24, flags | (start if i == 0 else BitVecVal(0, 8)) | (end if i == blocks - 1 else BitVecVal(0, 8)))
        length = ZeroExt(24, last_len) if i == blocks - 1 else BitVecVal(64, 32)
        cv = spec.compress(cv, block, counter, length, f)[:8]
    out = m.regions["key" if alias else "out"]
    if any(i not in out.bytes for i in range(32)):
        raise Unproved("the output is not wholly written")
    got = spec.words([out.bytes[i] for i in range(32)])
    for i in range(8):
        if not same(got[i], cv[i]):
            raise Unproved(f"output word {i} differs from the specification")
    return m.steps


def main():
    obj = assemble()
    for blocks in range(1, 17):
        for alias in (False, True):
            t = time.time()
            steps = prove_c1(obj, blocks, alias)
            print(f"c1, {blocks:2} blocks{', out = key' if alias else ''}: proved ({steps} instructions, {time.time() - t:.1f} s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Unproved as e:
        print(f"NOT PROVED: {e}")
        sys.exit(1)
