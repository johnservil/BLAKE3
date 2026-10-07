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
            self.m.v[i] = [BitVec(f"v{i}_{l}_in", 32) for l in range(4)]
        self.m.x["sp"] = Ptr("stack", STACK)
        self.m.x[30] = Ptr("return", 0)

    def run_and_check_abi(self):
        m = aarch64.run(self.m, self.entry)
        if not (isinstance(m.x[30], Ptr) and m.x[30].region == "return"):
            raise Unproved("returned through a changed x30")
        if not (isinstance(m.x["sp"], Ptr) and m.x["sp"].offset == STACK):
            raise Unproved(f"sp not restored: {m.x['sp']}")
        for i in range(19, 30):
            if isinstance(m.x[i], Ptr) or not same(m.x[i], BitVec(f"x{i}_in", 64)):
                raise Unproved(f"callee-saved x{i} changed")
        for i in range(8, 16):
            if not all(same(m.v[i][l], BitVec(f"v{i}_{l}_in", 32)) for l in (0, 1)):
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


def prove_table(obj, name, n, blocks, per_input_counter):
    """kernel(inputs, blocks, key, counter, packed, out) (the file's
    header): input i, at inputs[i], is `blocks` blocks hashed from key at
    counter + i (k kernels) or counter (p kernels); its chaining value goes
    to out[32 i..]. Flags as c1's."""
    inputs = [byte_symbols(f"in{i}_", 64 * blocks) for i in range(n)]
    key = byte_symbols("key", 32)
    regions = {f"input{i}": Region(f"input{i}", 64 * blocks, False, (lambda b: lambda o: b[o])(inputs[i])) for i in range(n)}
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    regions["out"] = Region("out", 32 * n, True, None)
    table = Region("table", 8 * n, False, None)
    for i in range(n):
        for j in range(8):
            table.bytes[8 * i + j] = (Ptr(f"input{i}", 0), j)
    regions["table"] = table
    s = Setup(obj, name, name + "_end", regions)
    counter = BitVec("counter", 64)
    flags, start, end, last_len = (BitVec(x, 8) for x in ("flags", "start", "end", "last_len"))
    m = s.m
    m.x[0], m.x[1], m.x[2], m.x[3] = Ptr("table", 0), BitVecVal(blocks, 64), Ptr("key", 0), counter
    m.x[4] = ZeroExt(32, Concat(last_len, end, start, flags))
    m.x[5] = Ptr("out", 0)
    s.run_and_check_abi()
    out = m.regions["out"]
    if any(i not in out.bytes for i in range(32 * n)):
        raise Unproved("the output is not wholly written")
    zero = BitVecVal(0, 8)
    for i in range(n):
        cv = spec.words(key)
        c = counter + i if per_input_counter else counter
        for b in range(blocks):
            block = spec.words(inputs[i][64 * b:64 * b + 64])
            f = ZeroExt(24, flags | (start if b == 0 else zero) | (end if b == blocks - 1 else zero))
            length = ZeroExt(24, last_len) if b == blocks - 1 else BitVecVal(64, 32)
            cv = spec.compress(cv, block, c, length, f)[:8]
        got = spec.words([out.bytes[32 * i + j] for j in range(32)])
        for w in range(8):
            if not same(got[w], cv[w]):
                raise Unproved(f"input {i}, output word {w} differs from the specification")
    return m.steps


def prove_partial(obj, n, pblocks):
    """q<n>(inputs, 16, key, counter, packed, out, partial): n whole chunks
    and a partial chunk of `pblocks` blocks (zero-padded) whose last block
    records last_len bytes, partial = pblocks | last_len << 8. Input i < n
    is chunk i at counter + i; the partial chunk, at counter + n, is the
    table's last input (q1's table is chunk, partial, chunk: its pair reads
    the chunk twice, and the third value is not part of the contract).
    The values go to out in that order."""
    chunks = [byte_symbols(f"in{i}_", 1024) for i in range(n)]
    part = byte_symbols("part", 64 * pblocks)
    key = byte_symbols("key", 32)
    regions = {f"input{i}": Region(f"input{i}", 1024, False, (lambda b: lambda o: b[o])(chunks[i])) for i in range(n)}
    regions["partial"] = Region("partial", len(part), False, lambda o: part[o])
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    slots = [f"input{i}" for i in range(n)] + ["partial"] if n > 1 else ["input0", "partial", "input0"]
    regions["out"] = Region("out", 32 * len(slots), True, None)
    table = Region("table", 8 * len(slots), False, None)
    for i, r in enumerate(slots):
        for j in range(8):
            table.bytes[8 * i + j] = (Ptr(r, 0), j)
    regions["table"] = table
    s = Setup(obj, f"blake3_hybrid_q{n}", f"blake3_hybrid_q{n}_end", regions)
    counter = BitVec("counter", 64)
    flags, start, end, last_len, plast = (BitVec(x, 8) for x in ("flags", "start", "end", "last_len", "plast"))
    m = s.m
    m.x[0], m.x[1], m.x[2], m.x[3] = Ptr("table", 0), BitVecVal(16, 64), Ptr("key", 0), counter
    m.x[4] = ZeroExt(32, Concat(last_len, end, start, flags))
    m.x[5] = Ptr("out", 0)
    m.x[6] = ZeroExt(48, Concat(plast, BitVecVal(pblocks, 8)))
    s.run_and_check_abi()
    out = m.regions["out"]
    if any(i not in out.bytes for i in range(32 * (n + 1))):
        raise Unproved("the output is not wholly written")
    zero = BitVecVal(0, 8)
    for i in range(n + 1):
        data, blocks, final = (chunks[i], 16, last_len) if i < n else (part, pblocks, plast)
        cv = spec.words(key)
        for b in range(blocks):
            block = spec.words(data[64 * b:64 * b + 64])
            f = ZeroExt(24, flags | (start if b == 0 else zero) | (end if b == blocks - 1 else zero))
            length = ZeroExt(24, final) if b == blocks - 1 else BitVecVal(64, 32)
            cv = spec.compress(cv, block, counter + i, length, f)[:8]
        got = spec.words([out.bytes[32 * i + j] for j in range(32)])
        for w in range(8):
            if not same(got[w], cv[w]):
                raise Unproved(f"input {i}, output word {w} differs from the specification")
    return m.steps


def cases(only):
    """Every (kernel, block count) case: c1 apart from and equal to key,
    the k and p kernels, and the q kernels at every partial chunk."""
    out = []
    for blocks in range(1, 17):
        for alias in (False, True):
            out.append(("c1", blocks, alias))
    for name in [f"k{n}" for n in range(2, 11)] + [f"p{n}" for n in (2, 3, 4, 5, 7, 8, 9)]:
        for blocks in (range(1, 17) if name[0] == "k" else [1]):
            out.append((name, blocks, False))
    for n in range(1, 10):
        for pblocks in range(1, 17):
            out.append((f"q{n}", pblocks, False))
    return [c for c in out if not only or c[0] in only]


OBJ = None


def prove_case(case):
    global OBJ
    if OBJ is None:
        OBJ = assemble()
    name, blocks, alias = case
    t = time.time()
    try:
        if name == "c1":
            steps = prove_c1(OBJ, blocks, alias)
        elif name[0] == "q":
            steps = prove_partial(OBJ, int(name[1:]), blocks)
        else:
            steps = prove_table(OBJ, f"blake3_hybrid_{name}", int(name[1:]), blocks, name[0] == "k")
    except Unproved as e:
        return case, f"NOT PROVED: {e}"
    what = f"partial chunk of {blocks} blocks" if name[0] == "q" else f"{blocks} blocks" + (", out = key" if alias else "")
    return case, f"proved ({what}, {steps} instructions, {time.time() - t:.1f} s)"


def main():
    import multiprocessing
    todo = cases(set(sys.argv[1:]))
    failed = 0
    with multiprocessing.Pool() as pool:
        for (name, _, _), verdict in pool.imap(prove_case, todo):
            print(f"{name}: {verdict}", flush=True)
            failed += verdict.startswith("NOT")
    print(f"{len(todo) - failed} of {len(todo)} cases proved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
