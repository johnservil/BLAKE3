"""Prove the SME2 kernels in c/blake3_sme2_aarch64.S equal to BLAKE3's
compression function (tools/verify/README.md), at the 512-bit streaming
vector length of Apple M4 (`cntw` = 16; the kernels return at once on any
other).

    python3 tools/verify/prove_sme2.py [KERNEL...]
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, Concat, ZeroExt, Extract, If
import aarch64
from aarch64 import Region, Ptr, Unproved
import prove_hybrid
from prove_hybrid import Setup, byte_symbols, same
import spec

ROOT_DIR = prove_hybrid.ROOT_DIR
# An assembler that knows SME2 (Clang/LLVM 17 and later).
CLANG = next(c for c in ("clang-19", "clang-18", "clang-17", "clang") if shutil.which(c))


def assemble():
    obj = os.path.join(tempfile.mkdtemp(), "sme2.o")
    subprocess.run([CLANG, "-c", "-march=armv9-a+sme2", os.path.join(ROOT_DIR, "c/blake3_sme2_aarch64.S"), "-o", obj], check=True)
    return obj


class SmeSetup(Setup):
    def __init__(self, obj, entry, regions):
        syms = aarch64.symbols(obj)
        real = aarch64.disassemble
        aarch64.disassemble = lambda o, a, b: real(o, a, b, llvm=True)
        try:
            super().__init__(obj, entry, "blake3_sme2_text_end", regions, start_name="blake3_sme2_hash16_chunks_512")
        finally:
            aarch64.disassemble = real


def check_out(out, start, cvs):
    for i, cv in enumerate(cvs):
        if any(start + 32 * i + j not in out.bytes for j in range(32)):
            raise Unproved(f"output {i} is not wholly written")
        got = spec.words([out.bytes[start + 32 * i + j] for j in range(32)])
        for w in range(8):
            if not same(got[w], cv[w]):
                raise Unproved(f"output {i}, word {w} differs from the specification")


def prove_chunks(obj, groups, stored, at):
    """hash16_chunks_512(inputs, key, counter, flags, out, groups), or with
    `at` hash16_chunks_at_512 (x2 a table of a counter per group): 16
    chunks a group, lane l of group g at counter + 16 g + l (or
    counters[g] + l); flags = flags | start << 8 | end << 16 | stored << 24
    with anything above bit 31 (ignored); the last group stores its first
    `stored` values (0: all). out holds exactly what is stored."""
    n = 16 * groups
    chunks = [byte_symbols(f"in{i}_", 1024) for i in range(n)]
    key = byte_symbols("key", 32)
    regions = {f"input{i}": Region(f"input{i}", 1024, False, (lambda b: lambda o: b[o])(chunks[i])) for i in range(n)}
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    kept = 16 * (groups - 1) + (stored or 16)
    regions["out"] = Region("out", 32 * kept, True, None)
    table = Region("table", 8 * n, False, None)
    for i in range(n):
        for j in range(8):
            table.bytes[8 * i + j] = (Ptr(f"input{i}", 0), j)
    regions["table"] = table
    counters = [BitVec(f"counters{g}", 64) for g in range(groups)]
    if at:
        cb = [Extract(8 * j + 7, 8 * j, counters[g]) for g in range(groups) for j in range(8)]
        regions["counters"] = Region("counters", 8 * groups, False, lambda o: cb[o])
    s = SmeSetup(obj, "blake3_sme2_hash16_chunks_at_512" if at else "blake3_sme2_hash16_chunks_512", regions)
    counter = BitVec("counter", 64)
    flags, start, end = (BitVec(x, 8) for x in ("flags", "start", "end"))
    m = s.m
    m.x[0], m.x[1] = Ptr("table", 0), Ptr("key", 0)
    m.x[2] = Ptr("counters", 0) if at else counter
    m.x[3] = Concat(BitVec("junk", 32), BitVecVal(stored, 8), end, start, flags)
    m.x[4], m.x[5] = Ptr("out", 0), BitVecVal(groups, 64)
    s.run_and_check_abi()
    if aarch64.concrete(m.x[0]) != 16:
        raise Unproved(f"returned {m.x[0]}, not 16")
    zero = BitVecVal(0, 8)
    cvs = []
    for i in range(kept):
        g, lane = divmod(i, 16)
        c = (counters[g] if at else counter + 16 * g) + lane
        cv = spec.words(key)
        for b in range(16):
            f = ZeroExt(24, flags | (start if b == 0 else zero) | (end if b == 15 else zero))
            cv = spec.compress(cv, spec.words(chunks[i][64 * b:64 * b + 64]), c, BitVecVal(64, 32), f)[:8]
        cvs.append(cv)
    check_out(m.regions["out"], 0, cvs)
    return m.steps


def prove_messages(obj, groups, blocks, stored):
    """hash16_messages_512(inputs, key, counter, flags, out, groups,
    blocks): 16 messages a group of `blocks` whole blocks (2 to 16), every
    lane of group g at counter + g * step (step = flags bits 32..39), the
    last block recording last_len bytes (flags bits 40..47: any of 1 to 63,
    or 0 for 64, the contract's range, as a 6-bit symbol),
    start on block 0, end on the last; stored as the chunk kernel's."""
    n = 16 * groups
    msgs = [byte_symbols(f"in{i}_", 64 * blocks) for i in range(n)]
    key = byte_symbols("key", 32)
    regions = {f"input{i}": Region(f"input{i}", 64 * blocks, False, (lambda b: lambda o: b[o])(msgs[i])) for i in range(n)}
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    kept = 16 * (groups - 1) + (stored or 16)
    regions["out"] = Region("out", 32 * kept, True, None)
    table = Region("table", 8 * n, False, None)
    for i in range(n):
        for j in range(8):
            table.bytes[8 * i + j] = (Ptr(f"input{i}", 0), j)
    regions["table"] = table
    s = SmeSetup(obj, "blake3_sme2_hash16_messages_512", regions)
    counter = BitVec("counter", 64)
    flags, start, end, step = (BitVec(x, 8) for x in ("flags", "start", "end", "step"))
    last_len = BitVec("last_len", 6)
    m = s.m
    m.x[0], m.x[1], m.x[2] = Ptr("table", 0), Ptr("key", 0), counter
    m.x[3] = Concat(BitVecVal(0, 18), last_len, step, BitVecVal(stored, 8), end, start, flags)
    m.x[4], m.x[5], m.x[6] = Ptr("out", 0), BitVecVal(groups, 64), BitVecVal(blocks, 64)
    s.run_and_check_abi()
    if aarch64.concrete(m.x[0]) != 16:
        raise Unproved(f"returned {m.x[0]}, not 16")
    zero = BitVecVal(0, 8)
    cvs = []
    for i in range(kept):
        g = i // 16
        c = counter + ZeroExt(56, step) * g
        cv = spec.words(key)
        for b in range(blocks):
            f = ZeroExt(24, flags | (start if b == 0 else zero) | (end if b == blocks - 1 else zero))
            length = If(last_len == 0, BitVecVal(64, 32), ZeroExt(26, last_len)) if b == blocks - 1 else BitVecVal(64, 32)
            cv = spec.compress(cv, spec.words(msgs[i][64 * b:64 * b + 64]), c, length, f)[:8]
        cvs.append(cv)
    check_out(m.regions["out"], 0, cvs)
    return m.steps


def prove_parents(obj, groups):
    """hash16_parents_512(pairs, key, counter, flags, out, groups): parent
    i compresses pairs[64 i..] under key at counter (unchanged), block
    length 64, the 32-bit flags word."""
    n = 16 * groups
    pairs = byte_symbols("pair", 64 * n)
    key = byte_symbols("key", 32)
    regions = {"pairs": Region("pairs", 64 * n, False, lambda o: pairs[o]),
               "key": Region("key", 32, False, lambda o: key[o]),
               "out": Region("out", 32 * n, True, None)}
    s = SmeSetup(obj, "blake3_sme2_hash16_parents_512", regions)
    counter, flags = BitVec("counter", 64), BitVec("flags", 32)
    m = s.m
    m.x[0], m.x[1], m.x[2] = Ptr("pairs", 0), Ptr("key", 0), counter
    m.x[3] = ZeroExt(32, flags)
    m.x[4], m.x[5] = Ptr("out", 0), BitVecVal(groups, 64)
    s.run_and_check_abi()
    if aarch64.concrete(m.x[0]) != 16:
        raise Unproved(f"returned {m.x[0]}, not 16")
    cvs = [spec.compress(spec.words(key), spec.words(pairs[64 * i:64 * i + 64]), counter, BitVecVal(64, 32), flags)[:8] for i in range(n)]
    check_out(m.regions["out"], 0, cvs)
    return m.steps


def prove_xof(obj, groups):
    """xof16_512(cv, block, counter, flags_len, out, groups): output block
    i, the 16 words of the compression of `block` under cv at counter + i,
    flags the low byte of flags_len and block length its second."""
    cvb, block = byte_symbols("cv", 32), byte_symbols("block", 64)
    regions = {"cv": Region("cv", 32, False, lambda o: cvb[o]),
               "block": Region("block", 64, False, lambda o: block[o]),
               "out": Region("out", 1024 * groups, True, None)}
    s = SmeSetup(obj, "blake3_sme2_xof16_512", regions)
    counter, flags, length = BitVec("counter", 64), BitVec("flags", 8), BitVec("last_len", 8)
    m = s.m
    m.x[0], m.x[1], m.x[2] = Ptr("cv", 0), Ptr("block", 0), counter
    m.x[3] = ZeroExt(48, Concat(length, flags))
    m.x[4], m.x[5] = Ptr("out", 0), BitVecVal(groups, 64)
    s.run_and_check_abi()
    if aarch64.concrete(m.x[0]) != 16:
        raise Unproved(f"returned {m.x[0]}, not 16")
    out = m.regions["out"]
    for i in range(16 * groups):
        want = spec.compress(spec.words(cvb), spec.words(block), counter + i, ZeroExt(24, length), ZeroExt(24, flags))
        if any(64 * i + j not in out.bytes for j in range(64)):
            raise Unproved(f"output block {i} is not wholly written")
        got = spec.words([out.bytes[64 * i + j] for j in range(64)])
        for w in range(16):
            if not same(got[w], want[w]):
                raise Unproved(f"output block {i}, word {w} differs from the specification")
    return m.steps


def cases(only):
    out = []
    for groups in (1, 2):
        for stored in (0, 1, 15):
            out += [("chunks", groups, stored), ("chunks_at", groups, stored)]
    for blocks in range(2, 17):
        out.append(("messages", 1, (blocks, 0)))
    for stored in (1, 15):
        out.append(("messages", 1, (5, stored)))
    out.append(("messages", 2, (3, 7)))
    out += [("parents", 1, None), ("parents", 2, None), ("xof", 1, None), ("xof", 2, None)]
    return [c for c in out if not only or c[0] in only]


OBJ = None


def prove_case(case):
    global OBJ
    if OBJ is None:
        OBJ = assemble()
    name, groups, arg = case
    t = time.time()
    try:
        if name in ("chunks", "chunks_at"):
            steps = prove_chunks(OBJ, groups, arg, name == "chunks_at")
            what = f"{groups} groups, the last storing {arg or 16}"
        elif name == "messages":
            blocks, stored = arg
            steps = prove_messages(OBJ, groups, blocks, stored)
            what = f"{groups} groups of {blocks}-block messages, the last block of any length, the last group storing {stored or 16}"
        elif name == "parents":
            steps, what = prove_parents(OBJ, groups), f"{groups} groups"
        else:
            steps, what = prove_xof(OBJ, groups), f"{groups} groups"
    except Unproved as e:
        return case, f"NOT PROVED: {e}"
    return case, f"proved ({what}, {steps} instructions, {time.time() - t:.1f} s)"


def main():
    import multiprocessing
    todo = cases(set(sys.argv[1:]))
    failed = 0
    # A fresh process per case: the normal forms' caches keep their terms
    # alive (canon.py), so a worker that kept them would only grow.
    with multiprocessing.Pool(min(8, multiprocessing.cpu_count()), maxtasksperchild=1) as pool:
        for (name, _, _), verdict in pool.imap_unordered(prove_case, todo):
            print(f"{name}: {verdict}", flush=True)
            failed += verdict.startswith("NOT")
    print(f"{len(todo) - failed} of {len(todo)} cases proved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
