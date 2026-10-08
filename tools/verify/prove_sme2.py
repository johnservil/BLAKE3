"""Prove the SME2 kernels in c/blake3_sme2_aarch64.S equal to BLAKE3's
compression function (tools/verify/README.md), at the 512-bit streaming
vector length of Apple M4 (`cntw` = 16; the kernels return at once on any
other).

    python3 tools/verify/prove_sme2.py [KERNEL...]
    python3 tools/verify/prove_sme2.py every   # the chunk kernel at every group count (3.5 hours)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, Concat, ZeroExt, Extract, If, LShR, simplify
import aarch64
from aarch64 import Region, Ptr, Unproved
import prove_hybrid
from prove_hybrid import Setup, byte_symbols, same
import lean_spec as spec

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


def prove_chunks(obj, groups, stored, at, check=True):
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
    if not check:
        return m.steps
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


def prove_parents(obj, groups, check=True):
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
    if not check:
        return m.steps
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


def loop_head(obj, run, taken):
    """The head of the loop whose backward jump a concrete `run` takes
    `taken` times, the only such."""
    import collections
    jumps, last, step = collections.Counter(), [None], aarch64.step
    def trace(m, pc, mnem, ops):
        if last[0] is not None and pc < last[0]:
            jumps[pc] += 1
        last[0] = pc
        return step(m, pc, mnem, ops)
    aarch64.step = trace
    try:
        run()
    finally:
        aarch64.step = step
    heads = [pc for pc, n in jumps.items() if n == taken]
    if len(heads) != 1:
        raise Unproved(f"expected one loop taken {taken} times, found {sorted(map(hex, heads))}")
    return heads[0]


def prove_chunks_every(obj, stored):
    """hash16_chunks_512 at every number of groups from 2 (1 and 2 are the
    concrete cases), by induction over its group loop (induction.py): group
    K's lane l hashes the chunk table[16 K + l] points to, at counter + 16 K
    + l, into out[32 (16 K + l)..]. Each group's chunks are arbitrary (one
    set of symbols, since each step is proved for any)."""
    import induction
    import canon
    from z3 import ULE, UGE, ULT
    lanes = [byte_symbols(f"in{l}_", 1024) for l in range(16)]
    key = byte_symbols("key", 32)
    g, k = BitVec("len_groups", 64), BitVec("len_k", 64)
    # Group G's chunk for lane l lies at offset 1024 G of lane l's region.
    # Each group's chunks are arbitrary: every group reads the same symbols,
    # and a read is allowed only inside the current group's chunk (group 0
    # in the concrete runs to the loop's head, K in the induction step), so
    # a kernel that reads another group's table entry or chunk fails.
    current = [0]
    def chunk_byte(l):
        def byte(offset):
            m = aarch64._machine[0]
            group = offset // 1024 if isinstance(offset, int) else LShR(offset, 10)
            within = offset % 1024 if isinstance(offset, int) else induction.forced(m, offset & 1023)
            same_group = (group == current[0]) if isinstance(group, int) and isinstance(current[0], int) else \
                aarch64.holds(m, aarch64.as_bv(group) == aarch64.as_bv(current[0]))
            if within is None or not same_group:
                raise Unproved(f"a read of lane {l} outside the current group's chunk")
            return lanes[l][within]
        return byte
    regions = {}
    for l in range(16):
        regions[f"lane{l}"] = Region(f"lane{l}", g * 1024, False, chunk_byte(l))
        regions[f"lane{l}"].symbolic_initial = True
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    regions["out"] = Region("out", g * 512, True, None)
    def entry(offset):
        """Table byte `offset`: byte offset % 8 of the pointer to group
        offset / 128's chunk for lane (offset % 128) / 8."""
        if isinstance(offset, int):
            group, within = offset // 128, offset % 128
            base = 1024 * group
        else:
            within = induction.forced(aarch64._machine[0], offset & 127)
            if within is None:
                raise Unproved("a table read whose lane the lengths leave open")
            # From the entry's first byte, so all eight bytes name one pointer.
            first = simplify(aarch64.as_bv(offset) - within % 8, som=True, bv_sort_ac=True)
            base = simplify(LShR(first, 7) * 1024, som=True, bv_sort_ac=True)
        return (Ptr(f"lane{within // 8}", base), within % 8)
    table = Region("table", g * 128, False, entry)
    table.symbolic_initial = True
    regions["table"] = table
    s = SmeSetup(obj, "blake3_sme2_hash16_chunks_512", regions)
    m = s.m
    # The counter takes part in conditions (its carry), each decided by the solver.
    m.lengths = {"len_groups", "len_k", "counter"}
    m.assumptions = [UGE(g, 2), ULE(g, 1 << 40)]
    counter = BitVec("counter", 64)
    flags, start, end = (BitVec(x, 8) for x in ("flags", "start", "end"))
    m.x[0], m.x[1], m.x[2] = Ptr("table", 0), Ptr("key", 0), counter
    m.x[3] = Concat(BitVec("junk", 32), BitVecVal(stored, 8), end, start, flags)
    m.x[4], m.x[5] = Ptr("out", 0), g
    head = loop_head(obj, lambda: prove_chunks(obj, 3, 0, False, check=False), 2)
    zero = BitVecVal(0, 8)
    def want_group(kval, count=16):
        cvs = []
        for l in range(count):
            c = counter + kval * 16 + l
            cv = spec.words(key)
            for b in range(16):
                f = ZeroExt(24, flags | (start if b == 0 else zero) | (end if b == 15 else zero))
                cv = spec.compress(cv, spec.words(lanes[l][64 * b:64 * b + 64]), c, BitVecVal(64, 32), f)[:8]
            cvs.append(cv)
        return cvs
    def check_group(mach, kval, count=16):
        out = mach.regions["out"]
        keys = set()
        for l, cv in enumerate(want_group(kval, count)):
            offs = [aarch64.key(aarch64.as_bv(kval * 512 if not isinstance(kval, int) else BitVecVal(kval * 512, 64)) + 32 * l + j) for j in range(32)]
            keys.update(offs)
            if any(o not in out.bytes for o in offs):
                raise Unproved(f"group output {l} is not wholly written")
            got = spec.words([out.bytes[o] for o in offs])
            for w in range(8):
                if not same(got[w], cv[w]):
                    raise Unproved(f"group output {l}, word {w}, differs from the specification")
        if set(out.bytes) != keys:
            raise Unproved("the group writes output bytes outside its own")
    s0 = induction.run_to(m, s.entry, head)
    if s0.regions["out"].bytes:
        raise Unproved("output written before the group loop")
    s1 = induction.run_to(induction.snapshot(s0), head, head)
    check_group(s1, 0)
    gen = induction.Generalized(s0, s1, k)
    current[0] = k
    import prove_rust
    def check(endm, how):
        if how == "head":
            check_group(endm, k)
            return
        if not aarch64.holds(endm, g == k + 1):
            raise Unproved("a path leaves the group loop before the last group")
        if aarch64.concrete(endm.x[0]) != 16:
            raise Unproved(f"returned {endm.x[0]}, not 16")
        check_group(endm, k, stored or 16)
        prove_hybrid.check_abi(endm)
    kinds = prove_rust.loop_step(gen, s0, k, list(s0.assumptions) + [ULT(k, g), ULE(k, 1 << 39)], head, check)
    return f"{kinds['head']} paths back to the group loop's head, {kinds['return']} out of it"


def prove_parents_every(obj):
    """hash16_parents_512 at every number of groups from 2 to 2^40 (1 and 2
    are the concrete cases), by induction over its group loop: group K's
    parent i compresses pairs[1024 K + 64 i..] under key at counter
    (unchanged), into out[32 (16 K + i)..]. Each group's pairs are arbitrary:
    word w of group G's pairs is pair_w + G, so each group's words range
    over every value while the groups differ: a kernel that hashed another
    group's pairs (it preloads the next group's) would compute other terms.
    Between groups the words differ by 1, a constant, so the loop invariant
    holds the next group's words as pair_w + K + 1."""
    import induction
    from z3 import ULE, UGE, ULT
    words = [BitVec(f"pair{w}", 32) for w in range(256)]
    key = byte_symbols("key", 32)
    g, k = BitVec("len_groups", 64), BitVec("len_k", 64)
    def pair_byte(offset):
        m = aarch64._machine[0]
        if isinstance(offset, int):
            group, within = BitVecVal(offset // 1024, 32), offset % 1024
        else:
            within = induction.forced(m, aarch64.as_bv(offset) & 1023)
            if within is None:
                raise Unproved("a read of pairs whose place in its group the lengths leave open")
            # From the word's first byte, so its four bytes are one word's.
            first = simplify(aarch64.as_bv(offset) - within % 4, som=True, bv_sort_ac=True)
            group = simplify(Extract(31, 0, LShR(first, 10)), som=True, bv_sort_ac=True)
        word = simplify(words[within // 4] + group, som=True, bv_sort_ac=True)
        return Extract(8 * (within % 4) + 7, 8 * (within % 4), word)
    regions = {"pairs": Region("pairs", g * 1024, False, pair_byte),
               "key": Region("key", 32, False, lambda o: key[o]),
               "out": Region("out", g * 512, True, None)}
    regions["pairs"].symbolic_initial = True
    s = SmeSetup(obj, "blake3_sme2_hash16_parents_512", regions)
    m = s.m
    m.lengths = {"len_groups", "len_k"}
    m.assumptions = [UGE(g, 2), ULE(g, 1 << 40)]
    counter, flags = BitVec("counter", 64), BitVec("flags", 32)
    m.x[0], m.x[1], m.x[2] = Ptr("pairs", 0), Ptr("key", 0), counter
    m.x[3] = ZeroExt(32, flags)
    m.x[4], m.x[5] = Ptr("out", 0), g
    head = loop_head(obj, lambda: prove_parents(obj, 3, check=False), 2)
    def group_cvs(kval):
        group = BitVecVal(kval, 32) if isinstance(kval, int) else Extract(31, 0, kval)
        block = [simplify(w + group, som=True, bv_sort_ac=True) for w in words]
        return [spec.compress(spec.words(key), block[16 * i:16 * i + 16], counter, BitVecVal(64, 32), flags)[:8]
                for i in range(16)]
    def check_group(mach, kval):
        out = mach.regions["out"]
        keys = set()
        for i, cv in enumerate(group_cvs(kval)):
            base = BitVecVal(kval * 512, 64) if isinstance(kval, int) else aarch64.as_bv(kval * 512)
            offs = [aarch64.key(base + 32 * i + j) for j in range(32)]
            keys.update(offs)
            if any(o not in out.bytes for o in offs):
                raise Unproved(f"group output {i} is not wholly written")
            got = spec.words([out.bytes[o] for o in offs])
            for w in range(8):
                if not same(got[w], cv[w]):
                    raise Unproved(f"group output {i}, word {w}, differs from the specification")
        if set(out.bytes) != keys:
            raise Unproved("the group writes output bytes outside its own")
    # The group loop jumps back to the kernel's entry: the first arrival at
    # its head is the call itself.
    s0 = induction.snapshot(m) if head == s.entry else induction.run_to(m, s.entry, head)
    if s0.regions["out"].bytes:
        raise Unproved("output written before the group loop")
    s1 = induction.run_to(induction.snapshot(s0), head, head)
    check_group(s1, 0)
    gen = induction.Generalized(s0, s1, k)
    import prove_rust
    def check(endm, how):
        check_group(endm, k)
        if how == "return":
            if not aarch64.holds(endm, g == k + 1):
                raise Unproved("a path leaves the group loop before the last group")
            if aarch64.concrete(endm.x[0]) != 16:
                raise Unproved(f"returned {endm.x[0]}, not 16")
            prove_hybrid.check_abi(endm)
    kinds = prove_rust.loop_step(gen, s0, k, list(s0.assumptions) + [ULT(k, g), ULE(k, 1 << 39)], head, check)
    return f"{kinds['head']} paths back to the group loop's head, {kinds['return']} out of it"


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
    # Hours long: run when named.
    out += [("every", None, None), ("parents_every", None, None)]
    return [c for c in out if (c[0] in only if only else not c[0].endswith("every"))]


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
        elif name == "every":
            what = prove_chunks_every(OBJ, 0)
            return case, f"proved (the chunk kernel at every group count, the last group storing 16: {what}, {time.time() - t:.0f} s)"
        elif name == "parents_every":
            what = prove_parents_every(OBJ)
            return case, f"proved (the parent kernel at every group count: {what}, {time.time() - t:.0f} s)"
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
    with multiprocessing.Pool(min(4, multiprocessing.cpu_count()), maxtasksperchild=1) as pool:
        for (name, _, _), verdict in pool.imap_unordered(prove_case, todo):
            print(f"{name}: {verdict}", flush=True)
            failed += verdict.startswith("NOT")
    print(f"{len(todo) - failed} of {len(todo)} cases proved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
