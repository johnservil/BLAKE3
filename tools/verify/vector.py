"""NEON instructions for aarch64.py. A vector register is four 32-bit
lane terms, lane 0 the least significant (the register's bits 31:0);
`.2d` operations pair lanes (0, 1) and (2, 3), `.16b` bitwise operations
act on the lanes alike. Each model follows the Arm architecture reference
manual's pseudocode for the arrangement the kernels use; others stop the
run."""

import re
from z3 import BitVecVal, Concat, Extract, RotateRight, LShR
import aarch64
from aarch64 import Unproved
import canon


def arrangement(op):
    m = re.match(r"v(\d+)\.(\w+)$", op)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def lanes(m, op):
    return m.v[int(re.match(r"[vqdsb](\d+)", op).group(1))]


def put(m, op, ls):
    assert len(ls) == 4 and all(l.size() == 32 for l in ls)
    m.v[int(re.match(r"[vqdsb](\d+)", op).group(1))] = list(ls)


def pairs(ls):
    """The .2d view: two (low lane, high lane) pairs."""
    return [(ls[0], ls[1]), (ls[2], ls[3])]


def xar_pair(lo, hi, k):
    """XAR's 64-bit rotate of hi:lo right by k. When the halves are one
    word A (the dup layout), rotr64(A:A, k) = rotr32(A, k mod 32) in each
    half, an identity; the halves' equality comes from canon, which is
    sound. Otherwise the 64-bit rotation itself."""
    if canon.canon(lo) == canon.canon(hi):
        r = RotateRight(lo, k % 32) if k % 32 else lo
        return r, r
    v = RotateRight(Concat(hi, lo), k) if k % 64 else Concat(hi, lo)
    return Extract(31, 0, v), Extract(63, 32, v)


def step(m, pc, mnem, ops):
    d, da = arrangement(ops[0]) if ops else (None, None)
    if mnem == "add" and da == "4s":
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        put(m, ops[0], [x + y for x, y in zip(a, b)])
        return True
    if mnem in ("eor", "orr", "and") and da == "16b":
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        f = {"eor": lambda x, y: x ^ y, "orr": lambda x, y: x | y, "and": lambda x, y: x & y}[mnem]
        put(m, ops[0], [f(x, y) for x, y in zip(a, b)])
        return True
    if mnem == "xar" and da == "2d":
        k = aarch64.imm(ops[3])
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        x = [p ^ q for p, q in zip(a, b)]
        out = []
        for lo, hi in pairs(x):
            out.extend(xar_pair(lo, hi, k))
        put(m, ops[0], out)
        return True
    if mnem in ("zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2") and da in ("4s", "2d"):
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        if da == "2d":
            A, B = pairs(a), pairs(b)
            if mnem == "zip1" or mnem == "trn1" or mnem == "uzp1":
                r = [A[0], B[0]]
            else:
                r = [A[1], B[1]]
            put(m, ops[0], [r[0][0], r[0][1], r[1][0], r[1][1]])
            return True
        if mnem == "zip1":
            r = [a[0], b[0], a[1], b[1]]
        elif mnem == "zip2":
            r = [a[2], b[2], a[3], b[3]]
        elif mnem == "uzp1":
            r = [a[0], a[2], b[0], b[2]]
        elif mnem == "uzp2":
            r = [a[1], a[3], b[1], b[3]]
        elif mnem == "trn1":
            r = [a[0], b[0], a[2], b[2]]
        else:
            r = [a[1], b[1], a[3], b[3]]
        put(m, ops[0], r)
        return True
    if mnem == "dup" and da == "4s":
        src = ops[1]
        e = re.match(r"v(\d+)\.s\[(\d)\]$", src)
        if e:
            x = m.v[int(e.group(1))][int(e.group(2))]
        else:
            x = m.reg(src)
            if isinstance(x, aarch64.Ptr):
                raise Unproved("a pointer duplicated into a vector")
        put(m, ops[0], [x] * 4)
        return True
    if mnem in ("shl", "ushr", "sri", "sli") and da == "4s":
        k = aarch64.imm(ops[2])
        n, old = lanes(m, ops[1]), lanes(m, ops[0])
        out = []
        for x, d_ in zip(n, old):
            if mnem == "shl":      # x << k
                out.append(Concat(Extract(31 - k, 0, x), BitVecVal(0, k)) if k else x)
            elif mnem == "ushr":   # x >> k
                out.append(Concat(BitVecVal(0, k), Extract(31, k, x)) if k else x)
            elif mnem == "sri":    # the top k bits of d kept, x >> k below them
                out.append(Concat(Extract(31, 32 - k, d_), Extract(31, k, x)))
            else:                  # sli: x << k above the low k bits of d
                out.append(Concat(Extract(31 - k, 0, x), Extract(k - 1, 0, d_)))
        put(m, ops[0], out)
        return True
    if mnem == "rev32" and da == "8h":
        # Each 32-bit lane's two halfwords swapped.
        put(m, ops[0], [Concat(Extract(15, 0, x), Extract(31, 16, x)) for x in lanes(m, ops[1])])
        return True
    if mnem == "tbl" and da == "16b":
        tables = re.findall(r"v(\d+)\.16b", ops[1])
        src = [m.v[int(t)] for t in tables]
        idx = lanes(m, ops[2])
        def byte(v, i):
            return Extract(8 * (i % 4) + 7, 8 * (i % 4), v[i // 4])
        out_bytes = []
        for i in range(16):
            k = aarch64.concrete(byte(idx, i))
            if k is None:
                raise Unproved("tbl with data-dependent indices")
            out_bytes.append(byte(src[k // 16], k % 16) if k < 16 * len(src) else BitVecVal(0, 8))
        put(m, ops[0], [Concat(*reversed(out_bytes[4 * l:4 * l + 4])) for l in range(4)])
        return True
    if mnem == "mov" and da == "16b":
        put(m, ops[0], lanes(m, ops[1]))
        return True
    if mnem in ("ldr", "str") and ops[0][0] in "qds":
        n = {"q": 16, "d": 8, "s": 4}[ops[0][0]]
        addr, wb, new = aarch64.mem_operand(m, ops, 1)
        if mnem == "ldr":
            ls = [m.load(addr + 4 * i, 4) for i in range(n // 4)]
            put(m, ops[0], ls + [BitVecVal(0, 32)] * (4 - len(ls)))
        else:
            ls = lanes(m, ops[0])
            for i in range(n // 4):
                m.store(addr + 4 * i, ls[i], 4)
        if wb:
            m.set(wb, new)
        return True
    if mnem in ("ldp", "stp") and ops[0][0] in "qd":
        n = {"q": 16, "d": 8}[ops[0][0]]
        addr, wb, new = aarch64.mem_operand(m, ops, 2)
        for j, r in enumerate(ops[:2]):
            a = addr + j * n
            if mnem == "ldp":
                ls = [m.load(a + 4 * i, 4) for i in range(n // 4)]
                put(m, r, ls + [BitVecVal(0, 32)] * (4 - len(ls)))
            else:
                ls = lanes(m, r)
                for i in range(n // 4):
                    m.store(a + 4 * i, ls[i], 4)
        if wb:
            m.set(wb, new)
        return True
    return False
