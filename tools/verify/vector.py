"""NEON instructions for aarch64.py. A vector register is four 32-bit
lane terms, lane 0 the least significant (the register's bits 31:0);
`.2d` operations pair lanes (0, 1) and (2, 3), `.16b` bitwise operations
act on the lanes alike. Each model follows the Arm architecture reference
manual's pseudocode for the arrangement the kernels use; others stop the
run."""

import re
from z3 import BitVecVal, Concat, Extract, RotateRight, LShR, is_app_of, is_bv_value, Z3_OP_CONCAT
import aarch64
from aarch64 import Unproved
import canon


def arrangement(op):
    m = re.match(r"v(\d+)\.(\w+)$", op)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def lanes(m, op):
    """The NEON register: the low four lanes of its Z register."""
    return m.v[int(re.match(r"[vqdsb](\d+)", op).group(1))][:4]


def put(m, op, ls):
    """A NEON write: four lanes, the Z register's bits above them zeroed."""
    assert len(ls) == 4 and all(l.size() == 32 for l in ls)
    m.v[int(re.match(r"[vqdsb](\d+)", op).group(1))] = list(ls) + [BitVecVal(0, 32)] * 12


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
    mnem = {"ldur": "ldr", "stur": "str"}.get(mnem, mnem)
    d, da = arrangement(ops[0]) if ops else (None, None)
    if mnem == "add" and da == "4s":
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        put(m, ops[0], [x + y for x, y in zip(a, b)])
        return True
    if mnem == "add" and da == "2d":
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        out = []
        for (alo, ahi), (blo, bhi) in zip(pairs(a), pairs(b)):
            v = Concat(ahi, alo) + Concat(bhi, blo)
            out += [Extract(31, 0, v), Extract(63, 32, v)]
        put(m, ops[0], out)
        return True
    if mnem == "addhn" and da == "2s":
        # The high halves of the 64-bit sums, narrowed into the low 64 bits;
        # the rest of the register cleared.
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        highs = [Extract(63, 32, Concat(ahi, alo) + Concat(bhi, blo)) for (alo, ahi), (blo, bhi) in zip(pairs(a), pairs(b))]
        put(m, ops[0], highs + [BitVecVal(0, 32)] * 2)
        return True
    if mnem == "addhn2" and da == "4s":
        # As addhn, into the upper 64 bits; the lower kept.
        a, b = lanes(m, ops[1]), lanes(m, ops[2])
        highs = [Extract(63, 32, Concat(ahi, alo) + Concat(bhi, blo)) for (alo, ahi), (blo, bhi) in zip(pairs(a), pairs(b))]
        put(m, ops[0], lanes(m, ops[0])[:2] + highs)
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
    if mnem == "usra" and da == "4s":
        # d + (n >> k). When d's low 32 - k bits are a literal zero (a shl's
        # result), the two have no bit in common and the sum is their
        # concatenation, an identity; otherwise the addition itself.
        k = aarch64.imm(ops[2])
        out = []
        for x, d_ in zip(lanes(m, ops[1]), lanes(m, ops[0])):
            low = 32 - k
            if (is_app_of(d_, Z3_OP_CONCAT) and d_.num_args() == 2 and is_bv_value(d_.arg(1))
                    and d_.arg(1).as_long() == 0 and d_.arg(1).size() == low):
                out.append(Concat(d_.arg(0), Extract(31, k, x)))
            else:
                out.append(d_ + LShR(x, k))
        put(m, ops[0], out)
        return True
    if mnem == "rev32" and da == "8h":
        # Each 32-bit lane's two halfwords swapped.
        put(m, ops[0], [Concat(Extract(15, 0, x), Extract(31, 16, x)) for x in lanes(m, ops[1])])
        return True
    if mnem == "tbl" and da == "16b":
        tables = re.findall(r"v(\d+)\.16b", ops[1])
        src = [m.v[int(t)][:4] for t in tables]
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
    if mnem == "dup" and da == "2d" and ops[1][0] == "x":
        x = m.reg(ops[1])
        if isinstance(x, aarch64.Ptr):
            raise Unproved("a pointer duplicated into a vector")
        lo, hi = Extract(31, 0, x), Extract(63, 32, x)
        put(m, ops[0], [lo, hi, lo, hi])
        return True
    if mnem in ("mov", "ins", "umov") and len(ops) == 2 and re.search(r"\.[sd]\[\d+\]$", ops[0] + " " + ops[1]):
        # Element moves: v.s[i] or v.d[i] (two lanes) to or from a vector
        # element or a w or x register; the destination's other elements kept.
        def elem(op):
            e = re.fullmatch(r"v(\d+)\.([sd])\[(\d+)\]", op)
            return (int(e.group(1)), e.group(2), int(e.group(3))) if e else None
        dst, src = elem(ops[0]), elem(ops[1])
        if src:
            r, size, i = src
            value = m.v[r][i:i + 1] if size == "s" else m.v[r][2 * i:2 * i + 2]
        else:
            x = m.reg(ops[1])
            if isinstance(x, aarch64.Ptr):
                raise Unproved("a pointer moved into a vector")
            value = [x] if x.size() == 32 else [Extract(31, 0, x), Extract(63, 32, x)]
        if dst:
            r, size, i = dst
            if len(value) != (1 if size == "s" else 2):
                raise Unproved(f"element sizes differ: {', '.join(ops)}")
            ls = list(m.v[r])
            at = i if size == "s" else 2 * i
            ls[at:at + len(value)] = value
            m.v[r] = ls
        else:
            if (ops[0][0] == "w") != (len(value) == 1):
                raise Unproved(f"element sizes differ: {', '.join(ops)}")
            m.set(ops[0], value[0] if len(value) == 1 else Concat(value[1], value[0]))
        return True
    if mnem == "mov" and da == "16b":
        put(m, ops[0], lanes(m, ops[1]))
        return True
    if mnem == "ld1r" and re.fullmatch(r"\{v\d+\.4s\}", ops[0]):
        addr, wb, new = aarch64.mem_operand(m, ops, 1)
        put(m, ops[0].strip("{}"), [m.load(addr, 4)] * 4)
        if wb:
            m.set(wb, new)
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
    import sme
    return sme.step(m, pc, mnem, ops)
