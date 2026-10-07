"""Streaming SVE and SME2 instructions for aarch64.py, at the 512-bit
streaming vector length of Apple M4 (`cntw` = 16: an assumption of every
SME2 proof, which the kernels check themselves and otherwise return).

A Z register is the same sixteen-lane list as the vector registers (its
low four lanes are the NEON register). A predicate is sixteen booleans,
Python or Z3 (a merging add under a data-dependent predicate is an exact
If; a load, store, or branch needs concrete ones). ZA's four .S tiles are
16 x 16 lane terms, `za[t][row][column]`; a horizontal slice is a row, a
vertical one a column. Entering or leaving streaming mode zeroes the Z
and P registers; enabling ZA zeroes it.
"""

import re
from z3 import BitVecVal, If, ULT, UGT, Extract, RotateRight, BoolVal, is_true, is_false, simplify
import aarch64
from aarch64 import Unproved

LANES = 16
ZERO = BitVecVal(0, 32)


def zreg(op):
    m = re.match(r"z(\d+)\.(\w)", op)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def preg(op):
    m = re.match(r"p(\d+)(?:\.\w|/[zm])?$", op)
    return int(m.group(1)) if m else None


def zlist(text):
    """`{ z16.s - z19.s }` or `{ z12.s, z13.s }`: register numbers."""
    regs = [int(x) for x in re.findall(r"z(\d+)", text)]
    if "-" in text:
        return list(range(regs[0], regs[1] + 1))
    return regs


def slices(text):
    """`za0v.s[w12, 0x0:0x3]` or `za0h.s[w12, 3]`: (tile, vertical, base
    register, first offset, count)."""
    m = re.match(r"\{?\s*za(\d)([hv])\.s\[(w\d+),\s*(0x[0-9a-f]+|\d+)(?::(0x[0-9a-f]+|\d+))?\]\s*\}?$", text.strip())
    if not m:
        return None
    first = int(m.group(4), 0)
    last = int(m.group(5), 0) if m.group(5) else first
    return int(m.group(1)), m.group(2) == "v", m.group(3), first, last - first + 1


def concrete_bool(b):
    if isinstance(b, bool):
        return b
    s = simplify(b)
    if is_true(s):
        return True
    if is_false(s):
        return False
    raise Unproved("a predicate that must be known depends on data")


def zero_streaming(m):
    for i in range(32):
        m.v[i] = [ZERO] * LANES
    m.p = {i: [False] * LANES for i in range(16)}


def slice_index(m, reg, offset, count=1):
    """The slice an access of `count` registers reaches at `offset`: the
    base register's value rounded down to a multiple of `count` (the CPU
    does so for multi-register moves; cross_check_sme.py found it), plus
    the offset, modulo the tile's sixteen slices."""
    base = aarch64.concrete(m.reg(reg))
    if base is None:
        raise Unproved("a ZA slice index depends on data")
    return (base - base % count + offset) % LANES


def step(m, pc, mnem, ops):
    if not hasattr(m, "za"):
        m.za, m.streaming = None, False
        m.p = {i: None for i in range(16)}
    if mnem in ("smstart", "smstop"):
        what = ops[0] if ops else "both"
        if what in ("sm", "both"):
            m.streaming = mnem == "smstart"
            zero_streaming(m)
        if what in ("za", "both"):
            m.za = [[[ZERO] * LANES for _ in range(LANES)] for _ in range(4)] if mnem == "smstart" else None
        return True
    if mnem == "cntw":
        if not m.streaming:
            raise Unproved("cntw outside streaming mode")
        m.set(ops[0], BitVecVal(LANES, 64))
        return True
    if mnem == "prfm":
        return True      # a hint: no architectural effect
    d, size = zreg(ops[0]) if ops else (None, None)
    if mnem == "ptrue" and ops[0].endswith(".s"):
        m.p[preg(ops[0].split(".")[0])] = [True] * LANES
        return True
    if mnem == "whilelo" and ops[0].endswith(".s"):
        a, b = aarch64.concrete(m.reg(ops[1])), aarch64.concrete(m.reg(ops[2]))
        if a is None or b is None:
            raise Unproved("whilelo on data")
        m.p[preg(ops[0].split(".")[0])] = [a + i < b for i in range(LANES)]
        return True
    if mnem in ("cmphi", "cmplo", "cmphs", "cmpls") and ops[0].endswith(".s"):
        g = m.p[preg(ops[1].split("/")[0])]
        a, b = m.v[zreg(ops[2])[0]], m.v[zreg(ops[3])[0]]
        f = {"cmphi": lambda x, y: UGT(x, y), "cmplo": lambda x, y: ULT(x, y)}[mnem]
        m.p[preg(ops[0].split(".")[0])] = [f(x, y) if concrete_bool(gi) else False for gi, x, y in zip(g, a, b)]
        return True
    if mnem == "index" and size == "s":
        base = m.reg(ops[1])
        stepv = m.reg(ops[2]) if not ops[2].startswith("#") else BitVecVal(aarch64.imm(ops[2]) % (1 << 32), 32)
        m.v[d] = [base + BitVecVal(i, 32) * stepv for i in range(LANES)]
        return True
    if mnem in ("mov", "dup") and size == "s" and len(ops) == 2 and not ops[1].startswith(("{", "za")):
        if ops[1].startswith("#"):
            x = BitVecVal(aarch64.imm(ops[1]) % (1 << 32), 32)
        else:
            x = m.reg(ops[1])
            if isinstance(x, aarch64.Ptr):
                raise Unproved("a pointer duplicated into a vector")
        m.v[d] = [x] * LANES
        return True
    if mnem == "add" and size == "s":
        if len(ops) == 3:
            a, b = m.v[zreg(ops[1])[0]], m.v[zreg(ops[2])[0]]
            m.v[d] = [x + y for x, y in zip(a, b)]
            return True
        g = m.p[preg(ops[1].split("/")[0])]
        assert ops[1].endswith("/m") and zreg(ops[2])[0] == d
        a, b = m.v[d], m.v[zreg(ops[3])[0]]
        m.v[d] = [x + y if gi is True else (x if gi is False else If(gi, x + y, x)) for gi, x, y in zip(g, a, b)]
        return True
    if mnem == "xar" and size == "s":
        k = aarch64.imm(ops[3])
        assert zreg(ops[1])[0] == d
        a, b = m.v[d], m.v[zreg(ops[2])[0]]
        m.v[d] = [RotateRight(x ^ y, k % 32) if k % 32 else x ^ y for x, y in zip(a, b)]
        return True
    if mnem == "eor" and size == "d" and len(ops) == 3:
        a, b = m.v[zreg(ops[1])[0]], m.v[zreg(ops[2])[0]]
        m.v[d] = [x ^ y for x, y in zip(a, b)]
        return True
    if mnem == "mov" and (ops[0].startswith("{") and "za" in ops[1] or ops[0].startswith("za")):
        if m.za is None:
            raise Unproved("ZA used while disabled")
        to_za = ops[0].startswith("za")
        t, vertical, reg, first, count = slices(ops[0] if to_za else ops[1])
        zs = zlist(ops[1] if to_za else ops[0])
        assert len(zs) == count
        for j, z in enumerate(zs):
            s = slice_index(m, reg, first + j, count)
            cell = lambda i: (i, s) if vertical else (s, i)
            if to_za:
                for i in range(LANES):
                    r, c = cell(i)
                    m.za[t][r][c] = m.v[z][i]
            else:
                m.v[z] = [m.za[t][cell(i)[0]][cell(i)[1]] for i in range(LANES)]
        return True
    if mnem in ("ld1w", "st1w") and ops[0].lstrip("{ ").startswith("za"):
        if m.za is None:
            raise Unproved("ZA used while disabled")
        t, vertical, reg, first, count = slices(ops[0])
        assert not vertical and count == 1
        row = slice_index(m, reg, first)
        g = m.p[preg(ops[1].split("/")[0])]
        addr, wb, new = aarch64.mem_operand(m, ops, 2)
        assert wb is None
        for i in range(LANES):
            active = concrete_bool(g[i])
            if mnem == "ld1w":
                m.za[t][row][i] = m.load(addr + 4 * i, 4) if active else ZERO
            elif active:
                m.store(addr + 4 * i, m.za[t][row][i], 4)
        return True
    return False
