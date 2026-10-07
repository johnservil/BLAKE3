"""Symbolic execution of AArch64 machine code, for proving the assembly
kernels equal to BLAKE3's specification (tools/verify/README.md).

The code comes from the assembled object, disassembled by objdump, so a
proof covers the instructions the CPU runs. Values are Z3 bitvector terms;
pointers are (region, offset) pairs, so every access is checked against
its region's bounds. Control flow must be concrete: a branch, a select, or
an address that depends on data stops the run with an error, so a run that
finishes also shows the path depends only on the lengths and counts.

Only the instructions the kernels use are modelled; an unknown one stops
the run. Each model is a few lines, checked against the Arm architecture
reference manual, and cross-checked on the CPU (cross_check.py).
"""

import re
import subprocess
from z3 import (BitVec, BitVecVal, Concat, Extract, LShR, RotateRight, ZeroExt,
                SignExt, simplify, is_bv_value)


class Unproved(Exception):
    """The run met something it cannot or must not do: the proof fails."""


def concrete(term):
    """`term`'s value if it is a constant, else None."""
    if isinstance(term, Ptr):
        return None
    t = simplify(term)
    return t.as_long() if is_bv_value(t) else None


class Ptr:
    """An address: `offset` bytes into region `region`."""
    def __init__(self, region, offset):
        self.region, self.offset = region, offset

    def __add__(self, n):
        return Ptr(self.region, self.offset + n)

    def __repr__(self):
        return f"{self.region}+{self.offset}"


class Region:
    """A block of memory: `size` bytes, readable from `lo`; each unwritten
    byte is `initial(offset)` (a symbol or a constant), or unreadable when
    `initial` is None (the stack: nothing may be read before it is written)."""
    def __init__(self, name, size, writable, initial, lo=0):
        self.name, self.size, self.writable, self.initial, self.lo = name, size, writable, initial, lo
        self.bytes = {}


ZERO64 = BitVecVal(0, 64)


class Machine:
    def __init__(self, code, regions):
        """`code`: {address: (mnemonic, operands)}; `regions`: {name: Region}."""
        self.code, self.regions = code, regions
        self.x = {}          # general registers, 64-bit terms or Ptr
        self.v = {}          # vector registers: four 32-bit lane terms, lane 0 first
        self.nzcv = None     # concrete flags (n, z, c, v)
        self.steps = 0

    # Registers.
    def reg(self, name):
        if name in ("xzr", "wzr"):
            return ZERO64 if name == "xzr" else BitVecVal(0, 32)
        if name == "sp":
            return self.x["sp"]
        n = int(name[1:])
        val = self.x[n]
        if name[0] == "x":
            return val
        if isinstance(val, Ptr):
            raise Unproved(f"32-bit read of a pointer in {name}")
        return Extract(31, 0, val)

    def set(self, name, val):
        if name in ("xzr", "wzr"):
            return
        if name == "sp":
            self.x["sp"] = val
            return
        n = int(name[1:])
        if name[0] == "w":
            if isinstance(val, Ptr):
                raise Unproved(f"pointer written to {name}")
            val = ZeroExt(32, val) if val.size() == 32 else val
        self.x[n] = val

    # Memory.
    def access(self, addr, n, write):
        if not isinstance(addr, Ptr):
            raise Unproved(f"access through a non-pointer {addr}")
        r = self.regions.get(addr.region)
        if r is None:
            raise Unproved(f"access to {addr.region}, which is not memory")
        lo = r.lo if r.name != "stack" else self.x["sp"].offset
        if addr.offset < lo or addr.offset + n > r.size:
            raise Unproved(f"{'write' if write else 'read'} of {n} bytes at {addr} outside [{lo}, {r.size})")
        if write and not r.writable:
            raise Unproved(f"write to read-only {r.name}")
        return r

    def load(self, addr, n):
        r = self.access(addr, n, False)
        parts = []
        for i in range(n):
            b = r.bytes.get(addr.offset + i)
            if b is None:
                if r.initial is None:
                    raise Unproved(f"read of unwritten {addr + i}")
                b = r.initial(addr.offset + i)
            parts.append(b)
        if all(isinstance(p, tuple) for p in parts) and n == 8:
            ptrs = {(p[0].region, p[0].offset) for p in parts}
            if len(ptrs) == 1 and [p[1] for p in parts] == list(range(8)):
                return parts[0][0]
        if any(isinstance(p, tuple) for p in parts):
            raise Unproved(f"partial read of a stored pointer at {addr}")
        return Concat(*reversed(parts)) if n > 1 else parts[0]

    def store(self, addr, val, n):
        r = self.access(addr, n, True)
        if isinstance(val, Ptr):
            assert n == 8, "a pointer stored in 8 bytes"
            for i in range(8):
                r.bytes[addr.offset + i] = (val, i)
            return
        for i in range(n):
            r.bytes[addr.offset + i] = Extract(8 * i + 7, 8 * i, val)


import shutil
# A disassembler that decodes SME2 (LLVM 17 and later).
LLVM_OBJDUMP = next((c for c in ("llvm-objdump-19", "llvm-objdump-18", "llvm-objdump-17", "llvm-objdump",
                                 "/usr/lib/llvm-19/bin/llvm-objdump", "/usr/lib/llvm-18/bin/llvm-objdump")
                     if shutil.which(c)), None)


def disassemble(obj, start, stop, llvm=False):
    """{address: (mnemonic, [operand strings])} for [start, stop) of `obj`.
    GNU objdump for the NEON file; LLVM's (`llvm`) for the SME2 file, whose
    instructions binutils 2.40 does not decode."""
    tool = [LLVM_OBJDUMP, "--mattr=+sme2,+sve2,+sve2-sha3"] if llvm else ["objdump"]
    out = subprocess.run(tool + ["-d", "--no-show-raw-insn", f"--start-address={start}",
                          f"--stop-address={stop}", obj], capture_output=True, text=True, check=True).stdout
    code = {}
    for line in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s*\t(\S+)\s*(.*)", line)
        if not m:
            continue
        addr, mnem, ops = int(m.group(1), 16), m.group(2), m.group(3).split("//")[0].strip()
        code[addr] = (mnem, split_operands(ops))
    return code


def text_bytes(obj):
    """The .text section's bytes, {address: int}."""
    out = subprocess.run(["objdump", "-s", "-j", ".text", obj], capture_output=True, text=True, check=True).stdout
    data = {}
    for line in out.splitlines():
        m = re.match(r"\s([0-9a-f]+) ((?:[0-9a-f]{2,8} ?){1,4})", line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        hexs = "".join(m.group(2).split())
        for i in range(0, len(hexs), 2):
            data[addr + i // 2] = int(hexs[i:i + 2], 16)
    return data


def symbols(obj):
    out = subprocess.run(["nm", obj], capture_output=True, text=True, check=True).stdout
    return {p[2]: int(p[0], 16) for p in (l.split() for l in out.splitlines()) if len(p) == 3}


def split_operands(ops):
    """Operands split at top-level commas: `[x0, #8]!` stays one."""
    parts, depth, cur = [], 0, ""
    for ch in ops:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def imm(s):
    s = s.lstrip("#")
    return int(s, 0)


CONDS = {
    "eq": lambda n, z, c, v: z, "ne": lambda n, z, c, v: not z,
    "hs": lambda n, z, c, v: c, "cs": lambda n, z, c, v: c,
    "lo": lambda n, z, c, v: not c, "cc": lambda n, z, c, v: not c,
    "mi": lambda n, z, c, v: n, "pl": lambda n, z, c, v: not n,
    "hi": lambda n, z, c, v: c and not z, "ls": lambda n, z, c, v: not (c and not z),
    "ge": lambda n, z, c, v: n == v, "lt": lambda n, z, c, v: n != v,
    "gt": lambda n, z, c, v: not z and n == v, "le": lambda n, z, c, v: z or n != v,
}


def width(reg):
    return 32 if reg[0] == "w" else 64


def run(m, entry, stop_at_ret=True, max_steps=10_000_000):
    """Execute from `entry` until `ret`; return the machine."""
    pc = entry
    while True:
        m.steps += 1
        if m.steps > max_steps:
            raise Unproved("too many steps")
        if pc not in m.code:
            raise Unproved(f"execution reached {pc:#x}, outside the code under proof")
        mnem, ops = m.code[pc]
        nxt = step(m, pc, mnem, ops)
        if nxt == "ret":
            return m
        pc = nxt if nxt is not None else pc + 4


def operand_value(m, op, w):
    """A register (with an optional shift) or an immediate, as a w-bit term."""
    if op.startswith("#"):
        return BitVecVal(imm(op) & ((1 << w) - 1), w)
    return m.reg(op)


def shifted(m, ops, w):
    """The last operand(s) of an arithmetic instruction: `reg`, `#imm`,
    `reg, lsl #n`, `#imm, lsl #n`."""
    if len(ops) == 2 and ops[1].startswith(("lsl", "lsr", "asr", "uxtw", "sxtw")):
        kind, *amount = ops[1].split()
        n = imm(amount[0]) if amount else 0
        v = operand_value(m, ops[0], w)
        if isinstance(v, Ptr):
            raise Unproved("shifted pointer")
        if kind == "lsl":
            return v << n
        if kind == "lsr":
            return LShR(v, n)
        if kind == "uxtw":
            return ZeroExt(32, Extract(31, 0, v)) << n
        raise Unproved(f"shift {kind}")
    return operand_value(m, ops[0], w)


def add_values(a, b, sub=False):
    if isinstance(a, Ptr):
        k = concrete(b)
        if k is None:
            raise Unproved(f"pointer {a} moved by a data-dependent amount")
        return a + (-k if sub else k)
    if isinstance(b, Ptr):
        if sub:
            raise Unproved("subtraction of a pointer")
        return add_values(b, a)
    return a - b if sub else a + b


def set_flags_sub(m, a, b, w):
    """NZCV of a - b, which must be concrete (or two pointers into one region)."""
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        if not (isinstance(a, Ptr) and isinstance(b, Ptr) and a.region == b.region):
            raise Unproved(f"comparison of {a} with {b}")
        x, y = a.offset % (1 << 64), b.offset % (1 << 64)
    else:
        x, y = concrete(a), concrete(b)
        if x is None or y is None:
            raise Unproved("a condition depends on data")
    mask = (1 << w) - 1
    x, y = x & mask, y & mask
    r = (x - y) & mask
    top = 1 << (w - 1)
    m.nzcv = (bool(r & top), r == 0, x >= y, bool(((x ^ y) & (x ^ r)) & top))


def mem_operand(m, ops, k):
    """Parse the memory operand at ops[k] (and a post-index at ops[k+1]):
    (address for the access, writeback register or None, new value)."""
    s = ops[k]
    pre = s.endswith("!")
    inner = s.rstrip("!").strip("[]")
    parts = [p.strip() for p in inner.split(",")]
    base = m.reg(parts[0])
    if len(parts) == 1:
        off = 0
    elif parts[1].startswith("#"):
        off = imm(parts[1])
    else:
        idx = m.reg(parts[1]) if parts[1][0] == "x" else ZeroExt(32, m.reg(parts[1]))
        sh = imm(parts[2].split()[1]) if len(parts) > 2 else 0
        k_ = concrete(idx)
        if k_ is None:
            raise Unproved("an address depends on data")
        off = k_ << sh
    if len(ops) > k + 1 and ops[k + 1].startswith("#"):
        post = imm(ops[k + 1])
        return add_values(base, BitVecVal(0, 64)), parts[0], add_values(base, BitVecVal(post, 64))
    addr = add_values(base, BitVecVal(off % (1 << 64), 64))
    return addr, (parts[0] if pre else None), addr


def step(m, pc, mnem, ops):
    w = width(ops[0]) if ops and ops[0][0] in "wx" else 64
    if mnem == "nop":
        return None
    if mnem in ("add", "sub", "adds", "subs") and ops[0][0] in "wxs":
        a = m.reg(ops[1])
        b = shifted(m, ops[2:], w)
        r = add_values(a, b, sub=mnem.startswith("sub"))
        if mnem.endswith("s"):
            set_flags_sub(m, a, b if mnem.startswith("sub") else -b, w)
        m.set(ops[0], r)
        return None
    if mnem == "cmp":
        set_flags_sub(m, m.reg(ops[0]), shifted(m, ops[1:], w), w)
        return None
    if mnem in ("eor", "orr", "and") and ops[0][0] in "wx":
        a, b = m.reg(ops[1]), shifted(m, ops[2:], w)
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unproved(f"{mnem} on a pointer")
        m.set(ops[0], {"eor": a ^ b, "orr": a | b, "and": a & b}[mnem])
        return None
    if mnem == "ror":
        m.set(ops[0], RotateRight(m.reg(ops[1]), imm(ops[2])))
        return None
    if mnem == "bfi":
        lsb, n = imm(ops[2]), imm(ops[3])
        old, src = m.reg(ops[0]), m.reg(ops[1])
        parts = []
        if lsb + n < w:
            parts.append(Extract(w - 1, lsb + n, old))
        parts.append(Extract(n - 1, 0, src))
        if lsb > 0:
            parts.append(Extract(lsb - 1, 0, old))
        m.set(ops[0], Concat(*parts))
        return None
    if mnem in ("lsr", "lsl"):
        a = m.reg(ops[1])
        m.set(ops[0], LShR(a, imm(ops[2])) if mnem == "lsr" else a << imm(ops[2]))
        return None
    if mnem == "ubfx":
        lsb, n = imm(ops[2]), imm(ops[3])
        m.set(ops[0], ZeroExt(w - n, Extract(lsb + n - 1, lsb, m.reg(ops[1]))))
        return None
    if mnem == "mov" and ops[0][0] in "wxs":
        if ops[1].startswith("#"):
            m.set(ops[0], BitVecVal(imm(ops[1]) % (1 << w), w))
        else:
            m.set(ops[0], m.reg(ops[1]))
        return None
    if mnem == "movk":
        sh = imm(ops[2].split()[1]) if len(ops) > 2 else 0
        old = m.reg(ops[0])
        new = BitVecVal(imm(ops[1]), 16)
        parts = []
        if sh + 16 < w:
            parts.append(Extract(w - 1, sh + 16, old))
        parts.append(new)
        if sh > 0:
            parts.append(Extract(sh - 1, 0, old))
        m.set(ops[0], Concat(*parts) if len(parts) > 1 else parts[0])
        return None
    if mnem == "csel":
        if m.nzcv is None:
            raise Unproved("csel before any comparison")
        m.set(ops[0], m.reg(ops[1]) if CONDS[ops[3]](*m.nzcv) else m.reg(ops[2]))
        return None
    if mnem == "adr":
        m.set(ops[0], Ptr("text", int(ops[1].split()[0], 16)))
        return None
    if mnem in ("ldr", "ldrb", "ldrh", "str", "strb", "strh") and ops[0][0] in "wx":
        n = {"b": 1, "h": 2}.get(mnem[-1], w // 8)
        addr, wb, new = mem_operand(m, ops, 1)
        if mnem.startswith("ldr"):
            val = m.load(addr, n)
            if isinstance(val, Ptr):
                m.set(ops[0], val)
            else:
                m.set(ops[0], ZeroExt(w - 8 * n, val) if 8 * n < w else val)
        else:
            val = m.reg(ops[0])
            m.store(addr, val if isinstance(val, Ptr) or 8 * n == w else Extract(8 * n - 1, 0, val), n)
        if wb:
            m.set(wb, new)
        return None
    if mnem in ("ldp", "stp") and ops[0][0] in "wx":
        n = w // 8
        addr, wb, new = mem_operand(m, ops, 2)
        for i, r in enumerate(ops[:2]):
            if mnem == "ldp":
                m.set(r, m.load(addr + i * n, n))
            else:
                m.store(addr + i * n, m.reg(r), n)
        if wb:
            m.set(wb, new)
        return None
    if mnem.startswith("b.") or mnem == "b":
        target = int(ops[0].split()[0], 16)
        if mnem == "b" or CONDS[mnem[2:]](*m.nzcv):
            return target
        return None
    if mnem in ("tbz", "tbnz"):
        v = m.reg(ops[0])
        n = imm(ops[1])
        bit = concrete(Extract(n, n, v)) if not isinstance(v, Ptr) else None
        if bit is None:
            raise Unproved("a branch depends on data")
        target = int(ops[2].split()[0], 16)
        return target if (bit == 0) == (mnem == "tbz") else None
    if mnem in ("cbz", "cbnz"):
        k = concrete(m.reg(ops[0])) if not isinstance(m.reg(ops[0]), Ptr) else 1
        if k is None:
            raise Unproved("a branch depends on data")
        target = int(ops[1].split()[0], 16)
        return target if (k == 0) == (mnem == "cbz") else None
    if mnem == "ret":
        return "ret"
    import vector
    if vector.step(m, pc, mnem, ops):
        return None
    raise Unproved(f"instruction not modelled: {mnem} {', '.join(ops)} at {pc:#x}")
