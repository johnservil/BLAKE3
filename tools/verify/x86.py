"""A model of the x86-64 instructions BLAKE3's kernels use, for proving
the assembled machine code (upstream's c/blake3_{sse41,avx2,avx512}_
x86-64_unix.S and their MSVC twins) as aarch64.py proves the Arm kernels.

The machine runs LLVM's disassembly (Intel syntax) of a linked object, so
rip-relative constants resolve to the image. General registers hold 64-bit
terms or pointers (aarch64.Ptr); vector registers 16 32-bit lane terms
(xmm the first 4, ymm 8, zmm 16); mask registers 16-bit terms; the flags
the operands of the last instruction that set them, read by condition.
Memory, pointers, and branch decisions are aarch64's: only lengths may
steer a branch or an address.

Counter words: the kernels compute a lane's counter halves with compares
and blends where the standard adds 64-bit counters. `facts` holds the
standard's terms for them; a small lane term the solver proves equal to
one is replaced by it, so the rounds' terms meet the specification's in
one normal form (canon.py). The replacement is the proved equality.
"""

import re
import subprocess

from z3 import (BitVec, BitVecVal, Concat, Extract, LShR, RotateRight, ZeroExt, If, ULT, UGT, ULE,
                UGE, Not, Solver, simplify, is_bv_value, unsat)

import aarch64
from aarch64 import Ptr, Region, Unproved, decide, as_bv, concrete

LLVM_OBJDUMP = "/usr/lib/llvm-19/bin/llvm-objdump"
LLD = "/usr/lib/llvm-19/bin/ld.lld"

GPR = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"] + [f"r{i}" for i in range(8, 16)]
SUB = {}
for base, e, x, l in (("rax", "eax", "ax", "al"), ("rbx", "ebx", "bx", "bl"), ("rcx", "ecx", "cx", "cl"),
                      ("rdx", "edx", "dx", "dl"), ("rsi", "esi", "si", "sil"), ("rdi", "edi", "di", "dil"),
                      ("rbp", "ebp", "bp", "bpl"), ("rsp", "esp", "sp", "spl")):
    SUB.update({base: (base, 64), e: (base, 32), x: (base, 16), l: (base, 8)})
for i in range(8, 16):
    SUB.update({f"r{i}": (f"r{i}", 64), f"r{i}d": (f"r{i}", 32), f"r{i}w": (f"r{i}", 16), f"r{i}b": (f"r{i}", 8)})
SIZES = {"byte": 1, "word": 2, "dword": 4, "qword": 8, "xmmword": 16, "ymmword": 32, "zmmword": 64}


def link(obj):
    """A shared object of `obj`, so its rip-relative constants resolve."""
    so = obj + ".so"
    subprocess.run([LLD, "-shared", obj, "-o", so], check=True)
    return so


def disassemble(so, start, stop):
    """{address: (mnemonic, [operands], next address)} for [start, stop)."""
    out = subprocess.run([LLVM_OBJDUMP, "-d", "--no-show-raw-insn", "-M", "intel", f"--start-address={start}",
                          f"--stop-address={stop}", so], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s*\t(\S+)(?:\t(.*))?$", line)
        if not m:
            continue
        addr, mnem, rest = int(m.group(1), 16), m.group(2), (m.group(3) or "")
        target = None
        if "#" in rest:
            rest, comment = rest.split("#", 1)
            t = re.match(r"\s*0x([0-9a-f]+)", comment)
            target = int(t.group(1), 16) if t else None
        rows.append((addr, mnem, split(rest.strip()), target))
    code = {}
    for i, (addr, mnem, ops, target) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else stop
        # A rip-relative operand: its absolute address from the comment.
        ops = [o.replace("[rip]", f"[abs {target:#x}]") if "[rip" in o else o for o in ops]
        ops = [re.sub(r"\[rip [+-] 0x[0-9a-f]+\]", f"[abs {target:#x}]", o) if "[rip" in o else o for o in ops]
        code[addr] = (mnem, ops, nxt)
    return code


def split(ops):
    """Operands at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in ops:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def symbols(so):
    out = subprocess.run([LLVM_OBJDUMP, "-t", so], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.match(r"^[0-9a-f]{16}$", parts[0]):
            syms[parts[-1]] = int(parts[0], 16)
    return syms


def text_end(so):
    out = subprocess.run([LLVM_OBJDUMP, "-h", so], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4 and p[1] == ".text":
            return int(p[3], 16) + int(p[2], 16)
    raise AssertionError("no .text")


def term_size(t, limit):
    seen, todo = set(), [t]
    while todo:
        x = todo.pop()
        if x.get_id() in seen:
            continue
        seen.add(x.get_id())
        if len(seen) > limit:
            return limit + 1
        todo.extend(x.children())
    return len(seen)


class Machine(aarch64.Machine):
    def __init__(self, code, regions):
        super().__init__(code, regions)
        self.k = {i: BitVecVal(0, 16) for i in range(8)}
        self.flags = None
        self.facts = []          # the standard's small terms (counter words): see the module's doc
        self.fact_symbols = set()

    # General registers.
    def gread(self, name):
        base, bits = SUB[name]
        v = self.x[base]
        if bits == 64:
            return v
        if isinstance(v, Ptr):
            raise Unproved(f"a {bits}-bit read of a pointer in {name}")
        return Extract(bits - 1, 0, v)

    def gwrite(self, name, v):
        base, bits = SUB[name]
        if bits == 64:
            self.x[base] = v
            return
        if isinstance(v, Ptr):
            raise Unproved(f"a pointer written to {name}")
        v = simplify(v)
        if bits == 32:
            self.x[base] = ZeroExt(32, v)
        else:
            old = self.x[base]
            if isinstance(old, Ptr):
                raise Unproved(f"a partial write of a pointer in {name}")
            self.x[base] = simplify(Concat(Extract(63, bits, old), v))

    # Vector registers.
    def lanes(self, name):
        return self.v[int(re.match(r"[xyz]mm(\d+)", name).group(1))]

    def vwrite(self, name, new, vex):
        """Write `new` (its width's lanes) to a vector register: VEX and
        EVEX forms zero the lanes above; legacy SSE keeps them."""
        n = int(re.match(r"[xyz]mm(\d+)", name).group(1))
        new = [self.normal(simplify(x)) for x in new]
        if vex:
            self.v[n] = new + [BitVecVal(0, 32)] * (16 - len(new))
        else:
            self.v[n] = new + self.v[n][len(new):]

    def normal(self, t):
        """The standard's form of a small term proved equal to one of
        `facts` (the module's doc), else `t`."""
        if not self.facts or is_bv_value(t):
            return t
        for f in self.facts:
            if t.eq(f):
                return t
        if term_size(t, 120) > 120 or not aarch64.term_symbols(t) <= self.fact_symbols:
            return t
        for f in self.facts:
            s = Solver()
            s.add(t != f)
            if s.check() == unsat:
                return f
        return t


def width(name):
    return {"x": 4, "y": 8, "z": 16}[name[0]]


def is_vreg(op):
    return re.match(r"^[xyz]mm\d+$", op) is not None


def imm(op):
    return int(op, 0)


def strip_mask(op):
    """An operand and its EVEX mask: ('zmm0', 1, zeroing)."""
    m = re.match(r"^(\S+)\s*\{k(\d)\}(\s*\{z\})?$", op)
    if m:
        return m.group(1), int(m.group(2)), bool(m.group(3))
    return op, None, False


def address(m, op):
    """A memory operand's size (bytes, or None) and address."""
    mt = re.match(r"^(?:(\w+) ptr )?\[(.*)\](\{1to\d+\})?$", op)
    if not mt:
        raise Unproved(f"unparsed memory operand {op}")
    size = SIZES.get(mt.group(1)) if mt.group(1) else None
    inside = mt.group(2).strip()
    if inside.startswith("abs "):
        return size, Ptr("image", int(inside[4:], 16)), mt.group(3)
    base, disp = None, 0
    for sign, term in re.findall(r"([+-]?)\s*([^+-]+)", inside):
        term = term.strip()
        neg = sign == "-"
        if re.match(r"^(0x[0-9a-f]+|\d+)$", term):
            disp += -int(term, 0) if neg else int(term, 0)
        elif "*" in term:
            sc, r = term.split("*")
            if sc.strip() in SUB:
                sc, r = r, sc
            val = m.gread(r.strip())
            if isinstance(val, Ptr):
                raise Unproved("a scaled pointer")
            idx = as_bv(val) * int(sc.strip(), 0)
            base = idx if base is None else (base + idx if not isinstance(base, Ptr) else base + idx)
        else:
            val = m.gread(term)
            if base is None:
                base = val
            elif isinstance(val, Ptr):
                base = val + base
            else:
                base = base + val
    if base is None:
        raise Unproved(f"an absolute address {op}")
    if not isinstance(base, Ptr):
        raise Unproved(f"an address that is no pointer: {op}")
    return size, base + disp, mt.group(3)


def load_lanes(m, op, w):
    size, addr, bcast = address(m, op)
    if bcast:
        word = m.load(addr, 4)
        return [word] * w
    n = 4 * w if size is None else size
    val = m.load(addr, n)
    return [Extract(32 * i + 31, 32 * i, val) for i in range(n // 4)]


def src_lanes(m, op, w):
    if is_vreg(op):
        return m.lanes(op)[:w]
    return load_lanes(m, op, w)


def gval(m, op, bits):
    """A general operand's value: register, immediate, or memory."""
    if op in SUB:
        return m.gread(op)
    if op.startswith("[") or " ptr [" in op:
        size, addr, _ = address(m, op)
        return m.load(addr, size or bits // 8)
    return BitVecVal(imm(op) % (1 << bits), bits)


def opbits(m, op):
    if op in SUB:
        return SUB[op][1]
    mt = re.match(r"^(\w+) ptr", op)
    return 8 * SIZES[mt.group(1)] if mt else 64


def ptr_add(a, b, sub=False):
    """a + b (or a - b) where a or b may be a pointer."""
    if isinstance(a, Ptr) and isinstance(b, Ptr):
        if not sub or a.region != b.region:
            raise Unproved("arithmetic on two pointers")
        return simplify(as_bv(a.offset) - as_bv(b.offset))
    if isinstance(b, Ptr):
        if sub:
            raise Unproved("a pointer subtracted from a number")
        return b + a
    if isinstance(a, Ptr):
        return a + (simplify(-as_bv(b)) if sub else b)
    return simplify(a - b if sub else a + b)


def cond(m, cc):
    """The condition `cc` on the flags the last setter left."""
    f = m.flags
    if f is None:
        raise Unproved("a condition on unset flags")
    kind = f[0]
    if kind == "sub":
        _, a, b = f
        table = {"e": a == b, "z": a == b, "ne": a != b, "nz": a != b, "b": ULT(a, b), "c": ULT(a, b),
                 "ae": UGE(a, b), "nc": UGE(a, b), "a": UGT(a, b), "be": ULE(a, b),
                 "l": a < b, "ge": a >= b, "g": a > b, "le": a <= b}
    elif kind == "logic":
        _, r = f
        z = r == 0
        table = {"e": z, "z": z, "ne": Not(z), "nz": Not(z), "s": r < 0, "ns": r >= 0, "b": False, "ae": True,
                 "a": Not(z), "be": z}
    elif kind == "dec":
        _, r = f
        table = {"e": r == 0, "z": r == 0, "ne": r != 0, "nz": r != 0}
    else:
        raise Unproved(f"flags of {kind}")
    if cc not in table:
        raise Unproved(f"condition {cc} after {kind}")
    return table[cc]


def flag_operands(a, b):
    """Two compared values as terms: pointers into one region by offset."""
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        if not (isinstance(a, Ptr) and isinstance(b, Ptr) and a.region == b.region):
            raise Unproved("a comparison of a pointer with a number")
        return as_bv(a.offset), as_bv(b.offset)
    return a, b


def lanes128(x):
    """A lane list as 128-bit lanes of four."""
    return [x[i:i + 4] for i in range(0, len(x), 4)]


def bytes_of(lane):
    return [Extract(8 * i + 7, 8 * i, lane) for i in range(4)]


def pshufb(src, mask):
    out = []
    for blk, mblk in zip(lanes128(src), lanes128(mask)):
        mb = []
        for word in mblk:
            c = concrete(word)
            if c is None:
                raise Unproved("pshufb with a mask that is not constant")
            mb += [(c >> (8 * i)) & 0xff for i in range(4)]
        sb = [b for w in blk for b in bytes_of(w)]
        for j in range(4):
            sel = mb[4 * j:4 * j + 4]
            rot = next((k for k in (1, 2, 3) if sel == [4 * j + (i + k) % 4 for i in range(4)]), None)
            if rot is not None:
                out.append(RotateRight(blk[j], 8 * rot))
                continue
            parts = [BitVecVal(0, 8) if s & 0x80 else sb[s & 15] for s in sel]
            out.append(Concat(*reversed(parts)))
    return out


def blend_words(a, b, imm8):
    """pblendw: 16-bit word i from b where imm8 bit i % 8."""
    out = []
    for j, (x, y) in enumerate(zip(a, b)):
        lo, hi = (imm8 >> ((2 * j) % 8)) & 1, (imm8 >> ((2 * j + 1) % 8)) & 1
        if lo == hi:
            out.append(y if lo else x)
        else:
            out.append(Concat(Extract(31, 16, y if hi else x), Extract(15, 0, y if lo else x)))
    return out


def kbit(m, k, j):
    return Extract(j, j, m.k[k]) == 1


def step(m, pc, mnem, ops, nxt):
    """Execute one instruction; the next pc, or 'ret'."""
    vex = mnem.startswith("v")
    base = mnem[1:] if vex and mnem not in ("vzeroupper",) else mnem
    # Branches.
    if mnem == "jmp":
        return int(ops[0].split()[0], 16)
    if mnem.startswith("j"):
        return int(ops[0].split()[0], 16) if decide(m, cond(m, mnem[1:])) else nxt
    if mnem == "ret":
        return "ret"
    if mnem in ("nop", "prefetcht0", "endbr64"):
        return nxt
    if mnem == "vzeroupper":
        for i in range(16):
            m.v[i] = m.v[i][:4] + [BitVecVal(0, 32)] * 12
        return nxt
    # General-register instructions.
    if mnem == "push":
        m.x["rsp"] = m.x["rsp"] + (-8)
        m.store(m.x["rsp"], m.gread(ops[0]), 8)
        return nxt
    if mnem == "pop":
        m.gwrite(ops[0], m.load(m.x["rsp"], 8))
        m.x["rsp"] = m.x["rsp"] + 8
        return nxt
    if mnem in ("mov", "movzx") and not any(is_vreg(o) for o in ops):
        if ops[0].startswith("[") or " ptr [" in ops[0]:
            size, addr, _ = address(m, ops[0])
            val = gval(m, ops[1], 8 * size)
            m.store(addr, val, size)
        else:
            bits = SUB[ops[0]][1]
            val = gval(m, ops[1], bits if mnem == "mov" else opbits(m, ops[1]))
            if mnem == "movzx":
                val = ZeroExt(bits - val.size(), val)
            m.gwrite(ops[0], val)
        return nxt
    if mnem == "lea":
        _, addr, _ = address(m, ops[1])
        m.gwrite(ops[0], addr)
        return nxt
    if mnem in ("add", "sub", "and", "or", "xor", "cmp", "test"):
        bits = opbits(m, ops[0])
        a = gval(m, ops[0], bits)
        b = gval(m, ops[1], bits)
        if mnem in ("add", "sub", "cmp"):
            r = ptr_add(a, b, sub=mnem != "add")
            if mnem in ("sub", "cmp"):
                m.flags = ("sub",) + flag_operands(a, b)
            else:
                m.flags = ("logic", r) if not isinstance(r, Ptr) else None
            if mnem != "cmp":
                m.gwrite(ops[0], r)
            return nxt
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unproved(f"{mnem} on a pointer")
        r = simplify({"and": a & b, "or": a | b, "xor": a ^ b, "test": a & b}[mnem])
        m.flags = ("logic", r)
        if mnem != "test":
            m.gwrite(ops[0], r)
        return nxt
    if mnem in ("shl", "shr", "neg", "dec", "inc"):
        bits = opbits(m, ops[0])
        a = gval(m, ops[0], bits)
        if isinstance(a, Ptr):
            raise Unproved(f"{mnem} on a pointer")
        if mnem == "shl":
            r = a << imm(ops[1])
        elif mnem == "shr":
            r = LShR(a, imm(ops[1]))
        elif mnem == "neg":
            r = -a
        else:
            r = a - 1 if mnem == "dec" else a + 1
        r = simplify(r)
        m.flags = ("dec", r) if mnem in ("dec", "inc") else None
        m.gwrite(ops[0], r)
        return nxt
    if mnem.startswith("cmov"):
        if decide(m, cond(m, mnem[4:])):
            m.gwrite(ops[0], gval(m, ops[1], SUB[ops[0]][1]))
        return nxt
    # Mask registers.
    if mnem == "kmovw":
        if ops[0].startswith("k"):
            m.k[int(ops[0][1])] = Extract(15, 0, gval(m, ops[1], 32))
        else:
            m.gwrite(ops[0], ZeroExt(16, m.k[int(ops[1][1])]))
        return nxt
    if mnem == "knotw":
        m.k[int(ops[0][1])] = simplify(~m.k[int(ops[1][1])])
        return nxt
    # Vector instructions.
    return vector(m, mnem, base, vex, ops, nxt)


def vector(m, mnem, base, vex, ops, nxt):
    dst, mask, zeroing = strip_mask(ops[0])
    # Stores.
    if not is_vreg(dst) and not dst.startswith("k") and dst not in SUB:
        size, addr, _ = address(m, dst)
        src = ops[-1]
        if base in ("extracti128", "extracti32x4"):
            sel = imm(ops[2]) & (1 if base == "extracti128" else 3)
            val = m.lanes(ops[1])[4 * sel:4 * sel + 4]
        elif base in ("movd",):
            val = m.lanes(src)[:1]
        elif base in ("movq",):
            val = m.lanes(src)[:2]
        else:
            val = m.lanes(src)[:size // 4]
        if mask is not None:
            raise Unproved("a masked store")
        m.store(addr, Concat(*reversed(val)) if len(val) > 1 else val[0], 4 * len(val))
        return nxt
    if dst in SUB:
        # movd / movq to a general register.
        src = m.lanes(ops[1])
        if base == "movd":
            m.gwrite(dst, src[0])
        elif base == "movq":
            m.gwrite(dst, Concat(src[1], src[0]))
        else:
            raise Unproved(f"{mnem} to a general register")
        return nxt
    if dst.startswith("k"):
        k = int(dst[1])
        if base == "pcmpltud":
            a, b = m.lanes(ops[1])[:width(ops[1])], src_lanes(m, ops[2], width(ops[1]))
            bits = [If(ULT(x, y), BitVecVal(1, 1), BitVecVal(0, 1)) for x, y in zip(a, b)]
            m.k[k] = simplify(ZeroExt(16 - len(bits), Concat(*reversed(bits))))
            return nxt
        raise Unproved(f"{mnem} to a mask register")
    w = width(dst)
    old = m.lanes(dst)[:w]
    srcs = ops[1:]
    two = not vex  # legacy SSE: dst is also the first source
    if two:
        a_op, rest = dst, srcs
    else:
        a_op, rest = (srcs[0] if srcs else dst), srcs[1:]

    def A():
        return src_lanes(m, a_op, w)

    def B(i=0):
        return src_lanes(m, rest[i], w)

    if base in ("movups", "movaps", "movdqu", "movdqa", "movdqu32", "movdqa32"):
        new = src_lanes(m, ops[1], w)
    elif base in ("movd", "movq"):
        s = ops[1]
        if s in SUB:
            val = gval(m, s, SUB[s][1])
            parts = [val] if base == "movd" else [Extract(31, 0, val), Extract(63, 32, val)]
        else:
            parts = load_lanes(m, s, 1 if base == "movd" else 2)
        new = (parts + [BitVecVal(0, 32)] * 4)[:4]
        w = 4
        m.vwrite(dst, new, True)
        return nxt
    elif base in ("paddd", "psubd", "pxor", "pxord", "por", "pand"):
        a, b = A(), B()
        op = {"paddd": lambda x, y: x + y, "psubd": lambda x, y: x - y, "pxor": lambda x, y: x ^ y,
              "pxord": lambda x, y: x ^ y, "por": lambda x, y: x | y, "pand": lambda x, y: x & y}[base]
        new = [op(x, y) for x, y in zip(a, b)]
    elif base in ("pslld", "psrld", "prord"):
        a, k = A(), imm(rest[-1])
        if base == "pslld":
            new = [x << k if k < 32 else BitVecVal(0, 32) for x in a]
        elif base == "psrld":
            new = [LShR(x, k) if k < 32 else BitVecVal(0, 32) for x in a]
        else:
            new = [RotateRight(x, k % 32) for x in a]
    elif base == "pshufb":
        new = pshufb(A(), B())
    elif base == "pshufd":
        s, k = src_lanes(m, srcs[0], w), imm(srcs[-1])
        new = [blk[(k >> (2 * j)) & 3] for blk in lanes128(s) for j in range(4)]
    elif base == "shufps":
        a, b, k = A(), B(), imm(rest[-1])
        new = []
        for x, y in zip(lanes128(a), lanes128(b)):
            new += [x[k & 3], x[(k >> 2) & 3], y[(k >> 4) & 3], y[(k >> 6) & 3]]
    elif base in ("punpckldq", "unpcklps", "punpckhdq", "unpckhps", "punpcklqdq", "unpcklpd", "punpckhqdq", "unpckhpd"):
        a, b = A(), B()
        new = []
        for x, y in zip(lanes128(a), lanes128(b)):
            new += {"punpckldq": [x[0], y[0], x[1], y[1]], "unpcklps": [x[0], y[0], x[1], y[1]],
                    "punpckhdq": [x[2], y[2], x[3], y[3]], "unpckhps": [x[2], y[2], x[3], y[3]],
                    "punpcklqdq": [x[0], x[1], y[0], y[1]], "unpcklpd": [x[0], x[1], y[0], y[1]],
                    "punpckhqdq": [x[2], x[3], y[2], y[3]], "unpckhpd": [x[2], x[3], y[2], y[3]]}[base]
    elif base == "pblendw":
        new = blend_words(A(), B(), imm(rest[-1]))
    elif base in ("pblendd", "blendps"):
        a, b, k = A(), B(), imm(rest[-1])
        new = [y if (k >> (j % 8)) & 1 else x for j, (x, y) in enumerate(zip(a, b))]
    elif base in ("blendvps",):
        if two:
            a, b, sel = A(), B(), m.lanes("xmm0")[:4]
        else:
            a, b, sel = A(), B(), src_lanes(m, rest[1], w)
        new = [If(Extract(31, 31, s) == 1, y, x) for x, y, s in zip(a, b, sel)]
    elif base == "pcmpgtd":
        a, b = A(), B()
        new = [If(x > y, BitVecVal(0xFFFFFFFF, 32), BitVecVal(0, 32)) for x, y in zip(a, b)]
    elif base == "pblendmd":
        a, b = A(), B()
        new = [If(kbit(m, mask, j), y, x) for j, (x, y) in enumerate(zip(a, b))]
        mask = None
    elif base in ("pinsrd",):
        a = A()
        s = rest[0]
        val = gval(m, s, 32) if (s in SUB or "ptr" in s) else None
        new = list(a)
        new[imm(rest[-1]) & 3] = val
    elif base in ("broadcasti128", "broadcasti32x4"):
        blk = load_lanes(m, srcs[0], 4)
        new = (blk * 4)[:w]
    elif base == "pbroadcastd":
        s = srcs[0]
        word = m.lanes(s)[0] if is_vreg(s) else load_lanes(m, s, 1)[0]
        new = [word] * w
    elif base in ("inserti128", "insertf128", "inserti32x4", "inserti64x4"):
        a, k = A(), imm(rest[-1])
        part = src_lanes(m, rest[0], 8 if base == "inserti64x4" else 4)
        n = len(part)
        at = (k & (1 if n == 8 or base in ("inserti128", "insertf128") else 3)) * n
        new = list(a)
        new[at:at + n] = part
    elif base in ("perm2i128", "perm2f128"):
        a, b, k = A(), B(), imm(rest[-1])
        table = lanes128(a) + lanes128(b)
        new = []
        for half in (k & 0xF, k >> 4):
            new += [BitVecVal(0, 32)] * 4 if half & 8 else table[half & 3]
    elif base == "permq":
        s, k = src_lanes(m, srcs[0], w), imm(srcs[-1])
        q = [s[i:i + 2] for i in range(0, len(s), 2)]
        new = []
        for blk in range(0, len(q), 4):
            for j in range(4):
                new += q[blk + ((k >> (2 * j)) & 3)]
    elif base == "shufi32x4":
        a, b, k = A(), B(), imm(rest[-1])
        la, lb = lanes128(a), lanes128(b)
        if w == 16:
            new = la[k & 3] + la[(k >> 2) & 3] + lb[(k >> 4) & 3] + lb[(k >> 6) & 3]
        else:
            new = la[k & 1] + lb[(k >> 1) & 1]
    elif base in ("permt2d", "permi2d"):
        idx = old if base == "permi2d" else A()
        tables = (A(), B()) if base == "permi2d" else (old, B())
        bits = 4 if w == 16 else 3
        new = []
        for j in range(w):
            c = concrete(idx[j])
            if c is None:
                raise Unproved(f"{mnem} with indices that are not constant")
            new.append(tables[(c >> bits) & 1][c & (w - 1)])
    else:
        raise Unproved(f"no model of {mnem} {', '.join(ops)}")
    if mask is not None:
        new = [If(kbit(m, mask, j), y, BitVecVal(0, 32) if zeroing else x) for j, (x, y) in enumerate(zip(old, new))]
    m.vwrite(dst, new, vex)
    return nxt


EXECUTED = set()


def run(m, entry, max_steps=2_000_000):
    """Execute from `entry` to the return through the entry's return
    address."""
    pc = entry
    while True:
        m.steps += 1
        if m.steps > max_steps:
            raise Unproved("too many steps")
        if pc not in m.code:
            raise Unproved(f"execution reached {pc:#x}, outside the code under proof")
        mnem, ops, nxt = m.code[pc]
        EXECUTED.add((mnem, tuple(ops)))
        pc = step(m, pc, mnem, ops, nxt)
        if pc == "ret":
            to = m.load(m.x["rsp"], 8)
            m.x["rsp"] = m.x["rsp"] + 8
            if isinstance(to, Ptr) and to.region == "return":
                return m
            raise Unproved(f"return to {to}")
