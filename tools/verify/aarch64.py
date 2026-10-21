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
                SignExt, ULT, simplify, is_bv_value, BoolRef, Not, And, Or, If, UGE, ULE, Solver, unsat,
                is_true, is_false, Z3_OP_UNINTERPRETED, Z3_OP_EXTRACT, Z3_OP_ZERO_EXT, Z3_OP_CONCAT)


class Unproved(Exception):
    """The run met something it cannot or must not do: the proof fails."""


class Undecided(Exception):
    """A branch on lengths that the run's assumptions leave open: the prover
    splits the run there, one path for each answer (induction.py)."""
    def __init__(self, cond):
        super().__init__("a branch the assumptions leave open")
        self.cond = cond


def term_symbols(term, out=None, seen=None):
    """The names of the uninterpreted constants in `term`."""
    out = set() if out is None else out
    seen = set() if seen is None else seen
    if term.get_id() in seen:
        return out
    seen.add(term.get_id())
    if term.num_args() == 0 and term.decl().kind() == Z3_OP_UNINTERPRETED:
        out.add(term.decl().name())
    for c in term.children():
        term_symbols(c, out, seen)
    return out


def lengths_only(m, term):
    """Whether `term` depends on the run's length symbols alone (never on
    data): only such terms may steer a branch or an address."""
    return bool(m.lengths) and term_symbols(term) <= m.lengths


def decide(m, cond):
    """A condition's truth: concrete, or settled by the run's assumptions
    for a condition on lengths; otherwise Undecided (on lengths) or Unproved
    (on data)."""
    if isinstance(cond, bool):
        return cond
    c = simplify(cond)
    if is_true(c):
        return True
    if is_false(c):
        return False
    if not lengths_only(m, c):
        raise Unproved("a branch depends on data")
    for answer, test in ((True, Not(c)), (False, c)):
        s = Solver()
        s.add(*m.assumptions, test)
        if s.check() == unsat:
            return answer
    raise Undecided(c)


def holds(m, cond):
    """Whether `cond` (on lengths) follows from the run's assumptions."""
    s = Solver()
    s.add(*m.assumptions, Not(cond))
    return s.check() == unsat


def key(offset):
    """A memory location's key: a concrete offset, or a symbolic one in
    normal form (a sum of products of length symbols and constants)."""
    if isinstance(offset, int):
        return offset
    t = simplify(offset, som=True, bv_sort_ac=True)
    return t.as_long() if is_bv_value(t) else t.sexpr()


def as_bv(offset):
    return BitVecVal(offset % (1 << 64), 64) if isinstance(offset, int) else offset


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
        if isinstance(self.offset, int) and isinstance(n, int):
            return Ptr(self.region, self.offset + n)
        return Ptr(self.region, simplify(as_bv(self.offset) + as_bv(n), som=True, bv_sort_ac=True))

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



def bits(hi, lo, t):
    """Bits hi..lo of t, taken through the Concat, ZeroExt and Extract
    terms it is built of: a d word packed in a register's top half
    (bfi) and taken out again (lsr) is the word itself, as the normal
    form needs it (canon.py works on 32-bit words)."""
    if lo == 0 and hi == t.size() - 1:
        return t
    kind = t.decl().kind()
    if kind == Z3_OP_EXTRACT:
        _, l2 = t.params()
        return bits(hi + l2, lo + l2, t.arg(0))
    if kind == Z3_OP_ZERO_EXT:
        u = t.arg(0)
        if lo >= u.size():
            return BitVecVal(0, hi - lo + 1)
        if hi < u.size():
            return bits(hi, lo, u)
        return Concat(BitVecVal(0, hi - u.size() + 1), bits(u.size() - 1, lo, u))
    if kind == Z3_OP_CONCAT:
        parts, top = [], t.size()
        for k in range(t.num_args()):
            a = t.arg(k)
            plo = top - a.size()
            if lo <= top - 1 and hi >= plo:
                parts.append(bits(min(hi, top - 1) - plo, max(lo, plo) - plo, a))
            top = plo
        return parts[0] if len(parts) == 1 else Concat(*parts)
    return Extract(hi, lo, t)

class Machine:
    def __init__(self, code, regions):
        """`code`: {address: (mnemonic, operands)}; `regions`: {name: Region}."""
        self.code, self.regions = code, regions
        self.x = {}          # general registers, 64-bit terms or Ptr
        self.v = {}          # vector registers: four 32-bit lane terms, lane 0 first
        self.nzcv = None     # flags (n, z, c, v): booleans, or Z3 conditions on lengths
        self.steps = 0
        self.lengths = set()     # symbols that may steer branches and addresses (induction.py)
        self.assumptions = []    # what the run assumes of them

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
        return bits(31, 0, val)

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
        if isinstance(addr.offset, int) and isinstance(r.size, int):
            inside = lo <= addr.offset and addr.offset + n <= r.size
        else:
            # Offsets and sizes on lengths: in bounds under the run's assumptions.
            off = as_bv(addr.offset)
            inside = holds(self, And(UGE(off, as_bv(lo)), ULE(off + n, as_bv(r.size)), ULE(off, off + n)))
        if not inside:
            raise Unproved(f"{'write' if write else 'read'} of {n} bytes at {addr} outside [{lo}, {r.size})")
        if write and not r.writable:
            raise Unproved(f"write to read-only {r.name}")
        return r

    def load(self, addr, n):
        r = self.access(addr, n, False)
        parts = []
        for i in range(n):
            k = key((addr + i).offset)
            b = r.bytes.get(k)
            if b is None:
                if r.initial is None:
                    raise Unproved(f"read of unwritten {addr + i}")
                if not isinstance(k, int):
                    # A symbolic offset: a region whose initial bytes depend on it.
                    if not getattr(r, "symbolic_initial", False):
                        raise Unproved(f"read of unwritten {addr + i}")
                    b = r.initial((addr + i).offset)
                else:
                    b = r.initial(k)
            parts.append(b)
        if all(isinstance(p, tuple) for p in parts) and n == 8:
            ptrs = {(p[0].region, p[0].offset) for p in parts}
            if len(ptrs) == 1 and [p[1] for p in parts] == list(range(8)):
                return parts[0][0]
        if any(isinstance(p, tuple) for p in parts):
            raise Unproved(f"partial read of a stored pointer at {addr}")
        return Concat(*reversed(parts)) if n > 1 else parts[0]

    def load_raw(self, addr):
        """The byte at `addr` as memory holds it: data, or a stored pointer's piece."""
        r = self.access(addr, 1, False)
        k = key(addr.offset)
        b = r.bytes.get(k)
        if b is None:
            if r.initial is None or not isinstance(k, int):
                raise Unproved(f"read of unwritten {addr}")
            b = r.initial(k)
        return b

    def store(self, addr, val, n):
        r = self.access(addr, n, True)
        if isinstance(val, Ptr):
            assert n == 8, "a pointer stored in 8 bytes"
            for i in range(8):
                r.bytes[key((addr + i).offset)] = (val, i)
            return
        for i in range(n):
            r.bytes[key((addr + i).offset)] = Extract(8 * i + 7, 8 * i, val)


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


def image_bytes(obj):
    """The loaded sections' bytes, {address: int}: the image a shared
    library maps, for its read-only constants. Sections that are not
    loaded (debug information) overlap these addresses and are left out."""
    heads = subprocess.run(["objdump", "-h", obj], capture_output=True, text=True, check=True).stdout.splitlines()
    loaded = [heads[i].split()[1] for i in range(len(heads) - 1)
              if re.match(r"\s+\d+ ", heads[i]) and "ALLOC" in heads[i + 1] and "CONTENTS" in heads[i + 1]]
    out = subprocess.run(["objdump", "-s"] + [f"-j{n}" for n in loaded] + [obj], capture_output=True, text=True, check=True).stdout
    data = {}
    for line in out.splitlines():
        mt = re.match(r"\s([0-9a-f]+) ((?:[0-9a-f]{2,8} ?){1,4})", line)
        if not mt:
            continue
        addr = int(mt.group(1), 16)
        hexs = "".join(mt.group(2).split())
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


def _not(a):
    return (not a) if isinstance(a, bool) else Not(a)


def _and(a, b):
    return (a and b) if isinstance(a, bool) and isinstance(b, bool) else And(a, b)


def _or(a, b):
    return (a or b) if isinstance(a, bool) and isinstance(b, bool) else Or(a, b)


def _eq(a, b):
    return (a == b) if isinstance(a, bool) and isinstance(b, bool) else (a == b)


CONDS = {
    "eq": lambda n, z, c, v: z, "ne": lambda n, z, c, v: _not(z),
    "hs": lambda n, z, c, v: c, "cs": lambda n, z, c, v: c,
    "lo": lambda n, z, c, v: _not(c), "cc": lambda n, z, c, v: _not(c),
    "mi": lambda n, z, c, v: n, "pl": lambda n, z, c, v: _not(n),
    "hi": lambda n, z, c, v: _and(c, _not(z)), "ls": lambda n, z, c, v: _not(_and(c, _not(z))),
    "ge": lambda n, z, c, v: _eq(n, v), "lt": lambda n, z, c, v: _not(_eq(n, v)),
    "gt": lambda n, z, c, v: _and(_not(z), _eq(n, v)), "le": lambda n, z, c, v: _or(z, _not(_eq(n, v))),
}


def cond(m, name):
    """Condition `name` on the current flags, decided (see decide)."""
    if m.nzcv is None:
        raise Unproved(f"condition {name} before any comparison")
    return decide(m, CONDS[name](*m.nzcv))


def width(reg):
    return 32 if reg[0] == "w" else 64


# Every (mnemonic, operands) executed in this process, for cross_check.py.
EXECUTED = set()


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
        EXECUTED.add((mnem, tuple(ops)))
        nxt = step(m, pc, mnem, ops)
        if nxt == "ret":
            to = m.x[30]
            if not isinstance(to, Ptr):
                raise Unproved("return through a non-address")
            if to.region == "return":
                return m
            if to.region != "code":
                raise Unproved(f"return to {to}")
            nxt = to.offset
        pc = nxt if nxt is not None else pc + 4


def operand_value(m, op, w):
    """A register (with an optional shift) or an immediate, as a w-bit term."""
    if op.startswith("#"):
        return BitVecVal(imm(op) & ((1 << w) - 1), w)
    return m.reg(op)


def shifted(m, ops, w):
    """The last operand(s) of an arithmetic or logical instruction: `reg`,
    `#imm`, or either followed by `lsl #n`, `lsr #n`, `asr #n`, `ror #n`,
    or `uxtw`. Any other form stops the run: an operand the model does not
    read must never be read as something simpler."""
    if len(ops) == 1:
        return operand_value(m, ops[0], w)
    if len(ops) != 2:
        raise Unproved(f"operands not modelled: {', '.join(ops)}")
    kind, *amount = ops[1].split()
    if kind not in ("lsl", "lsr", "asr", "ror", "uxtw") or len(amount) > 1:
        raise Unproved(f"operand modifier not modelled: {ops[1]}")
    n = imm(amount[0]) if amount else 0
    v = operand_value(m, ops[0], w)
    if isinstance(v, Ptr):
        raise Unproved("shifted pointer")
    if kind == "lsl":
        return v << n
    if kind == "lsr":
        return LShR(v, n)
    if kind == "asr":
        return v >> n
    if kind == "ror":
        return RotateRight(v, n) if n % v.size() else v
    return ZeroExt(32, Extract(31, 0, v)) << n


def add_values(a, b, sub=False):
    if isinstance(a, Ptr):
        k = concrete(b)
        if k is None:
            if not lengths_only(_machine[0], b):
                raise Unproved(f"pointer {a} moved by a data-dependent amount")
            b = b if b.size() == 64 else ZeroExt(64 - b.size(), b)
            return Ptr(a.region, simplify(as_bv(a.offset) + (-b if sub else b), som=True, bv_sort_ac=True))
        if k >= 1 << (b.size() - 1):
            k -= 1 << b.size()      # two's complement: a negative step
        return a + (-k if sub else k)
    if isinstance(b, Ptr):
        if sub:
            raise Unproved("subtraction of a pointer")
        return add_values(b, a)
    return a - b if sub else a + b


def set_flags_sub(m, a, b, w):
    """NZCV of a - b: concrete, or (on lengths) Z3 conditions."""
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        if not (isinstance(a, Ptr) and isinstance(b, Ptr) and a.region == b.region):
            raise Unproved(f"comparison of {a} with {b}")
        if isinstance(a.offset, int) and isinstance(b.offset, int):
            x, y = a.offset % (1 << 64), b.offset % (1 << 64)
        else:
            return symbolic_flags(m, as_bv(a.offset), as_bv(b.offset), 64)
    else:
        x, y = concrete(a), concrete(b)
        if x is None or y is None:
            if not (lengths_only(m, a) or concrete(a) is not None) or not (lengths_only(m, b) or concrete(b) is not None):
                raise Unproved("a condition depends on data")
            return symbolic_flags(m, a, b, w)
    mask = (1 << w) - 1
    x, y = x & mask, y & mask
    r = (x - y) & mask
    top = 1 << (w - 1)
    m.nzcv = (bool(r & top), r == 0, x >= y, bool(((x ^ y) & (x ^ r)) & top))


def set_flags_add(m, a, b, w):
    """NZCV of a + b (Arm's AddWithCarry, carry in 0): concrete, or (on
    lengths) Z3 conditions."""
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        raise Unproved(f"flags of the sum {a} + {b}")
    x, y = concrete(a), concrete(b)
    top = 1 << (w - 1)
    if x is None or y is None:
        if not (lengths_only(m, a) or x is not None) or not (lengths_only(m, b) or y is not None):
            raise Unproved("a condition depends on data")
        x, y = Extract(w - 1, 0, a) if a.size() > w else a, Extract(w - 1, 0, b) if b.size() > w else b
        r = x + y
        hi = lambda t: Extract(w - 1, w - 1, t) == 1
        m.nzcv = (hi(r), r == 0, ULT(r, x), hi(~(x ^ y) & (x ^ r)))
        return
    mask = (1 << w) - 1
    x, y = x & mask, y & mask
    r = (x + y) & mask
    m.nzcv = (bool(r & top), r == 0, x + y > mask, bool(~(x ^ y) & (x ^ r) & top))


def symbolic_flags(m, a, b, w):
    """NZCV of a - b on lengths, as Z3 conditions."""
    x, y = Extract(w - 1, 0, a) if a.size() > w else a, Extract(w - 1, 0, b) if b.size() > w else b
    r = x - y
    top = lambda t: Extract(w - 1, w - 1, t) == 1
    m.nzcv = (top(r), r == 0, UGE(x, y), top((x ^ y) & (x ^ r)))


# The machine whose step is running (add_values reads its length symbols).
_machine = [None]


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
        # A register offset: an x register, `lsl #n` or none; or a w
        # register, `uxtw` with or without `#n`. Any other form stops the run.
        ext = parts[2].split() if len(parts) > 2 else []
        if len(parts) > 3 or (ext and not (ext[0] == "lsl" and parts[1][0] == "x" or ext[0] == "uxtw" and parts[1][0] == "w")) or len(ext) > 2:
            raise Unproved(f"address form not modelled: {s}")
        if parts[1][0] == "w" and not ext:
            raise Unproved(f"address form not modelled: {s}")
        idx = m.reg(parts[1]) if parts[1][0] == "x" else ZeroExt(32, m.reg(parts[1]))
        sh = imm(ext[1]) if len(ext) == 2 else 0
        k_ = concrete(idx)
        if k_ is None:
            raise Unproved("an address depends on data")
        off = k_ << sh
    if len(ops) > k + 1 and ops[k + 1].startswith("#"):
        post = imm(ops[k + 1])
        return add_values(base, BitVecVal(0, 64)), parts[0], add_values(base, BitVecVal(post, 64))
    addr = add_values(base, BitVecVal(off % (1 << 64), 64))
    if pre:
        # Pre-indexed: the base moves first (so a push below sp is inside
        # the new frame), and there is nothing left to write back.
        m.set(parts[0], addr)
    return addr, None, addr


_clobbers = [0]


def clobber(m, keep_x0=False):
    """After a call by contract: the registers the calling convention lets a
    callee change (x0-x18 but what the contract returns, and v0-v7, v16-v31,
    and d8-d15's upper lanes) hold fresh values, and the flags none."""
    _clobbers[0] += 1
    n = _clobbers[0]
    for i in range(1 if keep_x0 else 0, 19):
        m.x[i] = BitVec(f"clobbered{n}_x{i}", 64)
    for i in list(range(0, 8)) + list(range(16, 32)):
        m.v[i] = [BitVec(f"clobbered{n}_v{i}_{l}", 32) for l in range(16)]
    for i in range(8, 16):
        m.v[i] = m.v[i][:2] + [BitVec(f"clobbered{n}_v{i}_{l}", 32) for l in range(2, 16)]
    m.nzcv = None


def library_call(m, name):
    """The C library functions compiled Rust calls, by their contracts:
    memcpy and memmove copy x2 bytes (a concrete count) from x1 to x0, and
    memset writes x2 copies of x1's low byte at x0; each returns x0. The
    buffers are checked against their regions."""
    if name == "memset":
        # memset(dst, byte, n): n (concrete) copies of the low byte.
        n = concrete(m.x[2])
        if n is None:
            raise Unproved("memset of a data-dependent length")
        byte = Extract(7, 0, m.x[1]) if not isinstance(m.x[1], Ptr) else None
        if byte is None:
            raise Unproved("memset with a pointer for its byte")
        for i in range(n):
            m.store(m.x[0] + i, byte, 1)
        return
    if name == "pthread_key_create":
        # pthread_key_create(key, destructor): a new key to *key (1: std
        # keeps 0 for "none yet"); 0.
        m.store(m.x[0], BitVecVal(1, 32), 4)
        m.x[0] = BitVecVal(0, 64)
        return
    if name in ("pthread_setspecific", "pthread_getspecific"):
        # This thread's value for a key (concrete): set (returning 0) or got
        # (none yet: 0).
        k = concrete(m.x[0])
        if k is None:
            raise Unproved(f"{name} of a data-dependent key")
        table = m.__dict__.setdefault("specific", {})
        if name == "pthread_setspecific":
            table[k] = m.x[1]
            m.x[0] = BitVecVal(0, 64)
        else:
            m.x[0] = table.get(k, BitVecVal(0, 64))
        return
    if name in ("posix_memalign", "malloc"):
        # posix_memalign(memptr, align, size) and malloc(size): a new block
        # of `size` bytes (concrete), its base aligned to `align` (malloc's
        # 16; a power of two up to a page), its bytes indeterminate (each a
        # fresh value, data: one that steered the run would stop it); its
        # address to *memptr and 0, or returned. (A failed allocation aborts
        # the process: Rust's allocator never returns it.)
        align, size = (concrete(m.x[1]), concrete(m.x[2])) if name == "posix_memalign" else (16, concrete(m.x[0]))
        if size is None or align is None or align & (align - 1) or align > 4096:
            raise Unproved(f"{name} of a data-dependent size or alignment")
        m.blocks = getattr(m, "blocks", 0) + 1
        block = f"heap{m.blocks}"
        m.regions[block] = Region(block, size, True, (lambda b: lambda o: BitVec(f"{b}_{o}", 8))(block))
        m.regions[block].align = align
        if name == "malloc":
            m.x[0] = Ptr(block, 0)
        else:
            m.store(m.x[0], Ptr(block, 0), 8)
            m.x[0] = BitVecVal(0, 64)
        return
    if name == "free":
        # free(p): a block (or null) given back; the run reads it no more.
        p = m.x[0]
        if not (isinstance(p, Ptr) and p.region.startswith("heap") and p.offset == 0 or concrete(p) == 0):
            raise Unproved(f"free of {p}")
        if isinstance(p, Ptr):
            del m.regions[p.region]
        return
    if name not in ("memcpy", "memmove"):
        raise Unproved(f"a call of {name}, which is not modelled")
    n = concrete(m.x[2])
    if n is None:
        raise Unproved(f"{name} of a data-dependent length")
    dst, src = m.x[0], m.x[1]
    # Byte by byte as memory holds them: a stored pointer's pieces copied as pieces.
    data = [m.load_raw(src + i) for i in range(n)]
    if n:
        m.access(dst, n, True)
    for i, b in enumerate(data):
        m.regions[dst.region].bytes[key((dst + i).offset)] = b
    m.x[0] = dst


# Loads and stores with an unscaled offset: the same access as the scaled form.
UNSCALED = {"ldur": "ldr", "stur": "str", "ldurb": "ldrb", "sturb": "strb", "ldurh": "ldrh", "sturh": "strh",
            # Acquire and release order other threads' views; one thread's run sees plain accesses.
            "ldar": "ldr", "stlr": "str", "ldarb": "ldrb", "stlrb": "strb",
            # An exclusive load: a plain one (its store, below, always succeeds).
            "ldxr": "ldr", "ldaxr": "ldr", "ldxrb": "ldrb", "ldaxrb": "ldrb"}


def step(m, pc, mnem, ops):
    _machine[0] = m
    mnem = UNSCALED.get(mnem, mnem)
    w = width(ops[0]) if ops and ops[0][0] in "wx" else 64
    if mnem == "nop":
        return None
    if mnem in ("add", "sub", "adds", "subs") and ops[0][0] in "wxs":
        a = m.reg(ops[1])
        b = shifted(m, ops[2:], w)
        r = add_values(a, b, sub=mnem.startswith("sub"))
        if mnem.endswith("s"):
            (set_flags_sub if mnem.startswith("sub") else set_flags_add)(m, a, b, w)
        m.set(ops[0], r)
        return None
    if mnem == "cmp":
        set_flags_sub(m, m.reg(ops[0]), shifted(m, ops[1:], w), w)
        return None
    if mnem == "cmn":
        # The flags of a + b, as adds sets them.
        set_flags_add(m, m.reg(ops[0]), shifted(m, ops[1:], w), w)
        return None
    if mnem in ("umull", "umaddl") and ops[0][0] == "x":
        # The 64-bit product of two w registers, unsigned (plus an x register).
        a, b = m.reg(ops[1]), m.reg(ops[2])
        p = ZeroExt(32, a) * ZeroExt(32, b)
        m.set(ops[0], p if mnem == "umull" else add_values(m.reg(ops[3]), p))
        return None
    if mnem in ("mul", "madd", "msub"):
        a, b = m.reg(ops[1]), m.reg(ops[2])
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unproved(f"{mnem} of a pointer")
        if mnem == "mul":
            m.set(ops[0], a * b)
        else:
            m.set(ops[0], add_values(m.reg(ops[3]), a * b, sub=mnem == "msub"))
        return None
    if mnem == "and" and ops[1][0] in "wx" and isinstance(m.reg(ops[1]), Ptr) and len(ops) == 3 and ops[2].startswith("#"):
        # A stack or heap address aligned down to a power of two up to its
        # base's alignment (the stack's a page, a block's what it asked
        # for), so the offset aligns the same way.
        a, mask = m.reg(ops[1]), imm(ops[2]) % (1 << 64)
        align = (1 << 64) - mask
        base_align = 4096 if a.region == "stack" else getattr(m.regions.get(a.region), "align", 0)
        if not (isinstance(a.offset, int) and align & (align - 1) == 0 and align <= base_align):
            raise Unproved(f"and of the pointer {a} with {ops[2]}")
        aligned = Ptr("stack", a.offset & mask)
        if ops[0] == "sp":
            m.x["sp"] = aligned
        else:
            m.set(ops[0], aligned)
        return None
    if mnem in ("eor", "orr", "and") and ops[0][0] in "wx":
        a, b = m.reg(ops[1]), shifted(m, ops[2:], w)
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unproved(f"{mnem} on a pointer")
        if mnem == "orr" and lengths_only(m, a) and concrete(b) is not None and holds(m, a & b == 0):
            # Bits in common none (under the assumptions): the or is the sum,
            # the form offsets take elsewhere (key).
            m.set(ops[0], a + b)
            return None
        m.set(ops[0], {"eor": a ^ b, "orr": a | b, "and": a & b}[mnem])
        return None
    if mnem == "ror":
        m.set(ops[0], RotateRight(m.reg(ops[1]), imm(ops[2])))
        return None
    if mnem == "bfxil":
        lsb, n = imm(ops[2]), imm(ops[3])
        old, src = m.reg(ops[0]), m.reg(ops[1])
        field = Extract(lsb + n - 1, lsb, src)
        m.set(ops[0], Concat(Extract(w - 1, n, old), field) if n < w else field)
        return None
    if mnem in ("tst", "ands"):
        # An AND that sets the flags (n and z from the result, c and v clear),
        # which must be concrete, as cmp's are. tst is ands into the zero register.
        first = 0 if mnem == "tst" else 1
        a, b = m.reg(ops[first]), shifted(m, ops[first + 1:], w)
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unproved(f"{mnem} of a pointer")
        r = concrete(a & b)
        if r is not None:
            m.nzcv = (bool(r >> (w - 1) & 1), r == 0, False, False)
            result = BitVecVal(r, w)
        elif lengths_only(m, a & b):
            result = a & b
            m.nzcv = (Extract(w - 1, w - 1, result) == 1, result == 0, False, False)
        else:
            raise Unproved("a condition depends on data")
        if mnem == "ands":
            m.set(ops[0], result)
        return None
    if mnem == "neg":
        v = m.reg(ops[1])
        if isinstance(v, Ptr):
            raise Unproved("a negated pointer")
        m.set(ops[0], -v)
        return None
    if mnem == "cset":
        if m.nzcv is None:
            raise Unproved("cset before any comparison")
        m.set(ops[0], BitVecVal(1 if cond(m, ops[1]) else 0, w))
        return None
    if mnem in ("ccmp", "ccmn"):
        # If the condition holds on the current flags, they become those of
        # the comparison (ccmn: of the sum); otherwise the immediate's (n z c v
        # from bit 3 down).
        if m.nzcv is None:
            raise Unproved(f"{mnem} before any comparison")
        if cond(m, ops[3]):
            b = operand_value(m, ops[1], w)
            (set_flags_sub if mnem == "ccmp" else set_flags_add)(m, m.reg(ops[0]), b, w)
        else:
            k = imm(ops[2])
            m.nzcv = (bool(k & 8), bool(k & 4), bool(k & 2), bool(k & 1))
        return None
    if mnem == "bfi":
        lsb, n = imm(ops[2]), imm(ops[3])
        old, src = m.reg(ops[0]), m.reg(ops[1])
        parts = []
        if lsb + n < w:
            parts.append(bits(w - 1, lsb + n, old))
        parts.append(bits(n - 1, 0, src))
        if lsb > 0:
            parts.append(bits(lsb - 1, 0, old))
        m.set(ops[0], Concat(*parts))
        return None
    if mnem in ("lsr", "lsl"):
        a = m.reg(ops[1])
        if ops[2].startswith("#"):
            n = imm(ops[2])
        else:
            # The amount in a register, taken modulo the width.
            n = m.reg(ops[2])
            if isinstance(n, Ptr):
                raise Unproved("a shift by a pointer")
            n = n & (w - 1)
        if mnem == "lsr" and isinstance(n, int) and 0 < n < w:
            m.set(ops[0], ZeroExt(n, bits(w - 1, n, a)))
        else:
            m.set(ops[0], LShR(a, n) if mnem == "lsr" else a << n)
        return None
    if mnem == "ubfx":
        lsb, n = imm(ops[2]), imm(ops[3])
        m.set(ops[0], ZeroExt(w - n, Extract(lsb + n - 1, lsb, m.reg(ops[1]))))
        return None
    if mnem == "ubfiz":
        lsb, n = imm(ops[2]), imm(ops[3])
        m.set(ops[0], ZeroExt(w - n, Extract(n - 1, 0, m.reg(ops[1]))) << lsb)
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
        m.set(ops[0], m.reg(ops[1]) if cond(m, ops[3]) else m.reg(ops[2]))
        return None
    if mnem == "clz":
        a = m.reg(ops[1])
        if isinstance(a, Ptr):
            raise Unproved("clz of a pointer")
        c = concrete(a)
        if c is not None:
            m.set(ops[0], BitVecVal(w - c.bit_length() if c else w, w))
        else:
            # The highest set bit, as a chain of cases.
            r = BitVecVal(w, w)
            for i in range(w):
                r = If(Extract(i, i, a) == 1, BitVecVal(w - 1 - i, w), r)
            m.set(ops[0], r)
        return None
    if mnem == "csinc":
        if m.nzcv is None:
            raise Unproved("csinc before any comparison")
        if cond(m, ops[3]):
            m.set(ops[0], m.reg(ops[1]))
        else:
            b = m.reg(ops[2])
            if isinstance(b, Ptr):
                raise Unproved("csinc of a pointer")
            m.set(ops[0], b + 1)
        return None
    if mnem == "cinc":
        # cinc d, n, c: n + 1 if c holds, else n.
        if m.nzcv is None:
            raise Unproved("cinc before any comparison")
        a = m.reg(ops[1])
        if cond(m, ops[2]):
            if isinstance(a, Ptr):
                raise Unproved("cinc of a pointer")
            a = a + 1
        m.set(ops[0], a)
        return None
    if mnem == "adr":
        m.set(ops[0], Ptr("text", int(ops[1].split()[0], 16)))
        return None
    if mnem == "adrp":
        # The page of a read-only constant: the image region (image_bytes).
        m.set(ops[0], Ptr("image", int(ops[1].split()[0], 16) & ~0xfff))
        return None
    if mnem == "mrs":
        # The clock the run fixes (`m.clock`: the virtual counter and its
        # frequency), and the thread pointer: the start of this thread's
        # TLS block (Variant I: its variables from offset 16).
        if ops[1] == "tpidr_el0":
            m.set(ops[0], Ptr("tls", 0))
        elif ops[1] in getattr(m, "clock", {}):
            m.set(ops[0], BitVecVal(m.clock[ops[1]], 64))
        else:
            raise Unproved(f"a read of {ops[1]}")
        return None
    if mnem == "prfm":
        # A prefetch: no effect the program can see.
        return None
    if mnem == "umulh":
        a, b = m.reg(ops[1]), m.reg(ops[2])
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unproved("umulh of a pointer")
        m.set(ops[0], Extract(127, 64, ZeroExt(64, a) * ZeroExt(64, b)))
        return None
    if mnem == "blr" and isinstance(m.reg(ops[0]), Ptr) and m.reg(ops[0]).region == "tlsdesc":
        # A TLS descriptor's resolver, as the dynamic linker binds it for
        # static TLS: x0, the descriptor, returns its second word, the
        # variable's offset from the thread pointer; no other register changes.
        m.set("x0", m.load(add_values(m.reg("x0"), BitVecVal(8, 64)), 8))
        return None
    if mnem == "br":
        # A jump through a computed code address (a jump table's target).
        t = m.reg(ops[0])
        if isinstance(t, Ptr) and not isinstance(t.offset, int):
            t = Ptr(t.region, concrete(t.offset))
        if not (isinstance(t, Ptr) and t.region in ("image", "text", "code") and isinstance(t.offset, int)
                and t.offset in m.code):
            raise Unproved(f"a jump through {t}, which is not code of the image")
        return t.offset
    if mnem == "blr":
        # A call through a pointer the image holds (a table of functions):
        # the function at that address, as a direct call.
        t = m.reg(ops[0])
        if not (isinstance(t, Ptr) and t.region in ("image", "text") and isinstance(t.offset, int) and t.offset in m.code):
            raise Unproved(f"a call through {t}, which is not a function of the image")
        contract = getattr(m, "contracts", {}).get(t.offset)
        if contract is not None:
            contract(m)
            clobber(m, keep_x0=True)
            return None
        m.x[30] = Ptr("code", pc + 4)
        return t.offset
    if mnem == "bl":
        target = int(ops[0].split()[0], 16)
        name = getattr(m, "plt", {}).get(target)
        if name is not None and name in getattr(m, "local", {}):
            # A call through the PLT of a function the library defines: the
            # dynamic linker binds it to that definition.
            m.x[30] = Ptr("code", pc + 4)
            return m.local[name]
        if name is not None:
            library_call(m, name)
            clobber(m, keep_x0=True)
            return None
        contract = getattr(m, "contracts", {}).get(target)
        if contract is not None:
            # The callee by its proved statement: the contract writes what
            # the callee writes and sets x0; the rest the convention lets
            # change is fresh.
            contract(m)
            clobber(m, keep_x0=True)
            return None
        m.x[30] = Ptr("code", pc + 4)
        return target
    if mnem in ("ldr", "ldrb", "ldrh", "str", "strb", "strh") and ops[0][0] in "wx":
        n = {"b": 1, "h": 2}.get(mnem[-1], w // 8)
        addr, wb, new = mem_operand(m, ops, 1)
        if mnem.startswith("ldr") and ops[0] in ("xzr", "wzr"):
            # A load that keeps nothing (a stack probe): the access alone,
            # checked against its region.
            m.access(addr, n, False)
            if wb:
                m.set(wb, new)
            return None
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
    if mnem in ("stxr", "stlxr", "stxrb", "stlxrb"):
        # An exclusive store: one thread's run holds the monitor, so it
        # stores and reports success (0).
        n = 1 if mnem.endswith("b") else width(ops[1]) // 8
        addr, _, _ = mem_operand(m, ops, 2)
        val = m.reg(ops[1])
        m.store(addr, val if isinstance(val, Ptr) or 8 * n == width(ops[1]) else Extract(8 * n - 1, 0, val), n)
        m.set(ops[0], BitVecVal(0, 32))
        return None
    if mnem == "ldrsw" and ops[0][0] == "x":
        addr, wb, new = mem_operand(m, ops, 1)
        val = m.load(addr, 4)
        if isinstance(val, Ptr):
            raise Unproved("a sign-extended load of a pointer")
        m.set(ops[0], SignExt(32, val))
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
        if mnem == "b" or cond(m, mnem[2:]):
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
        v = m.reg(ops[0])
        zero = False if isinstance(v, Ptr) else decide(m, v == 0)
        target = int(ops[1].split()[0], 16)
        return target if zero == (mnem == "cbz") else None
    if mnem == "ret":
        return "ret"
    import vector
    if vector.step(m, pc, mnem, ops):
        return None
    raise Unproved(f"instruction not modelled: {mnem} {', '.join(ops)} at {pc:#x}")
