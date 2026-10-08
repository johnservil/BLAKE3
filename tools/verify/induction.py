"""Proofs for every count: induction over a loop (tools/verify/README.md).

A kernel or Rust path proved at each count up to a bound leaves the counts
past it unproved. Here a loop is proved for every count by induction: two
successive arrivals at its head are generalized into one state at a
symbolic iteration K (each position equal in both kept, each that differs
by a constant becoming start + K x difference, each other one an unknown);
then every path from that state is run, splitting where the lengths leave
a branch open, and each must either return to the head in the state for
K + 1 or leave the loop with its output right. Positions left unknown are
sound: a path whose result depends on one fails its comparison.
"""

import copy

import canon

from z3 import BitVec, BitVecVal, ZeroExt, Extract, Not, ULE, UGE, simplify, is_bv_value, eq
import aarch64
from aarch64 import Ptr, Unproved, Undecided


def snapshot(m):
    """An independent copy of a machine's state (its code shared)."""
    c = copy.copy(m)
    c.x = dict(m.x)
    c.v = {k: list(v) for k, v in m.v.items()}
    c.regions = {}
    for name, r in m.regions.items():
        r2 = copy.copy(r)
        r2.bytes = dict(r.bytes)
        c.regions[name] = r2
    c.assumptions = list(m.assumptions)
    c.lengths = set(m.lengths)
    if getattr(m, "za", None) is not None:
        c.za = [[list(row) for row in tile] for tile in m.za]
    if hasattr(m, "p"):
        c.p = {k: (list(v) if v is not None else None) for k, v in m.p.items()}
    return c


def paths(m, pc, heads):
    """Every path from `pc`: (machine, "head" or "return"), splitting where
    a branch on lengths is open. `heads`: the pcs at which a path stops."""
    work, done = [(m, pc, True)], []
    while work:
        m, pc, first = work.pop()
        while True:
            if pc in heads and not first:
                done.append((m, "head"))
                break
            first = False
            if pc not in m.code:
                raise Unproved(f"execution reached {pc:#x}, outside the code under proof")
            mnem, ops = m.code[pc]
            aarch64.EXECUTED.add((mnem, tuple(ops)))
            try:
                nxt = aarch64.step(m, pc, mnem, ops)
            except Undecided as e:
                for answer in (e.cond, Not(e.cond)):
                    b = snapshot(m)
                    b.assumptions.append(answer)
                    work.append((b, pc, True))      # resume at the split, without stopping
                break
            m.steps += 1
            if nxt == "ret":
                to = m.x[30]
                if isinstance(to, Ptr) and to.region == "return":
                    done.append((m, "return"))
                    break
                if not (isinstance(to, Ptr) and to.region == "code"):
                    raise Unproved(f"return to {to}")
                nxt = to.offset
            pc = nxt if nxt is not None else pc + 4
    return done


def run_to(m, pc, head):
    """Run `m` from `pc` to its next arrival at `head` (one path only)."""
    ends = paths(m, pc, {head})
    if len(ends) != 1 or ends[0][1] != "head":
        raise Unproved("the run to the loop head splits or returns")
    return ends[0][0]


class Generalized:
    """Two arrivals at a loop head made one state at iteration K."""
    def __init__(self, s0, s1, k):
        self.k = k
        self.wild = 0
        self.x = {r: self.unify(s0.x[r], s1.x[r]) for r in s0.x}
        self.v = {r: [self.unify(a, b) for a, b in zip(s0.v[r], s1.v[r])] for r in s0.v}
        # SME2's state: the ZA tiles and the predicates.
        self.za = None
        if getattr(s0, "za", None) is not None and getattr(s1, "za", None) is not None:
            self.za = [[[self.unify(a, b) for a, b in zip(r0, r1)] for r0, r1 in zip(t0, t1)]
                       for t0, t1 in zip(s0.za, s1.za)]
        self.p = None
        if hasattr(s0, "p") and hasattr(s1, "p"):
            self.p = {k: (s0.p[k] if s0.p[k] == s1.p.get(k) and all(isinstance(x, bool) for x in (s0.p[k] or [])) else None)
                      for k in s0.p}
        self.mem = {}
        for name, r0 in s0.regions.items():
            if not r0.writable or name not in s1.regions:
                continue
            r1 = s1.regions[name]
            self.mem[name] = {}
            for off in set(r0.bytes) | set(r1.bytes):
                if off in r0.bytes and off in r1.bytes and same_term(r0.bytes[off], r1.bytes[off]):
                    self.mem[name][off] = r0.bytes[off]
                elif off in r0.bytes or off in r1.bytes:
                    self.mem[name][off] = None      # differs: unknown at K
        # 8-byte words that differ by a constant (a spilled counter or pointer).
        for name, mem in self.mem.items():
            r0, r1 = s0.regions[name], s1.regions[name]
            for off in [o for o, b in mem.items() if b is None and isinstance(o, int) and o % 8 == 0]:
                w0, w1 = word(r0, off), word(r1, off)
                if w0 is None or w1 is None:
                    continue
                g = self.unify(w0, w1, wildcard=False)
                if g is not None:
                    for i in range(8):
                        mem[off + i] = ("word", g, i)

    def forget(self, pos):
        """Make a position unknown in the invariant."""
        kind = pos[0]
        if kind == "x":
            self.x[pos[1]] = ("unknown", 64)
        elif kind == "v":
            self.v[pos[1]][pos[2]] = ("unknown", 32)
        elif kind == "za":
            self.za[pos[1]][pos[2]][pos[3]] = ("unknown", 32)
        elif kind == "p":
            self.p[pos[1]] = None
        elif kind == "mem":
            self.mem[pos[1]][pos[2]] = None
        elif kind == "word":
            for i in range(8):
                self.mem[pos[1]][pos[2] + i] = None

    def fresh(self, size):
        self.wild += 1
        return BitVec(f"unknown{self.wild}", size)

    def unify(self, a, b, wildcard=True):
        """The general form of a pair: kept, a + K x d, or an unknown."""
        if isinstance(a, Ptr) and isinstance(b, Ptr) and a.region == b.region:
            if isinstance(a.offset, int) and isinstance(b.offset, int):
                return a if a.offset == b.offset else ("ptr", a.region, a.offset, b.offset - a.offset)
            return None if not wildcard else ("unknown", 64)
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            return ("unknown", 64) if wildcard else None
        if same_term(a, b):
            return a
        d = constant_difference(a, b)
        if d is not None:
            return ("step", a, d)
        return ("unknown", a.size()) if wildcard else None

    def at(self, kval):
        """The value forms at iteration `kval` (a 64-bit term)."""
        def inst(g, size=64):
            if isinstance(g, tuple):
                if g[0] == "ptr":
                    return Ptr(g[1], simplify(BitVecVal(g[2], 64) + kval * g[3], som=True, bv_sort_ac=True))
                if g[0] == "step":
                    a, d = g[1], g[2]
                    kk = kval if a.size() == 64 else Extract(a.size() - 1, 0, kval)
                    return simplify(a + kk * d, som=True, bv_sort_ac=True)
                if g[0] == "unknown":
                    return None
            return g
        return inst

    def machine(self, template, kval):
        """A machine at the head, iteration `kval`, unknowns as fresh symbols."""
        m = aarch64_snapshot_blank(template)
        inst = self.at(kval)
        for r, g in self.x.items():
            v = inst(g)
            m.x[r] = v if v is not None else self.fresh(64)
        for r, lanes in self.v.items():
            m.v[r] = [(inst(g) if inst(g) is not None else self.fresh(32)) for g in lanes]
        for name, mem in self.mem.items():
            region = m.regions[name]
            region.bytes = {}
            words = {}
            for off, b in mem.items():
                if isinstance(b, tuple) and b[0] == "word":
                    w = words.setdefault(off - b[2], inst(b[1]))
                    region.bytes[off] = Extract(8 * b[2] + 7, 8 * b[2], w) if not isinstance(w, Ptr) else (w, b[2])
                elif b is not None:
                    region.bytes[off] = b
                # unknown bytes stay unwritten: a read of one fails the proof
        m.nzcv = None
        if self.za is not None:
            m.za = [[[(inst(g) if inst(g) is not None else self.fresh(32)) for g in row] for row in tile] for tile in self.za]
        if self.p is not None:
            m.p = {k: (list(v) if v is not None else None) for k, v in self.p.items()}
        return m


_mixes = {}


def mixes(t):
    """Whether `t` holds an xor or a rotation (BLAKE3's mixing)."""
    from z3 import Z3_OP_BXOR, Z3_OP_ROTATE_LEFT, Z3_OP_ROTATE_RIGHT, Z3_OP_EXT_ROTATE_LEFT, Z3_OP_EXT_ROTATE_RIGHT
    k = t.get_id()
    r = _mixes.get(k)
    if r is None:
        kinds = (Z3_OP_BXOR, Z3_OP_ROTATE_LEFT, Z3_OP_ROTATE_RIGHT, Z3_OP_EXT_ROTATE_LEFT, Z3_OP_EXT_ROTATE_RIGHT)
        r = _mixes[k] = (t, t.decl().kind() in kinds or any(mixes(c) for c in t.children()))
    return r[1]


def constant_difference(a, b):
    """`b - a` when it is the same for every value of the symbols, else None:
    by simplification, or by the solver (byte shuffles of a sum hide a
    constant difference from the simplifier). Values at random points first
    settle cheaply most differences that are not constant; terms holding
    BLAKE3's mixing are not tried."""
    import random
    from z3 import Solver, unsat, substitute, BitVecVal
    d = aarch64.concrete(b - a)
    if d is not None:
        return d
    # Positions holding a hash's state (its xors and rotations) change by no
    # constant: only data moved or added to is worth the solver.
    if mixes(a) or mixes(b):
        return None
    diff = simplify(b - a)
    syms = {}
    canon.symbols_of(diff, syms, set())
    values = set()
    for _ in range(3):
        point = [(v, BitVecVal(random.getrandbits(v.size()), v.size())) for v in syms.values()]
        values.add(simplify(substitute(diff, *point)).as_long())
        if len(values) > 1:
            return None
    value = values.pop()
    s = Solver()
    s.add(diff != BitVecVal(value, diff.size()))
    return value if s.check() == unsat else None


def aarch64_snapshot_blank(template):
    m = snapshot(template)
    return m


def word(r, off):
    parts = [r.bytes.get(off + i) for i in range(8)]
    if any(p is None for p in parts):
        return None
    if all(isinstance(p, tuple) for p in parts):
        return parts[0][0] if all(p[1] == i and p[0] is parts[0][0] for i, p in enumerate(parts)) else None
    if any(isinstance(p, tuple) for p in parts):
        return None
    from z3 import Concat
    return simplify(Concat(*reversed(parts)))


def same_term(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return a == b if not (isinstance(a, tuple) and isinstance(b, tuple)) else (a[1] == b[1] and same_ptr(a[0], b[0]))
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        return isinstance(a, Ptr) and isinstance(b, Ptr) and same_ptr(a, b)
    return eq(a, b) or eq(simplify(a), simplify(b))


def same_ptr(a, b):
    return a.region == b.region and aarch64.key(a.offset) == aarch64.key(b.offset)


def forced(m, term):
    """`term`'s value when the run's assumptions force one (a model's value,
    and no other possible), else None."""
    from z3 import Solver, sat
    s = Solver()
    s.add(*m.assumptions)
    if s.check() != sat:
        return None
    v = s.model().eval(term, model_completion=True)
    return v.as_long() if aarch64.holds(m, term == v) else None
