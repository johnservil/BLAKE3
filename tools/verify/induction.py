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
        d = aarch64.concrete(b - a)
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
        return m


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
