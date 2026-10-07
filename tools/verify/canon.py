"""A normal form for the terms the kernels compute, so that a kernel's
output and the specification's are compared by identity, without search.

BLAKE3's compression is sums, xors, and rotations of 32-bit words; the
kernels add byte shuffles (loads, stores, spills, lane moves), which are
concatenations and extractions of whole words. `canon` flattens and sorts
sums and xors (xor cancelling pairs), keeps rotations, undoes byte
shuffles that put a word back together, and hands every small term with
no sum, xor, or rotation in it (the inputs: block words, key words, the
counter's halves, the flags) to Z3's simplifier, which settles those
reliably. Two terms with one normal form are equal; the converse does not
hold, so a mismatch is a failed proof, never a false one.
"""

import sys
from z3 import (Z3_OP_BADD, Z3_OP_BXOR, Z3_OP_ROTATE_RIGHT, Z3_OP_ROTATE_LEFT, Z3_OP_EXTRACT,
                Z3_OP_CONCAT, Z3_OP_ZERO_EXT, Z3_OP_EXT_ROTATE_RIGHT, Z3_OP_EXT_ROTATE_LEFT,
                Z3_OP_UNINTERPRETED, BitVecVal, Solver, unsat, substitute, simplify, is_bv_value)
import random

sys.setrecursionlimit(1_000_000)

_ids = {}          # normal-form tuple -> small integer
# Caches keyed by Z3's ast id hold the ast too: Z3 reuses the id of a
# freed ast, and a stale entry would equate two different terms.
_memo = {}         # z3 ast id -> (ast, normal form id)
_arith = {}        # z3 ast id -> (ast, whether it holds a sum, xor, or rotation)

ARITH = (Z3_OP_BADD, Z3_OP_BXOR, Z3_OP_ROTATE_RIGHT, Z3_OP_ROTATE_LEFT, Z3_OP_EXT_ROTATE_RIGHT, Z3_OP_EXT_ROTATE_LEFT)
_nodes = {}        # normal-form id -> its tuple


def node(n):
    return _nodes[n]


def intern(t):
    n = _ids.get(t)
    if n is None:
        n = _ids[t] = len(_ids)
        _nodes[n] = t
    return n


def has_arith(t):
    k = t.get_id()
    r = _arith.get(k)
    if r is None:
        r = (t, t.decl().kind() in ARITH or any(has_arith(c) for c in t.children()))
        _arith[k] = r
    return r[1]


_leaves = {}       # fingerprint -> [(simplified leaf, id)]
_values = {}       # symbol name -> random values, one per fingerprint round
ROUNDS = 3


def symbols_of(t, out, seen):
    if t.get_id() in seen:
        return
    seen.add(t.get_id())
    if t.num_args() == 0 and t.decl().kind() == Z3_OP_UNINTERPRETED:
        out[t.decl().name()] = t
    for c in t.children():
        symbols_of(c, out, seen)


def leaf(t):
    """A term with no rounds in it: equal leaves get one id. Leaves whose
    values agree at random points are proved equal by the solver; the
    random points only choose which pairs to try."""
    t = simplify(t)
    syms = {}
    symbols_of(t, syms, set())
    print_ = []
    for r in range(ROUNDS):
        sub = [(v, BitVecVal(_values.setdefault((n, r), random.getrandbits(v.size())) % (1 << v.size()), v.size()))
               for n, v in syms.items()]
        print_.append(simplify(substitute(t, *sub)).as_long() if sub else simplify(t).as_long())
    key = (t.size(), tuple(print_))
    for other, n in _leaves.get(key, []):
        if other.eq(t):
            return n
        s = Solver()
        s.set("timeout", 60_000)
        s.add(other != t)
        if s.check() == unsat:
            return n
    n = intern(("leaf", t.sexpr()))
    _leaves.setdefault(key, []).append((t, n))
    return n


def pieces(t):
    """`t` as a list of (term, hi, lo) bit ranges, most significant first,
    seeing through concatenations, extractions, and zero extensions."""
    kind = t.decl().kind()
    if kind == Z3_OP_CONCAT:
        return [p for c in t.children() for p in pieces(c)]
    if kind == Z3_OP_EXTRACT:
        hi, lo = t.params()
        out, pos = [], t.arg(0).size()
        for (x, h, l) in pieces(t.arg(0)):
            w = h - l + 1
            top, bot = pos - 1, pos - w        # this piece's bits in t.arg(0)
            pos -= w
            a, b = min(top, hi), max(bot, lo)
            if a >= b:
                out.append((x, l + (a - bot), l + (b - bot)))
        return out
    if kind == Z3_OP_ZERO_EXT:
        n = t.params()[0]
        from z3 import BitVecVal
        return [(BitVecVal(0, n), n - 1, 0)] + pieces(t.arg(0))
    return [(t, t.size() - 1, 0)]


def merged(ps):
    """Adjacent ranges of one term joined."""
    out = []
    for x, h, l in ps:
        if out and out[-1][0].eq(x) and out[-1][2] == h + 1:
            out[-1] = (x, out[-1][1], l)
        else:
            out.append((x, h, l))
    return out


def canon(t):
    k = t.get_id()
    r = _memo.get(k)
    if r is not None:
        return r[1]
    n = _canon(t)
    _memo[k] = (t, n)
    return n


def _canon(t):
    if not has_arith(t):
        return leaf(t)
    kind = t.decl().kind()
    if kind in (Z3_OP_CONCAT, Z3_OP_EXTRACT, Z3_OP_ZERO_EXT):
        ps = merged(pieces(t))
        if len(ps) == 1:
            x, h, l = ps[0]
            if l == 0 and h == x.size() - 1:
                return canon(x)
        return intern(("cat", tuple((canon(x), h, l) for x, h, l in ps)))
    if kind == Z3_OP_BADD:
        terms = []
        for c in t.children():
            n = canon(c)
            terms.extend(node(n)[1] if node(n)[0] == "add" else [n])
        return intern(("add", tuple(sorted(terms))))
    if kind == Z3_OP_BXOR:
        seen = {}
        for c in t.children():
            n = canon(c)
            for x in (node(n)[1] if node(n)[0] == "xor" else [n]):
                seen[x] = seen.get(x, 0) ^ 1
        odd = sorted(x for x, o in seen.items() if o)
        return odd[0] if len(odd) == 1 else intern(("xor", tuple(odd)))
    if kind in (Z3_OP_ROTATE_RIGHT, Z3_OP_ROTATE_LEFT, Z3_OP_EXT_ROTATE_RIGHT, Z3_OP_EXT_ROTATE_LEFT):
        if kind in (Z3_OP_EXT_ROTATE_RIGHT, Z3_OP_EXT_ROTATE_LEFT):
            amount = simplify(t.arg(1))
            if not is_bv_value(amount):
                raise ValueError("rotation by a variable amount")
            n = amount.as_long() % t.size()
        else:
            n = t.params()[0] % t.size()
        if kind in (Z3_OP_ROTATE_LEFT, Z3_OP_EXT_ROTATE_LEFT):
            n = (t.size() - n) % t.size()
        inner = canon(t.arg(0))
        if node(inner)[0] == "ror":
            n, inner = (n + node(inner)[1]) % t.size(), node(inner)[2]
        return inner if n == 0 else intern(("ror", n, inner))
    return intern(("op", t.decl().name(), tuple(t.params()), tuple(canon(c) for c in t.children())))
