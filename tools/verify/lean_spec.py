"""BLAKE3's compression function as the proofs' definition: the C2SP Lean
specification's own (c2sp/BLAKE3/Blake3.lean), as the graph Lean computes
from it over symbolic terms (tools/verify/lean/compress.json, written by
tools/verify/lean/emit.py, where Bridge.lean proves the computed function
the specification's).

Each node means what its operation means on 32-bit words: `add` addition
mod 2^32, `xor`, `rotr` rotation right, `low` and `high` the counter's
words, `const` and `var` themselves.
"""

import json
import os

from z3 import BitVec, BitVecVal, Concat, Extract, RotateRight, substitute

HERE = os.path.dirname(os.path.abspath(__file__))

_template = None


def template():
    """The 16 output words over the symbols lean_h0-7, lean_m0-15,
    lean_t (64-bit), lean_len, lean_flags."""
    global _template
    if _template is None:
        graph = json.load(open(os.path.join(HERE, "lean", "compress.json")))
        t = BitVec("lean_t", 64)
        terms = []
        for node in graph["nodes"]:
            op = node[0]
            if op == "var":
                terms.append(BitVec(f"lean_{node[1]}", 32))
            elif op == "const":
                terms.append(BitVecVal(node[1], 32))
            elif op == "add":
                terms.append(terms[node[1]] + terms[node[2]])
            elif op == "xor":
                terms.append(terms[node[1]] ^ terms[node[2]])
            elif op == "rotr":
                terms.append(RotateRight(terms[node[1]], node[2]))
            elif op == "low":
                terms.append(Extract(31, 0, t))
            elif op == "high":
                terms.append(Extract(63, 32, t))
            else:
                raise ValueError(f"compress.json: unknown node {node}")
        outs = [terms[i] for i in graph["outputs"]]
        h = [BitVec(f"lean_h{i}", 32) for i in range(8)]
        m = [BitVec(f"lean_m{i}", 32) for i in range(16)]
        _template = (h, m, t, BitVec("lean_len", 32), BitVec("lean_flags", 32), outs)
    return _template


def compress(cv, block, counter, block_len, flags):
    """The specification's 16 output words for these terms: `cv` 8 and
    `block` 16 32-bit terms, `counter` 64-bit, `block_len` and `flags`
    32-bit."""
    h, m, t, length, f, outs = template()
    pairs = list(zip(h, cv)) + list(zip(m, block)) + [(t, counter), (length, block_len), (f, flags)]
    return [substitute(w, *pairs) for w in outs]


def words(byte_terms):
    """Little-endian 32-bit words from 8-bit terms (section 2.1)."""
    return [Concat(*reversed(byte_terms[4 * i:4 * i + 4])) for i in range(len(byte_terms) // 4)]
