"""The proofs reject wrong code: each mutant changes one instruction of a
proved kernel, and its proof must fail. A proof that passed a mutant would
be vacuous.

    python3 tools/verify/mutants.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import aarch64
import prove_hybrid
from aarch64 import Unproved


def mutants(code):
    """(description, mutated code) pairs: the first instance of each kind."""
    out = []
    def first(pred):
        return next(a for a, (m, ops) in sorted(code.items()) if pred(m, ops))
    def change(addr, mnem, ops, what):
        c = dict(code)
        c[addr] = (mnem, ops)
        out.append((f"{what} at {addr:#x}", c))
    a = first(lambda m, o: m == "ror")
    m, o = code[a]
    change(a, m, o[:2] + [f"#{(aarch64.imm(o[2]) + 1) % 32}"], "a rotation one bit off")
    a = first(lambda m, o: m == "eor" and o[0][0] == "w")
    change(a, "orr", code[a][1], "an eor made an orr")
    a = first(lambda m, o: m == "add" and o[0][0] == "w" and not o[2].startswith("#"))
    change(a, "eor", code[a][1], "an add made an eor")
    a = first(lambda m, o: m == "ldrb")
    m, o = code[a]
    base = o[1].strip("[]").split(",")
    off = aarch64.imm(base[1]) if len(base) > 1 else 0
    change(a, m, [o[0], f"[{base[0]}, #{off + 1}]"], "a schedule entry read from the next byte")
    a = first(lambda m, o: m == "orr" and o[0][0] == "w")
    change(a, "mov", [code[a][1][0], code[a][1][1]], "a flag left out")
    a = first(lambda m, o: m == "ldp" and o[2].startswith("[x2"))
    m, o = code[a]
    change(a, m, o[:2] + ["[x2, #32]"], "a key word read past the key")
    return out


def main():
    obj = prove_hybrid.assemble()
    real = aarch64.disassemble
    base = real(obj, *[aarch64.symbols(obj)[n] for n in ("blake3_hybrid_c1", "blake3_hybrid_k1_end")])
    failures = 0
    for what, code in mutants(base):
        aarch64.disassemble = lambda *a, code=code: code
        try:
            prove_hybrid.prove_c1(obj, 3, False)
            print(f"c1 mutant, {what}: PROVED (the proof is vacuous here)")
            failures += 1
        except (Unproved, KeyError) as e:
            print(f"c1 mutant, {what}: rejected ({e})")
    aarch64.disassemble = real
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
