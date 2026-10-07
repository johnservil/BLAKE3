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


def vector_mutants(code):
    out = []
    def first(pred):
        return next((a for a, (m, ops) in sorted(code.items()) if pred(m, ops)), None)
    def change(addr, mnem, ops, what):
        if addr is None:
            return
        c = dict(code)
        c[addr] = (mnem, ops)
        out.append((f"{what} at {addr:#x}", c))
    a = first(lambda m, o: m == "xar")
    if a is not None:
        m, o = code[a]
        change(a, m, o[:3] + [f"#{aarch64.imm(o[3]) + 1}"], "an xar rotation one bit off")
    a = first(lambda m, o: m == "sri")
    if a is not None:
        m, o = code[a]
        change(a, m, o[:2] + [f"#{aarch64.imm(o[2]) - 1}"], "an sri shift one bit off")
    a = first(lambda m, o: m == "add" and o[0].startswith("v"))
    if a is not None:
        m, o = code[a]
        change(a, "eor", [x.replace(".4s", ".16b") for x in o], "a vector add made an eor")
    a = first(lambda m, o: m == "zip1")
    if a is not None:
        change(a, "zip2", code[a][1], "a zip1 made a zip2")
    a = first(lambda m, o: m == "tbl")
    if a is not None:
        m, o = code[a]
        change(a, m, [o[0], o[1], o[0]], "a tbl through the wrong index table")
    a = first(lambda m, o: m == "lsr" and o[0][0] == "x" and o[1] == "x6")
    if a is not None:
        m, o = code[a]
        change(a, m, o[:2] + ["#9"], "the partial chunk's length field misread")
    return out


def sme_mutants(code):
    out = []
    def first(pred):
        return next((a for a, (m, ops) in sorted(code.items()) if pred(m, ops)), None)
    def change(addr, mnem, ops, what):
        if addr is None:
            return
        c = dict(code)
        c[addr] = (mnem, ops)
        out.append((f"{what} at {addr:#x}", c))
    a = first(lambda m, o: m == "xar")
    m, o = code[a]
    change(a, m, o[:3] + [f"#{aarch64.imm(o[3]) + 1}"], "an xar rotation one bit off")
    a = first(lambda m, o: m == "add" and o[0].startswith("z") and len(o) == 3)
    change(a, "eor", [x.replace(".s", ".d") for x in code[a][1]], "a vector add made an eor")
    a = first(lambda m, o: m == "mov" and "za0v.s[w13" in ", ".join(o))
    change(a, "mov", [code[a][1][0], code[a][1][1].replace("w13", "w12")], "a transpose reading the wrong slices")
    a = first(lambda m, o: m == "add" and len(o) == 4 and "/m" in o[1])
    change(a, "nop", [], "the counter's carry removed")
    a = first(lambda m, o: m == "st1w" and o[1] == "p2")
    change(a, "st1w", [code[a][1][0], "p0", code[a][1][2]], "a value stored with all sixteen lanes")
    a = first(lambda m, o: m == "index")
    if a is not None:
        m, o = code[a]
        change(a, m, o[:2] + ["#2"] if o[2].startswith("#") else o[:2] + ["w2"], "the counters' step changed")
    return out


def main():
    obj = prove_hybrid.assemble()
    real = aarch64.disassemble
    base = real(obj, *[aarch64.symbols(obj)[n] for n in ("blake3_hybrid_c1", "blake3_hybrid_k1_end")])
    failures = 0
    for what, code in mutants(base):
        aarch64.disassemble = lambda *a, code=code, **k: code
        try:
            prove_hybrid.prove_c1(obj, 3, False)
            print(f"c1 mutant, {what}: PROVED (the proof is vacuous here)")
            failures += 1
        except (Unproved, KeyError) as e:
            print(f"c1 mutant, {what}: rejected ({e})")
    syms = aarch64.symbols(obj)
    for name, prove in (("k2", lambda: prove_hybrid.prove_table(obj, "blake3_hybrid_k2", 2, 2, True)),
                        ("k7", lambda: prove_hybrid.prove_table(obj, "blake3_hybrid_k7", 7, 2, True)),
                        ("q3", lambda: prove_hybrid.prove_partial(obj, 3, 3))):
        base = real(obj, syms[f"blake3_hybrid_{name}"], syms[f"blake3_hybrid_{name}_end"])
        for what, code in vector_mutants(base):
            aarch64.disassemble = lambda *a, code=code, **k: code
            try:
                prove()
                print(f"{name} mutant, {what}: PROVED (the proof is vacuous here)")
                failures += 1
            except (Unproved, KeyError) as e:
                print(f"{name} mutant, {what}: rejected ({e})")
    import prove_sme2
    sobj = prove_sme2.assemble()
    ssyms = aarch64.symbols(sobj)
    sbase = real(sobj, ssyms["blake3_sme2_hash16_chunks_512"], ssyms["blake3_sme2_text_end"], llvm=True)
    for what, code in sme_mutants(sbase):
        aarch64.disassemble = lambda *a, code=code, **k: code
        # Both the path that stores every value and the one that stores a few.
        reasons = []
        for stored in (0, 3):
            try:
                prove_sme2.prove_chunks(sobj, 1, stored, False)
            except (Unproved, KeyError) as e:
                reasons.append(f"storing {stored or 16}: {e}")
        if reasons:
            print(f"SME2 chunks mutant, {what}: rejected ({reasons[0]})")
        else:
            print(f"SME2 chunks mutant, {what}: PROVED (the proof is vacuous here)")
            failures += 1
    aarch64.disassemble = real
    # The Rust paths: a wrong instruction among those each one runs.
    import prove_rust
    lib = prove_rust.Library(prove_rust.build())
    original = dict(lib.code)
    proofs = {"xof": lambda: prove_rust.prove_xof(lib, "verify_neon_xof_many", 5),
              "portable": lambda: prove_rust.prove_in_place(lib, "verify_portable_compress_in_place")}
    ran = {}
    for path, prove in proofs.items():
        traced = []
        step = aarch64.step
        aarch64.step = lambda m, pc, mnem, ops: (traced.append(pc), step(m, pc, mnem, ops))[1]
        prove()
        aarch64.step = step
        ran[path] = set(traced)
    def first(pred, path):
        return next(a for a, (m, o) in sorted(original.items()) if a in ran[path] and pred(m, o))
    rust = []
    a = first(lambda m, o: m == "usra", "xof")
    rust.append(("a NEON rotation's shift one bit off", a, ("usra", original[a][1][:2] + [f"#{aarch64.imm(original[a][1][2]) + 1}"]), "xof"))
    a = first(lambda m, o: m == "eor" and o[-1].startswith("ror"), "portable")
    rust.append(("a rotated operand one bit off", a, ("eor", original[a][1][:3] + [f"ror #{aarch64.imm(original[a][1][3].split()[1]) + 1}"]), "portable"))
    a = first(lambda m, o: m == "add" and o[0].endswith(".4s"), "xof")
    rust.append(("a vector add made an eor", a, ("eor", [x.replace(".4s", ".16b") for x in original[a][1]]), "xof"))
    a = first(lambda m, o: m == "mov" and re.search(r"\.d\[1\]", o[0] or ""), "xof")
    rust.append(("an element moved from the wrong lane", a, ("mov", [original[a][1][0], original[a][1][1].replace("[0]", "[1]")]), "xof"))
    for what, addr, instr, path in rust:
        lib.code = dict(original)
        lib.code[addr] = instr
        try:
            proofs[path]()
            print(f"Rust mutant, {what} at {addr:#x}: PROVED (the proof is vacuous here)")
            failures += 1
        except (Unproved, KeyError) as e:
            print(f"Rust mutant, {what} at {addr:#x}: rejected ({e})")
    # The induction over the NEON extended output's loop: its counter and
    # pointer steps, which only the step from one iteration to the next sees.
    lib.code = dict(original)
    function = "_RNvNtCs5y7Y5DyUngN_13blake3_servil8neon_xof8xof_many"
    head = prove_rust.loop_head(lib, "verify_neon_xof_many", function)
    back = next(a for a, (m, o) in sorted(original.items())
                if a > head and m.startswith("b.") and int(o[0].split()[0], 16) == head)
    body = [a for a in sorted(original) if head <= a <= back]
    find = prove_rust.loop_head
    prove_rust.loop_head = lambda *args: head
    for what, imm, new in (("the loop's counter step", ("#0x8",), "#0x9"), ("the loop's pointer step", ("#0x200",), "#0x1c0")):
        a = next(a for a in body if original[a][0] == "add" and original[a][1][-1] in imm)
        lib.code = dict(original)
        lib.code[a] = ("add", original[a][1][:-1] + [new])
        try:
            prove_rust.prove_xof_every(lib, "verify_neon_xof_many", function, 8, 16)
            print(f"induction mutant, {what} at {a:#x}: PROVED (the proof is vacuous here)")
            failures += 1
        except (Unproved, KeyError) as e:
            print(f"induction mutant, {what} at {a:#x}: rejected ({e})")
    prove_rust.loop_head = find
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
