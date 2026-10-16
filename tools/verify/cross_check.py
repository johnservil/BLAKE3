"""Check the instruction models against the CPU: every register-only
instruction form the proved kernels use (its mnemonic, arrangement, and
immediates as they occur), executed natively on random register states
and by the model, must agree bit for bit.

    pypy3 tools/verify/cross_check.py

The forms are the hybrid kernels' file's and those the Rust and
whole-call proofs run.

Loads, stores, and branches are not checked here: the proofs check every
address against its region, and their models are a few lines each.
"""

import ctypes
import os
import random
import re
import subprocess
import sys
import results

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVecVal
import aarch64
import prove_hybrid
from aarch64 import Machine

SKIP = {"ldr", "ldrb", "ldrh", "ldp", "str", "strb", "strh", "stp", "ldur", "stur", "ld1r", "b", "bl", "b.ne", "cbz", "cbnz",
        "tbz", "tbnz", "ret", "nop", "adr", "adrp", "cmp", "ccmp", "ccmn", "tst", "cset", "csel", "csinc", "subs", "ands", ".word",
        # Control through a register, and reads of the clock and the thread
        # pointer: the runs fix these by their contracts, which random states
        # do not exercise; flags in and out, as cmp and csinc.
        "br", "blr", "mrs", "cmn", "adds", "cinc"}


def streaming(mnem, ops):
    """Whether an instruction is a streaming one (SVE or SME registers):
    cross_check_sme.py's, which checks every form of the SME2 kernels' file."""
    return (re.match(r"(smst|rdsvl|addsvl|addspl|addvl|addpl|cnt[bhwd]$|inc[bhwd]$|dec[bhwd]$)", mnem) is not None
            or any(re.search(r"\b(z\d+|za\w*|p\d+)\b", o) for o in ops))


def forms(instructions):
    """Each register-only instruction form among `instructions`, its
    registers renamed to fixed ones (x9.. for general registers, v0.. for
    vectors), immediates kept."""
    seen = {}
    for mnem, ops in instructions:
        if (mnem in SKIP or mnem.startswith("b.") or "sp" in ops or streaming(mnem, ops)
                or any("[" in o and not re.search(r"\.[sd]\[\d+\]$", o) for o in ops)):
            continue
        names = {}
        def rename(mt):
            r = mt.group(0)
            kind = r[0]
            key = (kind if kind in "wx" else "v", r[1:].split(".")[0])
            if key not in names:
                pool = [k for k in names if k[0] == key[0]]
                names[key] = len(pool)
            n = names[key]
            return (f"{kind}{9 + n}" if kind in "wx" else f"v{n}") + (r[len(r.split('.')[0]):] if "." in r else "")
        text = re.sub(r"\b([wx](\d+)|v\d+(\.\w+)?)\b", rename, ", ".join(ops))
        seen.setdefault((mnem, text), None)
    return list(seen)


def build(form_list):
    """A shared library with one function per form: load x9-x12 from
    xs and v0-v3 from vs, run the instruction, store them back."""
    src = [".text"]
    for i, (mnem, text) in enumerate(form_list):
        src += [f".global f{i}", f"f{i}:",
                "  ldp x9, x10, [x0]", "  ldp x11, x12, [x0, #16]",
                "  ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]",
                f"  {mnem} {text}",
                "  stp x9, x10, [x0]", "  stp x11, x12, [x0, #16]",
                "  st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]", "  ret"]
    return ctypes.CDLL(results.assembled(["cc", "-shared", "-march=armv8.2-a+sha3"], "\n".join(src) + "\n", ".so"))


def model(mnem, text, xs, vs):
    m = Machine({0: (mnem, aarch64.split_operands(text))}, {})
    for i in range(31):
        m.x[i] = BitVecVal(0, 64)
    for i, x in enumerate(xs):
        m.x[9 + i] = BitVecVal(x, 64)
    for i in range(32):
        m.v[i] = [BitVecVal(0, 32)] * 4
    for i in range(4):
        m.v[i] = [BitVecVal(vs[4 * i + l], 32) for l in range(4)]
    m.x["sp"] = aarch64.Ptr("stack", 0)
    aarch64.step(m, 0, mnem, aarch64.split_operands(text))
    return ([aarch64.concrete(m.x[9 + i]) for i in range(4)],
            [aarch64.concrete(m.v[i][l]) for i in range(4) for l in range(4)])


def proved_forms():
    """Every instruction form the Rust, whole-call, and flat-walk proofs run: from their
    last passing runs on this library with these tools (results.py), or by
    running them. Stops if one fails."""
    import prove_rust
    import prove_timing
    import results
    so = prove_rust.build()
    forms = set()
    import prove_flat
    for kind, prove in (("rust", prove_rust.prove), ("timing", prove_timing.prove), ("flat", prove_flat.prove)):
        got = results.load_forms(kind, so)
        if got is None:
            failed, got = prove()
            if failed:
                sys.exit("a proof above failed")
        else:
            print(f"the {kind} proofs passed on this library with these tools: their forms reused (results.py)", flush=True)
        forms |= got
    aarch64.EXECUTED.update(forms)
    return forms


def main():
    rounds = 200
    # The hybrid kernels' file, and every instruction the Rust proofs and the
    # whole-call proofs run.
    proved_forms()
    # The streaming instructions the runs met: each one of the SME2 kernels'
    # file, whose forms cross_check_sme.py checks.
    import prove_sme2
    sme_file = {(mn, tuple(o)) for mn, o in aarch64.disassemble(prove_sme2.assemble(), 0, 1 << 30, llvm=True).values()}
    outside = sorted(f for f in aarch64.EXECUTED if streaming(*f) and f not in sme_file)
    if outside:
        sys.exit(f"streaming instructions outside the SME2 kernels' file: {outside[:5]}")
    obj = prove_hybrid.assemble()
    fl = forms(list(aarch64.disassemble(obj, 0, 1 << 30).values()) + sorted(aarch64.EXECUTED))
    lib = build(fl)
    bad = 0
    for i, (mnem, text) in enumerate(fl):
        f = getattr(lib, f"f{i}")
        for _ in range(rounds):
            xs = [random.getrandbits(64) for _ in range(4)]
            vs = [random.getrandbits(32) for _ in range(16)]
            if mnem == "tbl":   # index bytes in range, as the kernels' tables are
                k = int(re.findall(r"v(\d+)", text)[-1])
                vs[4 * k:4 * k + 4] = [random.getrandbits(32) & 0x0f0f0f0f for _ in range(4)]
            cx, cv = (ctypes.c_uint64 * 4)(*xs), (ctypes.c_uint32 * 16)(*vs)
            f(cx, cv)
            mx, mv = model(mnem, text, xs, vs)
            if list(cx) != mx or list(cv) != mv:
                print(f"MISMATCH: {mnem} {text}: cpu {list(cx)} {list(cv)}, model {mx} {mv}")
                bad += 1
                break
    print(f"{len(fl)} instruction forms, {rounds} random states each: {'all agree' if not bad else f'{bad} disagree'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
