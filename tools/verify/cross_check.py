"""Check the instruction models against the CPU: every register-only
instruction form the proved kernels use (its mnemonic, arrangement, and
immediates as they occur), executed natively on random register states
and by the model, must agree bit for bit.

    python3 tools/verify/cross_check.py [ROUNDS]

Loads, stores, and branches are not checked here: the proofs check every
address against its region, and their models are a few lines each.
"""

import ctypes
import os
import random
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVecVal
import aarch64
import prove_hybrid
from aarch64 import Machine

SKIP = {"ldr", "ldrb", "ldp", "str", "stp", "b", "b.ne", "cbz", "cbnz", "ret", "nop", "adr", "cmp", "csel", "subs", ".word"}


def forms(obj):
    """Each register-only instruction form, its registers renamed to fixed
    ones (x9.. for general registers, v0.. for vectors), immediates kept."""
    code = aarch64.disassemble(obj, 0, 1 << 30)
    seen = {}
    for mnem, ops in code.values():
        if mnem in SKIP or mnem.startswith("b."):
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
    d = tempfile.mkdtemp()
    src = [".text"]
    for i, (mnem, text) in enumerate(form_list):
        src += [f".global f{i}", f"f{i}:",
                "  ldp x9, x10, [x0]", "  ldp x11, x12, [x0, #16]",
                "  ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]",
                f"  {mnem} {text}",
                "  stp x9, x10, [x0]", "  stp x11, x12, [x0, #16]",
                "  st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]", "  ret"]
    path = os.path.join(d, "forms.S")
    open(path, "w").write("\n".join(src) + "\n")
    lib = os.path.join(d, "forms.so")
    subprocess.run(["cc", "-shared", "-march=armv8.2-a+sha3", path, "-o", lib], check=True)
    return ctypes.CDLL(lib)


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


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    obj = prove_hybrid.assemble()
    fl = forms(obj)
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
