"""Check the streaming SVE and SME2 models (sme.py) against the CPU, as
cross_check.py does for NEON: each register and ZA instruction form the
SME2 kernels use, with their own immediates and slice offsets, runs in
streaming mode on random z0-z3 and ZA tile 0 (and tile 1 for the forms
that write it), and in the model; the registers and tiles must agree.
Needs a CPU with SME2 at a 512-bit vector length (Apple M4, or a VM on
it); elsewhere it says so and exits 0.

    python3 tools/verify/cross_check_sme.py [ROUNDS]
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
import sme
import prove_sme2

L = 16


def has_sme2():
    try:
        return "sme2" in open("/proc/cpuinfo").read().split()
    except OSError:
        return False


def forms(obj):
    code = aarch64.disassemble(obj, 0, 1 << 30, llvm=True)
    seen = {}
    for mnem, ops in code.values():
        text = ", ".join(ops)
        if not re.search(r"\bz\d+|\bza\d", text):
            continue
        if mnem in ("smstart", "smstop"):
            continue
        # Fixed registers: z0-z3 (in order of appearance), p0-p2 kept, w12-w15 as given.
        names = {}
        def rz(mt):
            k = mt.group(1)
            if k not in names:
                names[k] = len(names)
            return f"z{names[k]}"
        # Register ranges keep their length: { z16.s - z19.s } becomes z0-z3.
        rng = re.search(r"\{ z(\d+)\.s - z(\d+)\.s \}", text)
        if rng:
            n = int(rng.group(2)) - int(rng.group(1)) + 1
            text = text.replace(rng.group(0), f"{{ z0.s - z{n - 1}.s }}")
            names.update({str(k): k for k in range(n)})
        t = re.sub(r"\bz(\d+)", rz, text)
        if mnem in ("ld1w", "st1w"):
            t = re.sub(r"\[x\d+\]", "[x11]", t)
        if "za" not in t:
            t = re.sub(r"\bw(\d+)\b", "w9", t)
        seen.setdefault((mnem, t), None)
    return [f for f in seen if max([int(x) for x in re.findall(r"\bz(\d+)", f[1])] or [0]) < 4]


def build(fl):
    """f_i(state): state = 4 Z registers (256 bytes), the four ZA .S tiles (4 KiB),
    x9, x11 (a pointer to 64 bytes of memory, for slice loads and stores),
    w12-w15, a predicate mask. Loads it, runs the form, stores it back."""
    src = [".arch armv9-a+sme2", ".text"]
    for i, (mnem, text) in enumerate(fl):
        src += [f".global f{i}", f"f{i}:",
                "  stp d8, d9, [sp, #-64]!", "  stp d10, d11, [sp, #16]", "  stp d12, d13, [sp, #32]", "  stp d14, d15, [sp, #48]",
                "  mov x16, x0", "  smstart",
                "  ptrue p0.s", "  mov x8, #8", "  whilelo p2.s, xzr, x8",
                "  ldr z0, [x16, #0, mul vl]", "  ldr z1, [x16, #1, mul vl]", "  ldr z2, [x16, #2, mul vl]", "  ldr z3, [x16, #3, mul vl]",
                "  add x17, x16, #256"]
        for t in range(4):
            for row in range(16):
                src += [f"  mov w12, #{row}", f"  ld1w {{za{t}h.s[w12, 0]}}, p0/z, [x17]", "  add x17, x17, #64"]
        src += ["  ldr x9, [x16, #4352]", "  ldr x11, [x16, #4360]",
                "  ldr w12, [x16, #4368]", "  ldr w13, [x16, #4372]", "  ldr w14, [x16, #4376]", "  ldr w15, [x16, #4380]",
                "  add x17, x16, #4000", "  add x17, x17, #384", "  ldr p1, [x17]",
                f"  {mnem} {text}",
                "  str z0, [x16, #0, mul vl]", "  str z1, [x16, #1, mul vl]", "  str z2, [x16, #2, mul vl]", "  str z3, [x16, #3, mul vl]",
                "  add x17, x16, #256"]
        for t in range(4):
            for row in range(16):
                src += [f"  mov w12, #{row}", f"  st1w {{za{t}h.s[w12, 0]}}, p0, [x17]", "  add x17, x17, #64"]
        src += ["  str x9, [x16, #4352]", "  add x17, x16, #4000", "  add x17, x17, #384", "  str p1, [x17]", "  smstop",
                "  ldp d10, d11, [sp, #16]", "  ldp d12, d13, [sp, #32]", "  ldp d14, d15, [sp, #48]", "  ldp d8, d9, [sp], #64", "  ret"]
    d = tempfile.mkdtemp()
    path = os.path.join(d, "f.S")
    open(path, "w").write("\n".join(src) + "\n")
    lib = os.path.join(d, "f.so")
    subprocess.run([prove_sme2.CLANG, "-shared", "-march=armv9-a+sme2", path, "-o", lib], check=True)
    return ctypes.CDLL(lib)


def model(mnem, text, z, za, x9, mem, ws, p1):
    m = aarch64.Machine({}, {"mem": aarch64.Region("mem", 64, True, lambda o: BitVecVal(mem[o], 8))})
    for i in range(31):
        m.x[i] = BitVecVal(0, 64)
    m.x[9] = BitVecVal(x9, 64)
    m.x[11] = aarch64.Ptr("mem", 0)
    for i, w in enumerate(ws):
        m.x[12 + i] = BitVecVal(w, 64)
    m.x["sp"] = aarch64.Ptr("stack", 0)
    m.v = {i: [BitVecVal(0, 32)] * L for i in range(32)}
    for i in range(4):
        m.v[i] = [BitVecVal(z[L * i + l], 32) for l in range(L)]
    m.za = [[[BitVecVal(za[256 * t + 16 * r + c], 32) for c in range(L)] for r in range(L)] for t in range(4)]
    m.streaming = True
    m.p = {i: [False] * L for i in range(16)}
    m.p[0] = [True] * L
    m.p[1] = list(p1)
    m.p[2] = [i < 8 for i in range(L)]
    aarch64.step(m, 0, mnem, aarch64.split_operands(text))
    c = aarch64.concrete
    zs = [c(m.v[i][l]) for i in range(4) for l in range(L)]
    tiles = [c(m.za[t][r][col]) for t in range(4) for r in range(L) for col in range(L)]
    out_mem = [c(m.regions["mem"].bytes[o]) if o in m.regions["mem"].bytes else mem[o] for o in range(64)]
    from z3 import simplify, is_true
    preds = [p if isinstance(p, bool) else is_true(simplify(p)) for p in m.p[1]]
    return zs, tiles, c(m.x[9]), out_mem, preds


def main():
    if not has_sme2():
        print("cross_check_sme: this CPU has no SME2; nothing checked")
        return
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    obj = prove_sme2.assemble()
    fl = forms(obj)
    lib = build(fl)
    mem = (ctypes.c_uint8 * 64)()
    bad = 0
    for i, (mnem, text) in enumerate(fl):
        f = getattr(lib, f"f{i}")
        for _ in range(rounds):
            z = [random.getrandbits(32) for _ in range(4 * L)]
            za = [random.getrandbits(32) for _ in range(4 * L * L)]
            if "cmphi" in mnem or "cmplo" in mnem:
                z = [random.getrandbits(3) for _ in range(4 * L)]
            x9 = random.getrandbits(32)
            ws = [random.randrange(16) for _ in range(4)]
            data = [random.getrandbits(8) for _ in range(64)]
            for o in range(64):
                mem[o] = data[o]
            state = (ctypes.c_uint32 * (4392 // 4))()
            pbits = random.getrandbits(16)
            buf64 = ctypes.cast(state, ctypes.POINTER(ctypes.c_uint64))
            buf64[4384 // 8] = sum(((pbits >> i) & 1) << (4 * i) for i in range(16))
            for k, v in enumerate(z):
                state[k] = v
            for k, v in enumerate(za):
                state[64 + k] = v
            buf = ctypes.cast(state, ctypes.POINTER(ctypes.c_uint64))
            buf[4352 // 8] = x9
            buf[4360 // 8] = ctypes.addressof(mem)
            for k, w in enumerate(ws):
                state[4368 // 4 + k] = w
            f(state)
            praw = buf64[4384 // 8]
            cpu = ([state[k] for k in range(64)], [state[64 + k] for k in range(1024)], buf[4352 // 8], list(mem),
                   [bool((praw >> (4 * i)) & 1) for i in range(16)])
            mod = model(mnem, text, z, za, x9, data, ws, [bool((pbits >> i) & 1) for i in range(16)])
            if cpu[0] != mod[0] or cpu[1] != mod[1] or cpu[3] != mod[3] or cpu[4] != mod[4]:
                print(f"MISMATCH: {mnem} {text}")
                bad += 1
                break
    print(f"{len(fl)} streaming instruction forms, {rounds} random states each: {'all agree' if not bad else f'{bad} disagree'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
