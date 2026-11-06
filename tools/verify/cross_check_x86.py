"""Check x86 vector register and memory forms against a Linux x86-64 CPU.

Every form used by the kernels gets 200 random states.
Memory forms compare the entire buffer as well as the registers. Unaligned
moves/arithmetic run at an unaligned address too; aligned moves have a
separate, bounded fault control. AVX2 is required; AVX-512 forms are
reported skipped when the CPU lacks AVX-512 F/VL.

    python3 tools/verify/cross_check_x86.py

General-register instructions, branches and prefetch effects remain outside
this CPU check. Address/trace and ABI-boundary controls are in test_x86_memory.py.
"""

import ctypes
import os
import random
random.seed(0)
import resource
import signal
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVecVal
import x86
import aarch64

FILES = ("sse41", "avx2", "avx512", "wide", "output")  # wide/output: generated AVX-512 kernels


def forms(code):
    """Each vector form, registers renamed: vectors to
    xmm/ymm/zmm 0-3 by first use, masks to k1-k2, general registers to
    r8d-r11d / r8-r11. Memory addresses become [rdx]; widths remain."""
    seen = {}
    for mnem, ops, _ in code.values():
        if not any(x86.is_vreg(x86.strip_mask(o)[0]) for o in ops):
            continue
        vec, gpr, msk = {}, {}, {}

        def rename(mt):
            r = mt.group(0)
            if re.match(r"[xyz]mm\d+$", r):
                n = vec.setdefault(int(r[3:]), len(vec))
                return f"{r[:3]}{n}"
            if re.match(r"k\d$", r):
                return f"k{msk.setdefault(r, len(msk) + 1)}"
            base, bits = x86.SUB[r]
            n = gpr.setdefault(base, len(gpr))
            return {64: f"r{8 + n}", 32: f"r{8 + n}d"}[bits]
        text = re.sub(r"\b([xyz]mm\d+|k\d|r[a-z0-9]+|e[a-z]{2})\b",
                      lambda mt: rename(mt) if (re.match(r"[xyz]mm\d+$|k\d$", mt.group(0)) or mt.group(0) in x86.SUB) else mt.group(0),
                      re.sub(r"\[[^]]*\]", "[MEMORY]", ", ".join(ops))).replace("[MEMORY]", "[rdx]")
        if max(list(vec.values()) + [0]) > 3 or len(gpr) > 4 or len(msk) > 2:
            continue
        if mnem in ("blendvps", "blendvpd", "pblendvb"):
            # The mask is xmm0 by definition; the disassembler prints it
            # as a third operand, which the assembler rejects. The model
            # takes the two-operand form (selector xmm0).
            text = ", ".join(text.split(", ")[:2])
        seen.setdefault((mnem, text), None)
    return list(seen)


def wide(text):
    return "zmm" in text or "k1" in text


def build(form_list, avx512):
    """One function per form: r8-r11 from xs, zmm/ymm 0-3 from vs (and k1,
    k2 from xs' low halves), the instruction, everything stored back."""
    d = tempfile.mkdtemp()
    src = [".intel_syntax noprefix", ".text"]
    for i, (mnem, text) in enumerate(form_list):
        load = (["vmovdqu32 zmm%d, [rsi + %d]" % (j, 64 * j) for j in range(4)] + ["kmovw k1, r8d", "kmovw k2, r9d"]
                if avx512 else ["vmovdqu ymm%d, [rsi + %d]" % (j, 64 * j) for j in range(4)])
        store = (["vmovdqu32 [rsi + %d], zmm%d" % (64 * j, j) for j in range(4)] if avx512
                 else ["vmovdqu [rsi + %d], ymm%d" % (64 * j, j) for j in range(4)])
        src += [f".global f{i}", f"f{i}:", "  mov r8, [rdi]", "  mov r9, [rdi + 8]", "  mov r10, [rdi + 16]",
                "  mov r11, [rdi + 24]"] + ["  " + x for x in load] + [f"  {mnem} {text}"] + ["  " + x for x in store] + [
                "  mov [rdi], r8", "  mov [rdi + 8], r9", "  mov [rdi + 16], r10", "  mov [rdi + 24], r11",
                "  vzeroupper", "  ret"]
    path = os.path.join(d, "forms.S")
    with open(path, "w") as f:
        f.write("\n".join(src) + '\n.section .note.GNU-stack,"",@progbits\n')
    lib = os.path.join(d, "forms.so")
    subprocess.run(["cc", "-shared", path, "-o", lib], check=True)
    return ctypes.CDLL(lib)


def model(mnem, text, xs, vs, avx512, memory, offset):
    regions = {} if memory is None else {"data": aarch64.Region(
        "data", len(memory), True, lambda i: BitVecVal(memory[i], 8), alignment=64)}
    m = x86.Machine({}, regions)
    for r in x86.GPR:
        m.x[r] = BitVecVal(0, 64)
    m.x["rsp"] = aarch64.Ptr("stack", 0)
    m.x["rdx"] = aarch64.Ptr("data", offset)
    for i, x in enumerate(xs):
        m.x[f"r{8 + i}"] = BitVecVal(x, 64)
    for i in range(32):
        m.v[i] = [BitVecVal(0, 32)] * 16
    for i in range(4):
        m.v[i] = [BitVecVal(vs[16 * i + l], 32) for l in range(16)] if avx512 else \
            [BitVecVal(vs[16 * i + l], 32) for l in range(8)] + [BitVecVal(0, 32)] * 8
    if avx512:
        m.k[1], m.k[2] = BitVecVal(xs[0] & 0xFFFF, 16), BitVecVal(xs[1] & 0xFFFF, 16)
    x86.step(m, 0, mnem, x86.split(text), 0)
    lanes = 16 if avx512 else 8
    written = m.regions["data"].bytes if memory is not None else {}
    want_memory = None if memory is None else [aarch64.concrete(written[i]) if i in written else b
                                              for i, b in enumerate(memory)]
    return ([aarch64.concrete(m.x[f"r{8 + i}"]) for i in range(4)],
            [aarch64.concrete(m.v[i][l]) if l < lanes else vs[16 * i + l] for i in range(4) for l in range(16)],
            want_memory)


def has_avx512():
    # A failed feature read must stop instead of silently dropping coverage.
    with open("/proc/cpuinfo") as f:
        flags = f.read().split()
    assert "avx2" in flags, "the CPU cross-check requires an AVX2-capable Linux x86-64 host"
    return "avx512f" in flags and "avx512vl" in flags


def memory_buffer(data):
    """An owned ctypes buffer and its 64-byte-aligned view."""
    owner = ctypes.create_string_buffer(len(data) + 63)
    address = (ctypes.addressof(owner) + 63) & ~63
    view = (ctypes.c_uint8 * len(data)).from_address(address)
    view[:] = data
    return owner, view


def alignment_fault(lib, index, mnem, text, avx512):
    """One deliberately unaligned call, confined to a bounded child process."""
    try:
        model(mnem, text, [0xffffffffffffffff] * 4, [0] * 64, avx512, [0] * 128, 1)
    except aarch64.Unproved as error:
        assert "alignment" in str(error), str(error)
    else:
        raise AssertionError(f"model accepted an unaligned {mnem}: {text}")
    result = subprocess.run([sys.executable, __file__, "--alignment-fault", lib._name, str(index)],
                            capture_output=True, timeout=20)
    assert result.returncode == -signal.SIGSEGV, (index, result.returncode, result.stderr)


def fault_child(path, index):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lib = ctypes.CDLL(path)
    # All mask lanes active: the access cannot be suppressed by a zero mask.
    xs = (ctypes.c_uint64 * 4)(*[0xffffffffffffffff] * 4)
    vs = (ctypes.c_uint32 * 64)()
    owner, data = memory_buffer([0] * 128)
    getattr(lib, f"f{index}")(xs, vs, ctypes.c_void_p(ctypes.addressof(data) + 1))


def main():
    rounds = 200
    avx512 = has_avx512()
    total, skipped, bad, memory_forms, faults = 0, 0, 0, 0, 0
    for isa in FILES:
        import prove_x86
        so = prove_x86.assemble(isa)
        fl = list(dict.fromkeys(forms(x86.disassemble(so, x86.sections(so)[".text"][0], x86.text_end(so)))))
        if any(wide(t) for _, t in fl) and not avx512:
            skipped += len(fl)
            continue
        lib = build(fl, avx512)
        for i, (mnem, text) in enumerate(fl):
            f = getattr(lib, f"f{i}")
            memory = "[rdx]" in text
            offsets = (0,)
            if memory:
                print(f"MEMORY {mnem} {text}", flush=True)
                memory_forms += 1
                vex = mnem.startswith("v")
                base = mnem[1:] if vex else mnem
                size = x86.SIZES[re.search(r"(\w+) ptr ", text).group(1)]
                if x86.memory_alignment(base, vex, size) == 1:
                    offsets = (0, 1)
                else:
                    alignment_fault(lib, i, mnem, text, avx512)
                    faults += 1
            for trial in range(rounds):
                xs = [random.getrandbits(64) for _ in range(4)]
                vs = [random.getrandbits(32) for _ in range(64)]
                if mnem in ("vpermt2d", "vpermi2d"):
                    # Index lanes in range, as the kernels' tables are.
                    k = 1 if mnem == "vpermt2d" else 0
                    vs[16 * k:16 * k + 16] = [random.getrandbits(5) for _ in range(16)]
                cx, cv = (ctypes.c_uint64 * 4)(*xs), (ctypes.c_uint32 * 64)(*vs)
                data = [random.getrandbits(8) for _ in range(128)] if memory else None
                offset = offsets[trial % len(offsets)]
                mx, mv, md = model(mnem, text, xs, vs, avx512, data, offset)
                owner, view = memory_buffer(data) if memory else (None, None)
                f(cx, cv, ctypes.c_void_p(ctypes.addressof(view) + offset) if memory else None)
                actual_memory = list(view) if memory else None
                if list(cx) != mx or list(cv) != mv or actual_memory != md:
                    print(f"MISMATCH: {mnem} {text} at offset {offset}: cpu {list(cx)} {list(cv)[:16]} {actual_memory}, model {mx} {mv[:16]} {md}")
                    bad += 1
                    break
            total += 1
    print(f"{total} vector instruction forms ({memory_forms} memory forms), {rounds} random states each, {faults} alignment fault controls: {'all agree' if not bad else f'{bad} disagree'}"
          + (f"; {skipped} AVX-512 forms skipped (no AVX-512 here)" if skipped else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--alignment-fault":
        fault_child(sys.argv[2], int(sys.argv[3]))
    else:
        assert len(sys.argv) == 1, "usage: cross_check_x86.py"
        main()
