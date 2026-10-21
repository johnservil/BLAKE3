#!/usr/bin/env pypy3
"""Generate the fork's x86-64 kernels: whole chunks and a partial one in one
pass, side by side, as the Arm q kernels do (c/blake3_partial_x86-64_unix.S
and c/blake3_partial_x86-64_windows_msvc.asm, the same code in both forms).

    pypy3 tools/gen_x86.py unix > c/blake3_partial_x86-64_unix.S
    pypy3 tools/gen_x86.py msvc > c/blake3_partial_x86-64_windows_msvc.asm

Two kernels, one per width W: blake3_partial_sse41 (W = 4, xmm, legacy SSE
encodings, for any CPU with SSE4.1) and blake3_partial_avx2 (W = 8, ymm,
VEX). Each takes one argument, a pointer to a parameter block that the
caller fills (src/ffi_partial.rs), in the first argument register of the
calling convention (rdi on Unix, rcx on Windows); nothing else differs
between the two forms but the registers each convention has the callee
keep (xmm6-xmm15 on Windows), and the directives.

The kernel hashes W lanes of 16 blocks each, block b of lane j read at
lane j's pointer + 64 b. Every per-lane, per-block value comes from the
block, so the kernel branches only on the block index:

    offset          contents (u32 words unless said; vectors are W words)
    MSG             W lane pointers (u64)
    KEY             8 vectors: the chaining value's words, one per lane
    IV              4 vectors: BLAKE3's IV words 0-3, broadcast
    ROT16, ROT8     the byte shuffles that rotate each word right by 16, 8
    CTR             2 vectors: each lane's counter, low and high words
    LENS            16 vectors: each block's length per lane
    FLAGS           16 vectors: each block's flags per lane
    LAST            u32: 64 x the block after which SNAP is taken
    H               8 vectors, out: each lane's chaining value after block 15
    SNAP            8 vectors, out: each lane's chaining value after block LAST

so a partial chunk takes a lane beside whole ones, its last block's length
and CHUNK_END in its own lane at its own block, its value in SNAP; the
lane's later blocks compute words nobody reads (its pointer must reach 16
blocks: the caller zero-pads it). The registers follow upstream BLAKE3's
x86 kernels: the state's four rows in registers 0-15, row c's first word
kept in a stack slot but while row c is added to, so register 8 is the
temporary the rotations need; the block's sixteen message words
transposed onto the stack first.
"""

import sys

W_OF = {"sse41": 4, "avx2": 8}
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
COLS = [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)]
DIAGS = [(0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)]


def schedule():
    s = [list(range(16))]
    for _ in range(6):
        s.append([s[-1][MSG_PERMUTATION[i]] for i in range(16)])
    return s


def layout(w):
    """The parameter block's offsets for width w, in bytes."""
    v = 4 * w
    off, o = {}, 0
    for name, size in [("MSG", 8 * w), ("KEY", 8 * v), ("IV", 4 * v), ("ROT16", v), ("ROT8", v), ("CTR", 2 * v),
                       ("LENS", 16 * v), ("FLAGS", 16 * v), ("LAST", 64), ("H", 8 * v), ("SNAP", 8 * v)]:
        off[name] = o
        o += size
    off["SIZE"] = o
    return off


class Kernel:
    def __init__(self, isa, windows):
        self.isa, self.windows = isa, windows
        self.w = W_OF[isa]
        self.v = 4 * self.w
        self.avx = isa == "avx2"
        self.reg = "ymm" if self.avx else "xmm"
        self.word = "ymmword" if self.avx else "xmmword"
        self.p = "rcx" if windows else "rdi"
        self.off = layout(self.w)
        self.lines = []
        # Stack frame: the sixteen message vectors, the spill slot, and on
        # Windows the callee's xmm6-xmm15.
        self.msg = 0
        self.spill = 16 * self.v
        self.xsave = self.spill + max(self.v, 16)
        self.frame = self.xsave + (160 if windows else 0)

    def e(self, s):
        self.lines.append("        " + s)

    def r(self, i):
        return f"{self.reg}{i}"

    def mem(self, base, disp):
        return f"{self.word} ptr [{base}+{disp}]" if disp else f"{self.word} ptr [{base}]"

    def m(self, i):
        return self.mem("rsp", self.msg + self.v * i)

    def pm(self, name, extra=0):
        return self.mem(self.p, self.off[name] + extra)

    # Operations, VEX three-operand or legacy two-operand (dst is a).
    def op(self, vex, legacy, dst, a, b, imm=None):
        tail = f", {imm}" if imm is not None else ""
        if self.avx:
            self.e(f"{vex} {dst}, {a}, {b}{tail}")
        else:
            if dst != a:
                self.e(f"movdqa {dst}, {a}")
            self.e(f"{legacy} {dst}, {b}{tail}")

    def add(self, d, a, b):
        self.op("vpaddd", "paddd", d, a, b)

    def xor(self, d, a, b):
        self.op("vpxor", "pxor", d, a, b)

    def orr(self, d, a, b):
        self.op("vpor", "por", d, a, b)

    def shuf(self, d, a, mask):
        self.op("vpshufb", "pshufb", d, a, mask)

    def shift(self, vex, legacy, d, a, n):
        if self.avx:
            self.e(f"{vex} {d}, {a}, {n}")
        else:
            if d != a:
                self.e(f"movdqa {d}, {a}")
            self.e(f"{legacy} {d}, {n}")

    def load(self, d, src):
        self.e(f"{'vmovdqu' if self.avx else 'movdqu'} {d}, {src}")

    def store(self, dst, s):
        self.e(f"{'vmovdqu' if self.avx else 'movdqu'} {dst}, {s}")

    def store_aligned(self, dst, s):
        self.e(f"{'vmovdqa' if self.avx else 'movdqa'} {dst}, {s}")

    def half(self, quads, words, rd, rb):
        """Half of four Gs side by side: a += m + b; d = (d ^ a) >>> rd;
        c += d; b = (b ^ c) >>> rb. Row c's register 8 is in the spill
        slot outside its addition, register 8 the temporary."""
        r, t = self.r, self.r(8)
        for (a, b, c, d), m in zip(quads, words):
            self.add(r(a), r(a), self.m(m))
        for a, b, c, d in quads:
            self.add(r(a), r(a), r(b))
        for a, b, c, d in quads:
            self.xor(r(d), r(d), r(a))
        self.load(t, self.pm("ROT16" if rd == 16 else "ROT8"))
        for a, b, c, d in quads:
            self.shuf(r(d), r(d), t)
        for a, b, c, d in quads:
            if c == 8:
                self.add(t, r(d), self.mem("rsp", self.spill))
            else:
                self.add(r(c), r(c), r(d))
        for a, b, c, d in quads:
            self.xor(r(b), r(b), r(c))
        self.store_aligned(self.mem("rsp", self.spill), t)
        for a, b, c, d in quads:
            self.shift("vpsrld", "psrld", t, r(b), rb)
            self.shift("vpslld", "pslld", r(b), r(b), 32 - rb)
            self.orr(r(b), r(b), t)

    def transpose(self):
        """Block rdx/64's sixteen words of every lane onto the stack, word i
        of lane j at element j of message vector i."""
        r = self.r
        lanes = [f"r{8 + j}" for j in range(self.w)]
        for q in range(4):
            d = f"+{16 * q}" if q else ""
            if self.avx:
                for k in range(4):
                    self.e(f"vmovdqu xmm{8 + k}, xmmword ptr [{lanes[k]}+rdx{d}]")
                    self.e(f"vinsertf128 ymm{8 + k}, ymm{8 + k}, xmmword ptr [{lanes[k + 4]}+rdx{d}], 1")
                self.e("vunpcklpd ymm12, ymm8, ymm9")
                self.e("vunpckhpd ymm13, ymm8, ymm9")
                self.e("vunpcklpd ymm14, ymm10, ymm11")
                self.e("vunpckhpd ymm15, ymm10, ymm11")
                for k, (lo, hi, imm) in enumerate([(12, 14, 136), (12, 14, 221), (13, 15, 136), (13, 15, 221)]):
                    self.e(f"vshufps ymm{8 + k}, ymm{lo}, ymm{hi}, {imm}")
                    self.store_aligned(self.m(4 * q + k), r(8 + k))
            else:
                for k in range(4):
                    self.e(f"movdqu xmm{8 + k}, xmmword ptr [{lanes[k]}+rdx{d}]")
                for dst, a, b, op in [(12, 8, 9, "unpcklpd"), (13, 8, 9, "unpckhpd"), (14, 10, 11, "unpcklpd"), (15, 10, 11, "unpckhpd")]:
                    self.e(f"movaps xmm{dst}, xmm{a}")
                    self.e(f"{op} xmm{dst}, xmm{b}")
                for k, (lo, hi, imm) in enumerate([(12, 14, 136), (12, 14, 221), (13, 15, 136), (13, 15, 221)]):
                    self.e(f"movaps xmm{8 + k}, xmm{lo}")
                    self.e(f"shufps xmm{8 + k}, xmm{hi}, {imm}")
                    self.store_aligned(self.m(4 * q + k), r(8 + k))

    def body(self, name):
        r, w, v = self.r, self.w, self.v
        L = self.lines
        saved = ["rbp", "rbx", "r12", "r13", "r14", "r15"]
        for g in saved:
            self.e(f"push {g}")
        self.e("mov rbp, rsp")
        self.e(f"sub rsp, {self.frame + 64}")
        self.e("and rsp, -64")
        if self.windows:
            for i in range(6, 16):
                self.e(f"{'vmovdqu' if self.avx else 'movdqu'} xmmword ptr [rsp+{self.xsave + 16 * (i - 6)}], xmm{i}")
        for j in range(w):
            self.e(f"mov r{8 + j}, qword ptr [{self.p}+{self.off['MSG'] + 8 * j}]")
        for i in range(8):
            self.load(r(i), self.pm("KEY", v * i))
        self.e("xor edx, edx")
        self.e("xor ebx, ebx")
        L.append(f"{name}_block:")
        self.transpose()
        self.load(r(8), self.pm("IV"))
        self.store_aligned(self.mem("rsp", self.spill), r(8))
        for i in range(1, 4):
            self.load(r(8 + i), self.pm("IV", v * i))
        self.load(r(12), self.pm("CTR"))
        self.load(r(13), self.pm("CTR", v))
        self.load(r(14), self.mem(f"{self.p}+rbx", self.off["LENS"]))
        self.load(r(15), self.mem(f"{self.p}+rbx", self.off["FLAGS"]))
        for s in schedule():
            for quads, base in ((COLS, 0), (DIAGS, 8)):
                self.half(quads, [s[base + 2 * k] for k in range(4)], 16, 12)
                self.half(quads, [s[base + 2 * k + 1] for k in range(4)], 8, 7)
        self.xor(r(0), r(0), self.mem("rsp", self.spill))
        for i in range(1, 8):
            self.xor(r(i), r(i), r(i + 8))
        self.e(f"cmp edx, dword ptr [{self.p}+{self.off['LAST']}]")
        self.e(f"jne {name}_next")
        for i in range(8):
            self.store(self.pm("SNAP", v * i), r(i))
        L.append(f"{name}_next:")
        self.e("add rdx, 64")
        self.e(f"add rbx, {v}")
        self.e("cmp rdx, 1024")
        self.e(f"jne {name}_block")
        for i in range(8):
            self.store(self.pm("H", v * i), r(i))
        if self.windows:
            for i in range(6, 16):
                self.e(f"{'vmovdqu' if self.avx else 'movdqu'} xmm{i}, xmmword ptr [rsp+{self.xsave + 16 * (i - 6)}]")
        if self.avx:
            self.e("vzeroupper")
        self.e("mov rsp, rbp")
        for g in reversed(saved):
            self.e(f"pop {g}")
        self.e("ret")


def unix():
    out = ["// Generated by tools/gen_x86.py; do not edit.",
           "#if defined(__ELF__) && defined(__linux__)",
           '.section .note.GNU-stack,"",%progbits',
           "#endif",
           "#if defined(__ELF__) && defined(__CET__) && defined(__has_include)",
           "#if __has_include(<cet.h>)",
           "#include <cet.h>",
           "#endif",
           "#endif",
           "#if !defined(_CET_ENDBR)",
           "#define _CET_ENDBR",
           "#endif",
           ".intel_syntax noprefix"]
    for isa in ("sse41", "avx2"):
        name = f"blake3_partial_{isa}"
        out += [f".global _{name}", f".global {name}"]
    out += ["#ifdef __APPLE__", ".text", "#else", ".section .text", "#endif"]
    for isa in ("sse41", "avx2"):
        name = f"blake3_partial_{isa}"
        k = Kernel(isa, windows=False)
        k.body(name)
        out += ["        .p2align 6", f"_{name}:", f"{name}:", "        _CET_ENDBR"] + k.lines
    return out


def msvc():
    out = ["; Generated by tools/gen_x86.py; do not edit."]
    for isa in ("sse41", "avx2"):
        name = f"blake3_partial_{isa}"
        out += [f"public _{name}", f"public {name}"]
    out += ["", "_TEXT   SEGMENT ALIGN(16) 'CODE'"]
    for isa in ("sse41", "avx2"):
        name = f"blake3_partial_{isa}"
        k = Kernel(isa, windows=True)
        k.body(name)
        out += ["", "ALIGN   16", f"{name} PROC", f"_{name} PROC"] + k.lines + [f"_{name} ENDP", f"{name} ENDP"]
    out += ["", "_TEXT ENDS", "END"]
    return out


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ("unix", "msvc"), "usage: gen_x86.py unix|msvc"
    print("\n".join(unix() if sys.argv[1] == "unix" else msvc()))
