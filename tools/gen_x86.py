#!/usr/bin/env python3
"""Generate BLAKE3-Servil's x86-64 kernels: whole chunks and a partial one in one
pass, side by side, as the Arm q kernels do (c/blake3_partial_x86-64_unix.S).

    python3 tools/gen_x86.py > c/blake3_partial_x86-64_unix.S
    python3 tools/gen_x86.py --last > c/blake3_last_x86-64_unix.S

The last-blocks family (--last, blake3_servil_last_*: BLAKE3-Servil #247)
is the same kernel for W items of equal block count, each one chunk's
root, last blocks included: it hashes END/64 blocks, not 16, and reads
block SWITCH/64 of each lane through LASTMSG (each item's last block,
copied zero-padded), so no lane reads past its item; its FLAGS carry
CHUNK_END and ROOT at that block, and H is then each lane's digest.

Two kernels, one per width W: blake3_servil_partial_sse41 (W = 4, xmm, legacy SSE
encodings, for any CPU with SSE4.1) and blake3_servil_partial_avx2 (W = 8, ymm,
VEX). Each takes one argument, a pointer to a parameter block that the
caller fills (src/ffi_partial.rs), in the first argument register (rdi,
the System V convention of Linux and macOS).

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


def layout(w, last=False):
    """The parameter block's offsets for width w, in bytes; `last`, the
    last-blocks family's (the same, and after it LASTMSG: W lane pointers
    (u64) to each lane's last block; END: u32, 64 x the blocks to hash;
    SWITCH: u32, 64 x the block read through LASTMSG)."""
    v = 4 * w
    off, o = {}, 0
    fields = [("MSG", 8 * w), ("KEY", 8 * v), ("IV", 4 * v), ("ROT16", v), ("ROT8", v), ("CTR", 2 * v),
              ("LENS", 16 * v), ("FLAGS", 16 * v), ("LAST", 64), ("H", 8 * v), ("SNAP", 8 * v)]
    if last:
        fields += [("LASTMSG", 8 * w), ("END", 4), ("SWITCH", 4)]
    for name, size in fields:
        off[name] = o
        o += size
    off["SIZE"] = o
    return off


class Kernel:
    def __init__(self, isa, last=False, mixed=False):
        self.isa = isa
        self.mixed = mixed
        assert not mixed or last and isa == "avx2"
        # The last-blocks family (blake3_servil_last_*): END blocks, not 16,
        # and block SWITCH/64 read through LASTMSG; no snapshot.
        self.last = last
        self.w = W_OF[isa]
        self.v = 4 * self.w
        self.avx = isa == "avx2"
        self.reg = "ymm" if self.avx else "xmm"
        self.word = "ymmword" if self.avx else "xmmword"
        self.p = "rdi"
        self.off = layout(self.w, last)
        if mixed:
            self.off['ENDS'] = (self.off['SIZE'] + 63) // 64 * 64
        self.lines = []
        # Stack frame: the sixteen message vectors and the spill slot.
        self.msg = 0
        self.spill = 16 * self.v
        self.xsave = self.spill + max(self.v, 16)
        self.frame = self.xsave

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
        index = "" if self.mixed else "+rdx"
        for q in range(4):
            d = f"+{16 * q}" if q else ""
            if self.avx:
                for k in range(4):
                    self.e(f"vmovdqu xmm{8 + k}, xmmword ptr [{lanes[k]}{index}{d}]")
                    self.e(f"vinsertf128 ymm{8 + k}, ymm{8 + k}, xmmword ptr [{lanes[k + 4]}{index}{d}], 1")
                self.e("vunpcklpd ymm12, ymm8, ymm9")
                self.e("vunpckhpd ymm13, ymm8, ymm9")
                self.e("vunpcklpd ymm14, ymm10, ymm11")
                self.e("vunpckhpd ymm15, ymm10, ymm11")
                for k, (lo, hi, imm) in enumerate([(12, 14, 136), (12, 14, 221), (13, 15, 136), (13, 15, 221)]):
                    self.e(f"vshufps ymm{8 + k}, ymm{lo}, ymm{hi}, {imm}")
                    self.store_aligned(self.m(4 * q + k), r(8 + k))
            else:
                for k in range(4):
                    self.e(f"movdqu xmm{8 + k}, xmmword ptr [{lanes[k]}{index}{d}]")
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
        for j in range(w):
            self.e(f"mov r{8 + j}, qword ptr [{self.p}+{self.off['MSG'] + 8 * j}]")
        for i in range(8):
            self.load(r(i), self.pm("KEY", v * i))
        self.e("xor edx, edx")
        self.e("xor ebx, ebx")
        L.append(f"{name}_block:")
        if self.mixed:
            for j in range(w):
                self.e(f"mov r{8+j}, qword ptr [{self.p}+{self.off['MSG']+8*j}]")
                self.e(f"add r{8+j}, rdx")
                self.e(f"cmp edx, dword ptr [{self.p}+{self.off['ENDS']+4*j}]")
                self.e(f"cmovae r{8+j}, qword ptr [{self.p}+{self.off['LASTMSG']+8*j}]")
        elif self.last:
            # At block SWITCH/64, each lane's pointer moves to its last
            # block (LASTMSG, less the block's offset, which the loads add).
            self.e(f"cmp edx, dword ptr [{self.p}+{self.off['SWITCH']}]")
            self.e(f"jne {name}_read")
            for j in range(w):
                self.e(f"mov r{8 + j}, qword ptr [{self.p}+{self.off['LASTMSG'] + 8 * j}]")
                self.e(f"sub r{8 + j}, rdx")
            L.append(f"{name}_read:")
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
        if self.mixed:
            self.e('vmovd xmm15, edx')
            self.e('vpbroadcastd ymm15, xmm15')
            self.e(f"vpcmpeqd ymm15, ymm15, {self.pm('ENDS')}")
            for i in range(8):
                self.load(r(8), self.pm('SNAP', v*i))
                self.e(f'vpblendvb ymm8, ymm8, ymm{i}, ymm15')
                self.store(self.pm('SNAP', v*i), r(8))
        elif not self.last:
            self.e(f"cmp edx, dword ptr [{self.p}+{self.off['LAST']}]")
            self.e(f"jne {name}_next")
            for i in range(8):
                self.store(self.pm("SNAP", v * i), r(i))
            L.append(f"{name}_next:")
        self.e("add rdx, 64")
        self.e(f"add rbx, {v}")
        if self.last:
            self.e(f"cmp edx, dword ptr [{self.p}+{self.off['END']}]")
        else:
            self.e("cmp rdx, 1024")
        self.e(f"jne {name}_block")
        for i in range(8):
            self.store(self.pm("H", v * i), r(i))
        if self.avx:
            self.e("vzeroupper")
        self.e("mov rsp, rbp")
        for g in reversed(saved):
            self.e(f"pop {g}")
        self.e("ret")


def last16(mixed=False):
    """Sixteen messages, last blocks included (equal block counts unless
    mixed: every lane has its own ending offset at ParamsMixed.ends=384).
    Mixed results are stored word-major, each lane masked at its end. The input
    and output transposes and G operations are the wide kernel's; message
    words and state live in registers across all seven rounds.

    ABI: one Params16 pointer (ffi_last.rs). Offsets: MSG=0 (16 pointers),
    LASTMSG=128 (16 pointers), LENS=256 (16 u32s), KEY=320 (8 u32s),
    K=352 (whole blocks before last), FLAGS=356, OUT=360 (16 digests).
    The caller zero-pads last blocks. Every message has k full blocks
    followed by its own last block, length 0..64 (zero only when k=0).
    AVX-512F, System V x86-64. No calls or shared writable state.
    """
    import contextlib
    import io
    import gen_avx512_32 as wide

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        e = wide.emit
        name = 'blake3_servil_mixed_avx512' if mixed else 'blake3_servil_last_avx512'
        print(f'.p2align 6\n.global {name}\n.global _{name}\n{name}:\n_{name}:\n.cfi_startproc')
        e('_CET_ENDBR')
        e('push rbp')
        print('.cfi_def_cfa_offset 16\n.cfi_offset rbp, -16')
        e('mov rbp, rsp')
        print('.cfi_def_cfa_register rbp')
        for i, reg in enumerate(('rbx', 'r12', 'r13', 'r14', 'r15')):
            e(f'push {reg}')
            print(f'.cfi_offset {reg}, {-24-8*i}')
        e('mov r12, rdi')
        e('mov r15, QWORD PTR [r12 + 360]')
        e('mov r13d, DWORD PTR [r12 + 352]')
        e('shl r13, 6')
        e('mov ebx, DWORD PTR [r12 + 356]')
        e('xor r14d, r14d')
        for w in range(8):
            e(f'vpbroadcastd zmm{w}, DWORD PTR [r12 + {320+4*w}]')
        print('.Llast16_block:')
        if mixed:
            for i in range(16):
                e(f'mov rax, QWORD PTR [r12 + {8*i}]')
                e('add rax, r14')
                e(f'mov rdx, QWORD PTR [r12 + {128+8*i}]')
                e(f'cmp r14d, DWORD PTR [r12 + {384+4*i}]')
                e('cmovae rax, rdx')
                e(f'vmovdqu32 zmm{16+i}, ZMMWORD PTR [rax]')
        else:
            e('mov r8, r12')
            e('mov r9, r14')
            e('cmp r14, r13')
            e('jne .Llast16_load')
            e('lea r8, [r12 + 128]')
            e('xor r9d, r9d')
            print('.Llast16_load:')
            for i in range(16):
                e(f'mov rax, QWORD PTR [r8 + {8*i}]')
                e(f'vmovdqu32 zmm{16+i}, ZMMWORD PTR [rax + r9]')
        rows = wide.transpose(list(range(16, 32)), 8)
        messages = [rows[wide.WORD[w]] for w in range(16)]
        upper = [r for r in range(8, 32) if r not in messages]
        registers = list(range(8)) + upper
        assert len(upper) == 8 and len(set(registers + messages)) == 32
        for w in range(4):
            e(f'vpbroadcastd zmm{registers[8+w]}, DWORD PTR [rip + .Llast16_iv + {4*w}]')
        for w in (12, 13):
            e(f'vpxord zmm{registers[w]}, zmm{registers[w]}, zmm{registers[w]}')
        e('mov eax, ebx')
        e('test r14, r14')
        e('jnz .Llast16_flags')
        e('or eax, 1')  # CHUNK_START
        print('.Llast16_flags:')
        e('mov ecx, 64')
        e(f'vpbroadcastd zmm{registers[14]}, ecx')
        if mixed:
            e(f'vpbroadcastd zmm{registers[15]}, r14d')
            e(f'vpcmpeqd k1, zmm{registers[15]}, ZMMWORD PTR [r12 + 384]')
            e(f'vmovdqu32 zmm{registers[14]}{{k1}}, ZMMWORD PTR [r12 + 256]')
            e(f'vpbroadcastd zmm{registers[15]}, eax')
            e(f'vpord zmm{registers[15]}{{k1}}, zmm{registers[15]}, ZMMWORD PTR [rip + .Llast16_endflags]')
        else:
            e('cmp r14, r13')
            e('jne .Llast16_rounds')
            e('or eax, 10')  # CHUNK_END | ROOT
            e(f'vmovdqu32 zmm{registers[14]}, ZMMWORD PTR [r12 + 256]')
            print('.Llast16_rounds:')
            e(f'vpbroadcastd zmm{registers[15]}, eax')
        for permutation in schedule():
            for half, positions in enumerate((COLS, DIAGS)):
                chains = [wide.g(*[f'zmm{registers[w]}' for w in pos],
                                 f'zmm{messages[permutation[8*half+2*i]]}',
                                 f'zmm{messages[permutation[8*half+2*i+1]]}')
                          for i, pos in enumerate(positions)]
                for instructions in zip(*chains):
                    for instruction in instructions:
                        e(instruction)
        for w in range(8):
            e(f'vpxord zmm{w}, zmm{w}, zmm{registers[8+w]}')
        if mixed:
            for w in range(8):
                e(f'vmovdqu32 ZMMWORD PTR [r15 + {64*w}]{{k1}}, zmm{w}')
        e('add r14, 64')
        e('cmp r14, r13')
        e('jbe .Llast16_block')
        if not mixed:
            wide.output_group(0, 0)
        e('vzeroupper')
        for reg in ('r15', 'r14', 'r13', 'r12', 'rbx'):
            e(f'pop {reg}')
            print(f'.cfi_restore {reg}')
        e('pop rbp')
        print('.cfi_def_cfa rsp, 8\n.cfi_restore rbp')
        e('ret')
        print('.cfi_endproc\n.p2align 4\n.Llast16_iv:')
        print('    .long ' + ', '.join(hex(v) for v in wide.IV))
        if mixed:
            print('.p2align 6\n.Llast16_endflags:\n    .long ' + ', '.join(['10'] * 16))
    text = output.getvalue()
    if mixed:
        text = text.replace('.Llast16_', '.Lmixed16_')
    return text.splitlines()


def unix(family="partial"):
    last = family == "last"
    out = ["// Generated by tools/gen_x86.py" + (" --last" if last else "") + "; do not edit.",
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
    isas = ("avx2",) if last else ("sse41", "avx2")
    for isa in isas:
        name = f"blake3_servil_{family}_{isa}"
        out += [f".global _{name}", f".global {name}"]
    out += ["#ifdef __APPLE__", ".text", "#else", ".section .text", "#endif"]
    for isa in isas:
        name = f"blake3_servil_{family}_{isa}"
        k = Kernel(isa, last)
        k.body(name)
        out += ["        .p2align 6", f"_{name}:", f"{name}:", "        _CET_ENDBR"] + k.lines
    if last:
        name = 'blake3_servil_mixed_avx2'
        k = Kernel('avx2', last=True, mixed=True)
        k.body(name)
        out += [f'.global {name}', f'.global _{name}', '.p2align 6', f'{name}:', f'_{name}:', '        _CET_ENDBR'] + k.lines
        out += last16() + last16(mixed=True)
    return out


if __name__ == "__main__":
    assert sys.argv[1:] in ([], ["--last"]), "usage: python3 tools/gen_x86.py [--last] > c/blake3_{partial,last}_x86-64_unix.S"
    print("\n".join(unix("last" if sys.argv[1:] == ["--last"] else "partial")))
