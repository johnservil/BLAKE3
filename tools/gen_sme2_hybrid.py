#!/usr/bin/env python3
"""Generate c/blake3_sme2_hybrid_aarch64.S: SME2 chunk kernels that carry
an integer BLAKE3 lane beside the sixteen streaming-vector lanes.

Why: the core issues SME2 vector ops to the SME unit one per cycle and
keeps executing independent integer ops beside them at about four per
cycle, on P- and E-cores alike (probe/sme-scalar, NOTES-servil.md). One
SME2 block step (sixteen chunks' compressions) takes about 700 cycles; a
scalar compression is a 168-cycle dependency chain. So `k` scalar blocks
fit beside each SME2 block step, and a group hashes 16 + k chunks: the
SME2 lanes take chunks 0-15, the integer lane chunks 16..16+k/16*16... in
turn, k blocks per step, so over sixteen steps it finishes k chunks.

Kernels (k divides 16, so integer chunks start and end at loop-iteration
boundaries): blake3_sme2x<k>_hash_chunks_512 for k in KS.

C ABI, like blake3_sme2_hash16_chunks_512:
    uint64_t f(const uint8_t *const *inputs, const uint32_t key[8],
               uint64_t counter, uint32_t flags, uint8_t *out, uint64_t groups)
inputs holds (16 + k) chunk pointers per group; chunk i of a group gets
counter + i; out receives (16 + k) chaining values per group in order;
flags packs flags | flags_start << 8 | flags_end << 16. Returns the
streaming vector length in 32-bit lanes; work happens only when it is 16.
x18 untouched; d8-d15 and x19-x30 preserved.

The SME2 instruction sequences are the macros of
c/blake3_sme2_aarch64.S, expanded here so the two instruction streams can
be interleaved one instruction at a time.

    python3 tools/gen_sme2_hybrid.py > c/blake3_sme2_hybrid_aarch64.S
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
SCHEDULE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
]
COLS = [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)]
DIAGS = [(0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)]
KS = [1, 2, 4]

# ---------------------------------------------------------------- SME2 side


def sme2_macros():
    """The .macro definitions of the SME2 kernel file: name -> (params, body lines)."""
    text = (ROOT / "c/blake3_sme2_aarch64.S").read_text()
    macros = {}
    for m in re.finditer(r"^\.macro (\w+)([^\n]*)\n(.*?)^\.endm", text, re.S | re.M):
        params = [p.strip() for p in m.group(2).split(",") if p.strip()]
        body = [l.strip() for l in m.group(3).splitlines()]
        body = [l for l in body if l and not l.startswith("/*") and not l.startswith("*") and not l.startswith("//")]
        macros[m.group(1)] = (params, body)
    return macros


MACROS = sme2_macros()


def expand(line):
    """One source line, macros expanded recursively, as instruction lines."""
    parts = line.split(None, 1)
    if not parts or parts[0] not in MACROS:
        return [line]
    params, body = MACROS[parts[0]]
    args = [a.strip() for a in parts[1].split(",")] if len(parts) > 1 else []
    assert len(args) == len(params), (line, params)
    out = []
    for b in body:
        for p, a in zip(params, args):
            b = re.sub(r"\\" + p + r"\b", a, b)
        out += expand(b)
    return out


# ------------------------------------------------------------- integer side

# Integer lane registers: 16 state words, one message temp, the running
# block pointer, the chunk counter, one temp.
STATE = ["w1", "w2", "w3", "w4", "w5", "w6", "w11", "w16", "w17", "w19", "w20", "w21", "w22", "w23", "w24", "w25"]
M = "w26"
P = "x27"
CTR = "x28"
T = "x30"

# Frame.
F_D8 = 0          # d8-d15
F_X19 = 64        # x19-x30
F_KEY = 160       # key pointer
F_CTR = 168       # counter of the group's first chunk
F_FLAGS = 176     # packed flags
F_OUT = 184       # out pointer
F_GROUPS = 192    # groups remaining
F_FMID = 200      # flags (word)
F_FFIRST = 204    # flags | flags_start
F_FLAST = 208     # flags | flags_end
FRAME = 224


def w_(reg):
    return "w" + reg[1:]


def x_(reg):
    return "x" + reg[1:]


def int_block(k, i):
    """Integer block i (0..k-1) of this iteration: block start (c = IV, d =
    counter, 64, flags), seven rounds, feed-forward, pointer advance. x7
    holds the SME2 block index b; the integer block's index in its chunk is
    j = (b * k + i) & 15, first at 0, last at 15."""
    s = STATE
    out = []
    # Flags: mid, | start at j == 0, | end at j == 15. j = (x7 * k + i) & 15.
    out += [f"lsl {T}, x7, #{k.bit_length() - 1}" if k > 1 else f"mov {T}, x7"]
    if i:
        out += [f"add {T}, {T}, #{i}"]
    out += [f"and {T}, {T}, #15",
            f"ldr {s[15]}, [sp, #{F_FMID}]",
            f"ldr {M}, [sp, #{F_FFIRST}]",
            f"cmp {T}, #0",
            f"csel {s[15]}, {M}, {s[15]}, eq",
            f"ldr {M}, [sp, #{F_FLAST}]",
            f"cmp {T}, #15",
            f"csel {s[15]}, {M}, {s[15]}, eq"]
    for n in range(4):
        out += [f"mov {s[8 + n]}, #{IV[n] & 0xffff}", f"movk {s[8 + n]}, #{IV[n] >> 16}, lsl #16"]
    out += [f"mov {s[12]}, {w_(CTR)}", f"lsr {x_(s[13])}, {CTR}, #32", f"mov {s[14]}, #64"]
    for r in range(7):
        for quads, base in ((COLS, 0), (DIAGS, 8)):
            for half in (0, 1):
                for (ai, bi, ci, di), q in zip(quads, range(4)):
                    a, b, c, d = s[ai], s[bi], s[ci], s[di]
                    word = SCHEDULE[r][base + 2 * q + half]
                    rot1, rot2 = (16, 12) if half == 0 else (8, 7)
                    out += [f"ldr {M}, [{P}, #{4 * word}]",
                            f"add {a}, {a}, {M}",
                            f"add {a}, {a}, {b}",
                            f"eor {d}, {d}, {a}",
                            f"ror {d}, {d}, #{rot1}",
                            f"add {c}, {c}, {d}",
                            f"eor {b}, {b}, {c}",
                            f"ror {b}, {b}, #{rot2}"]
    out += [f"eor {s[n]}, {s[n]}, {s[n + 8]}" for n in range(4)]
    out += [f"eor {s[4 + n]}, {s[4 + n]}, {s[12 + n]}" for n in range(4)]
    out += [f"add {P}, {P}, #64"]
    return out


def interleave(a, b):
    """Merge two instruction lists keeping each one's order, b spread evenly
    through a."""
    if not b:
        return list(a)
    if not a:
        return list(b)
    out, j = [], 0
    for i, ins in enumerate(a):
        out.append(ins)
        target = (i + 1) * len(b) // len(a)
        while j < target:
            out.append(b[j])
            j += 1
    out += b[j:]
    return out


def kernel(k):
    name = f"blake3_sme2x{k}_hash_chunks_512"
    per_chunk = 16 // k          # iterations per integer chunk
    shift = per_chunk.bit_length() - 1
    lanes = 16 + k
    L = []
    e = L.append
    e(".p2align 4")
    e(f"_{name}:")
    e(f"{name}:")
    e(f"sub sp, sp, #{FRAME}")
    for n in range(8, 16, 2):
        e(f"stp d{n}, d{n + 1}, [sp, #{F_D8 + 8 * (n - 8)}]")
    for n in range(19, 31, 2):
        e(f"stp x{n}, x{n + 1}, [sp, #{F_X19 + 8 * (n - 19)}]")
    e(f"str x1, [sp, #{F_KEY}]")
    e(f"str x2, [sp, #{F_CTR}]")
    e(f"str x3, [sp, #{F_FLAGS}]")
    e(f"str x4, [sp, #{F_OUT}]")
    e(f"str x5, [sp, #{F_GROUPS}]")
    # Flag words for the integer lane (the SME2 side reads them too).
    e("and w8, w3, #0xff")
    e(f"str w8, [sp, #{F_FMID}]")
    e("ubfx w9, w3, #8, #8")
    e("orr w9, w9, w8")
    e(f"str w9, [sp, #{F_FFIRST}]")
    e("ubfx w9, w3, #16, #8")
    e("orr w9, w9, w8")
    e(f"str w9, [sp, #{F_FLAST}]")
    e("smstart sm")
    e("smstart za")
    e("cntw x8")
    e("cmp x8, #16")
    e(f"b.eq {name}_vl_ok")
    e("mov x0, x8")
    e(f"b {name}_return")
    e(f"{name}_vl_ok:")
    e("ptrue p0.s")
    e("mov x8, #8")
    e("whilelo p2.s, xzr, x8")
    e("mov w12, #0")
    e("mov w13, #4")
    e("mov w14, #8")
    e("mov w15, #12")
    for n, reg in enumerate(range(16, 20)):
        e(f"mov w8, #{IV[n] & 0xffff}")
        e(f"movk w8, #{IV[n] >> 16}, lsl #16")
        e(f"dup z{reg}.s, w8")
    e("mov za2h.s[w12, 0:3], { z16.s - z19.s }")
    e(f"ldr x8, [sp, #{F_GROUPS}]")
    e(f"cbz x8, {name}_success")

    e(f"{name}_group:")
    # SME2 lanes: key and counters (as the plain kernel).
    e(f"ldr x1, [sp, #{F_KEY}]")
    for n in range(8):
        e(f"ldr w8, [x1, #{4 * n}]")
        e(f"dup z{n}.s, w8")
    e(f"ldr x2, [sp, #{F_CTR}]")
    e("index z12.s, w2, #1")
    e("lsr x8, x2, #32")
    e("dup z13.s, w8")
    e("dup z16.s, w2")
    e("cmplo p1.s, p0/z, z12.s, z16.s")
    e("mov z17.s, #1")
    e("add z13.s, p1/m, z13.s, z17.s")
    e("mov za3h.s[w12, 0:1], { z12.s - z13.s }")
    e("mov x9, #0")
    L.extend(expand("LOAD_BLOCK"))
    e("mov x7, #0")
    e("mov x9, #64")

    def iteration(last):
        body = []
        b = body.append
        # Integer chunk start at iterations b % per_chunk == 0: key into
        # the a/b rows, the chunk's pointer and counter.
        lbl = f"{name}_{'l' if last else 'p'}_nostart"
        if per_chunk > 1:
            b(f"tst x7, #{per_chunk - 1}")
            b(f"b.ne {lbl}")
        b(f"lsr {T}, x7, #{shift}")
        b(f"add {T}, {T}, #16")
        b(f"ldr {P}, [x0, {T}, lsl #3]")
        b(f"ldr {CTR}, [sp, #{F_CTR}]")
        b(f"add {CTR}, {CTR}, {T}")
        b(f"ldr x8, [sp, #{F_KEY}]")
        for n in range(0, 8, 2):
            b(f"ldp {STATE[n]}, {STATE[n + 1]}, [x8, #{4 * n}]")
        b(f"{lbl}:")
        # SME2 block flags into w10: first block, last block, else mid.
        if last:
            b(f"ldr w10, [sp, #{F_FLAST}]")
        else:
            b(f"ldr w10, [sp, #{F_FMID}]")
            b(f"ldr w8, [sp, #{F_FFIRST}]")
            b("cmp x7, #0")
            b("csel w10, w8, w10, eq")
        body.extend(expand("INITIALIZE_UPPER_STATE w10"))
        body.extend(expand("READ_MESSAGES"))
        sme = []
        if not last:
            # Prefetch the next group's SME2 chunks at this offset (as the
            # plain kernel), when there is a next group.
            pass
        sme += expand("SEVEN_ROUNDS" if last else "SEVEN_ROUNDS_PIPELINED")
        sme += expand("FEED_FORWARD")
        ints = []
        for i in range(k):
            ints += int_block(k, i)
        body.extend(interleave(sme, ints))
        # Integer chunk end at iterations (b + 1) % per_chunk == 0: its CV
        # to out + 32 * (16 + q).
        lbl2 = f"{name}_{'l' if last else 'p'}_noend"
        b(f"add x8, x7, #1")
        if per_chunk > 1:
            b(f"tst x8, #{per_chunk - 1}")
            b(f"b.ne {lbl2}")
        b(f"lsr {T}, x7, #{shift}")
        b(f"add {T}, {T}, #16")
        b(f"ldr x8, [sp, #{F_OUT}]")
        b(f"add x8, x8, {T}, lsl #5")
        for n in range(0, 8, 2):
            b(f"stp {STATE[n]}, {STATE[n + 1]}, [x8, #{4 * n}]")
        b(f"{lbl2}:")
        return body

    e(f"{name}_block:")
    L.extend(iteration(False))
    e("add x7, x7, #1")
    e("add x9, x9, #64")
    e("cmp x7, #15")
    e(f"b.lo {name}_block")
    L.extend(iteration(True))
    # SME2 outputs, as the plain kernel.
    e("mov za1v.s[w12, 0:3], { z0.s - z3.s }")
    e("mov za1v.s[w13, 0:3], { z4.s - z7.s }")
    e(f"ldr x4, [sp, #{F_OUT}]")
    for n, (sel, sl) in enumerate([(s, t) for s in ("w12", "w13", "w14", "w15") for t in range(4)]):
        e(f"add x8, x4, #{32 * n}")
        e(f"st1w {{ za1h.s[{sel}, {sl}] }}, p2, [x8]")
    # Next group.
    e(f"add x0, x0, #{8 * lanes}")
    e(f"add x4, x4, #{32 * lanes}")
    e(f"str x4, [sp, #{F_OUT}]")
    e(f"ldr x2, [sp, #{F_CTR}]")
    e(f"add x2, x2, #{lanes}")
    e(f"str x2, [sp, #{F_CTR}]")
    e(f"ldr x8, [sp, #{F_GROUPS}]")
    e("subs x8, x8, #1")
    e(f"str x8, [sp, #{F_GROUPS}]")
    e(f"b.ne {name}_group")
    e(f"{name}_success:")
    e("mov x0, #16")
    e(f"{name}_return:")
    e("smstop za")
    e("smstop sm")
    for n in range(8, 16, 2):
        e(f"ldp d{n}, d{n + 1}, [sp, #{F_D8 + 8 * (n - 8)}]")
    for n in range(19, 31, 2):
        e(f"ldp x{n}, x{n + 1}, [sp, #{F_X19 + 8 * (n - 19)}]")
    e(f"add sp, sp, #{FRAME}")
    e("ret")
    return name, L


HEADER = """\
// SME2 chunk kernels with an integer BLAKE3 lane beside the sixteen
// streaming-vector lanes. Generated by tools/gen_sme2_hybrid.py; edit the
// generator rather than this file. See the generator for the design and
// the ABI.

#if defined(__ELF__) && defined(__linux__)
.section .note.GNU-stack,"",%progbits
#endif

.arch armv9-a+sme2
"""


def main():
    out = [HEADER]
    names, bodies = [], []
    for k in KS:
        name, body = kernel(k)
        names.append(name)
        bodies.append(body)
    for name in names:
        out.append(f".global _{name}\n.global {name}")
    out.append("#ifdef __APPLE__\n.text\n#else\n.section .text\n#endif")
    for body in bodies:
        out.append("\n".join(l if l.endswith(":") or l.startswith(".") else "    " + l for l in body))
    sys.stdout.write("\n\n".join(out) + "\n")


if __name__ == "__main__":
    main()
