#!/usr/bin/env python3
"""Generate c/blake3_neon_hybrid_aarch64.S: AArch64 kernels that hash several
whole 1024-byte chunks at once, one chunk on the integer ALUs beside two or
four chunks per NEON unit.

Why: a single chunk is a dependency chain. The scalar G runs at 12 cycles a
step on the integer units; the NEON "dup layout" G (one 32-bit word held
twice per 64-bit lane so `xar` rotates it) at 16-18; the classic four-lane
NEON G at 27-29. Integer and vector units have separate pipes and register
files, so a scalar chunk beside NEON work costs nothing extra. NEON register
pressure is the other limit: 32 vectors hold two units of state, so
messages are pre-transposed onto the stack each block and one unit's d row
lives on the stack too. Out-of-order renaming means one architectural temp
serves every G in flight.

Kernels: k<n> hashes n contiguous chunks. Composition per kernel:

  k1: scalar
  k2: pair                       k3: scalar + pair
  k4: pair + pair                k5: scalar + pair + pair
  k8: quad + quad                k9: scalar + quad + quad

C ABI, every kernel:
  fn(base: *const u8, blocks: u64, key: *const u32, counter: u64,
     packed_flags: u64, out: *mut u8)
  chunk i starts at base + 1024 * i; blocks in 1..=16 is the same for all;
  the counter is counter + i for chunk i; packed_flags =
  flags | flags_start << 8 | flags_end << 16; 32 bytes per chunk to out in
  input order. Requires NEON and the SHA-3 extension (`xar`). x18 untouched.

Run from the repository root:

    python3 tools/gen_neon_hybrid.py > c/blake3_neon_hybrid_aarch64.S

The generated file is committed so that building needs no Python; CI checks
that it matches the generator's output.
"""

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
CHUNK = 1024
BLOCK = 64
ROT8_TABLE = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]

# Stack frame (offsets from sp once the frame is built). 16-byte alignment
# for every q-register slot.
F_X19 = 0        # x19..x30 (96 bytes)
F_D8 = 96        # d8..d15 (64 bytes)
F_PACKED = 160   # packed flags
F_TOTAL = 168    # total blocks
F_OUT = 176      # out pointer
F_IVT = 184      # IV[0..4] as words (16 bytes) -- scalar c row
F_SCTPL = 200    # per scalar chunk: d-row template ctr (64-bit), 64 (64-bit); two slots
F_SDSPILL = 232  # per scalar chunk: spilled d row (16 bytes); two slots
F_IVDUP = 272    # IV[0..4], each broadcast to a 128-bit vector (64 bytes)
F_UNIT = 336     # per NEON unit: 32 bytes of constants (ctr_lo, ctr_hi)
UNIT_CONST = 32
F_ROT8 = F_UNIT + 2 * UNIT_CONST      # tbl constant for rotate-right-8 (16 bytes)
F_DSPILL = F_ROT8 + 16                # spilled d row of NEON unit 1 (64 bytes)
F_MSG = F_DSPILL + 64                 # per unit: 16 transposed message vectors (256 bytes)
FRAME = F_MSG + 2 * 256
# `stp q, q, [sp, #imm]` reaches 1008; every paired store must stay below.
assert F_MSG + 2 * 256 - 32 <= 1008, F_MSG
assert FRAME % 16 == 0


def w(reg):
    return "w" + reg[1:]


def x(reg):
    return "x" + reg[1:]


class Scalar:
    """One chunk on the integer registers: a, b, c rows in 12 registers and
    the d row either in 4 more or spilled to the stack (its live range within
    a G is short: read at `d ^= a`, dead after `c += d`). One message temp;
    with a spilled d row, one d temp. `w2`..`w5` are free during the rounds."""

    def __init__(self, index, slot, state, mtemp, dtemp=None, stride=CHUNK, ctr_step=1):
        self.index, self.slot = index, slot
        self.stride, self.ctr_step = stride, ctr_step
        self.state = state  # 12 or 16 w-registers: a0..3 b0..3 c0..3 [d0..3]
        self.mtemp, self.dtemp = mtemp, dtemp
        self.spill_d = len(state) == 12
        self.tpl = F_SCTPL + 16 * index
        self.dmem = F_SDSPILL + 16 * index

    def r(self, i):
        if i >= 12 and self.spill_d:
            return self.dtemp
        return self.state[i]

    def d_load(self, di):
        return [f"ldr {self.dtemp}, [sp, #{self.dmem + 4 * (di - 12)}]"] if self.spill_d else []

    def d_store(self, di):
        return [f"str {self.dtemp}, [sp, #{self.dmem + 4 * (di - 12)}]"] if self.spill_d else []

    def half_g(self, half, ai, bi, ci, di, word):
        """Half a G (one message word, one rotate pair) as a macro call. The
        message load and, with a spilled d row, the d load/store are part of
        the macro; see MACROS."""
        a, b, c, d, m = self.r(ai), self.r(bi), self.r(ci), self.r(di), self.mtemp
        off = self.stride * self.slot + 4 * word
        assert not self.spill_d, "spilled scalars emit whole Gs"
        name = "SGA" if half == 0 else "SGB"
        return [f"{name} {a}, {b}, {c}, {d}, {m}, {off}"]

    def half_step(self, quads, sched, base):
        out = []
        if self.spill_d:
            # Whole Gs: a spilled d word is loaded and stored once per G.
            off = lambda word: self.stride * self.slot + 4 * word
            for (ai, bi, ci, di), k in zip(quads, range(0, 8, 2)):
                a, b, c, d, m = self.r(ai), self.r(bi), self.r(ci), self.r(di), self.mtemp
                out.append(f"SG_SPILL {a}, {b}, {c}, {d}, {m}, {off(sched[base + k])}, {off(sched[base + k + 1])}, {self.dmem + 4 * (di - 12)}")
            return out
        # The four first halves, then the four second halves: independent
        # ops sit closer together in the instruction stream, which is worth
        # ~0.3 cycles per G-step to the out-of-order scheduler.
        for half in (0, 1):
            for q, k in zip(quads, range(0, 8, 2)):
                out += self.half_g(half, *q, sched[base + k + half])
        return out

    def prologue(self):
        # Key into a/b rows; d-row template [counter, 64] on the stack.
        out = []
        for i in range(0, 8, 2):
            out.append(f"ldp {self.r(i)}, {self.r(i + 1)}, [x2, #{4 * i}]")
        out += [
            f"add x4, x3, #{self.slot * self.ctr_step}",
            "mov x5, #64",
            f"stp x4, x5, [sp, #{self.tpl}]",
        ]
        return out

    def block_start(self, flags):
        c = self.state[8:12]
        out = [
            f"ldp {c[0]}, {c[1]}, [sp, #{F_IVT}]",
            f"ldp {c[2]}, {c[3]}, [sp, #{F_IVT + 8}]",
            f"ldp x3, x4, [sp, #{self.tpl}]",
            f"bfi x4, {x(flags)}, #32, #32",
        ]
        if self.spill_d:
            out.append(f"stp x3, x4, [sp, #{self.dmem}]")
        else:
            d = self.state[12:16]
            out += [
                f"mov {d[0]}, w3",
                f"lsr {x(d[1])}, x3, #32",
                f"mov {d[2]}, w4",
                f"lsr {x(d[3])}, x4, #32",
            ]
        return out

    def block_end(self):
        s = self.state
        out = [f"eor {s[i]}, {s[i]}, {s[i + 8]}" for i in range(4)]
        if self.spill_d:
            for i in range(4):
                out += self.d_load(12 + i)
                out.append(f"eor {s[4 + i]}, {s[4 + i]}, {self.dtemp}")
        else:
            out += [f"eor {s[4 + i]}, {s[4 + i]}, {s[12 + i]}" for i in range(4)]
        return out

    def store(self, outp):
        s, o = self.state, 32 * self.slot
        return [f"stp {s[i]}, {s[i + 1]}, [{outp}, #{o + 4 * i}]" for i in range(0, 8, 2)]


class Unit:
    """A NEON unit: a Pair (two chunks, dup layout) or a Quad (four chunks,
    classic layout). `state` is 16 vector numbers (a b c d rows) or 12 when
    the d row is spilled to F_DSPILL. Messages are read from the unit's
    F_MSG area, transposed there at block start."""

    def __init__(self, index, slots, state, spill_d, mtemp, dtemp, msg_regs=None, stride=CHUNK, ctr_step=1):
        self.index, self.slots, self.state = index, slots, state
        self.stride, self.ctr_step = stride, ctr_step
        self.spill_d, self.mtemp, self.dtemp = spill_d, mtemp, dtemp
        self.msg_regs = msg_regs  # 16 vector numbers, or None for stack-resident messages
        self.const = F_UNIT + UNIT_CONST * index
        self.msg = F_MSG + 256 * index

    def v(self, i):
        if i >= 12 and self.spill_d:
            return self.dtemp
        return f"v{self.state[i]}"

    def d_load(self, di):
        return [f"ldr q{self.dtemp[1:]}, [sp, #{F_DSPILL + 16 * (di - 12)}]"] if self.spill_d else []

    def d_store(self, di):
        return [f"str q{self.dtemp[1:]}, [sp, #{F_DSPILL + 16 * (di - 12)}]"] if self.spill_d else []

    def msg_load(self, word):
        if self.msg_regs:
            return None
        return f"ldr q{self.mtemp[1:]}, [sp, #{self.msg + 16 * word}]"

    def m(self, word):
        return f"v{self.msg_regs[word]}" if self.msg_regs else self.mtemp

    def half_step(self, quads, sched, base):
        out = []
        for q, k in zip(quads, range(0, 8, 2)):
            out += self.g(*q, sched[base + k], sched[base + k + 1])
        return out

    def block_end(self):
        out = []
        for i in range(4):
            out.append(f"eor {self.v(i)}.16b, {self.v(i)}.16b, {self.v(i + 8)}.16b")
        for i in range(4):
            out += self.d_load(12 + i)
            out.append(f"eor {self.v(4 + i)}.16b, {self.v(4 + i)}.16b, {self.v(12 + i)}.16b")
        return out

    def start_rows(self, flags):
        # c row <- IV; d row <- [ctr_lo, ctr_hi, 64, flags].
        out = [f"ldr q{self.state[8 + i]}, [sp, #{F_IVDUP + 16 * i}]" for i in range(4)]
        d = [self.v(12 + i) for i in range(4)]
        out += [
            f"ldr q{d[0][1:]}, [sp, #{self.const}]",
        ] + self.d_store(12) + [
            f"ldr q{d[1][1:]}, [sp, #{self.const + 16}]",
        ] + self.d_store(13) + [
            f"movi {d[2]}.4s, #64",
        ] + self.d_store(14) + [
            f"dup {d[3]}.4s, {flags}",
        ] + self.d_store(15)
        return out


class Pair(Unit):
    def g(self, ai, bi, ci, di, mx, my):
        a, b, c, d = self.v(ai), self.v(bi), self.v(ci), self.v(di)
        if self.msg_regs:
            return [f"PG_REG {a}, {b}, {c}, {d}, {self.m(mx)}, {self.m(my)}"]
        m = self.mtemp[1:]
        if self.spill_d:
            return [f"PG_SPILL {a}, {b}, {c}, {d[1:]}, {m}, {self.msg + 16 * mx}, {self.msg + 16 * my}, {F_DSPILL + 16 * (di - 12)}"]
        return [f"PG {a}, {b}, {c}, {d}, {m}, {self.msg + 16 * mx}, {self.msg + 16 * my}"]

    def prologue(self):
        # Counters of the two chunks, each duplicated within its 64-bit lane.
        sa, sb = self.slots
        return [
            f"add x4, x3, #{sa * self.ctr_step}",
            f"add x5, x3, #{sb * self.ctr_step}",
            "mov w6, w4",
            "bfi x6, x6, #32, #32",
            "mov w7, w5",
            "bfi x7, x7, #32, #32",
            f"stp x6, x7, [sp, #{self.const}]",
            "lsr x6, x4, #32",
            "bfi x6, x6, #32, #32",
            "lsr x7, x5, #32",
            "bfi x7, x7, #32, #32",
            f"stp x6, x7, [sp, #{self.const + 16}]",
        ]

    def transpose(self, temps):
        # Message words of chunks A and B into dup layout: (a_w a_w b_w b_w).
        sa, sb = self.slots
        t0, t1, t2, t3, t4, t5, t6, t7 = temps[:8]
        out = []
        for q in range(4):
            out += [
                f"ldr q{t0}, [x0, #{self.stride * sa + 16 * q}]",
                f"ldr q{t1}, [x0, #{self.stride * sb + 16 * q}]",
                f"zip1 v{t2}.4s, v{t0}.4s, v{t1}.4s",
                f"zip2 v{t3}.4s, v{t0}.4s, v{t1}.4s",
            ]
            if self.msg_regs:
                r = self.msg_regs[4 * q:4 * q + 4]
                out += [
                    f"zip1 v{r[0]}.4s, v{t2}.4s, v{t2}.4s",
                    f"zip2 v{r[1]}.4s, v{t2}.4s, v{t2}.4s",
                    f"zip1 v{r[2]}.4s, v{t3}.4s, v{t3}.4s",
                    f"zip2 v{r[3]}.4s, v{t3}.4s, v{t3}.4s",
                ]
            else:
                out += [
                    f"zip1 v{t4}.4s, v{t2}.4s, v{t2}.4s",
                    f"zip2 v{t5}.4s, v{t2}.4s, v{t2}.4s",
                    f"zip1 v{t6}.4s, v{t3}.4s, v{t3}.4s",
                    f"zip2 v{t7}.4s, v{t3}.4s, v{t3}.4s",
                    f"stp q{t4}, q{t5}, [sp, #{self.msg + 64 * q}]",
                    f"stp q{t6}, q{t7}, [sp, #{self.msg + 64 * q + 32}]",
                ]
        return out

    def store(self, outp, temps):
        sa, sb = self.slots
        t0, t1, t2, t3 = temps[:4]
        out = []
        for i in (0, 4):
            h = [self.state[i + k] for k in range(4)]
            out += [
                f"zip1 v{t0}.2d, v{h[0]}.2d, v{h[1]}.2d",
                f"zip1 v{t1}.2d, v{h[2]}.2d, v{h[3]}.2d",
                f"zip2 v{t2}.2d, v{h[0]}.2d, v{h[1]}.2d",
                f"zip2 v{t3}.2d, v{h[2]}.2d, v{h[3]}.2d",
                f"uzp1 v{t0}.4s, v{t0}.4s, v{t1}.4s",
                f"uzp1 v{t2}.4s, v{t2}.4s, v{t3}.4s",
                f"str q{t0}, [{outp}, #{32 * sa + 4 * i}]",
                f"str q{t2}, [{outp}, #{32 * sb + 4 * i}]",
            ]
        return out


class Quad(Unit):
    """Classic layout: lane j of every vector belongs to chunk slots[j].
    Rotates by 12 and 7 go through `rtemp` (shl + sri); the value moves
    between `b` and `rtemp` and is back in `b` at the end of the G."""

    def __init__(self, *args, rtemp, **kwargs):
        super().__init__(*args, **kwargs)
        self.rtemp = rtemp

    def g(self, ai, bi, ci, di, mx, my):
        a, b, c, d, m, t = self.v(ai), self.v(bi), self.v(ci), self.v(di), self.mtemp[1:], self.rtemp
        if self.spill_d:
            return [f"QG_SPILL {a}, {b}, {c}, {d[1:]}, {m}, {t}, {self.msg + 16 * mx}, {self.msg + 16 * my}, {F_DSPILL + 16 * (di - 12)}"]
        return [f"QG {a}, {b}, {c}, {d}, {m}, {t}, {self.msg + 16 * mx}, {self.msg + 16 * my}"]

    def prologue(self):
        out = []
        for j, s in enumerate(self.slots):
            out += [f"add x4, x3, #{s * self.ctr_step}", f"str w4, [sp, #{self.const + 4 * j}]", "lsr x4, x4, #32", f"str w4, [sp, #{self.const + 16 + 4 * j}]"]
        return out

    def transpose(self, temps):
        t = temps[:8]
        out = []
        for q in range(4):
            for j in range(4):
                out.append(f"ldr q{t[j]}, [x0, #{self.stride * self.slots[j] + 16 * q}]")
            out += [
                f"trn1 v{t[4]}.4s, v{t[0]}.4s, v{t[1]}.4s",
                f"trn2 v{t[5]}.4s, v{t[0]}.4s, v{t[1]}.4s",
                f"trn1 v{t[6]}.4s, v{t[2]}.4s, v{t[3]}.4s",
                f"trn2 v{t[7]}.4s, v{t[2]}.4s, v{t[3]}.4s",
                f"trn1 v{t[0]}.2d, v{t[4]}.2d, v{t[6]}.2d",
                f"trn1 v{t[1]}.2d, v{t[5]}.2d, v{t[7]}.2d",
                f"trn2 v{t[2]}.2d, v{t[4]}.2d, v{t[6]}.2d",
                f"trn2 v{t[3]}.2d, v{t[5]}.2d, v{t[7]}.2d",
                f"stp q{t[0]}, q{t[1]}, [sp, #{self.msg + 64 * q}]",
                f"stp q{t[2]}, q{t[3]}, [sp, #{self.msg + 64 * q + 32}]",
            ]
        return out

    def store(self, outp, temps):
        t = temps[:8]
        out = []
        for half in (0, 4):
            h = [self.state[half + k] for k in range(4)]
            out += [
                f"trn1 v{t[0]}.4s, v{h[0]}.4s, v{h[1]}.4s",
                f"trn2 v{t[1]}.4s, v{h[0]}.4s, v{h[1]}.4s",
                f"trn1 v{t[2]}.4s, v{h[2]}.4s, v{h[3]}.4s",
                f"trn2 v{t[3]}.4s, v{h[2]}.4s, v{h[3]}.4s",
                f"trn1 v{t[4]}.2d, v{t[0]}.2d, v{t[2]}.2d",
                f"trn1 v{t[5]}.2d, v{t[1]}.2d, v{t[3]}.2d",
                f"trn2 v{t[6]}.2d, v{t[0]}.2d, v{t[2]}.2d",
                f"trn2 v{t[7]}.2d, v{t[1]}.2d, v{t[3]}.2d",
            ]
            for j in range(4):
                out.append(f"str q{t[4 + j]}, [{outp}, #{32 * self.slots[j] + 4 * half}]")
        return out


MACROS = r"""
// Scalar half-G: a, b, c, d state words; m message temp; mx the message
// word's byte offset from x0. SGA is the first half (rotates 16, 12), SGB
// the second (8, 7). The message add comes first so it overlaps the
// previous half.
.macro SGA a, b, c, d, m, mx
    ldr \m, [x0, #\mx]
    add \a, \a, \m
    add \a, \a, \b
    eor \d, \d, \a
    ror \d, \d, #16
    add \c, \c, \d
    eor \b, \b, \c
    ror \b, \b, #12
.endm

.macro SGB a, b, c, d, m, mx
    ldr \m, [x0, #\mx]
    add \a, \a, \m
    add \a, \a, \b
    eor \d, \d, \a
    ror \d, \d, #8
    add \c, \c, \d
    eor \b, \b, \c
    ror \b, \b, #7
.endm

// Whole G with the d word spilled at [sp, #doff]; d is a temp register.
.macro SG_SPILL a, b, c, d, m, mx, my, doff
    ldr \m, [x0, #\mx]
    ldr \d, [sp, #\doff]
    add \a, \a, \m
    add \a, \a, \b
    eor \d, \d, \a
    ror \d, \d, #16
    add \c, \c, \d
    eor \b, \b, \c
    ror \b, \b, #12
    ldr \m, [x0, #\my]
    add \a, \a, \m
    add \a, \a, \b
    eor \d, \d, \a
    ror \d, \d, #8
    add \c, \c, \d
    str \d, [sp, #\doff]
    eor \b, \b, \c
    ror \b, \b, #7
.endm

// Pair G (dup layout: each 32-bit word twice per 64-bit lane, so xar on
// 64-bit lanes rotates the 32-bit words). Messages in registers.
.macro PG_REG a, b, c, d, mx, my
    add \a\().4s, \a\().4s, \mx\().4s
    add \a\().4s, \a\().4s, \b\().4s
    xar \d\().2d, \d\().2d, \a\().2d, #16
    add \c\().4s, \c\().4s, \d\().4s
    xar \b\().2d, \b\().2d, \c\().2d, #12
    add \a\().4s, \a\().4s, \my\().4s
    add \a\().4s, \a\().4s, \b\().4s
    xar \d\().2d, \d\().2d, \a\().2d, #8
    add \c\().4s, \c\().4s, \d\().4s
    xar \b\().2d, \b\().2d, \c\().2d, #7
.endm

// Pair G with messages on the stack at [sp, #mx], [sp, #my]; m is a temp.
.macro PG a, b, c, d, m, mx, my
    ldr q\m, [sp, #\mx]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \b\().4s
    xar \d\().2d, \d\().2d, \a\().2d, #16
    add \c\().4s, \c\().4s, \d\().4s
    xar \b\().2d, \b\().2d, \c\().2d, #12
    ldr q\m, [sp, #\my]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \b\().4s
    xar \d\().2d, \d\().2d, \a\().2d, #8
    add \c\().4s, \c\().4s, \d\().4s
    xar \b\().2d, \b\().2d, \c\().2d, #7
.endm

// Pair G, messages on the stack, d row spilled at [sp, #doff]; d is a temp.
.macro PG_SPILL a, b, c, d, m, mx, my, doff
    ldr q\d, [sp, #\doff]
    ldr q\m, [sp, #\mx]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \b\().4s
    xar v\d\().2d, v\d\().2d, \a\().2d, #16
    add \c\().4s, \c\().4s, v\d\().4s
    xar \b\().2d, \b\().2d, \c\().2d, #12
    ldr q\m, [sp, #\my]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \b\().4s
    xar v\d\().2d, v\d\().2d, \a\().2d, #8
    add \c\().4s, \c\().4s, v\d\().4s
    str q\d, [sp, #\doff]
    xar \b\().2d, \b\().2d, \c\().2d, #7
.endm

// Quad G (classic layout: lane j = input j). Rotates by 16 and 8 are rev32
// and tbl (v31 holds the byte table); 12 and 7 are shl + sri through t.
.macro QG a, b, c, d, m, t, mx, my
    ldr q\m, [sp, #\mx]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \b\().4s
    eor \d\().16b, \d\().16b, \a\().16b
    rev32 \d\().8h, \d\().8h
    add \c\().4s, \c\().4s, \d\().4s
    eor \b\().16b, \b\().16b, \c\().16b
    shl \t\().4s, \b\().4s, #20
    sri \t\().4s, \b\().4s, #12
    ldr q\m, [sp, #\my]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \t\().4s
    eor \d\().16b, \d\().16b, \a\().16b
    tbl \d\().16b, {\d\().16b}, v31.16b
    add \c\().4s, \c\().4s, \d\().4s
    eor \t\().16b, \t\().16b, \c\().16b
    shl \b\().4s, \t\().4s, #25
    sri \b\().4s, \t\().4s, #7
.endm

.macro QG_SPILL a, b, c, d, m, t, mx, my, doff
    ldr q\d, [sp, #\doff]
    ldr q\m, [sp, #\mx]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \b\().4s
    eor v\d\().16b, v\d\().16b, \a\().16b
    rev32 v\d\().8h, v\d\().8h
    add \c\().4s, \c\().4s, v\d\().4s
    eor \b\().16b, \b\().16b, \c\().16b
    shl \t\().4s, \b\().4s, #20
    sri \t\().4s, \b\().4s, #12
    ldr q\m, [sp, #\my]
    add \a\().4s, \a\().4s, v\m\().4s
    add \a\().4s, \a\().4s, \t\().4s
    eor v\d\().16b, v\d\().16b, \a\().16b
    tbl v\d\().16b, {v\d\().16b}, v31.16b
    add \c\().4s, \c\().4s, v\d\().4s
    str q\d, [sp, #\doff]
    eor \t\().16b, \t\().16b, \c\().16b
    shl \b\().4s, \t\().4s, #25
    sri \b\().4s, \t\().4s, #7
.endm
"""


def interleave(lists):
    out, iters = [], [iter(l) for l in lists]
    live = list(range(len(lists)))
    while live:
        nxt = []
        for i in live:
            try:
                out.append(next(iters[i]))
                nxt.append(i)
            except StopIteration:
                pass
        live = nxt
    return out


def kernel(name, scalars, units):
    L = []
    e = L.append
    e(f"{name}:")
    e(f"sub sp, sp, #{FRAME}")
    for i in range(0, 12, 2):
        e(f"stp x{19 + i}, x{20 + i}, [sp, #{F_X19 + 8 * i}]")
    for i in range(0, 8, 2):
        e(f"stp d{8 + i}, d{9 + i}, [sp, #{F_D8 + 8 * i}]")
    e(f"str x4, [sp, #{F_PACKED}]")
    e(f"str x1, [sp, #{F_TOTAL}]")
    e(f"str x5, [sp, #{F_OUT}]")
    for i in range(0, 4, 2):
        e(f"mov w6, #{IV[i] & 0xffff}")
        e(f"movk w6, #{IV[i] >> 16}, lsl #16")
        e(f"mov w7, #{IV[i + 1] & 0xffff}")
        e(f"movk w7, #{IV[i + 1] >> 16}, lsl #16")
        e(f"stp w6, w7, [sp, #{F_IVT + 4 * i}]")
    if units:
        for i in range(4):
            e(f"ldr s0, [sp, #{F_IVT + 4 * i}]")
            e("dup v0.4s, v0.s[0]")
            e(f"str q0, [sp, #{F_IVDUP + 16 * i}]")
        lo = sum(ROT8_TABLE[i] << (8 * i) for i in range(8))
        hi = sum(ROT8_TABLE[8 + i] << (8 * i) for i in range(8))
        for reg, val in (("x6", lo), ("x7", hi)):
            e(f"mov {reg}, #{val & 0xffff}")
            for k in range(1, 4):
                e(f"movk {reg}, #{(val >> (16 * k)) & 0xffff}, lsl #{16 * k}")
        e(f"stp x6, x7, [sp, #{F_ROT8}]")
        e(f"ldr q31, [sp, #{F_ROT8}]")
        for u in units:
            L += u.prologue()
        # Key into every unit's a/b rows.
        for i in range(8):
            e(f"ldr s0, [x2, #{4 * i}]")
            for u in units:
                if u.state[i] != 0:
                    e(f"dup v{u.state[i]}.4s, v0.s[0]")
        for u in units:
            if u.state[0] == 0:
                e("ldr s0, [x2]")
                e("dup v0.4s, v0.s[0]")
    for sc in scalars:
        L += sc.prologue()
    e(f"{name}_loop:")
    e(f"ldr w3, [sp, #{F_PACKED}]")
    e(f"ldr x4, [sp, #{F_TOTAL}]")
    e("and w2, w3, #0xff")
    e("ubfx w5, w3, #8, #8")
    e("cmp x1, x4")
    e("csel w5, w5, wzr, eq")
    e("orr w2, w2, w5")
    e("ubfx w5, w3, #16, #8")
    e("cmp x1, #1")
    e("csel w5, w5, wzr, eq")
    e("orr w2, w2, w5")
    if units:
        # c and d rows of every unit are dead here: use them as transposition temps.
        # c and d rows of every unit are dead here: use them as transposition
        # temps. A unit with register-resident messages must not have its
        # message registers used as temps.
        temps = [n for u in units for n in u.state[8:]]
        if not any(u.msg_regs for u in units):
            temps += [int(units[0].mtemp[1:]), int(units[0].dtemp[1:])]
            if isinstance(units[0], Quad):
                temps.append(int(units[0].rtemp[1:]))
        for u in units:
            L += u.transpose(temps)
        for u in units:
            L += u.start_rows("w2")
    for sc in scalars:
        L += sc.block_start("w2")
    for r in range(7):
        for quads, base in ((COLS, 0), (DIAGS, 8)):
            lists = [sc.half_step(quads, SCHEDULE[r], base) for sc in scalars]
            vec = []
            for u in units:
                vec += u.half_step(quads, SCHEDULE[r], base)
            if vec:
                lists.append(vec)
            L += interleave(lists)
    for sc in scalars:
        L += sc.block_end()
    for u in units:
        L += u.block_end()
    e("add x0, x0, #64")
    e("subs x1, x1, #1")
    e(f"b.ne {name}_loop")
    e(f"ldr x2, [sp, #{F_OUT}]")
    for sc in scalars:
        L += sc.store("x2")
    if units:
        temps = [n for u in units for n in u.state[8:]] + [int(units[0].mtemp[1:]), int(units[0].dtemp[1:])]
        for u in units:
            L += u.store("x2", temps)
    for i in range(0, 12, 2):
        e(f"ldp x{19 + i}, x{20 + i}, [sp, #{F_X19 + 8 * i}]")
    for i in range(0, 8, 2):
        e(f"ldp d{8 + i}, d{9 + i}, [sp, #{F_D8 + 8 * i}]")
    e(f"add sp, sp, #{FRAME}")
    e("ret")
    return L


def build():
    # One scalar chunk: 16 state registers w6..w17, w19..w22; message temp w23.
    one = [f"w{i}" for i in list(range(6, 18)) + list(range(19, 23))]
    sc1 = lambda slot: [Scalar(0, slot, one, "w23")]
    # Two scalar chunks with spilled d rows: 12 + 12 state registers in
    # w6..w17 and w19..w30; message temps w2/w3... those are flag temps at
    # block start only, so w2..w5 serve as message and d temps in the rounds.
    two_a = [f"w{i}" for i in range(6, 18)]
    two_b = [f"w{i}" for i in range(19, 31)]
    sc2 = lambda s0, s1: [Scalar(0, s0, two_a, "w2", "w3"), Scalar(1, s1, two_b, "w4", "w5")]
    # NEON unit 0 keeps all 16 rows; unit 1 spills its d row. v28 message
    # temp, v29 d temp, v30 rotate temp, v31 rot8 table.
    u0 = list(range(0, 16))
    u1 = list(range(16, 28))
    pair = lambda idx, slots: Pair(idx, slots, u0 if idx == 0 else u1, idx == 1, "v28", "v29")
    # A lone pair keeps its messages in v16..v31 (no temps needed: xar is in place).
    lone_pair = lambda slots: Pair(0, slots, u0, False, "v28", "v29", msg_regs=list(range(16, 32)))
    quad = lambda idx, slots: Quad(idx, slots, u0 if idx == 0 else u1, idx == 1, "v28", "v29", rtemp="v30")
    P = dict(stride=BLOCK, ctr_step=0)
    ppair = lambda idx, slots: Pair(idx, slots, u0 if idx == 0 else u1, idx == 1, "v28", "v29", **P)
    plone = lambda slots: Pair(0, slots, u0, False, "v28", "v29", msg_regs=list(range(16, 32)), **P)
    pquad = lambda idx, slots: Quad(idx, slots, u0 if idx == 0 else u1, idx == 1, "v28", "v29", rtemp="v30", **P)
    return {
        "p2": kernel("blake3_hybrid_p2", [], [plone((0, 1))]),
        "p4": kernel("blake3_hybrid_p4", [], [ppair(0, (0, 1)), ppair(1, (2, 3))]),
        "p8": kernel("blake3_hybrid_p8", [], [pquad(0, (0, 1, 2, 3)), pquad(1, (4, 5, 6, 7))]),
        "k1": kernel("blake3_hybrid_k1", sc1(0), []),
        "k2": kernel("blake3_hybrid_k2", [], [lone_pair((0, 1))]),
        "k3": kernel("blake3_hybrid_k3", sc1(0), [lone_pair((1, 2))]),
        "k4": kernel("blake3_hybrid_k4", sc2(0, 1), [lone_pair((2, 3))]),
        "k5": kernel("blake3_hybrid_k5", sc1(0), [pair(0, (1, 2)), pair(1, (3, 4))]),
        "k6": kernel("blake3_hybrid_k6", sc2(0, 1), [pair(0, (2, 3)), pair(1, (4, 5))]),
        "k8": kernel("blake3_hybrid_k8", [], [quad(0, (0, 1, 2, 3)), quad(1, (4, 5, 6, 7))]),
        "k9": kernel("blake3_hybrid_k9", sc1(0), [quad(0, (1, 2, 3, 4)), quad(1, (5, 6, 7, 8))]),
        "k10": kernel("blake3_hybrid_k10", sc2(0, 1), [quad(0, (2, 3, 4, 5)), quad(1, (6, 7, 8, 9))]),
    }


HEADER = """\
// AArch64 kernels that hash whole BLAKE3 chunks on the integer ALUs and NEON
// at the same time. Generated by tools/gen_neon_hybrid.py; edit the
// generator rather than this file.
//
// One chunk's compression is a dependency chain. The scalar G runs at
// twelve cycles per step on the integer units; the NEON "dup layout" G
// (each 32-bit word held twice in a 64-bit lane so xar rotates it) at
// sixteen to eighteen; the classic four-lane NEON G at about twenty-eight.
// Integer and vector units have separate pipes and register files, so a
// chunk on the integer side beside NEON work finishes inside the NEON time.
//
// k<n> hashes n contiguous chunks: k1 scalar; k2 one NEON pair; k3 scalar +
// pair; k4 two scalars + pair; k5 scalar + two pairs; k6 two scalars + two
// pairs; k8 two four-lane quads; k9 scalar + two quads; k10 two scalars +
// two quads. p<n> hashes n contiguous 64-byte parent blocks with one shared
// counter: p2 pair, p4 two pairs, p8 two quads. Messages are transposed onto
// the stack each block and the second NEON unit's d row lives there too,
// which is what lets two units of state share 32 vector registers.
//
// Every kernel has the C signature
//   void kernel(const uint8_t *base, uint64_t blocks, const uint32_t key[8],
//               uint64_t counter, uint64_t packed_flags, uint8_t *out);
// Input i starts at base + stride * i, stride 1024 for k and 64 for p;
// blocks is in 1..=16 and the same for every input (1 for p); input i uses
// counter + i (k) or counter (p); packed_flags is
// flags | flags_start << 8 | flags_end << 16, applied per block; out
// receives 32 bytes per input in input order. The CPU must have NEON and
// the SHA-3 extension (xar). x18 is untouched; AAPCS64 otherwise.

#if defined(__ELF__) && defined(__linux__)
.section .note.GNU-stack,"",%progbits
#endif

.arch armv8.2-a+sha3
"""


def asm_source(kernels):
    names = [f"blake3_hybrid_{k}" for k in kernels]
    out = [HEADER]
    for n in names:
        out.append(f".global {n}")
        out.append(f".global _{n}")
    out.append("#ifdef __APPLE__\n.text\n#else\n.section .text\n#endif")
    out.append(MACROS)
    for name, body in zip(names, kernels.values()):
        out.append("        .p2align 6")
        out.append(f"_{name}:")
        for ins in body:
            if ins.endswith(":"):
                out.append(ins)
            else:
                out.append("        " + ins)
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(asm_source(build()), end="")
