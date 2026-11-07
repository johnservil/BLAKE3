"""Proofs of the x86-64 kernels (c/blake3_{sse41,avx2,avx512}_x86-64_unix.S,
c/blake3_partial_x86-64_unix.S and c/blake3_chunk_root_x86-64_unix.S, under
the System V calling convention of Linux and macOS), as assembled, equal to the
compression function of the Lean specification (lean_spec.py) for every
input, key, counter, and flags value, by the x86-64 model (x86.py): one
compression (compress_in_place, compress_xof), hash_many at every input
count up to two passes of its width and every block count from 1 to 16,
the generated pass of whole chunks with a partial one (each lane's
16 blocks, every per-lane, per-block length and flags, at every block
after which its snapshot is taken), and the generated wide AVX-512
kernels (c/blake3_avx512_32.S: 32 inputs, counters incrementing or fixed,
and 16 inputs, at 1 to 16 blocks).

    python3 tools/verify/prove_x86.py [--workers N] [CASE...]
    python3 tools/verify/prove_x86.py [--workers N] --representative

--workers (default 16) bounds the prover processes at once: the 16-input
AVX-512 cases take gigabytes each, so choose it from the host's memory.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from z3 import And, BitVec, BitVecVal, Concat, Extract, UGE, ULE, ZeroExt, simplify

import aarch64
import x86
from aarch64 import Ptr, Region, Unproved
from prove_hybrid import byte_symbols, same, table_cvs
import lean_spec as spec

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
CLANG = x86.tool("clang")
STACK = 1 << 16
# A 64-byte-aligned region origin; the ABI enters with rsp == 8 mod 16.
ENTRY_STACKS = tuple(STACK + residue for residue in (8, 24, 40, 56))
WIDTH = {"sse41": 4, "avx2": 8, "avx512": 16}
SYSV_ARGS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]


def assemble(isa):
    """`isa`'s file (sse41, avx2, avx512, partial: the generated pass, or
    chunk_root)."""
    if isa in OBJS:
        return OBJS[isa]
    d = tempfile.mkdtemp()
    obj = os.path.join(d, f"{isa}.o")
    # "chunk_root": the one file holding both chunk-root entries (sse41, avx512);
    # "wide": the generated 32- and 16-input AVX-512 kernels (k32, p32, p16).
    src = {"wide": "c/blake3_avx512_32.S", "output": "c/blake3_output16.S", "output8": "c/blake3_output8.S"}.get(isa, f"c/blake3_{isa}_x86-64_unix.S")
    subprocess.run([CLANG, "-c", "--target=x86_64-linux-gnu", os.path.join(ROOT_DIR, src), "-o", obj], check=True)
    so = x86.link(obj)
    OBJS[isa] = so
    return so


class Setup:
    """A machine at a kernel's entry: junk in every register, the return
    address on the stack, the arguments placed by the convention.
    The block/output counter is public; each harness adds its count symbols."""
    def __init__(self, so, entry, regions, args, stack=None):
        # The default proves every legal entry alignment at once: the
        # region origin is 64-byte aligned and rsp == 8 (mod 16) at entry.
        # A concrete stack remains available to the boundary unit tests.
        if stack is None:
            residue = BitVec("len_stack", 64)
            entry_stack = BitVecVal(STACK, 64) + residue
            entry_assumptions = [UGE(residue, 8), ULE(residue, 56), residue & 15 == 8]
        else:
            assert stack in ENTRY_STACKS, "a legal ABI entry-stack residue modulo 64"
            entry_stack = BitVecVal(stack, 64)
            entry_assumptions = []
        syms = x86.symbols(so)
        self.entry = syms[entry]
        code = x86.disassemble(so, self.entry, x86.text_end(so))
        img = x86.image(so)
        regions = dict(regions)
        regions["image"] = Region("image", max(img) + 1, False, lambda o: BitVecVal(img[o], 8), alignment=4096)
        regs = SYSV_ARGS
        first_stack = 8
        end = entry_stack + first_stack + 8 * max(0, len(args) - len(regs))
        regions["stack"] = Region("stack", end, True, None, alignment=64)
        self.m = m = x86.Machine(code, regions)
        m.lengths = {"counter", "len_stack"}
        m.assumptions = entry_assumptions
        m.red_zone = 128
        for r in x86.GPR:
            m.x[r] = BitVec(f"{r}_in", 64)
        for i in range(32):
            m.v[i] = [BitVec(f"v{i}_{l}_in", 32) for l in range(16)]
        for i in m.k:
            m.k[i] = BitVec(f"k{i}_in", 16)
        m.entry_stack = entry_stack
        m.entry_residue = residue if stack is None else None
        m.x["rsp"] = Ptr("stack", entry_stack)
        m.store(m.x["rsp"], Ptr("return", 0), 8)
        # Stack arguments above the return address.
        for i, a in enumerate(args):
            if i < len(regs):
                m.x[regs[i]] = a
            else:
                at = m.x["rsp"] + first_stack + 8 * (i - len(regs))
                m.store(at, a, 8)

    def run(self):
        m = x86.run(self.m, self.entry)
        check_abi(m)
        return m


def check_abi(m):
    if not (isinstance(m.x["rsp"], Ptr) and m.x["rsp"].region == "stack" and aarch64.holds(m, aarch64.as_bv(m.x["rsp"].offset) == m.entry_stack + 8)):
        raise Unproved(f"rsp not restored: {m.x['rsp']}")
    for r in ["rbx", "rbp", "r12", "r13", "r14", "r15"]:
        if isinstance(m.x[r], Ptr) or not same(m.x[r], BitVec(f"{r}_in", 64)):
            raise Unproved(f"callee-saved {r} changed")


def u8(name):
    return BitVec(name, 8)


def d_row_facts(m, counter, n, words):
    """The standard's d-row words (x86.py's `facts`): the counter's halves
    for counter + i, i below n, and the block length and flags `words`
    (32-bit terms of the 8-bit symbols)."""
    for i in range(n):
        c = counter + i if i else counter
        m.facts += (Extract(31, 0, c), Extract(63, 32, c))
    m.facts += tuple(words)
    m.fact_symbols |= {"counter"} | {s for w in words for s in aarch64.term_symbols(w)}


def prove_compress(so, isa, xof):
    """compress_in_place(cv, block, block_len, counter, flags): cv becomes
    the chaining value; compress_xof(cv, block, block_len, counter, flags,
    out): out the whole 64-byte output, cv left as it was."""
    cv, block = byte_symbols("cv", 32), byte_symbols("block", 64)
    counter, block_len, flags = BitVec("counter", 64), u8("block_len"), u8("flags")
    regions = {"cv": Region("cv", 32, not xof, lambda o: cv[o]), "block": Region("block", 64, False, lambda o: block[o])}
    if xof:
        regions["out"] = Region("out", 64, True, None)
    args = [Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, block_len), counter, ZeroExt(56, flags)] + ([Ptr("out", 0)] if xof else [])
    name = f"blake3_servil_compress_{'xof' if xof else 'in_place'}_{isa}"
    s = Setup(so, name, regions, args)
    d_row_facts(s.m, counter, 1, [ZeroExt(24, block_len), ZeroExt(24, flags)])
    m = s.run()
    want = spec.compress(spec.words(cv), spec.words(block), counter, ZeroExt(24, block_len), ZeroExt(24, flags))
    region, n = (m.regions["out"], 16) if xof else (m.regions["cv"], 8)
    if any(i not in region.bytes for i in range(4 * n)):
        raise Unproved("the output is not wholly written")
    got = spec.words([region.bytes[i] for i in range(4 * n)])
    for w in range(n):
        if not same(got[w], want[w]):
            raise Unproved(f"output word {w} differs from the specification")
    return m.steps


def prove_xof_many(so, n):
    """xof_many(cv, block, block_len, counter, flags, out, n) (AVX-512,
    Unix): block i of out the whole output of block at counter + i."""
    cv, block = byte_symbols("cv", 32), byte_symbols("block", 64)
    counter, block_len, flags = BitVec("counter", 64), u8("block_len"), u8("flags")
    regions = {"cv": Region("cv", 32, False, lambda o: cv[o]), "block": Region("block", 64, False, lambda o: block[o]),
               "out": Region("out", 64 * n, True, None)}
    args = [Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, block_len), counter, ZeroExt(56, flags), Ptr("out", 0), BitVecVal(n, 64)]
    s = Setup(so, "blake3_servil_xof_many_avx512", regions, args)
    d_row_facts(s.m, counter, n, [ZeroExt(24, block_len), ZeroExt(24, flags)])
    m = s.run()
    out = m.regions["out"]
    if set(out.bytes) != set(range(64 * n)):
        raise Unproved("the output is not exactly written")
    words = spec.words(cv)
    for i in range(n):
        want = spec.compress(words, spec.words(block), counter + i if i else counter, ZeroExt(24, block_len), ZeroExt(24, flags))
        got = spec.words([out.bytes[64 * i + j] for j in range(64)])
        for w in range(16):
            if not same(got[w], want[w]):
                raise Unproved(f"block {i}, output word {w} differs from the specification")
    return m.steps


def prove_hash_many(so, isa, n, blocks, increment):
    """hash_many(inputs, n, blocks, key, counter, increment, flags,
    flags_start, flags_end, out): input i hashed from key over `blocks`
    blocks at counter + i (increment) or counter, the first block with
    flags | flags_start, the last with flags | flags_end, every block 64
    bytes long; its chaining value to out[32 i..]."""
    inputs = [byte_symbols(f"in{i}_", 64 * blocks) for i in range(n)]
    key = byte_symbols("key", 32)
    regions = {f"input{i}": Region(f"input{i}", 64 * blocks, False, (lambda b: lambda o: b[o])(inputs[i])) for i in range(n)}
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    regions["out"] = Region("out", 32 * n, True, None)
    table = Region("table", 8 * n, False, None)
    for i in range(n):
        for j in range(8):
            table.bytes[8 * i + j] = (Ptr(f"input{i}", 0), j)
    regions["table"] = table
    counter = BitVec("counter", 64)
    flags, start, end = u8("flags"), u8("start"), u8("end")
    args = [Ptr("table", 0), BitVecVal(n, 64), BitVecVal(blocks, 64), Ptr("key", 0), counter,
            BitVecVal(1 if increment else 0, 64), ZeroExt(56, flags), ZeroExt(56, start), ZeroExt(56, end), Ptr("out", 0)]
    s = Setup(so, f"blake3_servil_hash_many_{isa}", regions, args)
    d_row_facts(s.m, counter, n if increment else 1, [ZeroExt(24, flags | start | end), ZeroExt(24, flags | start), ZeroExt(24, flags | end), ZeroExt(24, flags), BitVecVal(64, 32)])
    m = s.run()
    out = m.regions["out"]
    if set(out.bytes) != set(range(32 * n)):
        raise Unproved("the output is not exactly written")
    cvs = table_cvs(inputs, key, [counter + i if increment else counter for i in range(n)], flags, start, end,
                    BitVecVal(64, 8), blocks)
    for i in range(n):
        got = spec.words([out.bytes[32 * i + j] for j in range(32)])
        for w in range(8):
            if not same(got[w], cvs[i][w]):
                raise Unproved(f"input {i}, output word {w} differs from the specification")
    return m.steps


WIDE = {"k32": (32, True), "p32": (32, False), "p16": (16, False)}
WIDE_WORKERS = 3  # their memory: at most three wide cases at once (main)


def prove_wide(so, kernel, blocks):
    """blake3_servil_avx512_{k32,p32,p16}(inputs, blocks, key, counter,
    packed, out), the AArch64 kernel family's ABI (tools/gen_avx512_32.py):
    input i, at inputs[i], is `blocks` blocks hashed from key at counter + i
    (k32) or counter (p32, p16), the first block with flags | start, the
    last with flags | end and block length last_len; its chaining value to
    out[32 i..]. For every input byte, key, counter, flags and last_len."""
    n, increment = WIDE[kernel]
    inputs = [byte_symbols(f"in{i}_", 64 * blocks) for i in range(n)]
    key = byte_symbols("key", 32)
    regions = {f"input{i}": Region(f"input{i}", 64 * blocks, False, (lambda b: lambda o: b[o])(inputs[i])) for i in range(n)}
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    regions["out"] = Region("out", 32 * n, True, None)
    table = Region("table", 8 * n, False, None)
    for i in range(n):
        for j in range(8):
            table.bytes[8 * i + j] = (Ptr(f"input{i}", 0), j)
    regions["table"] = table
    counter = BitVec("counter", 64)
    flags, start, end, last_len = u8("flags"), u8("start"), u8("end"), u8("last_len")
    packed = ZeroExt(32, Concat(last_len, end, start, flags))
    args = [Ptr("table", 0), BitVecVal(blocks, 64), Ptr("key", 0), counter, packed, Ptr("out", 0)]
    s = Setup(so, f"blake3_servil_avx512_{kernel}", regions, args)
    d_row_facts(s.m, counter, n if increment else 1,
                [ZeroExt(24, flags | start | end), ZeroExt(24, flags | start), ZeroExt(24, flags | end), ZeroExt(24, flags),
                 ZeroExt(24, last_len), BitVecVal(64, 32)])
    m = s.run()
    out = m.regions["out"]
    if set(out.bytes) != set(range(32 * n)):
        raise Unproved("the output is not exactly written")
    cvs = table_cvs(inputs, key, [counter + i if increment else counter for i in range(n)], flags, start, end, last_len, blocks)
    for i in range(n):
        got = spec.words([out.bytes[32 * i + j] for j in range(32)])
        for w in range(8):
            if not same(got[w], cvs[i][w]):
                raise Unproved(f"input {i}, output word {w} differs from the specification")
    return m.steps


CHUNK_START, CHUNK_END, ROOT = 1, 2, 8


def prove_chunk_root(so, isa, whole):
    """blake3_{isa}_chunk_root(input, key, last, packed, out): the root
    output of one chunk at counter 0, `whole` whole blocks at input and the
    last block (64 readable bytes, its length last_len) at last; packed =
    flags | last_len << 8 | whole << 16; out its 32 bytes. For every key,
    message byte and last-block length, and mode flags with CHUNK_START
    clear (the kernel's contract: it sets the chunk flags itself), no
    alignment promised for any pointer."""
    key, blocks, last = byte_symbols("key", 32), byte_symbols("in", 64 * whole), byte_symbols("last", 64)
    last_len, flags = u8("last_len"), u8("flags") & 0xFE
    packed = ZeroExt(56, flags) | (ZeroExt(56, last_len) << 8) | BitVecVal(whole << 16, 64)
    regions = {"input": Region("input", 64 * whole, False, lambda o: blocks[o]), "key": Region("key", 32, False, lambda o: key[o]),
               "last": Region("last", 64, False, lambda o: last[o]), "out": Region("out", 32, True, None)}
    args = [Ptr("input", 0), Ptr("key", 0), Ptr("last", 0), packed, Ptr("out", 0)]
    s = Setup(so, f"blake3_servil_{isa}_chunk_root", regions, args)
    # The last block's length and the mode flags are public, as lengths are.
    s.m.lengths |= {"last_len", "flags"}
    f = ZeroExt(24, flags)
    d_row_facts(s.m, BitVecVal(0, 64), 1, [ZeroExt(24, last_len), BitVecVal(64, 32), f | (CHUNK_START | CHUNK_END | ROOT),
                                           f | CHUNK_START, f, f | (CHUNK_END | ROOT)])
    m = s.run()
    cv = spec.words(key)
    for b in range(whole):
        cv = spec.compress(cv, spec.words(blocks[64 * b:64 * (b + 1)]), BitVecVal(0, 64), BitVecVal(64, 32),
                           f | CHUNK_START if b == 0 else f)[:8]
    want = spec.compress(cv, spec.words(last), BitVecVal(0, 64), ZeroExt(24, last_len),
                         f | ((CHUNK_START if whole == 0 else 0) | CHUNK_END | ROOT))[:8]
    out = m.regions["out"]
    if set(out.bytes) != set(range(32)):
        raise Unproved("the output is not exactly written")
    got = spec.words([out.bytes[i] for i in range(32)])
    for w in range(8):
        if not same(got[w], want[w]):
            raise Unproved(f"output word {w} differs from the specification")


def prove_partial(so, kisa, last):
    """blake3_partial_{sse41,avx2}(params): for every lane j, from the
    lane's chaining value and counter, the 16 blocks at its pointer, block
    b with the lane's length and flags for b; H the lane's chaining value
    after block 15, SNAP after block `last` (tools/gen_x86.py's layout)."""
    sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))
    import gen_x86
    w = gen_x86.W_OF[kisa]
    off, v = gen_x86.layout(w), 4 * w
    inputs = [byte_symbols(f"in{j}_", 1024) for j in range(w)]
    key = [[BitVec(f"key{i}_{j}", 32) for j in range(w)] for i in range(8)]
    ctr = [[BitVec(f"ctr{h}_{j}", 32) for j in range(w)] for h in range(2)]
    lens = [[BitVec(f"len{b}_{j}", 32) for j in range(w)] for b in range(16)]
    flags = [[BitVec(f"flags{b}_{j}", 32) for j in range(w)] for b in range(16)]
    init = {}

    def put(at, word):
        for k in range(4):
            init[at + k] = Extract(8 * k + 7, 8 * k, word)

    for i in range(8):
        for j in range(w):
            put(off["KEY"] + v * i + 4 * j, key[i][j])
    for i in range(4):
        for j in range(w):
            put(off["IV"] + v * i + 4 * j, BitVecVal((0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A)[i], 32))
    for name, r in (("ROT16", 2), ("ROT8", 1)):
        for j in range(w):
            base = 4 * (j % 4)
            put(off[name] + 4 * j, BitVecVal(sum((base + (k + r) % 4) << (8 * k) for k in range(4)), 32))
    for h in range(2):
        for j in range(w):
            put(off["CTR"] + v * h + 4 * j, ctr[h][j])
    for b in range(16):
        for j in range(w):
            put(off["LENS"] + v * b + 4 * j, lens[b][j])
            put(off["FLAGS"] + v * b + 4 * j, flags[b][j])
    put(off["LAST"], BitVecVal(64 * last, 32))
    params = Region("params", off["SIZE"], True, lambda o: init.get(o, BitVecVal(0, 8)))
    for j in range(w):
        for k in range(8):
            params.bytes[off["MSG"] + 8 * j + k] = (Ptr(f"input{j}", 0), k)
    regions = {f"input{j}": Region(f"input{j}", 1024, False, (lambda b: lambda o: b[o])(inputs[j])) for j in range(w)}
    regions["params"] = params
    s = Setup(so, f"blake3_servil_partial_{kisa}", regions, [Ptr("params", 0)])
    m = s.run()
    out = m.regions["params"]
    for j in range(w):
        cv = [key[i][j] for i in range(8)]
        counter = Concat(ctr[1][j], ctr[0][j])
        snap = None
        for b in range(16):
            cv = spec.compress(cv, spec.words(inputs[j][64 * b:64 * b + 64]), counter, lens[b][j], flags[b][j])[:8]
            if b == last:
                snap = cv
        for name, want in (("H", cv), ("SNAP", snap)):
            at = [off[name] + v * i + 4 * j for i in range(8)]
            if any(a + k not in out.bytes for a in at for k in range(4)):
                raise Unproved(f"lane {j}'s {name} is not wholly written")
            for i in range(8):
                got = spec.words([out.bytes[at[i] + k] for k in range(4)])[0]
                if not same(got, want[i]):
                    raise Unproved(f"lane {j}, {name} word {i} differs from the specification")
    return m.steps


def cases(only):
    out = []
    for isa in ("sse41", "avx512"):
        out += [(isa, "compress", None), (isa, "xof", None)]
    for isa in ("sse41", "avx2", "avx512"):
        for n in range(1, 2 * WIDTH[isa] + 1):
            for blocks in (1, 2, 16):
                out.append((isa, "hash_many", (n, blocks, True)))
        out.append((isa, "hash_many", (WIDTH[isa], 1, False)))
        # Shape (a): whole blocks at one counter, every block count.
        for blocks in range(2, 17):
            out.append((isa, "hash_many", (WIDTH[isa], blocks, False)))
    # pair.rs's shape: a whole chunk and a partial one side by side, two
    # inputs at consecutive counters, over the partial chunk's blocks
    # before its last (1 to 15), on AVX2 and AVX-512 (#163).
    for isa in ("avx2", "avx512"):
        for blocks in range(3, 16):
            out.append((isa, "hash_many", (2, blocks, True)))
    for n in range(1, 33):
        out.append(("avx512", "xof_many", n))
    out += [(isa, "chunk_root", whole) for isa in ("sse41", "avx512") for whole in range(16)]
    out += [("wide", "wide", (kernel, blocks)) for kernel in WIDE for blocks in range(1, 17)]
    for kisa in ("sse41", "avx2"):
        for last in range(16):
            out.append(("partial", "partial", (kisa, last)))
    return [c for c in out if not only or c[0] in only or c[1] in only]


def representative_cases():
    """The promotion gate's direct (non-induction) cases.

    The all-count proof covers hash_many for every count. These direct
    cases independently exercise the finite harness at every block length
    for one full-width fixed-counter pass, and at the first/full/two-pass
    counts for 16-block incrementing counters, for every ISA.
    Compression, XOF and every xof_many count have no all-count substitute
    and are all included.
    """
    out = []
    for isa in ("sse41", "avx512"):
        out += [(isa, "compress", None), (isa, "xof", None)]
    for isa in ("sse41", "avx2", "avx512"):
        width = WIDTH[isa]
        out += [(isa, "hash_many", (width, blocks, False)) for blocks in range(1, 17)]
        out += [(isa, "hash_many", (n, 16, True)) for n in (1, width, 2 * width)]
    for isa in ("avx2", "avx512"):
        out += [(isa, "hash_many", (2, blocks, True)) for blocks in range(1, 16)]
    out += [("avx512", "xof_many", n) for n in range(1, 33)]
    out += [(isa, "chunk_root", whole) for isa in ("sse41", "avx512") for whole in range(16)]
    out += [("wide", "wide", (kernel, blocks)) for kernel in WIDE for blocks in range(1, 17)]
    return out


OBJS = {}


def prove_case(case):
    isa, kind, arg = case
    so = assemble("chunk_root" if kind == "chunk_root" else isa)
    try:
        if kind == "chunk_root":
            steps, what = prove_chunk_root(so, isa, arg), f"chunk root, {arg} whole blocks and the last"
        elif kind == "partial":
            kisa, last = arg
            steps, what = prove_partial(so, kisa, last), f"{kisa}, snapshot after block {last}"
        elif kind == "wide":
            kernel, blocks = arg
            steps, what = prove_wide(so, kernel, blocks), f"{kernel}, {WIDE[kernel][0]} inputs of {blocks} blocks"
        elif kind == "xof_many":
            steps, what = prove_xof_many(so, arg), f"xof_many, {arg} blocks"
        elif kind in ("compress", "xof"):
            steps, what = prove_compress(so, isa, kind == "xof"), kind
        else:
            n, blocks, inc = arg
            steps = prove_hash_many(so, isa, n, blocks, inc)
            what = f"hash_many, {n} inputs of {blocks} blocks{'' if inc else ', one counter'}"
    except Unproved as e:
        return case, f"NOT PROVED: {e}"
    return case, f"proved ({what}, every entry-stack alignment, {steps} instructions)"


def main():
    import workers
    args = sys.argv[1:]
    count = 16
    if args[:1] == ["--workers"]:
        assert len(args) >= 2 and args[1].isdigit() and int(args[1]) >= 1, "--workers takes a count of at least 1"
        count, args = int(args[1]), args[2:]
    todo = representative_cases() if args == ["--representative"] else cases(set(args))
    for isa in {"chunk_root" if c[1] == "chunk_root" else c[0] for c in todo}:
        so = assemble(isa)
        x86.symbols(so)
        x86.image(so)
    # The largest cases first, so the pool's tail is short: a case's work,
    # by kind (hash_many's inputs times blocks; a partial pass's lanes
    # times 16 blocks; a chunk root's blocks; xof_many's blocks).
    def work(c):
        kind, arg = c[1], c[2]
        if kind == "hash_many":
            return arg[0] * arg[1]
        if kind == "partial":
            return WIDTH[arg[0]] * 16
        if kind == "wide":
            return WIDE[arg[0]][0] * arg[1] * 2  # 16-lane vectors twice over: the heaviest cases
        return (arg or 0) + 1
    todo.sort(key=lambda c: -work(c))
    # A 32-input wide case holds up to 22 GB (k32 at 16 blocks, the VM,
    # October 26): those run WIDE_WORKERS at a time, the rest `count`.
    failed = 0
    for batch, n in (([c for c in todo if c[1] == "wide"], min(count, WIDE_WORKERS)), ([c for c in todo if c[1] != "wide"], count)):
        for (isa, kind, arg), verdict in workers.each(prove_case, batch, n):
            print(f"{isa} {kind} {arg or ''}: {verdict}", flush=True)
            failed += verdict.startswith("NOT")
    print(f"{len(todo) - failed} of {len(todo)} cases proved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
