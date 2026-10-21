"""Proofs of the x86-64 kernels (c/blake3_{sse41,avx2,avx512}_x86-64_unix.S,
c/blake3_partial_x86-64_unix.S, and their MSVC twins *_windows_msvc.asm
under the Windows calling convention), as assembled, equal to the
compression function of the Lean specification (lean_spec.py) for every
input, key, counter, and flags value, by the x86-64 model (x86.py): one
compression (compress_in_place, compress_xof), hash_many at every input
count up to two passes of its width and every block count from 1 to 16,
and the generated pass of whole chunks with a partial one (each lane's
16 blocks, every per-lane, per-block length and flags, at every block
after which its snapshot is taken).

    python3 tools/verify/prove_x86.py [CASE...]
"""

import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, Concat, Extract, ZeroExt, simplify

import aarch64
import x86
from aarch64 import Ptr, Region, Unproved
from prove_hybrid import byte_symbols, same, table_cvs
import lean_spec as spec

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
CLANG = x86.tool("clang")
LLVM_ML = x86.tool("llvm-ml")
STACK = 1 << 16
WIDTH = {"sse41": 4, "avx2": 8, "avx512": 16}
SYSV_ARGS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
WIN_ARGS = ["rcx", "rdx", "r8", "r9"]


def assemble(isa, windows):
    """`isa`'s file (sse41, avx2, avx512, or partial: the generated pass)."""
    d = tempfile.mkdtemp()
    if windows:
        obj = os.path.join(d, f"{isa}.obj")
        src = os.path.join(ROOT_DIR, f"c/blake3_{isa}_x86-64_windows_msvc.asm")
        subprocess.run([LLVM_ML, "-m64", "-c", src, f"-Fo{obj}"], check=True, stdout=subprocess.DEVNULL)
        # llvm-ml writes COFF; lld links it as a Windows DLL image.
        so = os.path.join(d, f"{isa}.dll")
        subprocess.run([x86.tool("lld-link"), "/dll", "/noentry", "/force:unresolved", "/debug:symtab", f"/out:{so}", obj],
                       check=True, stdout=subprocess.DEVNULL)
        return so
    obj = os.path.join(d, f"{isa}.o")
    subprocess.run([CLANG, "-c", "--target=x86_64-linux-gnu", os.path.join(ROOT_DIR, f"c/blake3_{isa}_x86-64_unix.S"),
                    "-o", obj], check=True)
    return x86.link(obj)


class Setup:
    """A machine at a kernel's entry: junk in every register, the return
    address on the stack, the arguments placed by the convention."""
    def __init__(self, so, entry, regions, args, windows):
        syms = x86.symbols(so)
        self.entry = syms[entry]
        code = x86.disassemble(so, self.entry, x86.text_end(so))
        img = x86.image(so)
        regions = dict(regions)
        regions["image"] = Region("image", max(img) + 1, False, lambda o: BitVecVal(img[o], 8))
        regions["stack"] = Region("stack", STACK + 8 * 16, True, None)
        self.m = m = x86.Machine(code, regions)
        m.red_zone = 0 if windows else 128
        self.windows = windows
        for r in x86.GPR:
            m.x[r] = BitVec(f"{r}_in", 64)
        for i in range(32):
            m.v[i] = [BitVec(f"v{i}_{l}_in", 32) for l in range(16)]
        m.x["rsp"] = Ptr("stack", STACK)
        m.store(m.x["rsp"], Ptr("return", 0), 8)
        regs = WIN_ARGS if windows else SYSV_ARGS
        # Stack arguments above the return address (and Windows's 32-byte
        # shadow space).
        first_stack = 8 + (32 if windows else 0)
        for i, a in enumerate(args):
            if i < len(regs):
                m.x[regs[i]] = a
            else:
                at = m.x["rsp"] + first_stack + 8 * (i - len(regs))
                m.store(at, a, 8)
        m.stack_args_end = STACK + first_stack + 8 * max(0, len(args) - len(regs))

    def run(self):
        m = x86.run(self.m, self.entry)
        check_abi(m, self.windows)
        return m


def check_abi(m, windows):
    if not (isinstance(m.x["rsp"], Ptr) and m.x["rsp"].region == "stack" and aarch64.concrete(aarch64.as_bv(m.x["rsp"].offset)) == STACK + 8):
        raise Unproved(f"rsp not restored: {m.x['rsp']}")
    saved = ["rbx", "rbp", "r12", "r13", "r14", "r15"] + (["rdi", "rsi"] if windows else [])
    for r in saved:
        if isinstance(m.x[r], Ptr) or not same(m.x[r], BitVec(f"{r}_in", 64)):
            raise Unproved(f"callee-saved {r} changed")
    if windows:
        for i in range(6, 16):
            if not all(same(m.v[i][l], BitVec(f"v{i}_{l}_in", 32)) for l in range(4)):
                raise Unproved(f"callee-saved xmm{i} changed")


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


def prove_compress(so, isa, windows, xof):
    """compress_in_place(cv, block, block_len, counter, flags): cv becomes
    the chaining value; compress_xof(cv, block, block_len, counter, flags,
    out): out the whole 64-byte output, cv left as it was."""
    cv, block = byte_symbols("cv", 32), byte_symbols("block", 64)
    counter, block_len, flags = BitVec("counter", 64), u8("block_len"), u8("flags")
    regions = {"cv": Region("cv", 32, not xof, lambda o: cv[o]), "block": Region("block", 64, False, lambda o: block[o])}
    if xof:
        regions["out"] = Region("out", 64, True, None)
    args = [Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, block_len), counter, ZeroExt(56, flags)] + ([Ptr("out", 0)] if xof else [])
    name = f"blake3_compress_{'xof' if xof else 'in_place'}_{isa}"
    s = Setup(so, name, regions, args, windows)
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
    s = Setup(so, "blake3_xof_many_avx512", regions, args, False)
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


def prove_hash_many(so, isa, windows, n, blocks, increment):
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
    s = Setup(so, f"blake3_hash_many_{isa}", regions, args, windows)
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


def prove_partial(so, kisa, windows, last):
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
    s = Setup(so, f"blake3_partial_{kisa}", regions, [Ptr("params", 0)], windows)
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
    for windows in (False, True):
        for isa in ("sse41", "avx512"):
            out += [(isa, windows, "compress", None), (isa, windows, "xof", None)]
        for isa in ("sse41", "avx2", "avx512"):
            for n in range(1, 2 * WIDTH[isa] + 1):
                for blocks in (1, 2, 16):
                    out.append((isa, windows, "hash_many", (n, blocks, True)))
            out.append((isa, windows, "hash_many", (WIDTH[isa], 1, False)))
    for n in range(1, 33):
        out.append(("avx512", False, "xof_many", n))
    for windows in (False, True):
        for kisa in ("sse41", "avx2"):
            for last in range(16):
                out.append(("partial", windows, "partial", (kisa, last)))
    return [c for c in out if not only or c[0] in only or c[2] in only or ("windows" if c[1] else "unix") in only]


OBJS = {}


def prove_case(case):
    isa, windows, kind, arg = case
    if (isa, windows) not in OBJS:
        OBJS[(isa, windows)] = assemble(isa, windows)
    so = OBJS[(isa, windows)]
    t = time.time()
    try:
        if kind == "partial":
            kisa, last = arg
            steps, what = prove_partial(so, kisa, windows, last), f"{kisa}, snapshot after block {last}"
        elif kind == "xof_many":
            steps, what = prove_xof_many(so, arg), f"xof_many, {arg} blocks"
        elif kind in ("compress", "xof"):
            steps = prove_compress(so, isa, windows, kind == "xof")
            what = kind
        else:
            n, blocks, inc = arg
            steps = prove_hash_many(so, isa, windows, n, blocks, inc)
            what = f"hash_many, {n} inputs of {blocks} blocks{'' if inc else ', one counter'}"
    except Unproved as e:
        return case, f"NOT PROVED: {e}"
    return case, f"proved ({what}, {steps} instructions, {time.time() - t:.1f} s)"


def main():
    import workers
    todo = cases(set(sys.argv[1:]))
    failed = 0
    for (isa, windows, kind, arg), verdict in workers.each(prove_case, todo, 16):
        print(f"{isa} {'windows' if windows else 'unix'} {kind} {arg or ''}: {verdict}", flush=True)
        failed += verdict.startswith("NOT")
    print(f"{len(todo) - failed} of {len(todo)} cases proved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
