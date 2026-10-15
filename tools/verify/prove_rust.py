"""Prove blake3-servil's Rust compression paths equal to the Lean
specification's compression (tools/verify/README.md): each path as the
compiler built it for AArch64, the Rust and any assembly it calls run
together by the symbolic executor.

    pypy3 tools/verify/prove_rust.py

The paths (tools/verify/rust/src/lib.rs, each a call of the library's own
`Platform` method): the portable compression in place and its extended
output, the NEON and SME2 platforms' compression in place (Rust around the
scalar kernel), every platform's compression of 1 to 16 whole blocks from
a slice of any length that holds them (Rust around the integer kernel, or
the portable loop), the portable and NEON platforms' hash_many of whole chunks
and of parents at every count to their degree (Rust around the hybrid
kernels' plans), and the NEON platform's extended output of 1 to 20
blocks (Rust NEON intrinsics, four blocks at a time, and the portable code
for the rest). The library is built as users build it (std). Each runs on symbolic chaining values, blocks, lengths,
counters, and flags.
"""

import bisect
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecSort, BitVecVal, Concat, Extract, Function, ULT, UGE, ZeroExt, simplify
import aarch64
import induction
import results
from aarch64 import Machine, Region, Ptr, Unproved
import lean_spec as spec
from prove_hybrid import same, byte_symbols, check_abi

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "/tmp/target/verify-rust"
STACK = 1 << 20


def build():
    # In rust/, whose rust-toolchain.toml names the compiler.
    subprocess.run(["cargo", "build", "-q", "--release"], cwd=os.path.join(HERE, "rust"),
                   env={**os.environ, "CARGO_TARGET_DIR": TARGET}, check=True)
    return os.path.join(TARGET, "release", "libverify_entries.so")


class Library:
    """The built library: its code, its image, and its PLT's entries."""
    def __init__(self, so):
        self.so = so
        self.syms = aarch64.symbols(so)
        self.code = aarch64.disassemble(so, 0, 1 << 30)
        # The SME2 assembly, which binutils does not decode: each function
        # holding an undecoded word, decoded again by LLVM's disassembler.
        starts = sorted(set(self.syms.values()))
        redo = set()
        for a, (mnem, _) in self.code.items():
            if mnem == ".inst":
                i = bisect.bisect_right(starts, a) - 1
                redo.add((starts[i], starts[i + 1] if i + 1 < len(starts) else a + 4))
        for lo, hi in redo:
            self.code.update(aarch64.disassemble(so, lo, hi, llvm=True))
        self.image = aarch64.image_bytes(so)
        listing = subprocess.run(["objdump", "-d", so], capture_output=True, text=True, check=True).stdout
        self.plt = {int(a, 16): n for a, n in re.findall(r"^([0-9a-f]+) <(\w+)@plt>:", listing, re.M)}

    def symbol(self, suffix):
        """The one symbol whose name ends with `suffix`: a Rust name's
        mangling carries a hash of the crate, which varies by compiler."""
        found = [n for n in self.syms if n.endswith(suffix)]
        if len(found) != 1:
            raise Unproved(f"expected one symbol ending {suffix}, found {found}")
        return found[0]


def machine(lib, regions):
    image = lib.image
    regions = dict(regions)
    for name in ("image", "text"):
        regions[name] = Region(name, max(image) + 1, False, lambda o: BitVecVal(image.get(o, 0), 8))
    regions["stack"] = Region("stack", STACK, True, None)
    m = Machine(lib.code, regions)
    m.plt = lib.plt
    for i in range(31):
        m.x[i] = BitVec(f"x{i}_in", 64)
    for i in range(32):
        m.v[i] = [BitVec(f"v{i}_{l}_in", 32) for l in range(16)]
    m.x["sp"], m.x[30] = Ptr("stack", STACK), Ptr("return", 0)
    return m


def run(lib, m, entry):
    aarch64.run(m, lib.syms[entry])
    check_abi(m)


def inputs():
    cvb, block = byte_symbols("cv", 32), byte_symbols("block", 64)
    return cvb, block, BitVec("last_len", 8), BitVec("counter", 64), BitVec("flags", 8)


def want(cvb, block, length, counter, flags):
    return spec.compress(spec.words(cvb), spec.words(block), counter, ZeroExt(24, length), ZeroExt(24, flags))


def compare(region, start, words):
    if any(start + j not in region.bytes for j in range(4 * len(words))):
        raise Unproved("the output is not wholly written")
    got = spec.words([region.bytes[start + j] for j in range(4 * len(words))])
    for w, (g, x) in enumerate(zip(got, words)):
        if not same(g, x):
            raise Unproved(f"output word {start // 4 + w} differs from the specification")


def prove_in_place(lib, entry):
    cvb, block, length, counter, flags = inputs()
    m = machine(lib, {"cv": Region("cv", 32, True, lambda o: cvb[o]), "block": Region("block", 64, False, lambda o: block[o])})
    m.x[0], m.x[1], m.x[2], m.x[3], m.x[4] = Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, length), counter, ZeroExt(56, flags)
    run(lib, m, entry)
    compare(m.regions["cv"], 0, want(cvb, block, length, counter, flags)[:8])
    return m.steps


def prove_blocks(lib, entry, count):
    """`count` whole blocks into the chaining value, from a slice of any
    length `len` of at least 64 * count bytes: every block at the counter,
    the first with flags | flags_start, the rest with flags."""
    cvb, counter, flags, start = byte_symbols("cv", 32), BitVec("counter", 64), BitVec("flags", 8), BitVec("start", 8)
    inp = byte_symbols("in", 64 * count)
    length = BitVec("len", 64)
    m = machine(lib, {"cv": Region("cv", 32, True, lambda o: cvb[o]),
                      "blocks": Region("blocks", length, False, lambda o: inp[o])})
    m.lengths = {"len"}
    m.assumptions = [UGE(length, 64 * count), ULT(length, 1 << 62)]
    m.x[0], m.x[1], m.x[2], m.x[3] = Ptr("cv", 0), Ptr("blocks", 0), length, BitVecVal(count, 64)
    m.x[4], m.x[5], m.x[6] = counter, ZeroExt(56, flags), ZeroExt(56, start)
    run(lib, m, entry)
    cv = spec.words(cvb)
    for i in range(count):
        f = ZeroExt(24, flags | (start if i == 0 else BitVecVal(0, 8)))
        cv = spec.compress(cv, spec.words(inp[64 * i:64 * i + 64]), counter, BitVecVal(64, 32), f)[:8]
    compare(m.regions["cv"], 0, cv)
    return m.steps


def read_bytes(m, ptr, n, reads):
    """`n` bytes from `ptr` through the machine (each read checked), noted
    in `reads` for the overlap check."""
    reads.append((ptr, n))
    return [m.load(ptr + j, 1) for j in range(n)]


def write_cvs(m, out, cvs, reads):
    """The chaining values to `out`, after checking that they overlap
    nothing the call read (the kernels' proofs keep every buffer apart)."""
    n = 32 * len(cvs)
    for ptr, k in reads:
        if ptr.region == out.region:
            a, b = aarch64.concrete(aarch64.as_bv(ptr.offset)), aarch64.concrete(aarch64.as_bv(out.offset))
            if a is None or b is None or not (a + k <= b or b + n <= a):
                raise Unproved(f"a kernel's output at {out} may overlap its input at {ptr}")
    for i, cv in enumerate(cvs):
        for w in range(8):
            m.store(out + 32 * i + 4 * w, cv[w], 4)


def proved_code(lib, obj, name, end):
    """Stop unless the library's bytes of kernel `name` (to the symbol
    `end`, or to the end of the object's text when `end` is None) are the
    bytes of the object its proof ran: a contract stands for the code its
    proof covered, and only for that."""
    syms, text = aarch64.symbols(obj), aarch64.text_bytes(obj)
    lo, hi = syms[name], (syms[end] if end else max(text) + 1)
    if [text[a] for a in range(lo, hi)] != [lib.image.get(lib.syms[name] + i) for i in range(hi - lo)]:
        raise Unproved(f"the library's {name} is not the code its proof covered")


# The kernels' results as uninterpreted functions of where their input lies
# (sme2_contracts' `opaque`): CH_j, word j of the chaining value of the chunk
# at input offset `offset`, counter, flags word, and key; PA_j, of the
# parent of 16 child words under key, counter, and flags. Each kernel's
# statement makes them the specification's compressions of the bytes there.
CH = [Function(f"CH{j}", BitVecSort(64), BitVecSort(64), BitVecSort(32), *[BitVecSort(32)] * 8, BitVecSort(32)) for j in range(8)]
PA = [Function(f"PA{j}", *[BitVecSort(32)] * 24, BitVecSort(64), BitVecSort(32), BitVecSort(32)) for j in range(8)]


def sme2_contracts(lib, opaque=False):
    """The SME2 kernels the library calls, by what prove_sme2.py's
    always-run cases prove of them (chunk_cvs and parent_cvs, the one
    statement each), at the group counts those cases cover
    (prove_sme2.GROUPS; prove_sme2.HYBRID_GROUPS for the kernel with an
    integer lane): {address: contract}. With `opaque`, each result is its
    CH or PA term instead: the chunks' bytes are never read, only where
    they lie, in a region nothing writes (prove_flat.py). Each argument is
    taken in simplest form, so it reads as the caller wrote it."""
    import prove_sme2

    def groups_of(m, proved, what):
        g = aarch64.concrete(m.x[5])
        if g not in proved:
            raise Unproved(f"an SME2 {what} kernel called with {g if g is not None else simplify(m.x[5])} groups, outside prove_sme2's cases")
        return g

    def chunk_kernel(lanes, proved, what):
        def run(m):
            g, word, reads = groups_of(m, proved, what), m.x[3], []
            if aarch64.concrete(Extract(31, 24, word)) != 0:
                raise Unproved("the chunk kernel storing part of its last group")
            key = [simplify(b) for b in read_bytes(m, m.x[1], 32, [])]
            ptrs = []
            for i in range(lanes * g):
                ptr = m.load(m.x[0] + 8 * i, 8)
                if not isinstance(ptr, Ptr):
                    raise Unproved("a chunk table entry that is not a pointer")
                ptrs.append(ptr)
            counters = [simplify(m.x[2] + i) for i in range(lanes * g)]
            if opaque:
                words, flags = spec.words(key), simplify(Extract(31, 0, word))
                cvs = []
                for ptr, c in zip(ptrs, counters):
                    if m.regions[ptr.region].writable:
                        raise Unproved("an opaque chunk in a region the call may write")
                    m.load(ptr, 1), m.load(ptr + 1023, 1)        # the chunk lies inside its region
                    reads.append((ptr, 1024))
                    cvs.append([CH[j](simplify(aarch64.as_bv(ptr.offset)), c, flags, *words) for j in range(8)])
            else:
                data = [read_bytes(m, ptr, 1024, reads) for ptr in ptrs]
                cvs = prove_sme2.chunk_cvs(data, key, counters, *(simplify(Extract(h, h - 7, word)) for h in (7, 15, 23)))
            write_cvs(m, m.x[4], cvs, reads)
            m.x[0] = BitVecVal(16, 64)
        return run

    def parents(m):
        g, reads = groups_of(m, prove_sme2.GROUPS, "parent"), []
        pairs = read_bytes(m, m.x[0], 64 * 16 * g, reads)
        key = [simplify(b) for b in read_bytes(m, m.x[1], 32, [])]
        if opaque:
            words, kw = [simplify(w) for w in spec.words(pairs)], [simplify(w) for w in spec.words(key)]
            c, f = simplify(m.x[2]), simplify(Extract(31, 0, m.x[3]))
            cvs = [[PA[j](*words[16 * i:16 * i + 16], *kw, c, f) for j in range(8)] for i in range(16 * g)]
        else:
            cvs = prove_sme2.parent_cvs(pairs, key, simplify(m.x[2]), simplify(Extract(31, 0, m.x[3])), 16 * g)
        write_cvs(m, m.x[4], cvs, reads)
        m.x[0] = BitVecVal(16, 64)

    if not getattr(lib, "sme2_checked", False):
        obj = prove_sme2.assemble()
        for name in ("blake3_sme2_hash16_chunks_512", "blake3_sme2_hash16_parents_512"):
            proved_code(lib, obj, name, "blake3_sme2_text_end")
        proved_code(lib, prove_sme2.assemble_hybrid(), "blake3_sme2x2_hash_chunks_512", None)
        lib.sme2_checked = True
    return {lib.syms["blake3_sme2_hash16_chunks_512"]: chunk_kernel(16, prove_sme2.GROUPS, "chunk"),
            lib.syms["blake3_sme2_hash16_parents_512"]: parents,
            lib.syms["blake3_sme2x2_hash_chunks_512"]: chunk_kernel(prove_sme2.HYBRID_GROUP, prove_sme2.HYBRID_GROUPS, "hybrid chunk")}


def hybrid_contracts(lib):
    """The NEON hybrid kernels hash_many's plans call (c/blake3_neon_hybrid_aarch64.S:
    k2 to k10, 1 to 16 blocks; the p kernels, one block), by what
    prove_hybrid.py proves of each (its table_cvs, the one statement):
    {address: contract}. A call outside what the proofs cover stops the run."""
    import prove_hybrid
    proved = {f"k{n}": range(1, 17) for n in range(2, 11)}
    proved.update({f"p{n}": (1,) for n in (2, 3, 4, 5, 7, 8, 9)})

    def contract(name):
        n, per_input = int(name[1:]), name[0] == "k"

        def run(m):
            blocks, packed, reads = aarch64.concrete(m.x[1]), m.x[4], []
            if blocks not in proved[name]:
                raise Unproved(f"{name} called with {m.x[1]} blocks, outside its proof")
            if aarch64.concrete(Extract(63, 32, packed)) != 0:
                raise Unproved(f"{name} called with flags above bit 31")
            key = [simplify(b) for b in read_bytes(m, m.x[2], 32, [])]
            data = []
            for i in range(n):
                ptr = m.load(m.x[0] + 8 * i, 8)
                if not isinstance(ptr, Ptr):
                    raise Unproved(f"{name}'s table entry {i} is not a pointer")
                data.append(read_bytes(m, ptr, 64 * blocks, reads))
            f = [simplify(Extract(h, h - 7, packed)) for h in (7, 15, 23, 31)]
            counters = [m.x[3] + i if per_input else m.x[3] for i in range(n)]
            write_cvs(m, m.x[5], prove_hybrid.table_cvs(data, key, counters, *f, blocks), reads)
        return run

    if not getattr(lib, "hybrid_checked", False):
        obj = prove_hybrid.assemble()
        for name in proved:
            proved_code(lib, obj, f"blake3_hybrid_{name}", f"blake3_hybrid_{name}_end")
        lib.hybrid_checked = True
    return {lib.syms[f"blake3_hybrid_{name}"]: contract(name) for name in proved}


def prove_hash_many(lib, entry, n, size, increment):
    """`n` inputs of `size` bytes, each at the pointer table's entry, each
    its own chaining value from the key: counter + i (with `increment`)
    or the counter, the first block with flags | start, the last with
    flags | end; the values to out in order. Parents (64 bytes) lie one
    after another in one buffer, as every caller in the library lays them
    out. The SME2 kernels run by their proved statements (sme2_contracts)."""
    inputs = [byte_symbols(f"in{i}_", size) for i in range(n)]
    key = byte_symbols("key", 32)
    if size == 64:
        flat = [b for x in inputs for b in x]
        regions = {"inputs": Region("inputs", size * n, False, lambda o: flat[o])}
        at = [Ptr("inputs", size * i) for i in range(n)]
    else:
        regions = {f"input{i}": Region(f"input{i}", size, False, (lambda b: lambda o: b[o])(inputs[i])) for i in range(n)}
        at = [Ptr(f"input{i}", 0) for i in range(n)]
    regions["key"] = Region("key", 32, False, lambda o: key[o])
    regions["out"] = Region("out", 32 * n, True, None)
    table = Region("table", 8 * n, False, None)
    for i in range(n):
        for j in range(8):
            table.bytes[8 * i + j] = (at[i], j)
    regions["table"] = table
    m = machine(lib, regions)
    m.contracts = {**sme2_contracts(lib), **hybrid_contracts(lib)}
    import prove_timing
    m.local = lib.syms
    # The state after `initialize()` on a CPU without SME2; the entry names
    # its platform's kernels itself.
    image_end, initial = prove_timing.process_image(lib, prove_timing.PLATFORMS["NEON"][0])
    for name in ("image", "text"):
        m.regions[name] = Region(name, image_end, True, initial)
    counter = BitVec("counter", 64)
    flags, start, end = (BitVec(x, 8) for x in ("flags", "start", "end"))
    m.x[0], m.x[1], m.x[2], m.x[3] = Ptr("table", 0), BitVecVal(n, 64), Ptr("key", 0), counter
    m.x[4], m.x[5], m.x[6], m.x[7] = ZeroExt(56, flags), ZeroExt(56, start), ZeroExt(56, end), Ptr("out", 0)
    run(lib, m, entry)
    import prove_hybrid
    cvs = prove_hybrid.table_cvs(inputs, key, [counter + i if increment else counter for i in range(n)],
                                 flags, start, end, BitVecVal(64, 8), size // 64)
    for i, cv in enumerate(cvs):
        compare(m.regions["out"], 32 * i, cv)
    return m.steps


def prove_xof(lib, entry, blocks):
    cvb, block, length, counter, flags = inputs()
    regions = {"cv": Region("cv", 32, False, lambda o: cvb[o]), "block": Region("block", 64, False, lambda o: block[o]),
               "out": Region("out", 64 * max(blocks, 1), True, None)}
    m = machine(lib, regions)
    m.x[0], m.x[1], m.x[2], m.x[3], m.x[4], m.x[5] = Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, length), counter, ZeroExt(56, flags), Ptr("out", 0)
    if blocks:
        m.x[6] = BitVecVal(blocks, 64)
    run(lib, m, entry)
    for i in range(max(blocks, 1)):
        compare(m.regions["out"], 64 * i, want(cvb, block, length, counter + i, flags))
    return m.steps


def loop_head(lib, entry, function):
    """The head of `function`'s loop: the target of its one backward jump that
    a run of 24 blocks takes twice."""
    import collections
    start = lib.syms[function]
    later = [a for a in lib.syms.values() if a > start]
    end = min(later) if later else start + (1 << 20)
    jumps = collections.Counter()
    step = aarch64.step
    last = [None]
    def trace(m, pc, mnem, ops):
        if last[0] is not None and pc < last[0] and start <= pc < end:
            jumps[pc] += 1
        last[0] = pc
        return step(m, pc, mnem, ops)
    aarch64.step = trace
    try:
        prove_xof(lib, entry, 24)
    finally:
        aarch64.step = step
    heads = [pc for pc, n in jumps.items() if n == 2]
    if len(heads) != 1:
        raise Unproved(f"{function}: expected one loop taken twice, found {sorted(map(hex, heads))}")
    return heads[0]


def prove_xof_every(lib, entry, function, per_iteration, from_count):
    """The extended output at every count from `from_count` blocks, by
    induction over the loop of `function`, which writes `per_iteration`
    blocks an iteration (induction.py). Counts below `from_count` are the
    concrete cases."""
    import induction
    from z3 import ULE, UGE, BitVecVal
    head = loop_head(lib, entry, function)
    cvb, block, length, counter, flags = inputs()
    n, k = BitVec("len_n", 64), BitVec("len_k", 64)
    regions = {"cv": Region("cv", 32, False, lambda o: cvb[o]), "block": Region("block", 64, False, lambda o: block[o]),
               "out": Region("out", n * 64, True, None)}
    m = machine(lib, regions)
    m.lengths = {"len_n", "len_k"}
    m.assumptions = [UGE(n, from_count), ULE(n, 1 << 57)]
    m.x[0], m.x[1], m.x[2], m.x[3], m.x[4], m.x[5], m.x[6] = (Ptr("cv", 0), Ptr("block", 0), ZeroExt(56, length), counter,
                                                              ZeroExt(56, flags), Ptr("out", 0), n)
    s0 = induction.run_to(m, lib.syms[entry], head)
    if s0.regions["out"].bytes:
        raise Unproved("output written before the loop")
    s1 = induction.run_to(induction.snapshot(s0), head, head)
    check_blocks(s1, 0, per_iteration, cvb, block, length, counter, flags, BitVecVal(0, 64))
    g = induction.Generalized(s0, s1, k)
    def check(end, how):
        if how == "head":
            check_blocks(end, 0, per_iteration, cvb, block, length, counter, flags, k * per_iteration, exact=True)
            return
        rest = next((r for r in range(per_iteration) if aarch64.holds(end, n == k * per_iteration + per_iteration + r)), None)
        if rest is None:
            raise Unproved("a path leaves the loop with a count the proof does not determine")
        check_blocks(end, 0, per_iteration + rest, cvb, block, length, counter, flags, k * per_iteration, exact=True)
        check_abi(end)
    kinds = loop_step(g, s0, k, list(s0.assumptions) + [ULE(k * per_iteration + per_iteration, n), ULE(k, 1 << 54)], head, check)
    return f"{kinds['head']} paths back to the loop's head, {kinds['return']} out of it"


def check_blocks(m, first, count, cvb, block, length, counter, flags, base_block, exact=False):
    """Output blocks base_block + first .. + count - 1 are the specification's
    at counter + their index; with `exact`, the only bytes written."""
    out = m.regions["out"]
    want_keys = set()
    for i in range(first, first + count):
        idx = base_block + i
        words = want(cvb, block, length, counter + idx, flags)
        offs = [aarch64.key(aarch64.as_bv(idx * 64 if not isinstance(idx, int) else BitVecVal(idx * 64, 64)) + j) for j in range(64)]
        want_keys.update(offs)
        if any(o not in out.bytes for o in offs):
            raise Unproved(f"output block {i} of the iteration is not wholly written")
        got = spec.words([out.bytes[o] for o in offs])
        for w, (gw, x) in enumerate(zip(got, words)):
            if not same(gw, x):
                raise Unproved(f"output block {i} of the iteration, word {w}, differs from the specification")
    if exact and set(out.bytes) != want_keys:
        raise Unproved("the iteration writes output bytes outside its blocks")


def mismatches(m, g, kval):
    """The positions where the machine differs from the generalized state at
    iteration `kval` (none: it is that state)."""
    inst = g.at(kval)
    def agree(a, b):
        if b is None:
            return True
        if isinstance(a, tuple) or isinstance(b, tuple):
            # A byte of a stored pointer.
            return (isinstance(a, tuple) and isinstance(b, tuple) and a[1] == b[1]
                    and agree(a[0], b[0]))
        if isinstance(b, Ptr) or isinstance(a, Ptr):
            return (isinstance(a, Ptr) and isinstance(b, Ptr) and a.region == b.region
                    and aarch64.holds(m, aarch64.as_bv(a.offset) == aarch64.as_bv(b.offset)))
        if same(a, b):
            return True
        return aarch64.lengths_only(m, a - b) and aarch64.holds(m, a == b)
    bad = []
    for r, gv in g.x.items():
        if not agree(m.x[r], inst(gv)):
            bad.append(("x", r))
    for r, lanes in g.v.items():
        for l, gv in enumerate(lanes):
            if not agree(m.v[r][l], inst(gv)):
                bad.append(("v", r, l))
    if g.za is not None:
        for t, tile in enumerate(g.za):
            for row, cells in enumerate(tile):
                for col, gv in enumerate(cells):
                    if not agree(m.za[t][row][col], inst(gv)):
                        bad.append(("za", t, row, col))
    if g.p is not None:
        for k, v in g.p.items():
            if v is not None and m.p.get(k) != v:
                bad.append(("p", k))
    for name, mem in g.mem.items():
        if name == "out":
            continue
        region = m.regions[name]
        for off, b in mem.items():
            if b is None or (isinstance(b, tuple) and b[0] == "word"):
                continue
            if off not in region.bytes or not agree(region.bytes[off], b):
                bad.append(("mem", name, off))
        for off in {o - b[2] for o, b in mem.items() if isinstance(b, tuple) and b[0] == "word"}:
            w = induction.word(region, off)
            if w is None or not agree(w, inst(mem[off][1])):
                bad.append(("word", name, off))
    return bad


def loop_step(g, start, kval, assumptions, head, check):
    """Every path of one iteration from `g` at `kval` (induction.py): `check`
    each path's end (`check(end, how)`); a path back at the head must be in the
    state for kval + 1. Positions where it differs become unknown in `g`, and
    the step is proved again: the invariant needs only what the iterations
    rely on (a weaker invariant still holds at the start). Returns the
    counts of paths."""
    import canon
    passes = 0
    while True:
        passes += 1
        mk = g.machine(start, kval)
        mk.assumptions = assumptions
        ends = induction.paths(mk, head, {head})
        forget = set()
        for end, how in ends:
            if how == "head":
                canon.set_context(end.assumptions, end.lengths)
                forget.update(mismatches(end, g, kval + 1))
        canon.set_context([], set())
        print(f"  induction pass {passes}: {len(ends)} paths, {len(forget)} positions left unknown {sorted(map(str, forget))[:8]}", flush=True)
        if not forget:
            break
        for pos in forget:
            g.forget(pos)
    # The invariant settled: each path's own obligations.
    kinds = {"head": 0, "return": 0}
    for end, how in ends:
        kinds[how] += 1
        canon.set_context(end.assumptions, end.lengths)
        check(end, how)
    canon.set_context([], set())
    return kinds


def prove_neon_xof_every(lib):
    return prove_xof_every(lib, "verify_neon_xof_many", lib.symbol("_13blake3_servil8neon_xof8xof_many"), 8, 16)


_LIB = [None]


def _start(so):
    _LIB[0] = Library(so)


def _run(case):
    """One case in a worker: its verdict line, whether it failed, and the
    instruction forms it ran (cross_check.py and isla_check.py check them)."""
    what, prove, args = case
    aarch64.EXECUTED.clear()
    t = time.time()
    try:
        result = prove(_LIB[0], *args)
        detail = f"{result} instructions" if isinstance(result, int) else result
        line, failed = f"{what}: proved ({detail}, {time.time() - t:.1f} s)", False
    except Unproved as e:
        line, failed = f"{what}: NOT PROVED: {e}", True
    return line, failed, sorted(aarch64.EXECUTED)


def run_cases(so, cases):
    """Each case, (what, function, arguments), its function called with the
    library first, in fresh worker processes; the verdicts printed as they
    finish; the forms they ran gathered into aarch64.EXECUTED. Returns the
    number that failed and the forms."""
    import multiprocessing
    workers = 6
    failed, ran = 0, set()
    spawn = multiprocessing.get_context("spawn")
    with spawn.Pool(workers, initializer=_start, initargs=(so,), maxtasksperchild=8) as pool:
        for line, bad, executed in pool.imap_unordered(_run, cases):
            print(line, flush=True)
            failed += bad
            ran.update(tuple(f) for f in executed)
    aarch64.EXECUTED.update(ran)
    print(f"{len(cases) - failed} of {len(cases)} cases proved")
    return failed, ran


def prove():
    """Every Rust case: the number that failed and the forms they ran. A
    passing run is recorded (results.py)."""
    cases = [("portable compression in place", prove_in_place, ("verify_portable_compress_in_place",)),
             ("portable extended output block", prove_xof, ("verify_portable_compress_xof", 0)),
             ("NEON platform's compression in place", prove_in_place, ("verify_neon_compress_in_place",)),
             ("SME2 platform's compression in place", prove_in_place, ("verify_sme2_compress_in_place",))]
    cases += [(f"{name} platform's compression of {n} blocks", prove_blocks, (f"verify_{name.lower()}_compress_blocks", n))
              for name in ("Portable", "NEON", "SME2") for n in range(1, 17)]
    # hash_many, whole chunks and parents, at every count to the platform's
    # degree (the portable platform's 1, NEON's 16, SME2's 128).
    cases += [(f"{name} platform's hash_many of {n} {what}", prove_hash_many,
               (f"verify_{name.lower()}_hash_{what}", n, size, inc))
              for name, degree in (("Portable", 1), ("NEON", 16), ("SME2", 128))
              for what, size, inc in (("chunks", 1024, True), ("parents", 64, False))
              for n in range(1, degree + 1)]
    cases += [(f"NEON platform's extended output, {n} blocks", prove_xof, ("verify_neon_xof_many", n)) for n in range(1, 21)]
    cases.append(("NEON platform's extended output, every count from 16 blocks (induction over its loop)",
                  prove_neon_xof_every, ()))
    so = build()
    failed, ran = run_cases(so, cases)
    if not failed:
        results.save_forms("rust", so, ran)
    return failed, ran


def main():
    failed, _ = prove()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
