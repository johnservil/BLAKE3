"""The x86 last-block kernels against the Lean BLAKE3 specification.

Each case hashes W items, each k whole blocks followed by a separate last
block, at counter zero. Last lengths and mode flags are public; keys and
all bytes remain arbitrary. Region bounds enforce no reads past each
whole-block region or its separate 64-byte last block. The result and
System V callee-saved registers are checked by the existing x86 executor.

    python3 tools/verify/prove_last.py [--workers N] [WIDTH:K ...]

Widths are 4 (SSE4.1), 8 (AVX2), 16 (AVX-512F); k is 0..15. The Rust
wrapper's padding and selection remain the Rust tests' responsibility.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVecVal, Extract, ZeroExt
from aarch64 import Ptr, Region, Unproved
from prove_hybrid import byte_symbols, same
from prove_x86 import Setup, assemble, d_row_facts, u8
import lean_spec as spec
import workers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gen_x86


def prove(case):
    w, k = case
    inputs = [byte_symbols(f"input{j}_", k * 64) for j in range(w)]
    lasts = [byte_symbols(f"last{j}_", 64) for j in range(w)]
    key = byte_symbols("key", 32)
    lengths = [u8(f"len_last{j}") for j in range(w)]
    flags = u8("flags") & 0xFE  # CHUNK_START clear: the kernel adds it.
    f = ZeroExt(24, flags)
    init = {}

    def put(at, word):
        for b in range(4):
            init[at+b] = Extract(8*b+7, 8*b, word)

    def pointer(at, region):
        for b in range(8):
            init[at+b] = (Ptr(region, 0), b)

    regions = {}
    for j in range(w):
        regions[f"input{j}"] = Region(f"input{j}", k*64, False, lambda o, j=j: inputs[j][o])
        regions[f"last{j}"] = Region(f"last{j}", 64, False, lambda o, j=j: lasts[j][o])

    if w == 16:
        for j in range(w):
            pointer(j*8, f"input{j}")
            pointer(128+j*8, f"last{j}")
            put(256+j*4, ZeroExt(24, lengths[j]))
        for b, value in enumerate(key):
            init[320+b] = value
        put(352, BitVecVal(k, 32))
        put(356, f)
        pointer(360, "out")
        size = 368
        regions["out"] = Region("out", 32*w, True, None)
        entry = "blake3_servil_last_avx512"
    else:
        off, v = gen_x86.layout(w, True), 4*w
        words = spec.words(key)
        for j in range(w):
            pointer(off["MSG"]+8*j, f"input{j}")
            pointer(off["LASTMSG"]+8*j, f"last{j}")
            for i in range(8):
                put(off["KEY"]+v*i+4*j, words[i])
            for i, value in enumerate((0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A)):
                put(off["IV"]+v*i+4*j, BitVecVal(value, 32))
            for name, rot in (("ROT16",2),("ROT8",1)):
                base = 4*(j%4)
                put(off[name]+4*j, BitVecVal(sum((base+(i+rot)%4) << (8*i) for i in range(4)),32))
            for b in range(16):
                put(off["LENS"]+v*b+4*j, ZeroExt(24,lengths[j]) if b==k else BitVecVal(64,32))
                bf = f | (1 if b==0 else 0) | (10 if b==k else 0)
                put(off["FLAGS"]+v*b+4*j, bf)
        put(off["END"],BitVecVal(64*(k+1),32))
        put(off["SWITCH"],BitVecVal(64*k,32))
        size = off["SIZE"]
        entry = f"blake3_servil_last_{'sse41' if w==4 else 'avx2'}"

    # Explicit initialization avoids inventing symbolic input for padding.
    params = Region("params", size, w!=16, None, alignment=64)
    params.bytes.update(init)
    if w != 16:
        # The unused fields can be read by the kernel only if initialized;
        # zero them as the Rust wrapper does. Pointer bytes above stay pairs.
        for b in range(size):
            params.bytes.setdefault(b, BitVecVal(0,8))
    regions["params"] = params
    s = Setup(assemble("last"), entry, regions, [Ptr("params",0)])
    s.m.lengths |= {"flags"} | {f"len_last{j}" for j in range(w)}
    d_row_facts(s.m,BitVecVal(0,64),1,
                [BitVecVal(64,32),f,f|1,f|10,f|11]+[ZeroExt(24,l) for l in lengths])
    m = s.run()
    if w == 16 and set(m.regions["out"].bytes) != set(range(32*w)):
        raise Unproved("output not exactly written")
    for j in range(w):
        cv = spec.words(key)
        for b in range(k):
            cv = spec.compress(cv,spec.words(inputs[j][64*b:64*(b+1)]),BitVecVal(0,64),BitVecVal(64,32),f|(1 if b==0 else 0))[:8]
        want = spec.compress(cv,spec.words(lasts[j]),BitVecVal(0,64),ZeroExt(24,lengths[j]),f|10|(1 if k==0 else 0))[:8]
        if w == 16:
            out = m.regions["out"].bytes
            got = spec.words([out[32*j+b] for b in range(32)])
        else:
            out = m.regions["params"].bytes
            got = [spec.words([out[off["H"]+v*i+4*j+b] for b in range(4)])[0] for i in range(8)]
        for i in range(8):
            if not same(got[i],want[i]):
                raise Unproved(f"{w} lanes, k={k}, lane {j}, word {i} differs")
    return f"proved {w}:{k}, {m.steps} instructions"


if __name__ == "__main__":
    args = sys.argv[1:]
    count = 4
    if args[:1] == ["--workers"]:
        count = int(args[1]); args = args[2:]
    cases = [tuple(map(int,a.split(":"))) for a in args] if args else [(w,k) for w in (4,8,16) for k in range(16)]
    assert count >= 1 and all(w in (4,8,16) and 0<=k<16 for w,k in cases)
    for result in workers.each(prove,cases,count):
        print(result,flush=True)
