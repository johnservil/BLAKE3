"""The x86 last-block kernels against the Lean BLAKE3 specification.

Each case hashes W items, each k whole blocks followed by a separate last
block, at counter zero. Last lengths and mode flags are public; keys and
all bytes remain arbitrary. Region bounds enforce no reads past each
whole-block region or its separate 64-byte last block. The result and
System V callee-saved registers are checked by the existing x86 executor.

    python3 tools/verify/prove_last.py [--workers N] [WIDTH:K ...]
    python3 tools/verify/prove_last.py --workers N --symbolic
    python3 tools/verify/prove_last.py --workers N --mixed

Widths are 8 (AVX2) and 16 (AVX-512F); K is 0..15. Symbolic mixed cases
WIDTH:K:-1 give each lane an independent public ending in 0..K, with K+1
steps in the kernel. For each output lane the proof checks each of its
possible ends, leaving every other lane's ending arbitrary. Thus the 32
symbolic runs cover all combinations, not just equal or rotated patterns.
Prefix regions end at that lane's symbolic ending; guarded pointer loads
must prove bounds for both feasible alternatives. Arbitrary old output
bytes ensure an omitted conditional store cannot count as a valid answer.
Keys, bytes, last lengths and mode flags are symbolic throughout. Last
lengths use arbitrary bytes (a superset of the wrapper's valid 0..64).

The older WIDTH:K:PHASE cases are retained as finite pattern controls:
lane j has (j+PHASE) mod (K+1) full blocks. --mixed runs those 384 cases.
Rust's actual builders are checked against the initialized ABI rows before
native test calls enter assembly; padding and dispatch also have independent
reference, occupancy and guard-page tests.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, Extract, ZeroExt, If, ULE, simplify, substitute
from aarch64 import Ptr, Region, Unproved
from prove_hybrid import byte_symbols, same
from prove_x86 import Setup, assemble, d_row_facts, u8
import lean_spec as spec
import workers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gen_x86


def prove(case):
    w, k = case[:2]
    mixed = len(case) == 3
    symbolic = mixed and case[2] == -1
    ks = ([k] * w if symbolic or not mixed else [(j + case[2]) % (k+1) for j in range(w)])
    ends = [BitVec(f"end{j}", 4) for j in range(w)] if symbolic else [BitVecVal(v, 4) for v in ks]
    inputs = [byte_symbols(f"input{j}_", ks[j] * 64) for j in range(w)]
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
        size = ZeroExt(60, ends[j]) * 64 if symbolic else ks[j] * 64
        regions[f"input{j}"] = Region(f"input{j}", size, False, lambda o, j=j: inputs[j][o])
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
        size = 448 if mixed else 368
        if mixed:
            for j in range(w):
                put(384+j*4, ZeroExt(28, ends[j]) * 64)
        # Arbitrary old output makes every omitted conditional store a real
        # difference, not invented zero or an unconstrained missing byte.
        previous = byte_symbols("previous_out", 32*w)
        regions["out"] = Region("out", 32*w, True, (lambda o: previous[o]) if symbolic else None)
        entry = "blake3_servil_mixed_avx512" if mixed else "blake3_servil_last_avx512"
    else:
        off, v = gen_x86.last8_layout(), 4*w
        for j in range(w):
            pointer(off['MSG']+8*j,f'input{j}')
            pointer(off['LASTMSG']+8*j,f'last{j}')
            put(off['LENS']+4*j,ZeroExt(24,lengths[j]))
            put(off['ENDS']+4*j,ZeroExt(28,ends[j])*64)
        for b,value in enumerate(key):init[off['KEY']+b]=value
        put(off['FLAGS'],f)
        put(off['END'],BitVecVal(64*(k+1),32))
        size=off['SIZE']
        entry='blake3_servil_mixed_avx2' if mixed else 'blake3_servil_last_avx2'

    # Explicit initialization avoids inventing symbolic input for padding.
    params = Region("params", size, w!=16, None, alignment=64)
    params.bytes.update(init)
    if w != 16:
        # Initialize the Rust fields, not the struct's padding: an accidental
        # read from alignment padding must stop rather than receive invented
        # zeros. The mixed kernel reads its old snapshot before updating it.
        for name, length in (("H",8*v),("SNAP",8*v)):
            for b in range(off[name],off[name]+length):
                params.bytes.setdefault(b,BitVecVal(0,8))
    regions["params"] = params
    s = Setup(assemble("last"), entry, regions, [Ptr("params",0)])
    s.m.lengths |= {"flags"} | {f"len_last{j}" for j in range(w)}
    if symbolic:
        s.m.lengths |= {f"end{j}" for j in range(w)}
        s.m.assumptions.extend(ULE(e, BitVecVal(k,4)) for e in ends)
    d_row_facts(s.m,BitVecVal(0,64),1,
                [BitVecVal(64,32),f,f|1,f|10,f|11]+[ZeroExt(24,l) for l in lengths])
    m = s.run()
    if w == 16 and set(m.regions["out"].bytes) != set(range(32*w)):
        raise Unproved("output not exactly written")
    for j in range(w):
        if w == 16:
            out = m.regions["out"].bytes
            got = ([spec.words([out[64*i+4*j+b] for b in range(4)])[0] for i in range(8)] if mixed
                   else spec.words([out[32*j+b] for b in range(32)]))
        else:
            out = m.regions["params"].bytes
            field = "SNAP" if mixed else "H"
            got = [spec.words([out[off[field]+v*i+4*j+b] for b in range(4)])[0] for i in range(8)]
        # Case-split only this lane's end on the specification side.
        # Other lanes' ends remain arbitrary in got: an accidental lane
        # dependency cannot pass by being fixed to a matching pattern.
        for end in range(k+1) if symbolic else [ks[j]]:
            cv = spec.words(key)
            for b in range(end):
                cv = spec.compress(cv,spec.words(inputs[j][64*b:64*(b+1)]),BitVecVal(0,64),BitVecVal(64,32),f|(1 if b==0 else 0))[:8]
            want = spec.compress(cv,spec.words(lasts[j]),BitVecVal(0,64),ZeroExt(24,lengths[j]),f|10|(1 if end==0 else 0))[:8]
            for i in range(8):
                actual = substitute(got[i], (ends[j],BitVecVal(end,4))) if symbolic else got[i]
                if not same(actual,want[i]):
                    raise Unproved(f"{w} lanes, k={k}, lane {j}, end {end}, word {i} differs")
    return f"proved {':'.join(map(str,case))}, {m.steps} instructions"


if __name__ == "__main__":
    args = sys.argv[1:]
    count = 4
    if args[:1] == ["--workers"]:
        count = int(args[1]); args = args[2:]
    cases = ([(w,k,-1) for w in (8,16) for k in range(16)] if args == ["--symbolic"] else
             [(w,k,p) for w in (8,16) for k in range(16) for p in range(w)] if args == ["--mixed"] else
             [tuple(map(int,a.split(":"))) for a in args] if args else [(w,k) for w in (8,16) for k in range(16)])
    assert count >= 1 and all(c[0] in (8,16) and 0<=c[1]<16 and (len(c)==2 or len(c)==3 and c[0] in (8,16) and -1<=c[2]<c[0]) for c in cases)
    for result in workers.each(prove,cases,count):
        print(result,flush=True)
