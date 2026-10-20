"""The AVX-512 xof_many kernel at one random point, against the
specification, for N blocks: the check that tells a model gap from a
kernel bug when prove_x86 says a word "differs" (its normal forms differ,
which proves nothing either way).

    pypy3 tools/verify/x86_xof_point.py N [SEED]

At 4 and 5 blocks (the half-filled mask path) prove_x86 cannot bring the
terms to one normal form; this agrees at every word, so the gap is the
model's (October 20, 2026).
"""
import sys, random
sys.path.insert(0, __import__('os').path.dirname(__file__))
import prove_x86 as P
from z3 import *
import lean_spec as spec
so = P.assemble("avx512", False)
n = int(sys.argv[1])
cv, block = P.byte_symbols("cv", 32), P.byte_symbols("block", 64)
counter, block_len, flags = BitVec("counter", 64), P.u8("block_len"), P.u8("flags")
regions = {"cv": P.Region("cv", 32, False, lambda o: cv[o]), "block": P.Region("block", 64, False, lambda o: block[o]), "out": P.Region("out", 64 * n, True, None)}
args = [P.Ptr("cv", 0), P.Ptr("block", 0), ZeroExt(56, block_len), counter, ZeroExt(56, flags), P.Ptr("out", 0), BitVecVal(n, 64)]
s = P.Setup(so, "blake3_xof_many_avx512", regions, args, False)
P.d_row_facts(s.m, counter, n, [ZeroExt(24, block_len), ZeroExt(24, flags)])
m = s.run()
out = m.regions["out"]
words = spec.words(cv)
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
sub = [(b, BitVecVal(random.randrange(256), 8)) for b in cv + block] + [(counter, BitVecVal(random.randrange(1<<40), 64)), (block_len, BitVecVal(64, 8)), (flags, BitVecVal(0, 8))]
for i in range(n):
    want = spec.compress(words, spec.words(block), counter + i if i else counter, ZeroExt(24, block_len), ZeroExt(24, flags))
    got = spec.words([out.bytes[64 * i + j] for j in range(64)])
    bad = [w for w in range(16) if simplify(substitute(got[w], *sub)).as_long() != simplify(substitute(want[w], *sub)).as_long()]
    print(f"block {i}: {'all 16 words agree' if not bad else 'DIFFER at words ' + str(bad)}")
