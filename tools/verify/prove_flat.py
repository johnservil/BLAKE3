"""Prove the SME2 flat walk (src/ffi_sme2.rs, the SME2 hook of the
library's tree walk and Hasher) returns the specification's subtree, as
compiled: at each of its whole subtrees (32 to 1024 chunks, a power of
two) and each number of values it keeps (128, the tree walk's; 2, the
root's children; 1, the Hasher's subtree), with nothing or the subtree's
own length to prefetch past it, for every input, key, counter, and flags.

The kernels it calls run by their statements, as uninterpreted functions
of where their input lies (prove_rust.sme2_contracts, `opaque`): the
chunk kernels' value for the chunk at input offset o is CH(o, counter,
flags, key), the parent kernel's PA(children, key, counter, flags). So the
proof is about which bytes reach which kernel and how the results combine:
the walk's output must equal the specification's tree over those terms,
level by level, until it keeps its number of values. prove_sme2.py proves
each kernel's statement (CH and PA are the specification's compressions
of the bytes there), so the walk returns the specification's values.

    pypy3 tools/verify/prove_flat.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, ZeroExt, simplify
import prove_rust
import prove_timing
import results
from aarch64 import Region, Ptr, Unproved
from prove_rust import CH, PA
import lean_spec as spec

ENTRIES = {128: "verify_sme2_flat", 2: "verify_sme2_flat_to_parent", 1: "verify_sme2_flat_to_cv"}


def expected(chunks, keep, key, counter, flags):
    """The values the walk keeps for `chunks` chunks: the specification's
    tree over CH and PA terms, each level of parents until `keep` remain."""
    cflags = simplify(ZeroExt(24, flags) | BitVecVal(1 << 8 | 2 << 16, 32))      # CHUNK_START, CHUNK_END
    pflags = simplify(ZeroExt(24, flags) | BitVecVal(4, 32))                    # PARENT
    level = [[CH[j](BitVecVal(1024 * i, 64), simplify(counter + i), cflags, *key) for j in range(8)] for i in range(chunks)]
    while len(level) > keep:
        level = [[PA[j](*level[2 * i], *level[2 * i + 1], *key, BitVecVal(0, 64), pflags) for j in range(8)]
                 for i in range(len(level) // 2)]
    return [w for cv in level for w in cv]


def prove_walk(lib, chunks, keep, ahead):
    n = 1024 * chunks
    m = prove_rust.machine(lib, {})
    m.local = lib.syms
    m.contracts = {**prove_rust.sme2_contracts(lib, opaque=True), **prove_rust.hybrid_contracts(lib)}
    m.regions["tls"] = prove_timing.tls_block(lib)
    end, initial = prove_timing.process_image(lib, prove_timing.PLATFORMS["SME2"][0])
    for name in ("image", "text"):
        m.regions[name] = Region(name, end, True, initial)
    keyb = [BitVec(f"key_{o}", 8) for o in range(32)]
    m.regions["input"] = Region("input", n, False, lambda o: BitVec(f"msg_{o}", 8))
    m.regions["key"] = Region("key", 32, False, lambda o: keyb[o])
    m.regions["out"] = Region("out", 128 * 32, True, None)
    m.regions["stack"] = Region("stack", prove_rust.STACK, True, lambda o: BitVec(f"stack_{o}", 8))
    m.clock = {"cntvct_el0": 10 ** 12, "cntfrq_el0": 24_000_000}
    counter, flags = BitVec("counter", 64), BitVec("flags", 8)
    m.x[0], m.x[1], m.x[2], m.x[3] = Ptr("input", 0), BitVecVal(n, 64), BitVecVal(ahead, 64), Ptr("key", 0)
    m.x[4], m.x[5], m.x[6] = counter, ZeroExt(56, flags), Ptr("out", 0)
    prove_rust.run(lib, m, ENTRIES[keep])
    want = expected(chunks, keep, [simplify(w) for w in spec.words(keyb)], counter, flags)
    out = m.regions["out"]
    if any(j not in out.bytes for j in range(4 * len(want))):
        raise Unproved("the output is not wholly written")
    for w, (g, x) in enumerate(zip(spec.words([out.bytes[j] for j in range(4 * len(want))]), want)):
        # Applications of uninterpreted functions over the inputs: equal
        # when they simplify to one term (unequal ones may stop the proof,
        # never pass it).
        if not simplify(g).eq(simplify(x)):
            raise Unproved(f"output word {w} differs from the specification's tree")
    return m.steps


def prove():
    """Every case: the number that failed and the forms they ran. A passing
    run is recorded (results.py)."""
    cases = [(f"flat walk, {chunks} chunks, keeping {keep}, {ahead} bytes ahead", prove_walk, (chunks, keep, ahead))
             for keep in (1, 2, 128) for chunks in (32, 64, 128, 256, 512, 1024) if keep < chunks
             for ahead in (0, 1024 * chunks)]
    so = prove_rust.build()
    failed, ran = prove_rust.run_cases(so, cases)
    if not failed:
        results.save_forms("flat", so, ran)
    return failed, ran


def main():
    failed, _ = prove()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
