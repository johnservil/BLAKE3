"""Prove what a whole hash runs depends only on its input's length
(QUALITY.md, "Timing and secrets", the first condition), as compiled.

The compiled library's `hash` (through `rust/`'s `verify_hash`) runs on a
message of symbolic bytes, on each platform's kernels in turn. The run may
take no branch, select, address, or table index that depends on those
bytes: the executor stops at the first. So each length's run is one path,
the same for every message: the instructions and the addresses they touch
depend on the length alone. The process's state is the one a hash meets
after its first call: the self-test passed, SME2's detection cached.

    python3 tools/verify/prove_timing.py [LENGTH...]
"""

import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVecVal
import aarch64
from aarch64 import Region, Ptr, Unproved
from prove_hybrid import byte_symbols
import prove_rust

# Lengths: every length to two blocks, the chunk boundaries, and powers of
# two and their neighbours to 64 KiB.
LENGTHS = sorted(set(list(range(0, 130)) + [1023, 1024, 1025, 2047, 2048, 2049, 3071, 3072, 3073]
                     + [2 ** k + d for k in range(12, 17) for d in (-1, 0, 1)]))

# The platforms: SME2's cached detection (1 no, 2 yes), and the longest
# input each is proved to (the SME2 path's pointer table, built in vector
# lanes, is past what the executor models).
PLATFORMS = {"NEON": (1, 1 << 16), "SME2": (2, 4096)}


def process_image(lib, sme2_cache):
    """The library as loaded: its file bytes, its relocated pointers, and
    `.bss`, zero but the self-test's state (passed) and SME2's cache."""
    sections = subprocess.run(["readelf", "-SW", lib.so], capture_output=True, text=True, check=True).stdout
    bss = re.search(r"\.bss\s+NOBITS\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)", sections)
    end = int(bss.group(1), 16) + int(bss.group(2), 16)
    fixed = {lib.syms[lib.symbol("9self_test5STATE")]: 2, lib.syms[lib.symbol("13sme2_detected5CACHE")]: sme2_cache}
    pointers = {}
    for line in subprocess.run(["readelf", "-rW", lib.so], capture_output=True, text=True, check=True).stdout.splitlines():
        f = line.split()
        if len(f) >= 4 and f[2] == "R_AARCH64_RELATIVE":
            for i in range(8):
                pointers[int(f[0], 16) + i] = (Ptr("image", int(f[3], 16)), i)
    image = lib.image
    def initial(o):
        if o in pointers:
            return pointers[o]
        return BitVecVal(fixed.get(o, image.get(o, 0)), 8)
    return end, initial


def prove_length(lib, n, sme2_cache):
    message = byte_symbols("msg", n)
    regions = {"input": Region("input", n, False, lambda o: message[o]), "out": Region("out", 32, True, None)}
    m = prove_rust.machine(lib, regions)
    m.local = lib.syms
    end, initial = process_image(lib, sme2_cache)
    for name in ("image", "text"):
        # Writable: the statics in .data and .bss are.
        m.regions[name] = Region(name, end, True, initial)
    m.x[0], m.x[1], m.x[2] = Ptr("input", 0), BitVecVal(n, 64), Ptr("out", 0)
    prove_rust.run(lib, m, "verify_hash")
    return m.steps


def main(exit=True):
    lib = prove_rust.Library(prove_rust.build())
    only = [int(a) for a in sys.argv[1:]] if exit else []
    failed = total = 0
    for platform, (cache, longest) in PLATFORMS.items():
        for n in only or [n for n in LENGTHS if n <= longest]:
            total += 1
            t = time.time()
            try:
                steps = prove_length(lib, n, cache)
                print(f"{platform}, {n} bytes: one path, {steps} instructions ({time.time() - t:.1f} s)", flush=True)
            except Unproved as e:
                failed += 1
                print(f"{platform}, {n} bytes: NOT PROVED: {e}", flush=True)
    print(f"{total - failed} of {total} lengths proved")
    if exit:
        sys.exit(1 if failed else 0)
    return failed


if __name__ == "__main__":
    main()
