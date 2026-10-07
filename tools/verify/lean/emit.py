"""Write the proofs' definition of BLAKE3's compression from the Lean specification.

    python3 tools/verify/lean/emit.py

1. Inserts the generated section of c2sp/BLAKE3/Blake3.lean (generate.py's
   translation of BLAKE3.md) into Generic.lean, verbatim, and checks it is
   the same text generate.py produces now.
2. Builds Generic.lean and Bridge.lean: Lean's kernel checks the generic copy
   at UInt32 is the specification's BLAKE3_COMPRESS.
3. Runs Emit.lean, which evaluates the generic copy over symbolic terms and
   writes compress.json: the compression's 16 output words as a graph of
   additions, xors, rotations, the counter's words, and constants over the
   inputs h0-h7, m0-m15, t, len, flags. tools/verify/lean_spec.py reads it.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "..", "..", "c2sp", "BLAKE3")
sys.path.insert(0, SPEC)
import generate


def main():
    md = open(os.path.join(SPEC, "..", "BLAKE3.md")).read()
    text = generate.generate(md)
    spec = open(os.path.join(SPEC, "Blake3.lean")).read()
    i, j = spec.index(generate.BEGIN), spec.index(generate.END) + len(generate.END) + 1
    if spec[i:j] != text:
        sys.exit("emit.py: Blake3.lean's generated section is not generate.py's output (run c2sp/BLAKE3/check.py)")
    path = os.path.join(HERE, "Generic.lean")
    new = generate.splice(open(path).read(), text)
    open(path, "w").write(new)
    for args in (["build"], ["env", "lean", "--run", "Emit.lean", "compress.json", "samples.json"]):
        r = subprocess.run(["lake", *args], cwd=HERE, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"emit.py: lake {' '.join(args)} failed:\n{r.stdout[-3000:]}{r.stderr[-2000:]}")
    # The graph, read as lean_spec.py reads it, at the specification's own sample outputs.
    sys.path.insert(0, os.path.dirname(HERE))
    import json
    import lean_spec
    from z3 import BitVecVal, simplify
    for row in json.load(open(os.path.join(HERE, "samples.json"))):
        ins = row["inputs"]
        w = lambda n: BitVecVal(ins[n], 32)
        got = lean_spec.compress([w(f"h{i}") for i in range(8)], [w(f"m{i}") for i in range(16)],
                                 BitVecVal(ins["t"], 64), w("len"), w("flags"))
        if [simplify(x).as_long() for x in got] != row["outputs"]:
            sys.exit("emit.py: compress.json, read by lean_spec.py, differs from the specification at a sample")
    os.remove(os.path.join(HERE, "samples.json"))
    print("emit.py: Generic.lean holds the specification's text; Bridge.lean and Sound.lean prove what"
          " Emit.lean writes is the specification's compression; compress.json written, and read by"
          " lean_spec.py it agrees with the specification at the samples")


if __name__ == "__main__":
    main()
