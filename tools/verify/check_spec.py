"""Check spec.py, the definition the proofs compare the kernels with,
against BLAKE3's official test vectors (test_vectors/test_vectors.json,
from the BLAKE3 team's repository): a whole BLAKE3 built on spec.compress
(chunks, the tree, keyed hashing, key derivation, extended output) must
reproduce every vector's hash, keyed hash, and derived key, all 131 bytes
of each.

    python3 tools/verify/check_spec.py

The tree and modes here follow the BLAKE3 paper (sections 2.1-2.6) and
are only this check's: the proofs use spec.compress alone.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVecVal, simplify
import spec

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEYED_HASH, DERIVE_KEY_CONTEXT, DERIVE_KEY_MATERIAL = 16, 32, 64


def compress(cv, block, counter, block_len, flags):
    """spec.compress on numbers: its terms built from constants, then
    evaluated by Z3."""
    out = spec.compress([BitVecVal(w, 32) for w in cv], [BitVecVal(w, 32) for w in block],
                        BitVecVal(counter, 64), BitVecVal(block_len, 32), BitVecVal(flags, 32))
    return [simplify(w).as_long() for w in out]


def words(data):
    data = data + bytes(64 - len(data))
    return [int.from_bytes(data[4 * i:4 * i + 4], "little") for i in range(16)]


def to_bytes(ws):
    return b"".join(w.to_bytes(4, "little") for w in ws)


class Output:
    def __init__(self, cv, block, counter, block_len, flags):
        self.cv, self.block, self.counter, self.block_len, self.flags = cv, block, counter, block_len, flags

    def chaining_value(self):
        return compress(self.cv, self.block, self.counter, self.block_len, self.flags)[:8]

    def root_bytes(self, n):
        out = b""
        i = 0
        while len(out) < n:
            out += to_bytes(compress(self.cv, self.block, i, self.block_len, self.flags | spec.ROOT))
            i += 1
        return out[:n]


def chunk_output(key, chunk, counter, flags):
    cv = key
    blocks = [chunk[i:i + 64] for i in range(0, len(chunk), 64)] or [b""]
    for i, b in enumerate(blocks):
        f = flags | (spec.CHUNK_START if i == 0 else 0)
        if i == len(blocks) - 1:
            return Output(cv, words(b), counter, len(b), f | spec.CHUNK_END)
        cv = compress(cv, words(b), counter, 64, f)[:8]


def parent_output(key, left, right, flags):
    return Output(key, left + right, 0, 64, flags | spec.PARENT)


def blake3(data, key, flags, n):
    """The paper's tree: chunk values pushed on a stack, merged while the
    count of chunks so far is even, the rest merged at the end."""
    chunks = [data[i:i + 1024] for i in range(0, len(data), 1024)] or [b""]
    stack = []
    for i, c in enumerate(chunks[:-1]):
        cv = chunk_output(key, c, i, flags).chaining_value()
        total = i + 1
        while total % 2 == 0:
            cv = parent_output(key, stack.pop(), cv, flags).chaining_value()
            total //= 2
        stack.append(cv)
    out = chunk_output(key, chunks[-1], len(chunks) - 1, flags)
    while stack:
        out = parent_output(key, stack.pop(), out.chaining_value(), flags)
    return out.root_bytes(n)


def main():
    vectors = json.load(open(os.path.join(ROOT_DIR, "test_vectors/test_vectors.json")))
    key = list(int.from_bytes(vectors["key"].encode()[4 * i:4 * i + 4], "little") for i in range(8))
    context = vectors["context_string"].encode()
    context_key = blake3(context, spec.IV, DERIVE_KEY_CONTEXT, 32)
    context_words = [int.from_bytes(context_key[4 * i:4 * i + 4], "little") for i in range(8)]
    bad = 0
    for case in vectors["cases"]:
        n = case["input_len"]
        data = bytes(i % 251 for i in range(n))
        for field, k, f in (("hash", spec.IV, 0), ("keyed_hash", key, KEYED_HASH), ("derive_key", context_words, DERIVE_KEY_MATERIAL)):
            want = bytes.fromhex(case[field])
            got = blake3(data, k, f, len(want))
            if got != want:
                print(f"MISMATCH: {field} of {n} bytes")
                bad += 1
        print(f"{n:6} bytes: hash, keyed_hash, derive_key {'agree' if not bad else ''}", flush=True)
    print(f"{len(vectors['cases'])} official test vectors, three modes, {'every one reproduced by spec.compress' if not bad else f'{bad} mismatches'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
