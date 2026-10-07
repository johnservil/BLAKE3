"""Check spec.py, the definition the proofs compare the kernels with,
against the C2SP specification's execution trace (the state after each
round of one compression) and BLAKE3's official test vectors (test_vectors/test_vectors.json,
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


# C2SP BLAKE3 v1.0.0, "Test Values", hash of a single block: "IETF", the
# state after each of the 7 rounds, and the output (words in hex).
C2SP_ROUNDS = """
d7737c52 a0d29b6a d3b4f608 e20caed2 49091c17 b1abb189 961f03ba c3474f4e a7590324 9c110e95 f77c59cc b47c3370 9c1aed89 b7c28f82 bab6db43 e634ca3e
4cce55f2 9cdfa58b 297f68b4 887fd036 4e620c26 321af343 b8e634b0 72737ae9 6f6ecf4a 628788fb df9428c1 a2c42d78 a51ddf7b 6cf97481 72dccb9c 1878acb8
8e99a713 bd202a18 d70c8d18 603ba3ad f411ae76 88ff9580 03db2909 a12e939f 19b81233 69787f12 d2b0c5b7 52034613 21baaea8 84e5fe6d c8c96ae8 422a96d8
eeb6ec2a 22f4289a 64900193 d9f751b3 216a610d f5aadf41 ddf5584d ae312167 c8f40fb3 97f06701 6eee4503 4827825d 3c59d243 473585da 90d24798 c5957f9d
11876617 4a71dc87 23a5b774 185e51fa a1ed35c0 729a3348 6da19311 9716237c f66bbb71 f303cf35 585dd137 e5c9c363 8b2b32ed 6add0d37 12b87a10 f96fde3e
02b010fc 345f4920 ce96e963 018a8afd c0e0faca 651d2baf 0b24a23d d1ffa8fc aa7de2ee d80796c0 ff96b6bd 7cfbf53a 292b8630 8d8e1a78 31c6cb9d b471de23
a4839e1a 064b478f bb47c942 3f4a0350 efd0bb79 61167ed0 356b01f5 b40f5364 ba5d3c99 adadb369 9fcea12a f08a4ddf 7ba07e35 9e94d896 e3dfca24 568e0272
"""
C2SP_OUTPUT = "1edea283 abe6f4e6 24896868 cfc04e8f 9470c54c ff82a646 d6b4cbd1 e2815116"


def check_trace():
    """spec.compress on C2SP's single-block example, round by round."""
    want = [[int(w, 16) for w in line.split()] for line in C2SP_ROUNDS.strip().splitlines()]
    got = []
    out = spec.compress([BitVecVal(w, 32) for w in spec.IV], [BitVecVal(w, 32) for w in words(b"IETF")],
                        BitVecVal(0, 64), BitVecVal(4, 32), BitVecVal(spec.CHUNK_START | spec.CHUNK_END | spec.ROOT, 32),
                        after_round=lambda r, v: got.append([simplify(w).as_long() for w in v]))
    for r in range(7):
        if got[r] != want[r]:
            print(f"MISMATCH: C2SP's trace, after round {r}")
            return False
    if [simplify(w).as_long() for w in out[:8]] != [int(w, 16) for w in C2SP_OUTPUT.split()]:
        print("MISMATCH: C2SP's trace, the output")
        return False
    print("C2SP's single-block trace: the state after each of the 7 rounds and the output agree")
    return True


def main():
    if not check_trace():
        sys.exit(1)
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
