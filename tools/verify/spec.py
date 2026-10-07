"""BLAKE3's compression function, written from the specification
(https://github.com/BLAKE3-team/BLAKE3-specs, section 2.2), over Z3
bitvectors. The proofs compare each kernel's outputs with these terms."""

from z3 import BitVecVal, RotateRight, Extract, Concat

IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
CHUNK_START, CHUNK_END, PARENT, ROOT = 1, 2, 4, 8


def g(v, a, b, c, d, x, y):
    v[a] = v[a] + v[b] + x
    v[d] = RotateRight(v[d] ^ v[a], 16)
    v[c] = v[c] + v[d]
    v[b] = RotateRight(v[b] ^ v[c], 12)
    v[a] = v[a] + v[b] + y
    v[d] = RotateRight(v[d] ^ v[a], 8)
    v[c] = v[c] + v[d]
    v[b] = RotateRight(v[b] ^ v[c], 7)


def compress(cv, block, counter, block_len, flags, after_round=None):
    """The 16 output words; `cv` 8 and `block` 16 32-bit terms, `counter`
    64-bit, `block_len` and `flags` 32-bit. `after_round(r, v)`, if given,
    sees the state after each round (for check_spec.py's trace)."""
    v = list(cv) + [BitVecVal(x, 32) for x in IV[:4]] + [
        Extract(31, 0, counter), Extract(63, 32, counter), block_len, flags]
    m = list(block)
    for r in range(7):
        g(v, 0, 4, 8, 12, m[0], m[1])
        g(v, 1, 5, 9, 13, m[2], m[3])
        g(v, 2, 6, 10, 14, m[4], m[5])
        g(v, 3, 7, 11, 15, m[6], m[7])
        g(v, 0, 5, 10, 15, m[8], m[9])
        g(v, 1, 6, 11, 12, m[10], m[11])
        g(v, 2, 7, 8, 13, m[12], m[13])
        g(v, 3, 4, 9, 14, m[14], m[15])
        if after_round:
            after_round(r, list(v))
        if r < 6:
            m = [m[PERMUTATION[i]] for i in range(16)]
    return [v[i] ^ v[i + 8] for i in range(8)] + [v[i + 8] ^ cv[i] for i in range(8)]


def words(byte_terms):
    """Little-endian 32-bit words from 8-bit terms."""
    return [Concat(*reversed(byte_terms[4 * i:4 * i + 4])) for i in range(len(byte_terms) // 4)]
