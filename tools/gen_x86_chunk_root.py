#!/usr/bin/env python3
"""Generate c/blake3_chunk_root_x86-64_unix.S: the root hash of an input of
one chunk or less, in one call that touches no stack memory.

    python3 tools/gen_x86_chunk_root.py > c/blake3_chunk_root_x86-64_unix.S

Why: a caller that writes a short message and hashes it at once has its
stores still in flight when the hash runs. A load of ours whose address
shares the message's low 12 bits is held behind them (4 KiB aliasing): on
an AMD EPYC 9255, hash() of 64 bytes ran at 1.05 or 1.45-1.55 ns/B by where
its stack frames fell (bench-hashes #65). The chunk path through ChunkState
copies the message to the stack and keeps the chaining value there; this
kernel keeps both in registers, as AArch64's c1 kernel already does (flat at
every offset on the M4 Max's VM, #65).

Contract (System V AMD64; Linux and macOS):

    blake3_{sse41,avx512}_chunk_root(input, key, last, packed, out)

- input: `whole` blocks of 64 bytes, read in place (whole = 0..15).
- key: eight little-endian u32 words.
- last: the final block, 64 readable bytes (in place when the input ends
  with a whole block; a zero-padded copy otherwise, or zeros when empty).
- packed: flags | last_len << 8 | whole << 16, last_len 0..64.
- out: 32 bytes, written once: the root output's first 32 bytes.
The chunk counter is 0. CHUNK_START goes on the first block, CHUNK_END and
ROOT on the last. Only the arguments' memory is read or written: no push, no
spill, no stack scratch, no constant loads (IV and shuffle masks are built
from immediates). Clobbers rax, rcx, rdx, rdi, r9-r11, xmm0-xmm15 and flags,
all caller-saved. The sse41 entry needs SSE4.1; the avx512 entry needs
AVX-512F and AVX-512VL (vprord on xmm).

The compression body is upstream BLAKE3's blake3_compress_in_place_sse41,
instruction for instruction, with its rotations rendered per ISA. Proof
tooling covers the upstream kernels; this one is not yet in it.
"""
import sys

CHUNK_START, CHUNK_END, ROOT = 1, 2, 8
IV = (0xBB67AE856A09E667, 0xA54FF53A3C6EF372)
ROT16 = (0x0504070601000302, 0x0D0C0F0E09080B0A)
ROT8 = (0x0407060500030201, 0x0C0F0E0D080B0A09)

# The round's instruction stream in SSE's two-operand form (destination
# first and also a source), as upstream writes it, with rotations named.
ROUND = '''
paddd xmm0 xmm4 | paddd xmm0 xmm1 | pxor xmm3 xmm0 | ror xmm3 16 | paddd xmm2 xmm3 | pxor xmm1 xmm2 | ror xmm1 12
paddd xmm0 xmm5 | paddd xmm0 xmm1 | pxor xmm3 xmm0 | ror xmm3 8 | paddd xmm2 xmm3 | pxor xmm1 xmm2 | ror xmm1 7
pshufd xmm0 xmm0 0x93 | pshufd xmm3 xmm3 0x4E | pshufd xmm2 xmm2 0x39
paddd xmm0 xmm6 | paddd xmm0 xmm1 | pxor xmm3 xmm0 | ror xmm3 16 | paddd xmm2 xmm3 | pxor xmm1 xmm2 | ror xmm1 12
paddd xmm0 xmm7 | paddd xmm0 xmm1 | pxor xmm3 xmm0 | ror xmm3 8 | paddd xmm2 xmm3 | pxor xmm1 xmm2 | ror xmm1 7
pshufd xmm0 xmm0 0x39 | pshufd xmm3 xmm3 0x4E | pshufd xmm2 xmm2 0x93
'''
PERMUTE = '''
movdqa xmm8 xmm4 | shufps xmm8 xmm5 214 | pshufd xmm9 xmm4 0x0F | pshufd xmm4 xmm8 0x39
movdqa xmm8 xmm6 | shufps xmm8 xmm7 250 | pblendw xmm9 xmm8 0xCC
movdqa xmm8 xmm7 | punpcklqdq xmm8 xmm5 | pblendw xmm8 xmm6 0xC0 | pshufd xmm8 xmm8 0x78
punpckhdq xmm5 xmm7 | punpckldq xmm6 xmm5 | pshufd xmm7 xmm6 0x1E
movdqa xmm5 xmm9 | movdqa xmm6 xmm8
'''
# Upstream's message load: the block's words into rounds' column order.
LOAD = '''
movups xmm4 [{p}] | movups xmm5 [{p}+0x10] | movaps xmm8 xmm4 | shufps xmm4 xmm5 136 | shufps xmm8 xmm5 221 | movaps xmm5 xmm8
movups xmm6 [{p}+0x20] | movups xmm7 [{p}+0x30] | movaps xmm8 xmm6 | shufps xmm6 xmm7 136 | pshufd xmm6 xmm6 0x93
shufps xmm8 xmm7 221 | pshufd xmm7 xmm8 0x93
'''


def stream(text):
    return [i.split() for line in text.strip().splitlines() for i in line.split('|')]


def render(isa, ins):
    """One instruction of the two-operand stream for `isa`."""
    op, *args = ins
    if op == 'ror':
        reg, n = args[0], int(args[1])
        if isa == 'avx512':
            return [f'vprord {reg}, {reg}, {n}']
        if n in (16, 8):
            return [f'pshufb {reg}, xmm{15 if n == 16 else 14}']
        return [f'movdqa xmm11, {reg}', f'pslld {reg}, {32 - n}', f'psrld xmm11, {n}', f'por {reg}, xmm11']
    mem = lambda a: a.replace('[', 'xmmword ptr [')
    if isa == 'sse41':
        return [f'{op} ' + ', '.join(mem(a) for a in args)]
    if op in ('movups', 'movaps', 'movdqa'):
        return [f'v{op} {args[0]}, {mem(args[1])}']
    if op == 'pshufd':
        return [f'vpshufd {args[0]}, {args[1]}, {args[2]}']
    # dst = dst op src (, imm): VEX three-operand form.
    return [f'v{op} {args[0]}, {args[0]}, ' + ', '.join(args[1:])]


def body(isa, block_ptr):
    """Compress xmm0..xmm3 with the block at block_ptr; CV out in xmm0/xmm1."""
    out = []
    for ins in stream(LOAD.format(p=block_ptr)):
        out += render(isa, ins)
    out.append('mov al, 7')
    out.append('9:')
    for ins in stream(ROUND):
        out += render(isa, ins)
    out += ['dec al', 'jz 8f']
    for ins in stream(PERMUTE):
        out += render(isa, ins)
    out += ['jmp 9b', '8:']
    if isa == 'avx512':
        out += ['vpxor xmm0, xmm0, xmm2', 'vpxor xmm1, xmm1, xmm3']
    else:
        out += ['pxor xmm0, xmm2', 'pxor xmm1, xmm3']
    return out


def qword_pair(isa, reg, pair):
    """reg = the 16 bytes (low qword, high qword), from immediates."""
    lo, hi = pair
    if isa == 'avx512':
        return [f'mov rax, {lo:#018x}', f'vmovq {reg}, rax', f'mov rax, {hi:#018x}', f'vpinsrq {reg}, {reg}, rax, 1']
    return [f'mov rax, {lo:#018x}', f'movq {reg}, rax', f'mov rax, {hi:#018x}', f'pinsrq {reg}, rax, 1']


def row3(isa):
    """xmm3 = (counter 0, 0, block_len, flags) from rax = len | flags << 32."""
    if isa == 'avx512':
        return ['vmovq xmm9, rax', 'vpxor xmm3, xmm3, xmm3', 'vpunpcklqdq xmm3, xmm3, xmm9']
    return ['movq xmm9, rax', 'pxor xmm3, xmm3', 'punpcklqdq xmm3, xmm9']


def kernel(isa):
    name = f'blake3_{isa}_chunk_root'
    x = 'v' if isa == 'avx512' else ''
    out = [f'.p2align 6', f'{name}:', f'_{name}:', '_CET_ENDBR']
    out += ['movzx r11d, cl', 'mov r10, rcx', 'shr r10, 8', 'and r10d, 0xff', 'mov r9, rcx', 'shr r9, 16', 'and r9d, 0xff']
    out += [f'{x}movups xmm0, xmmword ptr [rsi]', f'{x}movups xmm1, xmmword ptr [rsi+0x10]']
    if isa == 'sse41':
        out += qword_pair(isa, 'xmm15', ROT16) + qword_pair(isa, 'xmm14', ROT8)
    out += [f'mov ecx, {CHUNK_START}']  # the start flag, until the first block is done
    out += ['test r9, r9', 'jz 3f', '2:']
    out += qword_pair(isa, 'xmm2', IV)
    out += ['mov eax, r11d', 'or eax, ecx', 'shl rax, 32', 'or rax, 64'] + row3(isa)
    out += body(isa, 'rdi')
    out += ['add rdi, 64', 'xor ecx, ecx', 'dec r9', 'jnz 2b', '3:']
    out += qword_pair(isa, 'xmm2', IV)
    out += ['mov eax, r11d', 'or eax, ecx', f'or eax, {CHUNK_END | ROOT}', 'shl rax, 32', 'or rax, r10'] + row3(isa)
    out += body(isa, 'rdx')
    out += [f'{x}movups xmmword ptr [r8], xmm0', f'{x}movups xmmword ptr [r8+0x10], xmm1', 'ret']
    return out


def main():
    print('// Generated by tools/gen_x86_chunk_root.py: edit that, not this.')
    print('#if defined(__ELF__) && defined(__linux__)\n.section .note.GNU-stack,"",%progbits\n#endif')
    print('#if defined(__ELF__) && defined(__CET__) && defined(__has_include)\n#if __has_include(<cet.h>)\n#include <cet.h>\n#endif\n#endif')
    print('#if !defined(_CET_ENDBR)\n#define _CET_ENDBR\n#endif')
    print('.intel_syntax noprefix')
    for isa in ('sse41', 'avx512'):
        print(f'.global blake3_{isa}_chunk_root\n.global _blake3_{isa}_chunk_root')
    print('#ifdef __APPLE__\n.text\n#else\n.section .text\n#endif')
    for isa in ('sse41', 'avx512'):
        for line in kernel(isa):
            print(line if line.endswith(':') or line.startswith('.') or line == '_CET_ENDBR' else f'        {line}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
