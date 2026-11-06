"""Prove sixteen independent root-output compressions and their ABI/bounds.

Each lane has arbitrary CV words, 64 block bytes, a byte length and flags.
Root output block zero is spec.compress(cv, block, 0, len, flags|ROOT)[:8].
The descriptor table, CVs and blocks are read-only. Every legal SysV stack
residue is symbolic. Four planted faults must be rejected by Unproved.
"""
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecVal, ULE, ZeroExt
from aarch64 import Ptr, Region, Unproved
from prove_hybrid import byte_symbols, same
from prove_x86 import Setup, d_row_facts
import lean_spec as spec
import x86
import workers

ROOT = pathlib.Path(__file__).resolve().parents[2]


def prove(source):
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d)
        asm = path / 'output.S'
        asm.write_text(source)
        obj = path / 'output.o'
        subprocess.run([x86.tool('clang'), '-c', '--target=x86_64-linux-gnu', str(asm), '-o', str(obj)], check=True)
        so = x86.link(str(obj))
        cvs = [byte_symbols(f'cv{i}_', 32) for i in range(16)]
        blocks = [byte_symbols(f'block{i}_', 64) for i in range(16)]
        lens = [BitVec(f'last_length_{i}', 8) for i in range(16)]
        flags = [BitVec(f'root_flags_{i}', 8) for i in range(16)]
        regions = {'out': Region('out', 512, True, None)}
        table = Region('descriptors', 16 * 24, False, None, alignment=8)
        for i in range(16):
            regions[f'cv{i}'] = Region(f'cv{i}', 32, False, lambda o, i=i: cvs[i][o])
            regions[f'block{i}'] = Region(f'block{i}', 64, False, lambda o, i=i: blocks[i][o])
            for j in range(8):
                table.bytes[24*i+j] = (Ptr(f'block{i}', 0), j)
                table.bytes[24*i+8+j] = (Ptr(f'cv{i}', 0), j)
            table.bytes[24*i+16] = lens[i]
            table.bytes[24*i+17] = flags[i]
        regions['descriptors'] = table
        setup = Setup(so, 'blake3_servil_output16', regions, [Ptr('descriptors', 0), Ptr('out', 0)])
        setup.m.assumptions += [ULE(length, 64) for length in lens]
        words = [value for i in range(16) for value in [ZeroExt(24, lens[i]), ZeroExt(24, flags[i] | 8)]]
        d_row_facts(setup.m, BitVecVal(0, 64), 1, words)
        m = setup.run()
        out = m.regions['out']
        if set(out.bytes) != set(range(512)):
            raise Unproved('root outputs are not written exactly')
        for i in range(16):
            expected = spec.compress(spec.words(cvs[i]), spec.words(blocks[i]), BitVecVal(0, 64), ZeroExt(24, lens[i]), ZeroExt(24, flags[i] | 8))
            got = spec.words([out.bytes[32*i+j] for j in range(32)])
            for w in range(8):
                if not same(got[w], expected[w]):
                    raise Unproved(f'lane {i} word {w} differs from root compression')
        return m.steps


def prove_case(name):
    source = (ROOT / 'c/blake3_output16.S').read_text()
    if name == 'positive':
        return f'output16 positive: {prove(source)} steps; arbitrary per-lane CV/block/length/flags'
    controls = {
        'ROOT omitted': ('    movzx edx, BYTE PTR [rdi + 17]\n    or edx, 8\n    mov DWORD PTR [rsp + 320], edx',
                         '    movzx edx, BYTE PTR [rdi + 17]\n    or edx, 0\n    mov DWORD PTR [rsp + 320], edx'),
        'CV from another lane': ('    mov rax, QWORD PTR [rdi + 8]', '    mov rax, QWORD PTR [rdi + 32]'),
        'CV read exceeds 32 bytes': ('    vmovdqu32 ymm16, YMMWORD PTR [rax]', '    vmovdqu32 ymm16, YMMWORD PTR [rax + 4]'),
    }
    stores = [line for line in source.splitlines() if 'vmovdqu32 ZMMWORD PTR [r15 + 448]' in line]
    assert len(stores) == 1
    controls['last outputs omitted'] = (stores[0], '    nop')
    old, new = controls[name]
    assert source.count(old) == 1, name
    try:
        prove(source.replace(old, new))
    except Unproved as error:
        return name + ': rejected: ' + str(error)
    raise AssertionError(name + ' escaped the proof')


def main():
    cases = ['positive', 'ROOT omitted', 'CV from another lane', 'CV read exceeds 32 bytes', 'last outputs omitted']
    for result in workers.each(prove_case, cases, 3):
        print(result, flush=True)
    print('output16 proof and four fault controls passed', flush=True)


if __name__ == '__main__':
    main()
