"""Eight independent AVX2 counter-zero compressions, ABI and memory bounds.

All per-lane CV/block bytes, lengths0..64 and flag bytes are independent.
Params384 is read-only; output256 is written exactly. All legal SysV stack
residues are symbolic. Four genuine faults must be rejected by Unproved.
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
import workers
import x86
ROOT = pathlib.Path(__file__).resolve().parents[2]


def prove(source):
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory)
        (path/'kernel.S').write_text(source)
        subprocess.run([x86.tool('clang'),'-c','--target=x86_64-linux-gnu',str(path/'kernel.S'),'-o',str(path/'kernel.o')],check=True)
        so=x86.link(str(path/'kernel.o'))
        cvs=[byte_symbols(f'cv{i}_',32)for i in range(8)]
        blocks=[byte_symbols(f'block{i}_',64)for i in range(8)]
        lens=[BitVec(f'length_{i}',8)for i in range(8)]
        flags=[BitVec(f'flags_{i}',8)for i in range(8)]
        params=Region('params',384,False,None,alignment=64)
        regions={'params':params,'out':Region('out',256,True,None)}
        for i in range(8):
            regions[f'block{i}']=Region(f'block{i}',64,False,lambda offset,i=i:blocks[i][offset])
            for j in range(8):params.bytes[8*i+j]=(Ptr(f'block{i}',0),j)
            for w in range(8):
                for j in range(4):params.bytes[64+4*(8*w+i)+j]=cvs[i][4*w+j]
            for base,value in [(320,lens[i]),(352,flags[i])]:
                params.bytes[base+4*i]=value
                for j in range(1,4):params.bytes[base+4*i+j]=BitVecVal(0,8)
        setup=Setup(so,'blake3_servil_output8',regions,[Ptr('params',0),Ptr('out',0)])
        setup.m.assumptions += [ULE(length,64)for length in lens]
        d_row_facts(setup.m,BitVecVal(0,64),1,[ZeroExt(24,x)for pair in zip(lens,flags)for x in pair])
        m=setup.run();out=m.regions['out']
        if set(out.bytes)!=set(range(256)):raise Unproved('outputs not written exactly')
        for i in range(8):
            expected=spec.compress(spec.words(cvs[i]),spec.words(blocks[i]),BitVecVal(0,64),ZeroExt(24,lens[i]),ZeroExt(24,flags[i]))
            got=spec.words([out.bytes[32*i+j]for j in range(32)])
            for w in range(8):
                if not same(got[w],expected[w]):raise Unproved(f'lane{i} word{w} differs from compression')
        return m.steps


def one(name):
    source=(ROOT/'c/blake3_output8.S').read_text()
    if name=='positive':return f'output8 positive: {prove(source)} steps, independent CV/block/length/flags'
    faults={
        'wrong flags':('        vmovdqu ymm15, ymmword ptr [rdi+rbx+352]','        vmovdqu ymm15, ymmword ptr [rdi+rbx+320]'),
        'wrong CV word':('        vmovdqu ymm0, ymmword ptr [rdi+64]','        vmovdqu ymm0, ymmword ptr [rdi+96]'),
        'block overread':('        vmovdqu xmm8, xmmword ptr [r8+rdx+48]','        vmovdqu xmm8, xmmword ptr [r8+rdx+52]'),
    }
    stores=[line for line in source.splitlines()if 'vmovdqu ymmword ptr [rsi+224]'in line]
    assert len(stores)==1
    faults['missing final output']=(stores[0],'        nop')
    old,new=faults[name];assert source.count(old)==1,name
    try:prove(source.replace(old,new))
    except Unproved as error:return name+': rejected: '+str(error)
    raise AssertionError(name+' escaped')


if __name__=='__main__':
    for result in workers.each(one,['positive','wrong flags','wrong CV word','block overread','missing final output'],3):print(result,flush=True)
    print('output8 proof and four controls passed',flush=True)
