"""Public conditional addresses and stores retain their actual alternatives."""
import unittest
from z3 import BitVec, BitVecVal, Extract, ZeroExt, ULE, simplify, substitute, Solver, unsat
import x86
from aarch64 import Ptr, Region, Unproved, concrete


def machine():
    m=x86.Machine({}, {})
    m.x['rsp']=Ptr('stack',0)
    m.v={i:[BitVecVal(0,32)]*16 for i in range(32)}
    return m


def equal(a,b):
    s=Solver();s.add(a!=b)
    return s.check()==unsat


class SymbolicEnds(unittest.TestCase):
    def selected(self,block,shift=0):
        m=machine();end=BitVec('end',4);m.lengths={'end'}
        m.assumptions=[ULE(end,BitVecVal(3,4))]
        m.regions['prefix']=Region('prefix',ZeroExt(60,end)*64,False,lambda o:BitVecVal(o%256,8))
        m.regions['last']=Region('last',64,False,lambda o:BitVecVal(255-o,8))
        m.x['rax']=Ptr('prefix',block*64+shift);m.x['rdx']=Ptr('last',0)
        m.flags=('sub',BitVecVal(block*64,32),ZeroExt(28,end)*64)
        x86.step(m,0,'cmovae',['rax','rdx'],1)
        return m,end

    def test_pointer_choice_checks_only_the_selected_region(self):
        for block in range(4):
            m,end=self.selected(block)
            x86.step(m,0,'vmovdqu32',['zmm0','zmmword ptr [rax]'],1)
            for e in range(4):
                words=m.v[0]
                for j,word in enumerate(words):
                    first=4*j
                    wanted=sum(((255-first-i) if block>=e else ((block*64+first+i)%256)) << (8*i) for i in range(4))
                    self.assertEqual(concrete(substitute(word,(end,BitVecVal(e,4)))),wanted)

    def test_pointer_choice_rejects_a_one_byte_overread(self):
        m,_=self.selected(1,shift=1)
        with self.assertRaisesRegex(Unproved,'outside'):
            x86.step(m,0,'vmovdqu32',['zmm0','zmmword ptr [rax]'],1)

    def test_secret_choice_cannot_select_an_address(self):
        m=machine();m.x['rax']=Ptr('a',0);m.x['rdx']=Ptr('b',0)
        m.flags=('sub',BitVec('secret',32),BitVecVal(1,32))
        with self.assertRaisesRegex(Unproved,'depends on data'):
            x86.step(m,0,'cmovae',['rax','rdx'],1)

    def test_cmov_memory_source_is_read_even_when_not_taken(self):
        m=machine();m.regions['data']=Region('data',8,False,None)
        m.x['rax']=BitVecVal(123,64);m.x['rdi']=Ptr('data',8)
        m.flags=('sub',BitVecVal(1,32),BitVecVal(2,32))
        with self.assertRaisesRegex(Unproved,'outside'):
            x86.step(m,0,'cmova',['rax','qword ptr [rdi]'],1)

    def test_not_taken_cmov32_clears_the_upper_half(self):
        for destination,source,wanted in [('eax','edx',0x1234),('rax','rdx',0xdead000000001234)]:
            m=machine();m.x['rax']=BitVecVal(0xdead000000001234,64);m.x['rdx']=BitVecVal(987,64)
            m.flags=('sub',BitVecVal(1,32),BitVecVal(2,32))
            x86.step(m,0,'cmova',[destination,source],1)
            self.assertEqual(concrete(m.x['rax']),wanted)

    def test_register_pointer_is_not_dereferenced_by_cmov(self):
        m=machine();m.x['rax']=Ptr('a',0);m.x['rdx']=Ptr('not_memory',10000)
        m.flags=('sub',BitVecVal(1,32),BitVecVal(2,32))
        x86.step(m,0,'cmova',['rax','rdx'],1)
        self.assertEqual(m.x['rax'].region,'a')

    def test_symbolic_store_preserves_the_previous_word_when_disabled(self):
        m=machine();mask=BitVec('mask',16);m.lengths={'mask'}
        previous=[BitVec(f'old{j}',32) for j in range(16)]
        m.regions['out']=Region('out',64,True,lambda o:Extract(8*(o%4)+7,8*(o%4),previous[o//4]))
        m.x['rdi']=Ptr('out',0);m.k[1]=mask;m.v[0]=[BitVecVal(100+j,32) for j in range(16)]
        x86.step(m,0,'vmovdqu32',['zmmword ptr [rdi] {k1}','zmm0'],1)
        for bits in [0,1,0x8000,0x5555,0xffff]:
            for j in range(16):
                got=substitute(m.load(Ptr('out',4*j),4),(mask,BitVecVal(bits,16)))
                self.assertTrue(equal(got,BitVecVal(100+j,32) if (bits>>j)&1 else previous[j]))

    def test_symbolic_mask_never_grants_unbounded_output(self):
        m=machine();mask=BitVec('mask',16);m.lengths={'mask'}
        m.regions['out']=Region('out',4,True,lambda _:BitVecVal(0,8));m.x['rdi']=Ptr('out',0);m.k[1]=mask
        with self.assertRaisesRegex(Unproved,'outside'):
            x86.step(m,0,'vmovdqu32',['zmmword ptr [rdi] {k1}','zmm0'],1)
        # Once constrained to the first word, even an uninitialized output
        # is sufficient: the other fifteen disabled words access nothing.
        m.assumptions=[mask==1];m.regions['out']=Region('out',4,True,None)
        x86.step(m,0,'vmovdqu32',['zmmword ptr [rdi] {k1}','zmm0'],1)


if __name__=='__main__':unittest.main()
