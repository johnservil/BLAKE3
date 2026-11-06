"""Memory contracts: ABI frames, vector alignment and secret-independent addresses."""

import unittest
from unittest.mock import patch

from z3 import BitVec, BitVecVal
import aarch64 as arm
from aarch64 import Ptr, Region, Unproved
import prove_x86 as p
import x86


def machine(offset=0, alignment=64):
    m = x86.Machine({}, {"data": Region("data", 128, True, lambda _: BitVecVal(0, 8), alignment=alignment)})
    m.x["rsp"] = Ptr("stack", 0)
    m.x["rdi"] = Ptr("data", offset)
    m.v = {i: [BitVecVal(0, 32)] * 16 for i in range(32)}
    return m


class X86Memory(unittest.TestCase):
    def test_symbolic_entry_stack_covers_only_legal_residues(self):
        from z3 import BitVecVal, Solver, sat, unsat
        with patch.multiple(p.x86, symbols=lambda _: {"entry": 0}, disassemble=lambda *args: {},
                            text_end=lambda _: 1, image=lambda _: {0: 0}):
            setup = p.Setup("unused", "entry", {}, [])
        residue = setup.m.entry_stack - p.STACK
        for value, expected in [(8, sat), (24, sat), (40, sat), (56, sat),
                                (0, unsat), (16, unsat), (32, unsat), (48, unsat), (64, unsat)]:
            with self.subTest(residue=value):
                solver = Solver()
                solver.add(*setup.m.assumptions, residue == BitVecVal(value, 64))
                self.assertEqual(solver.check(), expected)
        self.assertIn("len_stack", setup.m.lengths)

    def test_stack_boundary_and_entry_residues(self):
        self.assertEqual({s % 64 for s in p.ENTRY_STACKS}, {8, 24, 40, 56})
        with patch.multiple(p.x86, symbols=lambda _: {"entry": 0}, disassemble=lambda *args: {},
                            text_end=lambda _: 1, image=lambda _: {0: 0}):
            for nargs in (0, 5, 6, 7, 10):
                for stack in p.ENTRY_STACKS:
                    with self.subTest(nargs=nargs, stack=stack):
                        setup = p.Setup("unused", "entry", {}, [BitVecVal(0, 64)] * nargs, stack)
                        end = stack + 8 + 8 * max(0, nargs - 6)
                        self.assertEqual(arm.concrete(setup.m.regions["stack"].size), end)
                        self.assertEqual(setup.m.regions["stack"].alignment, 64)
                        setup.m.access(Ptr("stack", end - 8), 8, True)
                        with self.assertRaises(Unproved):
                            setup.m.access(Ptr("stack", end), 1, True)

    def test_aligned_memory_forms(self):
        forms = [("movaps", ["xmm0", "xmmword ptr [rdi]"], 16),
                 ("paddd", ["xmm0", "xmmword ptr [rdi]"], 16),
                 ("vmovdqa", ["ymm0", "ymmword ptr [rdi]"], 32),
                 ("vmovdqa32", ["zmm0", "zmmword ptr [rdi]"], 64),
                 ("vmovdqa", ["ymmword ptr [rdi]", "ymm0"], 32)]
        for mnemonic, ops, alignment in forms:
            with self.subTest(mnemonic=mnemonic, ops=ops):
                x86.step(machine(alignment), 0, mnemonic, ops, 0)
                for m in (machine(4), machine(0, 4)):
                    with self.assertRaisesRegex(Unproved, "alignment"):
                        x86.step(m, 0, mnemonic, ops, 0)

    def test_unaligned_memory_forms(self):
        for mnemonic, ops in [("movups", ["xmm0", "xmmword ptr [rdi]"]),
                              ("movdqu", ["xmmword ptr [rdi]", "xmm0"]),
                              ("vpaddd", ["ymm0", "ymm1", "ymmword ptr [rdi]"])]:
            with self.subTest(mnemonic=mnemonic):
                x86.step(machine(1, 1), 0, mnemonic, ops, 0)

    def test_addresses_include_prefetch(self):
        m = machine()
        m.x["rax"] = BitVec("secret", 64) & 15
        with self.assertRaisesRegex(Unproved, "data-dependent memory address"):
            x86.address(m, "[rdi+rax]")
        with self.assertRaisesRegex(Unproved, "data-dependent memory address"):
            x86.step(m, 0, "prefetcht0", ["[rdi+rax]"], 0)
        m.x["rax"] = BitVec("len_offset", 64)
        m.lengths = {"len_offset"}
        x86.step(m, 0, "prefetcht0", ["[rdi+rax+128]"], 0)

    def test_memory_masks_are_public(self):
        m = machine()
        m.k[1] = BitVec("secret_mask", 16)
        # A secret-dependent register result has no memory effect.
        x86.step(m, 0, "vpaddd", ["zmm0 {k1} {z}", "zmm1", "zmm2"], 0)
        ops = ["zmm0 {k1} {z}", "zmmword ptr [rdi]"]
        with self.assertRaisesRegex(Unproved, "data-dependent memory mask"):
            x86.step(m, 0, "vmovdqu32", ops, 0)
        m.k[1] = BitVec("public_mask", 16)
        m.lengths = {"public_mask"}
        x86.step(m, 0, "vmovdqu32", ops, 0)

    def test_masked_dword_store_writes_enabled_lanes_only(self):
        for mask in [0, 1, 0x8000, 0x5555, 0xffff]:
            m = machine()
            m.v[0] = [BitVecVal(100+j, 32) for j in range(16)]
            m.k[1] = BitVecVal(mask, 16)
            x86.step(m, 0, "vmovdqu32", ["zmmword ptr [rdi] {k1}", "zmm0"], 0)
            self.assertEqual(set(m.regions["data"].bytes), {4*j+b for j in range(16) if (mask >> j) & 1 for b in range(4)})
            for j in range(16):
                if (mask >> j) & 1:
                    self.assertEqual(arm.concrete(m.load(Ptr("data",4*j),4)), 100+j)
        m = machine()
        m.k[1] = BitVec("secret_store_mask",16)
        with self.assertRaisesRegex(Unproved,"data-dependent memory mask"):
            x86.step(m,0,"vmovdqu32",["zmmword ptr [rdi] {k1}","zmm0"],0)

    def test_masked_store_keeps_bounds_for_each_active_lane(self):
        m = machine()
        m.regions["data"] = Region("data",4,True,None)
        m.k[1] = BitVecVal(1,16)
        x86.step(m,0,"vmovdqu32",["zmmword ptr [rdi] {k1}","zmm0"],0)
        m.k[1] = BitVecVal(2,16)
        with self.assertRaises(Unproved):
            x86.step(m,0,"vmovdqu32",["zmmword ptr [rdi] {k1}","zmm0"],0)

    def test_dword_equality_mask_has_one_bit_per_lane(self):
        for mask in [0, 1, 0x8000, 0x5555, 0xffff]:
            m = machine()
            m.v[0] = [BitVecVal(j,32) for j in range(16)]
            m.v[1] = [BitVecVal(j if (mask >> j) & 1 else j+1,32) for j in range(16)]
            x86.step(m,0,"vpcmpeqd",["k1","zmm0","zmm1"],0)
            self.assertEqual(arm.concrete(m.k[1]),mask)

    def test_unmodelled_store_stops(self):
        with self.assertRaisesRegex(Unproved, "unmodelled vector store"):
            x86.step(machine(), 0, "vpmaskmovd", ["ymmword ptr [rdi]", "ymm1", "ymm0"], 0)


if __name__ == "__main__":
    unittest.main()
