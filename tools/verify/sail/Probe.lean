import Armv9
open Sail Sail.ConcurrencyInterfaceV1 PreSail
open Armv9 Armv9.Functions

/-- Sail's shift by an integer, by zero. -/
theorem toNat_shiftLeft_int_zero {w : Nat} (x : BitVec w) : (x <<< (0 : Int)).toNat = x.toNat := by
  show (BitVec.shiftLeft x 0).toNat = _
  simp

theorem add_w0_w1_w2 (s : SequentialState RegisterType trivialChoiceSource) (a b : BitVec 64)
    (h1 : s.regs.get? Register.R1 = some a) (h2 : s.regs.get? Register.R2 = some b) :
    ∃ s', decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg
        0#5 1#5 0#6 2#5 0#2 0#1 0#1 0#1 s = .ok () s' ∧
      s'.regs.get? Register.R0 = some ((a.extractLsb 31 0 + b.extractLsb 31 0).zeroExtend 64) := by
  simp (config := { decide := true }) only [decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg]
  simp (config := { decide := true }) [execute_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg,
    X_read, X_set, get_R, set_R, read_gpr, write_gpr, ShiftReg, DecodeShift, AddWithCarry, simp_sail,
    bind, EStateM.bind, get, getThe, MonadStateOf.get, EStateM.get, modify, modifyGet,
    MonadStateOf.modifyGet, EStateM.modifyGet, pure, EStateM.pure, h1, h2,
    Armv9.undefined_bitvector, PreSail.undefined_bitvector, choose]
  simp only [integer_subrange, get_slice_int]
  apply congrArg
  apply BitVec.eq_of_toNat_eq
  simp [BitVec.extractLsb'_toNat, BitVec.toNat_ofInt, BitVec.toNat_add, BitVec.toNat_shiftLeft,
    BitVec.extractLsb_toNat, toNat_shiftLeft_int_zero, instHShiftLeftBitVecInt_sail]
  erw [toNat_shiftLeft_int_zero, BitVec.extractLsb_toNat]
  simp only [Nat.shiftRight_zero]
  omega

#print axioms add_w0_w1_w2
