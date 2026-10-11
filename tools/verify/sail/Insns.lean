import Gpr
open Sail Sail.ConcurrencyInterfaceV1 PreSail
open Armv9 Armv9.Functions

/-- Sail's shift by an integer, by zero. -/
theorem toNat_shiftLeft_int_zero {w : Nat} (x : BitVec w) : (x <<< (0 : Int)).toNat = x.toNat := by
  show (BitVec.shiftLeft x 0).toNat = _
  simp

theorem shiftLeft_int_zero {w : Nat} (x : BitVec w) : x <<< (0 : Int) = x := by
  show BitVec.shiftLeft x 0 = x
  simp

/-- The choice state is `Unit`: an undefined value leaves the state as it was. -/
@[simp] theorem with_choice (s : State) (c : Unit) : { s with choiceState := c } = s := by
  cases s; rfl

@[simp] theorem state_eta (s : State) (c : trivialChoiceSource.α) :
    ({ regs := s.regs, choiceState := c, mem := s.mem, tags := s.tags, cycleCount := s.cycleCount,
       sailOutput := s.sailOutput } : State) = s := by
  cases s; rfl

/-- An undefined value, under the trivial choice source: the state as it was. -/
theorem undefined_bitvector_run (s : State) (n : Nat) :
    Armv9.undefined_bitvector n s = .ok (trivialChoiceSource.choose (.bitvector n) ()) s := by
  cases s; rfl

/-- `add Wd, Wn, Wm`: register `d` receives the 32-bit sum, zero-extended. -/
theorem add32_run (s : State) (d n m : BitVec 5) (hd : d.toNat < 31) (hn : n.toNat < 31)
    (hm : m.toNat < 31) (a b : BitVec 64) (ha : gpr s n.toNat = some a) (hb : gpr s m.toNat = some b) :
    decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg d n 0#6 m 0#2 0#1 0#1 0#1 s
      = .ok () (setGpr s d.toNat ((a.extractLsb 31 0 + b.extractLsb 31 0).zeroExtend 64)) := by
  have hn1 : ((n.toNat : Int) ≤ 31) := by omega
  have hn2 : ((n.toNat : Int) ≠ 31) := by omega
  have hm1 : ((m.toNat : Int) ≤ 31) := by omega
  have hm2 : ((m.toNat : Int) ≠ 31) := by omega
  have hd1 : ((d.toNat : Int) ≤ 31) := by omega
  have hd2 : ((d.toNat : Int) ≠ 31) := by omega
  simp (config := { decide := true }) only [decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg]
  simp (config := { decide := true }) [execute_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg,
    X_read, X_set, get_R, set_R, ShiftReg, DecodeShift, AddWithCarry, simp_sail,
    bind, EStateM.bind, pure, EStateM.pure, undefined_bitvector_run,
    read_gpr_run s n.toNat hn a ha, read_gpr_run s m.toNat hm b hb, write_gpr_run s d.toNat hd,
    hn1, hn2, hm1, hm2, hd1, hd2, trivialChoiceSource]
  simp only [integer_subrange, get_slice_int]
  apply congrArg
  apply congrArg
  apply BitVec.eq_of_toNat_eq
  simp [BitVec.extractLsb'_toNat, BitVec.toNat_ofInt, BitVec.toNat_add, BitVec.extractLsb_toNat]
  erw [toNat_shiftLeft_int_zero, BitVec.extractLsb_toNat]
  simp only [Nat.shiftRight_zero]
  omega

theorem undefined_bool_run (s : State) :
    Armv9.undefined_bool () s = .ok (trivialChoiceSource.choose .bool ()) s := by
  cases s; rfl

theorem undefined_LogicalOp_run (s : State) :
    undefined_LogicalOp () s = .ok (Armv9.LogicalOp.LogicalOp_AND) s := by
  cases s; rfl

/-- `eor Wd, Wn, Wm`: register `d` receives the 32-bit exclusive or, zero-extended. -/
theorem eor32_run (s : State) (d n m : BitVec 5) (hd : d.toNat < 31) (hn : n.toNat < 31)
    (hm : m.toNat < 31) (a b : BitVec 64) (ha : gpr s n.toNat = some a) (hb : gpr s m.toNat = some b) :
    decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg d n 0#6 m 0#1 0#2 0b10#2 0#1 s
      = .ok () (setGpr s d.toNat ((a.extractLsb 31 0 ^^^ b.extractLsb 31 0).zeroExtend 64)) := by
  have hn1 : ((n.toNat : Int) ≤ 31) := by omega
  have hn2 : ((n.toNat : Int) ≠ 31) := by omega
  have hm1 : ((m.toNat : Int) ≤ 31) := by omega
  have hm2 : ((m.toNat : Int) ≠ 31) := by omega
  have hd1 : ((d.toNat : Int) ≤ 31) := by omega
  have hd2 : ((d.toNat : Int) ≠ 31) := by omega
  simp (config := { decide := true }) only [decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg]
  simp (config := { decide := true }) [execute_aarch64_instrs_integer_logical_shiftedreg,
    X_read, X_set, get_R, set_R, ShiftReg, DecodeShift, simp_sail,
    bind, EStateM.bind, pure, EStateM.pure, undefined_bitvector_run, undefined_bool_run, undefined_LogicalOp_run,
    read_gpr_run s n.toNat hn a ha, read_gpr_run s m.toNat hm b hb, write_gpr_run s d.toNat hd,
    hn1, hn2, hm1, hm2, hd1, hd2]
  congr 1
  erw [shiftLeft_int_zero]
  ext i
  simp only [BitVec.getElem_setWidth, BitVec.getElem_xor]
  erw [BitVec.getLsbD_xor, BitVec.getLsbD_extractLsb, BitVec.getLsbD_extractLsb]

/-- Bits k to k + 31 of x joined with itself: x rotated right by k. -/
theorem extr_rotate (x : BitVec 32) (k lo hi : Nat) (hk : k < 32) (hlo : lo = k) (hhi : hi = k + 31) :
    BitVec.setWidth 32 (BitVec.extractLsb hi lo (x ++ x)) = x.rotateRight k := by
  subst hlo hhi
  ext i hi
  simp [BitVec.getElem_rotateRight, BitVec.getElem_setWidth, hi]
  rw [BitVec.getLsbD_append]
  have hmod : lo % 32 = lo := Nat.mod_eq_of_lt hk
  simp only [hmod]
  split <;> split
  · rw [BitVec.getLsbD_eq_getElem]
  · exfalso; omega
  · exfalso; omega
  · have e : lo + i - 32 = i - (32 - lo) := by omega
    rw [e, BitVec.getLsbD_eq_getElem]

/-- `ror Wd, Wn, #k` (`extr Wd, Wn, Wn, #k`): register `d` receives the low 32
bits of `n` rotated right by `k`, zero-extended. -/
theorem ror32_run (s : State) (d n : BitVec 5) (k : Nat) (hk : k < 32) (hd : d.toNat < 31)
    (hn : n.toNat < 31) (a : BitVec 64) (ha : gpr s n.toNat = some a) :
    decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate d n (BitVec.ofNat 6 k) n 0#1 0#1 s
      = .ok () (setGpr s d.toNat (((a.extractLsb 31 0).rotateRight k).zeroExtend 64)) := by
  have hn1 : ((n.toNat : Int) ≤ 31) := by omega
  have hn2 : ((n.toNat : Int) ≠ 31) := by omega
  have hd1 : ((d.toNat : Int) ≤ 31) := by omega
  have hd2 : ((d.toNat : Int) ≠ 31) := by omega
  have hk6 : (BitVec.ofNat 6 k).toNat = k := by simp; omega
  have hb5 : (BitVec.ofNat 6 k)[5]! = false := by
    rw [getElem!_pos (BitVec.ofNat 6 k) 5 (by omega), BitVec.getElem_eq_testBit_toNat]
    simp only [BitVec.toNat_ofNat]
    apply Nat.testBit_lt_two_pow
    omega
  have hc : (BitVec.ofBoolListBE [(BitVec.ofBool (BitVec.ofNat 6 k)[5]!)[0]] == 1#1) = false := by
    rw [hb5]; simp [BitVec.ofBoolListBE]
  simp (config := { decide := true }) only [decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate]
  simp (config := { decide := true }) [execute_aarch64_instrs_integer_ins_ext_extract_immediate,
    X_read, X_set, get_R, set_R, simp_sail, hk6,
    bind, EStateM.bind, pure, EStateM.pure,
    read_gpr_run s n.toNat hn a ha, write_gpr_run s d.toNat hd,
    hn1, hn2, hd1, hd2]
  have hb5' : (BitVec.ofNat 6 k)[5] = false := by
    rw [BitVec.getElem_eq_testBit_toNat]
    simp only [BitVec.toNat_ofNat]
    apply Nat.testBit_lt_two_pow
    omega
  have hk2 : ((k : Int) + 32 - 1 < 64) := by omega
  split
  · rename_i hbad
    simp [BitVec.ofBoolListBE, hb5'] at hbad
  simp (config := { decide := true }) [execute_aarch64_instrs_integer_ins_ext_extract_immediate,
    X_read, X_set, get_R, set_R, simp_sail, hk6,
    bind, EStateM.bind, pure, EStateM.pure,
    read_gpr_run s n.toNat hn a ha, write_gpr_run s d.toNat hd,
    hn1, hn2, hd1, hd2, hk2]
  congr 2
  exact extr_rotate _ k _ _ hk hk6 (by omega)

#print axioms add32_run
#print axioms eor32_run
#print axioms ror32_run
