import Insns
open Sail Sail.ConcurrencyInterfaceV1 PreSail
open Armv9 Armv9.Functions

/-- BLAKE3's G as the compiled portable code computes it: a, b, c, d in
registers 0 to 3, the message words in 4 and 5. -/
def gSeq : SailM Unit := do
  decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 0#5 0#5 0#6 1#5 0#2 0#1 0#1 0#1
  decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 0#5 0#5 0#6 4#5 0#2 0#1 0#1 0#1
  decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg 3#5 3#5 0#6 0#5 0#1 0#2 0b10#2 0#1
  decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate 3#5 3#5 (BitVec.ofNat 6 16) 3#5 0#1 0#1
  decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 2#5 2#5 0#6 3#5 0#2 0#1 0#1 0#1
  decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg 1#5 1#5 0#6 2#5 0#1 0#2 0b10#2 0#1
  decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate 1#5 1#5 (BitVec.ofNat 6 12) 1#5 0#1 0#1
  decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 0#5 0#5 0#6 1#5 0#2 0#1 0#1 0#1
  decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 0#5 0#5 0#6 5#5 0#2 0#1 0#1 0#1
  decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg 3#5 3#5 0#6 0#5 0#1 0#2 0b10#2 0#1
  decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate 3#5 3#5 (BitVec.ofNat 6 8) 3#5 0#1 0#1
  decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 2#5 2#5 0#6 3#5 0#2 0#1 0#1 0#1
  decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg 1#5 1#5 0#6 2#5 0#1 0#2 0b10#2 0#1
  decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate 1#5 1#5 (BitVec.ofNat 6 7) 1#5 0#1 0#1

/-- The low 32 bits of a zero-extended 32-bit value. -/
@[simp] theorem lo32_zext (x : BitVec 32) : (x.zeroExtend 64).extractLsb 31 0 = x := by
  ext i hi; simp [hi]

@[simp] theorem lo32_setWidth (x : BitVec 32) : (x.setWidth 64).extractLsb 31 0 = x := by
  ext i hi; simp [hi]

@[simp] theorem lo32_self (x : BitVec 32) : x.extractLsb 31 0 = x := by
  ext i hi; simp [hi]

@[simp] theorem lo32_xor (x y : BitVec 64) : (x ^^^ y).extractLsb 31 0 = x.extractLsb 31 0 ^^^ y.extractLsb 31 0 := by
  ext i hi; simp

/-! The instruction lemmas with the values read from the state, for `simp`. -/

theorem add32_run' (s : State) (d n m : BitVec 5) (hd : d.toNat < 31) (hn : n.toNat < 31)
    (hm : m.toNat < 31) (hsn : (gpr s n.toNat).isSome) (hsm : (gpr s m.toNat).isSome) :
    decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg d n 0#6 m 0#2 0#1 0#1 0#1 s
      = .ok () (setGpr s d.toNat ((((gpr s n.toNat).getD 0).extractLsb 31 0
          + ((gpr s m.toNat).getD 0).extractLsb 31 0).zeroExtend 64)) := by
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp hsn
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp hsm
  rw [ha, hb]; exact add32_run s d n m hd hn hm a b ha hb

theorem eor32_run' (s : State) (d n m : BitVec 5) (hd : d.toNat < 31) (hn : n.toNat < 31)
    (hm : m.toNat < 31) (hsn : (gpr s n.toNat).isSome) (hsm : (gpr s m.toNat).isSome) :
    decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg d n 0#6 m 0#1 0#2 0b10#2 0#1 s
      = .ok () (setGpr s d.toNat ((((gpr s n.toNat).getD 0).extractLsb 31 0
          ^^^ ((gpr s m.toNat).getD 0).extractLsb 31 0).zeroExtend 64)) := by
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp hsn
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp hsm
  rw [ha, hb]; exact eor32_run s d n m hd hn hm a b ha hb

theorem ror32_run' (s : State) (d n : BitVec 5) (k : Nat) (hk : k < 32) (hd : d.toNat < 31)
    (hn : n.toNat < 31) (hsn : (gpr s n.toNat).isSome) :
    decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate d n (BitVec.ofNat 6 k) n 0#1 0#1 s
      = .ok () (setGpr s d.toNat (((((gpr s n.toNat).getD 0).extractLsb 31 0).rotateRight k).zeroExtend 64)) := by
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp hsn
  rw [ha]; exact ror32_run s d n k hk hd hn a ha

/-- G, as specified: the quarter-round on 32-bit words. -/
def gSpec (a b c d mx my : BitVec 32) : BitVec 32 × BitVec 32 × BitVec 32 × BitVec 32 :=
  let a := a + b + mx
  let d := (d ^^^ a).rotateRight 16
  let c := c + d
  let b := (b ^^^ c).rotateRight 12
  let a := a + b + my
  let d := (d ^^^ a).rotateRight 8
  let c := c + d
  let b := (b ^^^ c).rotateRight 7
  (a, b, c, d)

/-- Arm's specification runs `gSeq` to G's results, from any state whose
registers 0 to 5 hold a, b, c, d and the message words. -/
theorem gSeq_is_G (s : State) (a b c d mx my : BitVec 64)
    (h0 : gpr s 0 = some a) (h1 : gpr s 1 = some b) (h2 : gpr s 2 = some c) (h3 : gpr s 3 = some d)
    (h4 : gpr s 4 = some mx) (h5 : gpr s 5 = some my) :
    match gSeq s with
    | .ok () s' =>
      let g := gSpec (a.extractLsb 31 0) (b.extractLsb 31 0) (c.extractLsb 31 0) (d.extractLsb 31 0)
        (mx.extractLsb 31 0) (my.extractLsb 31 0)
      gpr s' 0 = some (g.1.zeroExtend 64) ∧ gpr s' 1 = some (g.2.1.zeroExtend 64) ∧
      gpr s' 2 = some (g.2.2.1.zeroExtend 64) ∧ gpr s' 3 = some (g.2.2.2.zeroExtend 64)
    | .error _ _ => False := by
  simp only [gSeq, bind, EStateM.bind]
  simp [add32_run', eor32_run', ror32_run', gpr_setGpr, h0, h1, h2, h3, h4, h5, gSpec]

#print axioms gSeq_is_G
