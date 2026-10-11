import Armv9
open Sail Sail.ConcurrencyInterfaceV1 PreSail
open Armv9 Armv9.Functions

/-- A state with general registers 0 to 30 set from `vals`. -/
def mkState (vals : Array (BitVec 64)) : SequentialState RegisterType trivialChoiceSource :=
  let regs : Std.ExtDHashMap Register RegisterType :=
    [Register.R0, .R1, .R2, .R3, .R4, .R5, .R6, .R7, .R8, .R9, .R10, .R11, .R12, .R13, .R14, .R15,
     .R16, .R17, .R18, .R19, .R20, .R21, .R22, .R23, .R24, .R25, .R26, .R27, .R28, .R29, .R30].zipIdx.foldl
      (fun m (r, i) => match r with
        | .R0 => m.insert .R0 vals[i]! | .R1 => m.insert .R1 vals[i]! | .R2 => m.insert .R2 vals[i]!
        | .R3 => m.insert .R3 vals[i]! | _ => m) {}
  ⟨regs, (), {}, (), 0, #[]⟩

def r0 (s : SequentialState RegisterType trivialChoiceSource) : BitVec 64 :=
  (s.regs.get? Register.R0).getD 0

/-- Each line: op a b k; prints Arm's specification's result in w0/x0. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let mut line ← stdin.getLine
  while line != "" do
    match line.trim.splitOn " " with
    | [op, a, b, k] =>
      let a := (BitVec.ofNat 64 a.toNat!); let b := (BitVec.ofNat 64 b.toNat!); let k := k.toNat!
      let s := mkState #[0, a, b, 0]
      let m : SailM Unit := match op with
        | "add" => decode_add_addsub_shift_aarch64_instrs_integer_arithmetic_add_sub_shiftedreg 0#5 1#5 0#6 2#5 0#2 0#1 0#1 0#1
        | "eor" => decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg 0#5 1#5 0#6 2#5 0#1 0#2 0b10#2 0#1
        | "eorror" => decode_eor_log_shift_aarch64_instrs_integer_logical_shiftedreg 0#5 1#5 (BitVec.ofNat 6 k) 2#5 0#1 0b11#2 0b10#2 0#1
        | _ => decode_extr_aarch64_instrs_integer_ins_ext_extract_immediate 0#5 1#5 (BitVec.ofNat 6 k) 1#5 0#1 0#1
      match m s with
      | .ok () s' => IO.println s!"{(r0 s').toNat}"
      | .error _ _ => IO.println "error"
    | _ => pure ()
    line ← stdin.getLine
