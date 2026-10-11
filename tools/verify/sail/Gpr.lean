import Armv9
open Sail Sail.ConcurrencyInterfaceV1 PreSail
open Armv9 Armv9.Functions

/-! General-purpose registers as plain values: Arm's specification keeps each
register in a map whose value type depends on the register; `gpr` and
`setGpr` read and write register `n` (0 to 30) as a 64-bit value. -/

abbrev State := SequentialState RegisterType trivialChoiceSource

def gpr (s : State) : Nat → Option (BitVec 64)
  | 0 => s.regs.get? .R0 | 1 => s.regs.get? .R1 | 2 => s.regs.get? .R2 | 3 => s.regs.get? .R3
  | 4 => s.regs.get? .R4 | 5 => s.regs.get? .R5 | 6 => s.regs.get? .R6 | 7 => s.regs.get? .R7
  | 8 => s.regs.get? .R8 | 9 => s.regs.get? .R9 | 10 => s.regs.get? .R10 | 11 => s.regs.get? .R11
  | 12 => s.regs.get? .R12 | 13 => s.regs.get? .R13 | 14 => s.regs.get? .R14 | 15 => s.regs.get? .R15
  | 16 => s.regs.get? .R16 | 17 => s.regs.get? .R17 | 18 => s.regs.get? .R18 | 19 => s.regs.get? .R19
  | 20 => s.regs.get? .R20 | 21 => s.regs.get? .R21 | 22 => s.regs.get? .R22 | 23 => s.regs.get? .R23
  | 24 => s.regs.get? .R24 | 25 => s.regs.get? .R25 | 26 => s.regs.get? .R26 | 27 => s.regs.get? .R27
  | 28 => s.regs.get? .R28 | 29 => s.regs.get? .R29 | 30 => s.regs.get? .R30 | _ => none

def setGpr (s : State) (n : Nat) (v : BitVec 64) : State :=
  { s with regs := match n with
    | 0 => s.regs.insert .R0 v | 1 => s.regs.insert .R1 v | 2 => s.regs.insert .R2 v
    | 3 => s.regs.insert .R3 v | 4 => s.regs.insert .R4 v | 5 => s.regs.insert .R5 v
    | 6 => s.regs.insert .R6 v | 7 => s.regs.insert .R7 v | 8 => s.regs.insert .R8 v
    | 9 => s.regs.insert .R9 v | 10 => s.regs.insert .R10 v | 11 => s.regs.insert .R11 v
    | 12 => s.regs.insert .R12 v | 13 => s.regs.insert .R13 v | 14 => s.regs.insert .R14 v
    | 15 => s.regs.insert .R15 v | 16 => s.regs.insert .R16 v | 17 => s.regs.insert .R17 v
    | 18 => s.regs.insert .R18 v | 19 => s.regs.insert .R19 v | 20 => s.regs.insert .R20 v
    | 21 => s.regs.insert .R21 v | 22 => s.regs.insert .R22 v | 23 => s.regs.insert .R23 v
    | 24 => s.regs.insert .R24 v | 25 => s.regs.insert .R25 v | 26 => s.regs.insert .R26 v
    | 27 => s.regs.insert .R27 v | 28 => s.regs.insert .R28 v | 29 => s.regs.insert .R29 v
    | 30 => s.regs.insert .R30 v | _ => s.regs }

/-- Arm's register read is `gpr`. -/
theorem read_gpr_run (s : State) (n : Nat) (h : n < 31) (v : BitVec 64) (hv : gpr s n = some v) :
    read_gpr n s = .ok v s := by
  match n, h with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ | 14, _ | 15, _ | 16, _ | 17, _ | 18, _ | 19, _ | 20, _ | 21, _ | 22, _ | 23, _ | 24, _ | 25, _ | 26, _ | 27, _ | 28, _ | 29, _ | 30, _ => simp_all [read_gpr, gpr, PreSail.readReg, get, getThe, MonadStateOf.get,
    EStateM.get, bind, EStateM.bind, pure, EStateM.pure]
  | _ + 31, h => omega

/-- Arm's register write is `setGpr`. -/
theorem write_gpr_run (s : State) (n : Nat) (h : n < 31) (v : BitVec 64) :
    write_gpr n v s = .ok () (setGpr s n v) := by
  match n, h with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ | 14, _ | 15, _ | 16, _ | 17, _ | 18, _ | 19, _ | 20, _ | 21, _ | 22, _ | 23, _ | 24, _ | 25, _ | 26, _ | 27, _ | 28, _ | 29, _ | 30, _ => rfl
  | _ + 31, h => omega

set_option maxHeartbeats 0 in
/-- A write to register `d` changes register `d` alone. -/
theorem gpr_setGpr (s : State) (d n : Nat) (hd : d < 31) (hn : n < 31) (v : BitVec 64) :
    gpr (setGpr s d v) n = if n = d then some v else gpr s n := by
  match d, hd with
  | _ + 31, hd => omega
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ | 14, _ | 15, _ | 16, _ | 17, _ | 18, _ | 19, _ | 20, _ | 21, _ | 22, _ | 23, _ | 24, _ | 25, _ | 26, _ | 27, _ | 28, _ | 29, _ | 30, _ =>
    match n, hn with
    | _ + 31, hn => omega
    | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ | 14, _ | 15, _ | 16, _ | 17, _ | 18, _ | 19, _ | 20, _ | 21, _ | 22, _ | 23, _ | 24, _ | 25, _ | 26, _ | 27, _ | 28, _ | 29, _ | 30, _ => simp [gpr, setGpr, Std.ExtDHashMap.get?_insert]
