import Generic
import Blake3

/-! Symbolic words for the generic compression (`Generic.lean`), and what each one means.

A `Term` records the operations the generic text applies: each `+`, `^^^`, `rotr`, numeral, and
counter word builds a node. `Term.eval` gives each node its meaning on 32-bit words, the one
`lean_spec.py` gives it in Z3. -/

open Blake3Generic

inductive Term where
  | var (name : String)
  | const (value : Nat)
  | add (a b : Term)
  | xor (a b : Term)
  | rotr (a : Term) (n : Nat)
  | low (counter : String)
  | high (counter : String)
  deriving Inhabited

instance : Add Term := ⟨.add⟩
instance : XorOp Term := ⟨.xor⟩
instance (n : Nat) : OfNat Term n := ⟨.const n⟩
instance : Rotr Term := ⟨.rotr⟩
instance : Halves String Term := ⟨.low, .high⟩

/-- A term's value: `env` gives each variable's word, `t` the counter. -/
def Term.eval (env : String → UInt32) (t : UInt64) : Term → UInt32
  | .var n => env n
  | .const v => v.toUInt32
  | .add a b => a.eval env t + b.eval env t
  | .xor a b => a.eval env t ^^^ b.eval env t
  | .rotr a n => Blake3.rotr (a.eval env t) n
  | .low _ => Blake3.low t
  | .high _ => Blake3.high t
