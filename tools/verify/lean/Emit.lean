import Generic
import Std.Data.HashMap

/-! The compression function over symbolic terms, written as a graph.

`Term` is a free term: each `+`, `^^^`, `rotr`, numeral, and counter word builds a node. The
generic compression (the specification's text, `Generic.lean`) applied to variables builds the
16 output terms; the graph is written with each shared subterm once, as the evaluation shared it.
Every operation of the generic text is one of these nodes, so the graph computes, under the
meaning `lean_spec.py` gives each node (32-bit addition, xor, rotation; the counter's low and
high words), what the generic text computes at `UInt32`: the specification's function
(`Bridge.lean`). -/

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
instance (n : Nat) : OfNat Term n := ⟨.const (n % 2 ^ 32)⟩
instance : Rotr Term := ⟨.rotr⟩
instance : Halves String Term := ⟨.low, .high⟩

/-- Nodes in order, each shared subterm once: sharing is the evaluation's own (a value read
twice is one object), found by address. -/
unsafe def graph (outs : List Term) : String := Id.run do
  let mut ids : Std.HashMap USize Nat := {}
  let mut nodes : Array String := #[]
  let mut stack : List (Term × Bool) := outs.reverse.map (·, false)
  while !stack.isEmpty do
    let (t, ready) := stack.head!
    stack := stack.tail!
    let key := ptrAddrUnsafe t
    if ids.contains key then continue
    let kids : List Term := match (t : Term) with
      | .add a b | .xor a b => [a, b]
      | .rotr a _ => [a]
      | _ => []
    if !ready && kids.any (fun k => !ids.contains (ptrAddrUnsafe k)) then
      stack := kids.map (·, false) ++ (t, true) :: stack
      continue
    let id (k : Term) : Nat := match ids[ptrAddrUnsafe k]? with
      | some n => n
      | none => panic! "a child written after its parent"
    let node : String := match (t : Term) with
      | .var n => s!"[\"var\", \"{n}\"]"
      | .const v => s!"[\"const\", {v}]"
      | .add a b => s!"[\"add\", {id a}, {id b}]"
      | .xor a b => s!"[\"xor\", {id a}, {id b}]"
      | .rotr a n => s!"[\"rotr\", {id a}, {n}]"
      | .low c => s!"[\"low\", \"{c}\"]"
      | .high c => s!"[\"high\", \"{c}\"]"
    ids := ids.insert key nodes.size
    nodes := nodes.push node
  let outIds := outs.map fun o => toString ids[ptrAddrUnsafe o]!
  return "{\"nodes\": [\n" ++ ",\n".intercalate nodes.toList ++ "\n],\n\"outputs\": [" ++ ", ".intercalate outIds ++ "]}\n"

@[implemented_by graph] opaque graphSafe (outs : List Term) : String

def main (args : List String) : IO Unit := do
  let h : Vector Term 8 := Vector.ofFn fun i => .var s!"h{i.val}"
  let m : Vector Term 16 := Vector.ofFn fun i => .var s!"m{i.val}"
  let out := BLAKE3_COMPRESS h m "t" (.var "len") (.var "flags")
  IO.FS.writeFile (args.head!) (graphSafe out.toList)
