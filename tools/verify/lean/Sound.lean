import Term
import Bridge

/-! What `Emit.lean` writes is the specification's compression function.

`symbolic` is the generic compression applied to variables, the terms `Emit.lean` writes as a
graph. `compress_map` proves that evaluating the generic compression's result over terms is the
generic compression of the evaluated inputs; with `Bridge.lean`, `symbolic_is_the_specifications`
proves that evaluating `symbolic` under any values of its variables gives the specification's
`BLAKE3_COMPRESS` of those values. -/

open Blake3Generic

/-- The generic compression over the variables `h0`-`h7`, `m0`-`m15`, `len`, `flags`, and the
counter `t`: what `Emit.lean` writes. -/
def symbolic : Vector Term 16 :=
  BLAKE3_COMPRESS (Vector.ofFn fun i => .var s!"h{i.val}") (Vector.ofFn fun i => .var s!"m{i.val}") "t"
    (.var "len") (.var "flags")

section
variable (env : String → UInt32) (t : UInt64)

theorem eval_add (a b : Term) : (a + b).eval env t = a.eval env t + b.eval env t := rfl
theorem eval_xor (a b : Term) : (a ^^^ b).eval env t = a.eval env t ^^^ b.eval env t := rfl
theorem eval_rotr (a : Term) (n : Nat) : (Rotr.rotr a n).eval env t = Blake3.rotr (a.eval env t) n := rfl
theorem eval_low (c : String) : (Halves.low c : Term).eval env t = (Halves.low t : UInt32) := rfl
theorem eval_high (c : String) : (Halves.high c : Term).eval env t = (Halves.high t : UInt32) := rfl

theorem eval_get {n : Nat} (w : Vector Term n) (j : Nat) (h : j < n) :
    (w[j]).eval env t = (w.map (Term.eval env t))[j] := by simp

theorem eval_getFin {n : Nat} (w : Vector Term n) (j : Fin n) :
    (w[j]).eval env t = (w.map (Term.eval env t))[j] := by simp

theorem g_map (v : Vector Term 16) (a b c d : Fin 16) (x y : Term) :
    (G v a b c d x y).map (Term.eval env t)
      = G (v.map (Term.eval env t)) a b c d (x.eval env t) (y.eval env t) := by
  simp only [G, Id.run, pure, Vector.map_set, eval_add, eval_xor, eval_rotr, eval_getFin]
  rfl

theorem permute_map (m : Vector Term 16) :
    (PERMUTE m).map (Term.eval env t) = PERMUTE (m.map (Term.eval env t)) := by
  ext i hi; simp [PERMUTE]

theorem iv_map : (IV : Vector Term 8).map (Term.eval env t) = (IV : Vector UInt32 8) := by
  ext i hi
  simp only [Vector.getElem_map]
  match i, hi with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ => rfl

theorem compress_map (h : Vector Term 8) (m : Vector Term 16) (len flags : Term) :
    (BLAKE3_COMPRESS h m "t" len flags).map (Term.eval env t)
      = BLAKE3_COMPRESS (h.map (Term.eval env t)) (m.map (Term.eval env t)) t (len.eval env t) (flags.eval env t) := by
  simp only [BLAKE3_COMPRESS, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.range'_succ, List.range'_zero,
    List.forIn_cons, List.forIn_nil, List.finRange_succ, List.finRange_zero, List.map_cons, List.map_nil,
    bind, pure, Std.Legacy.Range.size]
  simp only [Vector.map_set, g_map, permute_map, eval_get, eval_getFin, eval_xor, iv_map,
    eval_low, eval_high, Vector.map_replicate]
  rfl

/-- Evaluating `symbolic` under values of its variables gives the specification's compression of
those values. -/
theorem symbolic_is_the_specifications :
    symbolic.map (Term.eval env t)
      = Blake3.BLAKE3_COMPRESS (Vector.ofFn fun i => env s!"h{i.val}") (Vector.ofFn fun i => env s!"m{i.val}") t
          (env "len") (env "flags") := by
  rw [symbolic, compress_map, ← compress_is_the_specifications]
  simp only [Vector.map_ofFn]
  rfl

end

#print axioms symbolic_is_the_specifications
