import Proofs

/-! The SME2 flat walk's shape is the specification's tree.

The walk (src/ffi_sme2.rs, `walk_in`) hashes 2^k chunks, then pairs
adjacent values level by level until `keep` remain; tools/verify/prove_flat.py
proves its machine code does exactly that over its kernels' results.
`levels_eq_goIdx` proves that after `j` levels the values are the
specification's subtrees (`goIdx`) of 2^j chunks each, in order: with
j = k the whole subtree (the Hasher's `subtree_cv` contract), with
j = k - 1 its two children (the `children` contract). -/

namespace Bridge

/-- One level: adjacent pairs combined. -/
def pairUp {α : Type} (node : α → α → α) : List α → List α
  | a :: b :: rest => node a b :: pairUp node rest
  | l => l

/-- `j` levels of pairs. -/
def levels {α : Type} (node : α → α → α) (l : List α) : Nat → List α
  | 0 => l
  | j + 1 => levels node (pairUp node l) j

theorem log2_two_pow_sub_one (j : Nat) : (2 ^ (j + 1) - 1).log2 = j := by
  apply Nat.log2_eq_iff (by have := Nat.one_le_two_pow (n := j + 1); omega) |>.mpr
  have := Nat.one_le_two_pow (n := j)
  constructor
  · rw [Nat.pow_succ]; omega
  · rw [Nat.pow_succ]; omega

/-- The specification's subtree over 2^(j+1) chunks splits into two of 2^j. -/
theorem goIdx_two_pow {α : Type} (leaf : Nat → α) (node : α → α → α) (c j : Nat) :
    goIdx leaf node c (2 ^ (j + 1)) = node (goIdx leaf node c (2 ^ j)) (goIdx leaf node (c + 2 ^ j) (2 ^ j)) := by
  rw [goIdx]
  have h1 : ¬ 2 ^ (j + 1) ≤ 1 := by have := Nat.one_le_two_pow (n := j); rw [Nat.pow_succ]; omega
  simp only [h1, ↓reduceIte, log2_two_pow_sub_one]
  congr 2
  rw [Nat.pow_succ]; omega

theorem range_map_succ {α : Type} (f : Nat → α) (n : Nat) :
    (List.range (n + 1)).map f = f 0 :: (List.range n).map (fun i => f (i + 1)) := by
  rw [List.range_succ_eq_map, List.map_cons, List.map_map]; rfl

/-- Pairing the subtrees of 2^j chunks gives those of 2^(j+1). -/
theorem pairUp_subtrees {α : Type} (leaf : Nat → α) (node : α → α → α) (j : Nat) :
    ∀ m c, pairUp node ((List.range (2 * m)).map fun i => goIdx leaf node (c + i * 2 ^ j) (2 ^ j)) =
      (List.range m).map fun i => goIdx leaf node (c + i * 2 ^ (j + 1)) (2 ^ (j + 1)) := by
  intro m
  induction m with
  | zero => intro c; simp [pairUp]
  | succ m ih =>
    intro c
    rw [show 2 * (m + 1) = (2 * m + 1) + 1 by omega, range_map_succ, range_map_succ, pairUp, range_map_succ]
    simp only [Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.one_mul]
    rw [goIdx_two_pow leaf node c j]
    congr 1
    have h := ih (c + 2 ^ (j + 1))
    rw [show ((List.range (2 * m)).map fun i => goIdx leaf node (c + (i + 1 + 1) * 2 ^ j) (2 ^ j)) =
        (List.range (2 * m)).map fun i => goIdx leaf node (c + 2 ^ (j + 1) + i * 2 ^ j) (2 ^ j) from
        List.map_congr_left (fun i _ => by congr 1; rw [Nat.pow_succ]; ring), h]
    exact List.map_congr_left (fun i _ => by congr 1; ring)

/-- After `j` levels over the subtrees of 2^i chunks, those of 2^(i+j). -/
theorem levels_subtrees {α : Type} (leaf : Nat → α) (node : α → α → α) :
    ∀ j i m c, levels node ((List.range (2 ^ j * m)).map fun x => goIdx leaf node (c + x * 2 ^ i) (2 ^ i)) j =
      (List.range m).map fun x => goIdx leaf node (c + x * 2 ^ (i + j)) (2 ^ (i + j)) := by
  intro j
  induction j with
  | zero => intro i m c; simp [levels]
  | succ j ih =>
    intro i m c
    rw [levels, show 2 ^ (j + 1) * m = 2 * (2 ^ j * m) by rw [Nat.pow_succ]; ring, pairUp_subtrees, ih]
    rw [show i + 1 + j = i + (j + 1) by omega]

/-- The walk's values after `j` levels over 2^j m chunks from chunk c: the
specification's subtrees of 2^j chunks each, in order. -/
theorem levels_eq_goIdx {α : Type} (leaf : Nat → α) (node : α → α → α) (j m c : Nat) :
    levels node ((List.range (2 ^ j * m)).map fun x => leaf (c + x)) j =
      (List.range m).map fun x => goIdx leaf node (c + x * 2 ^ j) (2 ^ j) := by
  have h := levels_subtrees leaf node j 0 m c
  simp only [Nat.pow_zero, Nat.mul_one, Nat.zero_add] at h
  rw [← h]
  congr 2
  funext x
  rw [goIdx]; simp

#print axioms levels_eq_goIdx

end Bridge
