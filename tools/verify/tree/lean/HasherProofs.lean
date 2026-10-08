import Proofs
open Bridge

namespace HasherProof

/-- The chaining values the Hasher's stack holds after the first `c` chunks
(as `merge_cv_stack` leaves it): one per 1-bit of `c`, the subtree over the
aligned power-of-two block it stands for, largest first. -/
def stackCvs {α : Type} (leaf : Nat → α) (node : α → α → α) (c : Nat) : List α :=
  if c = 0 then []
  else
    let h := 2 ^ c.log2
    goIdx leaf node 0 h :: stackCvs (fun i => leaf (i + h)) node (c - h)
termination_by c
decreasing_by
  have := Nat.one_le_two_pow (n := c.log2)
  omega

/-- The left subtree of `c + 2^k` chunks, `c` a positive multiple of `2^k`,
is `c`'s highest block. -/
theorem split_of_aligned (c k : Nat) (hc : 0 < c) (hk : 2 ^ k ∣ c) :
    2 ^ (c + 2 ^ k - 1).log2 = 2 ^ c.log2 := by
  congr 1
  rw [Nat.log2_eq_iff (by have := Nat.one_le_two_pow (n := k); omega)]
  have h1 := Nat.log2_self_le (n := c) (by omega)
  have h2 := Nat.lt_log2_self (n := c)
  have hkc : 2 ^ k ≤ c := Nat.le_of_dvd hc hk
  have hkh : k ≤ c.log2 := by
    by_contra hn
    have : 2 ^ (c.log2 + 1) ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  -- c is below the multiple of 2^k that is 2^(log2 c + 1), so at most 2^k under it
  obtain ⟨q, hq⟩ := hk
  have hr : 2 ^ (c.log2 + 1) = 2 ^ k * 2 ^ (c.log2 + 1 - k) := by
    rw [← pow_add]; congr 1; omega
  have hq' : q < 2 ^ (c.log2 + 1 - k) := by
    by_contra hn
    have : 2 ^ k * 2 ^ (c.log2 + 1 - k) ≤ 2 ^ k * q := Nat.mul_le_mul_left _ (by omega)
    omega
  have : 2 ^ k * q + 2 ^ k ≤ 2 ^ k * 2 ^ (c.log2 + 1 - k) := by
    rw [← Nat.mul_succ]; exact Nat.mul_le_mul_left _ hq'
  rw [← hq, ← hr] at this
  have := Nat.one_le_two_pow (n := k)
  constructor <;> omega

/-- Finalization: the Hasher folds its stack, from the top, onto the last
aligned block's chaining value; that is the specification's tree over all
the chunks. -/
theorem root_of_stack {α : Type} (leaf : Nat → α) (node : α → α → α) (c k : Nat) (hk : 2 ^ k ∣ c) :
    goIdx leaf node 0 (c + 2 ^ k) = (stackCvs leaf node c).foldr node (goIdx leaf node c (2 ^ k)) := by
  induction c using Nat.strong_induction_on generalizing leaf with
  | _ c ih =>
  by_cases hc : c = 0
  · subst hc; simp [stackCvs]
  · rw [stackCvs, if_neg hc]
    have hsplit := split_of_aligned c k (by omega) hk
    have h1 := Nat.log2_self_le (n := c) (by omega)
    have hp := Nat.one_le_two_pow (n := c.log2)
    have hkc : 2 ^ k ≤ c := Nat.le_of_dvd (by omega) hk
    dsimp only
    rw [List.foldr_cons, goIdx]
    simp only [show ¬ c + 2 ^ k ≤ 1 by have := Nat.one_le_two_pow (n := k); omega, ↓reduceIte, hsplit]
    congr 1
    -- the rest: the block after c's highest, shifted to start at 0
    have hdvd : 2 ^ k ∣ c - 2 ^ c.log2 := by
      apply Nat.dvd_sub hk
      exact Nat.pow_dvd_pow 2 (by
        by_contra hn
        have : 2 ^ (c.log2 + 1) ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
        have := Nat.lt_log2_self (n := c)
        omega)
    have hih := ih (c - 2 ^ c.log2) (by omega) (fun i => leaf (i + 2 ^ c.log2)) hdvd
    have e1 := goIdx_shift leaf node 0 (2 ^ c.log2) (c + 2 ^ k - 2 ^ c.log2)
    have e2 := goIdx_shift leaf node (c - 2 ^ c.log2) (2 ^ c.log2) (2 ^ k)

    rw [show c - 2 ^ c.log2 + 2 ^ c.log2 = c by omega] at e2
    rw [e1, e2, show c + 2 ^ k - 2 ^ c.log2 = (c - 2 ^ c.log2) + 2 ^ k by omega, hih]

/-- The number of 1-bits of `c`, as stackCvs counts its blocks. -/
def ones (c : Nat) : Nat :=
  if c = 0 then 0 else 1 + ones (c - 2 ^ c.log2)
termination_by c
decreasing_by
  have := Nat.one_le_two_pow (n := c.log2)
  omega

theorem stackCvs_length {α : Type} (leaf : Nat → α) (node : α → α → α) (c : Nat) :
    (stackCvs leaf node c).length = ones c := by
  induction c using Nat.strong_induction_on generalizing leaf with
  | _ c ih =>
  rw [stackCvs, ones]
  split
  · rfl
  · have := Nat.one_le_two_pow (n := c.log2)
    have := Nat.log2_self_le (n := c) (by omega)
    simp only [List.length_cons]
    rw [ih _ (by omega)]; omega

/-- `merge_cv_stack`, on the stack top first: while it holds more than
`target` values (and two at least), the top two become their parent, the
lower one its left child. -/
def merge {α : Type} (node : α → α → α) (target : Nat) : List α → List α
  | b :: a :: rest => if target < rest.length + 2 then merge node target (node a b :: rest) else b :: a :: rest
  | l => l
termination_by l => l.length

/-- Merging leaves the stack's bottom value, when one value at least stays above it. -/
theorem merge_append {α : Type} (node : α → α → α) (p : Nat) (hp : 1 ≤ p) (rest : List α) (x : α) :
    merge node (p + 1) (rest ++ [x]) = merge node p rest ++ [x] := by
  induction rest using merge.induct node p with
  | case1 b a r h ih =>
    rw [List.cons_append, List.cons_append, merge, if_pos (by simp; omega), merge, if_pos h, ← ih,
      List.cons_append]
  | case2 b a r h =>
    rw [List.cons_append, List.cons_append, merge, if_neg (by simp; omega), merge, if_neg h]
    simp
  | case3 l hl =>
    match l, hl with
    | [], _ => simp [merge]
    | [b], _ => rw [List.cons_append, List.nil_append, merge, if_neg (by simp; try omega)]; simp [merge]
    | _ :: _ :: _, h => exact absurd rfl (h _ _ _)

/-- Merging to one value: the values above the bottom merged to one, and
the bottom its left sibling. -/
theorem merge_one {α : Type} (node : α → α → α) (rest : List α) (x r : α) (hr : merge node 1 rest = [r]) :
    merge node 1 (rest ++ [x]) = [node x r] := by
  induction rest using merge.induct node 1 generalizing r with
  | case1 b a t h ih =>
    rw [merge, if_pos h] at hr
    rw [List.cons_append, List.cons_append, merge, if_pos (by simp), ← List.cons_append]
    exact ih r hr
  | case2 b a t h => exact absurd (by omega) h
  | case3 l hl =>
    match l, hl with
    | [], _ => simp [merge] at hr
    | [b], _ =>
      simp only [merge] at hr
      injection hr with hr; subst hr
      rw [List.cons_append, List.nil_append, merge, if_pos (by simp)]
      simp [merge]
    | _ :: _ :: _, h => exact absurd rfl (h _ _ _)

theorem log2_add_of_lt (m s : Nat) (hs : s < 2 ^ m) : (2 ^ m + s).log2 = m := by
  rw [Nat.log2_eq_iff (by have := Nat.one_le_two_pow (n := m); omega)]
  constructor
  · omega
  · rw [pow_succ]; omega

theorem ones_add (m s : Nat) (hs : s < 2 ^ m) : ones (2 ^ m + s) = 1 + ones s := by
  rw [ones, if_neg (by have := Nat.one_le_two_pow (n := m); omega), log2_add_of_lt m s hs,
    show 2 ^ m + s - 2 ^ m = s by omega]

theorem ones_pos (c : Nat) (h : 0 < c) : 1 ≤ ones c := by
  rw [ones, if_neg (by omega)]; omega

theorem stackCvs_add {α : Type} (leaf : Nat → α) (node : α → α → α) (m s : Nat) (hs : s < 2 ^ m) :
    stackCvs leaf node (2 ^ m + s) = goIdx leaf node 0 (2 ^ m) :: stackCvs (fun i => leaf (i + 2 ^ m)) node s := by
  rw [stackCvs, if_neg (by have := Nat.one_le_two_pow (n := m); omega)]
  dsimp only
  rw [log2_add_of_lt m s hs, show 2 ^ m + s - 2 ^ m = s by omega]

/-- Aligned multiples: below a power of two that 2^k divides, a multiple of
2^k leaves room for another 2^k. -/
theorem add_le_of_dvd (r h k : Nat) (hr : 2 ^ k ∣ r) (hh : 2 ^ k ∣ h) (hlt : r < h) : r + 2 ^ k ≤ h := by
  obtain ⟨a, rfl⟩ := hr
  obtain ⟨b, rfl⟩ := hh
  have : a < b := by
    by_contra hn
    have : 2 ^ k * b ≤ 2 ^ k * a := Nat.mul_le_mul_left _ (by omega)
    omega
  have : 2 ^ k * (a + 1) ≤ 2 ^ k * b := Nat.mul_le_mul_left _ (by omega)
  rw [Nat.mul_succ] at this
  exact this

/-- `push_cv`: before the chaining value of the aligned block of 2^k chunks
at chunk c goes on the stack, `merge_cv_stack(c)` leaves c's stack; after
it, merging to the 1-bits of c + 2^k leaves the stack of c + 2^k. (The stack
top first.) -/
theorem merge_push {α : Type} (leaf : Nat → α) (node : α → α → α) (c k : Nat) (hk : 2 ^ k ∣ c) :
    merge node (ones (c + 2 ^ k)) (goIdx leaf node c (2 ^ k) :: (stackCvs leaf node c).reverse) =
      (stackCvs leaf node (c + 2 ^ k)).reverse := by
  induction c using Nat.strong_induction_on generalizing leaf with
  | _ c ih =>
  by_cases hc : c = 0
  · subst hc
    have := stackCvs_add leaf node k 0 (Nat.one_le_two_pow)
    simp only [Nat.add_zero] at this
    rw [Nat.zero_add, this, stackCvs, if_pos rfl]
    simp [stackCvs, merge]
  · -- c = h + r, h its highest block
    have h1 := Nat.log2_self_le (n := c) (by omega)
    have h2 := Nat.lt_log2_self (n := c)
    rw [pow_succ] at h2
    have hkc : 2 ^ k ≤ c := Nat.le_of_dvd (by omega) hk
    have hkm : k ≤ c.log2 := by
      by_contra hn
      have : 2 ^ (c.log2 + 1) ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
      rw [pow_succ] at this; omega
    generalize hm : c.log2 = m at *
    have hhk : 2 ^ k ∣ 2 ^ m := Nat.pow_dvd_pow 2 hkm
    have hr : 2 ^ k ∣ c - 2 ^ m := Nat.dvd_sub hk hhk
    have hle := add_le_of_dvd (c - 2 ^ m) (2 ^ m) k hr hhk (by omega)
    have hcs := stackCvs_add leaf node m (c - 2 ^ m) (by omega)
    rw [show 2 ^ m + (c - 2 ^ m) = c by omega] at hcs
    have hx := goIdx_shift leaf node (c - 2 ^ m) (2 ^ m) (2 ^ k)
    rw [show c - 2 ^ m + 2 ^ m = c by omega] at hx
    have hih := ih (c - 2 ^ m) (by omega) (fun i => leaf (i + 2 ^ m)) hr
    rw [hcs, hx, List.reverse_cons, ← List.cons_append]
    by_cases hlt : c - 2 ^ m + 2 ^ k < 2 ^ m
    · -- no carry into a new highest bit: the bottom block stays
      rw [show c + 2 ^ k = 2 ^ m + (c - 2 ^ m + 2 ^ k) by omega, ones_add m _ hlt,
        Nat.add_comm 1, merge_append node _ (ones_pos _ (by have := Nat.one_le_two_pow (n := k); omega)),
        hih, stackCvs_add leaf node m _ hlt, List.reverse_cons]
    · -- the carry makes one block of 2^(m + 1)
      have heq : c - 2 ^ m + 2 ^ k = 2 ^ m := by omega
      rw [heq] at hih
      have hsm := stackCvs_add (fun i => leaf (i + 2 ^ m)) node m 0 (Nat.one_le_two_pow)
      rw [Nat.add_zero, show stackCvs (fun i => leaf (i + 2 ^ m + 2 ^ m)) node 0 = [] by simp [stackCvs]] at hsm
      have hones : ones (2 ^ m) = 1 := by
        have := ones_add m 0 (Nat.one_le_two_pow); simp [ones] at this ⊢; try omega
      rw [hones, hsm] at hih
      simp only [List.reverse_singleton] at hih
      have htop : c + 2 ^ k = 2 ^ (m + 1) := by rw [pow_succ]; omega
      have hones2 : ones (2 ^ (m + 1)) = 1 := by
        have := ones_add (m + 1) 0 (Nat.one_le_two_pow); simp [ones] at this ⊢; try omega
      rw [htop, hones2, merge_one node _ _ _ hih]
      have hs2 := stackCvs_add leaf node (m + 1) 0 (Nat.one_le_two_pow)
      rw [Nat.add_zero, show stackCvs (fun i => leaf (i + 2 ^ (m + 1))) node 0 = [] by simp [stackCvs]] at hs2
      rw [hs2, List.reverse_singleton]
      -- the tree over 2^(m + 1) chunks: the parent of its two halves
      congr 1
      conv_rhs => rw [goIdx]
      have hsplit : 2 ^ (2 ^ (m + 1) - 1).log2 = 2 ^ m := by
        congr 1; rw [Nat.log2_eq_iff (by have := Nat.one_le_two_pow (n := m); rw [pow_succ]; omega)]
        rw [pow_succ]; have := Nat.one_le_two_pow (n := m); constructor <;> omega
      simp only [show ¬ 2 ^ (m + 1) ≤ 1 by have := Nat.one_le_two_pow (n := m); rw [pow_succ]; omega, ↓reduceIte, hsplit,
        show 2 ^ (m + 1) - 2 ^ m = 2 ^ m by rw [pow_succ]; omega]
      rw [goIdx_shift leaf node 0 (2 ^ m) (2 ^ m)]

/-- `final_output` with no partial chunk: the stack top first, after the last
push of the aligned block of 2^k chunks at chunk c. Its output starts as the
parent of the top two values and takes each lower value as its left
sibling: that is the specification's tree over the c + 2^k chunks. -/
theorem final_output_whole {α : Type} (leaf : Nat → α) (node : α → α → α) (c k : Nat) (hk : 2 ^ k ∣ c) :
    ((stackCvs leaf node c).reverse).foldl (fun acc s => node s acc) (goIdx leaf node c (2 ^ k)) =
      goIdx leaf node 0 (c + 2 ^ k) := by
  rw [List.foldl_reverse, root_of_stack leaf node c k hk]

/-- `final_output` with a partial chunk, number c: the stack merged to c's
(as `update` leaves it), then each value the left sibling of the output, from
the chunk's: the specification's tree over the c + 1 chunks. -/
theorem final_output_partial {α : Type} (leaf : Nat → α) (node : α → α → α) (c : Nat) :
    ((stackCvs leaf node c).reverse).foldl (fun acc s => node s acc) (leaf c) =
      goIdx leaf node 0 (c + 1) := by
  have := final_output_whole leaf node c 0 (by simp)
  simp only [pow_zero] at this
  rw [← this]
  congr 1
  rw [goIdx, if_pos (le_refl 1)]

end HasherProof

#print axioms HasherProof.root_of_stack
#print axioms HasherProof.merge_push
#print axioms HasherProof.final_output_whole
#print axioms HasherProof.final_output_partial
