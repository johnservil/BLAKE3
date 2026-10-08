import Treecore
import Blake3
open Aeneas Aeneas.Std Result

namespace treecore

/-- The loop doubles `p`, a power of two, while `p ≤ chunks / 2`: it ends at
the largest power of two at most `chunks`. -/
theorem left_len_loop_spec (chunks p : Std.Usize) (hc : 0 < chunks.val)
    (hp : ∃ j, p.val = 2 ^ j) (hle : p.val ≤ 2 ^ chunks.val.log2) :
    left_len_loop chunks p ⦃ r => r.val = 2 ^ chunks.val.log2 ⦄ := by
  unfold left_len_loop
  apply loop.spec_decr_nat (fun p : Std.Usize => chunks.val - p.val)
    (fun p : Std.Usize => (∃ j, p.val = 2 ^ j) ∧ p.val ≤ 2 ^ chunks.val.log2)
  · rintro p ⟨⟨j, hj⟩, hle⟩
    unfold left_len_loop.body
    have hlog := Nat.log2_self_le (n := chunks.val) (by omega)
    have hlt := Nat.lt_log2_self (n := chunks.val)
    have hj1 : 1 ≤ 2 ^ j := Nat.one_le_two_pow
    step*
    · -- the step: p1 = 2 p is the next power of two, still at most 2^log2
      have hpi : p.val ≤ chunks.val / 2 := by scalar_tac
      have h2 : 2 ^ (j + 1) < 2 ^ (chunks.val.log2 + 1) := by rw [pow_succ]; omega
      have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp h2
      refine ⟨⟨j + 1, by rw [p1_post, hj, pow_succ]⟩, ?_, by omega⟩
      rw [p1_post, hj, ← pow_succ]
      exact Nat.pow_le_pow_right (by norm_num) (by omega)
    · -- the exit: p > chunks / 2, so p is no smaller than 2^log2
      have hpi : chunks.val / 2 < p.val := by scalar_tac
      have hjl : j ≤ chunks.val.log2 := by
        rw [hj] at hle; exact (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mp hle
      rcases Nat.lt_or_ge j chunks.val.log2 with h | h
      · have := Nat.pow_le_pow_right (n := 2) (by norm_num) h
        rw [pow_succ] at this; omega
      · rw [hj]; congr 1; omega
  · exact ⟨hp, hle⟩

/-- `left_len` of an input longer than one chunk: 1024 times the largest
power of two at most its number of chunks before the last. -/
@[step]
theorem left_len_spec (len : Std.Usize) (h : 1024 < len.val) :
    left_len len ⦃ r => r.val = 1024 * 2 ^ ((len.val - 1) / 1024).log2 ⦄ := by
  unfold left_len
  have hc : 0 < (len.val - 1) / 1024 := by omega
  have hlog := Nat.log2_self_le (n := (len.val - 1) / 1024) (by omega)
  have hk : CHUNK_LEN.val = 1024 := by simp [CHUNK_LEN]
  step*
  have hchunks : chunks.val = (len.val - 1) / 1024 := by rw [chunks_post, i_post, hk]
  step with left_len_loop_spec
  · exact ⟨0, by simp⟩
  · simp
  -- p * CHUNK_LEN cannot overflow: p ≤ chunks, so p * 1024 ≤ len - 1.
  rw [hchunks] at p_post
  have : p.val * 1024 ≤ len.val - 1 := by rw [p_post]; omega
  step*

/-- The split point lies inside the input, after its first chunk. -/
theorem split_bounds (L : Nat) (h : 1024 < L) :
    1024 ≤ 1024 * 2 ^ ((L - 1) / 1024).log2 ∧ 1024 * 2 ^ ((L - 1) / 1024).log2 ≤ L - 1 := by
  have := Nat.log2_self_le (n := (L - 1) / 1024) (by omega)
  have := Nat.one_le_two_pow (n := ((L - 1) / 1024).log2)
  omega

abbrev Cv := Std.Array Std.U32 8#usize

/-- The walk's recursion on the input's bytes, its kernels as functions:
`chunk bytes counter` and `parent left right`. -/
def walk {α : Type} (chunk : List Std.U8 → Nat → α) (parent : α → α → α)
    (bytes : List Std.U8) (counter : Nat) : α :=
  if h : bytes.length ≤ 1024 then chunk bytes counter
  else
    let l := 1024 * 2 ^ ((bytes.length - 1) / 1024).log2
    parent (walk chunk parent (bytes.take l) counter)
      (walk chunk parent (bytes.drop l) (counter + l / 1024))
termination_by bytes.length
decreasing_by
  all_goals
    have := Nat.log2_self_le (n := (bytes.length - 1) / 1024) (by omega)
    have := Nat.one_le_two_pow (n := ((bytes.length - 1) / 1024).log2)
    simp [List.length_take, List.length_drop]
    omega

/-- `subtree` is `walk`, for kernels that are the given functions, on every
input whose chunk numbers fit in 64 bits. -/
theorem subtree_spec {K : Type} (KernelsInst : Kernels K) (k : K)
    (chunk : List Std.U8 → Nat → Cv) (parent : Cv → Cv → Cv)
    (hchunk : ∀ s c, KernelsInst.chunk k s c = ok (chunk s.val c.val))
    (hparent : ∀ a b, KernelsInst.parent k a b = ok (parent a b))
    (input : Slice Std.U8) (counter : Std.U64)
    (hfit : counter.val + input.length / 1024 + 1 ≤ Std.U64.max) :
    subtree KernelsInst k input counter ⦃ r => r = walk chunk parent input.val counter.val ⦄ := by
  induction hn : input.length using Nat.strong_induction_on generalizing input counter with
  | _ n ih =>
  unfold subtree
  rw [walk]
  have hk : CHUNK_LEN.val = 1024 := by simp [CHUNK_LEN]
  by_cases hs : input.length ≤ 1024
  · have hle : input.len ≤ CHUNK_LEN := by scalar_tac
    have hval : input.val.length ≤ 1024 := by simpa [Slice.length] using hs
    simp [hle, hval, hchunk]
  · have hle : ¬ input.len ≤ CHUNK_LEN := by scalar_tac
    have hval : ¬ input.val.length ≤ 1024 := by simpa [Slice.length] using hs
    simp only [hle, hval, ↓reduceIte, ↓reduceDIte]
    have hlen : input.len.val = input.length := by simp
    have hb := split_bounds input.length (by omega)
    step*

end treecore

open treecore

namespace Bridge

/-- The specification's recursion over chunk indexes (C2SP BLAKE3, section
4.3.2, `tree.go`), for any leaf and parent functions: the subtree over the
`n ≥ 1` chunks from `start`, its left subtree `2 ^ log2 (n - 1)` chunks. -/
def goIdx {α : Type} (leaf : Nat → α) (node : α → α → α) (start n : Nat) : α :=
  if n ≤ 1 then leaf start
  else
    let k := 2 ^ (n - 1).log2
    node (goIdx leaf node start k) (goIdx leaf node (start + k) (n - k))
termination_by n
decreasing_by
  all_goals
    have := Nat.log2_self_le (n := n - 1) (by omega)
    have := Nat.one_le_two_pow (n := (n - 1).log2)
    omega

/-- goIdx reads its leaf only at the indexes of its chunks. -/
theorem goIdx_congr {α : Type} (leaf leaf' : Nat → α) (node : α → α → α) (start n : Nat)
    (h : ∀ i, start ≤ i → i < start + n → leaf i = leaf' i) (hn : 1 ≤ n) :
    goIdx leaf node start n = goIdx leaf' node start n := by
  induction n using Nat.strong_induction_on generalizing start with
  | _ n ih =>
  rw [goIdx, goIdx]
  split
  · exact h start le_rfl (by omega)
  · have := Nat.log2_self_le (n := n - 1) (by omega)
    have := Nat.one_le_two_pow (n := (n - 1).log2)
    dsimp only
    rw [ih (2 ^ (n - 1).log2) (by omega) start (fun i h1 h2 => h i h1 (by omega)) (by omega),
        ih (n - 2 ^ (n - 1).log2) (by omega) (start + 2 ^ (n - 1).log2)
          (fun i h1 h2 => h i (by omega) (by omega)) (by omega)]

/-- Chunk `i` of bytes whose first chunk is chunk `c`. -/
def chunkOf (bytes : List Std.U8) (c i : Nat) : List Std.U8 :=
  (bytes.drop (1024 * (i - c))).take 1024

/-- The number of chunks of a nonempty input. -/
def chunks (len : Nat) : Nat := (len + 1023) / 1024

/-- walk is the specification's recursion over the input's chunks. -/
theorem walk_eq_goIdx {α : Type} (chunk : List Std.U8 → Nat → α) (parent : α → α → α)
    (bytes : List Std.U8) (c : Nat) (hne : bytes ≠ []) :
    walk chunk parent bytes c =
      goIdx (fun i => chunk (chunkOf bytes c i) i) parent c (chunks bytes.length) := by
  induction hn : bytes.length using Nat.strong_induction_on generalizing bytes c with
  | _ n ih =>
  have hpos : 0 < bytes.length := List.length_pos_iff.mpr hne
  rw [walk, goIdx]
  by_cases hs : bytes.length ≤ 1024
  · have : chunks n ≤ 1 := by unfold chunks; omega
    simp only [hs, ↓reduceDIte, this, ↓reduceIte, chunkOf, Nat.sub_self, Nat.mul_zero, List.drop_zero]
    rw [List.take_of_length_le hs]
  · have hc : chunks n - 1 = (n - 1) / 1024 := by unfold chunks; omega
    have hc2 : ¬ chunks n ≤ 1 := by unfold chunks; omega
    have hb := split_bounds n (by omega)
    have hs' : ¬ n ≤ 1024 := by omega
    simp only [hn, hs', ↓reduceDIte, hc2, ↓reduceIte, hc]
    have hK := Nat.one_le_two_pow (n := ((n - 1) / 1024).log2)
    have hKle := Nat.log2_self_le (n := (n - 1) / 1024) (by omega)
    generalize hKdef : 2 ^ ((n - 1) / 1024).log2 = K at *
    have hdiv : 1024 * K / 1024 = K := by omega
    rw [hdiv]
    -- The left half: the first 1024 K bytes, K chunks.
    have htake_len : (bytes.take (1024 * K)).length = 1024 * K := by simp; omega
    have hleft := ih (1024 * K) (by omega) (bytes.take (1024 * K)) c
      (by intro h; rw [h] at htake_len; simp at htake_len; omega) htake_len
    have hkc : chunks (1024 * K) = K := by simp only [chunks]; omega
    rw [hkc] at hleft
    -- The right half: the bytes after them, chunks n - K chunks from chunk c + K.
    have hdrop_len : (bytes.drop (1024 * K)).length = n - 1024 * K := by simp; omega
    have hright := ih (n - 1024 * K) (by omega) (bytes.drop (1024 * K)) (c + K)
      (by intro h; rw [h] at hdrop_len; simp at hdrop_len; omega) hdrop_len
    have hrc : chunks (n - 1024 * K) = chunks n - K := by simp only [chunks] at hc ⊢; omega
    rw [hrc] at hright
    rw [hleft, hright]
    congr 1
    · apply goIdx_congr _ _ _ _ _ _ (by omega)
      intro i h1 h2
      simp only [chunkOf, List.drop_take, List.take_take]
      congr 2
      omega
    · apply goIdx_congr _ _ _ _ _ _ (by omega)
      intro i h1 h2
      simp only [chunkOf, List.drop_drop]
      congr 3
      omega

/-- walk commutes with an abstraction of its values that commutes with
its parent function. -/
theorem walk_map {α β : Type} (abs : α → β) (chunk : List Std.U8 → Nat → α) (parent : α → α → α)
    (P : β → β → β) (hp : ∀ a b, abs (parent a b) = P (abs a) (abs b))
    (bytes : List Std.U8) (c : Nat) :
    abs (walk chunk parent bytes c) = walk (fun b i => abs (chunk b i)) P bytes c := by
  induction hn : bytes.length using Nat.strong_induction_on generalizing bytes c with
  | _ n ih =>
  rw [walk, walk]
  split
  · rfl
  · have := split_bounds bytes.length (by omega)
    dsimp only
    rw [hp, ih _ (by simp; omega) _ _ rfl, ih _ (by simp; omega) _ _ rfl]

/-- goIdx over the specification's chunk nodes is its tree's subtree. -/
theorem goIdx_tree (key : Vector Blake3.Word 8) (mode : Blake3.Word) (chunks : Array Blake3.Node)
    (start n : Nat) (hn : 1 ≤ n) :
    (Blake3.tree.go key mode chunks start n).cv =
      goIdx (fun i => chunks[i]!.cv) (fun a b => (Blake3.parentNode key mode a b).cv) start n := by
  induction n using Nat.strong_induction_on generalizing start with
  | _ n ih =>
  rw [Blake3.tree.go, goIdx]
  by_cases h : n ≤ 1
  · simp [h]
  · have := Nat.log2_self_le (n := n - 1) (by omega)
    have := Nat.one_le_two_pow (n := (n - 1).log2)
    simp only [h, ↓reduceDIte, ↓reduceIte, Blake3.leftCount]
    rw [ih _ (by omega) _ (by omega), ih _ (by omega) _ (by omega)]

/-- Aeneas's bytes as the specification's. -/
def toBA (l : List Std.U8) : ByteArray := ⟨(l.map fun b => UInt8.ofBitVec b.bv).toArray⟩

theorem toBA_extract (l : List Std.U8) (b e : Nat) :
    (toBA l).extract b e = toBA ((l.drop b).take (e - b)) := by
  have : ∀ x y : ByteArray, x.data.toList = y.data.toList → x = y := by
    intro x y h; cases x; cases y; simp_all [Array.toList_inj]
  apply this
  simp [toBA, ByteArray.data_extract, List.extract_eq_take_drop, List.map_take,
    List.map_drop]

theorem toBA_size (l : List Std.U8) : (toBA l).size = l.length := by
  simp [toBA, ByteArray.size]

/-- The walk over a nonempty input, with the specification's chunk and parent
functions, is the chaining value of the specification's tree over that
input's chunks. -/
theorem walk_is_tree (key : Vector Blake3.Word 8) (mode : Blake3.Word)
    (bytes : List Std.U8) (hne : bytes ≠ []) :
    walk (fun b i => (Blake3.chunkNode key mode (toBA b) i).cv)
        (fun a b => (Blake3.parentNode key mode a b).cv) bytes 0 =
      (Blake3.tree key mode ((Array.range (max 1 (((toBA bytes).size + 1023) / 1024))).map fun i =>
        Blake3.chunkNode key mode ((toBA bytes).extract (1024 * i) (1024 * (i + 1))) i)).cv := by
  have hpos : 0 < bytes.length := List.length_pos_iff.mpr hne
  rw [walk_eq_goIdx _ _ _ _ hne, Blake3.tree, goIdx_tree _ _ _ _ _ (by simp [toBA_size])]
  simp only [Array.size_map, Array.size_range, toBA_size]
  have hcount : max 1 ((bytes.length + 1023) / 1024) = chunks bytes.length := by unfold chunks; omega
  rw [hcount]
  apply goIdx_congr _ _ _ _ _ _ (by unfold chunks; omega)
  intro i _ hi
  simp only [Nat.zero_add] at hi
  rw [getElem!_pos _ i (by simp; omega)]
  simp only [Array.getElem_map, Array.getElem_range, chunkOf, Nat.sub_zero, toBA_extract]
  rw [show 1024 * (i + 1) - 1024 * i = 1024 by omega]

/-- The safe-Rust tree walk computes the C2SP specification's tree, for
kernels that meet their contract: under an abstraction `abs` of chaining
values, `chunk` gives the specification's chunk chaining value and
`parent` its parent chaining value. Every nonempty input whose chunk count
fits in 64 bits: no panic, no overflow, and the specification's result. -/
theorem subtree_is_tree {K : Type} (KernelsInst : Kernels K) (k : K)
    (key : Vector Blake3.Word 8) (mode : Blake3.Word) (abs : Cv → Vector Blake3.Word 8)
    (chunkf : List Std.U8 → Nat → Cv) (parentf : Cv → Cv → Cv)
    (hchunk : ∀ s c, KernelsInst.chunk k s c = ok (chunkf s.val c.val))
    (hparent : ∀ a b, KernelsInst.parent k a b = ok (parentf a b))
    (hchunk_spec : ∀ b i, abs (chunkf b i) = (Blake3.chunkNode key mode (toBA b) i).cv)
    (hparent_spec : ∀ a b, abs (parentf a b) = (Blake3.parentNode key mode (abs a) (abs b)).cv)
    (input : Slice Std.U8) (hne : input.val ≠ [])
    (hfit : input.length / 1024 + 1 ≤ Std.U64.max) :
    subtree KernelsInst k input 0#u64 ⦃ r =>
      abs r = (Blake3.tree key mode ((Array.range (max 1 (((toBA input.val).size + 1023) / 1024))).map
        fun i => Blake3.chunkNode key mode ((toBA input.val).extract (1024 * i) (1024 * (i + 1))) i)).cv ⦄ := by
  apply WP.spec_mono (subtree_spec KernelsInst k chunkf parentf hchunk hparent input 0#u64 (by simpa using hfit))
  intro r hr
  rw [hr, walk_map abs chunkf parentf (fun a b => (Blake3.parentNode key mode a b).cv) hparent_spec]
  simp only [hchunk_spec]
  exact walk_is_tree key mode input.val hne

/-! ## One layer of parents

The library's walk computes chaining values a layer at a time: each pair
of neighbours becomes their parent, an odd last one passes through. The
specification's tree over a layer is its tree over the layer below. -/

/-- The layer above leaves `f` that end before index `e`: leaf i of the
layer is the parent of leaves 2i and 2i + 1, or leaf 2i alone when it is
the last. -/
def lay {α : Type} (f : Nat → α) (node : α → α → α) (e : Nat) (i : Nat) : α :=
  if 2 * i + 1 < e then node (f (2 * i)) (f (2 * i + 1)) else f (2 * i)

/-- The left subtree of a layer is half the left subtree below it. -/
theorem split_half (n : Nat) (h : 3 ≤ n) :
    2 ^ ((n + 1) / 2 - 1).log2 * 2 = 2 ^ (n - 1).log2 := by
  rw [Nat.log2_def (n - 1), if_pos (by omega), show (n + 1) / 2 - 1 = (n - 1) / 2 by omega, pow_succ]

theorem goIdx_lay {α : Type} (f : Nat → α) (node : α → α → α) (n : Nat) (hn : 2 ≤ n) (s : Nat) :
    goIdx f node (2 * s) n = goIdx (lay f node (2 * s + n)) node s ((n + 1) / 2) := by
  induction n using Nat.strong_induction_on generalizing s with
  | _ n ih =>
  by_cases h2 : n = 2
  · subst h2
    rw [goIdx, goIdx]
    simp [goIdx, lay]
  · have hn3 : 3 ≤ n := by omega
    have hhalf := split_half n hn3
    have hlog := Nat.log2_self_le (n := n - 1) (by omega)
    generalize hk : 2 ^ (n - 1).log2 = k at *
    generalize hk' : 2 ^ ((n + 1) / 2 - 1).log2 = k' at *
    have hk'1 : 1 ≤ k' := by rw [← hk']; exact Nat.one_le_two_pow
    rw [goIdx, goIdx]
    simp only [show ¬ n ≤ 1 by omega, show ¬ (n + 1) / 2 ≤ 1 by omega, ↓reduceIte]
    rw [hk, hk']
    congr 1
    · -- the left subtree: k = 2 k' leaves, k' pairs
      rw [ih k (by omega) (by omega) s, show (k + 1) / 2 = k' by omega]
      apply goIdx_congr _ _ _ _ _ _ (by omega)
      intro i _ _
      simp only [lay]
      rw [if_pos (by omega), if_pos (by omega)]
    · -- the right subtree: n - k leaves from 2 (s + k')
      rw [show 2 * s + k = 2 * (s + k') by omega]
      by_cases h1 : n - k = 1
      · rw [h1, show (n + 1) / 2 - k' = 1 by omega, goIdx, goIdx]
        simp only [le_refl, ↓reduceIte, lay]
        rw [if_neg (by omega)]
      · rw [ih (n - k) (by omega) (by omega) (s + k'), show 2 * (s + k') + (n - k) = 2 * s + n by omega,
            show (n - k + 1) / 2 = (n + 1) / 2 - k' by omega]

/-- One layer of parents over a list. -/
def layerL {α : Type} (node : α → α → α) : List α → List α
  | a :: b :: rest => node a b :: layerL node rest
  | l => l

theorem layerL_length {α : Type} (node : α → α → α) (l : List α) :
    (layerL node l).length = (l.length + 1) / 2 := by
  induction l using layerL.induct with
  | case1 a b rest ih => simp [layerL, ih]; omega
  | case2 l h => cases l with
    | nil => simp [layerL]
    | cons a t => cases t with
      | nil => simp [layerL]
      | cons b r => exact absurd rfl (h a b r)

theorem layerL_getD {α : Type} [Inhabited α] (node : α → α → α) (l : List α) (i : Nat)
    (hi : i < (l.length + 1) / 2) :
    (layerL node l).getD i default = lay (fun j => l.getD j default) node l.length i := by
  induction l using layerL.induct generalizing i with
  | case1 a b rest ih =>
    cases i with
    | zero => simp [layerL, lay]
    | succ i =>
      simp only [layerL, List.getD_cons_succ, List.length_cons] at hi ⊢
      rw [ih i (by omega)]
      simp only [lay]
      rw [show 2 * (i + 1) = 2 * i + 2 by ring, show 2 * i + 2 + 1 = 2 * i + 1 + 2 by ring]
      simp only [List.getD_cons_succ]
      split <;> split <;> first | rfl | omega
  | case2 l h => cases l with
    | nil => simp at hi
    | cons a t => cases t with
      | nil => simp at hi; subst hi; simp [layerL, lay]
      | cons b r => exact absurd rfl (h a b r)

/-- The specification's tree over a list of leaves. -/
def treeL {α : Type} [Inhabited α] (node : α → α → α) (l : List α) : α :=
  goIdx (fun i => l.getD i default) node 0 l.length

/-- A layer of parents keeps the tree. -/
theorem treeL_layerL {α : Type} [Inhabited α] (node : α → α → α) (l : List α) (h : 2 ≤ l.length) :
    treeL node (layerL node l) = treeL node l := by
  unfold treeL
  rw [goIdx_lay _ node l.length h 0, layerL_length]
  simp only [Nat.mul_zero, Nat.zero_add]
  apply goIdx_congr _ _ _ _ _ _ (by omega)
  intro i _ hi
  exact layerL_getD node l i (by omega)

/-- goIdx depends on its start only through its leaves. -/
theorem goIdx_shift {α : Type} (f : Nat → α) (node : α → α → α) (start t n : Nat) :
    goIdx f node (start + t) n = goIdx (fun i => f (i + t)) node start n := by
  induction n using Nat.strong_induction_on generalizing start with
  | _ n ih =>
  rw [goIdx, goIdx]
  split
  · rfl
  · have := Nat.log2_self_le (n := n - 1) (by omega)
    have := Nat.one_le_two_pow (n := (n - 1).log2)
    dsimp only
    rw [ih _ (by omega) start, show start + t + 2 ^ (n - 1).log2 = (start + 2 ^ (n - 1).log2) + t by omega,
      ih _ (by omega) (start + 2 ^ (n - 1).log2)]

/-- The tree over a list whose first part is a power of two at least the
rest: the parent of the parts' trees. -/
theorem treeL_append {α : Type} [Inhabited α] (node : α → α → α) (a b : List α) (j : Nat)
    (ha : a.length = 2 ^ j) (hb1 : 1 ≤ b.length) (hb : b.length ≤ a.length) :
    treeL node (a ++ b) = node (treeL node a) (treeL node b) := by
  have hsplit : 2 ^ (a.length + b.length - 1).log2 = a.length := by
    rw [ha]; congr 1
    rw [Nat.log2_eq_iff (by omega)]
    constructor
    · omega
    · rw [pow_succ]; omega
  unfold treeL
  rw [goIdx]
  simp only [List.length_append, show ¬ a.length + b.length ≤ 1 by omega, ↓reduceIte,
    hsplit, Nat.zero_add, show a.length + b.length - a.length = b.length by omega]
  congr 1
  · apply goIdx_congr _ _ _ _ _ _ (by omega)
    intro i _ hi
    rw [List.getD_append _ _ _ _ (by omega)]
  · have : ∀ i, (a ++ b).getD (i + a.length) default = b.getD i default := by
      intro i; rw [List.getD_append_right _ _ _ _ (by omega)]; congr 1; omega
    rw [show a.length = 0 + a.length by omega, goIdx_shift]
    simp only [this]

/-- The library's wide walk over the `n ≥ 1` chunks from `start`, for
kernels of degree `D` (a power of two): up to `D` chunks give their
leaves, one per chunk; more split as the tree does, and the two halves'
values join; a layer of parents follows, unless the left half gave one
value (degree 1). -/
def wideI {α : Type} (D : Nat) (leaf : Nat → α) (node : α → α → α) (start n : Nat) : List α :=
  if n ≤ D then (List.range n).map fun i => leaf (start + i)
  else if n ≤ 1 then []
  else
    let k := 2 ^ (n - 1).log2
    let a := wideI D leaf node start k
    let b := wideI D leaf node (start + k) (n - k)
    if a.length = 1 then a ++ b else layerL node (a ++ b)
termination_by n
decreasing_by
  all_goals
    have := Nat.log2_self_le (n := n - 1) (by omega)
    have := Nat.one_le_two_pow (n := (n - 1).log2)
    omega

/-- A power of two at most `m ≥ 1` is at most `2 ^ log2 m`. -/
theorem pow_le_two_pow_log2 (j m : Nat) (h : 2 ^ j ≤ m) : 2 ^ j ≤ 2 ^ m.log2 := by
  have hm : m ≠ 0 := by have := Nat.one_le_two_pow (n := j); omega
  apply Nat.pow_le_pow_right (by norm_num)
  have := Nat.lt_log2_self (n := m)
  by_contra hc
  have : 2 ^ (m.log2 + 1) ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
  omega

/-- The wide walk's values have the specification's tree, and between 1 and
`max D 2` of them: exactly `n` up to `D` chunks, exactly `max D 2` for a
power of two of at least that many. -/
theorem wideI_spec {α : Type} [Inhabited α] (D d : Nat) (hD : D = 2 ^ d)
    (leaf : Nat → α) (node : α → α → α) (n : Nat) (hn : 1 ≤ n) (start : Nat) :
    treeL node (wideI D leaf node start n) = goIdx leaf node start n ∧
    1 ≤ (wideI D leaf node start n).length ∧ (wideI D leaf node start n).length ≤ max D 2 ∧
    (n ≤ D → (wideI D leaf node start n).length = n) ∧
    (∀ j, n = 2 ^ j → max D 2 ≤ n → (wideI D leaf node start n).length = max D 2) := by
  induction n using Nat.strong_induction_on generalizing start with
  | _ n ih =>
  have hD1 : 1 ≤ D := by rw [hD]; exact Nat.one_le_two_pow
  by_cases hsmall : n ≤ D
  · rw [wideI, if_pos hsmall]
    refine ⟨?_, by simp; omega, by simp; omega, by simp, ?_⟩
    · unfold treeL
      simp only [List.length_map, List.length_range]
      rw [show start = 0 + start by omega, goIdx_shift]
      apply goIdx_congr _ _ _ _ _ _ hn
      intro i _ hi
      have hi' : i < n := by omega
      simp [List.getD_eq_getElem?_getD, List.getElem?_range hi', Nat.add_comm]
    · intro j hj hmax
      simp
      omega
  · have hn2 : 2 ≤ n := by omega
    have hlog := Nat.log2_self_le (n := n - 1) (by omega)
    have hk1 := Nat.one_le_two_pow (n := (n - 1).log2)
    -- the left half is at least D chunks: D is a power of two at most n - 1
    have hkD : D ≤ 2 ^ (n - 1).log2 := by rw [hD] at hsmall ⊢; exact pow_le_two_pow_log2 d (n - 1) (by omega)
    have hkn : n - 2 ^ (n - 1).log2 ≤ 2 ^ (n - 1).log2 := by
      have := Nat.lt_log2_self (n := n - 1); rw [pow_succ] at this; omega
    rw [wideI, if_neg hsmall, if_neg (by omega)]
    dsimp only
    generalize hk : 2 ^ (n - 1).log2 = k at *
    obtain ⟨tA, hA1, hAle, hAsmall, hAfull⟩ := ih k (by omega) (by omega) start
    obtain ⟨tB, hB1, hBle, hBsmall, hBfull⟩ := ih (n - k) (by omega) (by omega) (start + k)
    generalize ha : wideI D leaf node start k = a at *
    generalize hb : wideI D leaf node (start + k) (n - k) = b at *
    have hsplit : goIdx leaf node start n = node (goIdx leaf node start k) (goIdx leaf node (start + k) (n - k)) := by
      rw [goIdx]; simp only [show ¬ n ≤ 1 by omega, ↓reduceIte, hk]
    -- a is a power of two of values, at least as many as b
    have hapow : ∃ j, a.length = 2 ^ j ∧ b.length ≤ a.length := by
      by_cases hkD' : k ≤ D
      · have hkeq : k = D := by omega
        refine ⟨d, by rw [hAsmall hkD', hkeq, hD], by rw [hAsmall hkD']; have := hBsmall (by omega); omega⟩
      · have hkpow : k = 2 ^ (n - 1).log2 := hk.symm
        have hfull := hAfull _ hkpow (by omega)
        by_cases hd0 : d = 0
        · subst hd0; simp at hD; subst hD
          exact ⟨1, by rw [hfull]; rfl, by rw [hfull]; omega⟩
        · have hD2 : 2 ≤ D := by
            rw [hD]
            calc 2 = 2 ^ 1 := by norm_num
            _ ≤ 2 ^ d := Nat.pow_le_pow_right (by norm_num) (by omega)
          have hmax : max D 2 = D := by omega
          exact ⟨d, by rw [hfull, hmax, hD], by rw [hfull]; omega⟩
    obtain ⟨j, hja, hba⟩ := hapow
    have htree : treeL node (a ++ b) = goIdx leaf node start n := by
      rw [treeL_append node a b j hja hB1 hba, tA, tB, hsplit]
    split
    · -- the left half gave one value: degree 1, two chunks
      refine ⟨htree, by simp; omega, ?_, by intro h; omega, ?_⟩
      · simp; omega
      · intro j' hj' hmax
        have := hAfull _ hk.symm
        have := hBsmall
        simp; omega
    · have hlen := layerL_length node (a ++ b)
      simp only [List.length_append] at hlen
      refine ⟨by rw [treeL_layerL node _ (by simp; omega), htree], by rw [hlen]; omega,
        by rw [hlen]; omega, by intro h; omega, ?_⟩
      intro j' hj' hmax
      -- a power of two of chunks: both halves are 2 ^ (j' - 1), each at least max D 2
      have hj'1 : 1 ≤ j' := by
        rcases j' with _ | j'
        · simp at hj'; omega
        · omega
      have hhalf : k = 2 ^ (j' - 1) := by
        rw [← hk]; congr 1
        rw [Nat.log2_eq_iff (by omega)]
        rw [hj']
        constructor
        · have : 2 ^ j' = 2 ^ (j' - 1) * 2 := by rw [← pow_succ]; congr 1; omega
          have := Nat.one_le_two_pow (n := j' - 1)
          omega
        · rw [show j' - 1 + 1 = j' by omega]; omega
      have hnk : n - k = 2 ^ (j' - 1) := by
        have : 2 ^ j' = 2 ^ (j' - 1) * 2 := by rw [← pow_succ]; congr 1; omega
        omega
      by_cases hbig : max D 2 ≤ k
      · rw [hlen, hAfull _ hhalf hbig, hBfull _ hnk (by omega)]; omega
      · have := hAsmall (by omega)
        omega

/-! The hypotheses can be met: kernels computed by the specification
itself satisfy them, through these conversions between the
representations. -/

/-- Aeneas's chaining value as the specification's. -/
def absCv (a : Cv) : Vector Blake3.Word 8 :=
  ⟨(a.val.map fun x => UInt32.ofBitVec x.bv).toArray, by simp [Std.Array.val, Data.ListN.ListN_length]⟩

/-- The specification's chaining value as Aeneas's. -/
def repCv (v : Vector Blake3.Word 8) : Cv :=
  Std.Array.make 8#usize (v.toList.map fun w => ⟨w.toBitVec⟩) (by simp)

theorem absCv_repCv (v : Vector Blake3.Word 8) : absCv (repCv v) = v := by
  apply Vector.ext
  intro i hi
  simp [absCv, repCv, Std.Array.make_val]

/-- The specification's kernels, as Rust's Kernels. -/
def specKernels (key : Vector Blake3.Word 8) (mode : Blake3.Word) : Kernels Unit where
  chunk := fun _ s c => ok (repCv (Blake3.chunkNode key mode (toBA s.val) c.val).cv)
  parent := fun _ a b => ok (repCv (Blake3.parentNode key mode (absCv a) (absCv b)).cv)

/-- subtree_is_tree applies to them: its hypotheses can hold. -/
theorem subtree_is_tree_satisfiable (key : Vector Blake3.Word 8) (mode : Blake3.Word)
    (input : Slice Std.U8) (hne : input.val ≠ []) (hfit : input.length / 1024 + 1 ≤ Std.U64.max) :
    subtree (specKernels key mode) () input 0#u64 ⦃ r =>
      absCv r = (Blake3.tree key mode ((Array.range (max 1 (((toBA input.val).size + 1023) / 1024))).map
        fun i => Blake3.chunkNode key mode ((toBA input.val).extract (1024 * i) (1024 * (i + 1))) i)).cv ⦄ :=
  subtree_is_tree (specKernels key mode) () key mode absCv
    (fun b i => repCv (Blake3.chunkNode key mode (toBA b) i).cv)
    (fun a b => repCv (Blake3.parentNode key mode (absCv a) (absCv b)).cv)
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => absCv_repCv _) (fun _ _ => absCv_repCv _)
    input hne hfit

end Bridge

#print axioms treecore.left_len_spec
#print axioms treecore.subtree_spec
#print axioms Bridge.walk_eq_goIdx
#print axioms Bridge.walk_is_tree
#print axioms Bridge.subtree_is_tree
#print axioms Bridge.subtree_is_tree_satisfiable
#print axioms Bridge.goIdx_lay
#print axioms Bridge.treeL_layerL
#print axioms Bridge.wideI_spec
