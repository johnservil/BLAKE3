import Proofs
import Widecore
open Aeneas Aeneas.Std Result

namespace widecore

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

end widecore

namespace WideBridge
open Bridge

/-- The wide walk depends on its leaves only at its chunks' indexes. -/
theorem wideI_congr {α : Type} (D : Nat) (leaf leaf' : Nat → α) (node : α → α → α) (start n : Nat)
    (h : ∀ i, start ≤ i → i < start + n → leaf i = leaf' i) :
    wideI D leaf node start n = wideI D leaf' node start n := by
  induction n using Nat.strong_induction_on generalizing start with
  | _ n ih =>
  rw [wideI, wideI]
  split
  · apply List.map_congr_left
    intro i hi
    simp at hi
    exact h _ (by omega) (by omega)
  · split
    · rfl
    · have := Nat.log2_self_le (n := n - 1) (by omega)
      have := Nat.one_le_two_pow (n := (n - 1).log2)
      dsimp only
      rw [ih _ (by omega) start (fun i h1 h2 => h i h1 (by omega)),
        ih _ (by omega) (start + _) (fun i h1 h2 => h i (by omega) (by omega))]

abbrev Cv := Std.Array Std.U8 32#usize

/-- The leaves of an input whose first chunk is chunk `c`. -/
def leafOf (chunkf : List Std.U8 → Nat → Cv) (bytes : List Std.U8) (c i : Nat) : Cv :=
  chunkf (chunkOf bytes c i) i

/-- The leaves of the first 1024 K bytes are the input's. -/
theorem leafOf_take (chunkf : List Std.U8 → Nat → Cv) (bytes : List Std.U8) (c K i : Nat)
    (h1 : c ≤ i) (h2 : i < c + K) :
    leafOf chunkf (bytes.take (1024 * K)) c i = leafOf chunkf bytes c i := by
  simp only [leafOf, chunkOf, List.drop_take, List.take_take]
  congr 2; omega

/-- The leaves of the bytes after them, from chunk c + K, are the input's. -/
theorem leafOf_drop (chunkf : List Std.U8 → Nat → Cv) (bytes : List Std.U8) (c K i : Nat)
    (h1 : c + K ≤ i) :
    leafOf chunkf (bytes.drop (1024 * K)) (c + K) i = leafOf chunkf bytes c i := by
  simp only [leafOf, chunkOf, List.drop_drop]
  congr 3; omega

theorem take_two_set {α : Type} (L : List α) (x y : α) (h : 2 ≤ L.length) :
    ((L.set 0 x).set 1 y).take 2 = [x, y] := by
  match L, h with
  | _ :: _ :: _, _ => simp

theorem take_one_eq {α : Type} (M : List α) (h : 1 ≤ M.length) : M.take 1 = [M[0]] := by
  match M, h with
  | _ :: _, _ => simp

set_option maxHeartbeats 4000000 in
theorem wide_spec {K : Type} (KI : widecore.Kernels K) (k : K) (D d : Nat) (hD : D = 2 ^ d) (hDmax : D ≤ 128)
    (chunkf : List Std.U8 → Nat → Cv) (parentf : Cv → Cv → Cv)
    (degU : Std.Usize) (hdegU : degU.val = D) (hdeg : KI.degree k = ok degU)
    (hchunks : ∀ (input : Slice Std.U8) (counter : Std.U64) (out : Slice Cv),
      1 ≤ input.length → input.length ≤ D * 1024 → chunks input.length ≤ out.length →
      KI.chunks k input counter out ⦃ r => r.1.val = chunks input.length ∧ r.2.length = out.length ∧
        r.2.val.take r.1.val = (List.range (chunks input.length)).map
          fun i => leafOf chunkf input.val counter.val (counter.val + i) ⦄)
    (hparents : ∀ (children out : Slice Cv),
      2 ≤ children.length → children.length ≤ 256 → (children.length + 1) / 2 ≤ out.length →
      KI.parents k children out ⦃ r => r.1.val = (children.length + 1) / 2 ∧ r.2.length = out.length ∧
        r.2.val.take r.1.val = layerL parentf children.val ⦄)
    (hsubtree : ∀ (input : Slice Std.U8) (ahead : Std.Usize) (counter : Std.U64) (out : Slice Cv),
      D * 1024 < input.length → max D 2 ≤ out.length →
      KI.subtree k input ahead counter out ⦃ r => r.2.length = out.length ∧
        ∀ n, r.1 = some n → n.val = (wideI D (leafOf chunkf input.val counter.val) parentf counter.val
          (chunks input.length)).length ∧
        r.2.val.take n.val = wideI D (leafOf chunkf input.val counter.val) parentf counter.val (chunks input.length) ⦄)
    (input : Slice Std.U8) (ahead : Std.Usize) (counter : Std.U64) (out : Slice Cv)
    (hne : 1 ≤ input.length) (hfit : counter.val + input.length / 1024 + 1 ≤ Std.U64.max)
    (hahead : input.length + ahead.val ≤ Std.Usize.max)
    (hout : min (chunks input.length) (max D 2) ≤ out.length) :
    widecore.wide KI k input ahead counter out ⦃ r =>
      r.1.val = (wideI D (leafOf chunkf input.val counter.val) parentf counter.val (chunks input.length)).length ∧
      r.2.length = out.length ∧
      r.2.val.take r.1.val = wideI D (leafOf chunkf input.val counter.val) parentf counter.val
        (chunks input.length) ⦄ := by
  induction hn : input.length using Nat.strong_induction_on generalizing input ahead counter out with
  | _ n ih =>
  unfold widecore.wide
  have hk : widecore.CHUNK_LEN.val = 1024 := by simp [widecore.CHUNK_LEN]
  simp only [hdeg]
  have hlen : input.len.val = input.length := by simp
  have hD1 : 1 ≤ D := by rw [hD]; exact Nat.one_le_two_pow
  by_cases hs : input.length ≤ D * 1024
  · -- one batch of chunks
    have hc : chunks input.length ≤ D := by unfold chunks; omega
    step*
    · omega
    · have hW : wideI D (leafOf chunkf input.val counter.val) parentf counter.val (chunks n) =
          (List.range (chunks input.length)).map fun i => leafOf chunkf input.val counter.val (counter.val + i) := by
        rw [wideI, if_pos (by rw [← hn]; exact hc), hn]
      rw [hW]
      exact ⟨by rw [r_post]; simp, r_post1, r_post2⟩
  · -- a split
    have hb := treecore.split_bounds input.length (by omega)
    have hcn : chunks input.length - 1 = (input.length - 1) / 1024 := by unfold chunks; omega
    have hKD : D ≤ 2 ^ ((input.length - 1) / 1024).log2 := by
      rw [hD] at hs ⊢; exact pow_le_two_pow_log2 d _ (by omega)
    generalize hKdef : 2 ^ ((input.length - 1) / 1024).log2 = K at *
    have hK1 : 1 ≤ K := by omega
    -- The halves' lists, over the input's leaves, and the walk's list from them.
    have hnc : 2 ≤ chunks n := by unfold chunks; omega
    have hKlog : 2 ^ (chunks n - 1).log2 = K := by rw [← hn, hcn, hKdef]
    have hKn : K < chunks n := by rw [← hKlog]; have := Nat.log2_self_le (n := chunks n - 1) (by omega); omega
    obtain ⟨_, hA1, hAle, hAsmall, hAfull⟩ :=
      wideI_spec D d hD (leafOf chunkf input.val counter.val) parentf K hK1 counter.val
    obtain ⟨_, hB1, hBle, hBsmall, hBfull⟩ :=
      wideI_spec D d hD (leafOf chunkf input.val counter.val) parentf (chunks n - K) (by omega) (counter.val + K)
    generalize hWA : wideI D (leafOf chunkf input.val counter.val) parentf counter.val K = WA at *
    generalize hWB : wideI D (leafOf chunkf input.val counter.val) parentf (counter.val + K) (chunks n - K) = WB at *
    have hW : wideI D (leafOf chunkf input.val counter.val) parentf counter.val (chunks n) =
        if WA.length = 1 then WA ++ WB else layerL parentf (WA ++ WB) := by
      rw [wideI, if_neg (by omega), if_neg (by omega)]
      dsimp only
      rw [hKlog, hWA, hWB]
    have hAeq : ∀ a : Slice Std.U8, a.val = input.val.take (1024 * K) →
        wideI D (leafOf chunkf a.val counter.val) parentf counter.val (chunks (1024 * K)) = WA := by
      intro a ha
      rw [← hWA, show chunks (1024 * K) = K by unfold chunks; omega, ha]
      apply wideI_congr
      intro i h1 h2
      exact leafOf_take chunkf input.val counter.val K i h1 h2
    have hBeq : ∀ b : Slice Std.U8, b.val = input.val.drop (1024 * K) →
        wideI D (leafOf chunkf b.val (counter.val + K)) parentf (counter.val + K) (chunks (n - 1024 * K)) = WB := by
      intro b hb'
      rw [← hWB, show chunks (n - 1024 * K) = chunks n - K by unfold chunks; omega, hb']
      apply wideI_congr
      intro i h1 h2
      exact leafOf_drop chunkf input.val counter.val K i h1
    rw [hn] at hout
    have hmin : min (chunks n) (max D 2) = max D 2 := by
      have : max D 2 ≤ chunks n := by
        by_cases hd : D ≤ 1
        · omega
        · have : D < chunks n := by unfold chunks; omega
          omega
      omega
    rw [hmin] at hout
    have hUmax : 256 ≤ Usize.max := by
      rcases Usize.bounds_eq with h | h <;> rw [h] <;> simp [U32.max, U64.max, U32.numBits, U64.numBits]
    rw [hW]
    have hcvlen : (Std.Array.repeat 256#usize (Std.Array.repeat 32#usize 0#u8)).val.length = 256 := by
      rw [Std.Array.repeat_val, List.length_replicate]; rfl
    generalize Std.Array.repeat 256#usize (Std.Array.repeat 32#usize 0#u8) = cvs0 at *
    step*
    rw [hlen, hKdef] at l_post
    by_cases hl1 : K = 1
    · -- the left half is one chunk: degree 1, two chunks
      have hlC : l = widecore.CHUNK_LEN := by scalar_tac
      have hcn2 : chunks n = 2 := by
        have := Nat.lt_log2_self (n := chunks n - 1)
        rw [pow_succ, hKlog] at this; omega
      have hWA1 : WA.length = 1 := by rw [hAsmall (by omega)]; exact hl1
      have hWB1 : WB.length = 1 := by rw [hBsmall (by omega)]; omega
      simp only [hlC, ↓reduceIte]
      rw [if_pos hWA1]
      step*
      all_goals (try have hs256 : s.length = 256 := (by rw [Slice.length, s_post]; exact hcvlen))
      all_goals (try rw [l_post] at ln_post)
      all_goals (try rw [l_post] at ln_post2)
      all_goals (try rw [hAeq a (by rw [a_post2, l_post])] at ln_post)
      all_goals (try rw [hAeq a (by rw [a_post2, l_post])] at ln_post2)
      all_goals (try have hi4 : i6.val = K := (by rw [i6_post]; (try rw [l_post]); rw [hk]; omega))
      all_goals (try have hi5 : i7.val = K := (by
        rw [i7_post, UScalar.cast_val_eq, hi4]
        apply Nat.mod_eq_of_lt
        have hm : U64.max = 2 ^ 64 - 1 := by simp [U64.max, U64.numBits]
        rw [hm] at hfit; simp only [UScalarTy.numBits]; omega))
      all_goals (try have hi6 : i8.val = counter.val + K := (by rw [i8_post, hi5]))
      all_goals (try rw [hi6, l_post, hn] at rn_post)
      all_goals (try rw [hi6, l_post, hn] at rn_post2)
      all_goals (try rw [hBeq b (by rw [a_post3, l_post])] at rn_post)
      all_goals (try rw [hBeq b (by rw [a_post3, l_post])] at rn_post2)
      all_goals (try have hla : chunks a.length = K := by rw [a_post, l_post]; unfold chunks; omega)
      all_goals (try have hlb : chunks b.length = chunks n - K := by rw [a_post1, l_post, hn]; unfold chunks; omega)
      all_goals (try simp only [show (0#usize).val = 0 from rfl, show (1#usize).val = 1 from rfl,
        show (2#usize).val = 2 from rfl] at *)
      all_goals (try omega)
      all_goals (try (rw [out3_post, Slice.set_length]; omega))
      -- the left half gave one value: the branch that pairs is unreachable
      all_goals (try exact absurd (UScalar.eq_of_val_eq (by rw [ln_post, hWA1]; rfl)) ‹¬ ln = 1#usize›)
      -- the two values, unpaired
      refine ⟨by simp [hWA1, hWB1], by rw [out4_post, Slice.set_length, out3_post, Slice.set_length, o_post], ?_⟩
      have hout4 : 2 ≤ out1.val.length := by simp [Slice.length] at hout o_post; omega
      rw [← ln_post2, ← rn_post2, show ln.val = 1 by omega, show rn.val = 1 by omega,
        out4_post, Slice.set_val_eq, out3_post, Slice.set_val_eq, a2_post, a3_post]
      simp only [show (0#usize).val = 0 from rfl, show (1#usize).val = 1 from rfl]
      rw [take_two_set _ _ _ hout4, take_one_eq _ (by simp [Slice.length] at ln_post1 lo_post; omega),
        take_one_eq _ (by simp [Slice.length] at rn_post1 lo_post1 hs256; omega)]
      rfl
    · have hlC : ¬ l = widecore.CHUNK_LEN := by
        intro h; have := congrArg UScalar.val h; rw [l_post, hk] at this; omega
      -- the left half is a power of two of at least max D 2 chunks: it fills its max D 2 values
      have hK2 : 2 ≤ K := by omega
      have hWAm : WA.length = max D 2 := by
        by_cases hKD' : K ≤ D
        · rw [hAsmall hKD']; omega
        · exact hAfull _ hKdef.symm (by omega)
      have hWA1 : ¬ WA.length = 1 := by omega
      simp only [hlC, ↓reduceIte]
      rw [if_neg hWA1]
      by_cases hd2 : degU < 2#usize
      · have hcv : max D 2 = 2 := by have : degU.val < 2 := (by scalar_tac); omega
        simp only [hd2, ↓reduceIte]
        step*
        all_goals (try have hs256 : s.length = 256 := (by rw [Slice.length, s_post]; exact hcvlen))
        all_goals (try rw [l_post] at ln_post)
        all_goals (try rw [l_post] at ln_post2)
        all_goals (try rw [hAeq a (by rw [a_post2, l_post])] at ln_post)
        all_goals (try rw [hAeq a (by rw [a_post2, l_post])] at ln_post2)
        all_goals (try have hi4 : i6.val = K := (by rw [i6_post]; (try rw [l_post]); rw [hk]; omega))
        all_goals (try have hi5 : i7.val = K := (by
          rw [i7_post, UScalar.cast_val_eq, hi4]
          apply Nat.mod_eq_of_lt
          have hm : U64.max = 2 ^ 64 - 1 := by simp [U64.max, U64.numBits]
          rw [hm] at hfit; simp only [UScalarTy.numBits]; omega))
        all_goals (try have hi6 : i8.val = counter.val + K := (by rw [i8_post, hi5]))
        all_goals (try rw [hi6, l_post, hn] at rn_post)
        all_goals (try rw [hi6, l_post, hn] at rn_post2)
        all_goals (try rw [hBeq b (by rw [a_post3, l_post])] at rn_post)
        all_goals (try rw [hBeq b (by rw [a_post3, l_post])] at rn_post2)
        all_goals (try have hla : chunks a.length = K := by rw [a_post, l_post]; unfold chunks; omega)
        all_goals (try have hlb : chunks b.length = chunks n - K := by rw [a_post1, l_post, hn]; unfold chunks; omega)
        all_goals (try simp only [show (0#usize).val = 0 from rfl, show (1#usize).val = 1 from rfl,
          show (2#usize).val = 2 from rfl] at *)
        all_goals (try omega)
        -- the unpaired branch is unreachable: the left half gave max D 2 ≥ 2 values
        all_goals (try (exfalso; have h1 := congrArg UScalar.val ‹ln = 1#usize›; rw [show (1#usize).val = 1 from rfl] at h1; omega))
        all_goals (try have hback := lo_post4 lo1 hi1 (by omega) (by omega))
        all_goals (try have hbuf : (to_slice_mut_back (split_at_mut_back (lo1, hi1))).to_slice.val = lo1.val ++ hi1.val := (by rw [to_slice_mut_back_post]; simp only [Std.Array.to_slice, Slice.from_val]; rw [Std.Array.from_slice_val _ _ (by rw [← Slice.length, hback.2, hs256]; rfl), hback.1]))
        all_goals (try have hbuflen : (to_slice_mut_back (split_at_mut_back (lo1, hi1))).to_slice.length = 256 := (by simp only [Slice.length] at ln_post1 rn_post1 lo_post lo_post1 hs256 ⊢; rw [hbuf, List.length_append]; omega))
        all_goals (try omega)
        -- the parents of the two halves' values
        have hlo1 : lo1.val = WA := by
          rw [← ln_post2, List.take_of_length_le]; simp [Slice.length] at ln_post1 lo_post; omega
        have hs2 : s2.val = WA ++ WB := by
          rw [s2_post, List.slice_zero_j, hbuf, i9_post, List.take_append, hlo1, List.take_of_length_le (by omega),
            show ln.val + rn.val - WA.length = rn.val by omega, rn_post2]
        refine ⟨?_, by rw [r_post1, o_post], by rw [r_post2, hs2]⟩
        rw [r_post, s2_post1, i9_post, layerL_length]; simp; omega
      · have hcv : max D 2 = D := by have : ¬ degU.val < 2 := (by scalar_tac); omega
        simp only [hd2, ↓reduceIte]
        step*
        all_goals (try have hs256 : s.length = 256 := (by rw [Slice.length, s_post]; exact hcvlen))
        all_goals (try rw [l_post] at ln_post)
        all_goals (try rw [l_post] at ln_post2)
        all_goals (try rw [hAeq a (by rw [a_post2, l_post])] at ln_post)
        all_goals (try rw [hAeq a (by rw [a_post2, l_post])] at ln_post2)
        all_goals (try have hi4 : i6.val = K := (by rw [i6_post]; (try rw [l_post]); rw [hk]; omega))
        all_goals (try have hi5 : i7.val = K := (by
          rw [i7_post, UScalar.cast_val_eq, hi4]
          apply Nat.mod_eq_of_lt
          have hm : U64.max = 2 ^ 64 - 1 := by simp [U64.max, U64.numBits]
          rw [hm] at hfit; simp only [UScalarTy.numBits]; omega))
        all_goals (try have hi6 : i8.val = counter.val + K := (by rw [i8_post, hi5]))
        all_goals (try rw [hi6, l_post, hn] at rn_post)
        all_goals (try rw [hi6, l_post, hn] at rn_post2)
        all_goals (try rw [hBeq b (by rw [a_post3, l_post])] at rn_post)
        all_goals (try rw [hBeq b (by rw [a_post3, l_post])] at rn_post2)
        all_goals (try have hla : chunks a.length = K := by rw [a_post, l_post]; unfold chunks; omega)
        all_goals (try have hlb : chunks b.length = chunks n - K := by rw [a_post1, l_post, hn]; unfold chunks; omega)
        all_goals (try simp only [show (0#usize).val = 0 from rfl, show (1#usize).val = 1 from rfl,
          show (2#usize).val = 2 from rfl] at *)
        all_goals (try omega)
        -- the unpaired branch is unreachable: the left half gave max D 2 ≥ 2 values
        all_goals (try (exfalso; have h1 := congrArg UScalar.val ‹ln = 1#usize›; rw [show (1#usize).val = 1 from rfl] at h1; omega))
        all_goals (try have hback := lo_post4 lo1 hi1 (by omega) (by omega))
        all_goals (try have hbuf : (to_slice_mut_back (split_at_mut_back (lo1, hi1))).to_slice.val = lo1.val ++ hi1.val := (by rw [to_slice_mut_back_post]; simp only [Std.Array.to_slice, Slice.from_val]; rw [Std.Array.from_slice_val _ _ (by rw [← Slice.length, hback.2, hs256]; rfl), hback.1]))
        all_goals (try have hbuflen : (to_slice_mut_back (split_at_mut_back (lo1, hi1))).to_slice.length = 256 := (by simp only [Slice.length] at ln_post1 rn_post1 lo_post lo_post1 hs256 ⊢; rw [hbuf, List.length_append]; omega))
        all_goals (try omega)
        -- the parents of the two halves' values
        have hlo1 : lo1.val = WA := by
          rw [← ln_post2, List.take_of_length_le]; simp [Slice.length] at ln_post1 lo_post; omega
        have hs2 : s2.val = WA ++ WB := by
          rw [s2_post, List.slice_zero_j, hbuf, i9_post, List.take_append, hlo1, List.take_of_length_le (by omega),
            show ln.val + rn.val - WA.length = rn.val by omega, rn_post2]
        refine ⟨?_, by rw [r_post1, o_post], by rw [r_post2, hs2]⟩
        rw [r_post, s2_post1, i9_post, layerL_length]; simp; omega

/-- The library-shaped walk computes the C2SP specification's tree: under
an abstraction `abs` of chaining values, with kernels that meet their
contracts (`chunkf` the specification's chunk chaining value, `parentf` its
parent), the specification's tree over the values `wide` returns is the
specification's tree over the input's chunks. -/
theorem wide_is_tree {K : Type} (KI : widecore.Kernels K) (k : K) (D d : Nat) (hD : D = 2 ^ d) (hDmax : D ≤ 128)
    (key : Vector Blake3.Word 8) (mode : Blake3.Word) (abs : Cv → Vector Blake3.Word 8)
    (chunkf : List Std.U8 → Nat → Cv) (parentf : Cv → Cv → Cv)
    (degU : Std.Usize) (hdegU : degU.val = D) (hdeg : KI.degree k = ok degU)
    (hchunks : ∀ (input : Slice Std.U8) (counter : Std.U64) (out : Slice Cv),
      1 ≤ input.length → input.length ≤ D * 1024 → chunks input.length ≤ out.length →
      KI.chunks k input counter out ⦃ r => r.1.val = chunks input.length ∧ r.2.length = out.length ∧
        r.2.val.take r.1.val = (List.range (chunks input.length)).map
          fun i => leafOf chunkf input.val counter.val (counter.val + i) ⦄)
    (hparents : ∀ (children out : Slice Cv),
      2 ≤ children.length → children.length ≤ 256 → (children.length + 1) / 2 ≤ out.length →
      KI.parents k children out ⦃ r => r.1.val = (children.length + 1) / 2 ∧ r.2.length = out.length ∧
        r.2.val.take r.1.val = layerL parentf children.val ⦄)
    (hsubtree : ∀ (input : Slice Std.U8) (ahead : Std.Usize) (counter : Std.U64) (out : Slice Cv),
      D * 1024 < input.length → max D 2 ≤ out.length →
      KI.subtree k input ahead counter out ⦃ r => r.2.length = out.length ∧
        ∀ n, r.1 = some n → n.val = (wideI D (leafOf chunkf input.val counter.val) parentf counter.val
          (chunks input.length)).length ∧
        r.2.val.take n.val = wideI D (leafOf chunkf input.val counter.val) parentf counter.val (chunks input.length) ⦄)
    (hchunk_spec : ∀ b i, abs (chunkf b i) = (Blake3.chunkNode key mode (toBA b) i).cv)
    (hparent_spec : ∀ a b, abs (parentf a b) = (Blake3.parentNode key mode (abs a) (abs b)).cv)
    (input : Slice Std.U8) (ahead : Std.Usize) (out : Slice Cv) (hne : input.val ≠ [])
    (hfit : input.length / 1024 + 1 ≤ Std.U64.max) (hahead : input.length + ahead.val ≤ Std.Usize.max)
    (hout : min (chunks input.length) (max D 2) ≤ out.length) :
    widecore.wide KI k input ahead 0#u64 out ⦃ r =>
      abs (treeL parentf (r.2.val.take r.1.val)) =
        (Blake3.tree key mode ((Array.range (max 1 (((toBA input.val).size + 1023) / 1024))).map
          fun i => Blake3.chunkNode key mode ((toBA input.val).extract (1024 * i) (1024 * (i + 1))) i)).cv ⦄ := by
  have hlen : 1 ≤ input.length := by
    have := List.length_pos_iff.mpr hne; simp only [Slice.length]; omega
  apply WP.spec_mono (wide_spec KI k D d hD hDmax chunkf parentf degU hdegU hdeg hchunks hparents hsubtree
    input ahead 0#u64 out hlen (by simpa using hfit) hahead hout)
  rintro r ⟨_, _, hr⟩
  have hc : 1 ≤ chunks input.length := by unfold chunks; omega
  rw [hr, (wideI_spec D d hD (leafOf chunkf input.val (0#u64 : Std.U64).val) parentf _ hc _).1]
  simp only [show (0#u64 : Std.U64).val = 0 from rfl]
  have hw := walk_eq_goIdx chunkf parentf input.val 0 hne
  simp only [Slice.length] at hw ⊢
  show abs (goIdx (fun i => chunkf (chunkOf input.val 0 i) i) parentf 0 (chunks input.val.length)) = _
  rw [← hw, walk_map abs chunkf parentf (fun a b => (Blake3.parentNode key mode a b).cv) hparent_spec]
  simp only [hchunk_spec]
  exact walk_is_tree key mode input.val hne

end WideBridge

#print axioms widecore.left_len_spec
#print axioms WideBridge.wideI_congr
#print axioms WideBridge.wide_spec
#print axioms WideBridge.wide_is_tree
