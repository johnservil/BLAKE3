import Blake3
import Generic

/-! The generic copy of the compression function, at the specification's types, is the
specification's, checked by Lean's kernel. So what the generic copy computes over another word
type (`Emit.lean`), it computes as the specification does. -/

instance : Blake3Generic.Rotr UInt32 := ⟨Blake3.rotr⟩
instance : Blake3Generic.Halves UInt64 UInt32 := ⟨Blake3.low, Blake3.high⟩

theorem iv_eq : (Blake3Generic.IV : Vector UInt32 8) = Blake3.IV := rfl

theorem permute_eq : (Blake3Generic.PERMUTE : Vector UInt32 16 → _) = Blake3.PERMUTE := rfl

theorem g_eq :
    (Blake3Generic.G : Vector UInt32 16 → Fin 16 → Fin 16 → Fin 16 → Fin 16 → UInt32 → UInt32 → _)
      = Blake3.G := rfl

theorem compress_is_the_specifications :
    (Blake3Generic.BLAKE3_COMPRESS : Vector UInt32 8 → Vector UInt32 16 → UInt64 → UInt32 → UInt32 → _)
      = Blake3.BLAKE3_COMPRESS := by
  funext h m t len flags
  simp only [Blake3Generic.BLAKE3_COMPRESS, Blake3.BLAKE3_COMPRESS, g_eq, permute_eq, iv_eq]
  rfl
