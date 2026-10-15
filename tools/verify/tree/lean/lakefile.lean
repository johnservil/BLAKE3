import Lake
open Lake DSL

require aeneas from "../aeneas/backends/lean"

-- Mathlib and the rest, kept beside the copy (check.sh).

package tree where
  packagesDir := "../packages"

lean_lib Widecore {}
lean_lib Blake3 {}
lean_lib Hashercore {}
@[default_target] lean_lib Proofs {}
@[default_target] lean_lib WideProofs {}
@[default_target] lean_lib HasherProofs {}
@[default_target] lean_lib StackProofs {}
@[default_target] lean_lib ChildrenProofs {}
@[default_target] lean_lib HashProofs {}
@[default_target] lean_lib HasherCoreProofs {}
@[default_target] lean_lib ChunkProofs {}
@[default_target] lean_lib OutputProofs {}
@[default_target] lean_lib FlatProofs {}
