import Lake
open Lake DSL

require aeneas from "../aeneas/backends/lean"

-- Mathlib and the rest, kept beside the copy (check.sh).

package tree where
  packagesDir := "../packages"

lean_lib Treecore {}
lean_lib Widecore {}
lean_lib Blake3 {}
lean_lib Stackcore {}
@[default_target] lean_lib Proofs {}
@[default_target] lean_lib WideProofs {}
@[default_target] lean_lib HasherProofs {}
@[default_target] lean_lib StackProofs {}
