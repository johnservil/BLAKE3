# Probe: official BLAKE3 master's x86-64 assembly

This branch is `servil` with four files replaced by the exact files from
BLAKE3-team/BLAKE3 `master` at f55849f89cd85083c1c9daa2c5de20766f309c00:

- `c/blake3_avx2_x86-64_unix.S`
- `c/blake3_avx2_x86-64_windows_msvc.asm`
- `c/blake3_avx512_x86-64_unix.S`
- `c/blake3_avx512_x86-64_windows_msvc.asm`

It exists only to run this fork's tests, including
`test_hash_many_each_input_against_a_guard_page`, against those kernels.
It is not a candidate for merging.
