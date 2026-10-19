# Probe: official BLAKE3 1.8.7 release assembly

This branch is `servil` with four files replaced by the exact files from
BLAKE3-team/BLAKE3 release 1.8.7, tag commit f3149ec5bb5449af877ba20377a11008ff499fa2:

- `c/blake3_avx2_x86-64_unix.S`
- `c/blake3_avx2_x86-64_windows_msvc.asm`
- `c/blake3_avx512_x86-64_unix.S`
- `c/blake3_avx512_x86-64_windows_msvc.asm`

It exists only to run this fork's tests, including
`test_hash_many_each_input_against_a_guard_page`, against those kernels.
It is not a candidate for merging.
