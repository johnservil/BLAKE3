# NEON: each lane its own last block (BLAKE3-Servil #247)

**Goal.** Small items of mixed lengths (a stream task's members,
`hash_each`'s items of up to a chunk) hash side by side, last blocks
included, in one call per group. Today each item's last block compresses
alone, one dependent compression per item. That gap is what separates
mixed-length messages (about 0.5 ns/B on one Zen 5 core) from equal-length
ones (about 0.14), and from hardware SHA-256 (about 0.45).

**Why no per-lane chaining values.** `hash_by_blocks` groups items by k,
the whole blocks before their last. In a group every lane runs k + 1
blocks at counter 0 with the same flags on every block: `CHUNK_START` on the
first, `CHUNK_END | ROOT` on the last. The lanes differ only in the last
block's length (0 to 64) and in its zero-padded bytes. With ROOT, a lane's
final chaining value is its digest.

**The kernel.** The parent family's kernels (fixed counter), with two
per-lane inputs used on the last block alone, from a table at x6:

    struct Last { lens: [u32; 16], blocks: [*const u8; 16] }

- **Lengths.** The vector units hold the state transposed: word 14 is one
  register holding every lane's length. On the last block `start_rows`
  loads it from `lens` (one `ldr q` per quad's four contiguous slots; a
  zip for the pair's dup layout), where it now broadcasts one value. A
  scalar lane loads its own.
- **Last blocks.** On the last block, before the transposes, each lane's
  input pointer is overwritten from `blocks`. The pointers are dead after
  the last block, so the loop is otherwise unchanged. The caller copies
  each item's last partial block into zeroed scratch (16 x 64 bytes on its
  stack), as the serial loop does now.

The rounds are untouched, so the proofs' new work is the two loads and the
pointer swap.

**The contract,** shared with x86's (Sam, gen_x86.py's partial kernel with
a second pointer table): `hash_by_blocks`' groups each in one call, every
lane's output its digest. The test is every lane against `hash()` at every
length 0 to 1024, groups of 1 to 16 lanes, mixed last lengths in a group.

**Steps.**
1. The generator change and its regenerated `.S`, beside the existing
   kernels (no existing kernel changes).
2. `ffi_neon_hybrid::hash_last_lanes`, the Rust entry, and the test above.
3. `hash_by_blocks` calls it on NEON; on SME2 machines too (#249's first
   review point).
4. The proofs (tools/verify) for the new kernels, as assembled.
5. The checks on the Mac and the VM; then measure the lanes each group
   fills in the bench's mixed cells before the per-lane end steps
   (Sam's note 3).
