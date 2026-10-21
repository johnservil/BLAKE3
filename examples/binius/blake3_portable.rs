// Vendored for a one-shot comparison from binius64
// (https://github.com/binius-zk/binius64, commit 92a7e33cb67da108f79f760e47e8447fc4b89191),
// crates/hash/src/blake3_portable.rs, by Thomas Coratger; MIT OR Apache-2.0.
// Changed only so that it builds alone, the kernel untouched:
// - binius' traits, rayon and the blake3 crate's three constants: the
//   constants are defined here (BLOCK_LEN 64, OUT_LEN 32), the traits and the
//   parallel wrappers left out;
// - MultiDigest's `update` and `digest` as inherent methods, their bodies as
//   they are;
// - the output type [u8; 32] in place of digest::Output<blake3::Hasher>.
// blake3_neon.rs beside it is copied unchanged.

// Copyright 2026 The Binius Developers

//! Experimental portable, auto-vectorized Blake3 multi-lane kernel.
//!
//! Two batched entry points share one block-compression core:
//! - Leaf hashing, for any message up to one 1024-byte chunk.
//! - Two-to-one inner-node compression, `blake3(left || right)` over a 64-byte pair.
//!
//! An alternative to driving the `blake3` crate's hand-written SIMD kernel.
//! The bet: LLVM auto-vectorizes plain lane loops into whatever the target has.
//!
//! - Each of the 16 compression-state words is held as `[u32; N]`, one lane per message.
//! - Every step is a fixed-width `0..N` loop of plain scalar `u32` arithmetic.
//! - No intrinsics, no `unsafe`, no per-target code.
//!
//! Lanes the vectorizer is expected to fill, per target:
//! - NEON (128-bit) on ARM64 -> 4 lanes per vector.
//! - AVX2 / AVX-512 on x86 -> 8 / 16 lanes per vector.
//! - SVE2 on capable ARM64 -> width-agnostic vectors.
//!
//! Output is bit-identical to `blake3::hash`, pinned to the reference in tests.
//! Scope: any message up to one 1024-byte chunk, including sub-block and partial-block leaves.

use std::{array, mem::MaybeUninit};

// The blake3 crate's constants (binius imports them from it).
const BLOCK_LEN: usize = 64;
const OUT_LEN: usize = 32;

/// Hand-written vector kernel for the block compression, used in place of the lane loops below.
///
/// Kept private so it stays an implementation detail of this module.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[path = "blake3_neon.rs"]
mod neon;

/// Blake3 domain-separation flag marking the first block of a chunk.
const CHUNK_START: u32 = 1 << 0;

/// Blake3 domain-separation flag marking the last block of a chunk.
const CHUNK_END: u32 = 1 << 1;

/// Blake3 domain-separation flag marking the last block of the whole tree.
const ROOT: u32 = 1 << 3;

/// Blake3 initial chaining value: the eight IV words, identical to the SHA-256 IV.
const IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Blake3 message permutation applied between rounds.
///
/// The single fixed schedule from section 2.2 of the Blake3 spec, Table 2.
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// The 7-round count of the Blake3 keyed permutation.
const N_ROUNDS: usize = 7;

/// Applies one Blake3 quarter-round across all `N` lanes.
///
/// The state words at positions `a, b, c, d` are mixed with two message words per lane.
/// Every line is an independent `0..N` map, which is what the vectorizer turns into SIMD.
#[inline(always)]
fn quarter_round<const N: usize>(
	v: &mut [[u32; N]; 16],
	a: usize,
	b: usize,
	c: usize,
	d: usize,
	mx: &[u32; N],
	my: &[u32; N],
) {
	// One lane per iteration; lanes are independent, so the loop vectorizes.
	for i in 0..N {
		v[a][i] = v[a][i].wrapping_add(v[b][i]).wrapping_add(mx[i]);
		v[d][i] = (v[d][i] ^ v[a][i]).rotate_right(16);
		v[c][i] = v[c][i].wrapping_add(v[d][i]);
		v[b][i] = (v[b][i] ^ v[c][i]).rotate_right(12);
		v[a][i] = v[a][i].wrapping_add(v[b][i]).wrapping_add(my[i]);
		v[d][i] = (v[d][i] ^ v[a][i]).rotate_right(8);
		v[c][i] = v[c][i].wrapping_add(v[d][i]);
		v[b][i] = (v[b][i] ^ v[c][i]).rotate_right(7);
	}
}

/// Applies one full Blake3 round: four column mixes, then four diagonal mixes.
///
/// Message words are consumed in order `m[0..16]`, two per quarter-round.
#[inline(always)]
fn round<const N: usize>(v: &mut [[u32; N]; 16], m: &[[u32; N]; 16]) {
	// Columns.
	quarter_round(v, 0, 4, 8, 12, &m[0], &m[1]);
	quarter_round(v, 1, 5, 9, 13, &m[2], &m[3]);
	quarter_round(v, 2, 6, 10, 14, &m[4], &m[5]);
	quarter_round(v, 3, 7, 11, 15, &m[6], &m[7]);
	// Diagonals.
	quarter_round(v, 0, 5, 10, 15, &m[8], &m[9]);
	quarter_round(v, 1, 6, 11, 12, &m[10], &m[11]);
	quarter_round(v, 2, 7, 8, 13, &m[12], &m[13]);
	quarter_round(v, 3, 4, 9, 14, &m[14], &m[15]);
}

/// Permutes the message words in place for the next round.
#[inline(always)]
fn permute<const N: usize>(m: &mut [[u32; N]; 16]) {
	// The permutation reads each slot from its source, so build into a fresh array.
	let permuted: [[u32; N]; 16] = array::from_fn(|i| m[MSG_PERMUTATION[i]]);
	*m = permuted;
}

/// Loads one 64-byte block per lane into 16 little-endian message words.
#[inline(always)]
fn load_block_words<const N: usize>(block: &[[u8; BLOCK_LEN]; N]) -> [[u32; N]; 16] {
	let mut m = [[0u32; N]; 16];
	for lane in 0..N {
		for (w, slot) in m.iter_mut().enumerate() {
			let off = w * 4;
			slot[lane] = u32::from_le_bytes([
				block[lane][off],
				block[lane][off + 1],
				block[lane][off + 2],
				block[lane][off + 3],
			]);
		}
	}
	m
}

/// Compresses one 64-byte block across all `N` lanes, updating the chaining value in place.
///
/// The counter, block length, and flags are shared by every lane, so they broadcast.
/// Only the input chaining value and the message differ per lane.
#[inline(always)]
fn compress_block<const N: usize>(
	cv: &mut [[u32; N]; 8],
	block: &[[u32; N]; 16],
	counter: u64,
	block_len: u32,
	flags: u32,
) {
	// On aarch64 the hand-written vector kernel holds the same state in registers.
	// Lane counts it does not cover fall through to the lane loops.
	#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
	if neon::handles_lanes(N) {
		neon::compress_block(cv, block, counter, block_len, flags);
		return;
	}

	compress_block_portable(cv, block, counter, block_len, flags);
}

/// Compresses one 64-byte block across all `N` lanes with plain lane loops.
///
/// Every target without a hand-written kernel runs this, and it is the reference the vector
/// kernels are tested against.
#[inline(always)]
fn compress_block_portable<const N: usize>(
	cv: &mut [[u32; N]; 8],
	block: &[[u32; N]; 16],
	counter: u64,
	block_len: u32,
	flags: u32,
) {
	// Split the 64-bit counter into its two 32-bit words.
	let counter_lo = counter as u32;
	let counter_hi = (counter >> 32) as u32;

	// Initialize the 16-word state: CV, four IV words, counter, block length, flags.
	let mut v: [[u32; N]; 16] = [
		cv[0],
		cv[1],
		cv[2],
		cv[3],
		cv[4],
		cv[5],
		cv[6],
		cv[7],
		[IV[0]; N],
		[IV[1]; N],
		[IV[2]; N],
		[IV[3]; N],
		[counter_lo; N],
		[counter_hi; N],
		[block_len; N],
		[flags; N],
	];

	// Run 7 rounds; permute the message between all but the last.
	let mut m = *block;
	for r in 0..N_ROUNDS {
		round(&mut v, &m);
		if r < N_ROUNDS - 1 {
			permute(&mut m);
		}
	}

	// Truncated output: h_i = v_i XOR v_{i+8}, feeding the next block or the final digest.
	for i in 0..8 {
		for lane in 0..N {
			cv[i][lane] = v[i][lane] ^ v[i + 8][lane];
		}
	}
}

/// Broadcasts the eight IV words across `N` lanes to seed a fresh chaining value.
#[inline(always)]
fn broadcast_iv<const N: usize>() -> [[u32; N]; 8] {
	array::from_fn(|w| [IV[w]; N])
}

/// Serializes one lane's eight-word chaining value into its 32-byte little-endian digest.
#[inline(always)]
fn serialize_cv_lane<const N: usize>(cv: &[[u32; N]; 8], lane: usize) -> [u8; OUT_LEN] {
	let mut digest = [0u8; OUT_LEN];
	for (w, chunk) in digest.chunks_exact_mut(4).enumerate() {
		chunk.copy_from_slice(&cv[w][lane].to_le_bytes());
	}
	digest
}

/// Portable multi-lane Blake3 leaf digest over `N` messages, hashed a block at a time.
///
/// One chunk, so the chaining value stays at counter 0 and needs no CV stack.
/// Each message is any length up to `CHUNK_LEN`; all `N` lanes must share that length.
///
/// A block is compressed only once the next block's first byte arrives, so the trailing block
/// is deferred to finalization, where it alone carries `CHUNK_END | ROOT`.
#[derive(Clone)]
pub struct PortableBlake3MultiDigest<const N: usize> {
	/// Running chaining value per lane; seeded from the IV.
	cv: [[u32; N]; 8],
	/// The current block being filled, one 64-byte buffer per lane.
	block: [[u8; BLOCK_LEN]; N],
	/// Bytes buffered in `block` so far, shared across lanes (all lanes share one length).
	block_len: usize,
	/// How many blocks have already been compressed into `cv`.
	blocks_compressed: usize,
}

impl<const N: usize> Default for PortableBlake3MultiDigest<N> {
	fn default() -> Self {
		// Fresh chaining value at the IV, empty block buffer, nothing compressed yet.
		Self {
			cv: broadcast_iv(),
			block: [[0u8; BLOCK_LEN]; N],
			block_len: 0,
			blocks_compressed: 0,
		}
	}
}

impl<const N: usize> PortableBlake3MultiDigest<N> {
	/// Compresses the buffered block as a full, non-final block, then empties the buffer.
	fn compress_full_block(&mut self) {
		// Only the very first block of the chunk carries CHUNK_START.
		let flags = if self.blocks_compressed == 0 {
			CHUNK_START
		} else {
			0
		};
		let m = load_block_words(&self.block);
		compress_block(&mut self.cv, &m, 0, BLOCK_LEN as u32, flags);
		self.blocks_compressed += 1;
		self.block_len = 0;
	}

	/// Compresses the trailing block as the chunk root and writes each lane's digest.
	///
	/// Runs on a copy of the state, so the hasher itself is left untouched for reset.
	fn write_root(&self, out: &mut [MaybeUninit<[u8; OUT_LEN]>; N]) {
		let mut cv = self.cv;
		let mut block = self.block;
		// Zero-pad the trailing block's unused tail, so padding never changes the digest.
		for lane in 0..N {
			block[lane][self.block_len..].fill(0);
		}
		// A single-block message has its only block be both the first and the root block.
		let start = if self.blocks_compressed == 0 {
			CHUNK_START
		} else {
			0
		};
		let m = load_block_words(&block);
		compress_block(&mut cv, &m, 0, self.block_len as u32, start | CHUNK_END | ROOT);

		// Serialize each lane's eight-word chaining value into its 32-byte digest.
		for lane in 0..N {
			out[lane].write(serialize_cv_lane(&cv, lane));
		}
	}
}

impl<const N: usize> PortableBlake3MultiDigest<N> {
	/// The trait method `MultiDigest::update`, unchanged.
	pub fn update(&mut self, data: [&[u8]; N]) {
		// Per-lane read cursor into this call's input.
		let mut consumed = [0usize; N];
		loop {
			// Bytes still pending this call; all present lanes share one length, so the max drives.
			let remaining = (0..N)
				.map(|i| data[i].len() - consumed[i])
				.max()
				.unwrap_or(0);
			if remaining == 0 {
				break;
			}
			// A full buffer with more input to come is a non-final block: compress and empty it.
			if self.block_len == BLOCK_LEN {
				self.compress_full_block();
			}
			// Fill the block buffer up to one block from the pending input.
			let take = (BLOCK_LEN - self.block_len).min(remaining);
			for lane in 0..N {
				let avail = data[lane].len() - consumed[lane];
				let n = take.min(avail);
				self.block[lane][self.block_len..self.block_len + n]
					.copy_from_slice(&data[lane][consumed[lane]..consumed[lane] + n]);
				consumed[lane] += n;
			}
			self.block_len += take;
		}
	}

	/// The trait method `MultiDigest::digest`, unchanged: new, update, finalize.
	pub fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<[u8; OUT_LEN]>; N]) {
		let mut hasher = Self::default();
		hasher.update(data);
		hasher.write_root(out);
	}
}
