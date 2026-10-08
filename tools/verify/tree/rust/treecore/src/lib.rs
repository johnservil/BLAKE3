#![no_std]
//! The single-thread tree walk of BLAKE3, in safe Rust over two kernels.

pub const CHUNK_LEN: usize = 1024;
pub type Cv = [u32; 8];

/// The two kernels the walk calls.
pub trait Kernels {
    /// The chaining value of `chunk` (1 to CHUNK_LEN bytes; 0 for the
    /// empty input's only chunk), chunk number `counter`.
    fn chunk(&self, chunk: &[u8], counter: u64) -> Cv;
    /// The chaining value of the parent of `left` and `right`.
    fn parent(&self, left: &Cv, right: &Cv) -> Cv;
}

/// The bytes of the left subtree of an input of `len` > CHUNK_LEN bytes:
/// the largest power of two chunks below its chunk count.
pub fn left_len(len: usize) -> usize {
    let chunks = (len - 1) / CHUNK_LEN; // full chunks before the last
    let mut p: usize = 1;
    while p <= chunks / 2 {
        p *= 2;
    }
    p * CHUNK_LEN
}

/// The chaining value of the subtree over `input` (at least one byte),
/// its first chunk numbered `counter`.
pub fn subtree<K: Kernels>(k: &K, input: &[u8], counter: u64) -> Cv {
    if input.len() <= CHUNK_LEN {
        return k.chunk(input, counter);
    }
    let l = left_len(input.len());
    let (a, b) = input.split_at(l);
    let left = subtree(k, a, counter);
    let right = subtree(k, b, counter + (l / CHUNK_LEN) as u64);
    k.parent(&left, &right)
}
