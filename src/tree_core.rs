// The single-thread tree walk in safe Rust, over its kernels. The library
// includes this file in src/tree.rs, and tools/verify/tree/rust/widecore
// includes it too: the walk tools/verify/tree proves computes the C2SP
// specification's tree. It uses CHUNK_LEN, OUT_LEN, Cv, and MAX from its
// includer.

/// The kernels the walk calls.
pub trait Kernels {
    /// The chunks hashed per call: a power of two, at most MAX.
    fn degree(&self) -> usize;
    /// The chaining values of the chunks of `input` (1 to degree() chunks:
    /// whole ones, then perhaps a shorter last one), the first numbered
    /// `counter`, into `out[..n]`; returns n, the number of chunks.
    fn chunks(&self, input: &[u8], counter: u64, out: &mut [Cv]) -> usize;
    /// The parents of the pairs of `children` (2 to 2 MAX of them) into
    /// `out`, an odd last child copied after them; returns their count.
    fn parents(&self, children: &[Cv], out: &mut [Cv]) -> usize;
    /// What `wide` returns for `input`, computed another way where the
    /// kernels have one (None: they have none for it). `ahead` bytes of the
    /// caller's input follow `input`.
    fn subtree(&self, input: &[u8], ahead: usize, counter: u64, out: &mut [Cv]) -> Option<usize>;
}

/// The bytes of the left subtree of an input of `len` > CHUNK_LEN bytes:
/// the largest power of two chunks below its chunk count.
pub fn left_len(len: usize) -> usize {
    let chunks = (len - 1) / CHUNK_LEN;
    let mut p: usize = 1;
    while p <= chunks / 2 {
        p *= 2;
    }
    p * CHUNK_LEN
}

/// The chaining values of a cut through the subtree over `input` (at least
/// one byte), its first chunk numbered `counter`, into `out[..n]`; returns
/// n, between 1 and max(degree, 2). `out` holds at least that many.
/// `ahead` bytes of the caller's input follow `input`.
pub fn wide<K: Kernels>(k: &K, input: &[u8], ahead: usize, counter: u64, out: &mut [Cv]) -> usize {
    if input.len() <= k.degree() * CHUNK_LEN {
        return k.chunks(input, counter, out);
    }
    if let Some(n) = k.subtree(input, ahead, counter, out) {
        return n;
    }
    let l = left_len(input.len());
    let (a, b) = input.split_at(l);
    let mut cvs = [[0u8; OUT_LEN]; 2 * MAX];
    // The left half's values: one when it is a single chunk (degree 1).
    let left_cap = if l == CHUNK_LEN { 1 } else if k.degree() < 2 { 2 } else { k.degree() };
    let (lo, hi) = cvs.split_at_mut(left_cap);
    let ln = wide(k, a, b.len() + ahead, counter, lo);
    let rn = wide(k, b, ahead, counter + (l / CHUNK_LEN) as u64, hi);
    if ln == 1 {
        out[0] = lo[0];
        out[1] = hi[0];
        return 2;
    }
    k.parents(&cvs[..ln + rn], out)
}

