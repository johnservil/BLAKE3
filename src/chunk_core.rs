// A chunk's state and a node's output in safe Rust, over their compression
// kernels: the library includes this file in src/lib.rs (ChunkState,
// Output), and tools/verify/tree/rust/hashercore includes it too, for
// Aeneas. It uses BLOCK_LEN, CHUNK_START, CHUNK_END, PARENT, ROOT, and the
// Hasher's Chunk trait from its includer.

/// The compression kernels a chunk's state calls.
pub trait Blocks {
    /// `cv` compressed with `block`, its first `block_len` bytes the
    /// message's and the rest zero.
    fn compress_in_place(&self, cv: &mut [u32; 8], block: &[u8; BLOCK_LEN], block_len: u8, counter: u64, flags: u8);
    /// `cv` compressed with the `count` (1 to 16) whole blocks at the start
    /// of `blocks` in turn: the first with `flags | flags_start`, the
    /// others with `flags`.
    fn compress_blocks(&self, cv: &mut [u32; 8], blocks: &[u8], count: usize, counter: u64, flags: u8, flags_start: u8);
}

/// A chunk's state: the chaining value after its blocks so far but the
/// last, which waits in `buf` (zero past `buf_len`) for the output. `buf`
/// comes first and the state is 64-byte aligned, as Output's block (below)
/// explains.
#[repr(C, align(64))]
pub struct ChunkState<P> {
    pub buf: [u8; BLOCK_LEN],
    pub cv: [u32; 8],
    pub chunk_counter: u64,
    pub buf_len: u8,
    pub blocks_compressed: u8,
    pub flags: u8,
    pub platform: P,
}

impl<P: Blocks> ChunkState<P> {
    pub fn new(key: &[u32; 8], chunk_counter: u64, flags: u8, platform: P) -> Self {
        ChunkState { buf: [0; BLOCK_LEN], cv: *key, chunk_counter, buf_len: 0, blocks_compressed: 0, flags, platform }
    }

    /// The bytes so far.
    pub fn count(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.buf_len as usize
    }

    /// As much of `input` as `buf` has room for, into it; returns how much.
    fn fill_buf(&mut self, input: &[u8]) -> usize {
        let want = BLOCK_LEN - self.buf_len as usize;
        let take = if want < input.len() { want } else { input.len() };
        let start = self.buf_len as usize;
        self.buf[start..start + take].copy_from_slice(&input[..take]);
        self.buf_len += take as u8;
        take
    }

    pub fn start_flag(&self) -> u8 {
        if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
    }

    /// The next bytes, `count() + input.len()` at most CHUNK_LEN. Whole
    /// blocks compress straight from `input`; the last block always waits.
    pub fn absorb(&mut self, input: &[u8]) {
        let mut input = input;
        if self.buf_len > 0 {
            let take = self.fill_buf(input);
            input = &input[take..];
            if !input.is_empty() {
                let block_flags = self.flags | self.start_flag();
                self.platform.compress_in_place(&mut self.cv, &self.buf, BLOCK_LEN as u8, self.chunk_counter, block_flags);
                self.buf_len = 0;
                self.buf = [0; BLOCK_LEN];
                self.blocks_compressed += 1;
            }
        }
        if input.len() > BLOCK_LEN {
            let count = (input.len() - 1) / BLOCK_LEN;
            let start_flag = self.start_flag();
            self.platform.compress_blocks(&mut self.cv, input, count, self.chunk_counter, self.flags, start_flag);
            self.blocks_compressed += count as u8;
            input = &input[count * BLOCK_LEN..];
        }
        self.fill_buf(input);
    }
}

/// A node: the inputs of its last compression. `block` comes first and the
/// node is 64-byte aligned: glibc's AVX-512 memcpy writes a 64-byte copy
/// with a pair of overlapping unaligned 64-byte vector stores, and the
/// immediate reload of the destination then fails store-to-load forwarding
/// whenever the stack frame leaves it misaligned (address & 31 != 0). Stack
/// layout decides which case each process gets, so the same binary hashed
/// 64-byte inputs 35-45% slower in some processes than in others (Zen 5).
/// Aligning the block pins every process to the forwarding-friendly case
/// (https://github.com/zooko/bench-hashes/issues/2,
/// https://github.com/BLAKE3-team/BLAKE3/pull/582).
#[repr(C, align(64))]
pub struct Output<P> {
    pub block: [u8; BLOCK_LEN],
    pub input_chaining_value: [u32; 8],
    pub block_len: u8,
    pub counter: u64,
    pub flags: u8,
    pub platform: P,
}

/// The bytes of `words`, least significant first.
pub fn le_bytes(words: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        let b = words[i].to_le_bytes();
        out[4 * i] = b[0];
        out[4 * i + 1] = b[1];
        out[4 * i + 2] = b[2];
        out[4 * i + 3] = b[3];
        i += 1;
    }
    out
}

impl<P: Blocks> Output<P> {
    /// The node's chaining value.
    pub fn chaining_value(&self) -> [u8; 32] {
        let mut cv = self.input_chaining_value;
        self.platform.compress_in_place(&mut cv, &self.block, self.block_len, self.counter, self.flags);
        le_bytes(&cv)
    }

    /// The first 32 bytes of the root's output (the node the root, its
    /// counter 0).
    #[inline]
    pub fn root_bytes(&self) -> [u8; 32] {
        let mut cv = self.input_chaining_value;
        self.platform.compress_in_place(&mut cv, &self.block, self.block_len, 0, self.flags | ROOT);
        le_bytes(&cv)
    }
}

impl<P: Blocks + Copy> ChunkState<P> {
    /// The chunk's node.
    pub fn output(&self) -> Output<P> {
        Output {
            block: self.buf,
            input_chaining_value: self.cv,
            block_len: self.buf_len,
            counter: self.chunk_counter,
            flags: self.flags | self.start_flag() | CHUNK_END,
            platform: self.platform,
        }
    }
}

/// The parent node of two chaining values.
pub fn parent_output<P>(left: &[u8; 32], right: &[u8; 32], key: &[u32; 8], flags: u8, platform: P) -> Output<P> {
    let mut block = [0u8; BLOCK_LEN];
    block[..32].copy_from_slice(left);
    block[32..].copy_from_slice(right);
    Output { block, input_chaining_value: *key, block_len: BLOCK_LEN as u8, counter: 0, flags: flags | PARENT, platform }
}

/// The chunk state the Hasher's loop (hasher_core.rs) works with.
impl<P: Blocks> Chunk for ChunkState<P> {
    #[inline(always)]
    fn counter(&self) -> u64 {
        self.chunk_counter
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.count()
    }

    #[inline(always)]
    fn update(&mut self, input: &[u8]) {
        self.absorb(input)
    }

    #[inline(always)]
    fn advance(&mut self, chunks: u64) {
        self.chunk_counter += chunks
    }
}
