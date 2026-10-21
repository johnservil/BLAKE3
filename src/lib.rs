//! The [BLAKE3] cryptographic hash function, in the servil fork: the same
//! digests as official BLAKE3, faster on 64-bit Arm (above all Apple
//! M4-class chips) and on x86-64, with a call for each way programs hash
//! (the table below).
//!
//! # Examples
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Once per process, before any other call (see Startup self-test).
//! blake3_servil::initialize();
//!
//! // Hash an input all at once.
//! let hash1 = blake3_servil::hash(b"foobarbaz");
//!
//! // Hash an input incrementally.
//! let mut hasher = blake3_servil::Hasher::new();
//! hasher.update(b"foo");
//! hasher.update(b"bar");
//! hasher.update(b"baz");
//! let hash2 = hasher.finalize();
//! assert_eq!(hash1, hash2);
//!
//! // Extended output. OutputReader also implements Read and Seek.
//! # #[cfg(feature = "std")] {
//! let mut output = [0; 1000];
//! let mut output_reader = hasher.finalize_xof();
//! output_reader.fill(&mut output);
//! assert_eq!(hash1, output[..32]);
//! # }
//!
//! // Print a hash as hex.
//! println!("{}", hash1);
//! # Ok(())
//! # }
//! ```
//!
//! # Which call to use
//!
//! | Your data | Call |
//! |---|---|
//! | One message in memory | [`hash_multithreaded`] |
//! | One message arriving in pieces | [`Hasher::update`] |
//! | Many messages of one length, in one buffer | [`hash_many_multithreaded`] |
//! | Many messages of different lengths, in memory | [`hash_each_multithreaded_with`] |
//! | Many messages arriving at once (a server's uploads) | [`Hasher::update_each`] |
//! | A stream of messages read, made, or arriving | [`stream`] |
//! | A message read in verified ranges (Bao, iroh-blobs) | [`outboard_with`], [`verify_range_with`], [`Verifier`] |
//!
//! Each call returns once its hash is done, and uses other CPU cores
//! where that is faster (the forms without `multithreaded` stay on the
//! calling thread). A [`stream`] is different: your program submits
//! messages and goes on to read or make the next ones while other threads
//! hash them; each digest arrives in the place your program named, a
//! little later. Choose it when reading or making the data keeps your
//! thread busy, such as files read from memory or a fast disk: it hashes
//! the most bytes per second.
//!
//! Short messages hash fastest together: 1024 messages of 64 bytes in one
//! [`hash_many`] call take about a fifth of the time of 1024 [`hash`]
//! calls.
//!
//! **Keyed hashing and key derivation** run at the plain hash's speed:
//! the keyed hash and key derivation, through a [`Mode`],
//! taken by its `_with` form ([`hash_with`], [`hash_many_with`], ...),
//! by [`Hasher::new_with`], and by a
//! [`stream`].
//!
//! **Hashing now and then** (rather than nonstop) is slower per byte on
//! every hash function, since its code and data must come back into the
//! CPU's caches and a resting core runs slower at first: hashing 64 KiB
//! after a pause takes about three times as long as nonstop on an Apple M4
//! Max.
//!
//! **Apple M4 and later** have one matrix unit that this crate uses for
//! inputs of 16 KiB and more; when several threads of a program hash such
//! inputs at the same moment, one runs at full speed and the others at
//! about two thirds of it. The multithreaded calls share the work among
//! the cores themselves, so a program gets the most from one thread making
//! them.
//!
//! # Memory
//!
//! - **A [`Hasher`] is about 18 KiB**; box it where that matters.
//! - **Single-threaded calls allocate nothing**: [`hash`], [`hash_with`], [`hash_many`], [`hash_many_with`],
//!   [`Hasher::update`] and [`Hasher::finalize`], and [`OutputReader`].
//!   [`Hasher::update_reader`] uses a 1 MiB buffer while it reads past
//!   64 KiB, and [`hash_each_with`] 32 bytes for each chunk of its messages
//!   of 2 to 15 chunks, each freed when it returns. [`outboard_with`]
//!   returns the outboard it allocates (64 bytes for each 16 KiB), and
//!   [`verify_range_with`] allocates 32 bytes for each group of its range,
//!   freed when it returns. A [`Verifier`] keeps up to 16 KiB, for a group
//!   that arrives in pieces, and 48 bytes for each level of the tree; each
//!   [`Verifier::update`] allocates about 100 bytes for each node its piece
//!   holds, and its groups' working space (about 60 bytes a chunk), freed when it
//!   returns.
//! - **Multithreaded calls** allocate nothing below 512 KiB, where they
//!   hash on the calling thread. Above it they keep a list of the pieces
//!   they hand out, freed when they return: 48 bytes for each 128 KiB of
//!   input, and at most 200 bytes for each CPU (1 GiB on 16 CPUs: 390 KiB).
//! - **A [`stream`]** takes its storage (its 2 MiB ring and its entries)
//!   from a pool the process keeps: a thread's first stream allocates it
//!   and later streams reuse it, never freed.
//! - **Once per process**: the startup self-test's buffers, about 45 KiB
//!   of the calling thread's stack while it runs (it allocates nothing),
//!   and the threads: by default a worker per CPU beyond the first, each with the
//!   standard library's stack, started by [`initialize_multithreaded`].
//!
//! # Startup self-test
//!
//! **Every program calls [`initialize`] (or [`initialize_multithreaded`])
//! once, before any other call of this crate**: a call before it panics, in
//! every build, saying so. The check costs nothing: it is part of the
//! platform load every call makes. **A program that makes multithreaded
//! calls or streams calls [`initialize_multithreaded`]**: a
//! multithreaded call or a stream before it panics (one load per call).
//!
//! In `initialize`, the crate checks the build and the machine running it
//! before the process first hashes: a miscompilation, a faulty CPU, a
//! binary built for another CPU, or tampering that breaks the digests.
//! (What the code computes is the tests' and the proofs' to establish.) It
//! hashes fixed inputs (42 on AArch64, 9 elsewhere), chosen so that
//! together they run every one of its assembly kernels on this machine (a
//! unit test checks it on every run), each input carrying the previous
//! output, and compares each result with the reference implementation's; on x86 it first checks that the CPU has the
//! LZCNT and BMI1 instructions a build for newer CPUs assumes, which older
//! ones run as other instructions instead of stopping. A difference means
//! this build or this CPU computes wrong digests, and stops the program
//! with a panic naming the code path. The check takes 0.1 to 0.2 ms once
//! per process on AArch64 (Apple M4 Max, and a Linux VM on it) and about
//! 28 µs on x86-64 (i7-12700K, a performance core; 47 µs on an efficiency
//! core), in every build, with or without the `std` feature. Other threads
//! that call `initialize` meanwhile wait for it, and stop with it if it
//! fails. Without `std`, the panic goes to the program's panic handler,
//! which aborts or halts.
//!
//! # Cargo Features
//!
//! The `std` feature (the only feature enabled by default) enables the
//! [`Write`] implementation and the [`update_reader`](Hasher::update_reader)
//! method for [`Hasher`], and also the [`Read`] and [`Seek`] implementations
//! for [`OutputReader`].
//!
//! The `zeroize` feature (off by default, on for [docs.rs])
//! implements
//! [`Zeroize`](https://docs.rs/zeroize/latest/zeroize/trait.Zeroize.html) for
//! this crate's types.
//!
//! The `serde` feature (off by default, on for [docs.rs]) implements
//! [`serde::Serialize`](https://docs.rs/serde/latest/serde/trait.Serialize.html) and
//! [`serde::Deserialize`](https://docs.rs/serde/latest/serde/trait.Deserialize.html)
//! for [`Hash`](struct@Hash).
//!
//! [BLAKE3]: https://blake3.io
//! [docs.rs]: https://docs.rs/
//! [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
//! [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
//! [`Seek`]: https://doc.rust-lang.org/std/io/trait.Seek.html
//! [`digest`]: https://crates.io/crates/digest
//! [`signature`]: https://crates.io/crates/signature

#![cfg_attr(not(feature = "std"), no_std)]

// The tests use std in every build, the no_std one included.
#[cfg(all(test, not(feature = "std")))]
#[macro_use]
extern crate std;

#[cfg(test)]
mod test;

#[cfg(feature = "std")]
mod each;
#[cfg(feature = "std")]
pub use each::{hash_each_multithreaded_with, hash_each_with};
#[cfg(feature = "std")]
mod outboard;
#[cfg(feature = "std")]
#[doc(hidden)]
#[cfg(feature = "std")]
pub use outboard::{GROUP_LEN, Verifier, outboard_multithreaded_with, outboard_with, verify_range_with};

mod platform;

// Platform-specific implementations of the compression function. These
// BLAKE3-specific cfg flags are set in build.rs.
#[cfg(blake3_avx2_ffi)]
#[path = "ffi_avx2.rs"]
mod avx2;
#[cfg(blake3_avx512_ffi)]
#[path = "ffi_avx512.rs"]
mod avx512;
#[cfg(blake3_neon)]
#[path = "ffi_neon_hybrid.rs"]
mod neon_hybrid;
mod portable;
#[cfg(blake3_sme2)]
#[path = "ffi_sme2.rs"]
mod sme2;
// Whole chunks and a partial one in one pass, on x86-64 (generated by
// tools/gen_x86.py, built with the SSE4.1 and AVX2 assembly).
#[cfg(blake3_sse41_ffi)]
mod ffi_partial;
#[cfg(blake3_sse41_ffi)]
#[path = "ffi_sse41.rs"]
mod sse41;

#[cfg(feature = "std")]
mod io;
#[cfg(feature = "std")]
mod lanes;
mod many;
#[cfg(feature = "std")]
#[cfg(feature = "std")]
#[cfg(feature = "std")]
mod stream;
mod tree;
#[cfg(feature = "std")]
pub use stream::{Stream, stream};
mod self_test;

/// A kind of kernel or path a call ran, for the tests that check which
/// path a call took. Each build notes the kinds it has.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum Ran {
    /// The call went to the pool.
    Pool = 1,
    /// The call wanted SME2 and another thread held the turn.
    TurnTaken = 2,
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    /// The kinds this thread's calls ran since the test last cleared it.
    pub(crate) static RAN: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Note that `kind` happened: in the tests' record, and nothing in any
/// other build (some of which have no kernel to note).
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn ran(kind: Ran) {
    #[cfg(all(test, feature = "std"))]
    RAN.with(|r| r.set(r.get() | kind as u32));
    let _ = kind;
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    /// The assembly entry points this thread's calls ran since the test
    /// last cleared it, by address.
    pub(crate) static RAN_ENTRIES: core::cell::RefCell<std::collections::BTreeSet<usize>> = const { core::cell::RefCell::new(std::collections::BTreeSet::new()) };
}

/// Note that the assembly entry point at `entry` ran (in the tests'
/// record; nothing in any other build).
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn ran_kernel(entry: *const ()) {
    #[cfg(all(test, feature = "std"))]
    RAN_ENTRIES.with(|r| r.borrow_mut().insert(entry as usize));
    let _ = entry;
}

use arrayvec::{ArrayString, ArrayVec};
use core::cmp;
use core::fmt;
use platform::{MAX_SIMD_DEGREE, MAX_SIMD_DEGREE_OR_2, Platform};
#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

/// The number of bytes in a [`Hash`](struct.Hash.html), 32.
pub const OUT_LEN: usize = 32;

/// The number of bytes in a key, 32.
pub const KEY_LEN: usize = 32;

/// The number of bytes in a block, 64.
///
/// You don't usually need to think about this number. One case where it matters is calling
/// [`OutputReader::fill`] in a loop, where using a `buf` argument that's a multiple of `BLOCK_LEN`
/// avoids repeating work.
pub const BLOCK_LEN: usize = 64;

/// The number of bytes in a chunk, 1024.
///
/// You don't usually need to think about this number, but it often comes up in benchmarks, because
/// the maximum degree of parallelism used by the implementation equals the number of chunks.
pub const CHUNK_LEN: usize = 1024;

const MAX_DEPTH: usize = 54; // 2^54 * CHUNK_LEN = 2^64

// While iterating the compression function within a chunk, the CV is
// represented as words, to avoid doing two extra endianness conversions for
// each compression in the portable implementation. But the hash_many interface
// needs to hash both input bytes and parent nodes, so it's better for its
// output CVs to be represented as bytes.
type CVWords = [u32; 8];
type CVBytes = [u8; 32]; // little-endian

const IV: &CVWords = &[0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

// These are the internal flags that we use to domain separate root/non-root,
// chunk/parent, and chunk beginning/middle/end. These get set at the high end
// of the block flags word in the compression function, so their values start
// high and go down.
const CHUNK_START: u8 = 1 << 0;
const CHUNK_END: u8 = 1 << 1;
const PARENT: u8 = 1 << 2;
const ROOT: u8 = 1 << 3;
const KEYED_HASH: u8 = 1 << 4;
const DERIVE_KEY_CONTEXT: u8 = 1 << 5;
const DERIVE_KEY_MATERIAL: u8 = 1 << 6;

#[inline]
fn counter_low(counter: u64) -> u32 {
    counter as u32
}

#[inline]
fn counter_high(counter: u64) -> u32 {
    (counter >> 32) as u32
}

/// An output of the default size, 32 bytes, which provides constant-time
/// equality checking.
///
/// `Hash` implements [`From`] and [`Into`] for `[u8; 32]`, and it provides
/// [`from_bytes`] and [`as_bytes`] for explicit conversions between itself and
/// `[u8; 32]`. However, byte arrays and slices don't provide constant-time
/// equality checking, which is often a security requirement in software that
/// handles private data. `Hash` doesn't implement [`Deref`] or [`AsRef`], to
/// avoid situations where a type conversion happens implicitly and the
/// constant-time property is accidentally lost.
///
/// `Hash` provides the [`to_hex`] and [`from_hex`] methods for converting to
/// and from hexadecimal. It also implements [`Display`] and [`FromStr`].
///
/// [`From`]: https://doc.rust-lang.org/std/convert/trait.From.html
/// [`Into`]: https://doc.rust-lang.org/std/convert/trait.Into.html
/// [`as_bytes`]: #method.as_bytes
/// [`from_bytes`]: #method.from_bytes
/// [`Deref`]: https://doc.rust-lang.org/stable/std/ops/trait.Deref.html
/// [`AsRef`]: https://doc.rust-lang.org/std/convert/trait.AsRef.html
/// [`to_hex`]: #method.to_hex
/// [`from_hex`]: #method.from_hex
/// [`Display`]: https://doc.rust-lang.org/std/fmt/trait.Display.html
/// [`FromStr`]: https://doc.rust-lang.org/std/str/trait.FromStr.html
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Hash, Eq)]
#[repr(transparent)]
pub struct Hash([u8; OUT_LEN]);

impl Hash {
    /// The raw bytes of the `Hash`. Note that byte arrays don't provide
    /// constant-time equality checking, so if you need to compare hashes,
    /// prefer the `Hash` type.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; OUT_LEN] {
        &self.0
    }

    /// Create a `Hash` from its raw bytes representation.
    pub const fn from_bytes(bytes: [u8; OUT_LEN]) -> Self {
        Self(bytes)
    }

    /// The raw bytes of the `Hash`, as a slice. Useful for serialization. Note that byte arrays
    /// don't provide constant-time equality checking, so if you need to compare hashes, prefer
    /// the `Hash` type.
    #[inline]
    pub const fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    /// Create a `Hash` from its raw bytes representation as a slice.
    ///
    /// Returns an error if the slice is not exactly 32 bytes long.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, core::array::TryFromSliceError> {
        Ok(Self::from_bytes(bytes.try_into()?))
    }

    /// Encode a `Hash` in lowercase hexadecimal.
    ///
    /// The returned [`ArrayString`] is a fixed size and doesn't allocate memory
    /// on the heap. Note that [`ArrayString`] doesn't provide constant-time
    /// equality checking, so if you need to compare hashes, prefer the `Hash`
    /// type.
    ///
    /// [`ArrayString`]: https://docs.rs/arrayvec/0.5.1/arrayvec/struct.ArrayString.html
    pub fn to_hex(&self) -> ArrayString<{ 2 * OUT_LEN }> {
        let mut s = ArrayString::new();
        let table = b"0123456789abcdef";
        for &b in self.0.iter() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }

    /// Decode a `Hash` from hexadecimal. Both uppercase and lowercase ASCII
    /// bytes are supported.
    ///
    /// Any byte outside the ranges `'0'...'9'`, `'a'...'f'`, and `'A'...'F'`
    /// results in an error. An input length other than 64 also results in an
    /// error.
    ///
    /// Note that `Hash` also implements `FromStr`, so `Hash::from_hex("...")`
    /// is equivalent to `"...".parse()`.
    pub fn from_hex(hex: impl AsRef<[u8]>) -> Result<Self, HexError> {
        fn hex_val(byte: u8) -> Result<u8, HexError> {
            match byte {
                b'A'..=b'F' => Ok(byte - b'A' + 10),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                b'0'..=b'9' => Ok(byte - b'0'),
                _ => Err(HexError(HexErrorInner::InvalidByte(byte))),
            }
        }
        let hex_bytes: &[u8] = hex.as_ref();
        if hex_bytes.len() != OUT_LEN * 2 {
            return Err(HexError(HexErrorInner::InvalidLen(hex_bytes.len())));
        }
        let mut hash_bytes: [u8; OUT_LEN] = [0; OUT_LEN];
        for i in 0..OUT_LEN {
            hash_bytes[i] = 16 * hex_val(hex_bytes[2 * i])? + hex_val(hex_bytes[2 * i + 1])?;
        }
        Ok(Hash::from(hash_bytes))
    }
}

impl From<[u8; OUT_LEN]> for Hash {
    #[inline]
    fn from(bytes: [u8; OUT_LEN]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<Hash> for [u8; OUT_LEN] {
    #[inline]
    fn from(hash: Hash) -> Self {
        hash.0
    }
}

impl core::str::FromStr for Hash {
    type Err = HexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Hash::from_hex(s)
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for Hash {
    fn zeroize(&mut self) {
        // Destructuring to trigger compile error as a reminder to update this impl.
        let Self(bytes) = self;
        bytes.zeroize();
    }
}

/// This implementation is constant-time.
impl PartialEq for Hash {
    #[inline]
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq_32(&self.0, &other.0)
    }
}

/// This implementation is constant-time.
impl PartialEq<[u8; OUT_LEN]> for Hash {
    #[inline]
    fn eq(&self, other: &[u8; OUT_LEN]) -> bool {
        constant_time_eq::constant_time_eq_32(&self.0, other)
    }
}

/// This implementation is constant-time if the target is 32 bytes long.
impl PartialEq<[u8]> for Hash {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        constant_time_eq::constant_time_eq(&self.0, other)
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Formatting field as `&str` to reduce code size since the `Debug`
        // dynamic dispatch table for `&str` is likely needed elsewhere already,
        // but that for `ArrayString<[u8; 64]>` is not.
        let hex = self.to_hex();
        let hex: &str = hex.as_str();

        f.write_str(hex)
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Formatting field as `&str` to reduce code size since the `Debug`
        // dynamic dispatch table for `&str` is likely needed elsewhere already,
        // but that for `ArrayString<[u8; 64]>` is not.
        let hex = self.to_hex();
        let hex: &str = hex.as_str();

        f.debug_tuple("Hash").field(&hex).finish()
    }
}

/// The error type for [`Hash::from_hex`].
///
/// The `.to_string()` representation of this error currently distinguishes between bad length
/// errors and bad character errors. This is to help with logging and debugging, but it isn't a
/// stable API detail, and it may change at any time.
#[derive(Clone, Debug)]
pub struct HexError(HexErrorInner);

#[derive(Clone, Debug)]
enum HexErrorInner {
    InvalidByte(u8),
    InvalidLen(usize),
}

impl fmt::Display for HexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            HexErrorInner::InvalidByte(byte) => {
                if byte < 128 {
                    write!(f, "invalid hex character: {:?}", byte as char)
                } else {
                    write!(f, "invalid hex character: 0x{:x}", byte)
                }
            }
            HexErrorInner::InvalidLen(len) => {
                write!(f, "expected 64 hex bytes, received {}", len)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HexError {}

type Output = hasher::Output<Platform>;

impl Clone for Output {
    fn clone(&self) -> Self {
        hasher::Output {
            block: self.block,
            input_chaining_value: self.input_chaining_value,
            block_len: self.block_len,
            counter: self.counter,
            flags: self.flags,
            platform: self.platform,
        }
    }
}

impl Output {
    fn root_hash(&self) -> Hash {
        debug_assert_eq!(self.counter, 0);
        Hash(self.root_bytes())
    }

    fn root_output_block(&self) -> [u8; 2 * OUT_LEN] {
        self.platform.compress_xof(&self.input_chaining_value, &self.block, self.block_len, self.counter, self.flags | ROOT)
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for Output {
    fn zeroize(&mut self) {
        // Destructuring to trigger compile error as a reminder to update this impl.
        let Self { input_chaining_value, block, block_len, counter, flags, platform: _ } = self;

        input_chaining_value.zeroize();
        block.zeroize();
        block_len.zeroize();
        counter.zeroize();
        flags.zeroize();
    }
}

type ChunkState = hasher::ChunkState<Platform>;

impl hasher::Blocks for Platform {
    #[inline(always)]
    fn compress_in_place(&self, cv: &mut CVWords, block: &[u8; BLOCK_LEN], block_len: u8, counter: u64, flags: u8) {
        Platform::compress_in_place(self, cv, block, block_len, counter, flags)
    }

    #[inline(always)]
    fn compress_blocks(&self, cv: &mut CVWords, blocks: &[u8], count: usize, counter: u64, flags: u8, flags_start: u8) {
        Platform::compress_blocks(self, cv, blocks, count, counter, flags, flags_start)
    }
}

impl Clone for ChunkState {
    fn clone(&self) -> Self {
        hasher::ChunkState {
            buf: self.buf,
            cv: self.cv,
            chunk_counter: self.chunk_counter,
            buf_len: self.buf_len,
            blocks_compressed: self.blocks_compressed,
            flags: self.flags,
            platform: self.platform,
        }
    }
}

impl ChunkState {
    // Try to avoid buffering as much as possible, by compressing directly from
    // the input slice when full blocks are available (src/hasher.rs).
    #[inline(always)]
    fn update(&mut self, input: &[u8]) -> &mut Self {
        self.absorb(input);
        debug_assert!(self.count() <= CHUNK_LEN);
        self
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for ChunkState {
    fn zeroize(&mut self) {
        // Destructuring to trigger compile error as a reminder to update this impl.
        let Self { cv, chunk_counter, buf, buf_len, blocks_compressed, flags, platform: _ } = self;

        cv.zeroize();
        chunk_counter.zeroize();
        buf.zeroize();
        buf_len.zeroize();
        blocks_compressed.zeroize();
        flags.zeroize();
    }
}

// IMPLEMENTATION NOTE
// ===================
/// Undocumented and unstable, for benchmarks only.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum IncrementCounter {
    Yes,
    No,
}

impl IncrementCounter {
    #[inline]
    fn yes(&self) -> bool {
        match self {
            IncrementCounter::Yes => true,
            IncrementCounter::No => false,
        }
    }
}

/// The lanes of the one pass that hashes `whole` whole chunks with a
/// partial one of `partial` bytes after them, where the pass costs less
/// than the partial chunk's blocks after the whole ones: one whole chunk
/// and a partial one of two blocks or more in four lanes (AVX2 and
/// AVX-512: the chain falls from 25 blocks to 16 at 1536 bytes), three
/// whole chunks in four lanes (SSE4.1 and AVX2), and in eight (AVX2) six or
/// seven, or five with a partial chunk of six blocks or more (with fewer,
/// an i7-12700K's efficiency core ran the eight lanes up to 7% slower; a
/// pass of eight lanes costs 1.6 times one of four on its performance
/// core). AVX-512 otherwise keeps its own widths (unmeasured).
#[cfg(blake3_sse41_ffi)]
#[inline]
fn spare_lane(platform: Platform, whole: usize, partial: usize) -> Option<usize> {
    match (platform, whole) {
        (Platform::AVX2, 1) if partial > BLOCK_LEN => Some(4),
        #[cfg(blake3_avx512_ffi)]
        (Platform::AVX512, 1) if partial > BLOCK_LEN => Some(4),
        (Platform::SSE41 | Platform::AVX2, 3) => Some(4),
        (Platform::AVX2, 6..=7) => Some(8),
        (Platform::AVX2, 5) if partial > 5 * BLOCK_LEN => Some(8),
        _ => None,
    }
}

// Use SIMD parallelism to hash up to MAX_SIMD_DEGREE chunks at the same time
// on a single thread. Write out the chunk chaining values and return the
// number of chunks hashed. These chunks are never the root and never empty;
// those cases use a different codepath.
fn compress_chunks_parallel(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8, platform: Platform, out: &mut [u8]) -> usize {
    debug_assert!(!input.is_empty(), "empty chunks below the root");
    debug_assert!(input.len() <= MAX_SIMD_DEGREE * CHUNK_LEN);

    let mut chunks_exact = input.chunks_exact(CHUNK_LEN);
    let mut chunks_array = ArrayVec::<&[u8; CHUNK_LEN], MAX_SIMD_DEGREE>::new();
    for chunk in &mut chunks_exact {
        chunks_array.push(chunk.try_into().unwrap());
    }
    // Whole chunks that leave a lane spare, and a partial one: one pass,
    // the partial chunk in the spare lane.
    #[cfg(blake3_sse41_ffi)]
    if let (false, Some(lanes)) = (chunks_exact.remainder().is_empty(), spare_lane(platform, chunks_array.len(), chunks_exact.remainder().len())) {
        let (whole, partial) = (&chunks_array[..], chunks_exact.remainder());
        // Sound: spare_lane admits each width only on a platform whose CPU
        // has its instructions.
        if lanes == 4 {
            unsafe { ffi_partial::hash_chunks_with_partial_sse41(whole, partial, key, chunk_counter, flags, out) };
        } else {
            unsafe { ffi_partial::hash_chunks_with_partial_avx2(whole, partial, key, chunk_counter, flags, out) };
        }
        return whole.len() + 1;
    }
    // Whole chunks and a partial one, on the NEON hybrids: where a q kernel
    // covers the count, the partial chunk runs on the scalar units beside
    // the whole chunks rather than after them.
    #[cfg(blake3_neon)]
    let hybrids = matches!(platform, Platform::NEON);
    #[cfg(blake3_sme2)]
    let hybrids = hybrids || matches!(platform, Platform::SME2);
    #[cfg(blake3_neon)]
    if !chunks_exact.remainder().is_empty()
        && hybrids
        && neon_hybrid::partial_pays(chunks_array.len(), chunks_exact.remainder().len())
        && neon_hybrid::sha3_detected()
    {
        // Safe: the kernel exists for this count and the CPU has SHA-3.
        unsafe {
            neon_hybrid::hash_chunks_with_partial(&chunks_array, chunks_exact.remainder(), key, chunk_counter, flags, CHUNK_START, CHUNK_END, out);
        }
        return chunks_array.len() + 1;
    }
    platform.hash_many(&chunks_array, key, chunk_counter, IncrementCounter::Yes, flags, CHUNK_START, CHUNK_END, out);

    // Hash the remaining partial chunk, if there is one. Note that the empty
    // chunk (meaning the empty message) is a different codepath.
    let chunks_so_far = chunks_array.len();
    if !chunks_exact.remainder().is_empty() {
        let counter = chunk_counter + chunks_so_far as u64;
        let mut chunk_state = ChunkState::new(key, counter, flags, platform);
        chunk_state.update(chunks_exact.remainder());
        *<&mut [u8; OUT_LEN]>::try_from(&mut out[chunks_so_far * OUT_LEN..][..OUT_LEN]).unwrap() = chunk_state.output().chaining_value();
        chunks_so_far + 1
    } else {
        chunks_so_far
    }
}

// Use SIMD parallelism to hash up to MAX_SIMD_DEGREE parents at the same time
// on a single thread. Write out the parent chaining values and return the
// number of parents hashed. (If there's an odd input chaining value left over,
// return it as an additional output.) These parents are never the root and
// never empty; those cases use a different codepath.
fn compress_parents_parallel(child_chaining_values: &[u8], key: &CVWords, flags: u8, platform: Platform, out: &mut [u8]) -> usize {
    debug_assert_eq!(child_chaining_values.len() % OUT_LEN, 0, "wacky hash bytes");
    let num_children = child_chaining_values.len() / OUT_LEN;
    debug_assert!(num_children >= 2, "not enough children");
    debug_assert!(num_children <= 2 * MAX_SIMD_DEGREE_OR_2, "too many");

    let mut parents_exact = child_chaining_values.chunks_exact(BLOCK_LEN);
    // Use MAX_SIMD_DEGREE_OR_2 rather than MAX_SIMD_DEGREE here: the walk
    // (src/tree.rs) returns at least 2 values.
    let mut parents_array = ArrayVec::<&[u8; BLOCK_LEN], MAX_SIMD_DEGREE_OR_2>::new();
    for parent in &mut parents_exact {
        parents_array.push(parent.try_into().unwrap());
    }
    platform.hash_many(
        &parents_array,
        key,
        0, // Parents always use counter 0.
        IncrementCounter::No,
        flags | PARENT,
        0, // Parents have no start flags.
        0, // Parents have no end flags.
        out,
    );

    // If there's an odd child left over, it becomes an output.
    let parents_so_far = parents_array.len();
    if !parents_exact.remainder().is_empty() {
        out[parents_so_far * OUT_LEN..][..OUT_LEN].copy_from_slice(parents_exact.remainder());
        parents_so_far + 1
    } else {
        parents_so_far
    }
}

// Hash a subtree with the walk (src/tree.rs, `children`), condensing its
// chaining values down to a single parent node. Don't compress that
// last parent node, however. Instead, return its message bytes (the
// concatenated chaining values of its children). This is necessary when the
// first call to update() supplies a complete subtree, because the topmost
// parent node of that subtree could end up being the root. It's also necessary
// for extended output in the general case.
//
// This function is not used on inputs of 1 chunk or less. That's a
// different codepath. Its arrays live in tree::root_children, out of line.
#[inline]
fn compress_subtree_to_parent_node(input: &[u8], ahead: usize, key: &CVWords, chunk_counter: u64, flags: u8, platform: Platform) -> [u8; BLOCK_LEN] {
    debug_assert!(input.len() > CHUNK_LEN);
    let kernels = tree::PlatformKernels { key, flags, platform };
    let [left, right] = tree::root_children(&kernels, input, ahead, chunk_counter);
    let mut block = [0u8; BLOCK_LEN];
    block[..OUT_LEN].copy_from_slice(&left);
    block[OUT_LEN..].copy_from_slice(&right);
    block
}

// The root hash of an input of one chunk or less, in one kernel call. The
// whole blocks are read in place, the final one too when it is full; a
// short final block (and the empty block for the empty input) is copied
// into a zero-padded buffer, which is the copy ChunkState::update() makes
// too. The scalar kernel runs on every AArch64 core, so this path needs no
// platform check.
#[cfg(blake3_neon)]
#[inline]
fn hash_one_chunk_root(input: &[u8], key: &CVWords, flags: u8) -> Hash {
    debug_assert!(input.len() <= CHUNK_LEN);
    let blocks = cmp::max(1, input.len().div_ceil(BLOCK_LEN));
    let whole = (blocks - 1) * BLOCK_LEN;
    let tail = &input[whole..];
    /// The padded last block, 64-byte aligned as hasher::Output's is
    /// (src/hasher.rs explains).
    #[repr(align(64))]
    struct Aligned64([u8; BLOCK_LEN]);
    let padded;
    let last: &[u8; BLOCK_LEN] = match tail.try_into() {
        Ok(full) => full,
        Err(_) => {
            let mut block = Aligned64([0; BLOCK_LEN]);
            block.0[..tail.len()].copy_from_slice(tail);
            padded = block;
            &padded.0
        }
    };
    let mut out = [0u8; OUT_LEN];
    // Safe: `input` holds `whole` bytes before `tail`, and blocks is 1..=16.
    unsafe {
        neon_hybrid::hash_chunk(input.as_ptr(), blocks, last, tail.len() as u8, key, 0, flags, CHUNK_START, CHUNK_END | ROOT, &mut out);
    }
    Hash(out)
}

// Hash a whole subtree at its chunk counter (or the whole input at zero).
// Unlike the wider helpers above, this also handles one chunk or less.
// The pool's pieces (lanes.rs) use it; the one-shot calls go through
// tree::hash.
#[cfg(feature = "std")]
#[inline]
fn hash_all_at_once(input: &[u8], key: &CVWords, chunk_counter: u64, flags: u8, platform: Platform) -> Output {
    // The input is a complete subtree at chunk_counter, or the whole
    // input (including empty) at counter zero. Its byte offset fits u64.
    debug_assert!(chunk_counter <= u64::MAX / CHUNK_LEN as u64);
    debug_assert!(chunk_counter == 0 || (!input.is_empty() && input.len() as u64 <= max_subtree_len(chunk_counter * CHUNK_LEN as u64).unwrap()));

    // If the whole subtree is one chunk, hash it directly with a ChunkState.
    if input.len() <= CHUNK_LEN {
        return ChunkState::new(key, chunk_counter, flags, platform).update(input).output();
    }

    // Otherwise construct an Output object from the parent node returned by
    // compress_subtree_to_parent_node().
    hasher::Output {
        input_chaining_value: *key,
        block: compress_subtree_to_parent_node(input, 0, key, chunk_counter, flags, platform),
        block_len: BLOCK_LEN as u8,
        counter: 0,
        flags: flags | PARENT,
        platform,
    }
}

/// The hash of `input`, on the calling thread.
///
/// ```
/// # blake3_servil::initialize();
/// let hash = blake3_servil::hash(b"foo");
/// # let hash1 = hash;
///
/// let hash = blake3_servil::Hasher::new().update(b"foo").finalize();
/// # let hash2 = hash;
/// # assert_eq!(hash1, hash2);
/// ```
///
/// Choose it for short inputs, or where your program wants one thread; for
/// any input in memory, [`hash_multithreaded`] is never slower. For output
/// sizes other than 32 bytes, see [`Hasher::finalize_xof`] and
/// [`OutputReader`].
///
/// **Speed**, nonstop on an Apple M4 Max: about 50 ns for 64 bytes, 0.7 µs
/// for 1 KiB, 4 GB/s at 16 KiB, and 6 GB/s from 1 MiB. A call after a pause
/// or after other work takes longer (its code and data have left the
/// CPU's caches): 64 bytes then take about 0.2 to 0.6 µs. On Apple M4 and
/// later, threads that hash inputs of 16 KiB or more at the same moment
/// share one accelerator: one runs at full speed and the others at about
/// two thirds of it.
///
/// The first hash in a process first runs a self-test (see the crate's
/// documentation), under 0.2 ms; [`initialize`] moves it to start-up.
// Inlined into its caller: its one call, out of line, sat apart from
// hash_serial, one more code line to fetch from DRAM in a call after a pause
// (i7-12700K, after idling, 64 B: -16.5%, 8 fresh layouts a side).
#[inline]
pub fn hash(input: &[u8]) -> Hash {
    hash_serial(input, IV, 0)
}

/// The hash of `input`, using other CPU cores where that is faster: the same
/// [`Hash`](struct@Hash) as [`hash`], never slower.
///
/// ```
/// # blake3_servil::initialize_multithreaded(None);
/// let hash = blake3_servil::hash_multithreaded(&[0u8; 1 << 20]);
/// assert_eq!(hash, blake3_servil::hash(&[0u8; 1 << 20]));
/// ```
///
/// The default choice for one message in memory.
///
/// **Speed**, on an Apple M4 Max (16 cores): below 512 KiB it hashes on the
/// calling thread, at [`hash`]'s speed; from 512 KiB it uses the other cores
/// too: about 12 GB/s at 1 MiB, 20 GB/s at 4 MiB, and 25 GB/s from 16 MiB,
/// 4 to 6 times [`hash`]. The extra threads wait asleep between calls and
/// use no CPU then.
///
/// **Several callers at once**: calls from several threads of one program
/// share the cores in turn, each getting a fair part; a call that finds the
/// cores all busy hashes on its own thread, at [`hash`]'s speed.
///
/// **CPUs with performance and efficiency cores** (Intel's since Alder
/// Lake): a multithreaded call or stream uses every core, and afterwards
/// the OS sometimes leaves the calling thread on an efficiency core for a
/// while, where single-threaded hashing runs at about half speed (AVX2's
/// 256-bit operations run as two halves there). Measured on an i7-12700K
/// under Linux 7.0: beside nonstop multithreaded streams, 7% of 16 KiB
/// single-threaded hashes ran partly on an efficiency core, 2.2 times
/// slower, about 6% on their average. A program that alternates the two
/// and needs its single-threaded speed steady can keep that thread on the
/// performance cores (`sched_setaffinity` on Linux).
///
/// The program calls [`initialize_multithreaded`] first; a call before it
/// panics.
#[cfg(feature = "std")]
pub fn hash_multithreaded(input: &[u8]) -> Hash {
    lanes::require();
    lanes::hash(input)
}

/// Run the startup self-test and detect the CPU: every program calls this
/// (or [`initialize_multithreaded`]) once, before any other call of this
/// crate, and a call before it panics. Under 0.2 ms, once per process
/// (see the crate's documentation); later calls return at once.
///
/// ```
/// blake3_servil::initialize();
/// let hash = blake3_servil::hash(b"foo");
/// # assert_eq!(hash, blake3_servil::Hasher::new().update(b"foo").finalize());
/// ```
pub fn initialize() {
    self_test::ensure();
    platform::Platform::detect_unchecked().record();
}

/// [`initialize`], and start the threads that the multithreaded calls and
/// [`stream`] use: every program that makes such calls calls this once,
/// before them, in place of `initialize`. It starts `threads - 1` workers,
/// and each multithreaded call hashes on its calling thread and the
/// workers (calls made at once share the workers), and on CPUs with SME2
/// (Apple M4) the SME2 thread too; `None` means one thread
/// a CPU (`std::thread::available_parallelism`, which follows the CPU
/// affinity the program runs under). With 1, calls hash on their calling
/// thread alone: no workers start, nor the SME2 thread. Otherwise it
/// starts the first worker and returns; the first worker starts the others
/// (each 20-30 us on Linux x86-64), and calls run on the workers started
/// so far. The
/// threads wait asleep whenever no call has work for them (their memory:
/// see [Memory](crate#memory)). Later calls return at once, and must ask
/// for the same number: another `Some` panics. A panic on a worker thread
/// (a bug in this crate) ends the process, its message printed, rather
/// than leaving the call that waits for that worker's piece waiting
/// forever.
///
/// ```
/// blake3_servil::initialize_multithreaded(None);
/// let hash = blake3_servil::hash_multithreaded(&[0u8; 1 << 20]);
/// assert_eq!(hash, blake3_servil::hash(&[0u8; 1 << 20]));
/// ```
#[cfg(feature = "std")]
pub fn initialize_multithreaded(threads: Option<core::num::NonZeroUsize>) {
    static STARTED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cpus = || std::thread::available_parallelism().map_or(1, |n| n.get());
    let started = *STARTED.get_or_init(|| {
        let threads = threads.map_or_else(cpus, |n| n.get());
        initialize();
        lanes::initialize(threads);
        threads
    });
    if let Some(threads) = threads {
        assert_eq!(
            started,
            threads.get(),
            "blake3-servil: the multithreaded calls were started with {started} threads; initialize_multithreaded cannot change that to {threads}"
        );
    }
}

/// An alias to distinguish [`hash_derive_key_context`] outputs from other keys.
pub type ContextKey = [u8; KEY_LEN];

/// Hash a key-derivation context string and return a [`ContextKey`].
///
/// This has the same security requirement as key derivation. **The context
/// string should be hardcoded, globally unique, and application-specific.**
///
/// Its one use is [`Mode::DeriveKeyMaterial`]: a program that derives many
/// keys from one context hashes the context once.
///
/// # Example
///
/// ```
/// # blake3_servil::initialize();
/// use blake3_servil::Mode;
///
/// let context_key = blake3_servil::hash_derive_key_context("foo");
/// let derived = blake3_servil::hash_with(Mode::DeriveKeyMaterial(&context_key), b"bar");
/// assert_eq!(derived, blake3_servil::hash_with(Mode::DeriveKey("foo"), b"bar"));
/// ```
pub fn hash_derive_key_context(context: &str) -> ContextKey {
    hash_serial(context.as_bytes(), IV, DERIVE_KEY_CONTEXT).0
}

/// The most bytes a subtree starting at byte `input_offset` of a message
/// can hold: `None` at offset zero (the whole message), else the largest
/// power-of-two multiple of a chunk the offset is aligned to. Requires a
/// whole-chunk offset.
#[inline(always)]
pub(crate) fn max_subtree_len(input_offset: u64) -> Option<u64> {
    if input_offset == 0 {
        return None;
    }
    assert_eq!(input_offset % CHUNK_LEN as u64, 0);
    let counter = input_offset / CHUNK_LEN as u64;
    let max_chunks = 1 << counter.trailing_zeros();
    Some(max_chunks * CHUNK_LEN as u64)
}

/// Which of BLAKE3's three functions a call computes: the plain hash
/// ([`hash`]), the keyed hash, or key derivation. Every call that takes a mode runs each one at the same
/// speed, but for `DeriveKey`, which first hashes its context string (once
/// per call; [`DeriveKeyMaterial`](Mode::DeriveKeyMaterial) takes it hashed
/// once for many calls).
#[derive(Clone, Copy)]
pub enum Mode<'a> {
    /// The default hash function, as [`hash`].
    Hash,
    /// The keyed hash function with this key.
    Keyed(&'a [u8; KEY_LEN]),
    /// The key derivation function with this context string; the input
    /// is the key material.
    DeriveKey(&'a str),
    /// The key derivation function with its context string already hashed
    /// ([`hash_derive_key_context`]); the input is the key material.
    DeriveKeyMaterial(&'a ContextKey),
}

impl Mode<'_> {
    /// The key words and flags this mode hashes with.
    fn key_and_flags(self) -> (CVWords, u8) {
        match self {
            Mode::Hash => (*IV, 0),
            Mode::Keyed(key) => (platform::words_from_le_bytes_32(key), KEYED_HASH),
            Mode::DeriveKey(context) => (platform::words_from_le_bytes_32(&hash_derive_key_context(context)), DERIVE_KEY_MATERIAL),
            Mode::DeriveKeyMaterial(context_key) => (platform::words_from_le_bytes_32(context_key), DERIVE_KEY_MATERIAL),
        }
    }
}

/// [`hash`] in any [`Mode`]: the plain hash, the keyed hash, or key
/// derivation, at [`hash`]'s speed.
///
/// ```
/// # blake3_servil::initialize();
/// use blake3_servil::Mode;
/// let key = [7u8; 32];
/// assert_eq!(blake3_servil::hash_with(Mode::Keyed(&key), b"foo").as_bytes().len(), 32);
/// ```
pub fn hash_with(mode: Mode, input: &[u8]) -> Hash {
    let (key, flags) = mode.key_and_flags();
    hash_serial(input, &key, flags)
}

/// [`hash_multithreaded`] in any [`Mode`], at its speed.
///
/// ```
/// # blake3_servil::initialize_multithreaded(None);
/// use blake3_servil::Mode;
/// let key = [7u8; 32];
/// let input = vec![1u8; 1 << 20];
/// assert_eq!(blake3_servil::hash_multithreaded_with(Mode::Keyed(&key), &input), blake3_servil::hash_with(Mode::Keyed(&key), &input));
/// ```
#[cfg(feature = "std")]
pub fn hash_multithreaded_with(mode: Mode, input: &[u8]) -> Hash {
    lanes::require();
    let (key, flags) = mode.key_and_flags();
    lanes::hash_with_key(input, &key, flags)
}

/// The whole input on the calling thread, in the given mode: the one-chunk
/// kernel call up to CHUNK_LEN, the subtree walk above. hash(),
/// hash() in its modes and hash_multithreaded()'s short path all come
/// through here.
#[inline]
fn hash_serial(input: &[u8], key: &CVWords, flags: u8) -> Hash {
    hash_serial_from(Platform::detect(), input, key, flags)
}

/// [`hash_serial`] from `platform`: the startup self-test's way in, which
/// has not run `Platform::detect`'s check (it is that check).
#[inline]
fn hash_serial_from(platform: Platform, input: &[u8], key: &CVWords, flags: u8) -> Hash {
    let turn = platform::Sme2Turn::take(platform, input.len() >= SME2_SIZED_LEN);
    // One comparison on the path of a chunk or less: everything else is
    // out of line, so that path's code stays as small as it was (a call
    // after other work fetches every line of it from DRAM).
    #[cfg(all(blake3_neon, feature = "std"))]
    if (CHUNK_LEN + 1..PREFETCH_BELOW).contains(&input.len()) {
        prefetch_after_pause(input.len(), turn.platform(), false);
    }
    hash_serial_on(input, key, flags, turn.platform())
}

/// Below this length a one-shot call that follows a pause prefetches its
/// kernels' code: after other work the call then fetches it from DRAM in
/// parallel, not one miss at a time (M4 Max, after other work: 4 KiB
/// x0.61, 8 KiB x0.51, 16 KiB x0.66; 32 KiB and up level, the hashing's
/// own time dwarfing it; jobs 971-975). The streaming paths (Hasher,
/// batches, the streams' tasks) run back to back with their code near.
/// A cold call of 2-8 KiB spends about a third of its time on its code
/// (10-24 KB of kernels and glue); from 3 KiB what remains is the kernels
/// starting before their lines arrive. Tried and set aside: prefetching
/// for a chunk or less (2-10% gained, less than the pause check costs);
/// prefetching the 16 KiB of Rust around the kernels too (every call
/// 400-500 ns slower: prefetches beyond some number in flight are
/// dropped); a rolled four-lane NEON path for calls after a pause (1.3 KB,
/// 1.2-1.5x slower: a four-lane NEON BLAKE3 is bound by its chains'
/// latency, where the hybrids' integer lanes fill the idle issue slots).
#[cfg(all(blake3_neon, feature = "std"))]
const PREFETCH_BELOW: usize = 64 * 1024;

/// After a pause (code_may_be_cold), prefetch the kernels hashing `len`
/// bytes runs: a one-shot call's (prefetch_kernels), or with `fresh_hasher`
/// a fresh Hasher's first update's (prefetch_update_kernels).
#[cfg(all(blake3_neon, feature = "std"))]
// Cold for layout: the path of a chunk or less falls through past its
// call, on the lines it had before the call existed.
#[cold]
#[inline(never)]
fn prefetch_after_pause(len: usize, platform: Platform, fresh_hasher: bool) {
    if code_may_be_cold() {
        if fresh_hasher {
            prefetch_update_kernels(len, platform);
        } else {
            prefetch_kernels(len, platform);
        }
    }
}

/// Whether this thread's last call in the prefetching range was over
/// 100 us ago (or never). Calls back to back find their code near, where
/// the prefetch would cost them 2-4% (jobs 972-975); one that follows a
/// pause is the one other work may have taken the code from.
#[cfg(all(blake3_neon, feature = "std"))]
#[inline]
fn code_may_be_cold() -> bool {
    std::thread_local! {
        static LAST_CALL: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
    }
    let now = platform::counter();
    let last = LAST_CALL.with(|last| last.replace(now));
    now.wrapping_sub(last) > platform::ticks_in_100_us()
}

/// Batches of one-block messages up to this many prefetch their code after
/// a pause: past it the hashing's own time dwarfs a cold kernel's.
#[cfg(all(blake3_neon, feature = "std"))]
const PREFETCH_BATCH_BELOW: usize = 64;

/// Batches of 1 to this many one-block messages run on the pair kernel
/// alone after a pause (neon_hybrid::hash_pairs): its code (2.9 KB) is a
/// fraction of the plans' for 3 to 16 (up to 17 KB) and less than the one
/// message's kernel (3.9 KB), and after a pause the code comes from memory
/// (Mac jobs 1320-1337: 12 and 16 run faster after idling and slower after
/// other work this way).
#[cfg(all(blake3_neon, feature = "std"))]
const PAIRS_AFTER_PAUSE_UPTO: usize = 8;

/// After a pause (code_may_be_cold), a batch of `count` one-block
/// messages: 1 to PAIRS_AFTER_PAUSE_UPTO hashed here on the pair kernel
/// (true: done); otherwise its kernels' code prefetched (false: hash it as
/// usual), SME2's message kernel where it takes the batch, else the
/// hybrids' parent plans.
#[cfg(all(blake3_neon, feature = "std"))]
#[cold]
#[inline(never)]
fn batch_after_pause(input: &[u8], message_len: usize, key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]], platform: Platform) -> bool {
    if !code_may_be_cold() {
        return false;
    }
    let count = out.len();
    #[cfg(blake3_sme2)]
    if matches!(platform, Platform::SME2) && count >= many::ONE_BLOCK_PAD_MIN {
        sme2::prefetch_code();
        return false;
    }
    let _ = platform;
    if !neon_hybrid::sha3_detected() {
        return false;
    }
    if message_len == 0 || !(1..=PAIRS_AFTER_PAUSE_UPTO).contains(&count) {
        neon_hybrid::prefetch_message_code(count);
        return false;
    }
    many::assert_slots(input, message_len, count);
    neon_hybrid::prefetch_pair_code();
    let mut table: [&[u8; BLOCK_LEN]; PAIRS_AFTER_PAUSE_UPTO] = [&[0; BLOCK_LEN]; PAIRS_AFTER_PAUSE_UPTO];
    for (slot, message) in table.iter_mut().zip(input.chunks_exact(BLOCK_LEN)) {
        *slot = message.try_into().unwrap();
    }
    // Sound: the SHA-3 extension is here (above).
    unsafe { neon_hybrid::hash_pairs(&table[..count], key, flags, CHUNK_START, CHUNK_END | ROOT, message_len, out.as_flattened_mut()) };
    true
}

/// Prefetch the code of the kernels hashing `len` bytes on `platform`
/// runs: SME2's from 16 chunks, else the hybrids', chunks to root.
#[cfg(all(blake3_neon, feature = "std"))]
fn prefetch_kernels(len: usize, platform: Platform) {
    #[cfg(blake3_sme2)]
    if matches!(platform, Platform::SME2) && len >= SME2_SIZED_LEN {
        sme2::prefetch_code();
        // Below the flat walk, sixteen chunks' values merge on the hybrids.
        if !sme2::flat_takes(len) {
            neon_hybrid::prefetch_parent_code(16);
        }
        return;
    }
    // Below 16 chunks the SME2 platform runs the hybrids too.
    #[cfg(blake3_sme2)]
    let hybrids = matches!(platform, Platform::NEON | Platform::SME2);
    #[cfg(not(blake3_sme2))]
    let hybrids = matches!(platform, Platform::NEON);
    if hybrids && neon_hybrid::sha3_detected() {
        let whole = (len / CHUNK_LEN).min(16);
        neon_hybrid::prefetch_tree_code(whole, if whole == 16 { 0 } else { len % CHUNK_LEN });
    }
}

/// Prefetch the code a fresh Hasher's update of `len` bytes runs: the
/// subtrees its loop cuts (hasher::subtree_len from counter 0, up to the last
/// chunk, which waits in the chunk state), each one's kernels.
#[cfg(all(blake3_neon, feature = "std"))]
fn prefetch_update_kernels(len: usize, platform: Platform) {
    if !neon_hybrid::sha3_detected() {
        return;
    }
    let (mut counter, mut left) = (0u64, len);
    while left > CHUNK_LEN {
        let subtree = hasher::subtree_len(counter, left);
        let chunks = subtree / CHUNK_LEN;
        #[cfg(blake3_sme2)]
        if matches!(platform, Platform::SME2) && chunks >= 16 {
            sme2::prefetch_code();
            if !sme2::flat_takes(subtree) {
                neon_hybrid::prefetch_parent_code(16);
            }
            counter += chunks as u64;
            left -= subtree;
            continue;
        }
        neon_hybrid::prefetch_tree_code(chunks.min(16), 0);
        counter += chunks as u64;
        left -= subtree;
    }
    let _ = platform;
}

/// The smallest input the SME2 kernels take part in: one group of sixteen
/// chunks. Below it the SME2 platform runs the NEON hybrids.
pub(crate) const SME2_SIZED_LEN: usize = 16 * CHUNK_LEN;

/// The fewest messages a batch has when the SME2 kernels take part: one
/// group of sixteen messages.
pub(crate) const SME2_SIZED_BATCH: usize = 16;

/// [`hash_serial`] with the tree's kernels chosen by the caller (a pool
/// piece runs the NEON hybrids); one chunk or less runs the scalar kernel
/// on every platform.
#[inline]
fn hash_serial_on(input: &[u8], key: &CVWords, flags: u8, platform: Platform) -> Hash {
    Hash(tree::hash(&tree::PlatformKernels { key, flags, platform }, input))
}

/// The hashes of many messages of one length, laid out in one buffer, on
/// the calling thread: `out[i]` becomes the [`hash`] of message `i`.
///
/// Message `i` starts at byte `i * stride` of `input`, where `stride` is
/// `message_len` rounded up to a multiple of 64 (64 for empty messages), so
/// messages whose length is a multiple of 64 sit back to back. `input`
/// holds exactly `out.len()` strides, and the bytes between a message's end
/// and the next stride are zero (debug builds check; another byte there
/// changes that message's hash). This is the shape of a Merkle tree's
/// layer: leaves of one size, or nodes of two 32-byte children
/// (`message_len` 64).
///
/// ```
/// # blake3_servil::initialize();
/// let leaves = vec![7u8; 1000 * 256];
/// let mut hashes = vec![[0u8; 32]; 1000];
/// blake3_servil::hash_many(&leaves, 256, &mut hashes);
/// assert_eq!(hashes[3], *blake3_servil::hash(&leaves[3 * 256..4 * 256]).as_bytes());
///
/// // 100-byte messages sit 128 bytes apart, zero between them.
/// let mut records = vec![0u8; 10 * 128];
/// for (i, record) in records.chunks_mut(128).enumerate() {
///     record[..100].fill(i as u8 + 1);
/// }
/// let mut digests = vec![[0u8; 32]; 10];
/// blake3_servil::hash_many(&records, 100, &mut digests);
/// assert_eq!(digests[9], *blake3_servil::hash(&[10u8; 100]).as_bytes());
/// ```
///
/// **Speed**: short messages hash several at a time, so a batch beats a
/// loop of [`hash`]: on an Apple M4 Max, 64-byte messages take about 10 to
/// 20 ns each in batches of 16 or more, against about 50 for one [`hash`]
/// each. Messages longer than 15 KiB cost what [`hash`] costs. For the
/// same hashes over several cores, see [`hash_many_multithreaded`].
pub fn hash_many(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    hash_many_serial(input, message_len, IV, 0, out);
}

/// [`hash_many`] in the mode of `key` and `flags`, on the calling thread.
#[inline]
pub(crate) fn hash_many_serial(input: &[u8], message_len: usize, key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]]) {
    hash_many_serial_from(Platform::detect(), input, message_len, key, flags, out);
}

/// [`hash_many_serial`] from `platform` (the self-test's way in, as
/// [`hash_serial_from`]).
#[inline]
fn hash_many_serial_from(platform: Platform, input: &[u8], message_len: usize, key: &CVWords, flags: u8, out: &mut [[u8; OUT_LEN]]) {
    let turn = platform::Sme2Turn::take(platform, many::sme2_sized(message_len, out.len()));
    // A small batch of one-block messages after a pause fetches its kernels'
    // code in parallel (as hash_serial does): its time is mostly that code.
    #[cfg(all(blake3_neon, feature = "std"))]
    if message_len <= BLOCK_LEN && (1..=PREFETCH_BATCH_BELOW).contains(&out.len()) && batch_after_pause(input, message_len, key, flags, out, turn.platform()) {
        return;
    }
    many::hash_many_on(input, message_len, key, flags, out, turn.platform());
}

/// [`hash_many`], using other CPU cores where that is faster: the same
/// hashes, never slower.
///
/// ```
/// # blake3_servil::initialize_multithreaded(None);
/// let leaves = vec![7u8; 4096 * 256];
/// let mut hashes = vec![[0u8; 32]; 4096];
/// blake3_servil::hash_many_multithreaded(&leaves, 256, &mut hashes);
/// assert!(hashes.iter().all(|h| h == blake3_servil::hash(&[7u8; 256]).as_bytes()));
/// ```
///
/// **Speed**: below 512 KiB in all it runs on the calling thread, at
/// [`hash_many`]'s speed; larger batches use the other cores too (on an
/// Apple M4 Max, 64-byte messages from about 10 down to 3 ns each). It
/// shares the cores with other callers as [`hash_multithreaded`] does.
#[cfg(feature = "std")]
pub fn hash_many_multithreaded(input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    lanes::require();
    lanes::hash_many(input, message_len, IV, 0, out);
}

/// [`hash_many`] in any [`Mode`], at its speed.
///
/// ```
/// # blake3_servil::initialize();
/// use blake3_servil::Mode;
/// let key = [7u8; 32];
/// let nodes = vec![1u8; 20 * 64];
/// let mut macs = vec![[0u8; 32]; 20];
/// blake3_servil::hash_many_with(Mode::Keyed(&key), &nodes, 64, &mut macs);
/// assert_eq!(macs[5], *blake3_servil::hash_with(blake3_servil::Mode::Keyed(&key), &[1u8; 64]).as_bytes());
/// ```
pub fn hash_many_with(mode: Mode, input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    let (key, flags) = mode.key_and_flags();
    hash_many_serial(input, message_len, &key, flags, out);
}

/// [`hash_many_multithreaded`] in any [`Mode`], at its speed.
///
/// ```
/// # blake3_servil::initialize_multithreaded(None);
/// use blake3_servil::Mode;
/// let key = [7u8; 32];
/// let nodes = vec![1u8; 20000 * 64];
/// let mut macs = vec![[0u8; 32]; 20000];
/// blake3_servil::hash_many_multithreaded_with(Mode::Keyed(&key), &nodes, 64, &mut macs);
/// assert_eq!(macs[5], *blake3_servil::hash_with(blake3_servil::Mode::Keyed(&key), &[1u8; 64]).as_bytes());
/// ```
#[cfg(feature = "std")]
pub fn hash_many_multithreaded_with(mode: Mode, input: &[u8], message_len: usize, out: &mut [[u8; OUT_LEN]]) {
    lanes::require();
    let (key, flags) = mode.key_and_flags();
    lanes::hash_many(input, message_len, &key, flags, out);
}

/// A message's hash, fed in pieces: [`update`](Hasher::update) with each
/// piece, in order, then [`finalize`](Hasher::finalize). The pieces may be
/// of any lengths; the hash is the same as [`hash`] of them all at once.
///
/// A `Hasher` is about 18 KiB (it holds short pieces until it has 16 KiB
/// to hash at once); box it where that matters.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # blake3_servil::initialize();
/// // Hash an input incrementally.
/// let mut hasher = blake3_servil::Hasher::new();
/// hasher.update(b"foo");
/// hasher.update(b"bar");
/// hasher.update(b"baz");
/// assert_eq!(hasher.finalize(), blake3_servil::hash(b"foobarbaz"));
///
/// // Extended output. OutputReader also implements Read and Seek.
/// # #[cfg(feature = "std")] {
/// let mut output = [0; 1000];
/// let mut output_reader = hasher.finalize_xof();
/// output_reader.fill(&mut output);
/// assert_eq!(&output[..32], blake3_servil::hash(b"foobarbaz").as_bytes());
/// # }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Hasher {
    core: HasherCore,
    /// The message's bytes from its last 16 KiB boundary, gathered while
    /// they arrive in shorter updates: the first `staged`.
    staged: usize,
    stage: [core::mem::MaybeUninit<u8>; STAGE_LEN],
}

/// The bytes a Hasher gathers before they hash side by side: 16 chunks,
/// the widest kernels' lanes.
const STAGE_LEN: usize = 16 * CHUNK_LEN;

/// The tree's state behind a [`Hasher`]: its chunk, and the chaining values
/// of its finished subtrees.
#[derive(Clone)]
pub(crate) struct HasherCore {
    key: CVWords,
    chunk_state: ChunkState,
    initial_chunk_counter: u64,
    // The stack size is MAX_DEPTH + 1 because we do lazy merging. For example,
    // with 7 chunks, we have 3 entries in the stack. Adding an 8th chunk
    // requires a 4th entry, rather than merging everything down to 1, because
    // we don't know whether more input is coming. This is different from how
    // the reference implementation does things.
    cv_stack: CvStack,
}

mod hasher;
use hasher::CvStack;

/// What the Hasher's kernels take from the library (hasher::Hooks): with
/// `pooled`, subtrees of lanes::MIN_SPLIT_LEN and more on the pool; whole
/// subtrees on `platform` (an SME2 turn's); `chunk_platform`, the chunk
/// state's, otherwise.
struct HasherHooks<'a> {
    key: &'a CVWords,
    flags: u8,
    chunk_platform: Platform,
    platform: Platform,
    pooled: bool,
}

impl hasher::Hooks for HasherHooks<'_> {
    #[inline(always)]
    #[allow(unused_variables)]
    fn subtree_cv(&self, input: &[u8], ahead: usize, counter: u64) -> Option<CVBytes> {
        // After the first input, no subtree is the root: on SME2 the flat
        // walk runs it down to its own chaining value, so one value goes
        // on the stack and about one merge runs on the core between SME2
        // kernels (about 0.2 us of other work between them puts the SME
        // unit in its slow state: `sme2`'s documentation).
        #[cfg(blake3_sme2)]
        if !self.pooled && matches!(self.platform, Platform::SME2) && sme2::flat_takes(input.len()) {
            // Safe: the SME2 platform is selected only where the CPU has it.
            return Some(unsafe { sme2::compress_subtree_flat_to_cv(input, ahead, self.key, counter, self.flags) });
        }
        None
    }

    #[inline(always)]
    fn children(&self, input: &[u8], ahead: usize, counter: u64) -> [CVBytes; 2] {
        #[cfg(feature = "std")]
        let pool_takes = self.pooled && input.len() >= lanes::MIN_SPLIT_LEN;
        #[cfg(not(feature = "std"))]
        let pool_takes = false;
        let pair = if pool_takes {
            #[cfg(feature = "std")]
            {
                lanes::subtree_children(input, self.key, counter, self.flags)
            }
            #[cfg(not(feature = "std"))]
            unreachable!()
        } else if self.pooled {
            // Pooled, the subtrees below take the lock one at a time, and
            // the pool's own never use SME2.
            let own = platform::Sme2Turn::take(self.chunk_platform, input.len() >= SME2_SIZED_LEN);
            compress_subtree_to_parent_node(input, ahead, self.key, counter, self.flags, own.platform())
        } else {
            compress_subtree_to_parent_node(input, ahead, self.key, counter, self.flags, self.platform)
        };
        [pair[..OUT_LEN].try_into().unwrap(), pair[OUT_LEN..].try_into().unwrap()]
    }
}

/// The Hasher's kernels: chunks, nodes, and parents on the chunk state's
/// platform, the rest from `hooks`.
type HasherKernels<'a> = hasher::Kernels<'a, Platform, HasherHooks<'a>>;

/// A Hasher's kernels on its key and chunk state (see HasherHooks).
#[inline(always)]
fn kernels_on<'a>(key: &'a CVWords, chunk_state: &ChunkState, platform: Platform, pooled: bool) -> HasherKernels<'a> {
    hasher::Kernels {
        key,
        flags: chunk_state.flags,
        platform: chunk_state.platform,
        hooks: HasherHooks { key, flags: chunk_state.flags, chunk_platform: chunk_state.platform, platform, pooled },
    }
}

impl Hasher {
    fn from_core(core: HasherCore) -> Self {
        Self { core, staged: 0, stage: [core::mem::MaybeUninit::uninit(); STAGE_LEN] }
    }

    /// The bytes gathered in the stage.
    fn staged_bytes(&self) -> &[u8] {
        // Sound: the first `staged` bytes of the stage were written (gather).
        unsafe { core::slice::from_raw_parts(self.stage.as_ptr() as *const u8, self.staged) }
    }

    /// Copy `bytes` into the stage after what it holds.
    fn stage(&mut self, bytes: &[u8]) {
        // Sound: MaybeUninit<u8> has u8's layout, and the bytes fit (gather).
        let room = unsafe { core::slice::from_raw_parts_mut(self.stage.as_mut_ptr().add(self.staged) as *mut u8, bytes.len()) };
        room.copy_from_slice(bytes);
        self.staged += bytes.len();
    }

    /// `update`'s gathering: an update of STAGE_LEN or more from a 16 KiB
    /// boundary of the message (or an empty stage) goes to the core whole;
    /// shorter ones gather in the stage until it reaches the next boundary
    /// (but a fresh hasher's first chunk goes to the core),
    /// and the stage goes to the core as one update, its chunks side by
    /// side in the lanes.
    fn gather(&mut self, input: &[u8], pooled: bool) -> &mut Self {
        self.gather_part(input, pooled, false);
        self
    }

    /// [`gather`](Hasher::gather), but with `hold`, a stage that fills a
    /// whole group (from offset zero) is kept for the caller to hash with
    /// others' (`held_group`), and the rest of `input`
    /// returned; else nothing is left.
    fn gather_part<'a>(&mut self, mut input: &'a [u8], pooled: bool, hold: bool) -> &'a [u8] {
        self.core.assert_room(self.count(), input.len());
        while !input.is_empty() {
            // Whole groups, and a fresh hasher's first chunk (which hashes
            // serially either way), skip the stage.
            if self.staged == 0 && (input.len() >= STAGE_LEN || self.core.count() + input.len() as u64 <= CHUNK_LEN as u64) {
                self.core.update_pooled(input, pooled);
                return &[];
            }
            let at = (self.core.initial_chunk_counter * CHUNK_LEN as u64 + self.count()) % STAGE_LEN as u64;
            let to_boundary = STAGE_LEN - at as usize;
            let take = cmp::min(to_boundary, input.len());
            // The loop's variant: every pass stages at least one byte of
            // the input, which is not empty (`at` is below STAGE_LEN).
            assert!(take > 0, "the stage took nothing");
            self.stage(&input[..take]);
            input = &input[take..];
            if take < to_boundary {
                return &[];
            }
            if hold && self.held_group().is_some() {
                return input;
            }
            let staged = core::mem::replace(&mut self.staged, 0);
            // Sound: the first `staged` bytes were written; the core reads
            // them before the stage is written again.
            let bytes = unsafe { core::slice::from_raw_parts(self.stage.as_ptr() as *const u8, staged) };
            self.core.update_pooled(bytes, pooled);
        }
        &[]
    }

    /// A whole group in the stage, from offset zero: its first chunk's
    /// counter and its bytes.
    #[cfg(feature = "std")]
    fn held_group(&self) -> Option<(u64, &[u8; STAGE_LEN])> {
        (self.staged == STAGE_LEN && self.core.initial_chunk_counter == 0)
            .then(|| (self.core.count() / CHUNK_LEN as u64, self.staged_bytes().try_into().unwrap()))
    }

    #[cfg(not(feature = "std"))]
    fn held_group(&self) -> Option<(u64, &[u8; STAGE_LEN])> {
        None
    }

    /// The held group's subtree, as [`HasherCore::update_with_results`]
    /// takes it (its two halves' values at chunk zero, else its own), into
    /// the core; the stage empty again.
    #[cfg(feature = "std")]
    fn take_held(&mut self, result: &[u8; BLOCK_LEN]) {
        let staged = core::mem::replace(&mut self.staged, 0);
        // Sound: as in gather_part.
        let bytes = unsafe { core::slice::from_raw_parts(self.stage.as_ptr() as *const u8, staged) };
        self.core.update_with_results(bytes, &mut core::iter::once(result));
    }

    /// The root's output: the core's, after the stage's bytes.
    fn final_output(&self) -> Output {
        if self.staged == 0 {
            return self.core.final_output();
        }
        let mut core = self.core.clone();
        core.update(self.staged_bytes());
        core.final_output()
    }

    /// Construct a new `Hasher` for the regular hash function.
    pub fn new() -> Self {
        Self::from_core(HasherCore::new_internal(IV, 0))
    }

    /// Construct a new `Hasher` in a [`Mode`]: the keyed hash, or key
    /// derivation with a context string or a context hashed once.
    pub fn new_with(mode: Mode) -> Self {
        let (key, flags) = mode.key_and_flags();
        Self::from_core(HasherCore::new_internal(&key, flags))
    }

    /// Reset the `Hasher` to its initial state.
    ///
    /// This is functionally the same as overwriting the `Hasher` with a new
    /// one, using the same key or context string if any.
    pub fn reset(&mut self) -> &mut Self {
        self.core.reset();
        self.staged = 0;
        self
    }

    /// Add the message's next piece, of any length, on the calling thread.
    ///
    /// **Speed**: pieces of any size hash at nearly the speed of whole
    /// buffers: short pieces are copied into the `Hasher` until it holds
    /// 16 KiB, which then hash at once. Pieces of 16 KiB and more, starting
    /// at a multiple of 16 KiB of the message, skip the copy (about 5% of the
    /// time on an Apple M4). For a whole message in memory, [`hash`] is a
    /// little faster; for a file or reader, see
    /// [`update_reader`](#method.update_reader).
    pub fn update(&mut self, input: &[u8]) -> &mut Self {
        // A fresh hasher's first update is a one-shot call's work (a
        // Hasher per message, the digest traits): it prefetches its code
        // after a pause, as hash() does.
        #[cfg(all(blake3_neon, feature = "std"))]
        if self.count() == 0 && (CHUNK_LEN + 1..PREFETCH_BELOW).contains(&input.len()) {
            prefetch_after_pause(input.len(), self.core.chunk_state.platform, true);
        }
        self.gather(input, false)
    }

    /// [`update`](Hasher::update), using other CPU cores where that is
    /// faster: the same result, never slower. Pieces of 512 KiB and more
    /// use them, as [`hash_multithreaded`] does; shorter ones hash on the
    /// calling thread. One `Hasher` may mix the two calls. When reading
    /// the pieces keeps your thread busy, a [`stream`] is faster
    /// ([`Stream::submit_reader`], or [`Stream::submit_with`]): it hashes
    /// each piece while your thread reads the next.
    ///
    /// ```
    /// # blake3_servil::initialize_multithreaded(None);
    /// let input = vec![7u8; 3 << 20];
    /// let mut hasher = blake3_servil::Hasher::new();
    /// for piece in input.chunks(64 * 1024) {
    ///     hasher.update_multithreaded(piece);
    /// }
    /// assert_eq!(hasher.finalize(), blake3_servil::hash(&input));
    /// ```
    #[cfg(feature = "std")]
    pub fn update_multithreaded(&mut self, input: &[u8]) -> &mut Self {
        lanes::require();
        self.gather(input, true)
    }

    /// Finalize the hash state and return the [`Hash`](struct.Hash.html) of
    /// the input.
    ///
    /// This method is idempotent. Calling it twice will give the same result.
    /// You can also add more input and finalize again.
    pub fn finalize(&self) -> Hash {
        assert_eq!(self.core.initial_chunk_counter, 0, "set_input_offset must be used with finalize_non_root",);
        self.final_output().root_hash()
    }

    /// Finalize the hash state and return an [`OutputReader`], which can
    /// supply any number of output bytes.
    ///
    /// This method is idempotent. Calling it twice will give the same result.
    /// You can also add more input and finalize again.
    ///
    /// [`OutputReader`]: struct.OutputReader.html
    pub fn finalize_xof(&self) -> OutputReader {
        assert_eq!(self.core.initial_chunk_counter, 0, "set_input_offset must be used with finalize_non_root",);
        OutputReader::new(self.final_output())
    }

    /// The number of bytes passed to [`update`](Hasher::update) so far.
    pub fn count(&self) -> u64 {
        self.core.count() + self.staged as u64
    }

    /// As [`update`](Hasher::update), but reading from a
    /// [`std::io::Read`](https://doc.rust-lang.org/std/io/trait.Read.html) implementation.
    ///
    /// [`Hasher`] implements
    /// [`std::io::Write`](https://doc.rust-lang.org/std/io/trait.Write.html), so it's possible to
    /// use [`std::io::copy`](https://doc.rust-lang.org/std/io/fn.copy.html) to update a [`Hasher`]
    /// from any reader. Unfortunately, this standard approach can limit performance, because
    /// `copy` currently uses an internal 8 KiB buffer that isn't big enough to take advantage of
    /// all SIMD instruction sets. (In particular, [AVX-512](https://en.wikipedia.org/wiki/AVX-512)
    /// needs a 16 KiB buffer.) `update_reader` avoids this performance problem and is slightly
    /// more convenient.
    ///
    /// The internal buffer size this method uses may change at any time, and it may be different
    /// for different targets. The only guarantee is that it will be large enough for all of this
    /// crate's SIMD implementations on the current platform.
    ///
    /// The most common implementer of
    /// [`std::io::Read`](https://doc.rust-lang.org/std/io/trait.Read.html) might be
    /// [`std::fs::File`](https://doc.rust-lang.org/std/fs/struct.File.html).
    ///
    /// This method requires the `std` Cargo feature, which is enabled by default.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::fs::File;
    /// # use std::io;
    /// # fn main() -> io::Result<()> {
    /// # blake3_servil::initialize();
    /// // Hash standard input.
    /// let mut hasher = blake3_servil::Hasher::new();
    /// hasher.update_reader(std::io::stdin().lock())?;
    /// println!("{}", hasher.finalize());
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "std")]
    pub fn update_reader(&mut self, reader: impl std::io::Read) -> std::io::Result<&mut Self> {
        io::copy_wide(reader, self)?;
        Ok(self)
    }
}

impl HasherCore {
    pub(crate) fn new_internal(key: &CVWords, flags: u8) -> Self {
        Self::new_from(key, flags, Platform::detect())
    }

    /// [`HasherCore::new_internal`] on `platform` (the self-test's way in,
    /// as [`hash_serial_from`]).
    pub(crate) fn new_from(key: &CVWords, flags: u8, platform: Platform) -> Self {
        Self { key: *key, chunk_state: ChunkState::new(key, 0, flags, platform), initial_chunk_counter: 0, cv_stack: CvStack::new() }
    }

    /// Back to the empty message, with the same key, flags, and platform.
    pub(crate) fn reset(&mut self) -> &mut Self {
        self.chunk_state = ChunkState::new(&self.key, 0, self.chunk_state.flags, self.chunk_state.platform);
        self.cv_stack.clear();
        self
    }

    // As described in push_cv() below, we do "lazy merging", delaying merges
    // until right before the next CV is about to be added. This is different
    // from the reference implementation. Another difference is that we aren't
    // always merging 1 chunk at a time. Instead, each CV might represent any
    // power-of-two number of chunks, as long as the smaller-above-larger stack
    // order is maintained. Instead of the "count the trailing 0-bits"
    // algorithm described in the spec (which assumes you're adding one chunk
    // at a time), we use a "count the total number of 1-bits" variant (which
    // doesn't assume that). The principle is the same: each CV that should
    // remain in the stack is represented by a 1-bit in the total number of
    // chunks (or bytes) so far.
    #[cfg(feature = "std")]
    fn merge_cv_stack(&mut self, chunk_counter: u64) {
        // Its callers (the queue's hashers, and a Hasher's held group) start
        // at offset zero; a positioned subtree's offset takes the core's path.
        debug_assert_eq!(self.initial_chunk_counter, 0, "a hasher at offset zero");
        let post_merge_stack_len = hasher::ones(chunk_counter);
        let kernels = kernels_on(&self.key, &self.chunk_state, self.chunk_state.platform, false);
        self.cv_stack.merge(&kernels, post_merge_stack_len);
    }

    /// The kernels on this Hasher's key and flags (see HasherKernels).
    #[inline(always)]
    fn kernels(&self, platform: Platform, pooled: bool) -> HasherKernels<'_> {
        kernels_on(&self.key, &self.chunk_state, platform, pooled)
    }

    // In reference_impl.rs, we merge the new CV with existing CVs from the
    // stack before pushing it. We can do that because we know more input is
    // coming, so we know none of the merges are root.
    //
    // This setting is different. We want to feed as much input as possible to
    // the walk (`tree::wide`), without setting aside anything for the
    // chunk_state. If the user gives us 64 KiB, we want to parallelize over
    // all 64 KiB at once as a single subtree, if at all possible.
    //
    // This leads to two problems:
    // 1) This 64 KiB input might be the only call that ever gets made to
    //    update. In this case, the root node of the 64 KiB subtree would be
    //    the root node of the whole tree, and it would need to be ROOT
    //    finalized. We can't compress it until we know.
    // 2) This 64 KiB input might complete a larger tree, whose root node is
    //    similarly going to be the root of the whole tree. For example,
    //    maybe we have 196 KiB (that is, 128 + 64) hashed so far. We can't
    //    compress the node at the root of the 256 KiB subtree until we know
    //    how to finalize it.
    //
    // The second problem is solved with "lazy merging". That is, when we're
    // about to add a CV to the stack, we don't merge it with anything first,
    // as the reference impl does. Instead we do merges using the *previous* CV
    // that was added, which is sitting on top of the stack, and we put the new
    // CV (unmerged) on top of the stack afterwards. This guarantees that we
    // never merge the root node until finalize().
    //
    // Solving the first problem requires an additional tool,
    // compress_subtree_to_parent_node(). That function always returns the top
    // *two* chaining values of the subtree it's compressing. We then do lazy
    // merging with each of them separately, so that the second CV will always
    // remain unmerged. (That also helps us support extendable output when
    // we're hashing an input all-at-once.)
    #[cfg(feature = "std")]
    fn push_cv(&mut self, new_cv: &CVBytes, chunk_counter: u64) {
        self.merge_cv_stack(chunk_counter);
        self.cv_stack.push(*new_cv);
    }

    /// [`update`](Hasher::update) with every whole subtree's result taken
    /// from `results`, one a whole subtree in order (in parts of
    /// [`lanes::TASK_LEN`], as [`hasher::subtree_len`] cuts them), as
    /// [`lanes::Task`] computes them.
    #[cfg(feature = "std")]
    pub(crate) fn update_with_results<'a>(&mut self, input: &[u8], results: &mut impl Iterator<Item = &'a [u8; BLOCK_LEN]>) {
        for part in input.chunks(lanes::TASK_LEN) {
            self.update_part_with_results(part, results);
        }
    }

    #[cfg(feature = "std")]
    fn update_part_with_results<'a>(&mut self, mut input: &[u8], results: &mut impl Iterator<Item = &'a [u8; BLOCK_LEN]>) {
        // Its one caller hands it a whole held group, from a group boundary,
        // where the core's chunk is empty (Hasher::take_held).
        debug_assert_eq!(self.chunk_state.count(), 0, "a held group starts where the core's chunk is empty");
        while input.len() > CHUNK_LEN {
            let len = hasher::subtree_len(self.chunk_state.chunk_counter, input.len());
            let chunks = (len / CHUNK_LEN) as u64;
            let result = results.next().expect("a result for every planned subtree");
            let counter = self.chunk_state.chunk_counter;
            if counter == 0 && len > CHUNK_LEN {
                self.push_cv(result[..OUT_LEN].try_into().unwrap(), counter);
                self.push_cv(result[OUT_LEN..].try_into().unwrap(), counter + chunks / 2);
            } else {
                self.push_cv(result[..OUT_LEN].try_into().unwrap(), counter);
            }
            self.chunk_state.chunk_counter += chunks;
            input = &input[len..];
        }
        if !input.is_empty() {
            self.chunk_state.update(input);
            self.merge_cv_stack(self.chunk_state.chunk_counter);
        }
    }

    /// `update`'s loop; with `pooled`, whole subtrees of
    /// `lanes::MIN_SPLIT_LEN` and more go to the pool.
    pub(crate) fn update_pooled(&mut self, input: &[u8], pooled: bool) -> &mut Self {
        #[cfg(not(feature = "std"))]
        let _ = pooled;
        self.assert_room(self.count(), input.len());
        // The partial chunk first (hasher::fill), then whole subtrees and the
        // last chunk (hasher::feed), as src/hasher.rs describes.
        let fill_kernels = kernels_on(&self.key, &self.chunk_state, self.chunk_state.platform, pooled);
        let rest = hasher::fill(&fill_kernels, &mut self.chunk_state, &mut self.cv_stack, self.initial_chunk_counter, input);
        if rest.is_empty() {
            return self;
        }
        // Pooled, the subtrees take the lock one at a time (children).
        let turn = platform::Sme2Turn::take(self.chunk_state.platform, !pooled && rest.len() >= SME2_SIZED_LEN);
        let kernels = kernels_on(&self.key, &self.chunk_state, turn.platform(), pooled);
        hasher::feed(&kernels, &mut self.chunk_state, &mut self.cv_stack, self.initial_chunk_counter, rest);
        self
    }

    pub(crate) fn final_output(&self) -> Output {
        hasher::final_node(&self.kernels(self.chunk_state.platform, false), &self.chunk_state, &self.cv_stack)
    }

    /// The serial update: whole subtrees now, the last chunk kept.
    pub(crate) fn update(&mut self, input: &[u8]) -> &mut Self {
        self.update_pooled(input, false)
    }

    /// Requires that `len` more bytes after `count` fit the subtree that
    /// set_input_offset started (max_subtree_len).
    pub(crate) fn assert_room(&self, count: u64, len: usize) {
        let input_offset = self.initial_chunk_counter * CHUNK_LEN as u64;
        if let Some(max) = max_subtree_len(input_offset) {
            assert!(len as u64 <= max - count, "the subtree starting at {} contains at most {} bytes (found {})", input_offset, max, count + len as u64,);
        }
    }

    /// Start the subtree at byte `offset` of the message (the outboard's
    /// positioned subtrees; set_input_offset in the tests).
    #[cfg(any(feature = "std", test))]
    pub(crate) fn set_input_offset(&mut self, offset: u64) -> &mut Self {
        assert_eq!(self.count(), 0, "hasher has already accepted input");
        assert_eq!(offset % CHUNK_LEN as u64, 0, "offset ({offset}) must be a chunk boundary (divisible by {CHUNK_LEN})",);
        let counter = offset / CHUNK_LEN as u64;
        self.chunk_state.chunk_counter = counter;
        self.initial_chunk_counter = counter;
        self
    }

    /// Bytes so far, counted from the input offset.
    pub(crate) fn count(&self) -> u64 {
        (self.chunk_state.chunk_counter - self.initial_chunk_counter) * CHUNK_LEN as u64 + self.chunk_state.count() as u64
    }
}

// Don't derive(Debug), because the state may be secret.
impl fmt::Debug for Hasher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Hasher").field("flags", &self.core.chunk_state.flags).field("platform", &self.core.chunk_state.platform).finish()
    }
}

impl Default for Hasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl std::io::Write for Hasher {
    /// This is equivalent to [`update`](#method.update).
    #[inline]
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        self.update(input);
        Ok(input.len())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for Hasher {
    fn zeroize(&mut self) {
        self.core.zeroize();
        self.stage.fill(core::mem::MaybeUninit::new(0));
        self.staged = 0;
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for HasherCore {
    fn zeroize(&mut self) {
        // Destructuring to trigger compile error as a reminder to update this impl.
        let Self { key, chunk_state, initial_chunk_counter, cv_stack } = self;

        key.zeroize();
        chunk_state.zeroize();
        initial_chunk_counter.zeroize();
        cv_stack.zeroize();
    }
}

/// An incremental reader for extended output, returned by
/// [`Hasher::finalize_xof`](struct.Hasher.html#method.finalize_xof).
///
/// Shorter BLAKE3 outputs are prefixes of longer ones, and explicitly requesting a short output is
/// equivalent to truncating the default-length output. Note that this is a difference between
/// BLAKE2 and BLAKE3.
///
/// # Security notes
///
/// Outputs shorter than the default length of 32 bytes (256 bits) provide less security. An N-bit
/// BLAKE3 output is intended to provide N bits of first and second preimage resistance and N/2
/// bits of collision resistance, for any N up to 256. Longer outputs don't provide any additional
/// security.
///
/// Avoid relying on the secrecy of the output offset, that is, the number of output bytes read or
/// the arguments to [`seek`](struct.OutputReader.html#method.seek) or
/// [`set_position`](struct.OutputReader.html#method.set_position). [_Block-Cipher-Based Tree
/// Hashing_ by Aldo Gunsing](https://eprint.iacr.org/2022/283) shows that an attacker who knows
/// both the message and the key (if any) can easily determine the offset of an extended output.
/// For comparison, AES-CTR has a similar property: if you know the key, you can decrypt a block
/// from an unknown position in the output stream to recover its block index. Callers with strong
/// secret keys aren't affected in practice, but secret offsets are a [design
/// smell](https://en.wikipedia.org/wiki/Design_smell) in any case.
#[derive(Clone)]
pub struct OutputReader {
    inner: Output,
    position_within_block: u8,
}

impl OutputReader {
    fn new(inner: Output) -> Self {
        Self { inner, position_within_block: 0 }
    }

    // This helper function handles both the case where the output buffer is
    // shorter than one block, and the case where our position_within_block is
    // non-zero.
    fn fill_one_block(&mut self, buf: &mut &mut [u8]) {
        let output_block: [u8; BLOCK_LEN] = self.inner.root_output_block();
        let output_bytes = &output_block[self.position_within_block as usize..];
        let take = cmp::min(buf.len(), output_bytes.len());
        buf[..take].copy_from_slice(&output_bytes[..take]);
        self.position_within_block += take as u8;
        if self.position_within_block == BLOCK_LEN as u8 {
            self.inner.counter += 1;
            self.position_within_block = 0;
        }
        // Advance the dest buffer. mem::take() is a borrowck workaround.
        *buf = &mut core::mem::take(buf)[take..];
    }

    /// Fill a buffer with output bytes and advance the position of the
    /// `OutputReader`. This is equivalent to [`Read::read`], except that it
    /// doesn't return a `Result`. Both methods always fill the entire buffer.
    ///
    /// Note that `OutputReader` doesn't buffer output bytes internally, so
    /// calling `fill` repeatedly with a short-length or odd-length slice will
    /// end up performing the same compression multiple times. If you're
    /// reading output in a loop, prefer a slice length that's a multiple of
    /// [`BLOCK_LEN`] (64 bytes).
    ///
    /// The maximum output size of BLAKE3 is 2<sup>64</sup>-1 bytes. If you try
    /// to extract more than that, for example by seeking near the end and
    /// reading further, the behavior is unspecified.
    ///
    /// [`Read::read`]: #method.read
    pub fn fill(&mut self, mut buf: &mut [u8]) {
        if buf.is_empty() {
            return;
        }

        // If we're partway through a block, try to get to a block boundary.
        if self.position_within_block != 0 {
            self.fill_one_block(&mut buf);
        }

        let full_blocks = buf.len() / BLOCK_LEN;
        let full_blocks_len = full_blocks * BLOCK_LEN;
        if full_blocks > 0 {
            debug_assert_eq!(0, self.position_within_block);
            self.inner.platform.xof_many(
                &self.inner.input_chaining_value,
                &self.inner.block,
                self.inner.block_len,
                self.inner.counter,
                self.inner.flags | ROOT,
                &mut buf[..full_blocks_len],
            );
            self.inner.counter += full_blocks as u64;
            buf = &mut buf[full_blocks * BLOCK_LEN..];
        }

        if !buf.is_empty() {
            debug_assert!(buf.len() < BLOCK_LEN);
            self.fill_one_block(&mut buf);
            debug_assert!(buf.is_empty());
        }
    }

    /// Return the current read position in the output stream. This is
    /// equivalent to [`Seek::stream_position`], except that it doesn't return
    /// a `Result`. The position of a new `OutputReader` starts at 0, and each
    /// call to [`fill`] or [`Read::read`] moves the position forward by the
    /// number of bytes read.
    ///
    /// [`Seek::stream_position`]: #method.stream_position
    /// [`fill`]: #method.fill
    /// [`Read::read`]: #method.read
    pub fn position(&self) -> u64 {
        self.inner.counter * BLOCK_LEN as u64 + self.position_within_block as u64
    }

    /// Seek to a new read position in the output stream. This is equivalent to
    /// calling [`Seek::seek`] with [`SeekFrom::Start`], except that it doesn't
    /// return a `Result`.
    ///
    /// [`Seek::seek`]: #method.seek
    /// [`SeekFrom::Start`]: https://doc.rust-lang.org/std/io/enum.SeekFrom.html
    pub fn set_position(&mut self, position: u64) {
        self.position_within_block = (position % BLOCK_LEN as u64) as u8;
        self.inner.counter = position / BLOCK_LEN as u64;
    }
}

// Don't derive(Debug), because the state may be secret.
impl fmt::Debug for OutputReader {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("OutputReader").field("position", &self.position()).finish()
    }
}

#[cfg(feature = "std")]
impl std::io::Read for OutputReader {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.fill(buf);
        Ok(buf.len())
    }
}

#[cfg(feature = "std")]
impl std::io::Seek for OutputReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let max_position = u64::MAX as i128;
        let target_position: i128 = match pos {
            std::io::SeekFrom::Start(x) => x as i128,
            std::io::SeekFrom::Current(x) => self.position() as i128 + x as i128,
            std::io::SeekFrom::End(_) => {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek from end not supported"));
            }
        };
        if target_position < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before start"));
        }
        self.set_position(cmp::min(target_position, max_position) as u64);
        Ok(self.position())
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for OutputReader {
    fn zeroize(&mut self) {
        // Destructuring to trigger compile error as a reminder to update this impl.
        let Self { inner, position_within_block } = self;

        inner.zeroize();
        position_within_block.zeroize();
    }
}
