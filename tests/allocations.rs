//! The crate's "Memory" section, held to its word: the single-threaded
//! calls allocate nothing, and `hash_each` only 32 bytes for each chunk of
//! its messages of 2 to 15 chunks and up to 128 for each such message. A
//! counting allocator, in a process of its own, counts the bytes each call
//! asks for.
#![cfg(feature = "std")]

#[path = "../tools/require_nextest.rs"]
mod require_nextest;

use blake3_servil::throughput::{self, Batch, Block, Hasher};
use blake3_servil::{Mode, convenience};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;

static BYTES: AtomicUsize = AtomicUsize::new(0);

// Sound: every call goes on to the system allocator, unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The bytes `call` asks the allocator for.
fn allocated(call: impl FnOnce()) -> usize {
    let before = BYTES.load(Relaxed);
    call();
    BYTES.load(Relaxed) - before
}

fn blocks(len: usize, seed: u8) -> Vec<Block> {
    let mut blocks = vec![Block::ZERO; len.div_ceil(64).max(1)];
    for (i, b) in Block::bytes_mut(&mut blocks).iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(seed);
    }
    blocks
}

#[test]
fn single_threaded_calls_allocate_as_the_memory_section_says() {
    blake3_servil::initialize();
    let input = blocks(64 * 1024, 1);
    let bytes = Block::bytes(&input);
    let mut out = [0u8; 100];
    for len in [0, 1, 64, 1024, 1025, 16 * 1024, 64 * 1024] {
        assert_eq!(
            allocated(|| {
                std::hint::black_box(convenience::hash(&bytes[..len]));
            }),
            0,
            "convenience::hash of {len} B"
        );
        assert_eq!(
            allocated(|| {
                std::hint::black_box(throughput::hash(Mode::Plain, &input, len));
            }),
            0,
            "throughput::hash of {len} B"
        );
        let mut hasher = Hasher::new(Mode::Plain);
        assert_eq!(
            allocated(|| {
                std::hint::black_box(hasher.update(&bytes[..len]).finalize());
            }),
            0,
            "Hasher of {len} B"
        );
        assert_eq!(allocated(|| hasher.finalize_xof().fill(&mut out)), 0, "OutputReader after {len} B");
    }
    let batch = Batch::from_blocks(&input[..64], 64);
    let mut digests = vec![[0u8; 32]; batch.len()];
    assert_eq!(allocated(|| throughput::hash_many(Mode::Plain, &batch, &mut digests)), 0, "hash_many");

    // A collection: items of up to a chunk allocate nothing; a message of
    // 2 to 15 chunks, 32 bytes a chunk.
    let lens = [0usize, 1, 63, 64, 65, 300, 700, 1000, 1024];
    let small: Vec<Vec<Block>> = (0..100).map(|i| blocks(lens[i % lens.len()], i as u8)).collect();
    let items: Vec<(&[Block], usize)> = small.iter().zip((0..100).map(|i| lens[i % lens.len()])).map(|(b, l)| (&b[..], l)).collect();
    let mut digests = vec![[0u8; 32]; items.len()];
    assert_eq!(allocated(|| throughput::hash_each(Mode::Plain, &items, &mut digests)), 0, "hash_each of 100 items of up to a chunk");
    let long = blocks(5 * 1024, 9);
    let mut mixed = items.clone();
    mixed.push((&long[..], 5 * 1024));
    let mut digests = vec![[0u8; 32]; mixed.len()];
    let bytes = allocated(|| throughput::hash_each(Mode::Plain, &mixed, &mut digests));
    assert!(bytes <= 32 * 5 + 128, "hash_each with one message of 5 chunks: {bytes} B");
    // Forty such messages: 32 bytes a chunk, and the bookkeeping a message.
    let longs: Vec<(&[Block], usize)> = (0..40).map(|_| (&long[..], 5 * 1024)).collect();
    let mut digests = vec![[0u8; 32]; longs.len()];
    let bytes = allocated(|| throughput::hash_each(Mode::Plain, &longs, &mut digests));
    assert!(bytes <= 40 * (32 * 5 + 128), "hash_each of 40 messages of 5 chunks: {bytes} B");
}
