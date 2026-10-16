//! A stream allocates nothing once a thread has used one: its storage is
//! the thread's from stream to stream. Every allocation in the process, on
//! any thread, is counted, and after warm-up rounds more rounds of every
//! kind of message must make none. Its own test binary, without libtest's
//! harness, so nothing allocates beside it (the harness's notice that a test
//! has run over 60 seconds allocates on the main thread). The pool needs
//! threads, which wasm lacks; there the binary does nothing.
#![cfg_attr(any(target_family = "wasm", miri), allow(unused))]

use blake3_servil::{Hash, Mode};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Run `round` `warm` times, then `rounds` times more, and return the
/// allocations those made.
fn allocations_after_warm_up(warm: usize, rounds: usize, mut round: impl FnMut()) -> usize {
    for _ in 0..warm {
        round();
    }
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    for _ in 0..rounds {
        round();
    }
    ALLOCATIONS.load(Ordering::SeqCst) - before
}

fn main() {
    blake3_servil::initialize();
    #[cfg(not(any(target_family = "wasm", miri)))]
    a_warm_stream_allocates_nothing();
    println!("a warm stream allocates nothing: ok");
}

#[cfg(not(any(target_family = "wasm", miri)))]
fn a_warm_stream_allocates_nothing() {
    blake3_servil::initialize_multithreaded();
    // The threads start in the background (each start allocates, for the
    // standard library's stack-overflow handler), within about 1 ms: the
    // first 100 ms are their warm-up.
    let made = allocations_after_warm_up(1, 1, || std::thread::sleep(std::time::Duration::from_millis(100)));
    assert_eq!(made, 0, "the pool's threads settling after initialize_multithreaded: allocations");
    let messages: Vec<Vec<u8>> = [64usize, 1024, 64 << 10, (1 << 20) + 5].iter().map(|&len| vec![7u8; len]).collect();
    let batch = vec![1u8; 64 * 16384];
    let mut sum = [0u8; 32];
    // Every place, made before the count starts.
    let mut places = vec![Hash::from_bytes([0; 32]); 3 * messages.len() + 16 + 16384];
    let made = allocations_after_warm_up(20, 200, || {
        let (singles, batches) = places.split_at_mut(3 * messages.len());
        let (small, large) = batches.split_at_mut(16);
        let mut singles = singles.iter_mut();
        blake3_servil::stream(Mode::Hash, |hashes: &mut [Hash]| hashes.iter().for_each(|h| sum[0] ^= h.as_bytes()[0]), |s| {
            for message in &messages {
                s.submit(message, singles.next().unwrap());
                s.submit_with(message.len(), |at, room| room.copy_from_slice(&message[at..][..room.len()]), singles.next().unwrap());
                s.submit_reader(&message[..], None, singles.next().unwrap()).unwrap();
            }
            s.submit_batch(&batch[..64 * 16], 64, small);
            s.submit_batch(&batch, 64, large);
            s.finish();
        });
    });
    std::hint::black_box(sum);
    assert_eq!(made, 0, "streams after warm-up: allocations");
}
