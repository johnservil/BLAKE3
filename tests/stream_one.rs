//! Streams and their kin on a pool of one thread (BLAKE3 #244): with no
//! worker threads, every task is hashed on the calling thread as it is
//! handed over, so the stream's waits can never be woken from elsewhere.
//! A wait that finds nothing done and nothing to run there is a broken
//! invariant and stops at once; these tests run the stream's kinds of
//! message on that pool, where the time-out mutants of the stream's
//! waits now fail in the act instead of hanging for the test clock.
//! Each test process starts its own pool, so this file starts it with
//! one thread, as `pool_threads` starts its with several.

use blake3_servil::{convenience, throughput, Hash, Mode};
use std::num::NonZeroUsize;

fn input(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u32 * 31 + seed as u32) as u8).collect()
}

fn pool_of_one() {
    blake3_servil::initialize_multithreaded(NonZeroUsize::new(1));
}

/// Many messages at once on one thread, more than the ring holds in
/// flight, of the stream's shapes: whole by reference and copied in,
/// empty, a block, and one long message; every digest in its place,
/// every place handed back in order.
#[test]
fn a_stream_on_one_thread_hands_everything_back() {
    pool_of_one();
    let lens: Vec<usize> = (0..240)
        .map(|i| [0, 1, 63, 64, 65, 1448, 4096, 65_537][i % 8])
        .chain([3 * 1024 * 1024])
        .collect();
    let messages: Vec<Vec<u8>> = lens.iter().enumerate().map(|(i, &len)| input(len, i as u8)).collect();
    let mut places = vec![Hash::from_bytes([0; 32]); messages.len()];
    let want: Vec<*const Hash> = places.iter().map(|p| p as *const Hash).collect();
    let mut order = Vec::new();
    throughput::stream(
        Mode::Plain,
        |done: &mut [Hash]| order.extend(done.iter().map(|p| p as *const Hash as usize)),
        |s| {
            for (i, (message, place)) in messages.iter().zip(&mut places).enumerate() {
                let at = place as *mut Hash;
                match i % 2 {
                    0 => s.submit_with(
                        message.len(),
                        |off, room| {
                            room.copy_from_slice(&message[off..][..room.len()]);
                            room.len()
                        },
                        // Sound: the stream holds the place until it hands it back.
                        unsafe { &mut *at },
                    ),
                    _ => assert_eq!(
                        s.submit_reader(&message[..], Some(message.len() as u64), unsafe { &mut *at }).unwrap(),
                        message.len() as u64
                    ),
                }
            }
            s.finish();
        },
    );
    assert_eq!(
        order,
        want.iter().map(|p| *p as usize).collect::<Vec<_>>(),
        "every place handed back once, in order"
    );
    for (message, place) in messages.iter().zip(&places) {
        assert_eq!(place, &convenience::hash(message), "{} bytes", message.len());
    }
}

/// One long message in fragments, on one thread: the incremental
/// multithreaded update hashes on the caller, and the digest is the
/// one-shot digest.
#[test]
fn a_long_message_in_pieces_on_one_thread() {
    pool_of_one();
    let message = input(3 * 1024 * 1024 + 17, 7);
    let mut hasher = throughput::Hasher::new(Mode::Plain);
    let mut at = 0;
    for len in [1448usize, 4096, 16 * 1024, 64 * 1024].into_iter().cycle() {
        if at == message.len() {
            break;
        }
        let end = (at + len).min(message.len());
        hasher.update_multithreaded(&message[at..end]);
        at = end;
    }
    assert_eq!(hasher.finalize(), convenience::hash(&message));
}

#[path = "../tools/require_nextest.rs"]
mod require_nextest;
