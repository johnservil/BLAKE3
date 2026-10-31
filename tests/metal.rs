#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
include!("../tools/require_nextest.rs");

use blake3_servil::{Hash, Mode, convenience, metal};

/// Messages whose digests the stream must give: lengths around every
/// boundary (a block, a chunk, a subtree, a piece of the CPU stream, a
/// power of two of the GPU's tree), growing, shrinking and repeating so
/// slots are allocated, reused and regrown.
fn messages() -> Vec<Vec<u8>> {
    let mut sizes = vec![
        0,
        1,
        3,
        63,
        64,
        65,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        3072,
        3073,
        4096,
        4097,
        16383,
        16384,
        16385,
        65536,
        65537,
        262143,
        262144,
        262145,
        1048576,
        1048577,
        4 * 1048576 + 17,
        8 << 20,
        16 * 1048576,
        (8 << 20) + 1,
        (8 << 20) - 1,
        0,
        65,
        16385,
    ];
    let mut random = 17u64;
    for _ in 0..24 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        sizes.push((random % 200000) as usize);
    }
    sizes.iter().enumerate().map(|(k, &n)| (0..n).map(|i| ((i + k) % 251) as u8).collect()).collect()
}

/// Every message through the stream, in `mode`, once with `collect`
/// between submissions: each digest is the CPU's, every digest arrives
/// once, and each device's arrive in submission order.
fn check(gpu: &mut metal::Hasher, mode: Mode, messages: &[Vec<u8>]) {
    let expected: Vec<Hash> = messages.iter().map(|m| convenience::hash_with(mode, m)).collect();
    let mut digests = vec![Hash::from_bytes([0; 32]); messages.len()];
    for use_collect in [false, true] {
        let mut handed_back = Vec::new();
        gpu.stream(
            mode,
            |place: &mut [Hash]| handed_back.push(place[0]),
            |s| {
                for (message, digest) in messages.iter().zip(&mut digests) {
                    s.submit_with(
                        message.len(),
                        |at, room| {
                            room.copy_from_slice(&message[at..][..room.len()]);
                            room.len()
                        },
                        digest,
                    );
                    if use_collect {
                        s.collect();
                    }
                }
                s.finish();
            },
        );
        for (k, (digest, expected)) in digests.iter().zip(&expected).enumerate() {
            assert_eq!(digest, expected, "message {k} of {} bytes, gpu from {}", messages[k].len(), gpu.gpu_from());
        }
        let on = |on_gpu: bool| -> Vec<Hash> {
            let from = gpu.gpu_from();
            digests.iter().copied().zip(messages).filter(|(_, m)| (m.len() >= from) == on_gpu).map(|(d, _)| d).collect()
        };
        for on_gpu in [true, false] {
            let order: Vec<Hash> = handed_back.iter().copied().filter(|h| on(on_gpu).contains(h)).collect();
            assert_eq!(order, on(on_gpu), "handed back in submission order on each device, gpu {on_gpu}");
        }
        assert_eq!(handed_back.len(), messages.len(), "every digest handed back once");
    }
}

#[test]
fn stream_matches_the_cpu_in_every_mode() {
    blake3_servil::initialize_multithreaded(None);
    let messages = messages();
    let key = [7u8; 32];
    let context_key = blake3_servil::performance::context_key("blake3-servil metal test 2026-10-31");
    let modes = [Mode::Plain, Mode::Keyed(&key), Mode::DeriveKey("blake3-servil metal test 2026-10-31"), Mode::DeriveKeyWith(&context_key)];
    // The device's threshold, then the GPU for every message (the odd and
    // short lengths exercise its kernels), then the CPU for every message.
    let mut hasher = metal::Hasher::new().expect("Metal GPU is required");
    assert!((1 << 20..=64 << 20).contains(&hasher.gpu_from()), "a threshold from the core count: {}", hasher.gpu_from());
    for mode in modes {
        check(&mut hasher, mode, &messages);
    }
    let mut all_gpu = metal::Hasher::with_gpu_from(0).expect("Metal GPU is required");
    let mut all_cpu = metal::Hasher::with_gpu_from(usize::MAX).expect("Metal GPU is required");
    for mode in modes {
        check(&mut all_gpu, mode, &messages);
        check(&mut all_cpu, mode, &messages);
    }
}

#[test]
fn submit_takes_blocks() {
    blake3_servil::initialize_multithreaded(None);
    let mut gpu = metal::Hasher::with_gpu_from(1 << 20).expect("Metal GPU is required");
    let mut batch = blake3_servil::performance::Batch::new(3 << 20, 1);
    let message: Vec<u8> = (0..3 << 20).map(|i| (i % 251) as u8).collect();
    batch.set(0, &message);
    let mut digests = [Hash::from_bytes([0; 32]); 2];
    let (gpu_digest, cpu_digest) = digests.split_at_mut(1);
    gpu.stream(
        Mode::Plain,
        |_: &mut [Hash]| {},
        |s| {
            s.submit(batch.blocks(), message.len(), &mut gpu_digest[0]);
            s.submit(batch.blocks(), 1000, &mut cpu_digest[0]);
            s.finish();
        },
    );
    assert_eq!(digests[0], convenience::hash(&message));
    assert_eq!(digests[1], convenience::hash(&message[..1000]));
}

#[test]
#[should_panic(expected = "a stream's body calls finish last")]
fn a_body_without_finish_stops() {
    blake3_servil::initialize_multithreaded(None);
    let mut gpu = metal::Hasher::with_gpu_from(0).expect("Metal GPU is required");
    let mut digest = Hash::from_bytes([0; 32]);
    gpu.stream(
        Mode::Plain,
        |_: &mut [Hash]| {},
        |s| {
            s.submit_with(
                100,
                |_, room| {
                    room.fill(1);
                    room.len()
                },
                &mut digest,
            );
        },
    );
}
