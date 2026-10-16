//! A stream on a process with one CPU (a container's quota, a pinned
//! process): the pool then has no worker, and on a CPU without SME2 no
//! thread at all to take tasks, so the caller's thread hashes every task
//! while it waits. Each kind of message completes with the reference
//! implementation's digests (once, the queue's tasks waited forever: fork
//! NOTES, "The queue on one CPU"). The test pins its process to one CPU
//! before the pool starts, so it is a binary of its own, with one test
//! (Linux, where the pool counts the process's affinity).
#![cfg(target_os = "linux")]

use blake3_servil::{Hash, Mode};

unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const core::ffi::c_ulong) -> i32;
}

/// Input byte i is i % 251, as the official vectors define their inputs.
fn input(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn reference(data: &[u8]) -> Hash {
    let mut hasher = reference_impl::Hasher::new();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    Hash::from(out)
}

#[test]
fn every_stream_completes_on_one_cpu() {
    // Pinned first, so that the pool starts with no worker.
    // A cpu_set_t is an array of unsigned longs, CPU n at bit n % 64 of
    // long n / 64 (a byte array would set CPU 56 on big-endian targets).
    let mask: [core::ffi::c_ulong; 1] = [1];
    // Sound: one long with CPU 0's bit, for this process.
    assert_eq!(unsafe { sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr()) }, 0, "pin the process to CPU 0");
    assert_eq!(std::thread::available_parallelism().unwrap().get(), 1);
    blake3_servil::initialize_multithreaded();
    // Messages short (gathered into tasks) and long (pieces), borrowed,
    // written, and read; and batches small and large.
    let lens = [0, 64, 1000, 16 << 10, 64 << 10, (1 << 20) + 1, 5 << 20];
    let messages: Vec<Vec<u8>> = lens.iter().map(|&len| input(len)).collect();
    let batch = input(64 * 4096);
    let (mut got, mut want) = (Vec::new(), Vec::new());
    blake3_servil::stream(Mode::Hash, |s| {
        for message in &messages {
            s.submit(message, |h| got.extend_from_slice(h));
            s.submit_with(message.len(), |at, room| room.copy_from_slice(&message[at..][..room.len()]), |h| got.extend_from_slice(h));
            s.submit_reader(&message[..], None, |h| got.extend_from_slice(h)).unwrap();
            want.extend([reference(message); 3]);
        }
        for count in [16, 4096] {
            s.submit_batch(&batch[..64 * count], 64, |h| got.extend_from_slice(h));
            want.extend(batch[..64 * count].chunks(64).map(reference));
        }
        s.finish(|h| got.extend_from_slice(h));
    });
    assert!(got == want, "every digest, in order, on one CPU");
}
