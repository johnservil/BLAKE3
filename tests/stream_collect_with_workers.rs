//! A stream whose caller loops on `collect()` for a free place, on a pool
//! with workers that have gone to sleep: the stream publishes a lone task
//! for them without waking one (a worker is woken for the second task in
//! waiting, the caller expected to take the first), and `collect()` takes
//! none, so the loop must wake one or it waits forever (the C API's stream
//! test hung so on an i7-12700K, BLAKE3 candidate/minimalism-0.8). A binary
//! of its own, since the pool is the process's; a watchdog fails the test
//! where it would hang.

use blake3_servil::{Hash, Mode};

fn reference(data: &[u8]) -> Hash {
    let mut hasher = reference_impl::Hasher::new();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    Hash::from(out)
}

#[test]
fn a_caller_looping_on_collect_makes_progress_with_sleeping_workers() {
    blake3_servil::initialize_multithreaded(None);
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let messages: Vec<Vec<u8>> = (0..200usize).map(|i| (0..1 + i * 37 % 5000).map(|j| (j % 251) as u8).collect()).collect();
        let mut places = vec![Hash::from([0; 32]); 1];
        let free = std::cell::RefCell::new(places.iter_mut().collect::<Vec<_>>());
        let got = std::cell::RefCell::new(Vec::new());
        blake3_servil::stream(
            Mode::Hash,
            |place: &mut [Hash]| {
                got.borrow_mut().push(place[0]);
                free.borrow_mut().push(&mut place[0]);
            },
            |s| {
                for message in &messages {
                    // Long enough for every worker to go back to sleep.
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    let place = loop {
                        if let Some(place) = free.borrow_mut().pop() {
                            break place;
                        }
                        s.collect();
                    };
                    s.submit(message, place);
                    // One place: wait for it before the next message.
                    while free.borrow().is_empty() {
                        s.collect();
                    }
                }
                s.finish();
            },
        );
        let got = got.into_inner();
        assert_eq!(got.len(), messages.len());
        for (i, message) in messages.iter().enumerate() {
            assert_eq!(got[i], reference(message), "message {i}");
        }
        done.send(()).unwrap();
    });
    finished.recv_timeout(std::time::Duration::from_secs(20)).expect("a caller looping on collect() made no progress in 20 s");
}
