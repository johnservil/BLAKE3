//! A stream whose caller loops on `collect()` for a free place, on a pool
//! with no workers (`initialize_multithreaded(Some(1))`, as one CPU gives):
//! the stream hashes each task on the caller as it hands it over, so the
//! loop makes progress (it spun forever once, waiting for takers that did
//! not exist: BLAKE3 #86). A binary of its own, since the pool is the
//! process's; a watchdog fails the test where it would hang. (Devon Jonte,
//! PR #88's test.)

use blake3_servil::{Hash, Mode};

fn reference(data: &[u8]) -> Hash {
    let mut hasher = reference_impl::Hasher::new();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    Hash::from(out)
}

#[test]
fn a_caller_looping_on_collect_makes_progress_without_workers() {
    blake3_servil::initialize_multithreaded(Some(std::num::NonZeroUsize::new(2).unwrap()));
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let messages: Vec<Vec<u8>> = (0..1000usize).map(|i| (0..1 + i * 7 % 3000).map(|j| (j % 251) as u8).collect()).collect();
        // Every worker parked before the stream opens.
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut places = vec![Hash::from([0; 32]); 4];
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
                    let place = loop {
                        if let Some(place) = free.borrow_mut().pop() {
                            break place;
                        }
                        s.collect();
                    };
                    s.submit(message, place);
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
    finished.recv_timeout(std::time::Duration::from_secs(60)).expect("the stream completes: collect() made no progress without workers");
}
