//! A stream whose caller loops on `collect()` for one place while every
//! worker is parked: the one waiting task must get a worker, or the loop
//! spins forever (it did on GitHub's four-CPU runner: the stream wakes a
//! worker on publish only for two waiting tasks, and a caller in
//! `collect` hashes nothing itself). A pool of three workers, parked
//! first; a binary of its own, since the pool is the process's; a
//! watchdog fails the test where it would hang. On a CPU with SME2 the
//! SME2 thread, woken on every publish, takes the task, so the condition
//! shows on CPUs without it (CI's x86-64 runners).

use blake3_servil::{Hash, Mode};

#[test]
fn a_caller_looping_on_collect_wakes_a_parked_worker() {
    blake3_servil::initialize_multithreaded(Some(std::num::NonZeroUsize::new(4).unwrap()));
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let message = vec![7u8; 3000];
        for round in 0..20 {
            // Every worker parked before the stream opens.
            std::thread::sleep(std::time::Duration::from_millis(20));
            let mut place = Hash::from([0; 32]);
            let got = std::cell::Cell::new(false);
            blake3_servil::stream(
                Mode::Hash,
                |_: &mut [Hash]| got.set(true),
                |s| {
                    s.submit(&message, &mut place);
                    while !got.get() {
                        s.collect();
                    }
                    s.finish();
                },
            );
            assert_eq!(place, blake3_servil::hash(&message), "round {round}");
        }
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the stream completes: a caller looping on collect waited on a task no worker was awake to take");
}
