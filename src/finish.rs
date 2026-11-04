//! A job's finish line: how many of a source's pieces are published and
//! how many finished, and the handshake between the caller waiting for
//! its last piece and the worker that finishes it. The one place a wake
//! can be lost, so the one place Shuttle checks every schedule it can:
//! `tests/shuttle_finish.rs` compiles this same file beside a `finish_sync`
//! of Shuttle's atomics, mutex and threads, where the library's `finish_sync`
//! is the standard library's. Nothing here names the rest of the crate,
//! so that the two builds see one source.
//!
//! Every atomic here is SeqCst. Shuttle (like a reader) assumes one total
//! order of the operations; a weaker ordering would open a class of bug
//! the model cannot see, and Sam Shackleton found one (October 24): a
//! Relaxed count load let `wait` return on its polling path before the
//! finished pieces' outputs were visible to the caller. One ordering, the
//! one the model checks, and the test below holds the file to it.
//!
//! The handshake lives in one word: the count of pieces finished, and in
//! its top bit whether the caller sleeps. The waiter sets the bit with
//! one read-modify-write and reads the count it came with; the finisher
//! adds its pieces with one and reads the bit it came with. Every
//! read-modify-write on one word sees the one before it, so exactly one
//! of the two sees the other's: the finisher sees the sleeper and
//! unparks, or the sleeper sees the count complete and returns. A word
//! apart for the flag would need the two sides' stores and loads in one
//! total order (SeqCst), which a reader must reason about; one word
//! needs only its own read-modify-writes.
//!
// The primitives come from the including module: the library's
// `finish_sync` is the standard library's, the Shuttle test's is Shuttle's.
use super::finish_sync::{AtomicUsize, Mutex, Ordering, thread};

/// Any value on a cache line of its own.
#[repr(align(128))]
pub(crate) struct Line<T>(pub(crate) T);

impl<T> std::ops::Deref for Line<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// The top bit of `state`: the caller is parked, waiting.
const SLEEPING: usize = 1 << (usize::BITS - 1);

pub(crate) struct Finish {
    /// The number past the last piece or task published.
    pub(crate) published: Line<AtomicUsize>,
    /// The pieces finished, all the source's jobs', and SLEEPING.
    state: Line<AtomicUsize>,
    /// The waiting thread: behind a lock, since the worker finishing the
    /// last piece may read it as the source passes to another thread (an
    /// unpark that reaches the new owner is a harmless spurious wake).
    owner: Mutex<Option<thread::Thread>>,
}

impl Finish {
    pub(crate) fn new() -> Finish {
        Finish { published: Line(AtomicUsize::new(0)), state: Line(AtomicUsize::new(0)), owner: Mutex::new(None) }
    }

    /// The pieces finished so far.
    pub(crate) fn finished_count(&self) -> usize {
        self.state.load(Ordering::SeqCst) & !SLEEPING
    }

    /// The thread that waits on this source from now on.
    pub(crate) fn own(&self) {
        *self.owner.lock().unwrap() = Some(thread::current());
    }

    /// Count `n` pieces finished (their outputs written before); the last
    /// one wakes a caller parked for it.
    #[inline]
    pub(crate) fn finished(&self, n: usize) {
        let before = self.state.fetch_add(n, Ordering::SeqCst);
        if before & SLEEPING != 0 && (before & !SLEEPING) + n == self.published.load(Ordering::SeqCst) {
            if let Some(owner) = &*self.owner.lock().unwrap() {
                #[cfg(all(test, not(shuttle)))]
                tests::UNPARKS.with(|u| u.set(u.get() + 1));
                owner.unpark();
            }
        }
    }

    /// Wait until the count reaches `target`: `poll` a while (it returns
    /// false when the while is up), then parked until the worker finishing
    /// the last piece unparks this thread.
    #[inline]
    pub(crate) fn wait(&self, target: usize, mut poll: impl FnMut() -> bool) {
        loop {
            if self.finished_count() >= target {
                return;
            }
            if !poll() {
                break;
            }
        }
        // The bit and the count in one read: a finisher after this sees
        // the bit; one before it left its count here.
        let mut count = self.state.fetch_or(SLEEPING, Ordering::SeqCst) & !SLEEPING;
        while count < target {
            thread::park();
            count = self.finished_count();
        }
        self.state.fetch_and(!SLEEPING, Ordering::SeqCst);
    }
}

// The standard library's test alone: under Shuttle (tests/shuttle_finish.rs)
// the file compiles beside Shuttle's primitives, which have no sleep.
#[cfg(all(test, not(shuttle)))]
mod tests {
    use super::*;
    use std::sync::Arc;

    std::thread_local! {
        /// The owner unparks finished() made on this thread.
        pub(super) static UNPARKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    }

    /// finished() unparks the owner only when it sleeps and the count is
    /// complete. Any other unpark is a system call a piece for a wake the
    /// owner sleeps through, which no result shows, so only the count tells
    /// (BLAKE3 #219).
    #[test]
    fn finished_unparks_only_a_sleeper_on_the_last_piece() {
        let finish = Finish::new();
        finish.own();
        finish.published.store(3, Ordering::SeqCst);
        UNPARKS.with(|u| u.set(0));
        // No sleeper: the last piece wakes nobody.
        for _ in 0..3 {
            finish.finished(1);
        }
        assert_eq!(UNPARKS.with(|u| u.get()), 0, "no sleeper, no unpark");
        // A sleeper (its bit, as wait() sets it): only the last piece wakes it.
        finish.published.store(6, Ordering::SeqCst);
        finish.state.fetch_or(SLEEPING, Ordering::SeqCst);
        finish.finished(1);
        finish.finished(1);
        assert_eq!(UNPARKS.with(|u| u.get()), 0, "a sleeper is not woken before its count is complete");
        finish.finished(1);
        assert_eq!(UNPARKS.with(|u| u.get()), 1, "the last piece wakes the sleeper once");
        // The token this thread was handed, taken so it outlives no test.
        thread::park();
    }

    /// The handshake files use one ordering, SeqCst, the one the model
    /// checker assumes; this fires on any other, with the reason above.
    #[test]
    fn every_atomic_in_the_handshake_files_is_seqcst() {
        // Read at run time, not embedded (include_str!), so the test binary
        // holds none of the files' text: an embedded copy changed with every
        // mutant of them, and code_hash.py found none same code (#219).
        for name in ["finish.rs", "park.rs"] {
            let text = std::fs::read_to_string(std::format!("{}/src/{name}", env!("CARGO_MANIFEST_DIR"))).expect("the file, beside the crate's manifest");
            for (n, line) in text.lines().enumerate() {
                if line.contains("Ordering::") && !line.contains("Ordering::SeqCst") && !line.trim_start().starts_with("//") {
                    panic!("{name}:{}: {} (every atomic here is SeqCst: see finish.rs's head)", n + 1, line.trim());
                }
            }
        }
    }

    /// A wake with no piece behind it, as the operating system may give
    /// (std::thread::park allows spurious wakeups), sends the waiter back
    /// to sleep: it returns only once the count is complete. Loom's park
    /// has no spurious wakes, so this is the standard library's test.
    #[test]
    fn a_spurious_wake_is_slept_through() {
        let finish = Arc::new(Finish::new());
        finish.published.store(2, Ordering::SeqCst);
        let returned = Arc::new(AtomicUsize::new(0));
        let waiter = {
            let (finish, returned) = (finish.clone(), returned.clone());
            thread::spawn(move || {
                finish.own();
                finish.wait(2, || false);
                returned.store(1, Ordering::SeqCst);
            })
        };
        let handle = waiter.thread().clone();
        // Spurious wakes while one piece is still to come: the waiter
        // must not return.
        finish.finished(1);
        for _ in 0..20 {
            handle.unpark();
            thread::sleep(std::time::Duration::from_millis(1));
            assert_eq!(returned.load(Ordering::SeqCst), 0, "the waiter returned before its count was complete");
        }
        finish.finished(1);
        waiter.join().unwrap();
        assert_eq!(returned.load(Ordering::SeqCst), 1);
        assert_eq!(finish.finished_count(), 2);
    }
}
