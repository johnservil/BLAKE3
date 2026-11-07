//! The reservation core used by the stream prototype. The production stream
//! adopts it after the ownership, cancellation and model tests pass.
//!
//! Memory checks: MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-env-forward=NEXTEST"
//! cargo +nightly miri nextest run --test admission
//! Add -Zmiri-tree-borrows to exercise that aliasing model too.
#![cfg(feature = "std")]

mod finish_sync {
    pub use std::sync::Mutex;
}
#[path = "../src/admission.rs"]
mod admission;
mod require_nextest {
    include!("../tools/require_nextest.rs");
}

use admission::{Lease, Pool};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct Signal(AtomicUsize);
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn ready<F: Future>(future: Pin<&mut F>, cx: &mut Context<'_>) -> F::Output {
    match future.poll(cx) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the capacity is reserved"),
    }
}

fn take<'a, T>(pool: &'a Pool<T>, cx: &mut Context<'_>) -> Lease<'a, T> {
    ready(std::pin::pin!(pool.acquire()), cx)
}

#[test]
fn granted_capacity_is_reserved_before_notification() {
    let pool = Pool::new(vec![17]);
    let signal = Arc::new(Signal::default());
    let waker = Waker::from(signal.clone());
    let mut cx = Context::from_waker(&waker);
    let held = take(&pool, &mut cx);
    let mut first = Box::pin(pool.acquire());
    assert!(first.as_mut().poll(&mut cx).is_pending());
    // Repeated polls retain one queue position and one notification.
    assert!(first.as_mut().poll(&mut cx).is_pending());
    held.release();
    assert_eq!(signal.0.load(Ordering::SeqCst), 1);
    let mut newcomer = Box::pin(pool.acquire());
    assert!(newcomer.as_mut().poll(&mut cx).is_pending(), "a new arrival cannot steal the grant");
    let granted = ready(first.as_mut(), &mut cx);
    assert_eq!(*granted, 17);
    granted.release();
    ready(newcomer.as_mut(), &mut cx).release();
}

#[test]
fn a_dropped_request_leaves_the_queue_at_every_position() {
    for cancel in 0..3 {
        let pool = Pool::new(vec![9]);
        let waker = Waker::from(Arc::new(Signal::default()));
        let mut cx = Context::from_waker(&waker);
        let held = take(&pool, &mut cx);
        let mut waiting: Vec<_> = (0..3).map(|_| Box::pin(pool.acquire())).collect();
        for request in &mut waiting {
            assert!(request.as_mut().poll(&mut cx).is_pending());
        }
        drop(waiting.remove(cancel));
        let mut appended = Box::pin(pool.acquire());
        assert!(appended.as_mut().poll(&mut cx).is_pending());
        waiting.push(appended);
        held.release();
        for mut request in waiting {
            let granted = ready(request.as_mut(), &mut cx);
            assert_eq!(*granted, 9);
            granted.release();
        }
        take(&pool, &mut cx).release();
    }
}

#[test]
fn dropping_the_woken_waiter_wakes_the_next() {
    let pool = Pool::new(vec![11]);
    let first_signal = Arc::new(Signal::default());
    let first_waker = Waker::from(first_signal.clone());
    let second_signal = Arc::new(Signal::default());
    let second_waker = Waker::from(second_signal.clone());
    let mut first_cx = Context::from_waker(&first_waker);
    let mut second_cx = Context::from_waker(&second_waker);
    let held = take(&pool, &mut first_cx);
    let mut first = Box::pin(pool.acquire());
    let mut second = Box::pin(pool.acquire());
    assert!(first.as_mut().poll(&mut first_cx).is_pending());
    assert!(second.as_mut().poll(&mut second_cx).is_pending());
    held.release();
    assert_eq!(first_signal.0.load(Ordering::SeqCst), 1);
    assert_eq!(second_signal.0.load(Ordering::SeqCst), 0);
    drop(first);
    assert_eq!(second_signal.0.load(Ordering::SeqCst), 1);
    ready(second.as_mut(), &mut second_cx).release();
}

#[test]
fn a_moved_pending_future_replaces_its_notification() {
    let pool = Pool::new(vec![1]);
    let old = Arc::new(Signal::default());
    let new = Arc::new(Signal::default());
    let old_waker = Waker::from(old.clone());
    let new_waker = Waker::from(new.clone());
    let mut old_cx = Context::from_waker(&old_waker);
    let mut new_cx = Context::from_waker(&new_waker);
    let held = take(&pool, &mut old_cx);
    let mut request = Box::pin(pool.acquire());
    assert!(request.as_mut().poll(&mut old_cx).is_pending());
    // Move the pinned owner, keeping its allocation in place.
    let mut moved = request;
    assert!(moved.as_mut().poll(&mut new_cx).is_pending());
    held.release();
    assert_eq!(old.0.load(Ordering::SeqCst), 0);
    assert_eq!(new.0.load(Ordering::SeqCst), 1);
    ready(moved.as_mut(), &mut new_cx).release();
}

#[test]
fn worker_writes_are_visible_when_the_buffer_returns() {
    let pool = Pool::new(vec![Box::new([0u8; 64])]);
    let waker = Waker::from(Arc::new(Signal::default()));
    let mut cx = Context::from_waker(&waker);
    let mut held = take(&pool, &mut cx);
    let address = held.as_ptr();
    let mut waiting = Box::pin(pool.acquire());
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    std::thread::scope(|s| {
        s.spawn(move || {
            held.fill(42);
            held.release();
        })
        .join()
        .unwrap();
    });
    let returned = ready(waiting.as_mut(), &mut cx);
    assert_eq!(returned.as_ptr(), address, "handoff preserves the buffer's address");
    assert_eq!(returned.as_slice(), &[42; 64]);
    returned.release();
}

#[test]
fn unused_requests_leave_capacity_available() {
    let pool = Pool::new(vec![23]);
    // Never polled: nothing is queued or granted, so dropping is harmless.
    drop(pool.acquire());
    let waker = Waker::from(Arc::new(Signal::default()));
    let mut cx = Context::from_waker(&waker);
    let granted = take(&pool, &mut cx);
    assert_eq!(*granted, 23);
    granted.release();
}

#[test]
fn capacity_returned_while_cloning_the_notification_is_taken() {
    use std::task::{RawWaker, RawWakerVTable};
    struct ReturnOnClone<'a> {
        held: std::sync::Mutex<Option<Lease<'a, usize>>>,
    }
    unsafe fn clone(data: *const ()) -> RawWaker {
        // Sound: this test keeps ReturnOnClone alive through every waker.
        let state = unsafe { &*data.cast::<ReturnOnClone<'_>>() };
        if let Some(held) = state.held.lock().unwrap().take() {
            held.release();
        }
        RawWaker::new(data, &VTABLE)
    }
    unsafe fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let pool = Pool::new(vec![37]);
    let mut noop_cx = Context::from_waker(Waker::noop());
    let held = take(&pool, &mut noop_cx);
    let state = ReturnOnClone { held: std::sync::Mutex::new(Some(held)) };
    // Sound: the state stays in place and every clone is dropped before it.
    let waker = unsafe { Waker::from_raw(RawWaker::new((&state as *const ReturnOnClone<'_>).cast(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut request = std::pin::pin!(pool.acquire());
    let granted = ready(request.as_mut(), &mut cx);
    assert_eq!(*granted, 37, "registration rechecks capacity after cloning the waker");
    granted.release();
}

#[test]
fn an_owned_grant_keeps_the_pool_alive_after_the_original_handle_drops() {
    let pool = Arc::new(Pool::new(vec![19]));
    let weak = Arc::downgrade(&pool);
    let waker = Waker::from(Arc::new(Signal::default()));
    let mut cx = Context::from_waker(&waker);
    let mut request = Box::pin(pool.clone().acquire_owned());
    let granted: admission::OwnedLease<usize> = ready(request.as_mut(), &mut cx);
    drop(pool);
    assert!(weak.upgrade().is_some());
    assert_eq!(*granted, 19);
    drop(request); // Completed: it holds nothing.
    granted.release();
    assert!(weak.upgrade().is_none(), "the final lease releases pool ownership");
}

#[test]
fn owned_and_borrowed_requests_share_one_fifo() {
    let pool = Arc::new(Pool::new(vec![29]));
    let mut cx = Context::from_waker(Waker::noop());
    let held = take(&pool, &mut cx);
    let mut borrowed = Box::pin(pool.acquire());
    let mut owned = Box::pin(pool.clone().acquire_owned());
    assert!(borrowed.as_mut().poll(&mut cx).is_pending());
    assert!(owned.as_mut().poll(&mut cx).is_pending());
    held.release();
    assert!(owned.as_mut().poll(&mut cx).is_pending(), "ownership form preserves queue order");
    ready(borrowed.as_mut(), &mut cx).release();
    let granted = ready(owned.as_mut(), &mut cx);
    assert_eq!(*granted, 29);
    granted.release();
}

/// Crash-only: a reserved resource dropped unreleased ends the process.
/// The case runs in a child process that must die by SIGABRT. A pending
/// request, which holds nothing, is dropped freely (the tests above).
#[test]
fn an_unreleased_resource_aborts_the_process() {
    if std::env::var_os("ADMISSION_MISUSE").is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "an_unreleased_resource_aborts_the_process", "--nocapture"])
            .env("ADMISSION_MISUSE", "resource")
            .env("NEXTEST", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("crash-only"), "the mistake is named: {stderr}");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(out.status.signal(), Some(libc_sigabrt()), "the process aborts");
        }
        return;
    }
    let pool = Pool::new(vec![5]);
    let mut cx = Context::from_waker(Waker::noop());
    let held = take(&pool, &mut cx);
    drop(held);
    // An undetected mistake ends here, quietly: the parent then sees no
    // abort and fails.
    std::process::exit(0);
}

fn libc_sigabrt() -> i32 {
    6
}
