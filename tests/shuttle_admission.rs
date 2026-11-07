//! The stream reservation source under Shuttle's mutex and executor.
//! A grant owns the resource before wakeup. Cancellation removes a waiter
//! or returns its grant. The file under test is the ordinary implementation.
//!
//! RUSTFLAGS="--cfg shuttle" cargo nextest run --release --test shuttle_admission
#![cfg(all(shuttle, feature = "std"))]

mod finish_sync {
    pub use shuttle::sync::Mutex;
}
#[path = "../src/admission.rs"]
mod admission;
#[path = "../tools/require_nextest.rs"]
mod require_nextest;

use admission::Pool;
use core::{future::Future, task::Context};
use shuttle::{future::block_on, thread};

#[test]
fn worker_release_races_with_waiter_registration() {
    shuttle::check_pct(
        || {
            let pool = Pool::new(vec![0]);
            let mut held: admission::Lease<'_, usize> = block_on(pool.acquire());
            thread::scope(|s| {
                s.spawn(move || {
                    *held = 42;
                    held.release();
                });
                let received = block_on(pool.acquire());
                assert_eq!(*received, 42, "the grant publishes the worker's writes");
                received.release();
            });
        },
        3000,
        3,
    );
}

#[test]
fn a_new_request_cannot_take_an_older_requests_grant() {
    shuttle::check_pct(
        || {
            let pool = Pool::new(vec![7]);
            let held = block_on(pool.acquire());
            let mut cx = Context::from_waker(std::task::Waker::noop());
            let mut first = std::pin::pin!(pool.acquire());
            assert!(first.as_mut().poll(&mut cx).is_pending());
            thread::scope(|s| {
                s.spawn(move || held.release());
                let mut newcomer = std::pin::pin!(pool.acquire());
                assert!(newcomer.as_mut().poll(&mut cx).is_pending());
                let granted = block_on(first.as_mut());
                assert_eq!(*granted, 7);
                granted.release();
                let granted = block_on(newcomer.as_mut());
                assert_eq!(*granted, 7);
                granted.release();
            });
        },
        3000,
        3,
    );
}

#[test]
fn cancellation_races_with_a_grant_and_preserves_the_next_request() {
    shuttle::check_pct(
        || {
            let pool = Pool::new(vec![11]);
            let held = block_on(pool.acquire());
            let mut cx = Context::from_waker(std::task::Waker::noop());
            let mut cancelled = Box::pin(pool.acquire());
            let mut next = std::pin::pin!(pool.acquire());
            assert!(cancelled.as_mut().poll(&mut cx).is_pending());
            assert!(next.as_mut().poll(&mut cx).is_pending());
            thread::scope(|s| {
                s.spawn(move || held.release());
                cancelled.as_mut().cancel();
                let granted = block_on(next.as_mut());
                assert_eq!(*granted, 11);
                granted.release();
            });
            let granted = block_on(pool.acquire());
            assert_eq!(*granted, 11);
            granted.release();
        },
        3000,
        3,
    );
}

#[test]
fn owned_resources_survive_the_original_pool_handle() {
    shuttle::check_pct(
        || {
            let pool = std::sync::Arc::new(Pool::new(vec![31]));
            let held: admission::OwnedLease<usize> = block_on(pool.clone().acquire_owned());
            let mut request = std::pin::pin!(pool.clone().acquire_owned());
            let mut cx = Context::from_waker(std::task::Waker::noop());
            assert!(request.as_mut().poll(&mut cx).is_pending());
            drop(pool);
            thread::scope(|s| {
                s.spawn(move || held.release());
                let granted = block_on(request.as_mut());
                assert_eq!(*granted, 31);
                granted.release();
            });
        },
        3000,
        3,
    );
}
