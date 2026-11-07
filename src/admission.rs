//! Reserved resources for the stream's producer, served in the order
//! asked. A returned resource wakes the oldest waiter, who takes it under
//! the pool's lock on its next poll: a resource is never parked in a
//! request nobody polls, so a pending request may be dropped as any future
//! is (a timeout, a `select!`), and the next waiter is served. Dropping a
//! request removes its ticket, the registration every future that waits
//! on a waker undoes; it holds no resource, so no cleanup runs in Drop.
//! A resource taken is released by an explicit call; one dropped unreleased
//! ends the process (crash-only: a lost resource would stall every waiter).
//!
//! The pool allocates its resource list and a waiter queue of the same
//! capacity in `new`; waiting beyond that capacity grows the queue.
//! Mutex metadata is initialized in `new` too (Darwin's lazy pthread box).
//!
//! Finish counts completed work; this pool transfers ownership of individual
//! resources. The two predicates differ. Both compile their actual source
//! against Shuttle's primitives for the handshake tests.

use super::finish_sync::Mutex;
use core::future::Future;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::collections::VecDeque;
use std::sync::Arc;

pub(crate) struct Pool<T> {
    state: Mutex<State<T>>,
}

struct State<T> {
    free: Vec<T>,
    /// Waiters in the order they asked: each with its ticket and the waker
    /// it last polled with.
    waiters: VecDeque<(u64, Waker)>,
    next_ticket: u64,
}

/// A request: its ticket once queued. `Unpin`: the queue holds no pointer
/// into it.
pub(crate) struct Acquire<P: Deref<Target = Pool<T>> + Unpin, T> {
    pool: Option<P>,
    ticket: Option<u64>,
}

pub(crate) struct Resource<P: Deref<Target = Pool<T>> + Unpin, T> {
    pool: P,
    value: Option<T>,
}

pub(crate) type Lease<'a, T> = Resource<&'a Pool<T>, T>;
pub(crate) type OwnedLease<T> = Resource<Arc<Pool<T>>, T>;

/// Initialize platform mutex metadata before the operation's allocation window.
pub(crate) fn initialized_mutex<T>(value: T) -> Mutex<T> {
    let mutex = Mutex::new(value);
    // Darwin initializes its pthread mutex on first lock (64 bytes).
    drop(mutex.lock().unwrap());
    mutex
}

impl<T> Pool<T> {
    /// The fixed, nonempty set of resources this pool circulates.
    pub(crate) fn new(resources: Vec<T>) -> Self {
        assert!(!resources.is_empty(), "a resource pool has capacity");
        let waiters = VecDeque::with_capacity(resources.len());
        Self { state: initialized_mutex(State { free: resources, waiters, next_ticket: 0 }) }
    }

    pub(crate) fn acquire(&self) -> Acquire<&Self, T> {
        Acquire { pool: Some(self), ticket: None }
    }

    /// Ownership of this pool follows the buffer through background work.
    /// Cloning an Arc reserves ownership, without allocating another pool.
    pub(crate) fn acquire_owned(self: Arc<Self>) -> Acquire<Arc<Self>, T> {
        Acquire { pool: Some(self), ticket: None }
    }

    fn give(&self, value: T) {
        let wake = {
            let mut state = self.state.lock().unwrap();
            state.free.push(value);
            state.waiters.front().map(|(_, waker)| waker.clone())
        };
        // Waker code runs outside the mutex.
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Future for Acquire<P, T> {
    type Output = Resource<P, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        let pool = this.pool.as_ref().expect("a completed resource request is consumed");
        // The waker's code (clone, drop) runs outside the pool's lock.
        let mut waker = Some(cx.waker().clone());
        let mut state = pool.state.lock().unwrap();
        let at_front = match this.ticket {
            None => state.waiters.is_empty(),
            Some(ticket) => state.waiters.front().is_some_and(|(t, _)| *t == ticket),
        };
        if at_front {
            if let Some(value) = state.free.pop() {
                if this.ticket.take().is_some() {
                    state.waiters.pop_front();
                }
                // The next waiter may take the next free resource now.
                let wake = state.free.last().and(state.waiters.front().map(|(_, w)| w.clone()));
                drop(state);
                drop(waker);
                if let Some(wake) = wake {
                    wake.wake();
                }
                return Poll::Ready(Resource { pool: this.pool.take().unwrap(), value: Some(value) });
            }
        }
        let old = match this.ticket {
            None => {
                let ticket = state.next_ticket;
                state.next_ticket += 1;
                state.waiters.push_back((ticket, waker.take().unwrap()));
                this.ticket = Some(ticket);
                None
            }
            Some(ticket) => {
                let (_, registered) = state.waiters.iter_mut().find(|(t, _)| *t == ticket).expect("a queued request keeps its ticket");
                if registered.will_wake(cx.waker()) { None } else { Some(core::mem::replace(registered, waker.take().unwrap())) }
            }
        };
        drop(state);
        drop(old);
        drop(waker);
        Poll::Pending
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Drop for Acquire<P, T> {
    /// A pending request leaves the queue (it holds no resource). If it was
    /// at the front with a resource free, the next waiter is woken.
    fn drop(&mut self) {
        let (Some(pool), Some(ticket)) = (&self.pool, self.ticket) else { return };
        let (removed, wake) = {
            let mut state = pool.state.lock().unwrap();
            let at = state.waiters.iter().position(|(t, _)| *t == ticket).expect("a queued request keeps its ticket");
            let removed = state.waiters.remove(at).map(|(_, w)| w);
            let wake = (at == 0 && !state.free.is_empty()).then(|| state.waiters.front().map(|(_, w)| w.clone())).flatten();
            (removed, wake)
        };
        drop(removed);
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}

/// A broken ownership contract ends the process at once, as the pool's
/// guards do: a lost resource would stall every waiter.
fn misuse(what: &str) -> ! {
    eprintln!("blake3-servil: {what}; the process ends (crash-only)");
    std::process::abort()
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Resource<P, T> {
    /// Give the resource back, waking the oldest waiter.
    pub(crate) fn release(mut self) {
        let value = self.value.take().unwrap();
        self.pool.give(value);
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Drop for Resource<P, T> {
    fn drop(&mut self) {
        if self.value.is_some() {
            misuse("a reserved resource was dropped without release()");
        }
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Deref for Resource<P, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().unwrap()
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> DerefMut for Resource<P, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().unwrap()
    }
}
