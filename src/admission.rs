//! Reserved resources for the stream's producer: a returned resource goes
//! straight to the oldest waiter, before its notification. A caller that
//! resumes owns its resource, independently of what other producers do.
//!
//! The future holds its own queue node. Pinning keeps that node in place.
//! Returns and cancellations are explicit operations (crash-only: no Drop
//! does lifecycle work). A pending request forgotten in the queue is a
//! caller's mistake that the pool detects when it next grants.
//! The pool initializes its fixed metadata in new. Waiting uses the caller's
//! future node. It makes no per-request allocation.
//!
//! Finish counts completed work; this pool transfers ownership of individual
//! resources. The two predicates differ. Both compile their actual source
//! against Shuttle's primitives for the handshake tests.

use super::finish_sync::Mutex;
use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomPinned;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::ptr;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;

pub(crate) struct Pool<T> {
    state: Mutex<State<T>>,
}

struct State<T> {
    free: Vec<T>,
    first: *mut Node<T>,
    last: *mut Node<T>,
}

struct Node<T> {
    previous: *mut Node<T>,
    next: *mut Node<T>,
    queued: bool,
    granted: Option<T>,
    waker: Option<Waker>,
}

// Sound: a queued node is pinned, alive through its unlink, and reached
// only under state. A resource crosses threads only when T permits it.
unsafe impl<T: Send> Send for State<T> {}

pub(crate) struct Acquire<P: Deref<Target = Pool<T>> + Unpin, T> {
    pool: Option<P>,
    node: UnsafeCell<Node<T>>,
    _pin: PhantomPinned,
}

// Sound: the node is private before its first poll. Once pinned, every
// access to it uses the pool's mutex, including grant and cancellation.
unsafe impl<P: Deref<Target = Pool<T>> + Unpin + Send, T: Send> Send for Acquire<P, T> {}

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
        Self { state: initialized_mutex(State { free: resources, first: ptr::null_mut(), last: ptr::null_mut() }) }
    }

    pub(crate) fn acquire(&self) -> Acquire<&Self, T> {
        Acquire::new(self)
    }

    /// Ownership of this pool follows the buffer through background work.
    /// Cloning an Arc reserves ownership, without allocating another pool.
    pub(crate) fn acquire_owned(self: Arc<Self>) -> Acquire<Arc<Self>, T> {
        Acquire::new(self)
    }

    fn give(&self, value: T) {
        let wake = {
            let mut state = self.state.lock().unwrap();
            if state.first.is_null() {
                state.free.push(value);
                return;
            }
            let first = state.first;
            // Sound: state owns access to this pinned, queued node. Unlink
            // and store its grant before exposing the notification.
            unsafe {
                state.unlink(first);
                (*first).granted = Some(value);
                (*first).waker.take().expect("a queued request has its notification")
            }
        };
        // Waker code runs outside the mutex, as do clone and drop below.
        wake.wake();
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Acquire<P, T> {
    fn new(pool: P) -> Self {
        Self {
            pool: Some(pool),
            node: UnsafeCell::new(Node { previous: ptr::null_mut(), next: ptr::null_mut(), queued: false, granted: None, waker: None }),
            _pin: PhantomPinned,
        }
    }
}

impl<T> State<T> {
    /// Caller holds the pool mutex; node is pinned and queued in this state.
    unsafe fn unlink(&mut self, node: *mut Node<T>) {
        unsafe {
            debug_assert!((*node).queued);
            let (previous, next) = ((*node).previous, (*node).next);
            if previous.is_null() {
                self.first = next;
            } else {
                (*previous).next = next;
            }
            if next.is_null() {
                self.last = previous;
            } else {
                (*next).previous = previous;
            }
            (*node).queued = false;
            (*node).previous = ptr::null_mut();
            (*node).next = ptr::null_mut();
        }
    }

    /// Caller holds the pool mutex; node is pinned and belongs to this request.
    unsafe fn take(&mut self, node: *mut Node<T>) -> Option<T> {
        unsafe {
            if let Some(value) = (*node).granted.take() {
                return Some(value);
            }
            if (*node).queued {
                return None;
            }
        }
        debug_assert!(self.first.is_null() || self.free.is_empty(), "waiting requests receive returned resources first");
        self.free.pop()
    }
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Future for Acquire<P, T> {
    type Output = Resource<P, T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Sound: neither this projection nor any operation below moves node.
        let this = unsafe { self.get_unchecked_mut() };
        let pool = this.pool.as_ref().expect("a completed resource request is consumed");
        let node = this.node.get();
        {
            let mut state = pool.state.lock().unwrap();
            // Sound: all node access uses this mutex, also on the fast path.
            if let Some(value) = unsafe { state.take(node) } {
                let old = unsafe { (*node).waker.take() };
                drop(state);
                drop(old);
                return Poll::Ready(Resource { pool: this.pool.take().unwrap(), value: Some(value) });
            }
            if unsafe { (*node).waker.as_ref().is_some_and(|w| w.will_wake(cx.waker())) } {
                return Poll::Pending;
            }
        }
        let new_waker = cx.waker().clone();
        let mut state = pool.state.lock().unwrap();
        // Capacity may have returned while the notification was cloned.
        // Sound: the same mutex protects this recheck and queue insertion.
        if let Some(value) = unsafe { state.take(node) } {
            let old = unsafe { (*node).waker.take() };
            drop(state);
            drop(old);
            drop(new_waker);
            return Poll::Ready(Resource { pool: this.pool.take().unwrap(), value: Some(value) });
        }
        let old = unsafe {
            if !(*node).queued {
                (*node).previous = state.last;
                if state.last.is_null() {
                    state.first = node;
                } else {
                    (*state.last).next = node;
                }
                state.last = node;
                (*node).queued = true;
            }
            (*node).waker.replace(new_waker)
        };
        drop(state);
        drop(old);
        Poll::Pending
    }
}

#[allow(dead_code)] // cancel: the integration tests' and the frontend's to come.
impl<P: Deref<Target = Pool<T>> + Unpin, T> Acquire<P, T> {
    /// Withdraw this request. A pending request leaves the queue; a grant
    /// received since the last poll returns to the pool. A completed
    /// request has nothing to withdraw and is a caller's mistake.
    pub(crate) fn cancel(self: Pin<&mut Self>) {
        // Sound: the node is unlinked under the mutex before this returns
        // and is never touched again through the queue.
        let this = unsafe { self.get_unchecked_mut() };
        let pool = this.pool.take().expect("a completed request has nothing to cancel");
        let node = this.node.get();
        let (value, wake) = {
            let mut state = pool.state.lock().unwrap();
            unsafe {
                if (*node).queued {
                    state.unlink(node);
                }
                ((*node).granted.take(), (*node).waker.take())
            }
        };
        drop(wake);
        if let Some(value) = value {
            pool.give(value);
        }
    }
}

/// A broken ownership contract ends the process at once, as the pool's
/// guards do. No cleanup runs: a queued node left behind would be read
/// after its memory is gone, and a lost resource would stall every waiter.
fn misuse(what: &str) -> ! {
    eprintln!("blake3-servil: {what}; the process ends (crash-only)");
    std::process::abort()
}

impl<P: Deref<Target = Pool<T>> + Unpin, T> Drop for Acquire<P, T> {
    fn drop(&mut self) {
        if self.pool.is_none() {
            return;
        }
        // Sound: pinned storage stays valid until this drop; the flag is
        // written only under the mutex, read here after the owner stopped
        // polling (misuse detection, no recovery).
        let node = self.node.get();
        if unsafe { (*node).queued || (*node).granted.is_some() } {
            misuse("a pending or granted resource request was dropped; cancel it or receive its grant");
        }
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

impl<P: Deref<Target = Pool<T>> + Unpin, T> Resource<P, T> {
    /// Give the resource back, waking the oldest waiter with it.
    pub(crate) fn release(mut self) {
        let value = self.value.take().unwrap();
        self.pool.give(value);
    }
}
