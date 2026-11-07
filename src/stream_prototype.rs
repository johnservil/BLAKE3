//! Incremental stream prototypes on the existing task source, worker pool
//! and kernels. This page stays test-only until the new frontend owns the
//! complete lifecycle. It supplies no alternative production worker engine.

use super::*;

struct ProducerWake(std::thread::Thread);
impl std::task::Wake for ProducerWake {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.0.unpark();
    }
}
fn producer_waker() -> std::task::Waker {
    std::task::Waker::from(std::sync::Arc::new(ProducerWake(std::thread::current())))
}
/// cx wakes this producer's thread. Pending input is waited for by parking.
fn wait_for<F: core::future::Future>(mut future: core::pin::Pin<&mut F>, cx: &mut core::task::Context<'_>) -> F::Output {
    loop {
        match future.as_mut().poll(cx) {
            core::task::Poll::Ready(value) => return value,
            core::task::Poll::Pending => std::thread::park(),
        }
    }
}

/// Input ownership makes a round trip through the actual worker pool.
/// The waiting producer parks; worker completion returns its buffer and
/// publishes the digest before waking it. Actions remain test-only until
/// the per-message frontend replaces the existing stream frontend.
#[test]
fn reserved_buffer_returns_from_an_actual_worker() {
    use crate::admission::{Lease, Pool};
    use core::future::Future;
    use core::task::{Context, Poll};

    crate::initialize_multithreaded(core::num::NonZeroUsize::new(2));
    let unwinding = crate::lanes::AbortOnUnwind;
    let source = Box::leak(Box::new(crate::lanes::sources::Source::new(2)));
    let slot = source.register().expect("room for the producer's source");
    let pool = Pool::new(vec![vec![crate::throughput::Block::ZERO; 256].into_boxed_slice()]);
    let waker = producer_waker();
    let mut cx = Context::from_waker(&waker);
    let mut initial = std::pin::pin!(pool.acquire());
    let Poll::Ready(mut input) = initial.as_mut().poll(&mut cx) else { panic!("the input buffer starts free") };
    crate::test::paint_test_input(crate::throughput::Block::bytes_mut(&mut input));
    let expected = *blake3::hash(crate::throughput::Block::bytes(&input)).as_bytes();
    let address = input.as_ptr();
    struct Work<'a> {
        input: UnsafeCell<Option<Lease<'a, Box<[crate::throughput::Block]>>>>,
        output: UnsafeCell<Hash>,
        worker: std::sync::Mutex<Option<std::thread::ThreadId>>,
        source: &'static crate::lanes::sources::Source,
        started: std::sync::Barrier,
        release: std::sync::Barrier,
    }
    unsafe fn run(context: *mut (), platform: Platform) {
        // Sound: the source barrier holds this context in place. This
        // worker alone takes the input and writes the output. Its grant
        // publishes those writes to the producer through the pool mutex.
        let work = unsafe { &*context.cast::<Work<'_>>() };
        let source = work.source;
        work.started.wait();
        work.release.wait();
        let input = unsafe { &mut *work.input.get() }.take().unwrap();
        let hash = crate::hash_serial_on(crate::throughput::Block::bytes(&input), crate::IV, 0, platform);
        unsafe { *work.output.get() = hash };
        *work.worker.lock().unwrap() = Some(std::thread::current().id());
        // All context reads precede returning input: a new owner may
        // immediately overwrite the returned buffer. The source is
        // persistent, so the final completion access stays valid too.
        drop(input);
        source.finished(1);
    }
    let work = Box::new(Work {
        input: UnsafeCell::new(Some(input)),
        output: UnsafeCell::new(Hash::from_bytes([0; 32])),
        worker: std::sync::Mutex::new(None),
        source,
        started: std::sync::Barrier::new(2),
        release: std::sync::Barrier::new(2),
    });
    let mut next_input = std::pin::pin!(pool.acquire());
    assert!(next_input.as_mut().poll(&mut cx).is_pending());
    // Sound: the worker takes work.input after publication. The scope
    // holds Work and Pool until the source has counted this action done.
    let bytes = unsafe { crate::throughput::Block::bytes((*work.input.get()).as_ref().unwrap()) };
    let mut task = Task::of(bytes, 0);
    task.action = Some((run, (&*work as *const Work<'_>).cast_mut().cast()));
    source.publish(task);
    source.wake_for_waiting();
    work.started.wait();
    assert!(next_input.as_mut().poll(&mut cx).is_pending(), "submission returned while the worker is deliberately held");
    work.release.wait();
    let mut returned = wait_for(next_input.as_mut(), &mut cx);
    assert_eq!(returned.as_ptr(), address, "the worker returns the same receive storage");
    crate::throughput::Block::bytes_mut(&mut returned).fill(0);
    source.drain_actions();
    assert_eq!(unsafe { (*work.output.get()).as_bytes() }, &expected);
    assert_ne!(work.worker.lock().unwrap().unwrap(), std::thread::current().id(), "hashing ran on the shared worker pool");
    crate::lanes::sources::Source::unregister(slot);
    core::mem::forget(unwinding);
}

/// One message continues across producer callbacks. Complete groups use
/// independent subtree kernels; final bytes and folding reuse HasherCore,
/// without the convenience Hasher's 16 KiB staging buffer.
#[test]
fn one_message_accepts_callback_pieces_and_finishes_on_the_workers() {
    use crate::admission::{Lease, Pool};
    use core::future::Future;
    use core::task::Context;
    const GROUP: usize = crate::incremental::STAGE_LEN;
    crate::initialize_multithreaded(core::num::NonZeroUsize::new(2));
    let unwinding = crate::lanes::AbortOnUnwind;
    let source = Box::leak(Box::new(crate::lanes::sources::Source::new(2)));
    let slot = source.register().expect("room for the producer's source");
    let pool = Pool::new(vec![vec![crate::throughput::Block::ZERO; GROUP / 64].into_boxed_slice()]);
    let waker = producer_waker();
    let mut cx = Context::from_waker(&waker);
    struct Work<'a> {
        input: UnsafeCell<Option<Lease<'a, Box<[crate::throughput::Block]>>>>,
        piece: UnsafeCell<(u64, usize, bool)>,
        core: std::sync::Mutex<crate::incremental::HasherCore>,
        output: UnsafeCell<Option<Hash>>,
        key: CVWords,
        flags: u8,
        source: &'static crate::lanes::sources::Source,
    }
    unsafe fn run(context: *mut (), platform: Platform) {
        // Sound: this action alone owns the input. The producer reuses its
        // context only after receiving that input back. Copy every immutable
        // field needed after the grant before returning the buffer.
        let work = unsafe { &*context.cast::<Work<'_>>() };
        let source = work.source;
        let input = unsafe { &mut *work.input.get() }.take().unwrap();
        let (counter, len, last) = unsafe { *work.piece.get() };
        let bytes = &crate::throughput::Block::bytes(&input)[..len];
        let mut result = [0; 64];
        if !last {
            if counter == 0 {
                result = crate::compress_subtree_to_parent_node(bytes, 0, &work.key, counter, work.flags, platform);
            } else {
                result[..32].copy_from_slice(&crate::hash_all_at_once(bytes, &work.key, counter, work.flags, platform).chaining_value());
            }
        }
        {
            let mut core = work.core.lock().unwrap();
            assert_eq!(core.count(), counter * CHUNK_LEN as u64, "this bounded prototype completes each prefix before reusing its one buffer");
            if last {
                core.update(bytes);
                unsafe { *work.output.get() = Some(core.final_output().root_hash()) };
            } else {
                core.push_group(&result);
            }
        }
        // The message lock is released before notifications can reenter.
        // No context access follows returning the buffer to the producer.
        drop(input);
        source.finished(1);
    }
    let key = [7; 32];
    const CONTEXT: &str = "BLAKE3-Servil pipeline callback test";
    let context_key = crate::throughput::context_key(CONTEXT);
    for mode in [Mode::Plain, Mode::Keyed(&key), Mode::DeriveKeyWith(&context_key)] {
        let (key_words, flags) = mode.key_and_flags();
        for len in [0, 1, 63, 64, 65, 1024, GROUP - 1, GROUP, GROUP + 1, 2 * GROUP, 2 * GROUP + 17] {
            let mut input = vec![0; len];
            crate::test::paint_test_input(&mut input);
            let expected = match mode {
                Mode::Plain => *blake3::hash(&input).as_bytes(),
                Mode::Keyed(key) => *blake3::keyed_hash(key, &input).as_bytes(),
                Mode::DeriveKey(_) | Mode::DeriveKeyWith(_) => blake3::derive_key(CONTEXT, &input),
            };
            let work = Box::new(Work {
                input: UnsafeCell::new(None),
                piece: UnsafeCell::new((0, 0, false)),
                core: std::sync::Mutex::new(crate::incremental::HasherCore::new_internal(&key_words, flags)),
                output: UnsafeCell::new(None),
                key: key_words,
                flags,
                source,
            });
            let full = len / GROUP * GROUP;
            let parts = input[..full].chunks(GROUP).map(|p| (p, false)).chain(core::iter::once((&input[full..], true)));
            for (i, (piece, last)) in parts.enumerate() {
                let mut request = std::pin::pin!(pool.acquire());
                let mut buffer = wait_for(request.as_mut(), &mut cx);
                let bytes = crate::throughput::Block::bytes_mut(&mut buffer);
                bytes[..piece.len()].copy_from_slice(piece);
                if last {
                    bytes[piece.len()..piece.len().max(1).div_ceil(64) * 64].fill(0);
                }
                // Sound: the last action returned this buffer before its
                // context can be reused. The source barrier holds Work.
                unsafe {
                    *work.input.get() = Some(buffer);
                    *work.piece.get() = ((i * GROUP / CHUNK_LEN) as u64, piece.len(), last);
                }
                let mut next = std::pin::pin!(pool.acquire());
                assert!(next.as_mut().poll(&mut cx).is_pending());
                let mut task = Task::of(&[], 0);
                // The final short update owns its own SME2 turn, as the
                // regular HasherCore path does. Groups use the worker's.
                task.members = usize::from(last);
                task.action = Some((run, (&*work as *const Work<'_>).cast_mut().cast()));
                source.publish(task);
                source.wake_for_waiting();
                drop(wait_for(next.as_mut(), &mut cx));
            }
            source.drain_actions();
            assert_eq!(unsafe { (*work.output.get()).unwrap() }.as_bytes(), &expected, "flags={flags} length={len}");
            assert_eq!(work.core.lock().unwrap().count(), len as u64);
        }
    }
    crate::lanes::sources::Source::unregister(slot);
    core::mem::forget(unwinding);
}
