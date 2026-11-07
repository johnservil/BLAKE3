//! The sketch's `Stream` over the receiver: messages and batches submitted
//! by one producer, completions delivered to its handler on its own thread
//! from `collect` and `finish`, and `ready` to wait for any of them.
//! Test-only until it replaces the submission frontend in stream.rs.

use super::receiver::{self, BatchCompletion, Completion, Receiver};
use crate::{Hash, Mode, OUT_LEN, throughput::Batch};
use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::sync::{Arc, Mutex};

pub(super) struct Limits {
    pub(super) buffers: usize,
    pub(super) messages: usize,
}

pub(super) enum Event<T> {
    Hashed {
        tag: T,
        hash: Hash,
    },
    Aborted {
        tag: T,
    },
    /// Every digest of the batch is in its places.
    Batch {
        tag: T,
    },
}

/// A completion the producer has yet to collect. Pushed with one poll, so
/// its waker is registered before `ready` can wait on it.
enum Pending<'a, T> {
    Message(T, Pin<Box<Completion>>),
    Batch(T, Pin<Box<BatchCompletion<'a>>>),
}

/// Completions wake the producer through this; `ready` waits on it.
#[derive(Default)]
struct Notify {
    flagged: bool,
    waker: Option<Waker>,
}
struct NotifyWake(Arc<Mutex<Notify>>);
impl std::task::Wake for NotifyWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let waker = {
            let mut notify = self.0.lock().unwrap();
            notify.flagged = true;
            notify.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

pub(super) struct Stream<'a, T, F: FnMut(Event<T>)> {
    receiver: Receiver,
    on_completion: RefCell<F>,
    pending: RefCell<Vec<Pending<'a, T>>>,
    notify: Arc<Mutex<Notify>>,
    completion_waker: Waker,
}

pub(super) struct Message<'s, 'a, T, F: FnMut(Event<T>)> {
    stream: &'s Stream<'a, T, F>,
    pub(super) inner: receiver::Message,
    tag: T,
}

impl<'a, T: Copy, F: FnMut(Event<T>)> Stream<'a, T, F> {
    pub(super) fn new(mode: Mode<'_>, limits: Limits, on_completion: F) -> Self {
        let notify = Arc::new(Mutex::new(Notify::default()));
        Self {
            receiver: Receiver::new(mode, limits.buffers, limits.messages),
            on_completion: RefCell::new(on_completion),
            pending: RefCell::new(Vec::with_capacity(limits.messages)),
            completion_waker: Waker::from(Arc::new(NotifyWake(notify.clone()))),
            notify,
        }
    }

    /// Begin a message once a message slot is reserved. While it waits, it
    /// delivers completions that are ready (a slot returns through them), as
    /// a stream's submission hands places back while it waits for room.
    pub(super) async fn message(&self, tag: T, length_hint: Option<u64>) -> Message<'_, 'a, T, F> {
        Message { stream: self, inner: self.admit(self.receiver.message(length_hint)).await, tag }
    }

    /// Submit a prepared batch once admitted (delivering completions as
    /// `message` does while it waits); its digests land in `places`.
    pub(super) async fn submit_batch(&self, batch: &'a Batch<'a>, places: &'a mut [[u8; OUT_LEN]], tag: T) {
        let completion = self.admit(self.receiver.submit_batch(batch, places)).await;
        self.push(Pending::Batch(tag, Box::pin(completion)));
    }

    /// Await an admission, collecting while it is pending: a slot held by an
    /// uncollected completion comes back through `collect`, so a wait that
    /// collected nothing would otherwise never end. Wakes on the admission
    /// itself and on any completion.
    async fn admit<R>(&self, admission: impl Future<Output = R>) -> R {
        let mut admission = core::pin::pin!(admission);
        core::future::poll_fn(|cx| {
            loop {
                if let Poll::Ready(r) = admission.as_mut().poll(cx) {
                    return Poll::Ready(r);
                }
                if self.collect() == 0 {
                    break;
                }
                // Something was delivered: a slot may be free now; poll again.
            }
            let mut notify = self.notify.lock().unwrap();
            notify.flagged = false;
            notify.waker = Some(cx.waker().clone());
            Poll::Pending
        })
        .await
    }

    /// Deliver every completion that is ready, without waiting: how many.
    pub(super) fn collect(&self) -> usize {
        let mut delivered = 0;
        let mut i = 0;
        loop {
            let event = {
                let mut pending = self.pending.borrow_mut();
                let Some(item) = pending.get_mut(i) else { break };
                let Some(event) = self.poll_pending(item) else {
                    i += 1;
                    continue;
                };
                pending.swap_remove(i);
                event
            };
            // The handler runs with no borrow held: it may submit.
            (self.on_completion.borrow_mut())(event);
            delivered += 1;
        }
        delivered
    }

    /// One poll of a pending completion with the stream's waker: its event
    /// once ready, its slot then retired.
    fn poll_pending(&self, item: &mut Pending<'a, T>) -> Option<Event<T>> {
        let mut cx = Context::from_waker(&self.completion_waker);
        match item {
            Pending::Message(tag, completion) => match completion.as_mut().poll(&mut cx) {
                Poll::Ready(finished) => {
                    let outcome = finished.outcome().copied();
                    finished.retire();
                    Some(match outcome {
                        Some(hash) => Event::Hashed { tag: *tag, hash },
                        None => Event::Aborted { tag: *tag },
                    })
                }
                Poll::Pending => None,
            },
            Pending::Batch(tag, completion) => match completion.as_mut().poll(&mut cx) {
                Poll::Ready(finished) => {
                    finished.retire();
                    Some(Event::Batch { tag: *tag })
                }
                Poll::Pending => None,
            },
        }
    }

    /// Keep a completion for `collect`, polled once now so that its waker
    /// is registered; one already ready is delivered at once.
    fn push(&self, mut item: Pending<'a, T>) {
        match self.poll_pending(&mut item) {
            Some(event) => (self.on_completion.borrow_mut())(event),
            None => self.pending.borrow_mut().push(item),
        }
    }

    /// Wait until some completion may be ready (then `collect`).
    pub(super) async fn ready(&self) {
        core::future::poll_fn(|cx| {
            let mut notify = self.notify.lock().unwrap();
            if core::mem::take(&mut notify.flagged) {
                return Poll::Ready(());
            }
            notify.waker = Some(cx.waker().clone());
            Poll::Pending
        })
        .await
    }

    /// Wait for every accepted piece of work, deliver the rest, and close.
    /// Every message has been ended or aborted.
    pub(super) fn finish(self) {
        self.receiver.drain();
        while !self.pending.borrow().is_empty() {
            assert!(self.collect() > 0, "after the drain, every pending completion is ready");
        }
        self.receiver.close();
    }
}

impl<'a, T: Copy, F: FnMut(Event<T>)> Message<'_, 'a, T, F> {
    pub(super) async fn input(&mut self) -> receiver::Input<'_> {
        self.inner.input().await
    }
    pub(super) fn end(self) {
        self.stream.push(Pending::Message(self.tag, Box::pin(self.inner.end())));
    }
    pub(super) fn abort(self) {
        self.stream.push(Pending::Message(self.tag, Box::pin(self.inner.abort())));
    }
}

#[cfg(test)]
mod tests {
    use super::super::receiver::tests::{expected, run};
    use super::*;
    use crate::throughput::Block;
    use std::cell::RefCell;

    fn stream<'a, T: Copy>(buffers: usize, messages: usize, events: &'a RefCell<Vec<Event<T>>>) -> Stream<'a, T, impl FnMut(Event<T>) + 'a> {
        crate::initialize_multithreaded(core::num::NonZeroUsize::new(3));
        Stream::new(Mode::Plain, Limits { buffers, messages }, move |event| events.borrow_mut().push(event))
    }

    /// A reader standing in for a socket, as the sketch's `read_fragment`.
    struct Reader<'a> {
        input: &'a [u8],
        at: usize,
        turn: usize,
        fail_at: Option<usize>,
    }
    impl Reader<'_> {
        fn read_fragment(&mut self, room: &mut [u8]) -> Result<(usize, bool), &'static str> {
            if self.fail_at == Some(self.turn) {
                return Err("connection reset");
            }
            let wanted = [1448, 4096, 16 * 1024, 64 * 1024][self.turn % 4];
            self.turn += 1;
            let n = wanted.min(room.len()).min(self.input.len() - self.at);
            room[..n].copy_from_slice(&self.input[self.at..self.at + n]);
            self.at += n;
            Ok((n, self.at == self.input.len()))
        }
    }

    /// The sketch's loop, verbatim but for `run` in place of `.await`.
    fn receive<'s, 'a, T: Copy, F: FnMut(Event<T>)>(stream: &'s Stream<'a, T, F>, tag: T, reader: &mut Reader<'_>) -> Result<(), &'static str> {
        let mut message = run(stream.message(tag, Some(reader.input.len() as u64)));
        loop {
            let mut input = run(message.input());
            let ended = match reader.read_fragment(input.bytes_mut()) {
                Ok((written, ended)) => {
                    input.commit(written);
                    ended
                }
                Err(error) => {
                    message.abort();
                    return Err(error);
                }
            };
            if ended {
                message.end();
                return Ok(());
            }
        }
    }

    #[test]
    fn messages_batches_and_an_abort_reach_the_handler() {
        // A batch and its places outlive the stream that borrows them: the
        // borrow checker holds the producer to it (declared first).
        let mut batch = Batch::new(100, 20);
        for i in 0..batch.len() {
            batch.set(i, &[i as u8; 100]);
        }
        let mut places = vec![[0u8; OUT_LEN]; batch.len()];
        let events = RefCell::new(Vec::new());
        let stream = stream(4, 4, &events);
        let mut input = vec![0u8; (1 << 20) + 777];
        crate::test::paint_test_input(&mut input);
        assert_eq!(receive(&stream, "long", &mut Reader { input: &input, at: 0, turn: 0, fail_at: None }), Ok(()));
        run(stream.submit_batch(&batch, &mut places, "batch"));
        assert_eq!(receive(&stream, "reset", &mut Reader { input: &input, at: 0, turn: 0, fail_at: Some(5) }), Err("connection reset"));
        assert_eq!(receive(&stream, "short", &mut Reader { input: &input[..1000], at: 0, turn: 0, fail_at: None }), Ok(()));
        // The producer waits for completions as an event loop would.
        while events.borrow().len() < 4 {
            run(stream.ready());
            stream.collect();
        }
        stream.finish();
        let mut got: Vec<String> = events
            .borrow()
            .iter()
            .map(|e| match e {
                Event::Hashed { tag, hash } => format!("{tag}: {}", hex::encode(hash.as_bytes())),
                Event::Aborted { tag } => format!("{tag}: aborted"),
                Event::Batch { tag } => format!("{tag}: batch"),
            })
            .collect();
        got.sort();
        let mut want = vec![
            format!("long: {}", hex::encode(expected(Mode::Plain, &input))),
            format!("short: {}", hex::encode(expected(Mode::Plain, &input[..1000]))),
            "reset: aborted".to_string(),
            "batch: batch".to_string(),
        ];
        want.sort();
        assert_eq!(got, want);
        for (i, place) in places.iter().enumerate() {
            assert_eq!(place, &expected(Mode::Plain, &[i as u8; 100]), "batch message {i}");
        }
    }

    /// A later short message is delivered while an earlier one is held.
    #[test]
    fn a_later_message_is_delivered_before_an_earlier_held_one() {
        let events = RefCell::new(Vec::new());
        let stream = stream(3, 2, &events);
        let mut long = vec![0u8; crate::incremental::STAGE_LEN];
        crate::test::paint_test_input(&mut long);
        let gate = super::super::receiver::Gate::new();
        let mut first = run(stream.message(1u8, None));
        first.inner.hold_next(gate.clone());
        {
            let mut room = run(first.input());
            room.bytes_mut().copy_from_slice(&long);
            room.commit(long.len());
        }
        gate.started.wait();
        let mut second = run(stream.message(2u8, Some(3)));
        {
            let mut room = run(second.input());
            room.bytes_mut()[..3].copy_from_slice(b"abc");
            room.commit(3);
        }
        second.end();
        // The completion may be delivered by end() itself when the worker
        // is quick; otherwise ready() waits for it.
        while events.borrow().is_empty() {
            run(stream.ready());
            stream.collect();
        }
        assert!(matches!(events.borrow()[0], Event::Hashed { tag: 2, .. }), "message 2 delivered while 1 is held");
        gate.release.wait();
        first.end();
        stream.finish();
        assert!(matches!(events.borrow()[1], Event::Hashed { tag: 1, hash } if hash.as_bytes() == &expected(Mode::Plain, &long)));
    }

    /// Many messages interleaved at random, in fragments of random lengths,
    /// over a tight budget: every digest matches, every slot and buffer
    /// comes back, with a fixed seed so a failure repeats.
    #[test]
    fn interleaved_messages_in_random_fragments_match_official() {
        use rand::{RngExt, SeedableRng};
        let events = RefCell::new(Vec::new());
        let stream = stream(3, 2, &events);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed_b3);
        let mut inputs: Vec<Vec<u8>> = (0..24)
            .map(|i| {
                let len = [0, 1, 63, 64, 65, 1000, 16384, 16385, 40000, 100_000][i % 10] + (i / 10) * 7;
                let mut v = vec![0u8; len];
                crate::test::paint_test_input(&mut v);
                v
            })
            .collect();
        let mut open: Vec<(usize, Message<'_, '_, usize, _>, usize)> = Vec::new();
        let mut next = 0;
        while next < inputs.len() || !open.is_empty() {
            if open.len() < 2 && next < inputs.len() {
                open.push((next, run(stream.message(next, Some(inputs[next].len() as u64))), 0));
                next += 1;
                continue;
            }
            let k = rng.random_range(0..open.len());
            let (i, message, at) = &mut open[k];
            let input = &inputs[*i];
            if *at == input.len() {
                let (i, message, _) = open.swap_remove(k);
                let _ = i;
                message.end();
                stream.collect();
                continue;
            }
            let mut room = run(message.input());
            let n = rng.random_range(1..=room.bytes_mut().len().min(input.len() - *at).max(1));
            room.bytes_mut()[..n].copy_from_slice(&input[*at..*at + n]);
            room.commit(n);
            *at += n;
        }
        stream.finish();
        let mut got: Vec<(usize, [u8; 32])> = events
            .borrow()
            .iter()
            .map(|e| match e {
                Event::Hashed { tag, hash } => (*tag, *hash.as_bytes()),
                _ => panic!("every message hashes"),
            })
            .collect();
        got.sort();
        assert_eq!(got.len(), inputs.len());
        for (tag, hash) in got {
            assert_eq!(hash, expected(Mode::Plain, &inputs[tag]), "message {tag}");
        }
        inputs.clear();
    }

    /// Whole-block storage is borrowed as a batch without a copy.
    #[test]
    fn a_borrowed_batch_fills_the_producers_places() {
        let mut blocks = vec![Block::ZERO; 8];
        crate::test::paint_test_input(Block::bytes_mut(&mut blocks));
        let borrowed = Batch::from_blocks(&blocks, 128);
        let mut places = [[0u8; OUT_LEN]; 4];
        let events = RefCell::new(Vec::new());
        let stream = stream(1, 1, &events);
        run(stream.submit_batch(&borrowed, &mut places, ()));
        stream.finish();
        assert_eq!(events.borrow().len(), 1);
        for (i, place) in places.iter().enumerate() {
            assert_eq!(place, &expected(Mode::Plain, &Block::bytes(&blocks)[i * 128..][..128]));
        }
    }
}
