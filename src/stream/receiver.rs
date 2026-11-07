//! Bounded per-message progress on the existing source and worker pool.
//!
//! Each input buffer reserves its metadata slot too. A completed prefix
//! remains in that slot until earlier prefixes of its message have folded.
//! This bounds queued work and completed-but-unmerged results together.
//!
//! Crash-only: no Drop does lifecycle work. Ending, aborting, releasing a
//! grant and retiring a result are explicit operations. A producer that
//! unwinds while its work is accepted ends the process (AbortOnUnwind), as
//! the stream scope does. This module is test-only until frontend integration.

use crate::admission::{Acquire, OwnedLease, Pool, initialized_mutex};
use crate::incremental::{HasherCore, STAGE_LEN};
use crate::lanes::{Task, sources::Source};
use crate::platform::Platform;
use crate::{
    CHUNK_LEN, CVWords, Hash, Mode, OUT_LEN,
    throughput::{Batch, Block},
};
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::collections::VecDeque;
use std::sync::{Arc, Barrier, Mutex};

const INPUT_BYTES: usize = STAGE_LEN;

struct Buffer {
    index: usize,
    blocks: Box<[Block]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Id {
    index: usize,
    generation: u64,
}

#[derive(Default)]
struct Hooks {
    before: Option<Arc<Gate>>,
    after: Option<Arc<Barrier>>,
}

pub(super) struct Gate {
    pub(super) started: Barrier,
    pub(super) release: Barrier,
}
impl Gate {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self { started: Barrier::new(2), release: Barrier::new(2) })
    }
}

struct Job {
    owner: Arc<Core>,
    id: Id,
    buffer: OwnedLease<Buffer>,
    counter: u64,
    len: usize,
    last: bool,
    hooks: Hooks,
}

struct Completed {
    job: Job,
    result: [u8; 64],
}

enum SlotState {
    Vacant,
    Queued(Job),
    Running,
    Ready(Completed),
}
struct Slot {
    state: Mutex<SlotState>,
}

#[derive(Debug)]
enum Outcome {
    Pending,
    Hashed(Hash),
    Aborted,
}

struct Progress {
    generation: u64,
    core: HasherCore,
    queue: VecDeque<usize>,
    writer_open: bool,
    ended: bool,
    aborted: bool,
    done: bool,
    receiver_gone: bool,
    slot: Option<OwnedLease<usize>>,
    output: Option<OwnedLease<Outcome>>,
}

struct Core {
    source: &'static Source,
    registration: usize,
    key: CVWords,
    flags: u8,
    buffers: Arc<Pool<Buffer>>,
    message_slots: Arc<Pool<usize>>,
    messages: Box<[Mutex<Progress>]>,
    outputs: Box<[Arc<Pool<Outcome>>]>,
    slots: Box<[Slot]>,
}

/// One test producer. Message and result owners can outlive this handle.
pub(super) struct Receiver {
    core: Arc<Core>,
}

impl Receiver {
    /// Program initialization has configured at least one background worker.
    pub(super) fn new(mode: Mode<'_>, buffers: usize, messages: usize) -> Self {
        assert!(buffers > 0 && messages > 0, "the receiver has input and message capacity");
        let (key, flags) = mode.key_and_flags();
        // Sources persist, as the existing registry requires. The eventual
        // frontend will reuse its storage pool; each test owns one source.
        let source = Box::leak(Box::new(Source::new(buffers)));
        let registration = source.register().expect("a source slot for this prototype");
        source.drain_actions(); // Initialize its platform completion metadata.
        let input = (0..buffers).map(|index| Buffer { index, blocks: vec![Block::ZERO; INPUT_BYTES / 64].into_boxed_slice() }).collect();
        let slots = (0..buffers).map(|_| Slot { state: initialized_mutex(SlotState::Vacant) }).collect();
        let states = (0..messages)
            .map(|_| {
                initialized_mutex(Progress {
                    generation: 0,
                    core: HasherCore::new_internal(&key, flags),
                    queue: VecDeque::with_capacity(buffers),
                    writer_open: false,
                    ended: true,
                    aborted: false,
                    done: true,
                    receiver_gone: true,
                    slot: None,
                    output: None,
                })
            })
            .collect();
        let outputs = (0..messages).map(|_| Arc::new(Pool::new(vec![Outcome::Pending]))).collect();
        Self {
            core: Arc::new(Core {
                source,
                registration,
                key,
                flags,
                buffers: Arc::new(Pool::new(input)),
                message_slots: Arc::new(Pool::new((0..messages).collect())),
                messages: states,
                outputs,
                slots,
            }),
        }
    }

    pub(super) fn message(&self, length_hint: Option<u64>) -> impl Future<Output = Message> + Send + 'static {
        let core = self.core.clone();
        async move {
            // Hints are advisory. The scheduling prototype accepts but does
            // not use them, as submit_reader does today.
            let _ = length_hint;
            let slot = core.message_slots.clone().acquire_owned().await;
            let index = *slot;
            let mut output = core.outputs[index].clone().acquire_owned().await;
            *output = Outcome::Pending;
            let id = {
                let mut state = core.messages[index].lock().unwrap();
                assert!(state.queue.is_empty() && state.slot.is_none(), "a returned message slot has retired its work");
                state.generation = state.generation.checked_add(1).expect("message generations fit u64");
                state.core.reset();
                state.writer_open = true;
                state.ended = false;
                state.aborted = false;
                state.done = false;
                state.receiver_gone = false;
                state.slot = Some(slot);
                state.output = Some(output);
                Id { index, generation: state.generation }
            };
            Message { current: None, used: 0, sent: 0, next_hooks: Hooks::default(), core, id, guard: crate::lanes::AbortOnUnwind }
        }
    }

    /// Hash a prepared batch in its compact layout, in `place`s (one per
    /// message). Completes at admission: a message-slot reservation and one
    /// bounded batch job on the pool. Its result token resolves when workers
    /// have written every digest. Borrowing keeps the batch and places in
    /// place until then; the result's lifetime carries that borrow.
    pub(super) fn submit_batch<'a>(&self, batch: &'a Batch<'a>, places: &'a mut [[u8; OUT_LEN]]) -> impl Future<Output = BatchCompletion<'a>> + 'a {
        assert_eq!(batch.len(), places.len(), "one place per batch message");
        let core = self.core.clone();
        async move {
            let slot = core.message_slots.clone().acquire_owned().await;
            let index = *slot;
            let mut output = core.outputs[index].clone().acquire_owned().await;
            *output = Outcome::Pending;
            let id = {
                let mut state = core.messages[index].lock().unwrap();
                assert!(state.queue.is_empty() && state.slot.is_none(), "a returned message slot has retired its work");
                state.generation = state.generation.checked_add(1).expect("message generations fit u64");
                state.writer_open = false;
                state.ended = true;
                state.aborted = false;
                state.done = false;
                state.receiver_gone = false;
                state.slot = Some(slot);
                state.output = Some(output);
                Id { index, generation: state.generation }
            };
            let job = Box::new(BatchJob {
                owner: core.clone(),
                id,
                input: Block::bytes(batch.blocks()).as_ptr(),
                input_len: Block::bytes(batch.blocks()).len(),
                message_len: batch.message_len(),
                out: places.as_mut_ptr(),
                count: places.len(),
            });
            let context = Box::into_raw(job);
            let mut task = Task::of(&[], 0);
            task.action = Some((execute_batch, context.cast()));
            core.source.publish(task);
            core.source.wake_for_waiting();
            BatchCompletion {
                request: core.outputs[id.index].clone().acquire_owned(),
                core,
                id,
                guard: Some(crate::lanes::AbortOnUnwind),
                borrow: core::marker::PhantomData,
            }
        }
    }

    /// Explicit test scope boundary, using the source's existing finish line.
    pub(super) fn drain(&self) {
        assert!(self.core.messages.iter().all(|s| !s.lock().unwrap().writer_open), "end or abort every message before draining");
        self.core.source.drain_actions();
    }
}

impl Receiver {
    /// Give the registry slot back. Every message has retired its result,
    /// and drain has run, so no task reaches this core again.
    pub(super) fn close(self) {
        assert!(self.core.messages.iter().all(|s| s.lock().unwrap().slot.is_none()), "every message retired its result before close");
        Source::unregister(self.core.registration);
    }
}

pub(super) struct Message {
    current: Option<OwnedLease<Buffer>>,
    used: usize,
    sent: u64,
    next_hooks: Hooks,
    core: Arc<Core>,
    id: Id,
    /// Dropped only if the producer unwinds with work accepted: the process
    /// ends, since workers may still write this message's state.
    guard: crate::lanes::AbortOnUnwind,
}

pub(super) struct Input<'a> {
    message: &'a mut Message,
}
impl Input<'_> {
    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        let message = &mut *self.message;
        &mut Block::bytes_mut(&mut message.current.as_mut().unwrap().blocks)[message.used..]
    }
    /// written is the number of bytes the producer wrote at the room's start.
    pub(super) fn commit(self, written: usize) {
        let message = &mut *self.message;
        assert!(written <= INPUT_BYTES - message.used, "the write fits its granted room");
        message.used += written;
        if message.used == INPUT_BYTES {
            message.submit(false);
        }
    }
}
impl Input<'_> {
    /// Give an unused room back without committing bytes.
    pub(super) fn release(self) {
        let message = &mut *self.message;
        assert_eq!(message.used, 0, "a room with committed bytes stays with its message");
        message.current.take().unwrap().release();
    }
}

impl Message {
    pub(super) async fn input(&mut self) -> Input<'_> {
        if self.current.is_none() {
            self.current = Some(self.core.buffers.clone().acquire_owned().await);
        }
        Input { message: self }
    }

    /// The next submitted part uses these controlled-schedule test hooks.
    pub(super) fn hold_next(&mut self, gate: Arc<Gate>) {
        self.next_hooks.before = Some(gate);
    }
    pub(super) fn observe_next(&mut self, barrier: Arc<Barrier>) {
        self.next_hooks.after = Some(barrier);
    }

    fn submit(&mut self, last: bool) {
        let mut buffer = self.current.take().expect("submission owns reserved input and metadata");
        let len = self.used;
        if last {
            let padded = len.max(1).div_ceil(64) * 64;
            Block::bytes_mut(&mut buffer.blocks)[len..padded].fill(0);
        } else {
            assert_eq!(len, INPUT_BYTES, "a nonfinal input fills whole algorithm groups");
        }
        let index = buffer.index;
        let owner = self.core.clone();
        let id = self.id;
        let job = Job { owner: owner.clone(), id, buffer, counter: self.sent / CHUNK_LEN as u64, len, last, hooks: core::mem::take(&mut self.next_hooks) };
        {
            let mut slot = owner.slots[index].state.lock().unwrap();
            assert!(matches!(*slot, SlotState::Vacant), "a buffer reserves its vacant metadata slot");
            *slot = SlotState::Queued(job);
        }
        {
            let mut state = owner.messages[id.index].lock().unwrap();
            assert_eq!(state.generation, id.generation);
            assert!(state.writer_open && !state.ended, "only the live writer submits input");
            assert!(state.queue.len() < state.queue.capacity(), "reserved metadata bounds each message's work queue");
            state.queue.push_back(index);
            if last {
                state.ended = true;
                state.writer_open = false;
            }
        }
        self.sent += len as u64;
        self.used = 0;
        let mut task = Task::of(&[], 0);
        // The final short update takes its own SME2 turn. A complete group
        // uses the worker's turn, as the existing stream's piece does.
        task.members = usize::from(last);
        task.action = Some((execute, (&owner.slots[index] as *const Slot).cast_mut().cast()));
        owner.source.publish(task);
        owner.source.wake_for_waiting();
    }

    pub(super) fn end(mut self) -> Completion {
        if self.current.is_some() && self.used > 0 {
            self.submit(true);
        } else {
            if let Some(room) = self.current.take() {
                room.release();
            }
            let mut state = self.core.messages[self.id.index].lock().unwrap();
            state.ended = true;
            state.writer_open = false;
        }
        let Message { core, id, guard, .. } = self;
        core::mem::forget(guard); // The completion carries its own guard.
        core.finalize_if_idle(id);
        Completion { request: core.outputs[id.index].clone().acquire_owned(), core, id, guard: Some(crate::lanes::AbortOnUnwind) }
    }

    /// Stop this message: no digest. Queued work for it is skipped; running
    /// jobs finish before its buffers return. The slot returns once the
    /// pool has retired that work.
    pub(super) fn abort(mut self) -> Aborted {
        if let Some(room) = self.current.take() {
            room.release();
        }
        let Message { core, id, guard, .. } = self;
        core::mem::forget(guard);
        core.forget(id, true);
        Aborted { core, id }
    }
}

/// An aborted message's slot comes back once its accepted work has retired.
/// The scope's drain waits for that work; then `retire` gives the slot back.
pub(super) struct Aborted {
    core: Arc<Core>,
    id: Id,
}
impl Aborted {
    pub(super) fn retire(self) {
        self.core.retire(self.id);
    }
}

/// A batch's bounded job: whole prepared messages and their digest places,
/// borrowed for the completion's lifetime.
struct BatchJob {
    owner: Arc<Core>,
    id: Id,
    input: *const u8,
    input_len: usize,
    message_len: usize,
    out: *mut [u8; OUT_LEN],
    count: usize,
}

/// A batch's result. The borrowed batch and places stay in place until the
/// receiver drains; a producer that unwinds first ends the process (a test
/// that panicked before draining once let a worker write into a freed stack
/// frame; the guard now aborts there).
pub(super) struct BatchCompletion<'a> {
    request: Acquire<Arc<Pool<Outcome>>, Outcome>,
    core: Arc<Core>,
    id: Id,
    guard: Option<crate::lanes::AbortOnUnwind>,
    borrow: core::marker::PhantomData<&'a mut [u8]>,
}
impl Future for BatchCompletion<'_> {
    type Output = BatchFinished;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<BatchFinished> {
        // Sound: request stays pinned; the other fields are Unpin and are
        // read only after the result transfers.
        let this = unsafe { self.get_unchecked_mut() };
        let request = unsafe { Pin::new_unchecked(&mut this.request) };
        match request.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                assert!(matches!(*result, Outcome::Hashed(_)), "a batch job completes with its digests written");
                core::mem::forget(this.guard.take().expect("the guard is live until the result transfers or the wait is cancelled"));
                Poll::Ready(BatchFinished { result, core: this.core.clone(), id: this.id })
            }
        }
    }
}
/// A finished batch: every digest is in its place. `retire` returns its slot.
pub(super) struct BatchFinished {
    result: OwnedLease<Outcome>,
    core: Arc<Core>,
    id: Id,
}
impl BatchFinished {
    pub(super) fn retire(self) {
        self.result.release();
        self.core.forget(self.id, false);
        self.core.retire(self.id);
    }
}

unsafe fn execute_batch(context: *mut (), platform: Platform) {
    // Sound: the job box was published by submit_batch and is claimed by
    // this one task. The batch and places outlive the receiver's drain.
    let job = unsafe { Box::from_raw(context.cast::<BatchJob>()) };
    let aborted = job.owner.messages[job.id.index].lock().unwrap().aborted;
    if !aborted {
        let input = unsafe { core::slice::from_raw_parts(job.input, job.input_len) };
        let out = unsafe { core::slice::from_raw_parts_mut(job.out, job.count) };
        crate::many::hash_many_on(input, job.message_len, &job.owner.key, job.owner.flags, out, platform);
    }
    let owner = job.owner.clone();
    let id = job.id;
    drop(job);
    {
        let mut state = owner.messages[id.index].lock().unwrap();
        assert_eq!(state.generation, id.generation);
        let mut output = state.output.take().expect("one completion token per batch");
        *output = if aborted { Outcome::Aborted } else { Outcome::Hashed(Hash::from_bytes([0; 32])) };
        state.done = true;
        drop(state);
        output.release();
    }
    owner.finalize_if_idle(id);
    owner.source.finished(1);
}

/// An ended message's pending result. Await it, or `cancel` it to stop
/// waiting; either way its slot returns through `retire`.
pub(super) struct Completion {
    request: Acquire<Arc<Pool<Outcome>>, Outcome>,
    core: Arc<Core>,
    id: Id,
    /// Live while the result is pending: dropped only if the producer
    /// unwinds, which ends the process. Taken and forgotten otherwise.
    guard: Option<crate::lanes::AbortOnUnwind>,
}
impl Future for Completion {
    type Output = Finished;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Finished> {
        // Sound: request stays pinned; the other fields are Unpin and are
        // read only after request has transferred its result.
        let this = unsafe { self.get_unchecked_mut() };
        let request = unsafe { Pin::new_unchecked(&mut this.request) };
        match request.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                core::mem::forget(this.guard.take().expect("the guard is live until the result transfers or the wait is cancelled"));
                Poll::Ready(Finished { result, core: this.core.clone(), id: this.id })
            }
        }
    }
}
impl Completion {
    /// Stop waiting for the result. Accepted work still runs to its end;
    /// the scope's drain waits for it, then `retire` gives the slot back.
    pub(super) fn cancel(self: Pin<&mut Self>) -> Aborted {
        // Sound: request stays pinned through cancel; the guard is taken
        // and forgotten, so no abort fires on this ordinary path.
        let this = unsafe { self.get_unchecked_mut() };
        unsafe { Pin::new_unchecked(&mut this.request) }.cancel();
        core::mem::forget(this.guard.take().expect("the guard is live until the result transfers or the wait is cancelled"));
        this.core.forget(this.id, true);
        Aborted { core: this.core.clone(), id: this.id }
    }
}

/// A finished message's digest. `retire` returns its result token and slot.
pub(super) struct Finished {
    result: OwnedLease<Outcome>,
    core: Arc<Core>,
    id: Id,
}
impl Finished {
    pub(super) fn hash(&self) -> &Hash {
        match &*self.result {
            Outcome::Hashed(hash) => hash,
            _ => panic!("only successful completion yields a digest"),
        }
    }
    pub(super) fn retire(self) {
        self.result.release();
        self.core.forget(self.id, false);
        self.core.retire(self.id);
    }
}

impl Core {
    fn forget(&self, id: Id, cancel: bool) {
        {
            let mut state = self.messages[id.index].lock().unwrap();
            assert_eq!(state.generation, id.generation, "the result retains its message slot");
            state.receiver_gone = true;
            if cancel && !state.done {
                state.aborted = true;
                state.ended = true;
                state.writer_open = false;
            }
        }
        self.finalize_if_idle(id);
    }

    fn finalize_if_idle(&self, id: Id) {
        let output = {
            let mut state = self.messages[id.index].lock().unwrap();
            if state.generation != id.generation || !state.queue.is_empty() {
                return;
            }
            let output = if state.ended && !state.done {
                let mut output = state.output.take().expect("one completion token per message");
                *output = if state.aborted { Outcome::Aborted } else { Outcome::Hashed(state.core.final_output().root_hash()) };
                state.done = true;
                Some(output)
            } else {
                None
            };
            output
        };
        // A waker may poll another operation immediately. The notification
        // happens after releasing the message lock.
        if let Some(output) = output {
            output.release();
        }
    }

    /// Give a message's slot back: its work has retired (the scope drained)
    /// and its result holder is gone.
    fn retire(&self, id: Id) {
        let slot = {
            let mut state = self.messages[id.index].lock().unwrap();
            assert_eq!(state.generation, id.generation);
            assert!(state.done && state.receiver_gone && state.queue.is_empty(), "retire after the result and the drain");
            state.slot.take().expect("one retire per message")
        };
        slot.release();
    }

    fn consume(&self, id: Id) {
        loop {
            let completed = {
                let mut state = self.messages[id.index].lock().unwrap();
                if state.generation != id.generation {
                    return;
                }
                let Some(&index) = state.queue.front() else {
                    drop(state);
                    self.finalize_if_idle(id);
                    return;
                };
                let mut slot = self.slots[index].state.lock().unwrap();
                if !matches!(*slot, SlotState::Ready(_)) {
                    return;
                }
                let SlotState::Ready(completed) = core::mem::replace(&mut *slot, SlotState::Vacant) else { unreachable!() };
                state.queue.pop_front();
                if !state.aborted {
                    assert_eq!(state.core.count(), completed.job.counter * CHUNK_LEN as u64, "fold prefixes in message order");
                    if completed.job.last {
                        let input = &Block::bytes(&completed.job.buffer.blocks)[..completed.job.len];
                        state.core.update(input);
                    } else {
                        state.core.push_group(&completed.result);
                    }
                }
                completed
            };
            // Return input and its metadata reservation outside both locks.
            // Another producer can reuse this slot immediately afterward.
            completed.job.buffer.release();
        }
    }
}

unsafe fn execute(context: *mut (), platform: Platform) {
    // Sound: the Queued Job owns the Core containing this stable slot.
    // Claiming the job transfers that ownership to this worker. No caller
    // can recover the buffer until the completed part has been retired.
    let slot = unsafe { &*context.cast::<Slot>() };
    let job = {
        let mut state = slot.state.lock().unwrap();
        let SlotState::Queued(job) = core::mem::replace(&mut *state, SlotState::Running) else { panic!("a task claims one queued job") };
        job
    };
    let owner = job.owner.clone();
    let id = job.id;
    let after = job.hooks.after.clone();
    if let Some(gate) = &job.hooks.before {
        gate.started.wait();
        gate.release.wait();
    }
    let aborted = owner.messages[id.index].lock().unwrap().aborted;
    let mut result = [0; 64];
    if !aborted && !job.last {
        let input = &Block::bytes(&job.buffer.blocks)[..job.len];
        if job.counter == 0 {
            result = crate::compress_subtree_to_parent_node(input, 0, &owner.key, 0, owner.flags, platform);
        } else {
            result[..32].copy_from_slice(&crate::hash_all_at_once(input, &owner.key, job.counter, owner.flags, platform).chaining_value());
        }
    }
    {
        let mut state = slot.state.lock().unwrap();
        assert!(matches!(*state, SlotState::Running));
        *state = SlotState::Ready(Completed { job, result });
    }
    // A different callback can now retire/reuse this slot. Use only the
    // cached owner and id beyond this point, never context or slot again.
    owner.consume(id);
    if let Some(barrier) = after {
        barrier.wait();
    }
    owner.source.finished(1);
    // Ownership ends after the source count, keeping all metadata alive.
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::task::Waker;

    fn producer_waker() -> Waker {
        struct Park(std::thread::Thread);
        impl std::task::Wake for Park {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        Waker::from(Arc::new(Park(std::thread::current())))
    }

    /// Run `future` on this thread, parking while its resources are pending.
    fn run<F: Future>(future: F) -> F::Output {
        let waker = producer_waker();
        let mut cx = Context::from_waker(&waker);
        let mut future = core::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
        let waker = producer_waker();
        future.poll(&mut Context::from_waker(&waker))
    }

    fn receiver(mode: Mode<'_>, buffers: usize, messages: usize) -> Receiver {
        crate::initialize_multithreaded(core::num::NonZeroUsize::new(3));
        Receiver::new(mode, buffers, messages)
    }

    /// Write `input` through `message` in pieces of up to `piece` bytes.
    fn feed(message: &mut Message, input: &[u8], piece: usize) {
        let mut at = 0;
        while at < input.len() {
            let mut room = run(message.input());
            let n = room.bytes_mut().len().min(piece).min(input.len() - at);
            room.bytes_mut()[..n].copy_from_slice(&input[at..at + n]);
            room.commit(n);
            at += n;
        }
    }

    fn expected(mode: Mode<'_>, input: &[u8]) -> [u8; 32] {
        match mode {
            Mode::Plain => *blake3::hash(input).as_bytes(),
            Mode::Keyed(key) => *blake3::keyed_hash(key, input).as_bytes(),
            Mode::DeriveKey(context) => blake3::derive_key(context, input),
            Mode::DeriveKeyWith(_) => unreachable!("tests name their context"),
        }
    }

    #[test]
    fn irregular_pieces_match_official_in_every_mode() {
        let key = [3u8; 32];
        for mode in [Mode::Plain, Mode::Keyed(&key), Mode::DeriveKey("receiver prototype test")] {
            let receiver = receiver(mode, 2, 1);
            for len in [0, 1, 63, 64, 65, 1024, INPUT_BYTES - 1, INPUT_BYTES, INPUT_BYTES + 1, 2 * INPUT_BYTES, 3 * INPUT_BYTES + 17] {
                let mut input = vec![0u8; len];
                crate::test::paint_test_input(&mut input);
                let mut message = run(receiver.message(Some(len as u64)));
                feed(&mut message, &input, 1448);
                let finished = run(message.end());
                assert_eq!(finished.hash().as_bytes(), &expected(mode, &input), "{len} bytes");
                receiver.drain();
                finished.retire();
            }
            receiver.close();
        }
    }

    /// Workers finish a message's second part before its first. The fold
    /// retires parts in message order and reuses the earlier slot after.
    #[test]
    fn parts_completed_out_of_order_fold_in_message_order() {
        let receiver = receiver(Mode::Plain, 2, 1);
        let mut input = vec![0u8; 2 * INPUT_BYTES + 100];
        crate::test::paint_test_input(&mut input);
        let mut message = run(receiver.message(None));
        let gate = Gate::new();
        let second_done = Arc::new(Barrier::new(2));
        message.hold_next(gate.clone());
        feed(&mut message, &input[..INPUT_BYTES], INPUT_BYTES);
        gate.started.wait();
        message.observe_next(second_done.clone());
        feed(&mut message, &input[INPUT_BYTES..2 * INPUT_BYTES], INPUT_BYTES);
        second_done.wait();
        // Both buffers are retained: the third part has no room yet.
        {
            let mut third = core::pin::pin!(message.input());
            assert!(poll_once(third.as_mut()).is_pending(), "retained completed parts count against the buffer budget");
            gate.release.wait();
            let mut room = run(third);
            room.bytes_mut()[..100].copy_from_slice(&input[2 * INPUT_BYTES..]);
            room.commit(100);
        }
        let finished = run(message.end());
        assert_eq!(finished.hash().as_bytes(), &expected(Mode::Plain, &input));
        receiver.drain();
        finished.retire();
        receiver.close();
    }

    /// A short message completes while an earlier message is still held.
    #[test]
    fn a_later_message_completes_while_an_earlier_one_is_unfinished() {
        let receiver = receiver(Mode::Plain, 3, 2);
        let mut long = vec![0u8; INPUT_BYTES];
        crate::test::paint_test_input(&mut long);
        let mut first = run(receiver.message(None));
        let gate = Gate::new();
        first.hold_next(gate.clone());
        feed(&mut first, &long, INPUT_BYTES);
        gate.started.wait();
        let mut second = run(receiver.message(Some(3)));
        feed(&mut second, b"abc", 3);
        let finished = run(second.end());
        assert_eq!(finished.hash().as_bytes(), &expected(Mode::Plain, b"abc"));
        finished.retire(); // Its own work retired with its result.
        gate.release.wait();
        let first = run(first.end());
        assert_eq!(first.hash().as_bytes(), &expected(Mode::Plain, &long));
        receiver.drain();
        first.retire();
        receiver.close();
    }

    /// Aborting a writer skips its queued work. After the drain retires that
    /// work, its slot and buffers return and the next message proceeds.
    #[test]
    fn aborting_a_writer_returns_its_resources_after_the_drain() {
        let receiver = receiver(Mode::Plain, 1, 1);
        let mut input = vec![0u8; INPUT_BYTES + 5];
        crate::test::paint_test_input(&mut input);
        let mut message = run(receiver.message(None));
        let gate = Gate::new();
        message.hold_next(gate.clone());
        feed(&mut message, &input[..INPUT_BYTES], INPUT_BYTES);
        gate.started.wait();
        let aborted = message.abort();
        gate.release.wait();
        receiver.drain();
        aborted.retire();
        let mut next = run(receiver.message(None));
        feed(&mut next, b"after abort", 11);
        let finished = run(next.end());
        assert_eq!(finished.hash().as_bytes(), &expected(Mode::Plain, b"after abort"));
        receiver.drain();
        finished.retire();
        receiver.close();
    }

    /// A prepared batch keeps its layout: one admission, one job, every
    /// digest written before its result resolves; message slots are shared
    /// with incremental messages.
    #[test]
    fn a_prepared_batch_completes_as_one_group_beside_incremental_messages() {
        let receiver = receiver(Mode::Plain, 2, 2);
        let mut batch = Batch::new(100, 37);
        for i in 0..batch.len() {
            batch.set(i, &[i as u8; 100]);
        }
        let mut places = vec![[0u8; OUT_LEN]; batch.len()];
        let mut message = run(receiver.message(None));
        let gate = Gate::new();
        message.hold_next(gate.clone());
        feed(&mut message, b"incremental", 11);
        let completion = message.end();
        gate.started.wait();
        // Admission first, then completion: every digest is written before
        // the result resolves.
        let admitted = run(receiver.submit_batch(&batch, &mut places));
        let batch_finished = run(admitted);
        for (i, place) in places.iter().enumerate() {
            assert_eq!(place, &expected(Mode::Plain, &[i as u8; 100]), "message {i}");
        }
        gate.release.wait();
        let finished = run(completion);
        assert_eq!(finished.hash().as_bytes(), &expected(Mode::Plain, b"incremental"));
        receiver.drain();
        batch_finished.retire();
        finished.retire();
        receiver.close();
    }

    /// An empty batch and a whole-block borrowed batch complete too.
    #[test]
    fn empty_and_borrowed_batches_complete() {
        let receiver = receiver(Mode::Plain, 1, 1);
        let empty = Batch::new(64, 0);
        let mut no_places: [[u8; OUT_LEN]; 0] = [];
        let empty_done = run(run(receiver.submit_batch(&empty, &mut no_places)));
        receiver.drain();
        empty_done.retire();
        let mut blocks = vec![Block::ZERO; 8];
        crate::test::paint_test_input(Block::bytes_mut(&mut blocks));
        let borrowed = Batch::from_blocks(&blocks, 128);
        let mut places = [[0u8; OUT_LEN]; 4];
        let borrowed_done = run(run(receiver.submit_batch(&borrowed, &mut places)));
        for (i, place) in places.iter().enumerate() {
            assert_eq!(place, &expected(Mode::Plain, &Block::bytes(&blocks)[i * 128..][..128]), "message {i}");
        }
        receiver.drain();
        borrowed_done.retire();
        receiver.close();
    }

    /// A producer that unwinds while its message has accepted work ends the
    /// process (crash-only): workers may still write that message's state.
    /// The child must die by SIGABRT, with the held job then released by
    /// nobody, which the dead process no longer needs.
    #[test]
    fn unwinding_with_accepted_work_aborts_the_process() {
        if std::env::var_os("RECEIVER_UNWIND").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "stream::prototype::receiver::tests::unwinding_with_accepted_work_aborts_the_process", "--nocapture"])
                .env("RECEIVER_UNWIND", "1")
                .env("NEXTEST", "1")
                .output()
                .unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                assert_eq!(out.status.signal(), Some(6), "SIGABRT: {}", String::from_utf8_lossy(&out.stderr));
            }
            return;
        }
        let receiver = receiver(Mode::Plain, 1, 1);
        let mut message = run(receiver.message(None));
        let gate = Gate::new();
        message.hold_next(gate.clone());
        feed(&mut message, &[7; INPUT_BYTES], INPUT_BYTES);
        gate.started.wait();
        // The producer panics with a part on a worker. Unwinding drops the
        // message's guard, which ends the process before the frame goes.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _message = message;
            panic!("the producer failed mid-message");
        }));
        std::process::exit(0);
    }

    /// Cancelling a pending completion stops the result wait. The message
    /// still finishes its accepted work; the slot returns after the drain.
    #[test]
    fn cancelling_a_completion_releases_the_message_slot_after_the_drain() {
        let receiver = receiver(Mode::Plain, 1, 1);
        let mut message = run(receiver.message(None));
        let gate = Gate::new();
        message.hold_next(gate.clone());
        feed(&mut message, b"held", 4);
        // A partial final piece is submitted by end().
        let mut completion = core::pin::pin!(message.end());
        gate.started.wait();
        assert!(poll_once(completion.as_mut()).is_pending());
        let aborted = completion.as_mut().cancel();
        gate.release.wait();
        receiver.drain();
        aborted.retire();
        let mut next = run(receiver.message(None));
        // A room taken and given back untouched leaves the budget whole.
        run(next.input()).release();
        feed(&mut next, b"next", 4);
        let finished = run(next.end());
        assert_eq!(finished.hash().as_bytes(), &expected(Mode::Plain, b"next"));
        receiver.drain();
        finished.retire();
        receiver.close();
    }
}
