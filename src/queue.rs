//! The queue: streams of inputs handed over by ownership, hashed behind
//! the caller, their results delivered to a handler.
//!
//! # How it runs
//!
//! One mechanism: one engine thread for every queue in the process takes
//! a queue's pending submissions a round at a time, hashes the round with
//! the pool, and delivers it in order.
//!
//! - **Submit.** The submission joins its queue's pending list, in order;
//!   a queue that had none is handed to the engine. Nothing else: no
//!   hashing, no wake of a worker.
//! - **A round.** The engine takes the queue's oldest pending submissions,
//!   up to ROUND_BYTES or ROUND_ITEMS (so the program refills the rest of
//!   what it keeps in flight while the round hashes), and cuts them into
//!   parts (`lanes::Part`): a message under TASK_MIN whole, a longer one
//!   or a piece into the subtrees `Hasher::update` would hash
//!   (`plan_subtrees`), a batch into ranges of its messages. One pool job
//!   hashes the round's parts (`lanes::hash_parts`: one-block messages side
//!   by side, ranges of about a thread's share over the pool's threads, a
//!   small round on the engine thread alone).
//! - **Deliver.** The engine replays `Hasher::update` with each message's
//!   or piece's results (for a message, then finalizes) and calls the
//!   handler, in submission order. A queue with more pending goes to the
//!   back of the engine's list, so queues take turns.
//!
//! The engine sleeps when no queue has pending submissions; the pool's
//! threads sleep between jobs. Nothing runs for work that may come
//! (AGENTS.md, "Serve real programs"). A round reuses its lists, so a warm
//! queue allocates nothing. The pool's workers never run user code; the
//! engine does, in the handler calls.

use crate::lanes::Part;
use crate::{CVWords, Hash, Mode, OUT_LEN};
use std::any::Any;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex};

/// The handler of a [`Queue::messages`]: one message per buffer, of any
/// length. See [`Queue`] for the rules every handler follows.
pub trait MessageHandler: Send + 'static {
    /// The program's buffer, handed over by [`Queue::submit`] and back here.
    type Buffer: AsRef<[u8]> + Send + 'static;
    /// `buffer` is hashed: `hash` is its digest in the queue's mode.
    fn hashed(&mut self, buffer: Self::Buffer, hash: Hash);
}

/// The handler of a [`Queue::pieces`]: one long message in pieces. See
/// [`Queue`] for the rules every handler follows.
pub trait PieceHandler: Send + 'static {
    /// The program's buffer, handed over by [`Queue::submit`] and back here.
    type Buffer: AsRef<[u8]> + Send + 'static;
    /// `buffer`'s bytes are part of the message now; the buffer is free.
    fn piece_done(&mut self, buffer: Self::Buffer);
    /// The message that [`Queue::finish`] ended has this digest.
    fn finished(&mut self, hash: Hash);
}

/// The handler of a [`Queue::fixed`]: messages of one length, back to
/// back in each buffer as [`hash_many`](crate::hash_many) takes them. See
/// [`Queue`] for the rules every handler follows.
pub trait FixedHandler: Send + 'static {
    /// The program's buffer of messages, handed over by [`Queue::submit`]
    /// and back here.
    type Buffer: AsRef<[u8]> + Send + 'static;
    /// The program's space for the digests, one per message.
    type Digests: AsMut<[[u8; OUT_LEN]]> + Send + 'static;
    /// `buffer` is hashed: `digests[i]` holds message i's digest.
    fn hashed(&mut self, buffer: Self::Buffer, digests: Self::Digests);
}

/// The three kinds of queue, as its second type parameter; a program
/// names them only to write down a queue's type.
pub mod shape {
    /// A [`Queue::messages`](crate::Queue::messages), the default.
    pub struct Messages;
    /// A [`Queue::pieces`](crate::Queue::pieces).
    pub struct Pieces;
    /// A [`Queue::fixed`](crate::Queue::fixed).
    pub struct Fixed;
}

/// A stream of inputs, hashed behind the program on every thread that
/// pays. Built for throughput (see [Which call to
/// use](crate#which-call-to-use)): the most bytes or messages hashed per
/// second. Each submission comes back after a handover, so
/// a single input takes longer than [`hash`](crate::hash) takes; for the
/// lowest latency per input, call the one-shot functions. The queue's
/// throughput is the hashing's when the program keeps enough in flight to
/// cover the round trip: a few buffers of 64 KiB and more, or many small
/// messages (or batches of them, [`Queue::fixed`]).
///
/// The program submits its buffers and moves on; each comes back, hashed,
/// through a call to the queue's handler, which the program implements.
/// No bytes are copied, and the buffers in flight are the ones the program
/// made: a program that cycles a fixed set (fill one, submit it, get it
/// back in a handler call, fill it again) hashes any amount in that much
/// memory. A queue takes one shape of input: messages of any length one
/// per buffer ([`Queue::messages`]), one long message in pieces
/// ([`Queue::pieces`]), or messages of one length back to back
/// ([`Queue::fixed`]). Every queue in a process shares one engine, so a
/// program makes a queue for each stream, on any thread. Buffers that
/// wait together are hashed together, over several threads where that
/// pays, so more buffers in flight keep more threads busy.
///
/// The rules every handler follows:
///
/// 1. **Short and never blocking**: every queue's results are delivered
///    from one thread, so a slow handler delays them all; heavy work
///    belongs on the program's own threads.
/// 2. **In order, one at a time**: a queue's handler is called in
///    submission order, one call at a time.
/// 3. **Submitting from inside is allowed**: a handler may submit to any
///    queue, its own included (refill and resubmit), and never waits.
/// 4. **A panic in a handler aborts the process.**
/// 5. **Dropping a queue cancels nothing**: every buffer submitted is
///    still hashed and comes back through the handler, which lives until
///    its last call.
///
/// A queue delivers on a thread of its own, so it needs a target with
/// threads: on one without them (wasm32-wasip1) the first submission
/// panics.
///
/// A program that allocates nothing once it runs makes its queue once and
/// keeps it, and carries what comes back to its own thread in a channel
/// made with room for every buffer it keeps in flight, such as the
/// standard library's `sync_channel`, whose ring is allocated when it is
/// made:
///
/// ```
/// use blake3_servil::{Hash, MessageHandler, Mode, Queue};
/// use std::sync::mpsc;
///
/// struct Digests(mpsc::SyncSender<(Vec<u8>, Hash)>);
/// impl MessageHandler for Digests {
///     type Buffer = Vec<u8>;
///     fn hashed(&mut self, buffer: Vec<u8>, hash: Hash) {
///         // Never waits: the channel has room for every buffer in flight.
///         self.0.try_send((buffer, hash)).unwrap();
///     }
/// }
///
/// # if cfg!(target_family = "wasm") { return; } // a queue needs threads
/// let in_flight = 2;
/// let (sender, results) = mpsc::sync_channel(in_flight);
/// let queue = Queue::messages(Mode::Hash, Digests(sender));
/// queue.submit(b"foo".to_vec());
/// queue.submit(b"bar".to_vec());
/// assert_eq!(results.recv().unwrap(), (b"foo".to_vec(), blake3_servil::hash(b"foo")));
/// assert_eq!(results.recv().unwrap(), (b"bar".to_vec(), blake3_servil::hash(b"bar")));
/// ```
pub struct Queue<H, S = shape::Messages> {
    /// An `Arc<Inner<H, item, S>>`, its item type fixed by the shape; the
    /// shape's methods know it and downcast.
    inner: Arc<dyn Any + Send + Sync>,
    shape: PhantomData<fn() -> (H, S)>,
}

/// A queue's state, shared by its handle and the engine.
struct Inner<H, I, S> {
    /// The submissions waiting for a round, in order, and whether the
    /// engine holds the queue (it has pending submissions or a round).
    pending: Mutex<(VecDeque<I>, bool)>,
    /// What the engine alone touches, during a round.
    work: Mutex<Work<H, I>>,
    key: CVWords,
    flags: u8,
    /// A [`Queue::fixed`]'s messages' length.
    message_len: usize,
    shape: PhantomData<fn() -> S>,
}

/// A round's lists, kept from round to round (no allocation once warm),
/// the handler, and a [`Queue::pieces`]'s message so far.
struct Work<H, I> {
    handling: H,
    round: Vec<I>,
    parts: Vec<Part>,
    results: Vec<[u8; crate::BLOCK_LEN]>,
    /// For each item of the round, its parts' range in `parts`.
    spans: Vec<(usize, usize)>,
    plan: crate::PlanState,
    hasher: crate::HasherCore,
}

/// A round takes submissions until they hold this many bytes, or ROUND_ITEMS
/// of them: about the shortest input the pool pays for, and well under what
/// a program keeps in flight to cover the round trip, so the program
/// refills while the round hashes.
const ROUND_BYTES: usize = crate::lanes::MIN_SPLIT_LEN;
const ROUND_ITEMS: usize = 256;

/// The shortest message or piece hashed as subtrees of its own: a shorter
/// message is one part; a shorter piece joins the message at delivery.
const TASK_MIN: usize = crate::SME2_SIZED_LEN;

/// A queue of pieces' submissions.
enum PieceItem<B> {
    Piece(B),
    Finish,
}

/// What the engine runs for a queue: one round, and whether it has more
/// pending.
trait Round: Send + Sync {
    fn round(&self) -> bool;
}

/// The engine's queues with pending submissions, in turn, whether it
/// sleeps, and the room its list keeps (one for every queue alive: no
/// allocation in flight).
struct Engine {
    queues: Mutex<(VecDeque<Arc<dyn Round>>, bool, usize)>,
    wake: Condvar,
}

static ENGINE: Engine = Engine { queues: Mutex::new((VecDeque::new(), false, 0)), wake: Condvar::new() };

impl Engine {
    /// Hand `queue`, which has just gained pending submissions, to the
    /// engine; start the engine with the first.
    fn hold(&self, queue: Arc<dyn Round>) {
        start_delivery();
        let mut queues = crate::lanes::lock_polling(&self.queues);
        queues.0.push_back(queue);
        if queues.1 {
            self.wake.notify_one();
        }
    }

    /// Room for one more queue in the engine's list.
    fn make_room(&self) {
        let mut queues = crate::lanes::lock_polling(&self.queues);
        queues.2 += 1;
        let (room, held) = (queues.2, queues.0.len());
        queues.0.reserve(room.saturating_sub(held));
    }

    /// Run rounds, one queue at a time in turn, until none has pending
    /// submissions; then sleep. A panic in a handler aborts the process
    /// (handler rule 4); so does one in the hashing, a bug.
    fn run(&self) {
        loop {
            let queue = {
                let mut queues = crate::lanes::lock_polling(&self.queues);
                loop {
                    if let Some(queue) = queues.0.pop_front() {
                        break queue;
                    }
                    queues.1 = true;
                    queues = self.wake.wait(queues).unwrap();
                    queues.1 = false;
                }
            };
            let more = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.round())).unwrap_or_else(|_| std::process::abort());
            if more {
                crate::lanes::lock_polling(&self.queues).0.push_back(queue);
            }
        }
    }
}

/// Start the engine thread, once per process: from
/// `initialize_multithreaded`, or else at a queue's first submission.
pub(crate) fn start_delivery() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        crate::lanes::prepare(&ENGINE.queues, &ENGINE.wake);
        // Return once the thread runs, as the pool's threads do: its start
        // allocates too (std's stack-overflow handler), which a warm queue
        // must not meet later.
        static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        std::thread::Builder::new()
            .name("blake3-servil-queue".into())
            .spawn(|| {
                RUNNING.store(true, std::sync::atomic::Ordering::SeqCst);
                ENGINE.run()
            })
            .expect("the queue's engine thread starts");
        while !RUNNING.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::yield_now();
        }
    });
}

impl<H: Send + 'static, S: 'static> Queue<H, S> {
    fn new<I: Send + 'static>(mode: Mode, handler: H, message_len: usize) -> Self
    where
        Inner<H, I, S>: Round,
    {
        let (key, flags) = mode.key_and_flags();
        ENGINE.make_room();
        let work = Work { handling: handler, round: Vec::with_capacity(ROUND_ITEMS), parts: Vec::new(), results: Vec::new(), spans: Vec::with_capacity(ROUND_ITEMS), plan: Default::default(), hasher: crate::HasherCore::new_internal(&key, flags) };
        let inner: Inner<H, I, S> = Inner { pending: Mutex::new((VecDeque::new(), false)), work: Mutex::new(work), key, flags, message_len, shape: PhantomData };
        crate::lanes::prepare(&inner.pending, &ENGINE.wake);
        Queue { inner: Arc::new(inner), shape: PhantomData }
    }

    fn inner<I: Send + 'static>(&self) -> &Inner<H, I, S> {
        self.inner.downcast_ref().unwrap_or_else(|| unreachable!("a queue's state has its shape's type"))
    }

    /// Put `item` in flight: on its queue's pending list, the queue handed
    /// to the engine if it had none.
    fn push<I: Send + 'static>(&self, item: I)
    where
        Inner<H, I, S>: Round,
    {
        let inner = self.inner::<I>();
        let mut pending = crate::lanes::lock_polling(&inner.pending);
        pending.0.push_back(item);
        if !pending.1 {
            pending.1 = true;
            drop(pending);
            let held: Arc<Inner<H, I, S>> = self.inner.clone().downcast().unwrap_or_else(|_| unreachable!("a queue's state has its shape's type"));
            ENGINE.hold(held);
        }
    }
}

impl<H, I, S> Inner<H, I, S> {
    /// Move the next round's submissions into `round` (`len` gives each
    /// one's bytes); returns whether more stay pending. With none left,
    /// the queue goes back to the submitters (the next one hands it over).
    fn take(&self, round: &mut Vec<I>, len: impl Fn(&I) -> usize) -> bool {
        let mut pending = crate::lanes::lock_polling(&self.pending);
        let mut bytes = 0;
        while bytes < ROUND_BYTES && round.len() < ROUND_ITEMS {
            let Some(item) = pending.0.pop_front() else { break };
            bytes += len(&item);
            round.push(item);
        }
        let more = !pending.0.is_empty();
        if !more {
            pending.1 = false;
        }
        more
    }
}

impl<H: MessageHandler> Round for Inner<H, H::Buffer, shape::Messages> {
    fn round(&self) -> bool {
        let mut work = crate::lanes::lock_polling(&self.work);
        let Work { handling, round, parts, results, spans, .. } = &mut *work;
        let more = self.take(round, |b| b.as_ref().len());
        // A short message is one part, its digest; a longer one the
        // subtrees Hasher::update hashes, replayed below.
        parts.clear();
        spans.clear();
        make_room(parts, results, round.iter().map(|b| b.as_ref().len()));
        for buffer in round.iter() {
            let bytes = buffer.as_ref();
            let start = parts.len();
            if bytes.len() < TASK_MIN {
                parts.push(Part { root: true, ..Part::of(bytes, 0) });
            } else {
                crate::plan_subtrees(&mut Default::default(), bytes, parts);
            }
            spans.push((start, parts.len()));
        }
        hash_round(parts, results, &self.key, self.flags);
        for (buffer, &(start, end)) in round.drain(..).zip(spans.iter()) {
            let hash = if end == start + 1 && parts[start].root {
                Hash(results[start][..OUT_LEN].try_into().unwrap())
            } else {
                let mut hasher = crate::HasherCore::new_internal(&self.key, self.flags);
                hasher.update_with_results(buffer.as_ref(), &mut results[start..end].iter());
                hasher.final_output().root_hash()
            };
            handling.hashed(buffer, hash);
        }
        more
    }
}

impl<H: PieceHandler> Round for Inner<H, PieceItem<H::Buffer>, shape::Pieces> {
    fn round(&self) -> bool {
        let mut work = crate::lanes::lock_polling(&self.work);
        let Work { handling, round, parts, results, spans, plan, hasher } = &mut *work;
        let more = self.take(round, |item| match item {
            PieceItem::Piece(piece) => piece.as_ref().len(),
            PieceItem::Finish => 0,
        });
        parts.clear();
        spans.clear();
        make_room(parts, results, round.iter().map(|item| match item {
            PieceItem::Piece(piece) => piece.as_ref().len(),
            PieceItem::Finish => 0,
        }));
        for item in round.iter() {
            let start = parts.len();
            match item {
                PieceItem::Piece(piece) => {
                    // The plan follows every piece; a short one's subtrees
                    // are its replay's (it joins the message there).
                    crate::plan_subtrees(plan, piece.as_ref(), parts);
                    if piece.as_ref().len() < TASK_MIN {
                        parts.truncate(start);
                    }
                }
                PieceItem::Finish => *plan = Default::default(),
            }
            spans.push((start, parts.len()));
        }
        hash_round(parts, results, &self.key, self.flags);
        for (item, &(start, end)) in round.drain(..).zip(spans.iter()) {
            match item {
                PieceItem::Piece(piece) => {
                    if start == end {
                        hasher.update(piece.as_ref());
                    } else {
                        hasher.update_with_results(piece.as_ref(), &mut results[start..end].iter());
                    }
                    handling.piece_done(piece);
                }
                PieceItem::Finish => {
                    let hash = hasher.final_output().root_hash();
                    hasher.reset();
                    handling.finished(hash);
                }
            }
        }
        more
    }
}

impl<H: FixedHandler> Round for Inner<H, (H::Buffer, H::Digests), shape::Fixed> {
    fn round(&self) -> bool {
        let mut work = crate::lanes::lock_polling(&self.work);
        let Work { handling, round, parts, results, .. } = &mut *work;
        let more = self.take(round, |(buffer, _)| buffer.as_ref().len());
        make_room(parts, results, round.iter().map(|(buffer, _)| buffer.as_ref().len()));
        // Each batch in ranges of about a task's bytes, their digests
        // straight into the program's space.
        let slot = crate::many::slot_len(self.message_len);
        let per_part = (crate::lanes::TASK_LEN / slot).max(1);
        parts.clear();
        for (buffer, digests) in round.iter_mut() {
            let out = digests.as_mut().as_mut_ptr();
            for (index, range) in buffer.as_ref().chunks(per_part * slot).enumerate() {
                // Sound: `per_part` digests per range, within the space.
                parts.push(Part { batch: Some(self.message_len), out: unsafe { out.add(index * per_part) } as *mut u8, ..Part::of(range, 0) });
            }
        }
        crate::lanes::hash_parts(parts, &self.key, self.flags);
        for (buffer, digests) in round.drain(..) {
            handling.hashed(buffer, digests);
        }
        more
    }
}

/// Room in `parts` and `results` for the most parts submissions of these
/// lengths can cut into, whatever their alignment (a part of up to
/// TASK_LEN cuts into at most 16 subtrees), so a program that cycles the
/// same lengths meets every list's size during warm-up.
fn make_room(parts: &mut Vec<Part>, results: &mut Vec<[u8; crate::BLOCK_LEN]>, lens: impl Iterator<Item = usize>) {
    let most: usize = lens.map(|len| 16 * (len.div_ceil(crate::lanes::TASK_LEN) + 1)).sum();
    parts.reserve(most.saturating_sub(parts.len()));
    results.reserve(most.saturating_sub(results.len()));
}

/// Hash a round's `parts`, each part's result to its place in `results`.
fn hash_round(parts: &mut [Part], results: &mut Vec<[u8; crate::BLOCK_LEN]>, key: &CVWords, flags: u8) {
    results.clear();
    results.resize(parts.len(), [0; crate::BLOCK_LEN]);
    for (part, result) in parts.iter_mut().zip(results.iter_mut()) {
        part.out = result.as_mut_ptr();
    }
    crate::lanes::hash_parts(parts, key, flags);
}

impl<H: MessageHandler> Queue<H, shape::Messages> {
    /// A queue of messages of any length, one per buffer, each digest in
    /// `mode` delivered to `handler.hashed` with its buffer.
    pub fn messages(mode: Mode, handler: H) -> Self {
        Self::new::<H::Buffer>(mode, handler, 0)
    }

    /// Hash `buffer`'s bytes as one message; returns at once.
    pub fn submit(&self, buffer: H::Buffer) {
        self.push(buffer);
    }
}

impl<H: PieceHandler> Queue<H, shape::Pieces> {
    /// A queue of one long message in pieces: each piece comes back to
    /// `handler.piece_done`, and after [`finish`](Self::finish) the
    /// message's digest in `mode` to `handler.finished`. The next piece
    /// after `finish` starts a new message.
    pub fn pieces(mode: Mode, handler: H) -> Self {
        Self::new::<PieceItem<H::Buffer>>(mode, handler, 0)
    }

    /// Append `piece`'s bytes to the message; returns at once.
    pub fn submit(&self, piece: H::Buffer) {
        self.push(PieceItem::Piece(piece));
    }

    /// End the message: its digest goes to `handler.finished` after every
    /// piece has come back. Returns at once.
    pub fn finish(&self) {
        self.push(PieceItem::<H::Buffer>::Finish);
    }
}

impl<H: FixedHandler> Queue<H, shape::Fixed> {
    /// A queue of messages of `message_len` bytes, many per buffer, laid
    /// out as [`hash_many`](crate::hash_many) takes them; their digests in
    /// `mode` come back to `handler.hashed` with the buffer, in the space
    /// the program submitted beside it.
    pub fn fixed(message_len: usize, mode: Mode, handler: H) -> Self {
        Self::new::<(H::Buffer, H::Digests)>(mode, handler, message_len)
    }

    /// Hash the messages in `buffer` into `digests`, one per message: the
    /// buffer holds exactly `digests.len()` messages under
    /// [`hash_many`](crate::hash_many)'s layout (checked here). Returns at
    /// once.
    pub fn submit(&self, buffer: H::Buffer, mut digests: H::Digests) {
        let message_len = self.inner::<(H::Buffer, H::Digests)>().message_len;
        let slot = crate::many::slot_len(message_len);
        assert_eq!(Some(buffer.as_ref().len()), slot.checked_mul(digests.as_mut().len()), "the buffer holds one slot of whole blocks per digest");
        self.push((buffer, digests));
    }
}

#[cfg(test)]
mod test {

    /// A few inputs through every shape of queue against
    /// hash(): small enough for Miri (the CI's smoketest runs it), which
    /// checks the hand-over of each round's items to the engine thread
    /// and back for data races.
    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "a queue needs threads, which this target lacks")]
    fn test_miri_queue_round_trips() {
        use crate::{Hash, MessageHandler, Mode, PieceHandler, Queue};
        use std::sync::mpsc;
        struct Back(mpsc::Sender<(std::vec::Vec<u8>, Hash)>);
        impl MessageHandler for Back {
            type Buffer = std::vec::Vec<u8>;
            fn hashed(&mut self, buffer: std::vec::Vec<u8>, hash: Hash) {
                self.0.send((buffer, hash)).unwrap();
            }
        }
        struct Fixed(mpsc::Sender<(std::vec::Vec<u8>, std::vec::Vec<[u8; 32]>)>);
        impl crate::FixedHandler for Fixed {
            type Buffer = std::vec::Vec<u8>;
            type Digests = std::vec::Vec<[u8; 32]>;
            fn hashed(&mut self, buffer: std::vec::Vec<u8>, digests: std::vec::Vec<[u8; 32]>) {
                self.0.send((buffer, digests)).unwrap();
            }
        }
        struct Pieces(mpsc::Sender<Option<Hash>>);
        impl PieceHandler for Pieces {
            type Buffer = std::vec::Vec<u8>;
            fn piece_done(&mut self, _: std::vec::Vec<u8>) {
                self.0.send(None).unwrap();
            }
            fn finished(&mut self, hash: Hash) {
                self.0.send(Some(hash)).unwrap();
            }
        }
        {
            let (sender, back) = mpsc::channel();
            let queue = Queue::messages(Mode::Hash, Back(sender));
            // 40 KiB: whole subtrees as tasks of the pool's threads.
            let inputs: std::vec::Vec<std::vec::Vec<u8>> = (0..7).map(|i| std::vec![i as u8; [0, 1, 64, 65, 1024, 1500, 40 << 10][i]]).collect();
            for input in &inputs {
                queue.submit(input.clone());
            }
            for input in &inputs {
                let (buffer, hash) = back.recv().unwrap();
                assert_eq!(&buffer, input, "in order");
                assert_eq!(hash, crate::hash(input));
            }
            let (sender, back) = mpsc::channel();
            let queue = Queue::pieces(Mode::Hash, Pieces(sender));
            // Pieces of 700 B (hashed at delivery) and of 20 KiB (tasks).
            for (len, piece) in [(3000usize, 700usize), (60 << 10, 20 << 10)] {
                let message = std::vec![7u8; len];
                for piece in message.chunks(piece) {
                    queue.submit(piece.to_vec());
                }
                queue.finish();
                let mut done = 0;
                let hash = loop {
                    match back.recv().unwrap() {
                        None => done += 1,
                        Some(hash) => break hash,
                    }
                };
                assert_eq!(done, len.div_ceil(piece));
                assert_eq!(hash, crate::hash(&message));
            }
            let (sender, back) = mpsc::channel();
            let queue = Queue::fixed(64, Mode::Hash, Fixed(sender));
            let batch: std::vec::Vec<u8> = (0..20 * 64).map(|i| (i % 251) as u8).collect();
            for _ in 0..3 {
                queue.submit(batch.clone(), std::vec![[0u8; 32]; 20]);
            }
            for _ in 0..3 {
                let (_, digests) = back.recv().unwrap();
                for (i, digest) in digests.iter().enumerate() {
                    assert_eq!(digest, crate::hash(&batch[i * 64..][..64]).as_bytes());
                }
            }
        }
    }
    use crate::platform::Platform;
    use crate::{BLOCK_LEN, CHUNK_LEN};
    use std::sync::atomic::AtomicUsize;

    /// Pieces planned into whole subtrees, hashed as tasks, and replayed
    /// give Hasher::update's digest, for runs of pieces of many lengths
    /// (whole chunks, partial chunks, single chunks, a chunk and a byte),
    /// starting from states Hasher::update left (empty, mid-chunk, a full
    /// chunk), in batches of every size.
    #[test]
    fn planned_pieces_match_update() {
        let lens: [usize; 14] = [CHUNK_LEN + 1, 1, 1000, CHUNK_LEN, 0, 2 * CHUNK_LEN, 3 * CHUNK_LEN, 4096 + 17, 16 * CHUNK_LEN, 64 * CHUNK_LEN, 65 * CHUNK_LEN, 100_000, 128 * CHUNK_LEN, 3 * 65536];
        let input: Vec<u8> = (0..4 << 20).map(|i: u32| (i.wrapping_mul(0x9E37_79B1) >> 24) as u8).collect();
        let key = [9u32; 8];
        for start in [0, 500, CHUNK_LEN, 3 * CHUNK_LEN] {
            for (step, batch) in [(1usize, 1usize), (3, 2), (5, 3), (7, 4), (11, 6)] {
                let pieces: Vec<usize> = (0..12).map(|i| lens[(i * step + start) % lens.len()]).collect();
                let mut want = crate::HasherCore::new_internal(&key, crate::KEYED_HASH);
                let mut got = want.clone();
                want.update(&input[..start]);
                got.update(&input[..start]);
                let mut plan = crate::PlanState::default();
                crate::plan_subtrees(&mut plan, &input[..start], &mut Vec::new());
                let mut offset = start;
                for group in pieces.chunks(batch) {
                    let slices: Vec<&[u8]> = group.iter().map(|&len| {
                        let piece = &input[offset..offset + len];
                        offset += len;
                        piece
                    }).collect();
                    let mut tasks = Vec::new();
                    for piece in &slices {
                        crate::plan_subtrees(&mut plan, piece, &mut tasks);
                    }
                    let mut results = vec![[0u8; BLOCK_LEN]; tasks.len()];
                    for (task, result) in tasks.iter_mut().zip(&mut results) {
                        task.out = result.as_mut_ptr();
                    }
                    crate::lanes::hash_parts(&tasks, &key, crate::KEYED_HASH);
                    let mut replay = results.iter();
                    for piece in &slices {
                        want.update(piece);
                        got.update_with_results(piece, &mut replay);
                    }
                    assert!(replay.as_slice().is_empty(), "every result replayed");
                    assert_eq!(got.final_output().root_hash(), want.final_output().root_hash(), "start {start}, pieces {pieces:?}, batches of {batch}, after {offset} bytes");
                }
            }
        }
    }
}
