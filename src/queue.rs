//! The queue: streams of inputs handed over by ownership, hashed behind
//! the caller, their results delivered to a handler.
//!
//! # How it runs
//!
//! One mechanism: `submit` turns a submission into tasks on the caller's
//! thread and returns; the pool's threads hash the tasks; one delivery
//! thread hands the results back in order.
//!
//! - **Submit.** The submission takes a slot (slots live in blocks that
//!   never move and are recycled: no allocation after warm-up) and is
//!   linked at the tail of the queue's chain of entries, in submission
//!   order. Its tasks are the whole subtrees `Hasher::update` would hash
//!   in it, in parts of at most `lanes::TASK_LEN` (`plan_subtrees`: a
//!   piece's chunk counters follow from its offset in the message; a
//!   message is a stream of one piece). Messages shorter than `TASK_MIN`,
//!   and `Queue::fixed` batches under `BATCH_TASK_MIN`, go several to a
//!   task instead (`lanes::Member`), handed over when it is full or when
//!   the delivery thread waits on it. Tasks go onto the pool's task list
//!   (`lanes::TASKS`), which wakes a thread per task in flight.
//! - **Hash.** The SME2 thread and the workers pop tasks; each writes its
//!   result into its entry and counts it down. The SME2 thread hashes
//!   subtrees on SME2 and gathered tasks on NEON; the workers on NEON.
//! - **Deliver.** The delivery thread follows each queue's chain from the
//!   last entry it delivered, takes each entry once its count is zero,
//!   replays `Hasher::update` with its results (and for a message
//!   finalizes), and calls the handler. Entries without tasks it hashes
//!   itself: pieces shorter than `TASK_MIN` (their bytes join the message
//!   in order), and everything in a pool with no thread to take tasks
//!   (one CPU to the process, no SME2), which the delivery thread hashes
//!   alone. The submitters and the delivery thread
//!   share no lock on these paths: delivered slots go back through
//!   `returned`, and the queue's hold passes by a store-then-recheck
//!   handshake (`Inner::activate`, the end of `deliver_with`). The
//!   delivery thread polls while any entry is in flight, and sleeps when
//!   none is; the pool's threads sleep as soon as no task waits. Nothing
//!   runs for work that may come (AGENTS.md, "Serve real programs").
//!
//! The pool's workers never run user code; the delivery thread does, in
//! the handler calls.

use crate::lanes::{OwnLine, TASKS, Task};
use crate::{CVWords, Hash, Mode, OUT_LEN};
use std::any::Any;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
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

/// One queue's state, shared by its handle and the delivery thread.
///
/// The submitters and the delivery thread share no lock on their common
/// paths (each lock's line would move between their cores on every
/// submission: about 120 ns of a 64-byte message's 200 on the VM). Entries
/// in flight form a chain in submission order: a submitter links each new
/// slot after the last one (`State::tail`); the delivery thread follows the
/// links from the last slot it delivered (`head`, kept until the next is
/// delivered, so a submitter always has a slot to link after). Delivered
/// slots go back through `returned`, a batch per delivery round, and the
/// submitters take the whole batch when their own free slots run out.
struct Inner<H, I, S> {
    /// The submitters' side. It, `returned`, and the delivery thread's own
    /// fields each on lines of their own: the submitters take the one lock
    /// and the delivery thread the other for every message.
    state: OwnLine<Mutex<State<I>>>,
    /// Slots delivered since the submitters last took them.
    returned: OwnLine<Mutex<Slots<I>>>,
    /// What the delivery thread alone touches.
    handling: OwnLine<Mutex<Handling<H>>>,
    /// The delivery thread's own: the slot it delivered last.
    head: AtomicPtr<Slot<I>>,
    /// The delivery thread's own: the unfinished count of the entry its
    /// last look stopped at, which it polls alone.
    waiting: AtomicPtr<AtomicUsize>,
    /// The delivery thread's own: how many rounds it has polled `waiting`.
    polls: AtomicUsize,
    /// Whether the delivery thread holds this queue (entries in flight).
    active: OwnLine<AtomicBool>,
    key: CVWords,
    flags: u8,
    /// Whether the pool's threads hash this queue's entries (as tasks):
    /// when the pool has a thread to take them; otherwise the delivery
    /// thread hashes each entry.
    tasks: bool,
    /// The fixed-length queue's message length.
    message_len: usize,
    shape: PhantomData<fn() -> S>,
}

/*
 * A queue's storage is allocated once and recycled, as io_uring's rings
 * are: submissions in flight occupy slots, which live in blocks that never
 * move (so tasks and the chain may point into them) and are handed back
 * after delivery; a block is added only when more submissions are in
 * flight than ever before, and the lists below keep their capacity. So a
 * program cycling a fixed set of buffers makes the queue allocate nothing
 * once as many of its submissions have waited undelivered at once as ever
 * will: at most every buffer and one more. How many wait depends on how
 * far the delivery thread trails the program, so the last blocks may come
 * well after the first round (bench-hashes' producer met one in about one
 * run of ten).
 */
struct State<I> {
    /// The slots' blocks, each SLOT_BLOCK slots (from Box::into_raw; freed
    /// on drop).
    blocks: Vec<*mut [Slot<I>; SLOT_BLOCK]>,
    /// The slots free (room for every slot, as `returned` has).
    free: Slots<I>,
    /// The last slot in the chain.
    tail: *mut Slot<I>,
    /// Room to plan a submission's tasks in.
    tasks: Vec<Task>,
    /// The most tasks a submission has had: every slot's results make room
    /// for as many when handed out, so no slot grows later than another.
    most: usize,
    /// Where planning stands: past every piece submitted (the hasher
    /// stands past the delivered ones).
    plan: crate::PlanState,
    /// Short messages gathered into one task, not yet handed to the pool.
    open: Option<Task>,    /// Whether this queue hands tasks to the pool, and the room it made in
    /// the pool's task list for them (lanes::Tasks::make_room).
    tasks_room: Option<usize>,
}

/// Slots, by address.
struct Slots<I>(Vec<*mut Slot<I>>);

// Sound: the blocks and slots are the queue's own, reached through its
// locks or, for a slot in flight, by the one thread that holds it (below).
unsafe impl<I: Send> Send for State<I> {}
unsafe impl<I: Send> Send for Slots<I> {}

const SLOT_BLOCK: usize = 16;

/// A submission in flight: its item, its tasks' results (none: hashed at
/// delivery), how many of its tasks are unfinished, and the next entry.
/// The results keep their capacity from one submission to the next.
struct Slot<I> {
    item: Option<I>,
    results: Vec<[u8; crate::BLOCK_LEN]>,
    left: AtomicUsize,
    /// Whether `results` holds the message's digest (a short message
    /// hashed with others), not its subtrees' results.
    digest: bool,
    /// The entry submitted after this one, once linked.
    next: AtomicPtr<Slot<I>>,
}

impl<I> State<I> {
    /// A new block's slots, added to `free`, and room for every slot in
    /// `free` and `returned`.
    fn add_block(&mut self, returned: &mut Slots<I>) {
        let most = self.most.max(1);
        let block = Box::into_raw(Box::new(std::array::from_fn(|_| Slot { item: None, results: Vec::with_capacity(most), left: AtomicUsize::new(0), digest: false, next: AtomicPtr::new(core::ptr::null_mut()) })));
        self.blocks.push(block);
        if let Some(room) = &mut self.tasks_room {
            TASKS.make_room(SLOT_BLOCK * most);
            *room += SLOT_BLOCK * most;
        }
        let total = self.blocks.len() * SLOT_BLOCK;
        self.free.0.reserve(total - self.free.0.len());
        returned.0.reserve(total - returned.0.len());
        // Sound: a fresh block, SLOT_BLOCK slots.
        self.free.0.extend((0..SLOT_BLOCK).rev().map(|k| unsafe { (block as *mut Slot<I>).add(k) }));
    }

    /// A free slot: from the free list, else every returned one, else a new
    /// block's. It is the caller's until linked.
    fn take_slot(&mut self, returned: &Mutex<Slots<I>>) -> *mut Slot<I> {
        if self.free.0.is_empty() {
            let mut returned = crate::lanes::lock_polling(returned);
            // Both lists keep room for every slot.
            std::mem::swap(&mut self.free, &mut *returned);
            // The longest returned first: its lines have had the longest to
            // leave the delivery thread's cache.
            self.free.0.reverse();
            if self.free.0.is_empty() {
                self.add_block(&mut returned);
            }
        }
        let slot = self.free.0.pop().unwrap();
        // Sound: a free slot is untouched by any other thread.
        unsafe { &(*slot).next }.store(core::ptr::null_mut(), Ordering::Relaxed);
        slot
    }

    /// The open batch of short messages, to hand to the pool (after the
    /// state's lock is released) when it is full or (with `waited`) when
    /// the delivery thread waits on it; otherwise it goes on filling.
    fn close_open(&mut self, waited: bool) -> Option<Task> {
        // Small batches fill a task up to a task's bytes (short messages
        // fill all 64 places: 4 KiB messages measured 10-30% slower at 16).
        let full = self.open.as_ref().is_some_and(|open| open.members == crate::lanes::MEMBERS || (open.batch.is_some() && open.len >= crate::lanes::TASK_LEN));
        if full || waited { self.open.take() } else { None }
    }
}

impl<I> State<I> {
    /// Link `slot`, filled, after the chain's last slot. A raw pointer: once
    /// linked, the slot is the delivery thread's to read, and a `&mut`
    /// argument would claim it until this call returns.
    fn link(&mut self, slot: *mut Slot<I>) {
        // Sound: the tail is linked (or the first, delivered slot), and the
        // delivery thread hands it back only after this link. Only its
        // `next` is touched, through the field alone: the delivery thread
        // may be taking the slot's item meanwhile, and a reference to the
        // whole slot would race with that (Miri: undefined behaviour).
        unsafe { &(*self.tail).next }.store(slot, Ordering::SeqCst);
        self.tail = slot;
    }
}

impl<I> Drop for State<I> {
    fn drop(&mut self) {
        DELIVERY.give_room();
        if let Some(room) = self.tasks_room {
            TASKS.give_room(room);
        }
        for &block in &self.blocks {
            // Sound: from Box::into_raw, dropped once, with nothing in flight
            // (the delivery thread holds the queue until its last delivery).
            drop(unsafe { Box::from_raw(block) });
        }
    }
}

struct Handling<H> {
    handler: H,
    hasher: crate::HasherCore,
}

/// A queue of pieces' submissions.
enum PieceItem<B> {
    Piece(B),
    Finish,
}

/// The shortest message or piece hashed as tasks of its own: shorter
/// messages go several to a task (`lanes::MEMBERS`), shorter pieces are
/// hashed at delivery (a piece's bytes join the message in order).
const TASK_MIN: usize = crate::SME2_SIZED_LEN;

/// The shortest batch of fixed-length messages hashed as tasks of its own
/// (a task's bytes): shorter ones go several to a task (on the Mac a task
/// of its own cost a batch of 16 64-byte messages 68 ns per message, twice
/// hashing it at delivery, and one of 64 messages shared 36-50).
const BATCH_TASK_MIN: usize = crate::lanes::TASK_LEN;

impl<H: Send + 'static, S: 'static> Queue<H, S> {
    fn new<I: Send + 'static>(mode: Mode, handler: H, message_len: usize) -> Self
    where
        Inner<H, I, S>: Deliver,
    {
        let (key, flags) = mode.key_and_flags();
        let tasks = crate::lanes::takes_tasks();
        let mut returned = Slots(Vec::new());
        let mut state = State { blocks: Vec::new(), free: Slots(Vec::new()), tail: core::ptr::null_mut(), tasks: Vec::new(), most: 0, plan: Default::default(), open: None, tasks_room: tasks.then_some(0) };
        state.add_block(&mut returned);
        DELIVERY.make_room();
        // The chain starts at a slot delivered already.
        let first = state.free.0.pop().unwrap();
        state.tail = first;
        let inner = Inner::<H, I, S> {
            state: OwnLine(Mutex::new(state)),
            returned: OwnLine(Mutex::new(returned)),
            handling: OwnLine(Mutex::new(Handling { handler, hasher: crate::HasherCore::new_internal(&key, flags) })),
            head: AtomicPtr::new(first),
            waiting: AtomicPtr::new(core::ptr::null_mut()),
            polls: AtomicUsize::new(0),
            active: OwnLine(AtomicBool::new(false)),
            key,
            flags,
            tasks,
            message_len,
            shape: PhantomData,
        };
        Queue { inner: Arc::new(inner), shape: PhantomData }
    }

    /// The queue's state as its concrete type (a type comparison), and
    /// its owner, which the delivery thread holds while entries are in
    /// flight (cloned only then: a clone per submission would bounce the
    /// count's line between the submitter and the delivery thread).
    fn inner<I: Send + 'static>(&self) -> (&Inner<H, I, S>, &Arc<dyn Any + Send + Sync>) {
        (self.inner.downcast_ref().unwrap_or_else(|| unreachable!("a queue's state has its shape's type")), &self.inner)
    }
}

impl<H: Send + 'static, I: Send + 'static, S: 'static> Inner<H, I, S>
where
    Inner<H, I, S>: Deliver,
{
    /// Put `item` in flight, with tasks for `plan` to cut from it (it
    /// appends their inputs and chunk counters to `tasks`, and returns
    /// whether they are the entry's work or the entry is hashed at delivery).
    fn submit(&self, owner: &Arc<dyn Any + Send + Sync>, item: I, plan: impl FnOnce(&mut I, &mut crate::PlanState, &mut Vec<Task>) -> bool) {
        let mut guard = crate::lanes::lock_polling(&self.state);
        let state = &mut *guard;
        let slot = state.take_slot(&self.returned);
        // Sound: a free slot is this thread's until linked.
        let slot = unsafe { &mut *slot };
        let item = slot.item.insert(item);
        slot.results.clear();
        slot.digest = false;
        let mut tasks = std::mem::take(&mut state.tasks);
        let planned = plan(item, &mut state.plan, &mut tasks);
        if tasks.len() > state.most {
            if let Some(room) = &mut state.tasks_room {
                // Every slot may hold a submission of this many tasks.
                let more = state.blocks.len() * SLOT_BLOCK * (tasks.len() - state.most.max(1).min(tasks.len()));
                TASKS.make_room(more);
                *room += more;
            }
            state.most = tasks.len();
            // Every free slot makes room now, the ones in flight when next
            // handed out (below): growth follows the program's submissions,
            // not which slot timing hands out.
            let returned = crate::lanes::lock_polling(&self.returned);
            for &free in state.free.0.iter().chain(&returned.0) {
                // Sound: a free slot is untouched by any task.
                unsafe { &mut *free }.results.reserve(state.most);
            }
        }
        slot.results.reserve(state.most);
        if planned && !tasks.is_empty() {
            slot.results.resize(tasks.len(), [0; crate::BLOCK_LEN]);
            slot.left.store(tasks.len(), Ordering::Relaxed);
            let (out, left) = (slot.results.as_mut_ptr(), &slot.left as *const AtomicUsize);
            TASKS.push(tasks.drain(..).enumerate().map(|(index, task)| Task {
                key: self.key,
                flags: self.flags,
                // Sound: the slot's block and results stay in place until
                // its delivery, after `left` reaches zero.
                out: task.out.is_null().then(|| unsafe { out.add(index) } as *mut u8).unwrap_or(task.out),
                left,
                ..task
            }));
        }
        tasks.clear();
        state.tasks = tasks;
        state.link(slot as *mut Slot<I>);
        drop(guard);
        self.activate(owner);
    }

    /// Put a short message (with `batch`, a small batch of messages of
    /// that length) in flight as a member of the queue's open task: `member`
    /// gives its bytes and, for a batch, its digest space, from the item,
    /// which stays in its slot.
    fn submit_member(&self, owner: &Arc<dyn Any + Send + Sync>, item: I, batch: Option<usize>, member: impl FnOnce(&mut I) -> (*const u8, usize, Option<*mut u8>)) {
        let mut guard = crate::lanes::lock_polling(&self.state);
        let state = &mut *guard;
        let slot = state.take_slot(&self.returned);
        // Sound: a free slot is this thread's until linked.
        let slot = unsafe { &mut *slot };
        let (input, len, out) = member(slot.item.insert(item));
        // A message's digest lands in results[0]; a slot that held one
        // keeps it (rewriting it would take the line from the worker that
        // last wrote it).
        if slot.results.len() != 1 {
            slot.results.clear();
            slot.results.resize(1, [0; crate::BLOCK_LEN]);
        }
        slot.digest = true;
        slot.left.store(1, Ordering::Relaxed);
        let open = state.open.get_or_insert_with(|| Task::members(&self.key, self.flags, batch));
        // Sound: the slot's block, bytes, and result (or the digest space)
        // stay in place until its delivery, after `left` reaches zero.
        open.member[open.members] = crate::lanes::Member { input, len, out: out.unwrap_or(slot.results.as_mut_ptr() as *mut u8), left: &slot.left };
        open.members += 1;
        open.len += len;
        let closed = state.close_open(false);
        state.link(slot as *mut Slot<I>);
        drop(guard);
        if let Some(task) = closed {
            TASKS.push(core::iter::once(task));
        }
        self.activate(owner);
    }

    /// Have the delivery thread hold this queue, unless it does; after a
    /// link (see `deliver_with` for the other side of the handshake).
    fn activate(&self, owner: &Arc<dyn Any + Send + Sync>) {
        if !self.active.load(Ordering::SeqCst) && !self.active.swap(true, Ordering::SeqCst) {
            DELIVERY.hold(Self::owned(owner));
        }
    }

    /// The queue's state, owned, for the delivery thread to hold.
    fn owned(owner: &Arc<dyn Any + Send + Sync>) -> Arc<Self> {
        owner.clone().downcast().unwrap_or_else(|_| unreachable!("a queue's state has its shape's type"))
    }

    /*
     * The delivery thread's side: hand back every entry at the front whose
     * tasks are done (at most DELIVER_AT_ONCE), each through `deliver`,
     * with no lock the submitters take (the handler may submit to this
     * queue). Returns whether it delivered any, and whether the queue
     * still has entries in flight.
     */
    fn deliver_with(&self, deliver: impl Fn(&Self, &mut Handling<H>, I, &[[u8; crate::BLOCK_LEN]], bool)) -> (bool, bool) {
        let waiting = self.waiting.load(Ordering::Relaxed);
        // Sound: the entry stays linked and in place until this thread
        // delivers it.
        if !waiting.is_null() && unsafe { &*waiting }.load(Ordering::Acquire) > 0 {
            // The entry waited on may be in the open task of short
            // messages, which a stream fills first: hand it over after a
            // while (handing each over at once took the submitters' lock
            // and made tasks of a few, a loop that ran batches of 16 at a
            // fifth of their speed in some samples).
            let polls = self.polls.fetch_add(1, Ordering::Relaxed) + 1;
            if polls == CLOSE_AFTER_POLLS {
                let closed = crate::lanes::lock_polling(&self.state).close_open(true);
                if let Some(task) = closed {
                    TASKS.push(core::iter::once(task));
                }
            }
            return (false, true);
        }
        let mut head = self.head.load(Ordering::Relaxed);
        let mut count = 0;
        let mut unfinished = core::ptr::null_mut();
        let mut handling = None;
        while count < DELIVER_AT_ONCE {
            // Sound: `head` is this thread's (delivered, not yet handed
            // back), and a linked slot is in flight until delivered here.
            let next = unsafe { &(*head).next }.load(Ordering::SeqCst);
            if next.is_null() {
                break;
            }
            let left = unsafe { &(*next).left };
            if left.load(Ordering::Acquire) > 0 {
                unfinished = left as *const AtomicUsize as *mut AtomicUsize;
                break;
            }
            // The slot before it goes back before the handler runs (which
            // may let the program submit again): the queue's slots then
            // follow what the program keeps in flight, whatever the threads'
            // timing. Room for every slot: no allocation.
            crate::lanes::lock_polling(&self.returned).0.push(head);
            head = next;
            let handling = handling.get_or_insert_with(|| self.handling.lock().expect("no lock is poisoned"));
            // Sound: its tasks are finished; it is this thread's now.
            let item = unsafe { (*next).item.take() }.expect("a slot in flight holds its item");
            deliver(self, handling, item, unsafe { &(*next).results }, unsafe { (*next).digest });
            count += 1;
        }
        drop(handling);
        self.head.store(head, Ordering::Relaxed);
        self.waiting.store(unfinished, Ordering::Relaxed);
        self.polls.store(0, Ordering::Relaxed);
        if !unfinished.is_null() {
            return (count > 0, true);
        }
        if count == DELIVER_AT_ONCE {
            return (true, true);
        }
        // Nothing linked after `head`: let the queue go, unless a submitter
        // linked meanwhile. A submitter links, then sets `active`; this
        // thread clears `active`, then looks for a link (all SeqCst): either
        // it sees the link, or the submitter sees `active` clear and hands
        // the queue over again.
        self.active.store(false, Ordering::SeqCst);
        let linked = !unsafe { &(*head).next }.load(Ordering::SeqCst).is_null();
        // Relinked and still ours to hold, or handed over again (let this
        // copy go).
        let in_flight = linked && !self.active.swap(true, Ordering::SeqCst);
        (count > 0, in_flight)
    }
}

/// How many rounds the delivery thread waits on an entry before handing
/// over the open task of short messages, which may hold it: about a
/// microsecond (a round is a look at every queue and a pause).
const CLOSE_AFTER_POLLS: usize = 16;

/// The most entries of one queue delivered between two looks at its state.
const DELIVER_AT_ONCE: usize = 64;

/// What the delivery thread runs for a queue in flight.
trait Deliver: Send + Sync + 'static {
    /// Deliver what is done: (whether any was, whether entries remain).
    fn deliver(&self) -> (bool, bool);
}

impl<H: MessageHandler> Deliver for Inner<H, H::Buffer, shape::Messages> {
    fn deliver(&self) -> (bool, bool) {
        self.deliver_with(|queue, handling, buffer, results, digest| {
            let hash = if digest {
                Hash(results[0][..OUT_LEN].try_into().unwrap())
            } else if results.is_empty() {
                crate::hash_serial(buffer.as_ref(), &queue.key, queue.flags)
            } else {
                // A message is a stream of one piece.
                let mut hasher = crate::HasherCore::new_internal(&queue.key, queue.flags);
                hasher.update_with_results(buffer.as_ref(), &mut results.iter());
                hasher.final_output().root_hash()
            };
            handling.handler.hashed(buffer, hash);
        })
    }
}

impl<H: PieceHandler> Deliver for Inner<H, PieceItem<H::Buffer>, shape::Pieces> {
    fn deliver(&self) -> (bool, bool) {
        self.deliver_with(|_, handling, item, results, _| match item {
            PieceItem::Piece(piece) => {
                if results.is_empty() {
                    handling.hasher.update(piece.as_ref());
                } else {
                    handling.hasher.update_with_results(piece.as_ref(), &mut results.iter());
                }
                handling.handler.piece_done(piece);
            }
            PieceItem::Finish => {
                let hash = handling.hasher.final_output().root_hash();
                handling.hasher.reset();
                handling.handler.finished(hash);
            }
        })
    }
}

impl<H: FixedHandler> Deliver for Inner<H, (H::Buffer, H::Digests), shape::Fixed> {
    fn deliver(&self) -> (bool, bool) {
        self.deliver_with(|queue, handling, (buffer, mut digests), results, hashed| {
            // A small batch was hashed as a member of a task (`hashed`), a
            // larger one as tasks of its own (results); any other here.
            if results.is_empty() && !hashed {
                crate::hash_many_serial(buffer.as_ref(), queue.message_len, &queue.key, queue.flags, digests.as_mut());
            }
            handling.handler.hashed(buffer, digests);
        })
    }
}

impl<H: MessageHandler> Queue<H, shape::Messages> {
    /// A queue of messages of any length, one per buffer, each digest in
    /// `mode` delivered to `handler.hashed` with its buffer.
    pub fn messages(mode: Mode, handler: H) -> Self {
        Self::new::<H::Buffer>(mode, handler, 0)
    }

    /// Hash `buffer`'s bytes as one message; returns at once.
    pub fn submit(&self, buffer: H::Buffer) {
        let (inner, owner) = self.inner::<H::Buffer>();
        let tasks = inner.tasks;
        if tasks && buffer.as_ref().len() < TASK_MIN {
            return inner.submit_member(owner, buffer, None, |buffer| (buffer.as_ref().as_ptr(), buffer.as_ref().len(), None));
        }
        inner.submit(owner, buffer, |buffer, _, out| {
            let bytes = buffer.as_ref();
            tasks && bytes.len() >= TASK_MIN && {
                crate::plan_subtrees(&mut Default::default(), bytes, out);
                true
            }
        });
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
        let (inner, owner) = self.inner::<PieceItem<H::Buffer>>();
        let tasks = inner.tasks;
        inner.submit(owner, PieceItem::Piece(piece), |item, plan, out| {
            let PieceItem::Piece(piece) = item else { unreachable!("a piece") };
            let bytes = piece.as_ref();
            crate::plan_subtrees(plan, bytes, out);
            tasks && bytes.len() >= TASK_MIN
        });
    }

    /// End the message: its digest goes to `handler.finished` after every
    /// piece has come back. Returns at once.
    pub fn finish(&self) {
        let (inner, owner) = self.inner::<PieceItem<H::Buffer>>();
        inner.submit(owner, PieceItem::Finish, |_, plan, _| {
            *plan = Default::default();
            false
        });
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
        let (inner, owner) = self.inner::<(H::Buffer, H::Digests)>();
        let (message_len, tasks) = (inner.message_len, inner.tasks);
        let slot = crate::many::slot_len(message_len);
        assert_eq!(Some(buffer.as_ref().len()), slot.checked_mul(digests.as_mut().len()), "the buffer holds one slot of whole blocks per digest");
        let len = buffer.as_ref().len();
        if tasks && len > 0 && len < BATCH_TASK_MIN {
            // Small batches go several to a task, as short messages do.
            return inner.submit_member(owner, (buffer, digests), Some(message_len), |(buffer, digests)| (buffer.as_ref().as_ptr(), buffer.as_ref().len(), Some(digests.as_mut().as_mut_ptr() as *mut u8)));
        }
        inner.submit(owner, (buffer, digests), |(buffer, digests), _, out| {
            let bytes = buffer.as_ref();
            // The digests' space is the program's, in the slot with its
            // buffer, which stays put until delivery.
            let digests = digests.as_mut().as_mut_ptr();
            let per_task = (crate::lanes::TASK_LEN / slot).max(1);
            for (index, range) in bytes.chunks(per_task * slot).enumerate() {
                let mut task = Task::of(range, 0);
                task.batch = Some(message_len);
                // Sound: `per_task` digests per range, within the space.
                task.out = unsafe { digests.add(index * per_task) } as *mut u8;
                out.push(task);
            }
            // A batch that costs more to hash than a handover to the pool
            // (about a microsecond) hashes there; a shorter one at delivery.
            tasks && bytes.len() >= BATCH_TASK_MIN
        });
    }
}

/// The delivery thread's queues in flight, and whether it sleeps.
struct Delivery {
    /// The queues handed over, whether the thread sleeps, and the room
    /// both lists keep: two for every queue alive, since a queue handed
    /// over again while the thread lets it go is held twice for a moment
    /// (`Inner::activate`). Room made when a queue is made: no allocation
    /// in flight.
    queues: Mutex<(Vec<Arc<dyn Deliver>>, bool, usize)>,
    wake: Condvar,
}

static DELIVERY: Delivery = Delivery { queues: Mutex::new((Vec::new(), false, 0)), wake: Condvar::new() };

/// Start the delivery thread, once per process: from
/// `initialize_multithreaded`, or else at a queue's first entry in flight.
/// It sleeps whenever no queue has an entry in flight.
pub(crate) fn start_delivery() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        crate::lanes::prepare(&DELIVERY.queues, &DELIVERY.wake);
        // Return once the thread runs, as the pool's threads do: its start
        // allocates too (std's stack-overflow handler), which a warm queue
        // must not meet later.
        static RUNNING: AtomicBool = AtomicBool::new(false);
        std::thread::Builder::new()
            .name("blake3-servil-queue".into())
            .spawn(|| {
                RUNNING.store(true, Ordering::SeqCst);
                DELIVERY.run()
            })
            .expect("the queue's delivery thread starts");
        while !RUNNING.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
    });
}

impl Delivery {
    /// Give the delivery thread `queue`, which has just put an entry in
    /// flight; start the thread with the first.
    fn hold(&self, queue: Arc<dyn Deliver>) {
        start_delivery();
        let mut queues = crate::lanes::lock_polling(&self.queues);
        queues.0.push(queue);
        if queues.1 {
            self.wake.notify_one();
        }
    }

    /// Room for one more queue (`give_room` gives it back).
    fn make_room(&self) {
        let mut queues = crate::lanes::lock_polling(&self.queues);
        queues.2 += 2;
        let more = queues.2 - queues.0.len();
        queues.0.reserve(more);
    }

    fn give_room(&self) {
        crate::lanes::lock_polling(&self.queues).2 -= 2;
    }

    /// Deliver from every queue in flight, in turn; poll while any entry
    /// waits on its tasks, and sleep while no queue is in flight. A panic in a
    /// handler aborts the process (handler rule 4); so does one in the
    /// hashing, a bug.
    fn run(&self) {
        let mut polled = std::time::Instant::now();
        // The queues this round serves, swapped with the held list's (both
        // keep their capacity: no allocation).
        let mut queues: Vec<Arc<dyn Deliver>> = Vec::new();
        loop {
            {
                let mut held = crate::lanes::lock_polling(&self.queues);
                while held.0.is_empty() {
                    // Nothing in flight: this thread sleeps.
                    held.1 = true;
                    held = self.wake.wait(held).unwrap();
                    held.1 = false;
                }
                std::mem::swap(&mut held.0, &mut queues);
                let room = held.2;
                held.0.reserve(room);
            }
            let mut delivered = false;
            queues.retain(|queue| {
                let (any, in_flight) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.deliver())).unwrap_or_else(|_| std::process::abort());
                delivered |= any;
                in_flight
            });
            let mut held = crate::lanes::lock_polling(&self.queues);
            held.0.append(&mut queues);
            if !delivered {
                crate::lanes::poll_pause(&mut polled);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::lanes::Task;

    /// A few inputs through every shape of queue against
    /// hash(): small enough for Miri (the CI's smoketest runs it), which
    /// checks the chain's links, the slots' hand-over, and the delivery
    /// thread's reads for data races.
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
    use crate::{BLOCK_LEN, CHUNK_LEN, Hasher};
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
                    let left = AtomicUsize::new(tasks.len());
                    for (task, result) in tasks.into_iter().zip(&mut results) {
                        Task { key, flags: crate::KEYED_HASH, out: result.as_mut_ptr(), left: &left, ..task }.run(Platform::detect());
                    }
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
