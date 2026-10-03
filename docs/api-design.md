# The servil API: why it is shaped so, and how the benchmark measures it

For the servil team. Users read the crate docs: `src/lib.rs`, "Which call
to use" (four questions and a table that lead to one call), and each
call's own documentation (its speed and rules; the batch layout
in `hash_many`; the handler rules in `Queue`). This file holds the reasons
behind them, the decisions with their dates, the open questions (each
marked **Q**), and how the benchmark measures each call. bench-hashes'
`FROZEN.md` turns it into measurements.

## Decided October 3, 2026, to build next (Zooko)

1. **Every cell charges the program's write**: each byte is written into
   memory (a copy from a fixed source, standing in for a read) inside the
   timed work, in every use case, so that cells compare as the whole job a
   user's program does. Today the after-a-gap cells time the write apart and
   the nonstop cells include it.
2. **A stream that owns its buffer** replaces the queue: the program writes
   each message straight into space the stream lends it, threads hash each
   64 KiB segment as it is committed, and each message's hash goes to a
   function of the program's with the program's own tag for it. A
   prototype (probe/owned-buffer, Mac job 1194) hashed 64 MiB messages,
   write included, 1.4x as fast as writing them whole and calling
   `hash_multithreaded`, and 2.3x the queue. The API as planned:

   ```rust
   /// A stream that hashes messages written straight into its own buffer.
   pub fn hash_stream<T, F>(mode: Mode, on_hash: F) -> HashStream<T, F>
   where T: Send + 'static, F: Fn(T, Hash) + Send + Sync + 'static;

   /// The unit the stream hashes: 64 aligned chunks, one whole subtree.
   pub const SEGMENT_LEN: usize = 64 * 1024;

   impl<T, F> HashStream<T, F> {
       /// The next free bytes of the current message, a whole number of
       /// segments; waits until some are free.
       pub fn space(&mut self) -> Space;
       /// Ends the current message, named `tag`, and returns at once; it
       /// ends once every Space taken for it is committed. `on_hash(tag,
       /// hash)` is called with its hash once that is ready, on whichever
       /// thread finishes hashing it; calls may overlap and come in any
       /// order, and should return quickly (a hashing thread runs them).
       /// The next space() starts the next message.
       pub fn finish(&mut self, tag: T);
   }

   /// Bytes of a stream's buffer, yours to write until you commit them.
   pub struct Space { /* its place in its message, its length */ }

   impl Space {
       pub fn bytes(&mut self) -> &mut [u8];
       /// Two Spaces, split at `at`, a multiple of SEGMENT_LEN.
       pub fn split_at(self, at: usize) -> (Space, Space);
       /// You wrote the first `len` bytes; the stream hashes them. `len` is
       /// a multiple of SEGMENT_LEN, unless these are the message's last
       /// bytes, after which `finish` comes next. Any thread may commit.
       pub fn commit(self, len: usize);
   }
   ```

   - **Who writes**: a thread of the program's reading or computing into
     `space()`; io_uring or a device writing into split Spaces, committed
     as each write completes, in any order; several threads filling parts
     of one message. Data already in memory the program does not control
     (a mapped file, a network library's buffers) goes to
     `hash_multithreaded`, or is copied in (to measure: NOTES-servil.md,
     Future work).
   - **The hash, by tag**: `finish(tag)` names the message, and `on_hash`
     receives that name with its hash; nothing is matched by order.
   - **`on_hash` is `Fn + Sync`** (Zooko, October 3, 2026): the hashing
     thread that completes a message calls it directly, with no thread or
     handover between, and two may call it at once.
   - **Lifetimes**: the stream adds no threads (it uses the crate's pool,
     which lasts for the process). Its buffer is shared by the stream, each
     outstanding Space, and each thread hashing a segment of it, and is
     freed when the last lets go (AGENTS.md, "Crash-only": releasing is
     simpler here than keeping and reusing, for tests above all). Dropping
     the stream drops its unfinished message; finished ones still reach
     `on_hash`.
   - **Open**: short messages each start a segment, so many small ones
     waste the buffer (the benchmark of short messages of different
     lengths, Future work); a short read mid-message breaks the segment
     rule, which a helper that fills a Space from a reader would serve.
3. **No thread lingers**, and the benchmark's "in pieces" row goes: pieces
   lent to `update_multithreaded` hash as one buffer of their length does.
4. **The API docs are the doors**: each function's documentation is the one
   place for its contract and behaviour, published, holding no measured
   numbers; the graph, the guide, and the READMEs link to it.
5. **Every cell charges the use of each result** (Zooko, October 3, 2026),
   as item 1 charges the write: every cell makes the same use of each hash
   (the benchmark's `consume`, which FROZEN.md defines), wherever its
   design delivers it (on the calling thread for a returned hash, inside
   `on_hash` for the stream), and its timed work ends when the last hash
   has been used. Latency, lock contention, handovers, or copies that a
   design adds between a hash existing and its use count against that
   design.

## Four questions lead a user to one call

The crate docs put the questions right after their first example (Zooko,
September 28, 2026), so a
reader learns that the batch calls and the queue exist before reaching for
a loop over `hash`. `hash` keeps its name: a message in one buffer, now
and then, is likely the commonest case, and `hash` and
`hash_multithreaded` are its natural names.

**Does your thread keep up?** Three rates decide it: S, the data's arrival from outside; p, the
thread's own time per byte to get or make it; h, hashing's. Taking turns
(a synchronous call) runs at min(S, 1/(p+h)); pipelining (the queue) at
min(S, 1/p, the hashing threads' capacity), less its handovers, so it wins
only where S >= 1/(p+h). "Cannot tell" answers yes: a synchronous call is
never far wrong, and the queue pays only when there is something to
overlap (handovers, and wakes of 15-45 us when the workers sleep between
pieces).

**Whose buffer?** Owned buffers cycled through the queue cost no copy and
no allocation, stay in cache, and take the layout the program chooses
(whole 64 KiB pieces, padded batches). A buffer lent only until the call
returns would cost a copy to queue, so it gets a synchronous call on
several threads. One exception pays: copying tiny messages side by side
into a batch buffer (about 9 ns a 64-byte message through `hash_many`
against 42 through `hash`, single-threaded, Mac).

**Why each column holds its calls:**
- One thread: a synchronous loop is the best it can do; nothing overlaps
  without a second thread.
- Keeps up, several threads: the multithreaded one-buffer and batch calls
  shorten only the wait for a large input's digest. Pieces stay on
  `Hasher::update`: threads could shorten only the last piece's hash
  (pieces of 512 KiB or more arriving slowly are the exception, where
  `update_multithreaded` helps).
- Falls behind, buffer yours: the one column that pipelines; a future
  design with BLAKE3 owning the buffers or doing the reads goes here
  (NOTES-servil.md, "Future work").
- Falls behind, buffer lent: each call spreads over threads before
  returning, while the program and the hashing still take turns. Pieces
  shorter than 512 KiB stay on the calling thread, as `update`; a program
  that wants them hashed on other threads hands them over to a
  `Queue::pieces`.

**One concept per choice** (Zooko, October 2, 2026). Threading is in a
call's name (`hash` and `hash_multithreaded`, `hash_many` and
`hash_many_multithreaded`, `Hasher::update` and `update_multithreaded`;
the queue always multithreaded), and a call's only option is its mode:
each one-shot call has a `_with(mode, ...)` form, and each queue takes a
mode. Two concepts left the API, each costing more than it gave:
- **Thread budgets** (`Threads::Budget`, earlier the `..._with_budget`
  functions): few programs need one, and b3sum's `--num-threads`, the one
  user, now warns that it is ignored.
- **Time or energy** (`Efficiency` on the queue): few users would want it
  or know how to choose it. Saving energy stays possible future work
  (NOTES-servil.md, "Future work").

- **Q**: the multithreaded `Hasher` form's name (`update_multithreaded`
  is a placeholder).

## The synchronous calls

Each call returns its result; nothing keeps running for a call that may
come (AGENTS.md, "Serve real programs"), and no thread lingers: a worker
sleeps as soon as it finds nothing to take (Zooko, October 2, 2026).
Performance-sensitive programs pipeline through a queue, which needs no
lingering; a thread that lingered for the synchronous calls would spin
for work that may never come.

**Settled, as the code documents them:** initialization (`initialize`,
`initialize_multithreaded`; September 27), the batch layout (`hash_many`;
September 26), the modes (`Mode`, every call at the same speed; upstream's
`keyed_hash` and `derive_key` stay for drop-in use; September 27).
Signatures: inputs as `&[u8]` or owned buffers (the queue), one digest as
`Hash` (constant-time equality), batch digests as `&mut [[u8; 32]]`,
contract violations panic naming the rule broken.

## The queue

The queue maximises throughput, bytes or messages per second, and spends
latency to buy it. It owes that handovers never slow
the hashing threads, so its throughput is the hashing's for a program
that keeps enough in flight (Little's law: in flight = rate x round
trip). Its handler rules, shapes, and example are in `src/queue.rs`'s
docs; the reasons:

- **Event-based, through traits** (Zooko, September 27-28, 2026): results
  arrive as calls to a handler the user implements; no polling, no
  blocking, `submit` returns at once.
- **Zero copying**: buffers pass by ownership, as Rust's io_uring
  libraries do; an io_uring program reads into registered buffers and
  submits the same ones.
- **Zero allocation after warm-up** (`tests/queue_no_alloc.rs`); the
  program's side matches it with a queue kept for good and a
  `sync_channel` with room for everything in flight, as the docs' example
  and the benchmark do (Zooko, September 28, 2026).
- **One engine per process** (September 27, 2026): the pool, its hashing
  threads, the SME2 unit's turn, and one delivery thread, which the user
  never makes or configures. A queue is a cheap handle per stream, `Send
  + Sync`, `submit(&self)`, so a handler may hold its own queue and
  resubmit. Back-pressure is the program's own buffers.

**Later, optional**: an io_uring layer on Linux (a read's completion hands
its buffer to the queue; a handler posts into the program's ring with
`IORING_OP_MSG_RING`), and chaining (a compressor's output buffer becomes
the hasher's input).

## How the benchmark measures each

The benchmark measures each call only as its contract says users call
it, so that it guides us to optimise each one for them and flags only
what users meet (Zooko, September 28, 2026). Calls in the keeps-up columns follow the program's other work (the
gap); calls in the falls-behind columns run back to back. Every cell
records wall time and cycles.

**Keeps up: two shapes measured**, on one thread and several. Whole
messages and batches use their single-threaded or multithreaded entry
points. Each message or batch comes after a gap, of each kind. Message
lengths: 64 B-128 MiB; batches: 1-262144 messages of 64 B. Pieces, which
use `Hasher::update` in both columns, go unmeasured here: each piece
costs about what `hash` costs on a buffer that long, and the one-buffer
cells show it (Zooko, October 1, 2026; bench-hashes FROZEN.md).

**Falls behind, buffer yours: two axes.** Messages of one length,
64 B-64 MiB, produced into owned buffers: `Queue::messages` up to 64 KiB,
`Queue::pieces` beyond. Batches of 16-65536 64-byte messages through
`Queue::fixed`. The program keeps about 1 MiB or 1024 buffers in flight,
whichever is fewer, with its queue and bounded return channel kept
across samples. Reads and hashing are timed end to end. The other
contenders run the same producer through their synchronous calls.

**Falls behind, buffer lent: three axes.** Whole messages and
batches at the continuous axes' sizes, and 64 MiB messages in 64 KiB
pieces (one length: a long message shows the rate a multithreaded
incremental call sustains, which no one-buffer size predicts). Every input
is read into a kept buffer and lent to a synchronous call until it
returns; reads and hashing take turns, timed end to end. Whole messages
use `hash` or `hash_multithreaded`, pieces `update` or
`update_multithreaded`, and batches `hash_many` or
`hash_many_multithreaded`. The single-threaded cells also measure the
one-thread column when its thread falls behind.

**The gap**, of two kinds, each measured (Zooko, September 30, 2026;
`clocks::Gap`). *Amid other work*: a fixed other program (about 1 MiB of
distinct code, run once), a complete walk of a kept 128 MiB buffer, then
integer work to fill any remaining part of 1 ms, as on a machine busy
with other programs. *After idling*: a 1 ms sleep, as a server waiting
for its next request. Then the program writes the input (timed
separately, outside the hash sample) and calls. The timing helper is
`clocks::measure_after_gaps_prepared`; preparation and hashing each carry
wall time and thread counts separately. The other code is there because
a data walk alone left the call's code in the core's instruction cache,
by an amount the harness's own work decided (bench-hashes NOTES, "The
cause: where the hash's code is").

- perf_regress judges after-gap synchronous calls at 20%, continuous
  cells at 3% solo and 10% shared.
- **Modes**: keyed and derive-key spot checks at a few sizes in
  perf_regress (same cost as plain; a check that it stays so), no graph
  axis.

## The freeze

bench-hashes calls exactly the interfaces above in exactly these
patterns. Its `FROZEN.md` lists every use case, the servil call it makes,
its call pattern, its points, and its scenarios, each with its reason and
the decision's date; the test `frozen_contract_matches_frozen_md`
compares it with the code. A change to what the benchmark asks of servil
is Zooko's decision, recorded there.
