# The servil API: why it is shaped so, and how the benchmark measures it

For the servil team. Users read the crate docs: `src/lib.rs`, "Which call
to use" (four questions and a table that lead to one call), and each
call's own documentation (its speed and rules; the batch layout
in `hash_many`; the handler rules in `Queue`). This file holds the reasons
behind them, the decisions with their dates, the open questions (each
marked **Q**), and how the benchmark measures each call. bench-hashes'
`FROZEN.md` turns it into measurements.

## Use cases, the APIs they need, and the measurements to settle them (October 3, 2026)

Zooko's plan: catalogue the programs that hash, and the APIs, kept and
considered; then build the benchmarks (and the tests) that settle every
question below, before designing any API or the mechanisms they might
share. Fear of the implementation's complexity exploding is answered by
measurements and tests, not by guesses.

### What programs hash, and how

| Program | What it hashes | How the data comes |
|---|---|---|
| b3sum, a backup of one large file, a container layer | one message, MB to TB | read or downloaded as it goes |
| `find -print0 \| xargs -0 b3sum`, `git add .` | many files, mostly KB, a few large | on disk, all available |
| `for F in ...; do b3sum $F; done` | one small file per process | a new process each time |
| Zero-knowledge proofs (Remco: 2^20 messages of 256 B), Merkle tree layers (64 B nodes) | many messages of one length | in memory |
| A server receiving uploads; Iroh and QUIC transfers; BitTorrent | many messages at once, KB to GB | arriving interleaved, in packets |
| TLS, WireGuard, HMAC per packet, signatures | one small message | in memory, latency-bound |
| Content-addressed stores naming a collection: Snix castore, Casita, Bazel REv2/REv3 (every input of every action), Nix (store paths as NARs), git (objects: blobs, trees, commits) | thousands to millions of items, mostly small, every length; each item's hash is its name, used to look it up, to refer to it (a directory lists its entries' hashes), and to check it | available at once (a tree being added, a pack being indexed), or one after another from a decompressor |
| Hugging Face Buckets (Xet), Epic's Lore, Peergos | large files cut into content-defined chunks (Xet: about 64 KiB), a tree over the chunks' hashes | chunks one after another |
| Roc bundles, Nixtamal, Subresource Integrity | one download or bundle, verified | at network or decompression speed |
| Iroh-blobs, Bao; Nix range reads (`narOffset`); SRI as it could be | verification before processing (CVE-2024-45593: a cache in the download path reached vulnerable code before the final hash check), and fetching one range of a file with its proof | a stream or a range, with the tree's inner chaining values |

Users and possible users today: NixOS/nix (issue 11999), Nixtamal, Snix,
Roc, Hugging Face Buckets, Epic Games Lore, BDASL, Peergos, Casita;
Subresource Integrity, Laut, Gradient, git (libra-tools git-internal),
Bazel REv2/REv3. Many single streams arrive at network or decompression
speed, below one core's hashing rate (about 4-6 GB/s on the M4 Max), where
a faster hash saves CPU time and energy rather than waiting.

### APIs kept, and the use case that pays for each

- `hash`, `keyed_hash`, `derive_key`, `hash_with(mode, ...)`: one message
  in memory on the caller's thread; nearly every program, and the fastest
  path for one small message.
- `Hasher` (`update`, `finalize`, `finalize_xof`, `update_reader`, keyed
  and derive-key forms): data in pieces; a `Hasher` per upload on a
  server; long outputs.
- `hash_multithreaded`: one large message in memory (about 6x `hash` at
  8 MiB on the M4 Max).
- `hash_many`, `hash_many_multithreaded`: many messages of one length
  (about 5x a loop of `hash` at 64 B).
- `hazmat`: subtrees at an offset and their merging, the base of Bao.
- `initialize`, `initialize_multithreaded`, `kernel_report`, `Mode`.
- **Questioned**: `Hasher::update_multithreaded`. Without lingering it
  helps only pieces of 512 KiB and more, the stream's use case done less
  well; if the stream lands, it likely goes (Simplicity, a second
  solution), and b3sum's read path moves to the stream.
- **Leaving**: the `Queue`, in its three shapes; the stream replaces it.

### APIs added, and the one still considered

1. **Verified streaming (Bao, iroh-blobs)**, in servil since October 4,
   2026: `outboard_with(mode, input) -> (Hash, Vec<u8>)` and
   `outboard_multithreaded_with` build a message's outboard (its tree's
   parent nodes above 16 KiB groups, pre-order: byte for byte bao-tree's
   pre-order outboard with 16 KiB blocks, as iroh-blobs stores it), and
   `verify_range_with(mode, hash, len, outboard, first, bytes)` checks any
   range of whole groups against the hash, as it arrives. One mechanism:
   groups hash in batches through the many-inputs kernels, on the pool
   for the multithreaded form (a third kind of pool work). VM, 64 MiB:
   build 0.173 ns/B on one thread and 0.026 on the pool, against
   bao-tree's 0.41-0.43 (2.4x, 16x); verify a megabyte at a time at about
   building's cost, one group at a time at twice it. The benchmark's
   outboard cell measures the build, with each message written first.
   Next: a stream verifier keeping the parents it has checked, so groups
   arriving one at a time cost building's price; ranges' proofs (the path
   nodes alone) for a reader that holds no outboard.
2. **A collection of messages of different lengths**, in servil since
   October 4, 2026: `hash_each_with(mode, items, out)`, short messages
   side by side in the SIMD lanes; items under 16 KiB 26-28% less time
   than one `hash` each, whole collections 9-11% (Mac job 1204); the
   benchmark's collection cell uses it for BLAKE3 servil on one thread.
   Next: items of one chunk or less (a quarter of git's objects) still
   hash one at a time; a multithreaded form for collections larger than
   a core's share.
3. **The stream does not replace the queue** (decided October 5, 2026,
   on Zooko's request to evaluate and judge). The queue pays most where
   pipelining matters most, and the stream has no form there: on the Mac
   (job 1213) pipelined 1 KiB messages take 0.121 ns/B against 0.683
   waiting for each call (5.6x), 64 KiB 0.061 against 0.229, batches of 16
   5.1 ns per message against 19.5. The stream beat the queue on long
   messages alone (64 MiB: 0.026 against 0.058, Mac job 1194; about level
   in the VM), where waiting for each call beats the queue too (0.046):
   the queue's long-message path is the gap, so it is the queue that
   improves there, and a second pipelined API for long messages would be
   a second solution. Simplicity: the queue is about 1,300 lines (queue.rs
   and its task list in lanes.rs) behind three shapes, three handler
   traits, and a delivery thread; the stream prototype about 300, for long
   messages only. The prototype stays on probe/owned-buffer. The history:
   one to
   three long messages arriving faster than one core hashes; 1.33x the
   next-best way with one stream, 1.10x with two, level at four, 19%
   slower at sixteen (probe/owned-buffer, Mac jobs 1196 on battery and
   1199 on mains, within 3% of each other); in b3sum, today's reader
   thread beat it (jobs 1197-1198). Kept until native Linux with io_uring
   is measured (NOTES-servil.md, Future work); a prototype, not in the
   crate.

4. **The API frozen for the optimisation pass** (Zooko, October 5, 2026:
   "add the lanes across messages before freezing the API, so that we can
   experiment with optimizations without thawing it"). Each new call
   lands first in its simplest correct form, so the benchmark (0.15.0)
   calls exactly what the optimisations will speed up:
   - `Verifier` (Zooko's "14. Yes"): a whole message verified as it
     arrives in iroh-blobs' wire format, in pieces of any length.
   - `Hasher::update_each(hashers, pieces)`, its twin
     `update_each_multithreaded`, and `Hasher::finalize_each(hashers,
     which, out)`: many messages in progress, a turn's pieces in one call,
     so different messages' bytes can share the lanes. Today they loop
     over `update`, `update_multithreaded`, and `finalize`.
   - `hash_each_multithreaded_with`: the multithreaded twin of
     `hash_each_with`. Zooko's reading of the two servil contenders: a
     program chooses one thread (it has no threads, or one thread per
     workload) or allows many, and each contender makes the fastest call
     under its choice; a multithreaded call is never slower than the
     single-threaded one on the same task. Today it hashes on the calling
     thread.
   - The Hasher gathers each message's pieces into whole 16 KiB itself
     (Zooko, October 5); no new call, and the docs of `update` tell
     callers who can cheaply batch that whole 16 KiB from a 16 KiB
     boundary skip the copy.

The two added APIs share the batched group hashing of `outboard.rs`
(`group_cvs_into`); `hash_each_with` lanes its own chunks. A stream, if it
lands, would hash its segments the same way.

### Many messages at once: the caller gathers, or the Hasher does (for Zooko)

The one large cell BLAKE3 servil loses (Mac job 1213: 0.504 ns/B against
SHA-256's 0.309) flips when each message's pieces reach `Hasher::update` in
whole 16 KiB from a 16 KiB boundary of the message (probe/gather-16k, Mac
job 1215, the benchmark's schedule, the copy charged): 0.283 ns/B, and 0.251
gathering 64 KiB, against 0.484 updating per piece (VM alike: 0.297, 0.268,
0.472). Updates of 16 KiB or more off that boundary gain a tenth (0.424).
Two ways to the cell; Zooko chose the Hasher's (October 5, 2026), with a
note in the docs for callers who can cheaply batch:
- **The caller's job** (his contract idea of October 3): `Hasher::update`'s
  docs say it is fastest given whole multiples of 16 KiB from a 16 KiB
  boundary, and the benchmark's servil contenders gather so in this cell
  (a change to what the benchmark asks). No code in the crate; the memory
  (16 KiB a message) is the program's to choose.
- **The Hasher's**: it keeps a 16 KiB staging buffer and gathers itself;
  the cell stays as it is, and every `Hasher` grows from about 1.9 KiB to
  about 18 KiB (git-internal already boxes `Hasher` for its size).

### The measurements that settle them

Each with the question it answers. Every cell charges the program's
write and the use of each hash (decisions 1 and 5 below). Each API's
implementation comes with the fork's tests against fixed answers
(`QUALITY.md`).

1. **b3sum's read paths** (`bench-hashes b3sum`, as it is: files from
   4 KiB to 1 GiB and a tree of 1000 files, warm and cold; each variant a
   fork branch built by `tools/b3sum-contenders.sh`): today's (a mapped
   file through `hash_multithreaded`, otherwise 4 MiB reads through
   `update_multithreaded`); reads into the stream's space; reads into
   buffers handed to `Queue::pieces`; on Linux, io_uring reads into the
   stream's space; one thread reading and calling `update`; official
   b3sum. Settles whether the stream pays in b3sum, warm and cold, and
   whether io_uring adds to it. With it, **one process per file** (the
   `for` loop): 1000 files of 16 KiB, a b3sum process each, start to
   exit; settles whether the start-up (the self-test, the pool) costs
   more than the hashing.
   *Measured* (Mac jobs 1197 on battery and 1198 on mains, October 3,
   2026; every cell within 1-3% between them, the second busy in 11 of 414
   load windows): the stream does not pay in b3sum. A cached file is fastest
   mapped (1 GiB: 38.8 ms, no copy); read instead, today's reader thread
   with two 4 MiB buffers and `update_multithreaded` beats the stream
   (51.6 ms against 58.1); from storage every multithreaded path reads at
   the drive's 6.7 GB/s (1 GiB: 159-160 ms), about one core's hashing
   rate, so they tie; small files lose with the stream or the queue
   (1000 files of 16 KiB, cached: 29.8 and 46.7 ms against 19.6), from
   their own pools and buffers. b3sum's own pipelining, a reader thread
   ahead of the hashing, already gets the overlap the stream offers: the
   stream's 1.33x (job 1196) was against writing whole messages, then
   hashing. One process per file: 1.40 ms a file today, against official
   b3sum's 1.58, the same as on one thread: the start-up is the process's,
   not the crate's.
   *Libra* (git-internal's main user; bench-hashes apps/libra-bench, Mac
   job 1202, October 4, 2026): in Libra's own benchmark the hash is a small
   part of every scenario (SHA-1 and BLAKE3 level, the fork's kernels level);
   hashing in place saves 7% in `fsck` and 12% in `add`, by the copies it
   removes; its time goes to thread handoffs (about 106 context switches per
   file added, 330 per object checked).
   *What Libra taught the API* (October 4, 2026; johnservil/libra
   `faster-add`, 7.37 s against 9.75 for `add`, nearly all of it
   independent of the hash):
   - **Framing is the commonest copy.** Five places in Libra copied each
     object behind its `"<type> <size>\0"` header before hashing. A `Mode`
     whose derive-key context names the object type replaces the header, and
     the crate's docs should show it, with the context key computed once
     (`Mode::DeriveKey(&str)` derives it again per call; a mode taking a
     precomputed `hazmat::ContextKey` would serve many small calls).
   - **A program's per-item bookkeeping dwarfs the hash** (database rows,
     marker files, ignore lookups, thread handoffs, about 1.5 ms per file
     against microseconds of hashing), so a hashing API pays only beside
     batched bookkeeping, and the API should make batching natural: results
     by index, one call per batch.
   - **`hash_each_with(mode, items, out)`** (probe/hash-each, tested against
     `hash_with`) fits that shape; on the collection cell's items it takes
     26-28% less time than one `hash` per item for items under 16 KiB, and
     9-11% for whole collections (Mac job 1204, on battery, quiet; the VM
     alike). Items of one chunk or less still go one at a time (a quarter of
     git's objects): laning them needs a kernel that starts each lane from
     the key over a different number of blocks. It one mode per call (group by object type), the
     caller's own buffers, so the caller keeps each file's stat with its
     bytes.
   - **The largest remaining cost is the file system** (per-file opens,
     stats, path resolution), which only an API that does the reading itself
     (`hash_files`, Level 3) could batch.
   - **The stream played no part:** these are many small items.
2. **A collection of items of different lengths**: a real collection's
   sizes, in a fixed order (the objects of a git repository at a fixed
   commit, the sizes listed in the code), in memory, each hashed once
   and its hash stored in its slot; every contender's one-shot call per
   item on one thread, and the same over the program's threads. Settles
   what a varied-length batch call could win, against `hash_many` at one
   length as the bound.
3. **Bao**: build the outboard of a 1 GiB message in memory; verify a
   1 GiB stream against it as it arrives in 16 KiB and 64 KiB pieces;
   verify one 1 MiB range. Contenders: the `bao` and `bao-tree` crates,
   later the fork's own; `hash_multithreaded` on the same 1 GiB as the
   bound for building. Settles how far today's crates are from what the
   hash allows.
4. **Many messages at once, pieces gathered**: the existing cell, plus
   BLAKE3 servil with each message's pieces gathered into 16 KiB before
   each `update`. Settles the "16 KiB per update" obligation for a
   `Hasher` per message.
5. **Batches of 256 B messages, up to 2^20**: beside the 64 B batches.
   Settles whether `hash_many` serves zero-knowledge proofs' messages as
   it does Merkle nodes.
6. **Several long streams at once**: 1, 2, 4, and 16 writer threads in one
   program, each writing and hashing 16 MiB messages; contenders: each
   message written whole then `hash_multithreaded`, a `Hasher` per writer,
   later the stream. Settles where the stream stops paying.

Order: 1 first (the stream's main question, on a tool that exists),
then 2 and 3 (the two new capabilities), then 4 to 6.

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

   /// The most one Space holds: half the stream's buffer.
   pub const MAX_SPACE_LEN: usize = 2 * 1024 * 1024;

   impl<T, F> HashStream<T, F> {
       /// Blocks until `len` bytes of the buffer are free, then lends them
       /// to you as the current message's next bytes. Requires `len` a
       /// multiple of SEGMENT_LEN, at most MAX_SPACE_LEN.
       pub fn space(&mut self, len: usize) -> Space;
       /// Ends the current message, named `tag`, and returns at once; it
       /// ends once every Space taken for it is committed. `on_hash(tag,
       /// hash)` is called with its hash once that is ready, on whichever
       /// thread finishes hashing it; calls may overlap and come in any
       /// order, and should return quickly (a hashing thread runs them).
       /// The next space() starts the next message.
       pub fn finish(&mut self, tag: T);
       /// Abandons the current message: `on_hash` is never called for it,
       /// and its space comes back once hashing already under way on it
       /// ends. The next space() starts a new message.
       pub fn cancel(&mut self);
   }

   /// Bytes of a stream's buffer, yours to write until you commit them.
   pub struct Space { /* its place in its message, its length */ }

   impl Space {
       pub fn bytes(&mut self) -> &mut [u8];
       /// You wrote the first `len` bytes; the stream hashes them. `len` is
       /// a multiple of SEGMENT_LEN, unless these are the message's last
       /// bytes, after which `finish` comes next. Any thread may commit.
       pub fn commit(self, len: usize);
   }
   ```

   - **Who writes**: a thread of the program's reading or computing into
     a Space; io_uring or a device writing into several Spaces at once,
     one taken per write, each committed as its write completes, in any
     order; several threads filling parts
     of one message. Data already in memory the program does not control
     (a mapped file, a network library's buffers) goes to
     `hash_multithreaded`, or is copied in (to measure: NOTES-servil.md,
     Future work).
   - **Space comes back a half at a time**: the buffer has two halves;
     `space` lends from the current one, and when that has fewer than
     `len` bytes free, it blocks until the other half is wholly hashed and
     lends from there (what the current half had left waits for its next
     turn).
   - **Merging in batches**: chaining values merge into their parents only
     when a level holds 32 adjacent ones (sixteen parents, one call of the
     widest SIMD kernels: SME2, AVX-512), or when the message is finished;
     the thread that completes such a run merges it.
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
   - **Cancelling** (Zooko, October 3, 2026): a server must not let a
     client hold its resources by sending half a message and stopping.
     `cancel` frees the stream for its next message; deciding when to give
     up (a timeout, a limit) is the caller's. A `Hasher` cancels by being
     dropped. A Space dropped uncommitted is part of no message: `finish`
     after one fails stop, and `cancel` is the way out.
   - **What the stream leaves to `Hasher`**: output longer than 32 bytes
     (`finalize_xof`); the stream's `on_hash` gets a `Hash`.
   - **Many messages at once** (a server's uploads): a `Hasher` per
     message, on the thread that receives it; the server's own threads
     keep the cores busy, and a thread hashing is a thread not reading,
     so TCP pushes back on the sender. The caller's part: at least 16 KiB
     per `update`, so BLAKE3 hashes 16 chunks at once (to measure: the
     benchmark's "many messages at once" cell with the pieces gathered
     first).
   - **Several streams at once share one pool** (to measure: 1, 2, 4, and
     16 streams, against writing each message whole and calling
     `hash_multithreaded`). Each stream's buffer, two halves of 2 MiB, is
     sized for the whole machine's speed; whether a caller should choose
     it follows from that measurement.
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
