# Prepared bulk and incremental background hashing: Rust API sketch

**Proposal for review.** The `Block`/`Batch` types and direct bulk calls below exist today and stay in the design. The incremental pipeline, moved convenience Hasher and async submission signatures are proposed. Their names, lifetime parameters and cancellation behavior remain provisional.

BLAKE3-Servil has two intended interfaces:

- **Convenience:** ordinary bytes, synchronous one-shot or incremental hashing.
- **Throughput:** prepared storage, including a pipeline that hashes earlier bytes while the application receives or produces later bytes of the same message.

The [contract proposal](https://github.com/servil-software-factory/BLAKE3/issues/267) specifies behavior. The [implementation prototype](https://github.com/servil-software-factory/BLAKE3/pull/272) exercises reservations and worker handoff.

## API combinations

Convenience is tuned for occasional hashing amid other work. Throughput is tuned for repeated hashing. Its direct calls offer caller-thread and multithread-capable forms independently of the background pipeline.

| Your work / call family | Tuned for | Caller-thread entry | Workers / background entry |
|---|---|---|---|
| Ordinary bytes, one message | Occasional / cold | `convenience::hash`, `hash_with`, `hash_xof` | — |
| Ordinary bytes arriving in pieces | Occasional / convenient incremental state | `convenience::Hasher` **(proposed move)** | — |
| Repeated aligned messages | Repeated / warm | `throughput::hash` | `throughput::hash_multithreaded` |
| Equal-length messages in a `Batch` | Repeated / warm, side-by-side hashing | `throughput::hash_many` | `throughput::hash_many_multithreaded` |
| Mixed-length collection in `Block`s | Repeated / warm | `throughput::hash_each` | `throughput::hash_each_multithreaded` |
| Read/generate complete messages, batches, or incremental pieces | Ongoing production overlapped with hashing | — | `throughput::Stream`: typed `submit` / `submit_batch`, or message `input` / `commit` **(proposed forms)** |

- The first row's three functions are in `convenience`. Batching and library-managed parallelism belong to throughput. Its direct calls and prepared types already exist and are retained.
- Multithread-capable means the call may use workers where that helps. It still completes synchronously. Initialize those entry points with `initialize_multithreaded`.
- The background stream lets the producer continue while hashing runs. It uses the same initialized worker pool, within its configured budget.
- The current shared `throughput::Hasher` has hybrid tuning and parallel-update methods. The proposal splits that role into a convenience Hasher and throughput message handles. Grouped incremental Hasher operations need a settled migration path, as discussed below.

## Prepared inputs stay part of throughput

The throughput interface carries layout guarantees in types:

| Type | Guarantee |
|---|---|
| `Block` | 64 accessible bytes on a 64-byte boundary. Its contents can be any bytes. |
| `Batch` | Messages of one logical length, each starting on a block boundary, with protected zero padding after each message. |

Constructing a value establishes the guarantee. Hashing callers pass that value, so the implementation can trust the alignment and layout. `Block::from_bytes` checks existing storage once and returns a borrowed view when it is aligned and whole-block sized. It performs no copy.

The single-message and collection calls still ignore bytes beyond each logical end. `Block` alone does **not** promise a zero tail. `Batch` carries that stronger invariant because batch kernels read those bytes in place.

### Direct prepared-bulk calls: retained existing API

These are synchronous operations on work already ready to hash. They retain direct kernel paths, without forcing inputs through pipeline setup or completion callbacks.

| Input | Caller-thread call | Multithread-capable call |
|---|---|---|
| One aligned message: `&[Block]` and logical length | `throughput::hash` | `throughput::hash_multithreaded` |
| Equal-length messages: `&Batch` | `throughput::hash_many` | `throughput::hash_many_multithreaded` |
| Mixed lengths: `&[(&[Block], usize)]` | `throughput::hash_each` | `throughput::hash_each_multithreaded` |

Each call takes `Mode`. The multi-result calls write into the caller's output array.

#### One million 32-byte messages

This is a direct prepared-batch operation, already available today:

```rust
use blake3_servil::{Mode, throughput::{self, Batch, Block}};

blake3_servil::initialize_multithreaded(None); // Once per program.
const COUNT: usize = 1_000_000;
let mut batch = Batch::new(32, COUNT);
for i in 0..COUNT {
    produce_message_into(batch.message_mut(i)); // Writes directly into 32 bytes.
}
let mut digests = vec![[0u8; 32]; COUNT];
throughput::hash_many_multithreaded(Mode::Plain, &batch, &mut digests);
```

`produce_message_into` is application code. For existing messages, `batch.set(i, &messages[i])` copies each one into prepared storage. That preparation belongs in an end-to-end measurement.

Each message occupies a 64-byte-aligned block: 32 logical bytes plus 32 protected zero bytes. This batch occupies 64,000,000 bytes, and the digest array 32,000,000 bytes. A continuing producer can reuse smaller batches through the pipeline.

For caller-thread hashing, replace the final call with `throughput::hash_many(Mode::Plain, &batch, &mut digests)`. Convenience exposes ordinary byte-oriented hashing. These batch operations belong to throughput.

Already-aligned whole storage can also be borrowed without copying:

```rust
// aligned_whole_storage holds complete blocks, unlike packed 32-byte records.
let blocks = Block::from_bytes(aligned_whole_storage)
    .expect("the producer supplied aligned, whole-block storage");
let digest = throughput::hash(Mode::Plain, blocks, logical_message_bytes);
let nodes = Batch::from_blocks(blocks, 64); // Complete 64-byte messages.
```

`Batch::from_blocks` is a zero-copy read-only borrow when messages fill whole blocks. Throughput hashing retains prepared typed inputs.

### Pipelined prepared batches: proposed admission form

The stream also accepts ready work in its prepared representation. The proposed async form waits for admission, then returns without waiting for hashing:

```rust
stream.submit(blocks, logical_message_bytes, &mut place, tag).await;
stream.submit_batch(&batch, &mut batch_places, batch_tag).await;
```

Here `batch_places` is a slice of `Hash`, one per batch message. A producer can choose bounded groups independently of the million-message direct-call example. A whole batch retains its compact layout and group completion. The handler receives its output slice together, as the current stream does. A single message returns a slice of one.

The exact accounting of batch admission against stream limits remains provisional. Preserving the efficient batch representation is a requirement. Input ownership and output lifetimes are part of the review below.

## Convenience

```rust
let mut hasher = convenience::Hasher::new(Mode::Plain, None);
hasher.update(first_piece);
hasher.update(second_piece);
let digest = hasher.finalize();
hasher.reset(Some(likely_next_message_bytes));
```

The optional hint estimates the whole message's length. Actual bytes determine the digest. `update` returns with its input incorporated. The caller can immediately reuse that input. `finalize` returns the digest synchronously. There is no parallel-update or background-submit variant here.

One-shot hashing, keyed hashing, key derivation and extended output remain convenience operations. `Mode::Plain` means ordinary BLAKE3. A keyed or key-derivation mode uses the same incremental interface.

## Receive into the throughput pipeline

Program setup calls `initialize_multithreaded` once. Its configured thread count includes the producer. Background execution requires at least one worker besides it. The library chooses kernels and GPU use within the execution budget.

Inside an application's async receive routine, `reader.read_fragment(room)` writes into `room` and returns `(bytes_written, message_ended)`. This is an **application-supplied helper**. The application's protocol defines that message boundary.

```rust
use blake3_servil::{Hash, Mode, throughput};
use blake3_servil::throughput::{Completion, Limits};

let stream = throughput::Stream::new(
    Mode::Plain,
    Limits { buffered_bytes: 2 << 20, messages: 32 },
    |completion| match completion {
        Completion::Hashed { tag, hash } => println!("{tag}: {hash}"),
        Completion::Aborted { tag } => eprintln!("{tag}: aborted"),
    },
);

let mut message = stream.message(7u64, None).await;
let read_result = loop {
    let mut input = message.input().await;
    let ended = match reader.read_fragment(input.bytes_mut()).await {
        Ok((written, ended)) => {
            input.commit(written);
            ended
        }
        Err(error) => {
            // The borrow ends here; abort returns the room.
            message.abort();
            break Err(error);
        }
    };
    if ended {
        message.end();
        break Ok(());
    }
};

stream.finish().await;
read_result?;
```

The tag is caller data, returned unchanged with its completion. An incremental message's digest comes with its completion, by value: the producer lends no output place. `buffered_bytes` bounds retained and reserved input, including writable grants. `messages` bounds messages whose output places have yet to return. These limits are stream-wide.

`message.input().await` grants writable library-owned storage **and its submission capacity**. Available capacity completes immediately. Otherwise the operation registers once and the application handles other events. Space belongs to that request before its waker is notified.

`input.commit(written)` consumes the writable guard. It records the written bytes, submits complete prepared portions and preserves a partial portion for the next callback. It returns without waiting for hashing. A zero-byte commit records no bytes. `end` alone declares the message complete. The `Input` guard owns nothing: dropped unused, its room stays with the message for the next callback.

`Input::bytes_mut` is a controlled writer into library-owned storage, not a hashing call accepting arbitrary byte slices. Its preparation helpers establish alignment, whole-unit boundaries and the final zero tail. Workers receive immutable prepared input. They hash it in place. A filesystem read or socket receive can fill that storage directly, with no intermediate input buffer.

`message.end()` consumes the write handle and seals its final portion. `abort()` also consumes the handle, produces no digest and returns storage when active jobs release it. `finish().await` consumes the stream and requires all messages ended or aborted. It waits for accepted work and invokes remaining completion handlers on the caller's thread.

### Crash-only lifecycle

Every lifecycle step is an explicit call: `commit`, `end`, `abort`, `finish`. No handle does that work in `Drop`. Dropping a `Message` or granted `Input` with work accepted, or unwinding through a stream scope, is a broken contract: the process ends at once, since workers may still be reading or writing that memory. This is the rule the current `stream` already follows. A prototype that cleaned up in `Drop` let a panicking producer free a frame a worker was writing; the explicit form removes that path.

Async cancellation of accepted work means calling `abort()` (or letting `finish` drain); dropping a future that holds accepted work ends the process. A pending `message()` or `input()` request holds nothing yet: dropping it, as a timeout or `select!` does, is an ordinary cancellation, and the next waiter is served.

## Several unfinished messages

One producer owns each stream. Its methods use shared access, so one message's write handle leaves the stream available to other messages.

```rust
let mut a = stream.message(1u64, None).await;
let mut b = stream.message(2u64, Some(3)).await;

let mut input = b.input().await;
input.bytes_mut()[..3].copy_from_slice(b"abc");
input.commit(3);
b.end();

stream.ready().await;
stream.collect(); // Can deliver message 2 while message 1 is unfinished.

a.abort();
stream.finish().await;
```

`ready` notifies the producer about completed work. `collect` invokes handlers for ready completions without waiting. Workers signal readiness and compute hashes. Completion handlers run when the producer collects or finishes.

## Provisional surface

These are Rust-style operation signatures. Private fields, exact lifetime parameters and trait bounds are omitted. This block is notation for review, not a compilable API declaration.

```text
// Existing direct prepared-input calls remain as listed above.
Stream::new(mode, limits, on_completion) -> Stream
Stream::submit(&self, blocks: &[Block], len, output_place, tag)
    -> Future<Output = ()>             // Admission, not hash completion.
Stream::submit_batch(&self, batch: &Batch, output_places, tag)
    -> Future<Output = ()>             // Preserves the batch representation.
Stream::message(&self, tag, length_hint)
    -> Future<Output = Message>      // Its digest arrives with its completion.

Message::input(&mut self) -> Future<Output = Input>
Input::bytes_mut(&mut self) -> &mut [u8]
Input::commit(self, written: usize)
Message::end(self)
Message::abort(self)

Stream::ready(&self) -> Future<Output = ()>
Stream::collect(&self) -> usize
Stream::finish(self) -> Future<Output = ()>
```

The futures use Rust's standard polling/waker interface. BLAKE3-Servil chooses no async runtime. A callback adapter can arrange notification through an event loop. The blocking reader adapter waits internally on the same resource requests.

## Existing prepared inputs and blocking readers

The current `Stream::submit_reader`, `submit_with` and `submit_batch_with` already read or generate directly into library-owned storage. Direct-fill message and batch adapters stay in the design. The new capability is preserving one unfinished message across separate producer turns, with capacity waiting supplied by the library.

The existing grouped incremental Hasher operations (`update_each`, `update_each_multithreaded`, `finalize_each`) also need an explicit migration path. Their final replacement in the per-message pipeline remains unsettled. The ready-bulk table above stays independently of that migration.

The public C API is convenience-only in this proposal. Prepared bulk inputs, the background pipeline and library-managed multithreading remain Rust interfaces.

## Feedback requested

1. Do the retained typed bulk calls and proposed `submit_batch` form cover your ready-work flow? Is `input().await` → write → `commit(n)` clear for incremental production?
2. **Output ownership:** incremental messages hand their digest over by value with the completion (the pipeline owns the result; one 32-byte copy), while batches keep caller-provided digest places under the crash-only lifecycle above. Does that split fit your programs?
3. Are the byte/message budgets and advisory total-length hint enough to predict and control resource use? An incorrect hint changes neither bytes consumed nor end-of-message behavior.
4. Which concrete receive/generate workflow would make you choose this pipeline over the convenience Hasher or direct prepared-bulk calls?

The public Rust signatures will be settled after this review and implementation checks. The sketch makes no speed claims.
