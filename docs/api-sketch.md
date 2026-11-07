# Incremental background hashing: Rust API sketch

**Proposal for review. These methods are not available in a release or in the current library.** Names, lifetime parameters and cancellation behavior remain provisional.

BLAKE3-Servil has two intended interfaces:

- **Convenience:** ordinary bytes, synchronous one-shot or incremental hashing.
- **Throughput:** prepared storage, including a pipeline that hashes earlier bytes while the application receives or produces later bytes of the same message.

The [contract proposal](https://github.com/servil-software-factory/BLAKE3/issues/267) specifies behavior. The [implementation prototype](https://github.com/servil-software-factory/BLAKE3/pull/272) exercises reservations and worker handoff.

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

let mut place = Hash::from_bytes([0; 32]);
let stream = throughput::Stream::new(
    Mode::Plain,
    Limits { buffered_bytes: 2 << 20, messages: 32 },
    |completion| match completion {
        Completion::Hashed { tag, place } => println!("{tag}: {place}"),
        Completion::Aborted { tag, .. } => eprintln!("{tag}: aborted"),
    },
);

let mut message = stream.message(&mut place, 7u64, None).await;
let read_result = loop {
    let mut input = message.input().await;
    match reader.read_fragment(input.bytes_mut()).await {
        Ok((written, ended)) => {
            input.commit(written);
            if ended {
                message.end();
                break Ok(());
            }
        }
        Err(error) => {
            drop(input);
            message.abort();
            break Err(error);
        }
    }
};

stream.finish().await;
read_result?;
```

The tag is caller data, returned unchanged with its completion. `buffered_bytes` bounds retained and reserved input, including writable grants. `messages` bounds messages whose output places have yet to return. These limits are stream-wide.

`message.input().await` grants writable library-owned storage **and its submission capacity**. Available capacity completes immediately. Otherwise the operation registers once and the application handles other events. Space belongs to that request before its waker is notified.

`input.commit(written)` consumes the writable guard. It records the written bytes, submits complete prepared portions and preserves a partial portion for the next callback. It returns without waiting for hashing. A zero-byte commit records no bytes. `end` alone declares the message complete.

The producer's preparation helpers establish alignment, whole-unit boundaries and the final zero tail. Workers receive immutable prepared input. They hash it in place. A filesystem read or socket receive can fill that storage directly, with no intermediate input buffer.

`message.end()` consumes the write handle and seals its final portion. `abort()` also consumes the handle, produces no digest and returns storage when active jobs release it. `finish().await` consumes the stream and requires all messages ended or aborted. It waits for accepted work and invokes remaining completion handlers on the caller's thread.

## Several unfinished messages

One producer owns each stream. Its methods use shared access, so one message's write handle leaves the stream available to other messages.

```rust
let mut a = stream.message(&mut place_a, 1u64, None).await;
let mut b = stream.message(&mut place_b, 2u64, Some(3)).await;

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
Stream::new(mode, limits, on_completion) -> Stream
Stream::message(&self, output_place, tag, length_hint)
    -> Future<Output = Message>

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

Already-prepared messages and `Batch` values retain direct bulk operations with caller-supplied outputs. Their layouts can select efficient kernels without pipeline setup or per-message completion traffic.

The current `Stream::submit_reader` and `submit_with` already read or generate directly into library-owned storage. That direct-fill ability stays. The new capability is preserving one unfinished message across separate producer turns, with capacity waiting supplied by the library.

The public C API is convenience-only in this proposal. Prepared bulk inputs, the background pipeline and library-managed multithreading remain Rust interfaces.

## Feedback requested

1. Is `input().await` → write → `commit(n)` clear for your producer? Are `end`, `abort`, `ready` and `collect` the right operations?
2. **Output ownership and cancellation:** this sketch keeps the contract's caller-provided output places and producer-thread handler. Those places must remain live through completion or the stream's drain boundary. The current contract fails stop if a borrowed-work scope ends prematurely. Does that fit async cancellation, or should the pipeline instead own output slots and return per-message completion leases? That is a design alternative for review, not an additional supported API.
3. Are the byte/message budgets and advisory total-length hint enough to predict and control resource use? An incorrect hint changes neither bytes consumed nor end-of-message behavior.
4. Which concrete receive/generate workflow would make you choose this pipeline over the convenience Hasher or direct prepared-bulk calls?

The public Rust signatures will be settled after this review and implementation checks. The sketch makes no speed claims.
