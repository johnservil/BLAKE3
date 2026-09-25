//! [`Stream`]: hashing that runs behind the caller. The caller fills the
//! stream's buffers (reading straight into them, or copying with
//! [`Stream::update`]); each full buffer goes to a hashing thread while the
//! caller fills the next. When every buffer is out, the caller's next
//! [`Stream::buffer`] waits for one to come back: that is the back-pressure.
//!
//! Each buffer is one whole subtree (BUFFER_LEN, a power of two
//! chunks), so the hashing thread feeds it to a [`Hasher`] whole, which
//! runs the flat walk (and, multithreaded, the pool) on it. The hashing
//! thread belongs to the thread that made the stream and is kept for its
//! next stream, asleep between them; so are the buffers. A stream that
//! never fills a buffer wakes no thread: `finalize` hashes it in place.

use crate::{Hash, Hasher};
use std::cell::{Cell, RefCell};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

/// Bytes per buffer: 1 MiB, the flat walk's largest subtree, at which
/// `Hasher::update` runs at `hash`'s speed.
pub(crate) const BUFFER_LEN: usize = 1 << 20;

/// Buffers per stream: the caller fills one while up to three wait for or
/// sit in the hashing thread.
const BUFFERS: usize = 4;

type Buffer = Box<[u8]>;

enum ToHasher {
    Start { multithreaded: bool },
    Buffer(Buffer),
    Finish,
}

enum FromHasher {
    Free(Buffer),
    Done(Hasher),
}

/// A connection to a hashing thread.
struct Link {
    to: SyncSender<ToHasher>,
    from: Receiver<FromHasher>,
}

thread_local! {
    /// This thread's idle hashing thread and spare buffers, for its next stream.
    static IDLE: RefCell<(Option<Link>, Vec<Buffer>)> = const { RefCell::new((None, Vec::new())) };
    /// The first spare buffer, in a slot of its own: a stream shorter than
    /// a buffer takes and returns only this one, with no borrow and no
    /// list (it cost a short stream 3-4 ns against a Hasher's update).
    static FIRST: Cell<Option<Buffer>> = const { Cell::new(None) };
}

fn spare_buffer() -> Buffer {
    if let Some(buffer) = FIRST.take() {
        return buffer;
    }
    IDLE.with(|idle| idle.borrow_mut().1.pop()).unwrap_or_else(|| vec![0u8; BUFFER_LEN].into_boxed_slice())
}

fn keep_buffer(buffer: Buffer) {
    /* The first slot takes it; a buffer already there goes to the list. */
    let Some(buffer) = FIRST.replace(Some(buffer)) else { return };
    IDLE.with(|idle| {
        let spare = &mut idle.borrow_mut().1;
        if spare.len() < BUFFERS {
            spare.push(buffer);
        }
    });
}

fn hashing_thread(from_caller: Receiver<ToHasher>, to_caller: SyncSender<FromHasher>) {
    let mut hasher = Hasher::new();
    let mut multithreaded = false;
    for message in from_caller {
        match message {
            ToHasher::Start { multithreaded: m } => {
                hasher = Hasher::new();
                multithreaded = m;
            }
            ToHasher::Buffer(buffer) => {
                if multithreaded {
                    hasher.update_multithreaded(&buffer);
                } else {
                    hasher.update(&buffer);
                }
                if to_caller.send(FromHasher::Free(buffer)).is_err() {
                    return;
                }
            }
            ToHasher::Finish => {
                let done = std::mem::replace(&mut hasher, Hasher::new());
                if to_caller.send(FromHasher::Done(done)).is_err() {
                    return;
                }
            }
        }
    }
}

fn take_link() -> Link {
    if let Some(link) = IDLE.with(|idle| idle.borrow_mut().0.take()) {
        return link;
    }
    let (to, from_caller) = sync_channel(BUFFERS + 2);
    let (to_caller, from) = sync_channel(BUFFERS + 2);
    std::thread::Builder::new()
        .name("blake3-stream".into())
        .spawn(move || hashing_thread(from_caller, to_caller))
        .expect("spawning a BLAKE3 stream thread");
    Link { to, from }
}

fn keep_link(link: Link) {
    IDLE.with(|idle| {
        let slot = &mut idle.borrow_mut().0;
        if slot.is_none() {
            *slot = Some(link);
        }
    });
}

/// A hash computed behind the caller: fill the stream's buffers, and a
/// hashing thread hashes each full one while the caller fills the next.
/// [`finalize`](Stream::finalize) returns the same [`Hash`] as [`crate::hash`]
/// of all the bytes filled, in order.
///
/// ```
/// use std::io::Read;
/// let data = vec![7u8; 3 << 20];
/// let mut source = &data[..];
/// let mut stream = blake3_servil::Stream::new();
/// loop {
///     let n = source.read(stream.buffer()).unwrap();
///     if n == 0 {
///         break;
///     }
///     stream.filled(n);
/// }
/// assert_eq!(stream.finalize(), blake3_servil::hash(&data));
/// ```
///
/// A stream holds four buffers of 1 MiB. When all four
/// are out, full and waiting for the hashing thread, [`buffer`](Stream::buffer)
/// blocks until one comes back, so a caller that produces faster than the
/// stream hashes waits instead of piling up memory. The hashing thread and
/// the buffers belong to the thread that made the stream and are kept,
/// asleep, for its next stream; a stream shorter than one buffer uses
/// neither the thread nor a second buffer. Dropping a stream without
/// finalizing it waits for the buffers already handed over.
pub struct Stream {
    multithreaded: bool,
    current: Option<Buffer>,
    filled: usize,
    /// Buffers with the hashing thread.
    out: usize,
    link: Option<Link>,
}

impl Stream {
    /// A stream whose buffers are each hashed on one thread, behind the
    /// caller's, as [`Hasher::update`] hashes them.
    #[inline]
    pub fn new() -> Self {
        Stream { multithreaded: false, current: None, filled: 0, out: 0, link: None }
    }

    /// A stream whose buffers are each hashed over this crate's worker
    /// threads, as [`Hasher::update_multithreaded`] hashes them.
    #[inline]
    pub fn new_multithreaded() -> Self {
        Stream { multithreaded: true, current: None, filled: 0, out: 0, link: None }
    }

    /// The free part of the current buffer, never empty: the next bytes
    /// of the input go here, then [`filled`](Stream::filled) says how many.
    /// Blocks while every buffer is with the hashing thread.
    #[inline]
    pub fn buffer(&mut self) -> &mut [u8] {
        if self.filled == BUFFER_LEN {
            self.hand_over();
        }
        let filled = self.filled;
        &mut self.current.get_or_insert_with(spare_buffer)[filled..]
    }

    /// The first `n` bytes of [`buffer`](Stream::buffer) now hold the
    /// next `n` bytes of the input. `n` is at most that buffer's length.
    #[inline]
    pub fn filled(&mut self, n: usize) {
        let room = self.current.as_ref().map_or(0, |_| BUFFER_LEN - self.filled);
        assert!(n <= room, "filled {n} bytes of a buffer with {room} free; call buffer() first");
        self.filled += n;
    }

    /// Copy `input` into the stream: [`buffer`](Stream::buffer) and
    /// [`filled`](Stream::filled) until all of it is in.
    #[inline]
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            let buffer = self.buffer();
            let n = buffer.len().min(input.len());
            buffer[..n].copy_from_slice(&input[..n]);
            self.filled(n);
            input = &input[n..];
        }
        self
    }

    /// Send the full current buffer to the hashing thread and take a free
    /// one, waiting for it when all are out.
    #[inline(never)]
    fn hand_over(&mut self) {
        let full = self.current.take().expect("a full buffer to hand over");
        let link = self.link.get_or_insert_with(|| {
            let link = take_link();
            link.to.send(ToHasher::Start { multithreaded: self.multithreaded }).expect("the stream thread runs");
            link
        });
        link.to.send(ToHasher::Buffer(full)).expect("the stream thread runs");
        self.out += 1;
        let next = if self.out < BUFFERS {
            spare_buffer()
        } else {
            match link.from.recv().expect("the stream thread runs") {
                FromHasher::Free(buffer) => {
                    self.out -= 1;
                    buffer
                }
                FromHasher::Done(_) => unreachable!("the stream thread finishes only when asked"),
            }
        };
        self.current = Some(next);
        self.filled = 0;
    }

    /// Wait for the hashing thread's [`Hasher`], keeping the buffers it
    /// returns; None when no buffer went to it.
    fn finish(&mut self) -> Option<Hasher> {
        let link = self.link.take()?;
        link.to.send(ToHasher::Finish).expect("the stream thread runs");
        let hasher = loop {
            match link.from.recv().expect("the stream thread runs") {
                FromHasher::Free(buffer) => {
                    self.out -= 1;
                    keep_buffer(buffer);
                }
                FromHasher::Done(hasher) => break hasher,
            }
        };
        debug_assert_eq!(self.out, 0);
        keep_link(link);
        Some(hasher)
    }

    /// The hash of every byte filled, in order.
    #[inline]
    pub fn finalize(mut self) -> Hash {
        /* A stream shorter than one buffer: hashed in place, the buffer kept. */
        if self.link.is_none() {
            let tail = self.current.take();
            let bytes = tail.as_ref().map_or(&[][..], |b| &b[..self.filled]);
            let hash = short_hash(bytes, self.multithreaded);
            if let Some(buffer) = tail {
                keep_buffer(buffer);
            }
            return hash;
        }
        self.finalize_handed_over()
    }

    #[inline(never)]
    fn finalize_handed_over(mut self) -> Hash {
        let tail = self.current.take();
        let tail_bytes = tail.as_ref().map_or(&[][..], |b| &b[..self.filled]);
        let hash = match self.finish() {
            None if self.multithreaded => crate::hash_multithreaded(tail_bytes),
            None => crate::hash(tail_bytes),
            Some(mut hasher) => {
                if self.multithreaded {
                    hasher.update_multithreaded(tail_bytes);
                } else {
                    hasher.update(tail_bytes);
                }
                hasher.finalize()
            }
        };
        if let Some(buffer) = tail {
            keep_buffer(buffer);
        }
        hash
    }
}

/// hash() or hash_multithreaded() of a stream that never handed over a
/// buffer; one chunk or less goes straight to the one-chunk code, which
/// is what both call there, without their platform check.
#[inline]
fn short_hash(bytes: &[u8], multithreaded: bool) -> Hash {
    #[cfg(blake3_neon_hybrid)]
    if bytes.len() <= crate::CHUNK_LEN {
        return crate::hash_one_chunk_root(bytes, crate::IV, 0);
    }
    if multithreaded { crate::hash_multithreaded(bytes) } else { crate::hash(bytes) }
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl Drop for Stream {
    #[inline]
    fn drop(&mut self) {
        if self.link.is_some() {
            self.finish();
        }
        if let Some(buffer) = self.current.take() {
            keep_buffer(buffer);
        }
    }
}

impl std::io::Write for Stream {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        self.update(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn input(len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        crate::test::paint_test_input(&mut v);
        v
    }

    #[test]
    fn test_stream_matches_hash() {
        let data = input(5 * BUFFER_LEN + BUFFER_LEN / 2 + 3);
        for len in [0, 1, 1023, 1024, 1025, BUFFER_LEN - 1, BUFFER_LEN, BUFFER_LEN + 1, 4 * BUFFER_LEN, 5 * BUFFER_LEN + 7, data.len()] {
            let expected = crate::hash(&data[..len]);
            for multithreaded in [false, true] {
                // Copies in pieces of several sizes.
                for piece in [1usize << 16, 1000, 3 << 20] {
                    let mut stream = if multithreaded { Stream::new_multithreaded() } else { Stream::new() };
                    for p in data[..len].chunks(piece) {
                        stream.update(p);
                    }
                    assert_eq!(stream.finalize(), expected, "len {len}, piece {piece}, mt {multithreaded}");
                }
                // Short fills into the buffers themselves, as reads return.
                let mut stream = if multithreaded { Stream::new_multithreaded() } else { Stream::new() };
                let mut rest = &data[..len];
                let mut step = 1;
                while !rest.is_empty() {
                    let buffer = stream.buffer();
                    let n = buffer.len().min(rest.len()).min(step);
                    buffer[..n].copy_from_slice(&rest[..n]);
                    stream.filled(n);
                    rest = &rest[n..];
                    step = step * 7 % 100_003 + 1;
                }
                assert_eq!(stream.finalize(), expected, "len {len}, short fills, mt {multithreaded}");
            }
        }
    }

    #[test]
    fn test_streams_side_by_side_and_dropped() {
        let data = input(3 * BUFFER_LEN + 99);
        let (mut a, mut b) = (Stream::new(), Stream::new_multithreaded());
        for p in data.chunks(1 << 16) {
            a.update(p);
            b.update(p);
        }
        let mut dropped = Stream::new();
        dropped.update(&data);
        drop(dropped);
        assert_eq!(a.finalize(), crate::hash(&data));
        assert_eq!(b.finalize(), crate::hash(&data));
        let mut again = Stream::new();
        again.update(&data[..2 * BUFFER_LEN]);
        assert_eq!(again.finalize(), crate::hash(&data[..2 * BUFFER_LEN]));
    }

    #[test]
    #[should_panic(expected = "call buffer() first")]
    fn test_filled_past_the_buffer_stops() {
        let mut stream = Stream::new();
        stream.buffer();
        stream.filled(BUFFER_LEN + 1);
    }
}
