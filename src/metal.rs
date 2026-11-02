//! Experimental Apple Silicon Metal hashing (`metal` feature): a stream of
//! messages hashed on the GPU and the CPU pool together ([`Hasher::stream`]),
//! on a [`Hasher`] reused across streams (it keeps the GPU's buffers). Call
//! [`crate::initialize_multithreaded`] before constructing one.
//!
//! **A stream of large messages is where the GPU pays.** The program writes
//! each message into the stream's own memory and goes on to the next while
//! the hashing runs; several messages stay queued on the GPU, so it holds
//! its full clock. On an Apple M4 Pro, 64 MiB messages already in place
//! stream at 111 GB/s against the CPU stream's 35, and at 48 against 28
//! when each is copied in (the copy then bounds both); the stream uses about
//! 5% of the CPU time. A GPU hashing one message at a time loses instead:
//! it idles at 338 MHz between calls and reaches 1578 MHz only under a
//! queue, and no public API sets the state.
//!
//! Messages shorter than [`Hasher::gpu_from`] go to the CPU pool's
//! [`crate::performance::stream`], which is faster for them: each message
//! costs the GPU a command and a pass a tree level, which on the M4 Pro's
//! 20 cores cost what the GPU saves at 4 MiB and twice it at 1 MiB, so the
//! GPU takes messages of 8 MiB and more there; the threshold scales with the
//! GPU's core count, unmeasured on smaller GPUs. A message must fit the
//! GPU's largest buffer. The `metal` example measures streams with the
//! repository's `clocks`.

use crate::{Hash, Mode};
use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn blake3_servil_metal_create(cores: *mut u32, error: *mut c_char) -> *mut c_void;
    fn blake3_servil_metal_destroy(ctx: *mut c_void);
    fn blake3_servil_metal_stream_reserve(ctx: *mut c_void, slot: usize, capacity: usize, error: *mut c_char) -> *mut u8;
    fn blake3_servil_metal_stream_submit(ctx: *mut c_void, slot: usize, length: usize, key: *const u32, mode: u32, error: *mut c_char) -> bool;
    fn blake3_servil_metal_stream_done(ctx: *mut c_void, slot: usize) -> bool;
    fn blake3_servil_metal_stream_wait(ctx: *mut c_void, slot: usize, out: *mut u8, error: *mut c_char) -> bool;
}

/// Reusable GPU state: one per calling thread. Construction returns an
/// error if Metal, the shader's compilation, or the GPU's core count fails.
pub struct Hasher {
    context: NonNull<c_void>,
    gpu_from: usize,
}

fn error_message(error: &[c_char; 512]) -> String {
    // The bridge writes a NUL-terminated message into this zeroed buffer.
    unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy().into_owned()
}

/// Slots a stream keeps for messages in flight on the GPU: enough that the
/// GPU always has the next message queued while the program fills a slot.
const IN_FLIGHT: usize = 4;

/// Where the GPU starts to pay on the 20 GPU cores of an Apple M4 Pro.
const GPU_FROM_ON_20_CORES: usize = 8 << 20;

impl Hasher {
    /// Create the GPU state, after CPU pool initialization: the threshold
    /// for the GPU ([`gpu_from`](Self::gpu_from)) comes from its core count.
    pub fn new() -> Result<Self, String> {
        let mut cores = 0;
        let mut hasher = Self::create(&mut cores)?;
        hasher.gpu_from = GPU_FROM_ON_20_CORES * 20 / cores as usize;
        Ok(hasher)
    }

    /// A hasher whose GPU takes every message of `gpu_from` bytes and more
    /// (0: every message), for measuring and testing the GPU on sizes the
    /// threshold keeps from it.
    #[doc(hidden)]
    pub fn with_gpu_from(gpu_from: usize) -> Result<Self, String> {
        let mut hasher = Self::create(&mut 0)?;
        hasher.gpu_from = gpu_from;
        Ok(hasher)
    }

    fn create(cores: &mut u32) -> Result<Self, String> {
        crate::lanes::require();
        let mut error = [0; 512];
        // The returned pointer uniquely owns the bridge; Drop releases it.
        let context = unsafe { blake3_servil_metal_create(cores, error.as_mut_ptr()) };
        NonNull::new(context).map(|context| Self { context, gpu_from: 0 }).ok_or_else(|| error_message(&error))
    }

    /// The length from which a stream's messages go to the GPU.
    pub fn gpu_from(&self) -> usize {
        self.gpu_from
    }

    /// A stream of messages hashed in `mode`, each where it is fastest:
    /// those of [`gpu_from`](Self::gpu_from) bytes and more whole on the
    /// GPU, written into the GPU's own memory ([`Stream::submit_with`]),
    /// shorter ones on the CPU pool's [`crate::performance::stream`]. Each
    /// digest is handed back through `done` on the program's thread, a
    /// little later, as it is done. **The order is narrower than the CPU
    /// stream's:** each device hands its digests back in submission order,
    /// and none holds between the two (each place names its message). The
    /// GPU hashes one message while the program fills the next, and stays
    /// at its full clock while messages keep coming. The body calls
    /// [`Stream::finish`] last.
    pub fn stream<'h, 'env: 'h, D: FnMut(&'env mut [Hash]) + 'h, T>(
        &'h mut self,
        mode: Mode<'env>,
        done: D,
        body: impl FnOnce(&mut Stream<'h, 'env, D>) -> T,
    ) -> T {
        let (key, flags) = mode.key_and_flags();
        let mut stream = Stream {
            hasher: self,
            mode,
            key,
            flags: u32::from(flags),
            done: std::rc::Rc::new(std::cell::UnsafeCell::new(done)),
            cpu: None,
            in_flight: std::collections::VecDeque::new(),
            free: (0..IN_FLIGHT).collect(),
        };
        let result = body(&mut stream);
        assert!(stream.in_flight.is_empty(), "a stream's body calls finish last");
        result
    }
}

type CpuStream<'h, 'env> = crate::stream::Stream<'env, Box<dyn FnMut(&'env mut [Hash]) + 'h>>;

/// A stream of messages hashed on the GPU and the CPU ([`Hasher::stream`]).
pub struct Stream<'h, 'env, D> {
    hasher: &'h mut Hasher,
    mode: Mode<'env>,
    key: [u32; 8],
    flags: u32,
    /// The handler, shared with the CPU stream. Sound: both callers run on
    /// this thread, the CPU stream's only inside its calls, which this
    /// stream makes while holding no reference to the handler.
    done: std::rc::Rc<std::cell::UnsafeCell<D>>,
    /// The CPU pool's stream, opened by the first short message: open, it
    /// keeps the pool's workers looking for its messages.
    cpu: Option<CpuStream<'h, 'env>>,
    /// The slots whose commands are in flight, oldest first, each with the
    /// place its digest goes. Sound: each place is an exclusive borrow for
    /// 'env, written once by this thread when its command completes and
    /// handed back once.
    in_flight: std::collections::VecDeque<(usize, *mut Hash)>,
    free: Vec<usize>,
}

impl<'h, 'env: 'h, D: FnMut(&'env mut [Hash]) + 'h> Stream<'h, 'env, D> {
    /// Hash a message of `len` bytes that `fill` writes in place, its
    /// digest into `out`: `fill(at, room)` writes the message's bytes from
    /// `at` at the start of `room` and returns how many it wrote, as
    /// [`crate::performance::Stream::submit_with`]'s does, called again
    /// until the room is full (one room, the whole message, for the GPU;
    /// pieces of up to 64 KiB for the CPU). The stream keeps the GPU memory
    /// of the largest messages it has seen, [`IN_FLIGHT`] of them. Waits
    /// for the oldest GPU message when every slot is in flight.
    pub fn submit_with(&mut self, len: usize, mut fill: impl FnMut(usize, &mut [u8]) -> usize, out: &'env mut Hash) {
        if len < self.hasher.gpu_from {
            return self.cpu_stream().submit_with(len, fill, out);
        }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => self.wait_oldest(),
        };
        let mut error = [0; 512];
        let room = unsafe { blake3_servil_metal_stream_reserve(self.hasher.context.as_ptr(), slot, len, error.as_mut_ptr()) };
        assert!(!room.is_null(), "{}", error_message(&error));
        // Sound: the slot's buffer holds len bytes and no command reads it.
        let room = unsafe { core::slice::from_raw_parts_mut(room, len) };
        let mut done = 0;
        while done < len {
            let n = fill(done, &mut room[done..]);
            assert!(n != 0, "fill wrote nothing with {} of its {len} bytes left: its source ended before the message's length", len - done);
            assert!(n <= len - done, "fill says it wrote {n} bytes into a room of {}", len - done);
            done += n;
        }
        let ok = unsafe { blake3_servil_metal_stream_submit(self.hasher.context.as_ptr(), slot, len, self.key.as_ptr(), self.flags, error.as_mut_ptr()) };
        assert!(ok, "{}", error_message(&error));
        self.in_flight.push_back((slot, out));
    }

    /// Hash the message of `len` bytes that starts `message`, its digest
    /// into `out`: where it is on the CPU, from a copy in the GPU's memory
    /// on the GPU.
    pub fn submit(&mut self, message: &'env [crate::performance::Block], len: usize, out: &'env mut Hash) {
        if len < self.hasher.gpu_from {
            return self.cpu_stream().submit(message, len, out);
        }
        let bytes = &crate::performance::Block::bytes(message)[..len];
        self.submit_with(
            len,
            |at, room| {
                room.copy_from_slice(&bytes[at..][..room.len()]);
                room.len()
            },
            out,
        )
    }

    /// The CPU stream, opened by the first short message.
    fn cpu_stream(&mut self) -> &mut CpuStream<'h, 'env> {
        self.cpu.get_or_insert_with(|| {
            let done = self.done.clone();
            // Sound: as the field says; the reference lives for this call only.
            crate::stream::Stream::open(self.mode, Box::new(move |place: &'env mut [Hash]| (unsafe { &mut *done.get() })(place)))
        })
    }

    /// Hand back every digest that is done, without waiting.
    pub fn collect(&mut self) {
        if let Some(cpu) = &mut self.cpu {
            cpu.collect();
        }
        while let Some(&(slot, _)) = self.in_flight.front() {
            if !unsafe { blake3_servil_metal_stream_done(self.hasher.context.as_ptr(), slot) } {
                break;
            }
            let slot = self.wait_oldest();
            self.free.push(slot);
        }
    }

    /// Hand back every digest, once it is done.
    pub fn finish(&mut self) {
        while !self.in_flight.is_empty() {
            let slot = self.wait_oldest();
            self.free.push(slot);
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.finish();
        }
    }

    /// Wait for the oldest GPU message and hand back its digest: its slot.
    fn wait_oldest(&mut self) -> usize {
        let (slot, out) = self.in_flight.pop_front().expect("a message in flight");
        let mut digest = [0; 32];
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_metal_stream_wait(self.hasher.context.as_ptr(), slot, digest.as_mut_ptr(), error.as_mut_ptr()) };
        assert!(ok, "{}", error_message(&error));
        // Sound: see in_flight; the place is handed back once, here.
        let place = unsafe { core::slice::from_raw_parts_mut(out, 1) };
        place[0] = Hash::from(digest);
        // Sound: see done; no other reference exists during this call.
        (unsafe { &mut *self.done.get() })(place);
        slot
    }
}

impl Drop for Hasher {
    fn drop(&mut self) {
        // Unique owner; every stream waits for its commands before it ends.
        unsafe { blake3_servil_metal_destroy(self.context.as_ptr()) };
    }
}
