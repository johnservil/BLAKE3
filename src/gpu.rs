//! The GPU of an Apple Silicon Mac, through Metal: where a stream hashes
//! long messages that the program writes in ([`crate::stream`]).
//!
//! **What it costs and gives** (Apple M4 Max; the probe branch
//! `probe/metal-gpu`, jobs 1705-1710). Starting Metal takes about 160 ms
//! the first time in a process (the driver compiles the shader) and 16
//! threads of Metal's own. Then a stream of 4 to 64 MiB messages that the
//! program writes into GPU memory hashes at 58-65 GB/s, twice the CPU
//! stream's 32-33, on one core where the CPU stream keeps twelve busy,
//! at an eighth of its energy a byte (0.18-0.22 nJ against 1.4-1.8); the
//! program's own writing is then the limit (in place the GPU reaches 208
//! GB/s). Idle, the GPU slows within about 5 ms; after 2 s idle its first
//! 64 MiB message took 17.8 ms, where the CPU pool takes 1.4: the stream
//! never waits on it (see `stream`'s race).
//!
//! **Engaging.** A stream takes the GPU only under sustained load: once
//! its run of long messages has lasted 10 ms (`Run`). The first such
//! message starts the GPU on a thread of its own and goes to the CPU;
//! later ones use the GPU once it is ready. A program that never streams
//! long messages for that long never starts Metal. The start checks the GPU's digests against the CPU's and stops
//! the program if they differ (a wrong digest is never returned). Where
//! Metal has no device, the CPU hashes everything, as on any machine.

use crate::{CVWords, Hash};
use core::sync::atomic::{AtomicU8, Ordering};
use std::ffi::{CStr, c_char, c_void};

unsafe extern "C" {
    fn blake3_servil_gpu_start(error: *mut c_char) -> bool;
    fn blake3_servil_gpu_slots_new(count: usize) -> *mut c_void;
    fn blake3_servil_gpu_room(slots: *mut c_void, slot: usize, capacity: usize, error: *mut c_char) -> *mut u8;
    fn blake3_servil_gpu_submit(slots: *mut c_void, slot: usize, length: usize, key: *const u32, flags: u32, error: *mut c_char) -> bool;
    fn blake3_servil_gpu_done(slots: *mut c_void, slot: usize) -> bool;
    fn blake3_servil_gpu_wait(slots: *mut c_void, slot: usize, out: *mut u8, error: *mut c_char) -> bool;
}

/// The shortest message a stream hashes on the GPU: at 4 MiB its
/// per-message cost (about 50 us a command) is a fraction of the message,
/// and the GPU stream ran 1.8x the CPU stream (1 MiB: 0.8x). A GPU slower
/// than the CPU on some machine costs little: its slots stay busy and the
/// CPU takes the other messages.
pub(crate) const MIN_LEN: usize = 4 << 20;

/// The longest: a slot holds a whole message, kept for the thread's next
/// stream, so a stream keeps at most SLOTS times this of GPU memory.
pub(crate) const MAX_LEN: usize = 64 << 20;

/// The most slots a stream has (see SLOT_BYTES): enough that a warm GPU
/// always has the next message while the program writes another (the
/// probe's GPU stream, with four, ran 58-65 GB/s at 4-64 MiB).
pub(crate) const SLOTS: usize = 4;

/// A stream engages the GPU once its run of long messages (each within
/// RUN_GAP of the one before) has lasted RUN_ENGAGE: the GPU takes 30-40 ms
/// of load to reach its clock and cools within about 5 ms of idle, so a
/// burst shorter than this finds it cold, and the CPU, which hashes a
/// message's pieces as the program writes them, finishes first. A program
/// that streams only bursts never starts Metal.
const RUN_GAP: std::time::Duration = std::time::Duration::from_millis(5);
const RUN_ENGAGE: std::time::Duration = std::time::Duration::from_millis(10);

/// A stream storage's run of long messages.
pub(crate) struct Run {
    started: std::time::Instant,
    last: std::time::Instant,
}

impl Run {
    pub(crate) fn new() -> Self {
        let now = std::time::Instant::now();
        Run { started: now, last: now - RUN_GAP - RUN_GAP }
    }

    /// Count a long message in the run: whether the run has lasted long
    /// enough to engage the GPU.
    #[inline]
    pub(crate) fn engaged(&mut self) -> bool {
        let now = std::time::Instant::now();
        if now.duration_since(self.last) > RUN_GAP {
            self.started = now;
        }
        self.last = now;
        now.duration_since(self.started) >= RUN_ENGAGE
    }
}

const COLD: u8 = 0;
const STARTING: u8 = 1;
const READY: u8 = 2;
const ABSENT: u8 = 3;

/// The process's GPU: cold until a stream first asks, then starting on a
/// thread of its own, then ready (or absent, for good).
static STATE: AtomicU8 = AtomicU8::new(COLD);

/// Whether the GPU is ready: true from the moment its start (and its check)
/// completed. The first call starts it and returns false.
#[inline]
pub(crate) fn ready() -> bool {
    match STATE.load(Ordering::Acquire) {
        READY => true,
        COLD => {
            // One caller wins the start; the others see it starting.
            if STATE.compare_exchange(COLD, STARTING, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                std::thread::Builder::new().name("blake3-servil GPU start".into()).spawn(start).expect("a thread to start the GPU");
            }
            false
        }
        _ => false,
    }
}

fn message(error: &[c_char; 512]) -> String {
    // The bridge writes a NUL-terminated message into this zeroed buffer.
    unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy().into_owned()
}

/// Start Metal and check the GPU against the CPU: every mode's flags, a
/// one-chunk message, an odd tree, and a long one.
fn start() {
    let mut error = [0; 512];
    // Sound: called once (STATE's compare-exchange), before READY.
    if !unsafe { blake3_servil_gpu_start(error.as_mut_ptr()) } {
        STATE.store(ABSENT, Ordering::Release);
        return;
    }
    let mut slots = Slots::new();
    let input: std::vec::Vec<u8> = (0..(5 << 20) + 17).map(|i| (i % 251) as u8).collect();
    let key = [7u8; 32];
    for mode in [crate::Mode::Plain, crate::Mode::Keyed(&key), crate::Mode::DeriveKey("blake3-servil GPU check")] {
        for len in [1, 1024, 1025, 3 * 1024 + 7, input.len()] {
            let (words, flags) = mode.key_and_flags();
            let gpu = check_one(&mut slots, &input[..len], &words, flags);
            let cpu = crate::convenience::hash_with(mode, &input[..len]);
            if gpu != cpu {
                // A GPU that hashes wrongly stops the program (never a wrong digest).
                eprintln!("blake3-servil: the GPU's digest differs from the CPU's ({len} bytes); stopping");
                std::process::abort();
            }
        }
    }
    STATE.store(READY, Ordering::Release);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    Free,
    /// Its command's digest goes to `place`, for the stream's entry `entry`,
    /// whose count (in the storage's `lefts`) is 1 until it is written.
    Owned {
        entry: usize,
        place: *mut Hash,
    },
    /// Its entry is done another way (the race, or the stream ended); the
    /// command may still run, reading the slot, and its digest is dropped.
    Orphaned,
}

/// A stream storage's slots of GPU memory, kept with it for the thread.
pub(crate) struct Slots {
    handle: *mut c_void,
    uses: [Use; SLOTS],
    /// Each slot's memory, as `room` last gave it (it moves only when the
    /// slot grows, which only `room` does).
    memory: [*mut u8; SLOTS],
}

/// The most GPU memory a stream's slots use: 128 MiB, so up to four slots
/// for messages of up to 32 MiB, two for longer ones (writing a message,
/// ~65 GB/s, is slower than the GPU's hashing, 200+, so two keep a warm
/// GPU fed even then).
const SLOT_BYTES: usize = 128 << 20;

impl Slots {
    /// New slots (no memory until first used). Requires a started GPU.
    pub(crate) fn new() -> Self {
        // Sound: Metal started (READY, or inside start).
        Slots { handle: unsafe { blake3_servil_gpu_slots_new(SLOTS) }, uses: [Use::Free; SLOTS], memory: [core::ptr::null_mut(); SLOTS] }
    }

    /// Every slot whose command has ended: an owned one writes its digest
    /// into its place and counts its entry done (0 in `lefts`), as a
    /// task's threads do; an orphaned one is free again.
    pub(crate) fn reap(&mut self, lefts: &[crate::lanes::OwnLine<core::sync::atomic::AtomicUsize>]) {
        for slot in 0..SLOTS {
            if self.uses[slot] == Use::Free || !self.done(slot) {
                continue;
            }
            let digest = self.wait(slot);
            if let Use::Owned { entry, place } = self.uses[slot] {
                // Sound: the place is the program's, lent for the stream's
                // scope and not yet handed back (its entry counts 1).
                unsafe { *place = digest };
                lefts[entry].store(0, Ordering::Relaxed);
            }
            self.uses[slot] = Use::Free;
        }
    }

    /// A free slot for a message of `len` bytes, after reaping.
    pub(crate) fn free(&mut self, len: usize, lefts: &[crate::lanes::OwnLine<core::sync::atomic::AtomicUsize>]) -> Option<usize> {
        self.reap(lefts);
        (0..(SLOT_BYTES / len).clamp(2, SLOTS)).find(|&slot| self.uses[slot] == Use::Free)
    }

    /// The free slot's memory for a message of `len` bytes (1..=MAX_LEN).
    pub(crate) fn room(&mut self, slot: usize, len: usize) -> &mut [u8] {
        debug_assert!(self.uses[slot] == Use::Free);
        let mut error = [0; 512];
        // Sound: the slot has no command in flight (it is free).
        let room = unsafe { blake3_servil_gpu_room(self.handle, slot, len, error.as_mut_ptr()) };
        assert!(!room.is_null(), "{}", message(&error));
        self.memory[slot] = room;
        // Sound: the slot's memory holds at least len bytes, and no command reads it.
        unsafe { core::slice::from_raw_parts_mut(room, len) }
    }

    /// The bytes of an owned slot's message, for the race.
    pub(crate) fn bytes(&self, slot: usize, len: usize) -> &[u8] {
        debug_assert!(matches!(self.uses[slot], Use::Owned { .. }));
        // Sound: an owned slot's memory holds its message, len bytes, which
        // nothing writes until the slot is free again (`room` alone writes).
        unsafe { core::slice::from_raw_parts(self.memory[slot], len) }
    }

    /// Start hashing the slot's first `len` bytes under `key` and `flags`,
    /// its digest for entry `entry`, into `place`.
    pub(crate) fn submit(&mut self, slot: usize, len: usize, key: &CVWords, flags: u8, entry: usize, place: *mut Hash) {
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_gpu_submit(self.handle, slot, len, key.as_ptr(), u32::from(flags), error.as_mut_ptr()) };
        assert!(ok, "{}", message(&error));
        self.uses[slot] = Use::Owned { entry, place };
    }

    /// Whether the slot's command has ended.
    #[inline]
    pub(crate) fn done(&self, slot: usize) -> bool {
        unsafe { blake3_servil_gpu_done(self.handle, slot) }
    }

    /// The slot still owing entry `entry` its digest.
    pub(crate) fn owing(&self, entry: usize) -> Option<usize> {
        (0..SLOTS).find(|&slot| matches!(self.uses[slot], Use::Owned { entry: e, .. } if e == entry))
    }

    /// The slot's entry is done another way: drop its command's digest.
    pub(crate) fn orphan(&mut self, slot: usize) {
        debug_assert!(matches!(self.uses[slot], Use::Owned { .. }));
        self.uses[slot] = Use::Orphaned;
    }

    /// Wait for the slot's command, and its digest.
    fn wait(&mut self, slot: usize) -> Hash {
        let mut digest = [0; 32];
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_gpu_wait(self.handle, slot, digest.as_mut_ptr(), error.as_mut_ptr()) };
        assert!(ok, "the GPU failed: {}", message(&error));
        Hash::from(digest)
    }
}

/// The start's own check: one message on slot 0, waited for.
fn check_one(slots: &mut Slots, input: &[u8], key: &CVWords, flags: u8) -> Hash {
    slots.room(0, input.len()).copy_from_slice(input);
    slots.submit(0, input.len(), key, flags, 0, core::ptr::null_mut());
    let digest = slots.wait(0);
    slots.uses[0] = Use::Free;
    digest
}
