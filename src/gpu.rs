//! The GPU of an Apple Silicon Mac, through Metal: one more worker for a
//! stream's long messages ([`crate::stream`]), taking their tasks in
//! batches.
//!
//! **How.** Once a stream's run of long messages engages the GPU (`Run`),
//! the program writes their pieces into the stream's GPU ring, 64 MiB of
//! memory the CPU and the GPU share, as tasks like any other: the pool's
//! workers take them one at a time. Between pieces, the stream's thread
//! claims the oldest BATCH tasks waiting, in one step, while fewer than
//! DEPTH of its commands are in flight; one command hashes them, and as it
//! ends, Metal's thread writes each task's output and counts it done, as a
//! worker does. So the GPU takes the work the pool has not reached. At the
//! stream's finish, with nothing else left to do, the stream takes back
//! the tasks of every command still in flight and gives them to the pool
//! (`Gpu::reclaim`): a stream's end waits for the pool's time on at most
//! DEPTH batches, never for the GPU's latency, and the GPU's work on them
//! is dropped.
//!
//! **What it costs and gives** (Apple M4 Max, against the CPU alone, the
//! probe branch `probe/gpu-4`, jobs 1901-1902, two runs, quiet). Streams of
//! 16 messages or more, of 4 or 16 MiB: 0.79-0.85 of the time alone and
//! 0.61-0.97 with two of a program's threads streaming at once, at
//! 0.63-0.92 of the CPU time, the GPU taking 25-48% of the pieces.
//! Streams of 1 to 4 messages: 0.98-1.01 alone; two at once, 16 MiB x 4,
//! 1.03 and 1.11. Without the reclaim, a batch claimed near a stream's end
//! made 4-message streams up to 1.20 alone (job 1898): one command takes
//! 0.2-1 ms from commit to its end. Starting Metal takes about 160 ms the
//! first time in a process (the driver compiles the shader), on a thread
//! of its own.
//!
//! **Engaging.** A stream uses its GPU ring only under sustained load: once
//! its run of long messages has lasted 10 ms (`Run`). The first such
//! message starts the GPU; later ones use it once it is ready. A program
//! that never streams long messages for that long never starts Metal. The
//! start checks the GPU's outputs against the CPU's and stops the program
//! if they differ (a wrong digest is never returned). Where Metal has no
//! device, the CPU hashes everything, as on any machine.

use crate::lanes::Task;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::ffi::{CStr, c_char, c_void};

unsafe extern "C" {
    fn blake3_servil_gpu_start(error: *mut c_char) -> bool;
    fn blake3_servil_gpu_buffer(length: usize, contents: *mut *mut u8, error: *mut c_char) -> *mut c_void;
    fn blake3_servil_gpu_commit(
        ring: *mut c_void,
        out: *mut c_void,
        pieces: *const Piece,
        count: usize,
        key: *const u32,
        flags: u32,
        done: extern "C" fn(*mut c_void, bool),
        context: *mut c_void,
        error: *mut c_char,
    ) -> bool;
}

/// A stream's GPU ring: 64 MiB, as many pieces as a stream has entries.
pub(crate) const RING_BYTES: usize = 64 << 20;

/// The tasks a command takes: 128, 8 MiB (job 1889: 39 GB/s with 4 in
/// flight, where 4 MiB reached 18; jobs 1901-1902: 16 MiB, with fewer
/// batches to share, slower on 16-message streams).
pub(crate) const BATCH: usize = 128;

/// The most commands a stream has in flight: 4, so the GPU always has the
/// next command queued as one ends.
const DEPTH: usize = 4;

/// A stream engages the GPU once its run of long messages (each within
/// RUN_GAP of the one before) has lasted RUN_ENGAGE: the GPU takes 30-40 ms
/// of load to reach its clock and cools within about 5 ms of idle, so a
/// burst shorter than this finds it cold. A program that streams only
/// bursts never starts Metal.
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
    /// enough to engage the GPU (out of line: the short messages' path
    /// keeps only the length's test).
    #[inline(never)]
    pub(crate) fn engaged(&mut self) -> bool {
        // The stream tests engage it outright: a run's timing is the
        // machine's (test_a_run_engages_after_10_ms tests it exactly).
        #[cfg(test)]
        if test::ENGAGED.load(Ordering::Relaxed) {
            return true;
        }
        self.engaged_at(std::time::Instant::now())
    }

    /// `engaged`, for a long message at `now`.
    fn engaged_at(&mut self, now: std::time::Instant) -> bool {
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

/// Start Metal and check the GPU against the CPU: in every mode, a batch
/// of tasks at chunk 0 (a message's first piece, its two halves' values)
/// and elsewhere, the largest counters' high words included.
fn start() {
    let mut error = [0; 512];
    // Sound: called once (STATE's compare-exchange), before READY.
    if !unsafe { blake3_servil_gpu_start(error.as_mut_ptr()) } {
        STATE.store(ABSENT, Ordering::Release);
        return;
    }
    let gpu = Gpu::new(8 * PIECE);
    // Sound: the ring is this GPU's own, and no command is in flight.
    let ring = unsafe { core::slice::from_raw_parts_mut(gpu.ring, 8 * PIECE) };
    for (i, b) in ring.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let counters = [0, 64, 1 << 20, (1 << 32) - 64, (1 << 40) + 128, 0, 3 << 32, 64 * 7];
    let key = [7u8; 32];
    for mode in [crate::Mode::Plain, crate::Mode::Keyed(&key), crate::Mode::DeriveKey("blake3-servil GPU check")] {
        let (words, flags) = mode.key_and_flags();
        let outs: std::vec::Vec<crate::lanes::OwnLine<UnsafeCell<[u8; 64]>>> = (0..16).map(|_| crate::lanes::OwnLine(UnsafeCell::new([0; 64]))).collect();
        let lefts: std::vec::Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(1)).collect();
        let task = |i: usize| Task {
            key: words,
            flags,
            out: outs[i].0.get() as *mut u8,
            left: &lefts[i],
            ..Task::of(&ring[(i % 8) * PIECE..][..PIECE], counters[i % 8])
        };
        let tasks: std::vec::Vec<Task> = (0..8).map(task).collect();
        assert!(gpu.commit(tasks.iter(), &words, flags), "a fresh GPU has a free command");
        for i in 8..16 {
            task(i).run(crate::platform::Platform::detect());
        }
        while lefts[..8].iter().any(|l| l.load(Ordering::Acquire) != 0) {
            std::thread::yield_now();
        }
        for i in 0..8 {
            // Sound: both tasks are done (their lefts read 0, Acquire).
            let (g, c) = unsafe { (*outs[i].0.get(), *outs[i + 8].0.get()) };
            let n = if counters[i] == 0 { 64 } else { 32 };
            if g[..n] != c[..n] {
                // A GPU that hashes wrongly stops the program (never a wrong digest).
                eprintln!("blake3-servil: the GPU's output differs from the CPU's (a piece at chunk {}); stopping", counters[i]);
                std::process::abort();
            }
        }
    }
    STATE.store(READY, Ordering::Release);
}

const PIECE: usize = crate::lanes::TASK_LEN;

/// A task as the kernel reads it: its piece's offset in the ring and its
/// first chunk's index.
#[repr(C)]
#[derive(Clone, Copy)]
struct Piece {
    offset: u64,
    counter: u64,
}

/// One of a GPU's commands, as it last was: kept for good, so Metal's
/// thread may reach it until the command ends.
struct Command {
    /// FREE, FLIGHT (committed), DONE (its end writing its tasks' outputs),
    /// or RECLAIMED (the stream took its tasks back at its finish; its end
    /// drops its outputs). FLIGHT moves on by one compare-and-swap, to DONE
    /// on Metal's thread or to RECLAIMED on the stream's: exactly one of
    /// them writes the tasks' outputs.
    state: AtomicU8,
    out: *mut c_void,
    /// BATCH * 64 bytes of output, 64 a task.
    out_bytes: *mut u8,
    /// Its tasks' outputs and counts, written by the stream's thread while
    /// the command is not busy, read by Metal's while it is.
    tasks: UnsafeCell<([Held; BATCH], usize)>,
}

/// A task as a command holds it: its piece, its chunk counter, its output
/// and its count.
#[derive(Clone, Copy)]
pub(crate) struct Held {
    pub(crate) input: *const u8,
    pub(crate) counter: u64,
    pub(crate) out: *mut u8,
    pub(crate) left: *const AtomicUsize,
}

/// The most tasks a stream's commands hold at once.
pub(crate) const HELD: usize = BATCH * DEPTH;

const FREE: u8 = 0;
const FLIGHT: u8 = 1;
const DONE: u8 = 2;
const RECLAIMED: u8 = 3;

/// A stream storage's GPU: its ring and its commands, kept with it for
/// the thread.
pub(crate) struct Gpu {
    ring_buffer: *mut c_void,
    /// The ring's bytes: the program writes some while the pool's workers
    /// and the GPU read others, so reached by pointer alone.
    pub(crate) ring: *mut u8,
    ring_len: usize,
    commands: Box<[Command]>,
}

impl Gpu {
    /// A GPU with a ring of `ring_len` bytes. Requires a started GPU.
    pub(crate) fn new(ring_len: usize) -> Gpu {
        let buffer = |len: usize| {
            let mut error = [0; 512];
            let mut contents = core::ptr::null_mut();
            // Sound: Metal started (READY, or inside start).
            let handle = unsafe { blake3_servil_gpu_buffer(len, &mut contents, error.as_mut_ptr()) };
            assert!(!handle.is_null(), "{}", message(&error));
            (handle, contents)
        };
        let (ring_buffer, ring) = buffer(ring_len);
        let commands = (0..DEPTH)
            .map(|_| {
                let (out, out_bytes) = buffer(BATCH * 64);
                let none = Held { input: core::ptr::null(), counter: 0, out: core::ptr::null_mut(), left: core::ptr::null() };
                Command { state: AtomicU8::new(FREE), out, out_bytes, tasks: UnsafeCell::new(([none; BATCH], 0)) }
            })
            .collect();
        Gpu { ring_buffer, ring, ring_len, commands }
    }

    /// Hash `tasks` (1 to BATCH whole pieces in the ring, of the mode of
    /// `key` and `flags`) on a free command: false, with nothing done,
    /// when every command is in flight. Each task's output and count, as
    /// its worker would, once the command ends.
    pub(crate) fn commit<'a>(&self, tasks: impl Iterator<Item = &'a Task>, key: &crate::CVWords, flags: u8) -> bool {
        let Some(command) = self.commands.iter().find(|c| c.state.load(Ordering::Acquire) == FREE) else { return false };
        let mut pieces = [Piece { offset: 0, counter: 0 }; BATCH];
        // Sound: the command is not busy, so Metal's thread is done with it.
        let held = unsafe { &mut *command.tasks.get() };
        let mut count = 0;
        for (i, task) in tasks.enumerate() {
            let offset = (task.input as usize).wrapping_sub(self.ring as usize);
            // Every build: a piece outside the ring would hash other bytes.
            assert!(task.len == PIECE && offset <= self.ring_len - PIECE, "a GPU task is a whole piece in the ring");
            debug_assert!(task.members == 0 && task.key == *key && task.flags == flags);
            pieces[i] = Piece { offset: offset as u64, counter: task.counter };
            held.0[i] = Held { input: task.input, counter: task.counter, out: task.out, left: task.left };
            count = i + 1;
        }
        debug_assert!(count >= 1);
        held.1 = count;
        command.state.store(FLIGHT, Ordering::Release);
        let mut error = [0; 512];
        // Sound: the ring and the output live for good; the pieces lie in
        // the ring, and the tasks' outputs and counts stay in place until
        // each count is zero (lanes::Task).
        let ok = unsafe {
            blake3_servil_gpu_commit(
                self.ring_buffer,
                command.out,
                pieces.as_ptr(),
                count,
                key.as_ptr(),
                u32::from(flags),
                completed,
                command as *const Command as *mut c_void,
                error.as_mut_ptr(),
            )
        };
        assert!(ok, "{}", message(&error));
        true
    }

    /// The held-command fixture starts with the warm-up's commands done.
    /// Its later HOLD applies only to work submitted by that fixture.
    #[cfg(test)]
    pub(crate) fn wait_idle_for_test(&self) {
        while self.commands.iter().any(|c| c.state.load(Ordering::Acquire) != FREE) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Whether any command is in flight: loads alone, for the finish's
    /// loop, which asks on every pass.
    #[inline]
    pub(crate) fn in_flight(&self) -> bool {
        self.commands.iter().any(|c| c.state.load(Ordering::Acquire) == FLIGHT)
    }

    /// Take back every command's tasks still in flight (the stream's
    /// finish, when nothing else is left to do): `f` gets each, for the
    /// pool's workers; the commands' ends drop their outputs. Whether it
    /// took any.
    pub(crate) fn reclaim(&self, mut f: impl FnMut(Held)) -> bool {
        let mut any = false;
        for command in self.commands.iter() {
            if command.state.compare_exchange(FLIGHT, RECLAIMED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                // Sound: RECLAIMED, so Metal's thread will not read the tasks.
                let (tasks, count) = unsafe { &*command.tasks.get() };
                tasks[..*count].iter().for_each(|&t| f(t));
                any = true;
            }
        }
        any
    }
}

/// A command ended (on Metal's thread): each task's 64 bytes of output to
/// its place, then its count down, as a worker finishes a task; then the
/// command is free. A failed command stops the program: its tasks have no
/// output, and their stream would wait for them for ever.
extern "C" fn completed(context: *mut c_void, ok: bool) {
    if !ok {
        eprintln!("blake3-servil: a GPU command failed; stopping");
        std::process::abort();
    }
    // A test holds the commands' ends, so the stream's finish reclaims.
    #[cfg(test)]
    while test::HOLD.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // Sound: the context is a command kept for good, not FREE until below.
    let command = unsafe { &*(context as *const Command) };
    if command.state.compare_exchange(FLIGHT, DONE, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        let (tasks, count) = unsafe { &*command.tasks.get() };
        for (i, t) in tasks[..*count].iter().enumerate() {
            // Sound: the command has ended, so its output is written; each
            // task's place (64 bytes) and count stay until the count is zero.
            unsafe {
                core::ptr::copy_nonoverlapping(command.out_bytes.add(i * 64), t.out, 64);
                (*t.left).fetch_sub(1, Ordering::Release);
            }
        }
    }
    command.state.store(FREE, Ordering::Release);
}

/// Give the GPU BATCH of `source`'s tasks, if it may take them now: the
/// oldest waiting, all at positions `from..to` (a run of whole pieces in
/// its ring), on a free command. Whether it gave any. Only the stream's
/// own thread calls this.
pub(crate) fn feed(gpu: &Gpu, source: &crate::lanes::sources::Source, from: usize, to: usize, key: &crate::CVWords, flags: u8) -> bool {
    if !gpu.commands.iter().any(|c| c.state.load(Ordering::Acquire) == FREE) {
        return false;
    }
    #[cfg(test)]
    test::FEEDS.fetch_add(1, Ordering::Relaxed);
    let mut taken = false;
    source.take_run(from, to, BATCH, |tasks| {
        taken = gpu.commit(tasks, key, flags);
        assert!(taken, "a command stays free on the stream's own thread");
    });
    #[cfg(test)]
    if taken {
        test::GPU_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
    taken
}

// Sound: the raw pointers are Metal's buffers and the commands' tasks,
// kept for good; a command's tasks move between threads across `state`.
unsafe impl Send for Gpu {}
unsafe impl Sync for Command {}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use std::time::{Duration, Instant};

    /// Batches the GPU took, in this process.
    pub(crate) static GPU_BATCHES: AtomicUsize = AtomicUsize::new(0);
    /// Finishes that took the GPU's tasks back, in this process.
    pub(crate) static RECLAIMS: AtomicUsize = AtomicUsize::new(0);
    /// Calls of `feed`, in this process.
    pub(crate) static FEEDS: AtomicUsize = AtomicUsize::new(0);
    /// While set, every run is engaged (the stream tests of the GPU).
    pub(crate) static ENGAGED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    /// While set, a command's end waits (on Metal's thread).
    pub(crate) static HOLD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

    /// A run engages once its messages, each within 5 ms of the one
    /// before, have lasted 10 ms; a longer gap starts a new run.
    #[test]
    fn test_a_run_engages_after_10_ms() {
        assert!(!Run::new().engaged(), "a run's first message never engages");
        let (ms, ns) = (Duration::from_millis, Duration::from_nanos);
        let mut run = Run::new();
        let t = Instant::now();
        assert!(!run.engaged_at(t));
        assert!(!run.engaged_at(t + ms(5)), "5 ms apart: the same run, 5 ms long");
        assert!(run.engaged_at(t + ms(10)), "10 ms long: engaged");
        assert!(!run.engaged_at(t + ms(15) + ns(1)), "more than 5 ms apart: a new run");
        assert!(!run.engaged_at(t + ms(20) + ns(1)), "5 ms into the new run");
        assert!(run.engaged_at(t + ms(25) + ns(1)), "10 ms into the new run");
    }

    #[test]
    fn test_message_reads_the_bridges_text() {
        let mut error = [0; 512];
        for (e, b) in error.iter_mut().zip(b"no GPU") {
            *e = *b as c_char;
        }
        assert_eq!(message(&error), "no GPU");
    }
}
