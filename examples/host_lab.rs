//! probe/owned-buffer: a prototype of a stream hasher that owns its input
//! buffer (Zooko, October 3, 2026), against the fork's Queue::pieces, on one
//! long message after another.
//!
//! The prototype: BLAKE3 owns two halves of one buffer, each a whole number
//! of 64 KiB segments (a segment: 64 aligned chunks, one subtree of the
//! tree). The program writes its data straight into the current half
//! (`space`) and commits whole segments (only a message's last may be
//! shorter). Hashing threads claim committed segments in order through one
//! counter, and write each segment's chaining value into that segment's
//! slot of a results array. The program's thread merges the chaining values
//! in order, in its own calls, and moves into the other half only once
//! everything that half held is hashed; until then it hashes unclaimed
//! segments itself, and waits only for segments other threads are hashing.
//! Hashing threads sleep as soon as nothing is committed and unclaimed.

use blake3_servil::hazmat::{self, HasherExt, Mode};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::Arc;

const SEG: usize = 64 * 1024;

/// Diagnostics: the program's thread's time copying, committing, and
/// waiting for room; wakes by the program's thread and by workers; parks.
static COPY_NS: AtomicU64 = AtomicU64::new(0);
static COMMIT_NS: AtomicU64 = AtomicU64::new(0);
static WAIT_NS: AtomicU64 = AtomicU64::new(0);
static FINISH_NS: AtomicU64 = AtomicU64::new(0);
static WAKES_PROGRAM: AtomicU64 = AtomicU64::new(0);
static WAKES_WORKER: AtomicU64 = AtomicU64::new(0);
static PARKS: AtomicU64 = AtomicU64::new(0);
static HELPED: AtomicU64 = AtomicU64::new(0);
static MERGE_NS: AtomicU64 = AtomicU64::new(0);
static PUBLISH_NS: AtomicU64 = AtomicU64::new(0);

/// When the program's commit wakes a sleeping hashing thread.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Wake {
    /// Whenever a commit leaves segments unclaimed and a thread sleeps.
    Every,
    /// Only once half a half's segments wait unclaimed.
    Half,
}

struct Shared {
    data: Box<[UnsafeCell<u8>]>,
    /// Segments in the whole buffer (two halves).
    segs: u64,
    /// Segments committed and claimed, counted from the stream's start.
    committed: AtomicU64,
    claimed: AtomicU64,
    /// Per slot: the segment's input offset and length, written before its commit.
    offset: Box<[UnsafeCell<u64>]>,
    len: Box<[UnsafeCell<usize>]>,
    /// Per slot: 1 + the index of the segment whose chaining value it holds.
    done: Box<[AtomicU64]>,
    cvs: Box<[UnsafeCell<[u8; 32]>]>,
    idle: Box<[AtomicBool]>,
    threads: std::sync::OnceLock<Vec<std::thread::Thread>>,
    stop: AtomicBool,
}

unsafe impl Sync for Shared {}

impl Shared {
    fn slot(&self, k: u64) -> usize {
        (k % self.segs) as usize
    }

    /// Hash segment k (claimed by this thread) into its slot.
    fn hash(&self, k: u64) {
        let s = self.slot(k);
        let (offset, len) = unsafe { (*self.offset[s].get(), *self.len[s].get()) };
        let bytes = unsafe { std::slice::from_raw_parts(self.data[s * SEG].get() as *const u8, len) };
        let mut h = blake3_servil::Hasher::new();
        h.set_input_offset(offset);
        h.update(bytes);
        unsafe { *self.cvs[s].get() = h.finalize_non_root() };
        self.done[s].store(k + 1, Release);
    }

    /// Claim the next committed segment, if any.
    fn claim(&self) -> Option<u64> {
        let mut c = self.claimed.load(Relaxed);
        loop {
            if c >= self.committed.load(Acquire) {
                return None;
            }
            match self.claimed.compare_exchange_weak(c, c + 1, AcqRel, Relaxed) {
                Ok(_) => return Some(c),
                Err(now) => c = now,
            }
        }
    }

    /// Wake one sleeping hashing thread, if one sleeps.
    fn wake_one(&self) {
        for (i, idle) in self.idle.iter().enumerate() {
            if idle.load(Relaxed) && idle.compare_exchange(true, false, AcqRel, Relaxed).is_ok() {
                self.threads.get().unwrap()[i].unpark();
                return;
            }
        }
    }

    fn worker(&self, me: usize) {
        while !self.stop.load(Relaxed) {
            if let Some(k) = self.claim() {
                // More waiting: pass the wake on.
                if self.claimed.load(Relaxed) < self.committed.load(Relaxed) {
                    WAKES_WORKER.fetch_add(1, Relaxed);
                    self.wake_one();
                }
                self.hash(k);
                continue;
            }
            self.idle[me].store(true, SeqCst);
            if self.claimed.load(SeqCst) < self.committed.load(SeqCst) || self.stop.load(SeqCst) {
                let _ = self.idle[me].compare_exchange(true, false, AcqRel, Relaxed);
                continue;
            }
            PARKS.fetch_add(1, Relaxed);
            std::thread::park();
            self.idle[me].store(false, Relaxed);
        }
    }
}

/// When the program's thread may write into freed space.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Layout {
    /// Two halves: a half is written only once all it held is hashed.
    Halves,
    /// A ring: each segment is written once the one it held is hashed.
    Ring,
}

struct Stream {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    wake: Wake,
    layout: Layout,
    /// Whether the program's thread hashes unclaimed segments while it waits.
    help: bool,
    /// The next segment to write, and the bytes already in it.
    write: u64,
    /// Segments merged into the chaining-value stack.
    merged: u64,
    /// The message's first segment, its bytes so far, and its stack.
    start: u64,
    bytes: u64,
    stack: Vec<[u8; 32]>,
    count: u64,
}

impl Stream {
    fn new(half_segs: u64, threads: usize, wake: Wake, layout: Layout, help: bool) -> Stream {
        let segs = 2 * half_segs;
        let shared = Arc::new(Shared {
            data: (0..segs as usize * SEG).map(|_| UnsafeCell::new(0)).collect(),
            segs,
            committed: AtomicU64::new(0),
            claimed: AtomicU64::new(0),
            offset: (0..segs).map(|_| UnsafeCell::new(0)).collect(),
            len: (0..segs).map(|_| UnsafeCell::new(0)).collect(),
            done: (0..segs).map(|_| AtomicU64::new(0)).collect(),
            cvs: (0..segs).map(|_| UnsafeCell::new([0; 32])).collect(),
            idle: (0..threads).map(|_| AtomicBool::new(false)).collect(),
            threads: std::sync::OnceLock::new(),
            stop: AtomicBool::new(false),
        });
        let workers: Vec<_> = (0..threads)
            .map(|i| {
                let s = shared.clone();
                std::thread::spawn(move || s.worker(i))
            })
            .collect();
        shared.threads.set(workers.iter().map(|w| w.thread().clone()).collect()).unwrap();
        Stream { shared, workers, wake, layout, help, write: 0, merged: 0, start: 0, bytes: 0, stack: Vec::new(), count: 0 }
    }

    fn half(&self) -> u64 {
        self.shared.segs / 2
    }

    /// Merge every segment hashed, in order. Returns whether it merged any.
    fn merge(&mut self) -> bool {
        let sh = &*self.shared;
        let mut moved = false;
        while self.merged < sh.committed.load(Relaxed) && sh.done[sh.slot(self.merged)].load(Acquire) == self.merged + 1 {
            let cv = unsafe { *sh.cvs[sh.slot(self.merged)].get() };
            while self.stack.len() > self.count.count_ones() as usize {
                let right = self.stack.pop().unwrap();
                let left = self.stack.pop().unwrap();
                self.stack.push(hazmat::merge_subtrees_non_root(&left, &right, Mode::Hash));
            }
            self.stack.push(cv);
            self.count += 1;
            self.merged += 1;
            moved = true;
        }
        moved
    }

    /// One step of waiting for hashing to finish: merge what is ready;
    /// else hash an unclaimed segment (with `help`), or wake a thread for
    /// it; else spin a moment (another thread is hashing).
    fn wait_step(&mut self) {
        if self.merge() {
            return;
        }
        let sh = &*self.shared;
        if self.help {
            if let Some(k) = sh.claim() {
                HELPED.fetch_add(1, Relaxed);
                sh.hash(k);
                return;
            }
        } else if sh.claimed.load(Relaxed) < sh.committed.load(Relaxed) && sh.idle.iter().all(|i| i.load(Relaxed)) {
            // Waiting on segments nobody has taken, every thread asleep.
            WAKES_PROGRAM.fetch_add(1, Relaxed);
            sh.wake_one();
        }
        std::hint::spin_loop();
    }

    /// The free bytes of the current half, after waiting (hashing) for
    /// room. Requires the last commit to have been whole segments.
    fn space(&mut self) -> &mut [u8] {
        let (half, segs) = (self.half(), self.shared.segs);
        let end = match self.layout {
            Layout::Halves => {
                if self.write % half == 0 {
                    // Entering a half: everything it held must be merged.
                    while self.merged + half < self.write {
                        self.wait_step();
                    }
                }
                (self.write / half + 1) * half
            }
            Layout::Ring => {
                // A free segment: the one it held is merged.
                while self.merged + segs <= self.write {
                    self.wait_step();
                }
                (self.merged + segs).min((self.write / segs + 1) * segs)
            }
        };
        let s = self.shared.slot(self.write);
        let len = (end - self.write) as usize * SEG;
        unsafe { std::slice::from_raw_parts_mut(self.shared.data[s * SEG].get(), len) }
    }

    /// `n` bytes written at the start of `space()`: whole segments, or a
    /// message's last bytes (then `finish` comes next).
    fn commit(&mut self, n: usize) {
        let t = clocks::now();
        let segs = n.div_ceil(SEG) as u64;
        for i in 0..segs {
            let s = self.shared.slot(self.write + i);
            unsafe {
                *self.shared.offset[s].get() = (self.write + i - self.start) * SEG as u64;
                *self.shared.len[s].get() = SEG.min(n - i as usize * SEG);
            }
        }
        self.write += segs;
        self.bytes += n as u64;
        self.shared.committed.store(self.write, Release);
        PUBLISH_NS.fetch_add(clocks::since_ns(t), Relaxed);
        let waiting = self.write - self.shared.claimed.load(Relaxed);
        if waiting > 0 && (self.wake == Wake::Every || waiting >= self.half() / 2) {
            WAKES_PROGRAM.fetch_add(1, Relaxed);
            self.shared.wake_one();
        }
        let t = clocks::now();
        self.merge();
        MERGE_NS.fetch_add(clocks::since_ns(t), Relaxed);
    }

    /// The message's hash; the next message starts at the next segment.
    fn finish(&mut self) -> blake3_servil::Hash {
        while self.merged < self.write {
            self.wait_step();
        }
        let hash = if self.bytes <= SEG as u64 {
            let s = self.shared.slot(self.start);
            let bytes = unsafe { std::slice::from_raw_parts(self.shared.data[s * SEG].get() as *const u8, self.bytes as usize) };
            blake3_servil::hash(bytes)
        } else {
            let mut right = self.stack.pop().unwrap();
            while self.stack.len() > 1 {
                right = hazmat::merge_subtrees_non_root(&self.stack.pop().unwrap(), &right, Mode::Hash);
            }
            hazmat::merge_subtrees_root(&self.stack.pop().unwrap(), &right, Mode::Hash)
        };
        self.stack.clear();
        self.count = 0;
        self.start = self.write;
        self.bytes = 0;
        hash
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            w.join().unwrap();
        }
    }
}

/// One message from `source` through the stream, written `piece` bytes at a time.
fn through_stream(stream: &mut Stream, source: &[u8], piece: usize) -> blake3_servil::Hash {
    let mut at = 0;
    while at < source.len() {
        let t = clocks::now();
        let space = stream.space();
        WAIT_NS.fetch_add(clocks::since_ns(t), Relaxed);
        let t = clocks::now();
        let n = piece.min(space.len()).min(source.len() - at);
        space[..n].copy_from_slice(&source[at..at + n]);
        COPY_NS.fetch_add(clocks::since_ns(t), Relaxed);
        let t = clocks::now();
        stream.commit(n);
        COMMIT_NS.fetch_add(clocks::since_ns(t), Relaxed);
        at += n;
    }
    let t = clocks::now();
    let hash = stream.finish();
    FINISH_NS.fetch_add(clocks::since_ns(t), Relaxed);
    hash
}

/// The fork's queue as bench-hashes uses it: owned buffers of `piece`
/// bytes, `count` in flight, back through a channel.
struct Back(std::sync::mpsc::SyncSender<Option<Vec<u8>>>, std::sync::mpsc::SyncSender<blake3_servil::Hash>);
impl blake3_servil::PieceHandler for Back {
    type Buffer = Vec<u8>;
    fn piece_done(&mut self, buffer: Vec<u8>) {
        self.0.send(Some(buffer)).unwrap();
    }
    fn finished(&mut self, hash: blake3_servil::Hash) {
        self.1.send(hash).unwrap();
    }
}

fn main() {
    let threads = std::thread::available_parallelism().unwrap().get();
    blake3_servil::initialize_multithreaded();
    let len = 64 << 20;
    let source: Vec<u8> = (0..len).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]).collect();
    assert_eq!(blake3_servil::hash(&source), blake3_servil::hash_multithreaded(&source));
    // Correctness first, at lengths around the segments and halves.
    {
        for layout in [Layout::Halves, Layout::Ring] {
        let mut stream = Stream::new(4, threads - 1, Wake::Every, layout, true);
        for l in [0, 1, 1024, 1025, SEG - 1, SEG, SEG + 1, 3 * SEG + 7, 8 * SEG, 9 * SEG + 1, 40 * SEG + 5, len] {
            for piece in [SEG, 4 * SEG, 1 << 20] {
                assert_eq!(through_stream(&mut stream, &source[..l], piece), blake3_servil::hash(&source[..l]), "len {l}, piece {piece}");
            }
        }
        }
    }
    let rounds = 6;
    for len in [64 << 20, 1 << 20] {
    let source = &source[..len];
    let expected = blake3_servil::hash(source);
    let report = |name: &str, f: &mut dyn FnMut()| {
        let cpu = clocks::process_cpu_ns();
        let batches = clocks::measure(rounds, 200_000_000, &mut *f);
        let cpu = clocks::process_cpu_ns() - cpu;
        let calls: u64 = batches.iter().map(|b| b.calls).sum();
        let wall: u64 = batches.iter().map(|b| b.wall_ns).sum();
        let cpu_calls = calls + batches[0].calls;
        let ns_b = |ns: u64, calls: u64| (ns as u128 * 10_000 / (calls as u128 * len as u128)) as u64;
        let w = ns_b(wall, calls);
        let c = ns_b(cpu, cpu_calls);
        println!("  {name:62} wall {}.{:04} ns/B   CPU {}.{:04} ns/B ({}.{:02} CPUs)", w / 10000, w % 10000, c / 10000, c % 10000, c * 100 / w.max(1) / 100, c * 100 / w.max(1) % 100);
    };
    println!("\n{} MiB messages, one after another, {threads} CPUs; ns per byte of message (lower is better)", len >> 20);
    println!(" The whole job: each byte written into memory (a copy, standing in for a read), and hashed");
    {
        let mut whole = vec![0u8; len];
        report("write the message, then Hasher::update (one thread)", &mut || {
            whole.copy_from_slice(source);
            let mut h = blake3_servil::Hasher::new();
            h.update(&whole);
            assert_eq!(h.finalize(), expected);
        });
        report("write the message, then hash_multithreaded", &mut || {
            whole.copy_from_slice(source);
            assert_eq!(blake3_servil::hash_multithreaded(&whole), expected);
        });
    }
    {
        let mut piece = vec![0u8; 4 << 20];
        report("write 4 MiB at a time, update_multithreaded each", &mut || {
            let mut h = blake3_servil::Hasher::new();
            for p in source.chunks(4 << 20) {
                piece[..p.len()].copy_from_slice(p);
                h.update_multithreaded(&piece[..p.len()]);
            }
            assert_eq!(h.finalize(), expected);
        });
    }
    {
        let (pieces, back) = std::sync::mpsc::sync_channel(64);
        let (hashes, digest) = std::sync::mpsc::sync_channel(4);
        let queue = blake3_servil::Queue::pieces(blake3_servil::Mode::Hash, Back(pieces, hashes));
        let mut free: Vec<Vec<u8>> = (0..16).map(|_| Vec::with_capacity(SEG)).collect();
        report("write 64 KiB buffers, Queue::pieces, 16 in flight (now)", &mut || {
            for p in source.chunks(SEG) {
                while free.is_empty() {
                    free.push(back.recv().unwrap().unwrap());
                }
                let mut b = free.pop().unwrap();
                b.clear();
                b.extend_from_slice(p);
                queue.submit(b);
            }
            queue.finish();
            assert_eq!(digest.recv().unwrap(), expected);
            while free.len() < 16 {
                free.push(back.recv().unwrap().unwrap());
            }
        });
    }
    for (layout, half_segs) in [(Layout::Halves, 16), (Layout::Halves, 32), (Layout::Halves, 128), (Layout::Ring, 32)] {
        let mut stream = Stream::new(half_segs, threads - 1, Wake::Half, layout, false);
        let name = format!("write 64 KiB at a time into the owned buffer, {layout:?}, 2 x {} MiB", half_segs * 64 >> 10);
        report(&name, &mut || assert_eq!(through_stream(&mut stream, source, SEG), expected));
    }
    {
        let mut stream = Stream::new(32, threads - 1, Wake::Every, Layout::Halves, false);
        report("  the same, Halves, 2 x 2 MiB, waking a thread at every commit", &mut || assert_eq!(through_stream(&mut stream, source, SEG), expected));
    }
    println!(" The hashing alone: the data already in place, nothing written");
    report("hash_multithreaded on the message in memory", &mut || assert_eq!(blake3_servil::hash_multithreaded(source), expected));
    {
        let mut stream = Stream::new(32, threads - 1, Wake::Half, Layout::Halves, false);
        // Fill the buffer once; the commits then hash whatever it holds.
        report("owned buffer, Halves, 2 x 2 MiB, commits without writing", &mut || {
            let mut left = len;
            while left > 0 {
                let n = stream.space().len().min(SEG).min(left);
                stream.commit(n);
                left -= n;
            }
            std::hint::black_box(stream.finish());
        });
    }
    }
    eprintln!("host_lab: {}", clocks::load::describe(&clocks::load::windows()));
}
