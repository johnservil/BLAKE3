//! probe/owned-buffer: several streams at once, each a buffer BLAKE3 owns
//! (two halves of 2 MiB, 64 KiB segments), sharing one pool of hashing
//! threads; against writing each message whole and calling
//! hash_multithreaded, and against Hasher::update on the writer's thread.
//! Each stream has a writer thread of its own; every byte is written (a
//! copy, standing in for a read) and hashed, and each hash is stored in
//! the writer's slot for its message.

use blake3_servil::hazmat::{self, HasherExt, Mode};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, OnceLock};

const SEG: usize = 64 * 1024;
const HALF_SEGS: u64 = 32;

/// One stream's buffer and its hashing state, shared with the pool.
struct Shared {
    data: Box<[UnsafeCell<u8>]>,
    segs: u64,
    committed: AtomicU64,
    claimed: AtomicU64,
    offset: Box<[UnsafeCell<u64>]>,
    len: Box<[UnsafeCell<usize>]>,
    done: Box<[AtomicU64]>,
    cvs: Box<[UnsafeCell<[u8; 32]>]>,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    fn slot(&self, k: u64) -> usize {
        (k % self.segs) as usize
    }
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
    fn waiting(&self) -> bool {
        self.claimed.load(Relaxed) < self.committed.load(Relaxed)
    }
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
}

/// The hashing threads every stream shares; they sleep when no stream has
/// a committed segment unclaimed.
struct Pool {
    /// Every stream, fixed before the measurements (the prototype's
    /// simplification: no lock on the workers' path).
    streams: OnceLock<Vec<Arc<Shared>>>,
    idle: Box<[AtomicBool]>,
    threads: OnceLock<Vec<std::thread::Thread>>,
    next: AtomicUsize,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().unwrap().get() - 1;
        let pool: &'static Pool = Box::leak(Box::new(Pool { streams: OnceLock::new(), idle: (0..n).map(|_| AtomicBool::new(false)).collect(), threads: OnceLock::new(), next: AtomicUsize::new(0) }));
        let threads = (0..n).map(|i| std::thread::spawn(move || pool.worker(i)).thread().clone()).collect();
        pool.threads.set(threads).unwrap();
        pool
    })
}

impl Pool {
    fn wake_one(&self) {
        for (i, idle) in self.idle.iter().enumerate() {
            if idle.load(Relaxed) && idle.compare_exchange(true, false, AcqRel, Relaxed).is_ok() {
                self.threads.get().unwrap()[i].unpark();
                return;
            }
        }
    }
    fn any_waiting(&self) -> bool {
        self.streams.get().map_or(false, |all| all.iter().any(|s| s.waiting()))
    }
    /// A committed, unclaimed segment from some stream, the streams taken
    /// in turn.
    fn take(&self) -> Option<(Arc<Shared>, u64)> {
        let streams = self.streams.get()?;
        let n = streams.len();
        let start = self.next.fetch_add(1, Relaxed);
        (0..n).find_map(|i| {
            let s = &streams[(start + i) % n];
            s.claim().map(|k| (s.clone(), k))
        })
    }
    fn worker(&self, me: usize) {
        loop {
            if let Some((s, k)) = self.take() {
                if self.any_waiting() {
                    self.wake_one();
                }
                s.hash(k);
                continue;
            }
            self.idle[me].store(true, SeqCst);
            if self.any_waiting() {
                let _ = self.idle[me].compare_exchange(true, false, AcqRel, Relaxed);
                continue;
            }
            std::thread::park();
            self.idle[me].store(false, Relaxed);
        }
    }
}

/// The writer's side of one stream.
struct Stream {
    shared: Arc<Shared>,
    write: u64,
    merged: u64,
    start: u64,
    bytes: u64,
    stack: Vec<[u8; 32]>,
    count: u64,
}

impl Stream {
    fn new() -> Stream {
        let segs = 2 * HALF_SEGS;
        let shared = Arc::new(Shared {
            data: (0..segs as usize * SEG).map(|_| UnsafeCell::new(1)).collect(),
            segs,
            committed: AtomicU64::new(0),
            claimed: AtomicU64::new(0),
            offset: (0..segs).map(|_| UnsafeCell::new(0)).collect(),
            len: (0..segs).map(|_| UnsafeCell::new(0)).collect(),
            done: (0..segs).map(|_| AtomicU64::new(0)).collect(),
            cvs: (0..segs).map(|_| UnsafeCell::new([0; 32])).collect(),
        });
        Stream { shared, write: 0, merged: 0, start: 0, bytes: 0, stack: Vec::new(), count: 0 }
    }
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
    /// Wait for hashing under way: merge what is ready, wake a thread if
    /// segments wait unclaimed and every thread sleeps, else yield.
    fn wait_step(&mut self) {
        if self.merge() {
            return;
        }
        if self.shared.waiting() && pool().idle.iter().all(|i| i.load(Relaxed)) {
            pool().wake_one();
        }
        std::thread::yield_now();
    }
    fn space(&mut self) -> &mut [u8] {
        if self.write % HALF_SEGS == 0 {
            while self.merged + HALF_SEGS < self.write {
                self.wait_step();
            }
        }
        let end = (self.write / HALF_SEGS + 1) * HALF_SEGS;
        let s = self.shared.slot(self.write);
        unsafe { std::slice::from_raw_parts_mut(self.shared.data[s * SEG].get(), (end - self.write) as usize * SEG) }
    }
    fn commit(&mut self, n: usize) {
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
        if self.write - self.shared.claimed.load(Relaxed) >= HALF_SEGS / 2 {
            pool().wake_one();
        }
        self.merge();
    }
    fn finish(&mut self) -> blake3_servil::Hash {
        while self.merged < self.write {
            self.wait_step();
        }
        let hash = if self.bytes <= SEG as u64 {
            let s = self.shared.slot(self.start);
            blake3_servil::hash(unsafe { std::slice::from_raw_parts(self.shared.data[s * SEG].get() as *const u8, self.bytes as usize) })
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

/// One message from `source` through the stream, written 64 KiB at a time.
fn through_stream(stream: &mut Stream, source: &[u8]) -> blake3_servil::Hash {
    let mut at = 0;
    while at < source.len() {
        let space = stream.space();
        let n = SEG.min(space.len()).min(source.len() - at);
        space[..n].copy_from_slice(&source[at..at + n]);
        stream.commit(n);
        at += n;
    }
    stream.finish()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum How {
    /// Each message written whole into the writer's buffer, then hash_multithreaded.
    WholeThenMt,
    /// Each 64 KiB written into the writer's buffer, then Hasher::update on its thread.
    UpdateEach,
    /// Each 64 KiB written into the stream's space, then committed.
    Stream,
}

/// `writers` threads, each hashing `messages` messages of `source` in the
/// way `how` says, each hash stored in the writer's slot. Returns the wall
/// time from the first write to the last hash stored, and the process's CPU time.
fn run(how: How, writers: usize, messages: usize, source: &Arc<Vec<u8>>, streams: &mut Vec<Option<Stream>>) -> (u64, u64) {
    let start = Arc::new(std::sync::Barrier::new(writers + 1));
    let handles: Vec<_> = (0..writers)
        .map(|w| {
            let (source, start) = (source.clone(), start.clone());
            let mut stream = streams[w].take();
            std::thread::spawn(move || {
                let mut slots = vec![[0u8; 32]; messages];
                let mut whole = if how == How::WholeThenMt { vec![1u8; source.len()] } else { vec![1u8; SEG] };
                start.wait();
                for slot in slots.iter_mut() {
                    let hash = match how {
                        How::WholeThenMt => {
                            whole.copy_from_slice(&source);
                            blake3_servil::hash_multithreaded(&whole)
                        }
                        How::UpdateEach => {
                            let mut h = blake3_servil::Hasher::new();
                            for piece in source.chunks(SEG) {
                                whole[..piece.len()].copy_from_slice(piece);
                                h.update(&whole[..piece.len()]);
                            }
                            h.finalize()
                        }
                        How::Stream => through_stream(stream.as_mut().unwrap(), &source),
                    };
                    *slot = *hash.as_bytes();
                }
                std::hint::black_box(&slots);
                stream
            })
        })
        .collect();
    let cpu = clocks::process_cpu_ns();
    let t = clocks::now();
    start.wait();
    for (w, h) in handles.into_iter().enumerate() {
        streams[w] = h.join().unwrap();
    }
    (clocks::since_ns(t), clocks::process_cpu_ns() - cpu)
}

fn main() {
    blake3_servil::initialize_multithreaded();
    pool();
    let len = 16 << 20;
    let source: Arc<Vec<u8>> = Arc::new((0..len).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]).collect());
    let max_writers = 16;
    let mut streams: Vec<Option<Stream>> = (0..max_writers + 1).map(|_| Some(Stream::new())).collect();
    pool().streams.set(streams.iter().map(|s| s.as_ref().unwrap().shared.clone()).collect()).unwrap_or_else(|_| unreachable!());
    // Correctness of the stream at lengths around segments and halves.
    {
        let mut stream = streams.pop().unwrap().unwrap();
        for l in [0, 1, 1025, SEG - 1, SEG, SEG + 1, 3 * SEG + 7, 40 * SEG + 5, 100 * SEG, len] {
            assert_eq!(through_stream(&mut stream, &source[..l]), blake3_servil::hash(&source[..l]), "len {l}");
        }
    }
    let total_bytes = 2u64 << 30; // per measurement, over all writers
    let repeats = 5;
    println!("{} MiB messages; every byte written (a copy) and hashed, each hash stored; {} CPUs", len >> 20, std::thread::available_parallelism().unwrap());
    println!("{:>8} {:<44} {:>14} {:>14} {:>10}", "writers", "how", "wall ns/B", "CPU ns/B", "CPUs busy");
    for writers in [1usize, 2, 4, 16] {
        let messages = (total_bytes / len as u64 / writers as u64).max(1) as usize;
        let bytes = (messages * writers * len) as u128;
        for how in [How::WholeThenMt, How::UpdateEach, How::Stream] {
            let mut walls = Vec::new();
            let mut cpus = Vec::new();
            run(how, writers, 1, &source, &mut streams); // warm-up
            for _ in 0..repeats {
                let (wall, cpu) = run(how, writers, messages, &source, &mut streams);
                walls.push(wall);
                cpus.push(cpu);
            }
            let per = |ns: u64| ((ns as u128 * 100_000 + bytes / 2) / bytes) as u64; // ns/B x 1e5
            let wall: Vec<String> = walls.iter().map(|&w| format!("{}.{:05}", per(w) / 100_000, per(w) % 100_000)).collect();
            let mean_wall: u64 = walls.iter().sum::<u64>() / repeats as u64;
            let mean_cpu: u64 = cpus.iter().sum::<u64>() / repeats as u64;
            let (w, c) = (per(mean_wall), per(mean_cpu));
            println!("{writers:>8} {:<44} {:>4}.{:05} {:>8}.{:05} {:>7}.{:02}   ({})", format!("{how:?}"), w / 100_000, w % 100_000, c / 100_000, c % 100_000, c * 100 / w.max(1) / 100, c * 100 / w.max(1) % 100, wall.join(" "));
        }
    }
    eprintln!("host_lab: {}", clocks::load::describe(&clocks::load::windows()));
}
