//! probe/b3sum-stream: the stream prototype (probe/owned-buffer, job 1196)
//! for b3sum's reads: a buffer this module owns (two halves of 2 MiB,
//! 64 KiB segments), read into in place, hashed by a pool of its own as
//! each read is committed. Plain mode only.

use blake3::hazmat::{self, HasherExt, Mode};
use std::cell::UnsafeCell;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Arc, OnceLock};

const SEG: usize = 64 * 1024;
const HALF_SEGS: u64 = 32;
/// Each read asks for this much (16 segments), so hashing starts while the rest is read.
const READ: usize = 16 * SEG;

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
        let mut h = blake3::Hasher::new();
        h.set_input_offset(offset);
        h.update(bytes);
        unsafe { *self.cvs[s].get() = h.finalize_non_root() };
        self.done[s].store(k + 1, Release);
    }
}

struct Pool {
    stream: Arc<Shared>,
    idle: Box<[AtomicBool]>,
    threads: OnceLock<Vec<std::thread::Thread>>,
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
    fn worker(&self, me: usize) {
        loop {
            if let Some(k) = self.stream.claim() {
                if self.stream.waiting() {
                    self.wake_one();
                }
                self.stream.hash(k);
                continue;
            }
            self.idle[me].store(true, SeqCst);
            if self.stream.waiting() {
                let _ = self.idle[me].compare_exchange(true, false, AcqRel, Relaxed);
                continue;
            }
            std::thread::park();
            self.idle[me].store(false, Relaxed);
        }
    }
}

/// The process's one stream (b3sum hashes one input at a time), made at
/// its first use and kept.
fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let segs = 2 * HALF_SEGS;
        let stream = Arc::new(Shared {
            data: (0..segs as usize * SEG).map(|_| UnsafeCell::new(1)).collect(),
            segs,
            committed: AtomicU64::new(0),
            claimed: AtomicU64::new(0),
            offset: (0..segs).map(|_| UnsafeCell::new(0)).collect(),
            len: (0..segs).map(|_| UnsafeCell::new(0)).collect(),
            done: (0..segs).map(|_| AtomicU64::new(0)).collect(),
            cvs: (0..segs).map(|_| UnsafeCell::new([0; 32])).collect(),
        });
        let n = std::thread::available_parallelism().map_or(1, |n| n.get()).max(2) - 1;
        let pool: &'static Pool = Box::leak(Box::new(Pool { stream, idle: (0..n).map(|_| AtomicBool::new(false)).collect(), threads: OnceLock::new() }));
        let threads = (0..n).map(|i| std::thread::spawn(move || pool.worker(i)).thread().clone()).collect();
        pool.threads.set(threads).unwrap();
        pool
    })
}

/// Where the process's stream stands, kept from input to input.
struct Writer {
    write: u64,
    merged: u64,
}

static WRITER: std::sync::Mutex<Writer> = std::sync::Mutex::new(Writer { write: 0, merged: 0 });

/// Hash all of `reader`, read straight into the stream's buffer.
pub fn hash_reader(mut reader: impl Read) -> io::Result<blake3::OutputReader> {
    let pool = pool();
    let sh = &*pool.stream;
    let mut w = WRITER.lock().unwrap();
    let start = w.write;
    let mut bytes = 0u64;
    let mut stack: Vec<[u8; 32]> = Vec::new();
    let mut count = 0u64;
    let merge = |w: &mut Writer, stack: &mut Vec<[u8; 32]>, count: &mut u64| {
        let mut moved = false;
        while w.merged < sh.committed.load(Relaxed) && sh.done[sh.slot(w.merged)].load(Acquire) == w.merged + 1 {
            let cv = unsafe { *sh.cvs[sh.slot(w.merged)].get() };
            while stack.len() > count.count_ones() as usize {
                let right = stack.pop().unwrap();
                let left = stack.pop().unwrap();
                stack.push(hazmat::merge_subtrees_non_root(&left, &right, Mode::Hash));
            }
            stack.push(cv);
            *count += 1;
            w.merged += 1;
            moved = true;
        }
        moved
    };
    let wait = |w: &mut Writer, stack: &mut Vec<[u8; 32]>, count: &mut u64| {
        if !merge(w, stack, count) {
            if sh.waiting() && pool.idle.iter().all(|i| i.load(Relaxed)) {
                pool.wake_one();
            }
            std::thread::yield_now();
        }
    };
    loop {
        if w.write % HALF_SEGS == 0 {
            while w.merged + HALF_SEGS < w.write {
                wait(&mut w, &mut stack, &mut count);
            }
        }
        let end = (w.write / HALF_SEGS + 1) * HALF_SEGS;
        let s = sh.slot(w.write);
        let room = ((end - w.write) as usize * SEG).min(READ);
        // Sound: these segments' last hashing is merged, so no thread reads them.
        let space = unsafe { std::slice::from_raw_parts_mut(sh.data[s * SEG].get(), room) };
        let n = fill(&mut reader, space)?;
        let segs = n.div_ceil(SEG) as u64;
        for i in 0..segs {
            let s = sh.slot(w.write + i);
            unsafe {
                *sh.offset[s].get() = (w.write + i - start) * SEG as u64;
                *sh.len[s].get() = SEG.min(n - i as usize * SEG);
            }
        }
        w.write += segs;
        bytes += n as u64;
        sh.committed.store(w.write, Release);
        if w.write - sh.claimed.load(Relaxed) >= HALF_SEGS / 2 {
            pool.wake_one();
        }
        merge(&mut w, &mut stack, &mut count);
        if n < room {
            break;
        }
    }
    while w.merged < w.write {
        wait(&mut w, &mut stack, &mut count);
    }
    if bytes <= SEG as u64 {
        let s = sh.slot(start);
        let data = unsafe { std::slice::from_raw_parts(sh.data[s * SEG].get() as *const u8, bytes as usize) };
        return Ok(blake3::Hasher::new().update(data).finalize_xof());
    }
    let mut right = stack.pop().unwrap();
    while stack.len() > 1 {
        right = hazmat::merge_subtrees_non_root(&stack.pop().unwrap(), &right, Mode::Hash);
    }
    Ok(hazmat::merge_subtrees_root_xof(&stack.pop().unwrap(), &right, Mode::Hash))
}

/// Read from `reader` until `buffer` is full or the input ends; how much was read.
fn fill(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}
