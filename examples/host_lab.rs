//! probe/lazy-sync: which allocations happen after initialize_multithreaded
//! returns, as the pool's threads settle (first sleeps), and as a queue
//! starts? Each one is printed with its thread and backtrace.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);
thread_local! { static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
static REPORT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn trace(size: usize) {
    if ARMED.load(Ordering::SeqCst) && !INSIDE.with(|i| i.replace(true)) {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        let frames: Vec<&str> = bt.lines().filter(|l| l.contains("blake3_servil") || l.contains("std::sys") || l.contains("host_lab")).take(14).collect();
        let s = format!("ALLOC {size} B on {:?}\n{}\n", std::thread::current().name(), frames.join("\n"));
        REPORT.lock().unwrap().push_str(&s);
        INSIDE.with(|i| i.set(false));
    }
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        trace(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        trace(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

struct Back(std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>);
impl blake3_servil::MessageHandler for Back {
    type Buffer = Vec<u8>;
    fn hashed(&mut self, buffer: Vec<u8>, hash: blake3_servil::Hash) {
        std::hint::black_box(hash);
        self.0.lock().unwrap().push(buffer);
    }
}

fn main() {
    let mut out = String::new();
    blake3_servil::initialize_multithreaded();
    ARMED.store(true, Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(20));
    ARMED.store(false, Ordering::SeqCst);
    out += &format!("after initialize_multithreaded, 20 ms of settling: {} allocations\n", ALLOCATIONS.swap(0, Ordering::SeqCst));
    out += &std::mem::take(&mut *REPORT.lock().unwrap());
    // A queue's first rounds, then later rounds (the test's shape: 64 B, Time).
    let back = std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(5)));
    let queue = blake3_servil::Queue::messages(blake3_servil::Mode::Hash, blake3_servil::Efficiency::Time, Back(back.clone()));
    for _ in 0..4 { back.lock().unwrap().push(vec![7u8; 64]); }
    let round = |n: usize| for _ in 0..n {
        let b = loop { if let Some(b) = back.lock().unwrap().pop() { break b; } std::hint::spin_loop(); };
        queue.submit(b);
    };
    round(50);
    ARMED.store(true, Ordering::SeqCst);
    round(500);
    std::thread::sleep(std::time::Duration::from_millis(20));
    round(500);
    ARMED.store(false, Ordering::SeqCst);
    out += &format!("a queue of 64 B messages after 50 warm-up rounds, 1000 more (a 20 ms pause between): {} allocations\n", ALLOCATIONS.load(Ordering::SeqCst));
    out += &std::mem::take(&mut *REPORT.lock().unwrap());
    print!("{out}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(out.as_bytes()).unwrap();
}
