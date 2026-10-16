//! Queue overhead probe, 64-byte messages: where a message's time goes.
//! Modes (one per process; clocks::measure 2 ms batches; clocks::summary mean):
//!   batch    hash_many over 1024 messages (the hashing alone)
//!   channel  sync_channel send+recv of a buffer per message, one thread
//!   queue    Queue::messages, 1024 kept buffers cycled through a
//!            sync_channel as bench-hashes' owned-buffer cell does
//! Prints ns per message with load qualification, and for the queue each
//! thread's CPU time over the wall time (clocks::threads, Linux).
use blake3_servil::{Hash, MessageHandler, Mode, Queue};
use std::sync::mpsc;

const MESSAGES: usize = 1024;

struct Back(mpsc::SyncSender<Vec<u8>>);
impl MessageHandler for Back {
    type Buffer = Vec<u8>;
    fn hashed(&mut self, buffer: Vec<u8>, hash: Hash) {
        std::hint::black_box(hash);
        self.0.send(buffer).unwrap();
    }
}

fn main() {
    blake3_servil::initialize_multithreaded();
    let mode = std::env::args().nth(1).expect("batch | channel | queue");
    let input: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let batches = match mode.as_str() {
        "batch" => {
            let many: Vec<u8> = input.iter().copied().cycle().take(64 * MESSAGES).collect();
            let mut out = vec![[0u8; 32]; MESSAGES];
            clocks::measure(64, 2_000_000, || {
                blake3_servil::hash_many(std::hint::black_box(&many), 64, &mut out);
                std::hint::black_box(&out);
            })
        }
        "channel" => {
            let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(MESSAGES);
            let mut buf = Some(vec![0u8; 64]);
            clocks::measure(64, 2_000_000, || {
                for _ in 0..MESSAGES {
                    let mut b = buf.take().unwrap();
                    b.copy_from_slice(std::hint::black_box(&input));
                    tx.send(b).unwrap();
                    buf = Some(rx.recv().unwrap());
                }
            })
        }
        "queue" => {
            let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(MESSAGES);
            let queue = Queue::messages(Mode::Hash, Back(tx));
            let mut free: Vec<Vec<u8>> = (0..MESSAGES).map(|_| vec![1u8; 64]).collect();
            // Warm: one full cycle outside timing.
            for b in free.drain(..) {
                queue.submit(b);
            }
            for _ in 0..MESSAGES {
                free.push(rx.recv().unwrap());
            }
            // Steady state: buffers stay in flight across calls; a free
            // buffer comes back from the handler when none is kept.
            let before = clocks::threads::snapshot();
            let t0 = clocks::now();
            let batches = clocks::measure(64, 2_000_000, || {
                for _ in 0..MESSAGES {
                    let mut b = match free.pop() {
                        Some(b) => b,
                        None => rx.recv().unwrap(),
                    };
                    b.clear();
                    b.extend_from_slice(std::hint::black_box(&input));
                    queue.submit(b);
                }
            });
            let wall = clocks::since_ns(t0);
            if let (Some(before), Some(after)) = (&before, clocks::threads::snapshot()) {
                eprintln!("CPU time over wall time: {}", clocks::threads::describe(&clocks::threads::since(before, &after), wall));
            }
            let mut back = 0;
            while free.len() + back < MESSAGES {
                rx.recv().unwrap();
                back += 1;
            }
            batches
        }
        _ => panic!("batch | channel | queue"),
    };
    let mean = clocks::summary::mean(batches.iter().map(|b| (b.wall_ns, b.calls * MESSAGES as u64)));
    let ns = (mean * 1000 + (1 << 63)) >> 64; // thousandths of a ns
    let windows = clocks::load::windows();
    println!("{mode}: {}.{:03} ns/message; {}", ns / 1000, ns % 1000, clocks::load::describe(&windows));
}
