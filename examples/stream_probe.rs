//! `stream` (messages borrowed, hashed in place) against `Queue::messages`
//! (owned buffers cycled back through a channel, bench-hashes' pattern):
//! the same messages, already in memory, in each; neither copies.
//!
//!   stream_probe samples API OUT [LEN...]
//!       (API oneshot-job | oneshot-stream, with -many for batches of LEN
//!       64-byte messages: one call a message or batch; collection-each | collection-stream: LEN 0 git objects, 1
//!       Nix files, hash_each_multithreaded_with against a stream;
//!       stream-mixed | queue-mixed: lengths from 1 byte to LEN;
//!       batch-queue | batch-stream: LEN is a batch's 64-byte messages,
//!       Queue::fixed against a stream of each batch's messages; written: each message read straight into the stream's bytes
//!       (submit_with); rounds: each round of messages read into the program's
//!       buffers, then a stream of them; queue-read: each read into a free
//!       buffer, then Queue::messages, as bench-hashes' ContinuousMessages)
//!       API stream | queue | new (lone messages only: a stream a message),
//!       each length (default 64 B to 1 MiB), one bench-hashes samples v4
//!       file for `bench-hashes compare`, with three scenarios:
//!       solo   one program's messages back to back (a sample: 1024
//!              messages, 64 at 1 MiB; ns per byte)
//!       two    two programs' at once, each as solo (both threads' samples)
//!       lone   one message after a 1 ms idle gap, its digest awaited (a
//!              sample: 25 messages; ns per message)
//!
//! The file's header records the CPUs busy during each cell (clocks::threads).
use blake3_servil::{Hash, Stream};
use std::sync::mpsc;

/// Places for a stream that runs on: chunks of `k` digests, each taken
/// for a submission and put back as the stream hands it back.
struct Free<'a>(std::cell::RefCell<Vec<&'a mut [Hash]>>);

impl<'a> Free<'a> {
    fn new(places: &'a mut [Hash], k: usize) -> Self {
        Free(std::cell::RefCell::new(places.chunks_mut(k).collect()))
    }

    /// The stream's handler: a place back, its digests looked at.
    fn back(&self, d: &'a mut [Hash]) {
        std::hint::black_box(d.first());
        self.0.borrow_mut().push(d);
    }

    /// A free place, collecting until one is.
    fn take<D: FnMut(&'a mut [Hash])>(&self, s: &mut Stream<'a, D>) -> &'a mut [Hash] {
        loop {
            if let Some(p) = self.0.borrow_mut().pop() {
                return p;
            }
            s.collect();
        }
    }
}

fn zeros(n: usize) -> Vec<Hash> {
    vec![Hash::from_bytes([0; 32]); n]
}

const PER_CALL: usize = 1024;
const LENS: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1 << 20];

struct Back(mpsc::SyncSender<Vec<u8>>);
impl blake3_servil::MessageHandler for Back {
    type Buffer = Vec<u8>;
    fn hashed(&mut self, buffer: Vec<u8>, hash: blake3_servil::Hash) {
        std::hint::black_box(hash);
        self.0.send(buffer).unwrap();
    }
}

/// A sample: wall ns, units (bytes or messages), start ms.
type Sample = (u64, u64, u64);

/// One mode's batches at messages of `len` bytes (1 MiB: 64 a batch).
fn solo(api: &str, len: usize) -> Vec<Sample> {
    let per_call = if len >= 1 << 20 { 64 } else { PER_CALL };
    // API-mixed: lengths spread from 1 byte to LEN, a fixed sequence.
    let (api, mixed) = match api.strip_suffix("-mixed") {
        Some(api) => (api, true),
        None => (api, false),
    };
    let length = |m: usize| if mixed { (m * 2_654_435_761 % len) + 1 } else { len };
    let messages: Vec<Vec<u8>> = (0..per_call).map(|m| (0..length(m)).map(|i| ((i + m) % 251) as u8).collect()).collect();
    let bytes: usize = messages.iter().map(|m| m.len()).sum();
    let batches = match api {
        "stream" => {
            let mut places = zeros(2 * per_call);
            let free = Free::new(&mut places, 1);
            blake3_servil::stream(
                blake3_servil::Mode::Hash,
                |d| free.back(d),
                |s| {
                    let b = clocks::measure(64, 2_000_000, || {
                        for m in &messages {
                            let p = free.take(s);
                            s.submit(m, &mut p[0]);
                        }
                        s.collect();
                    });
                    s.finish();
                    b
                },
            )
        }
        "written" => {
            // Each message read straight into the stream's bytes.
            let mut places = zeros(2 * per_call);
            let free = Free::new(&mut places, 1);
            blake3_servil::stream(
                blake3_servil::Mode::Hash,
                |d| free.back(d),
                |s| {
                    let b = clocks::measure(64, 2_000_000, || {
                        for m in &messages {
                            let p = free.take(s);
                            s.submit_with(m.len(), |at, room| room.copy_from_slice(&m[at..][..room.len()]), &mut p[0]);
                        }
                        s.collect();
                    });
                    s.finish();
                    b
                },
            )
        }
        "rounds" => {
            // A read of each message into the program's buffers, then a
            // stream of them: a buffer is refilled only after its scope.
            let mut buffers = messages.clone();
            let mut places = zeros(per_call);
            clocks::measure(64, 2_000_000, || {
                for (b, m) in buffers.iter_mut().zip(&messages) {
                    b.copy_from_slice(m);
                }
                blake3_servil::stream(blake3_servil::Mode::Hash, |d| { std::hint::black_box(d); }, |s| {
                    for (b, p) in buffers.iter().zip(&mut places) {
                        s.submit(b, p);
                    }
                    s.finish();
                });
            })
        }
        "queue-read" => {
            // As bench-hashes' ContinuousMessages: each message read into a
            // free buffer, then submitted.
            let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(per_call);
            let queue = blake3_servil::Queue::messages(blake3_servil::Mode::Hash, Back(tx));
            let mut free = messages.clone();
            let b = clocks::measure(64, 2_000_000, || {
                for m in &messages {
                    let mut buf = match free.pop() {
                        Some(b) => b,
                        None => rx.recv().unwrap(),
                    };
                    buf.copy_from_slice(m);
                    queue.submit(buf);
                }
            });
            for _ in free.len()..per_call {
                rx.recv().unwrap();
            }
            b
        }
        "queue" => {
            let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(per_call);
            let queue = blake3_servil::Queue::messages(blake3_servil::Mode::Hash, Back(tx));
            let mut free = messages;
            let b = clocks::measure(64, 2_000_000, || {
                for _ in 0..per_call {
                    queue.submit(match free.pop() {
                        Some(b) => b,
                        None => rx.recv().unwrap(),
                    });
                }
            });
            for _ in free.len()..per_call {
                rx.recv().unwrap();
            }
            b
        }
        _ => panic!("solo and two: stream | queue"),
    };
    batches.iter().map(|b| (b.wall_ns, b.calls * bytes as u64, b.started_ns / 1_000_000)).collect()
}

struct BatchBack(mpsc::SyncSender<(Vec<u8>, Vec<[u8; 32]>)>);
impl blake3_servil::FixedHandler for BatchBack {
    type Buffer = Vec<u8>;
    type Digests = Vec<[u8; 32]>;
    fn hashed(&mut self, buffer: Vec<u8>, digests: Vec<[u8; 32]>) {
        self.0.send((buffer, digests)).unwrap();
    }
}

/// Batches of `n` 64-byte messages, already in memory, 4 MiB of them a
/// sample: Queue::fixed (each batch's buffer and digest space cycled back),
/// or a stream's submit_batch, its digests into the batch's digest space.
fn batches(api: &str, n: usize) -> Vec<Sample> {
    let len = n * 64;
    let per_call = (4 << 20) / len;
    let buffers: Vec<Vec<u8>> = (0..per_call).map(|b| (0..len).map(|i| ((i + b) % 251) as u8).collect()).collect();
    let batches = match api {
        "batch-queue" => {
            let (tx, rx) = mpsc::sync_channel(per_call);
            let queue = blake3_servil::Queue::fixed(64, blake3_servil::Mode::Hash, BatchBack(tx));
            let mut free: Vec<(Vec<u8>, Vec<[u8; 32]>)> = buffers.into_iter().map(|b| (b, vec![[0u8; 32]; n])).collect();
            // Each digest copied out as it comes back, as the stream's are.
            let total = per_call * n;
            let mask = total - 1;
            assert!(total.is_power_of_two());
            let mut kept = vec![[0u8; 32]; total];
            let mut k = 0;
            let b = clocks::measure(64, 2_000_000, || {
                for _ in 0..per_call {
                    let (buffer, digests) = match free.pop() {
                        Some(p) => p,
                        None => {
                            let (buffer, digests) = rx.recv().unwrap();
                            for d in &digests {
                                kept[k & mask] = *d;
                                k += 1;
                            }
                            (buffer, digests)
                        }
                    };
                    queue.submit(buffer, digests);
                }
            });
            std::hint::black_box(&kept);
            for _ in free.len()..per_call {
                rx.recv().unwrap();
            }
            b
        }
        "batch-written" => {
            // Each batch copied into the stream's ring as it is read; its
            // digests into a place of the batch's size, cycled.
            let mut places = zeros(2 * per_call * n);
            let free = Free::new(&mut places, n);
            blake3_servil::stream(
                blake3_servil::Mode::Hash,
                |d| free.back(d),
                |s| {
                    let b = clocks::measure(64, 2_000_000, || {
                        for buffer in &buffers {
                            let p = free.take(s);
                            s.submit_batch_with(64, |at, room| room.copy_from_slice(&buffer[at..][..room.len()]), p);
                        }
                        s.collect();
                    });
                    s.finish();
                    b
                },
            )
        }
        "batch-stream" => {
            // Each batch's digests straight into a place of the batch's
            // size, cycled (Queue::fixed's digest space, as batch-queue).
            let mut places = zeros(2 * per_call * n);
            let free = Free::new(&mut places, n);
            blake3_servil::stream(
                blake3_servil::Mode::Hash,
                |d| free.back(d),
                |s| {
                    let b = clocks::measure(64, 2_000_000, || {
                        for buffer in &buffers {
                            let p = free.take(s);
                            s.submit_batch(buffer, 64, p);
                        }
                        s.collect();
                    });
                    s.finish();
                    b
                },
            )
        }
        _ => panic!("batches: batch-queue | batch-stream"),
    };
    batches.iter().map(|b| (b.wall_ns, b.calls * (per_call * len) as u64, b.started_ns / 1_000_000)).collect()
}

/// bench-hashes' collections (Collection): (count, length) classes.
const COLLECTIONS: [&[(usize, usize)]; 2] = [
    &[
        (1, 24),
        (13, 48),
        (12, 96),
        (42, 192),
        (182, 384),
        (237, 768),
        (135, 1536),
        (141, 3072),
        (173, 6144),
        (447, 12288),
        (321, 24576),
        (231, 49152),
        (84, 98304),
        (22, 196608),
        (4, 393216),
        (3, 786432),
    ],
    &[
        (6, 1),
        (1, 6),
        (1, 12),
        (4, 24),
        (11, 48),
        (44, 96),
        (139, 192),
        (133, 384),
        (439, 768),
        (428, 1536),
        (264, 3072),
        (179, 6144),
        (130, 12288),
        (96, 24576),
        (70, 49152),
        (46, 98304),
        (31, 196608),
        (17, 393216),
        (9, 786432),
    ],
];

/// Collection `index` (0 git objects, 1 Nix files) in memory, every item
/// hashed in one call: hash_each_multithreaded_with, or a stream of them.
fn collection(api: &str, index: usize) -> Vec<Sample> {
    let lengths: Vec<usize> = COLLECTIONS[index].iter().flat_map(|&(n, len)| std::iter::repeat_n(len, n)).collect();
    let count = lengths.len();
    let items: Vec<Vec<u8>> = (0..count).map(|j| (0..lengths[j * 389 % count]).map(|i| ((i + j) % 251) as u8).collect()).collect();
    let slices: Vec<&[u8]> = items.iter().map(|m| m.as_slice()).collect();
    let bytes: usize = lengths.iter().sum();
    let mut digests = vec![[0u8; 32]; count];
    let batches = match api {
        "collection-each" => clocks::measure(16, 2_000_000, || blake3_servil::hash_each_multithreaded_with(blake3_servil::Mode::Hash, &slices, &mut digests)),
        "collection-stream" => {
            let mut places = zeros(count);
            clocks::measure(16, 2_000_000, || {
                blake3_servil::stream(blake3_servil::Mode::Hash, |_| {}, |s| {
                    for (item, p) in slices.iter().zip(&mut places) {
                        s.submit(item, p);
                    }
                    s.finish();
                });
            })
        }
        _ => panic!("collection-each | collection-stream"),
    };
    std::hint::black_box(&digests);
    batches.iter().map(|b| (b.wall_ns, b.calls * bytes as u64, b.started_ns / 1_000_000)).collect()
}

/// One call per message (LEN bytes) or per batch (LEN 64-byte messages,
/// the API ending -many), back to back: the pool's one-shot calls
/// (hash_multithreaded, hash_many_multithreaded) against a stream of one
/// message or one batch.
fn oneshot(api: &str, len: usize) -> Vec<Sample> {
    let many = api.ends_with("-many");
    let bytes = if many { len * 64 } else { len };
    let input: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    let mut digests = vec![[0u8; 32]; if many { len } else { 1 }];
    let batches = match api {
        "oneshot-job" => clocks::measure(64, 2_000_000, || {
            digests[0] = *blake3_servil::hash_multithreaded(&input).as_bytes();
        }),
        "oneshot-job-many" => clocks::measure(64, 2_000_000, || blake3_servil::hash_many_multithreaded(&input, 64, &mut digests)),
        "oneshot-stream" | "oneshot-stream-many" => {
            let mut places = zeros(digests.len());
            clocks::measure(64, 2_000_000, || {
                blake3_servil::stream(blake3_servil::Mode::Hash, |_| {}, |s| {
                    if many {
                        s.submit_batch(&input, 64, &mut places);
                    } else {
                        s.submit(&input, &mut places[0]);
                    }
                    s.finish();
                });
            })
        }
        // One message from a reader (an in-memory one: each read a copy).
        "oneshot-reader-stream" => {
            let mut place = Hash::from_bytes([0; 32]);
            clocks::measure(64, 2_000_000, || {
                blake3_servil::stream(blake3_servil::Mode::Hash, |_| {}, |s| {
                    s.submit_reader(&input[..], None, &mut place).unwrap();
                    s.finish();
                });
            })
        }
        "oneshot-reader-hasher" => clocks::measure(64, 2_000_000, || {
            digests[0] = *blake3_servil::Hasher::new().update_reader(&input[..]).unwrap().finalize().as_bytes();
        }),
        _ => panic!("oneshot-job | oneshot-stream | oneshot-reader-stream | oneshot-reader-hasher, -many"),
    };
    std::hint::black_box(&digests);
    batches.iter().map(|b| (b.wall_ns, b.calls * bytes as u64, b.started_ns / 1_000_000)).collect()
}

/// One message at a time after a 1 ms idle gap, its digest awaited: a
/// server's request, a wait, the next.
fn lone(api: &str, len: usize) -> Vec<Sample> {
    let message: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let (batches, calls) = (20, 25);
    let mut unit = ();
    let mut gaps = |f: &mut dyn FnMut()| -> Vec<Sample> {
        (0..batches)
            .map(|_| {
                let started = clocks::now();
                let b = clocks::measure_after_gaps_prepared(calls, clocks::Gap::Idle, 1_000_000, &mut unit, |_| {}, |_| f());
                (b.calls.wall_ns, calls, clocks::since_ns(started) / 1_000_000)
            })
            .collect()
    };
    match api {
        "stream" => {
            let mut places = zeros(2);
            let free = Free::new(&mut places, 1);
            blake3_servil::stream(
                blake3_servil::Mode::Hash,
                |d| free.back(d),
                |s| {
                    let s = std::cell::RefCell::new(s);
                    gaps(&mut || {
                        let mut s = s.borrow_mut();
                        let p = free.take(&mut s);
                        s.submit(&message, &mut p[0]);
                        s.finish();
                    })
                },
            )
        }
        "new" => {
            let mut place = Hash::from_bytes([0; 32]);
            gaps(&mut || {
                blake3_servil::stream(blake3_servil::Mode::Hash, |d| { std::hint::black_box(d); }, |s| {
                    s.submit(&message, &mut place);
                    s.finish();
                })
            })
        }
        "queue" => {
            let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
            let queue = blake3_servil::Queue::messages(blake3_servil::Mode::Hash, Back(tx));
            let mut free = Some(message.clone());
            gaps(&mut || {
                queue.submit(free.take().unwrap());
                free = Some(rx.recv().unwrap());
            })
        }
        _ => panic!("lone: stream | queue | new"),
    }
}

fn main() {
    blake3_servil::initialize_multithreaded();
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args[0], "samples", "stream_probe samples API OUT [LEN...]");
    let (api, out) = (args[1].as_str(), &args[2]);
    let lens: Vec<usize> = if args.len() > 3 { args[3..].iter().map(|a| a.parse().unwrap()).collect() } else { LENS.to_vec() };
    let scenarios: &[&str] = match api {
        "new" => &["lone"],
        "rounds" | "queue-read" | "written" | "batch-queue" | "batch-stream" | "batch-written" => &["solo", "two"],
        mixed if mixed.ends_with("-mixed") || mixed.starts_with("collection") || mixed.starts_with("oneshot") => &["solo", "two"],
        _ => &["solo", "two", "lone"],
    };
    let (mut rows, mut busy) = (String::new(), String::new());
    for &scenario in scenarios {
        for &len in &lens {
            let before = clocks::threads::snapshot();
            let started = clocks::now();
            let samples = match scenario {
                "solo" if api.starts_with("oneshot") => oneshot(api, len),
                "two" if api.starts_with("oneshot") => std::thread::scope(|t| {
                    let other = t.spawn(|| oneshot(api, len));
                    let mut samples = oneshot(api, len);
                    samples.extend(other.join().unwrap());
                    samples
                }),
                "solo" if api.starts_with("collection") => collection(api, len),
                "two" if api.starts_with("collection") => std::thread::scope(|t| {
                    let other = t.spawn(|| collection(api, len));
                    let mut samples = collection(api, len);
                    samples.extend(other.join().unwrap());
                    samples
                }),
                "solo" if api.starts_with("batch") => batches(api, len),
                "two" if api.starts_with("batch") => std::thread::scope(|t| {
                    let other = t.spawn(|| batches(api, len));
                    let mut samples = batches(api, len);
                    samples.extend(other.join().unwrap());
                    samples
                }),
                "solo" => solo(api, len),
                "two" => std::thread::scope(|t| {
                    let other = t.spawn(|| solo(api, len));
                    let mut samples = solo(api, len);
                    samples.extend(other.join().unwrap());
                    samples
                }),
                _ => lone(api, len),
            };
            let wall = clocks::since_ns(started);
            if let (Some(b), Some(a)) = (before, clocks::threads::snapshot()) {
                let total: u64 = clocks::threads::since(&b, &a).iter().map(|x| x.cpu_ns).sum();
                busy += &format!("# cpus busy, {scenario} {len} B: {}.{:02}\n", total / wall, total % wall * 100 / wall);
            }
            let unit = if scenario == "lone" { "message" } else { "B" };
            let ns: Vec<String> = samples.iter().map(|(w, u, _)| format!("{w}/{u}")).collect();
            let starts: Vec<String> = samples.iter().map(|(_, _, s)| s.to_string()).collect();
            rows += &format!("probe\t{scenario}\tMessages\t{len} B\t{unit}\t{}\t{}\n", ns.join(","), starts.join(","));
        }
    }
    let load = clocks::load::describe(&clocks::load::windows());
    std::fs::write(out, format!("# bench-hashes samples v4\n# probe: stream_probe {api}\n# load: {load}\n# power: not reported by this probe\n{busy}contender\tscenario\tuse_case\tpoint\tunit\tns/units\tstart ms\n{rows}")).unwrap();
}
