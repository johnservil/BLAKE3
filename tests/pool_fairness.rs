//! Two programs calling the pool at once get fair shares of it (#213): two
//! threads hashing call after call for a second each complete about as
//! many calls. The pool serves simultaneous arrivals first come, first
//! served, whole (one finishes in its alone time, the other after it), so
//! a caller that keeps calling must not keep its place ahead of one that
//! waits: served in a fixed order, one caller completed everything while
//! the other starved. Two shapes: one message of 16 MiB (a tree job a
//! call), and a collection of 2048 items of 24 B to 768 KiB (the bench's
//! git objects: many small jobs and some trees a call). Measured on an
//! i7-12700K: 1.00-1.10 either way; starvation reads 2 and more. The
//! bound leaves room for a loaded machine, and a loaded one still fails
//! it (1.6-1.9 on a 4-vCPU guest under a mutation check, #240), so it
//! runs on purpose with the machine to itself (`cargo nextest run -P
//! machine`); the order the pool serves sources in, which makes the
//! shares fair, has its own test with no clock (lanes::sources::test).
#![cfg(feature = "std")]

use blake3_servil::Mode;
use blake3_servil::throughput::{Block, hash_each_multithreaded, hash_multithreaded};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const BOUND: f64 = 1.5;
const RUN: Duration = Duration::from_secs(1);

/// The CPU the calling thread runs on (Linux; None elsewhere): the
/// callers' cores beside a failed ratio, since on a chip with two kinds
/// of core a caller on the slower kind completes fewer calls of its own
/// work whatever the pool does (Sam Shackleton, #215).
fn current_cpu() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn sched_getcpu() -> i32;
        }
        // Sound: a plain libc call with no arguments.
        let cpu = unsafe { sched_getcpu() };
        return (cpu >= 0).then_some(cpu as usize);
    }
    #[allow(unreachable_code)]
    None
}

/// Two callers of `call`, each built from `make(who)`, free-running for RUN
/// from one moment: the more calls completed over the fewer, and the CPUs
/// each caller's thread was seen on (one in ten calls), in order of first
/// sight.
fn share<T: Send + 'static>(make: impl Fn(usize) -> T + Send + Sync + 'static, call: impl Fn(&T) + Send + Sync + 'static) -> (f64, [Vec<usize>; 2]) {
    let make = Arc::new(make);
    let call = Arc::new(call);
    let counts: Arc<[AtomicUsize; 2]> = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let go = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|who| {
            let (make, call, counts, go) = (make.clone(), call.clone(), counts.clone(), go.clone());
            std::thread::spawn(move || {
                let input = make(who);
                let mut cpus = Vec::new();
                go.wait();
                let end = Instant::now() + RUN;
                while Instant::now() < end {
                    call(&input);
                    let n = counts[who].fetch_add(1, Ordering::Relaxed);
                    if n % 10 == 0 {
                        if let Some(cpu) = current_cpu() {
                            if !cpus.contains(&cpu) {
                                cpus.push(cpu);
                            }
                        }
                    }
                }
                cpus
            })
        })
        .collect();
    let mut seen: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (who, h) in handles.into_iter().enumerate() {
        seen[who] = h.join().unwrap();
    }
    let (a, b) = (counts[0].load(Ordering::Relaxed), counts[1].load(Ordering::Relaxed));
    assert!(a.min(b) > 0, "each caller completed some calls: {a} and {b}");
    (a.max(b) as f64 / a.min(b) as f64, seen)
}

fn blocks(len: usize, seed: u8) -> Vec<Block> {
    (0..len.div_ceil(64).max(1)).map(|i| Block([(i as u8).wrapping_mul(31).wrapping_add(seed); 64])).collect()
}

#[test]
fn two_callers_share_the_pool() {
    blake3_servil::initialize_multithreaded(None);
    if std::thread::available_parallelism().map_or(1, |n| n.get()) < 4 {
        return; // a pool of one or two: nothing to share
    }
    if std::env::var_os("BLAKE3_EMULATED").is_some() {
        // Under an emulator (CI's QEMU Cortex-A72) the threads' speeds are the
        // emulator's scheduling, not the pool's: 1.71 and 2.00 there (#213).
        return;
    }
    let len = 16 << 20;
    let (messages, cpus) = share(
        move |who| blocks(len, who as u8),
        move |b| {
            std::hint::black_box(hash_multithreaded(Mode::Plain, b, len));
        },
    );
    assert!(
        messages <= BOUND,
        "16 MiB messages: one caller completed {messages:.2} times the other's calls; the callers ran on CPUs {:?} and {:?}",
        cpus[0],
        cpus[1]
    );

    const CLASSES: &[(usize, usize)] = &[
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
    ];
    let lengths: Vec<usize> = CLASSES.iter().flat_map(|&(n, l)| std::iter::repeat_n(l, n)).collect();
    let order: Arc<Vec<usize>> = Arc::new((0..2048).map(|j| lengths[j * 389 % 2048]).collect());
    let o = order.clone();
    let (collections, cpus) = share(
        move |who| o.iter().enumerate().map(|(i, &l)| (blocks(l, (i + who) as u8), l)).collect::<Vec<_>>(),
        |items: &Vec<(Vec<Block>, usize)>| {
            let refs: Vec<(&[Block], usize)> = items.iter().map(|(b, l)| (&b[..], *l)).collect();
            let mut out = vec![[0u8; 32]; refs.len()];
            hash_each_multithreaded(Mode::Plain, &refs, &mut out);
            std::hint::black_box(&out);
        },
    );
    assert!(
        collections <= BOUND,
        "collections: one caller completed {collections:.2} times the other's calls; the callers ran on CPUs {:?} and {:?}",
        cpus[0],
        cpus[1]
    );
}
