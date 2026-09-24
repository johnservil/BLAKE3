//! Timing for probes, shared by `#[path = "support/clocks.rs"] mod clocks;`.
//!
//! Every measurement reads both clocks: wall time, which is what a caller
//! waits for, and the thread's core cycles per core kind (macOS
//! `thread_selfcounts`), which do not change with the core's clock speed
//! and do not count time the core spends waiting on the SME unit. Their
//! ratio, cycles per ns, says which state a measurement ran in: the core
//! kind's clock, and on SME2 work the unit's fast or slow state (M4 Max
//! P-core: 3.93 against 3.20-3.30). Judge speed by wall time within one
//! state and report the state mix; let cycles explain (AGENTS.md,
//! "Measuring"). Off Apple the cycle counts are absent and say so.
#![allow(dead_code)]

use std::time::Instant;

/// One batch of calls: per-call wall time and per-call core cycles on
/// P-cores and E-cores (zero where the OS gives no counts).
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub ns: f64,
    pub p_cycles: f64,
    pub e_cycles: f64,
}

impl Sample {
    /// Whether this platform counted cycles.
    pub fn has_cycles(&self) -> bool {
        self.p_cycles + self.e_cycles > 0.0
    }

    pub fn cycles(&self) -> f64 {
        self.p_cycles + self.e_cycles
    }

    /// Cycles per ns: the clock the core ran at, lower where it waited on
    /// the SME unit.
    pub fn per_ns(&self) -> f64 {
        self.cycles() / self.ns
    }

    /// Share of the cycles spent on E-cores.
    pub fn e_share(&self) -> f64 {
        if self.has_cycles() { self.e_cycles / self.cycles() } else { 0.0 }
    }

    /// "1234 ns, 4850 cyc, 3.93/ns P" (or "1234 ns, no cycle counts").
    pub fn show(&self) -> String {
        if !self.has_cycles() {
            return format!("{:.0} ns, no cycle counts", self.ns);
        }
        let kind = match self.e_share() {
            s if s < 0.05 => "P",
            s if s > 0.95 => "E",
            _ => "mixed",
        };
        format!("{:.0} ns, {:.0} cyc, {:.2}/ns {kind}", self.ns, self.cycles(), self.per_ns())
    }
}

/// The batches of one measurement, in the order taken.
pub struct Measurement {
    pub batches: Vec<Sample>,
}

impl Measurement {
    /// The batch with the lowest wall time, with its own cycles.
    pub fn fastest(&self) -> Sample {
        *self.batches.iter().min_by(|a, b| a.ns.total_cmp(&b.ns)).expect("at least one batch")
    }

    /// The batch with the median wall time, with its own cycles.
    pub fn median(&self) -> Sample {
        let mut sorted = self.batches.clone();
        sorted.sort_by(|a, b| a.ns.total_cmp(&b.ns));
        sorted[sorted.len() / 2]
    }
}

/// `batches` batches of about `batch_us` microseconds each of `f`, after a
/// warm-up that sizes the batch.
pub fn measure(batches: usize, batch_us: u128, mut f: impl FnMut()) -> Measurement {
    assert!(batches > 0 && batch_us > 0, "a measurement takes at least one batch of some length");
    let started = Instant::now();
    let mut reps = 0u64;
    while started.elapsed().as_micros() < batch_us {
        f();
        reps += 1;
    }
    let batches = (0..batches)
        .map(|_| {
            let (p0, e0) = cycles();
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            let ns = t.elapsed().as_nanos() as f64;
            let (p1, e1) = cycles();
            let per = reps as f64;
            Sample { ns: ns / per, p_cycles: (p1 - p0) as f64 / per, e_cycles: (e1 - e0) as f64 / per }
        })
        .collect();
    Measurement { batches }
}

/// macOS QoS classes: user-interactive runs on P-cores, background on E-cores.
pub const USER_INTERACTIVE: u32 = 0x21;
pub const BACKGROUND: u32 = 0x09;

#[cfg(target_vendor = "apple")]
pub fn set_qos(class: u32) {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, relpri: i32) -> i32;
    }
    assert_eq!(unsafe { pthread_set_qos_class_self_np(class, 0) }, 0, "pthread_set_qos_class_self_np");
}

#[cfg(not(target_vendor = "apple"))]
pub fn set_qos(_class: u32) {}

/// The thread's core cycles so far on P-cores and on E-cores.
#[cfg(target_vendor = "apple")]
pub fn cycles() -> (u64, u64) {
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Cpi {
        instructions: u64,
        cycles: u64,
        user: u64,
        system: u64,
    }
    unsafe extern "C" {
        fn thread_selfcounts(kind: u32, dst: *mut std::ffi::c_void, size: usize) -> i32;
    }
    let mut levels = [Cpi::default(); 2];
    assert_eq!(unsafe { thread_selfcounts(4, levels.as_mut_ptr().cast(), std::mem::size_of_val(&levels)) }, 0, "thread_selfcounts");
    (levels[0].cycles, levels[1].cycles)
}

#[cfg(not(target_vendor = "apple"))]
pub fn cycles() -> (u64, u64) {
    (0, 0)
}
