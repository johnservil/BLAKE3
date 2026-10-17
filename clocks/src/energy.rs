//! The energy this process has used, as the OS estimates it, beside what
//! its caller waits for (wall time) and the cycles its threads ran (CPU
//! time).
//!
//! **What it counts.** macOS: `proc_pid_rusage(RUSAGE_INFO_V6)`'s
//! `ri_energy_nj` for this process, all its threads, exited ones included,
//! in nanojoules; `ri_penergy_nj` is the P-cores' share. The kernel's
//! estimate from each core's activity, not a power meter: whether it
//! includes the SME unit's and the memory's energy is what
//! `examples/energy_probe.rs` asks. Elsewhere `None`, and results say so.
//! Linux's figure, the whole package's (RAPL), is another measurement.

/// This process's energy so far, and the P-cores' share of it, in
/// nanojoules: `None` where the platform gives no such estimate.
pub fn process_nj() -> Option<(u64, u64)> {
    imp::process_nj()
}

#[cfg(target_vendor = "apple")]
mod imp {
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
        fn getpid() -> i32;
    }
    const RUSAGE_INFO_V6: i32 = 6;
    // rusage_info_v6 as u64 words (<sys/resource.h>; clocks::child's other
    // fields): ri_energy_nj 42, ri_penergy_nj 43.
    const ENERGY_NJ: usize = 42;
    const PENERGY_NJ: usize = 43;

    pub fn process_nj() -> Option<(u64, u64)> {
        let mut words = [0u64; 64];
        // Sound: `words` is writable and longer than rusage_info_v6.
        let rc = unsafe { proc_pid_rusage(getpid(), RUSAGE_INFO_V6, words.as_mut_ptr()) };
        (rc == 0).then(|| (words[ENERGY_NJ], words[PENERGY_NJ]))
    }
}

#[cfg(not(target_vendor = "apple"))]
mod imp {
    pub fn process_nj() -> Option<(u64, u64)> {
        None
    }
}
