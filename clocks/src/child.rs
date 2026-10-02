//! Another program's run, measured as the person who starts it waits for
//! it: wall time from just before its start to its exit, with the counts
//! the operating system keeps for the whole process (every thread).
//!
//! - **Wall time**: [`crate::now`] before the spawn and once the process
//!   has exited (before it is reaped, on macOS, so reading its counts
//!   costs the interval nothing).
//! - **CPU time, peak memory, major page faults** (Unix): `wait4`'s
//!   `rusage` for the process alone.
//! - **Bytes read from storage**: Linux, `ru_inblock` (512-byte units);
//!   macOS, `ri_diskio_bytesread`. Reads the page cache served count
//!   nothing, so the count tells a cold read from a warm one.
//! - **Cycles and instructions per core kind** (macOS): the exited
//!   process's `proc_pid_rusage(RUSAGE_INFO_V6)`, read before it is
//!   reaped: `ri_cycles` and `ri_instructions` in all, `ri_pcycles` and
//!   `ri_pinstructions` on performance cores, `ri_user_ptime` and
//!   `ri_system_ptime` the P-cores' share of its CPU time. Elsewhere
//!   `None`, and results say so.
//!
//! Windows gives wall time alone here.

use crate::Counts;

/// One run of another program.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// When it started, as [`crate::load::now_ns`].
    pub started_ns: u64,
    pub wall_ns: u64,
    /// Whether it exited with status 0.
    pub success: bool,
    /// User and system CPU time, all its threads.
    pub cpu_ns: Option<u64>,
    pub max_rss_bytes: Option<u64>,
    pub storage_read_bytes: Option<u64>,
    pub major_faults: Option<u64>,
    pub counts: Option<Counts>,
}

/// Run `command` (its arguments and standard streams set by the caller)
/// to its exit and measure it. Panics when it cannot be started: the
/// caller named a program that is absent.
pub fn run(command: &mut std::process::Command) -> Run {
    crate::load::tick();
    let started_ns = crate::load::now_ns();
    let started = crate::now();
    let child = command.spawn().unwrap_or_else(|e| panic!("cannot start {command:?}: {e}"));
    imp::finish(child, started, started_ns)
}

/// A `struct rusage`'s user and system time, as 64-bit words.
#[cfg(unix)]
pub(crate) fn rusage_cpu_ns(words: &[i64; 18]) -> u64 {
    rusage::cpu_ns(words)
}

/// This process's own usage so far, all its threads: user and system CPU
/// time and page faults (minor: a page already in memory mapped in; major:
/// one read from storage). `None` off Unix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub user_ns: u64,
    pub system_ns: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
}

impl Usage {
    pub fn own() -> Option<Usage> {
        #[cfg(unix)]
        {
            unsafe extern "C" {
                fn getrusage(who: i32, usage: *mut i64) -> i32;
            }
            const RUSAGE_SELF: i32 = 0;
            let mut w = [0i64; 18];
            // Sound: `w` is writable and as long as a struct rusage.
            assert_eq!(unsafe { getrusage(RUSAGE_SELF, w.as_mut_ptr()) }, 0, "getrusage(RUSAGE_SELF)");
            let tv = |sec: i64, usec: i64| sec as u64 * 1_000_000_000 + u64::from(usec as u32) * 1000;
            Some(Usage { user_ns: tv(w[0], w[1]), system_ns: tv(w[2], w[3]), minor_faults: w[8] as u64, major_faults: w[9] as u64 })
        }
        #[cfg(not(unix))]
        None
    }

    /// The usage between `earlier` and this reading.
    pub fn since(self, earlier: Usage) -> Usage {
        Usage {
            user_ns: self.user_ns - earlier.user_ns,
            system_ns: self.system_ns - earlier.system_ns,
            minor_faults: self.minor_faults - earlier.minor_faults,
            major_faults: self.major_faults - earlier.major_faults,
        }
    }
}

#[cfg(unix)]
mod rusage {
    /// `struct rusage` as 64-bit words (Linux and macOS, 64-bit): two
    /// timevals, then the longs, ru_maxrss first.
    pub type Words = [i64; 18];

    unsafe extern "C" {
        fn wait4(pid: i32, status: *mut i32, options: i32, rusage: *mut i64) -> i32;
    }

    /// Reap `pid`: its exit status and its rusage.
    pub fn reap(pid: i32) -> (i32, Words) {
        let mut status = 0;
        let mut words: Words = [0; 18];
        // Sound: `status` and `words` are writable, `words` as long as a rusage.
        let rc = unsafe { wait4(pid, &mut status, 0, words.as_mut_ptr()) };
        assert_eq!(rc, pid, "wait4 on the child: {}", std::io::Error::last_os_error());
        (status, words)
    }

    /// A timeval's microseconds as nanoseconds (macOS's tv_usec is a
    /// 32-bit int in a padded word).
    fn timeval_ns(sec: i64, usec: i64) -> u64 {
        sec as u64 * 1_000_000_000 + u64::from(usec as u32) * 1000
    }

    pub(crate) fn cpu_ns(w: &Words) -> u64 {
        timeval_ns(w[0], w[1]) + timeval_ns(w[2], w[3])
    }

    pub fn max_rss_bytes(w: &Words) -> u64 {
        // Linux counts KiB, macOS bytes.
        if cfg!(target_vendor = "apple") { w[4] as u64 } else { w[4] as u64 * 1024 }
    }

    pub fn major_faults(w: &Words) -> u64 {
        w[9] as u64
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn inblock(w: &Words) -> u64 {
        w[11] as u64
    }
}

#[cfg(all(unix, not(target_vendor = "apple")))]
mod imp {
    use super::{Run, rusage};

    pub fn finish(child: std::process::Child, started: std::time::Instant, started_ns: u64) -> Run {
        let (status, words) = rusage::reap(child.id() as i32);
        let wall_ns = crate::since_ns(started);
        Run {
            started_ns,
            wall_ns,
            success: status == 0,
            cpu_ns: Some(rusage::cpu_ns(&words)),
            max_rss_bytes: Some(rusage::max_rss_bytes(&words)),
            storage_read_bytes: Some(rusage::inblock(&words) * 512),
            major_faults: Some(rusage::major_faults(&words)),
            counts: None,
        }
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    use super::{Run, rusage};
    use crate::{Counts, Level};

    unsafe extern "C" {
        fn waitid(idtype: i32, id: u32, info: *mut u8, options: i32) -> i32;
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }
    const P_PID: i32 = 1;
    const WEXITED: i32 = 0x04;
    const WNOWAIT: i32 = 0x20;
    const RUSAGE_INFO_V6: i32 = 6;
    // rusage_info_v6 as u64 words (<sys/resource.h>): ri_uuid (2 words),
    // ri_user_time 2, ri_system_time 3, ..., ri_diskio_bytesread 18, ...,
    // ri_instructions 31, ri_cycles 32, ..., ri_user_ptime 38,
    // ri_system_ptime 39, ri_pinstructions 40, ri_pcycles 41.
    const USER_TIME: usize = 2;
    const SYSTEM_TIME: usize = 3;
    const DISKIO_BYTESREAD: usize = 18;
    const INSTRUCTIONS: usize = 31;
    const CYCLES: usize = 32;
    const USER_PTIME: usize = 38;
    const SYSTEM_PTIME: usize = 39;
    const PINSTRUCTIONS: usize = 40;
    const PCYCLES: usize = 41;

    pub fn finish(child: std::process::Child, started: std::time::Instant, started_ns: u64) -> Run {
        let pid = child.id() as i32;
        // Wait for the exit, leaving the process to read before it is reaped.
        let mut info = [0u8; 128];
        // Sound: `info` is writable and larger than a siginfo_t.
        let rc = unsafe { waitid(P_PID, pid as u32, info.as_mut_ptr(), WEXITED | WNOWAIT) };
        assert_eq!(rc, 0, "waitid on the child: {}", std::io::Error::last_os_error());
        let wall_ns = crate::since_ns(started);
        let mut words = [0u64; 64];
        // Sound: `words` is writable and longer than rusage_info_v6.
        let usage = (unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V6, words.as_mut_ptr()) } == 0).then_some(words);
        let (status, rusage_words) = rusage::reap(pid);
        let ns = crate::imp::mach_to_ns;
        let counts = usage.map(|w| {
            let (time, ptime) = (w[USER_TIME] + w[SYSTEM_TIME], w[USER_PTIME] + w[SYSTEM_PTIME]);
            Counts {
                p: Level { cycles: w[PCYCLES], instructions: w[PINSTRUCTIONS], time_ns: ns(ptime) },
                e: Level { cycles: w[CYCLES] - w[PCYCLES], instructions: w[INSTRUCTIONS] - w[PINSTRUCTIONS], time_ns: ns(time - ptime) },
            }
        });
        Run {
            started_ns,
            wall_ns,
            success: status == 0,
            cpu_ns: Some(rusage::cpu_ns(&rusage_words)),
            max_rss_bytes: Some(rusage::max_rss_bytes(&rusage_words)),
            storage_read_bytes: usage.map(|w| w[DISKIO_BYTESREAD]),
            major_faults: Some(rusage::major_faults(&rusage_words)),
            counts,
        }
    }
}

#[cfg(not(unix))]
mod imp {
    use super::Run;

    pub fn finish(mut child: std::process::Child, started: std::time::Instant, started_ns: u64) -> Run {
        let status = child.wait().expect("wait on the child");
        let wall_ns = crate::since_ns(started);
        Run { started_ns, wall_ns, success: status.success(), cpu_ns: None, max_rss_bytes: None, storage_read_bytes: None, major_faults: None, counts: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shell that spins briefly: it succeeds, its wall time covers its
    /// CPU time, and a failing one reports failure.
    #[test]
    #[cfg(unix)]
    fn a_child_is_timed_and_counted() {
        let ok = run(std::process::Command::new("sh").args(["-c", "i=0; while [ $i -lt 20000 ]; do i=$((i+1)); done"]));
        assert!(ok.success, "{ok:?}");
        let cpu = ok.cpu_ns.expect("unix counts CPU time");
        assert!(cpu > 0 && cpu <= ok.wall_ns + 20_000_000, "{ok:?}");
        assert!(ok.max_rss_bytes.unwrap() > 100_000, "{ok:?}");
        if let Some(counts) = ok.counts {
            let ns = counts.p.time_ns + counts.e.time_ns;
            assert!(ns * 2 >= cpu && ns <= cpu * 2, "the per-kind times add up to the CPU time: {ok:?}");
        }
        assert!(!run(std::process::Command::new("sh").args(["-c", "exit 3"])).success);
    }

    /// A reaped child's CPU time counts as this process's own, so the
    /// load it puts on the machine is never read as other programs'.
    #[test]
    #[cfg(unix)]
    fn a_reaped_childs_cpu_time_is_this_processs() {
        let before = crate::process_cpu_ns();
        let child = run(std::process::Command::new("sh").args(["-c", "i=0; while [ $i -lt 50000 ]; do i=$((i+1)); done"]));
        let after = crate::process_cpu_ns();
        assert!(after - before >= child.cpu_ns.unwrap(), "{before} {after} {child:?}");
    }
}
