//! Other programs' load on the machine while this process measures,
//! recorded by every measurement in this crate, with nothing to call.
//!
//! **What it counts.** How many CPUs other programs kept busy over a
//! stretch of wall time, in milli-CPUs (1000: one CPU busy throughout).
//! On Linux it sums their CPU time from /proc: every process but kernel
//! threads and this one's own tree, each with the children it has waited
//! for. Kernel work no process is charged with (kernel threads, other
//! CPUs' interrupts), much of it caused by this process's own children,
//! is no other program's. On macOS, the CPU time every CPU spent busy
//! less this process's own (every thread and child). On Linux the OS
//! also counts steal time, which a hypervisor took from a VM's CPUs for
//! other work on the host, load a guest sees no other way.
//!
//! **When it reads.** A reading costs system calls, which take far longer
//! than a short timed call, so it never happens inside a timed interval
//! or per call. Each measurement in this crate calls [`tick`] outside its
//! timed intervals, between batches and before each gap; a tick reads
//! the machine only once a window of [`WINDOW_NS`] has passed since the
//! last reading, and otherwise costs one wall-clock read and one atomic
//! load. Measured in the Linux VM (`reading_cost` below, September 30,
//! 2026): a reading 8.6 us, a tick between readings 18 ns. A program
//! that times with [`crate::now`] directly calls [`tick`] itself, between
//! samples.
//!
//! **Placing samples in windows.** Every [`crate::Batch`] records when it
//! started, as [`now_ns`] (nanoseconds since this process's first
//! measurement), on the same scale as each [`Window`]'s bounds.
//!
//! **Resolution.** Both OSes count machine CPU time in ticks of 10 ms per
//! CPU. A window's reading therefore carries an error of up to a tick per
//! CPU: at most 0.16 CPUs over a 1 s window on 16 CPUs, well under
//! [`BUSY_MILLI_CPUS`].
//!
//! A window whose other load or steal reaches [`BUSY_MILLI_CPUS`] is
//! busy: when one closes, this module says so on stderr, so every probe
//! and benchmark shows it with no code of its own; [`windows`] gives
//! every window for reports and samples files.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// A window's least length: a reading ends it once this much wall time
/// has passed since the one before.
pub const WINDOW_NS: u64 = 1_000_000_000;
/// Other programs (or a hypervisor) keeping a whole CPU busy make a
/// window busy: a core taken from the shared scenario and from the
/// multithreaded contenders, which use every CPU. An M4 Max running a
/// Linux VM beside the benchmark keeps 0.40-0.56 CPUs busy, with results
/// level with a quieter run's.
pub const BUSY_MILLI_CPUS: u64 = 1000;

/// One stretch of wall time and the load other programs put on the
/// machine in it. Bounds are [`now_ns`] values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub start_ns: u64,
    pub end_ns: u64,
    /// CPUs other programs kept busy, in milli-CPUs.
    pub other_milli_cpus: u64,
    /// CPUs a hypervisor withheld, in milli-CPUs (0 where none is counted).
    pub steal_milli_cpus: u64,
}

impl Window {
    /// Whether other programs or a hypervisor took a CPU or more.
    pub fn busy(&self) -> bool {
        self.other_milli_cpus.max(self.steal_milli_cpus) >= BUSY_MILLI_CPUS
    }

    /// Whether the window holds the instant `at_ns` (a [`now_ns`] value).
    pub fn holds(&self, at_ns: u64) -> bool {
        self.start_ns <= at_ns && at_ns < self.end_ns
    }
}

/// Nanoseconds of wall time since this process's first measurement (its
/// first call of this function, [`tick`], or a measurement in this crate).
pub fn now_ns() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    crate::since_ns(*ORIGIN.get_or_init(Instant::now))
}

/// A machine reading: when, the machine's CPU times, this process's.
#[derive(Clone, Copy)]
struct Reading {
    at_ns: u64,
    machine: imp::CpuTimes,
    own_ns: u64,
}

impl Reading {
    fn take() -> Option<Reading> {
        let at_ns = now_ns();
        let machine = imp::read()?;
        Some(Reading { at_ns, machine, own_ns: crate::process_cpu_ns()? })
    }
}

/// The window between two readings.
fn between(from: Reading, to: Reading) -> Window {
    assert!(to.at_ns > from.at_ns, "a window lasts some time");
    let (busy_ns, steal_ns) = imp::busy_and_steal_ns(from.machine, to.machine);
    let milli = |ns: u64| u64::try_from(u128::from(ns) * 1000 / u128::from(to.at_ns - from.at_ns)).expect("load fits in u64");
    // Where the platform counts other programs' time itself (Linux), that
    // is the load. Elsewhere it is the machine's busy time less this
    // process's own; tick rounding can put the first a little under the
    // second, which reads as none.
    let other_ns = if imp::COUNTS_OTHER_PROGRAMS { busy_ns } else { busy_ns.saturating_sub(to.own_ns - from.own_ns) };
    Window { start_ns: from.at_ns, end_ns: to.at_ns, other_milli_cpus: milli(other_ns), steal_milli_cpus: milli(steal_ns) }
}

struct State {
    last: Option<Reading>,
    /// The reading the last closed window started at, until a tail joins it.
    last_start: Option<Reading>,
    windows: Vec<Window>,
}

static STATE: Mutex<State> = Mutex::new(State { last: None, last_start: None, windows: Vec::new() });
/// The [`now_ns`] at which the next tick reads the machine; u64::MAX where
/// the platform gives no CPU times.
static DUE_NS: AtomicU64 = AtomicU64::new(0);

/// Close the running window if `least_ns` have passed since the last
/// reading (take the first reading when there is none).
fn close(state: &mut State, least_ns: u64) {
    let Some(reading) = Reading::take() else {
        DUE_NS.store(u64::MAX, Ordering::Relaxed);
        return;
    };
    match state.last {
        Some(last) if reading.at_ns - last.at_ns < least_ns => return,
        Some(last) => {
            push(state, between(last, reading));
            state.last_start = Some(last);
        }
        None => {}
    }
    state.last = Some(reading);
    DUE_NS.store(reading.at_ns + WINDOW_NS, Ordering::Relaxed);
}

/// Add a window, reporting it on stderr when busy.
fn push(state: &mut State, window: Window) {
    if window.busy() {
        eprintln!("clocks: {}; measurements in that time may read slower than this machine runs", describe_window(&window));
    }
    state.windows.push(window);
}

/// Join the final tail using the last window's original counter baseline.
fn join_tail(state: &mut State, start: Reading, reading: Reading) {
    if state.windows.last().expect("a tail joins a closed window").busy() {
        // Keep this observation: a quiet tail could average it below the
        // busy threshold. The run already has no speed verdict.
        return;
    }
    state.windows.pop();
    push(state, between(start, reading));
    state.last = Some(reading);
}

/// How long until the next tick reads the machine, in ns (0: now; u64::MAX
/// where the platform gives no CPU times).
pub fn until_due_ns() -> u64 {
    DUE_NS.load(Ordering::Relaxed).saturating_sub(now_ns())
}

/// Read the machine once a window has passed since the last reading;
/// otherwise nothing. Call it outside timed intervals: every measurement
/// in this crate does. Another thread's tick in progress makes this one
/// return at once.
pub fn tick() {
    if now_ns() < DUE_NS.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut state) = STATE.try_lock() {
        close(&mut state, WINDOW_NS);
    }
}

/// Every window so far, in order, the running one closed first when it
/// has lasted half a window or more; a shorter one joins the last closed
/// quiet window, read again from that window's start, so the run's last samples
/// lie in a window too (once: a tail joins a window that no tail has
/// joined, so repeated calls never dilute a burst into a longer average).
/// An observed busy window stays intact; the run already has no speed verdict.
/// Empty where the platform gives no CPU times, or before a window has
/// closed (a run under half a second).
pub fn windows() -> Vec<Window> {
    let mut state = STATE.lock().expect("load state lock");
    if DUE_NS.load(Ordering::Relaxed) != u64::MAX {
        let closed = state.windows.len();
        close(&mut state, WINDOW_NS / 2);
        if state.windows.len() == closed {
            if let (Some(start), Some(reading)) = (state.last_start.take(), Reading::take()) {
                join_tail(&mut state, start, reading);
            }
        }
    }
    state.windows.clone()
}

/// Other programs' load over the next `ns` of wall time, in milli-CPUs,
/// from two readings of its own (the windows' rule, apart from their
/// state): the tests' measure of the load beside them. 0 where the
/// platform gives no CPU times.
#[cfg(all(test, unix))]
pub(crate) fn other_milli_cpus_over(ns: u64) -> u64 {
    let Some(before) = Reading::take() else { return 0 };
    std::thread::sleep(std::time::Duration::from_nanos(ns));
    let after = Reading::take().expect("CPU times read a moment ago");
    between(before, after).other_milli_cpus
}

/// The tests that use this process's windows (they tick, read, or run a
/// child, whose watcher ticks) take turns: the tests run on many threads,
/// and one's windows are another's.
#[cfg(all(test, unix))]
pub(crate) fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Milli-CPUs as "1.23".
pub fn cpus(milli: u64) -> String {
    let hundredths = (milli + 5) / 10;
    format!("{}.{:02}", hundredths / 100, hundredths % 100)
}

/// Nanoseconds as seconds with one decimal: "12.3".
fn seconds(ns: u64) -> String {
    let tenths = (ns + 50_000_000) / 100_000_000;
    format!("{}.{}", tenths / 10, tenths % 10)
}

/// "other programs kept 1.40 CPUs busy from 12.0 s to 13.0 s" (with the
/// hypervisor's share where it took any).
pub fn describe_window(window: &Window) -> String {
    let mut line =
        format!("other programs kept {} CPUs busy from {} s to {} s", cpus(window.other_milli_cpus), seconds(window.start_ns), seconds(window.end_ns));
    if window.steal_milli_cpus > 0 {
        line += &format!(", the hypervisor withheld {} CPUs", cpus(window.steal_milli_cpus));
    }
    line
}

/// One line for readers about `windows`: "quiet: other programs kept 0.02
/// CPUs busy on average, 0.10 in the busiest second", or "busy: ..., busy
/// in 2 of 40 windows (12.0-13.0 s, 20.1-21.1 s)"; for none, "not
/// measured on this platform", or where the OS counts CPU time "not
/// measured: the run lasted under half a second" (a window closes after
/// half a second at the least).
pub fn describe(windows: &[Window]) -> String {
    let Some(first) = windows.first() else {
        return if cfg!(any(target_os = "linux", target_vendor = "apple")) {
            "not measured: the run lasted under half a second"
        } else {
            "not measured on this platform"
        }
        .to_owned();
    };
    let span = windows.last().unwrap().end_ns - first.start_ns;
    let weighted = |pick: fn(&Window) -> u64| {
        let sum: u128 = windows.iter().map(|w| u128::from(pick(w)) * u128::from(w.end_ns - w.start_ns)).sum();
        u64::try_from((sum + u128::from(span) / 2) / u128::from(span)).expect("load fits in u64")
    };
    let busy: Vec<&Window> = windows.iter().filter(|w| w.busy()).collect();
    let mut line = format!(
        "{}: other programs kept {} CPUs busy on average, {} in the busiest window",
        if busy.is_empty() { "quiet" } else { "busy" },
        cpus(weighted(|w| w.other_milli_cpus)),
        cpus(windows.iter().map(|w| w.other_milli_cpus).max().unwrap())
    );
    let steal = windows.iter().map(|w| w.steal_milli_cpus).max().unwrap();
    if steal > 0 {
        line += &format!("; the hypervisor withheld {} CPUs on average, {} at most", cpus(weighted(|w| w.steal_milli_cpus)), cpus(steal));
    }
    if !busy.is_empty() {
        let list: Vec<String> = busy.iter().map(|w| format!("{}-{} s", seconds(w.start_ns), seconds(w.end_ns))).collect();
        line += &format!("; busy in {} of {} windows ({})", busy.len(), windows.len(), list.join(", "));
    }
    line
}

#[cfg(target_os = "linux")]
mod imp {
    /// Linux counts every process's CPU time, so other programs' load is
    /// summed from theirs, not inferred from the machine's busy time less
    /// ours. The inference counted, as other programs, kernel work our own
    /// children caused and no process is charged with: kernel threads
    /// cleaning up after them, and other CPUs' interrupt handling. On an
    /// EPYC 9255 official b3sum run once per file (a thread pool over 48
    /// CPUs each time) made 0.61 CPUs of it, where the sum reads 0.00, and
    /// another program hashing back to back reads 1.00 either way
    /// (October 24, 2026).
    pub const COUNTS_OTHER_PROGRAMS: bool = true;

    /// CPU times in ticks: other programs' (`busy`), every process but
    /// kernel threads, this one and its descendants, each with the time
    /// of the children it has waited for (so a short-lived program still
    /// counts, in its parent's); and the machine's time stolen by a
    /// hypervisor, all CPUs summed.
    #[derive(Clone, Copy)]
    pub struct CpuTimes {
        pub(super) busy: u64,
        pub(super) steal: u64,
    }

    fn ticks_per_second() -> u64 {
        unsafe extern "C" {
            fn sysconf(name: i32) -> i64;
        }
        const SC_CLK_TCK: i32 = 2;
        // Sound: a plain query.
        u64::try_from(unsafe { sysconf(SC_CLK_TCK) }).expect("sysconf(_SC_CLK_TCK) is positive")
    }

    pub fn read() -> Option<CpuTimes> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let fields: Vec<u64> =
            stat.lines().next()?.strip_prefix("cpu ")?.split_whitespace().map(|field| field.parse().expect("/proc/stat counts are integers")).collect();
        // user nice system idle iowait irq softirq steal (guest time is inside user).
        assert!(fields.len() >= 8, "/proc/stat's cpu line has at least eight fields");
        Some(CpuTimes { busy: other_programs_ticks()?, steal: fields[7] })
    }

    /// One process's /proc/PID/stat: (parent, whether a kernel thread,
    /// utime + stime + cutime + cstime in ticks).
    pub(super) fn parse_stat(stat: &str) -> Option<(u32, bool, u64)> {
        // The command name, in parentheses, may hold anything: fields
        // resume after its last ')'.
        let rest = stat.get(stat.rfind(')')? + 2..)?;
        let f: Vec<&str> = rest.split_whitespace().collect();
        // Fields 3.. of proc(5): state ppid ... flags(9) ... utime(14) stime cutime cstime.
        let ppid = f.get(1)?.parse().ok()?;
        const PF_KTHREAD: u64 = 0x0020_0000;
        let kernel = f.get(6)?.parse::<u64>().ok()? & PF_KTHREAD != 0;
        let ticks = f.get(11..15)?.iter().map(|t| t.parse::<u64>().ok()).sum::<Option<u64>>()?;
        Some((ppid, kernel, ticks))
    }

    /// Kernel threads seen, by pid: skipped while their pid stays listed
    /// (most of /proc on a many-CPU machine: 542 of 574 on the EPYC 9255).
    /// Pids are handed out in turn up to pid_max before any is reused, so
    /// a listed pid stays the kernel thread it was within a run.
    static KERNEL_THREADS: std::sync::Mutex<Option<std::collections::HashSet<u32>>> = std::sync::Mutex::new(None);

    /// The CPU ticks of every process but kernel threads and this one's
    /// own tree, children waited for included. A process gone between
    /// reading /proc and its stat is skipped.
    fn other_programs_ticks() -> Option<u64> {
        Some(sum_other(std::process::id(), &scan()?))
    }

    /// Whether `pid`, as /proc lists it now, counts as another program's.
    #[cfg(test)]
    pub(super) fn counts_as_other(pid: u32) -> bool {
        let procs = scan().expect("/proc");
        procs.contains_key(&pid) && !ours(std::process::id(), pid, &procs)
    }

    /// Every listed process but kernel threads: pid to (parent, false, ticks).
    fn scan() -> Option<std::collections::HashMap<u32, (u32, bool, u64)>> {
        let mut kernel = KERNEL_THREADS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let kernel = kernel.get_or_insert_with(Default::default);
        let mut listed = std::collections::HashSet::new();
        let mut procs = std::collections::HashMap::new();
        let mut path = String::new();
        let mut stat = String::new();
        for entry in std::fs::read_dir("/proc").ok()?.flatten() {
            let Some(pid) = entry.file_name().to_str().and_then(|name| name.parse::<u32>().ok()) else { continue };
            listed.insert(pid);
            if kernel.contains(&pid) {
                continue;
            }
            use std::fmt::Write as _;
            use std::io::Read as _;
            path.clear();
            write!(path, "/proc/{pid}/stat").unwrap();
            stat.clear();
            let Ok(_) = std::fs::File::open(&path).and_then(|mut file| file.read_to_string(&mut stat)) else { continue };
            match parse_stat(&stat) {
                Some((_, true, _)) => {
                    kernel.insert(pid);
                }
                Some(parsed) => {
                    procs.insert(pid, parsed);
                }
                None => {}
            }
        }
        kernel.retain(|pid| listed.contains(pid));
        Some(procs)
    }

    /// The sum over `procs` (pid: parent, kernel thread, ticks) of all but
    /// kernel threads and `me`'s tree.
    pub(super) fn sum_other(me: u32, procs: &std::collections::HashMap<u32, (u32, bool, u64)>) -> u64 {
        procs.iter().filter(|&(&pid, &(_, kernel, _))| !kernel && !ours(me, pid, procs)).map(|(_, &(_, _, ticks))| ticks).sum()
    }

    /// Whether `pid` is `me` or descends from it in `procs`.
    fn ours(me: u32, mut pid: u32, procs: &std::collections::HashMap<u32, (u32, bool, u64)>) -> bool {
        // Walk up the parents; init's parent is 0. Bounded, for a table
        // read while processes come and go.
        for _ in 0..procs.len() + 1 {
            if pid == me {
                return true;
            }
            match procs.get(&pid) {
                Some(&(parent, _, _)) if parent != 0 && parent != pid => pid = parent,
                _ => return false,
            }
        }
        false
    }

    pub fn busy_and_steal_ns(from: CpuTimes, to: CpuTimes) -> (u64, u64) {
        let ns = |ticks: u64| ticks * 1_000_000_000 / ticks_per_second();
        // Other programs' sum can fall: one exits unwaited for (its parent
        // ignores SIGCHLD), taking its time with it. That reads as none.
        (ns(to.busy.saturating_sub(from.busy)), ns(to.steal - from.steal))
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    /// The machine's busy time less this process's own.
    pub const COUNTS_OTHER_PROGRAMS: bool = false;

    /// host_statistics(HOST_CPU_LOAD_INFO): user, system, idle, nice ticks,
    /// all CPUs summed, each a 32-bit counter that wraps.
    #[derive(Clone, Copy)]
    pub struct CpuTimes(pub(super) [u32; 4]);

    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics(host: u32, flavor: i32, info: *mut u32, count: *mut u32) -> i32;
        fn sysconf(name: i32) -> i64;
    }

    pub fn read() -> Option<CpuTimes> {
        const HOST_CPU_LOAD_INFO: i32 = 3;
        let mut ticks = [0u32; 4];
        let mut count = 4u32;
        // Sound: `ticks` is writable for `count` words.
        let rc = unsafe { host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, ticks.as_mut_ptr(), &mut count) };
        (rc == 0 && count == 4).then_some(CpuTimes(ticks))
    }

    pub fn busy_and_steal_ns(from: CpuTimes, to: CpuTimes) -> (u64, u64) {
        const SC_CLK_TCK: i32 = 3;
        // Sound: a plain query.
        let per_second = u64::try_from(unsafe { sysconf(SC_CLK_TCK) }).expect("sysconf(_SC_CLK_TCK) is positive");
        // Each counter wraps on its own: difference each, then sum.
        let busy: u64 = [0, 1, 3].iter().map(|&i| u64::from(to.0[i].wrapping_sub(from.0[i]))).sum();
        (busy * 1_000_000_000 / per_second, 0)
    }
}

#[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
mod imp {
    pub const COUNTS_OTHER_PROGRAMS: bool = false;

    #[derive(Clone, Copy)]
    pub struct CpuTimes;

    pub fn read() -> Option<CpuTimes> {
        None
    }

    pub fn busy_and_steal_ns(_: CpuTimes, _: CpuTimes) -> (u64, u64) {
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start_ns: u64, end_ns: u64, other_milli_cpus: u64, steal_milli_cpus: u64) -> Window {
        Window { start_ns, end_ns, other_milli_cpus, steal_milli_cpus }
    }

    #[test]
    fn busy_starts_at_one_cpu_of_other_load_or_steal() {
        assert!(!window(0, 1, 999, 999).busy());
        assert!(window(0, 1, 1000, 0).busy());
        assert!(window(0, 1, 0, 1000).busy());
    }

    #[test]
    fn a_window_holds_its_start_and_not_its_end() {
        let w = window(10, 20, 0, 0);
        assert!(w.holds(10) && w.holds(19) && !w.holds(20) && !w.holds(9));
    }

    #[test]
    fn descriptions_weigh_windows_by_length_and_name_busy_ones() {
        let none = if cfg!(any(target_os = "linux", target_vendor = "apple")) {
            "not measured: the run lasted under half a second"
        } else {
            "not measured on this platform"
        };
        assert_eq!(describe(&[]), none);
        let quiet = [window(0, 1_000_000_000, 100, 0), window(1_000_000_000, 4_000_000_000, 500, 0)];
        assert_eq!(describe(&quiet), "quiet: other programs kept 0.40 CPUs busy on average, 0.50 in the busiest window");
        let busy = [window(0, 1_000_000_000, 1400, 0), window(1_000_000_000, 2_000_000_000, 200, 300)];
        assert_eq!(
            describe(&busy),
            "busy: other programs kept 0.80 CPUs busy on average, 1.40 in the busiest window; \
            the hypervisor withheld 0.15 CPUs on average, 0.30 at most; busy in 1 of 2 windows (0.0-1.0 s)"
        );
    }

    #[cfg(target_os = "linux")]
    fn tail_fixture(busy: u64, steal: u64, tail_busy: u64, tail_steal: u64) -> State {
        let start = Reading { at_ns: 0, machine: imp::CpuTimes { busy: 0, steal: 0 }, own_ns: 0 };
        let last = Reading { at_ns: WINDOW_NS, machine: imp::CpuTimes { busy, steal }, own_ns: 0 };
        let mut state = State { last: Some(last), last_start: None, windows: vec![between(start, last)] };
        let tail = Reading { at_ns: WINDOW_NS + 400_000_000, machine: imp::CpuTimes { busy: tail_busy, steal: tail_steal }, own_ns: 0 };
        join_tail(&mut state, start, tail);
        state
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn quiet_tail_preserves_observed_other_load() {
        let state = tail_fixture(110, 0, 110, 0); // 1.10 CPUs, then no other work.
        assert!(state.windows.iter().any(Window::busy), "a quiet tail preserves the prior busy finding");
        assert!(describe(&state.windows).starts_with("busy:"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn quiet_tail_preserves_observed_steal() {
        let state = tail_fixture(0, 110, 0, 110);
        assert!(state.windows.iter().any(Window::busy), "a quiet tail preserves the prior steal finding");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn quiet_tail_still_extends_a_quiet_window() {
        let state = tail_fixture(20, 0, 28, 0);
        assert_eq!(state.windows, vec![window(0, 1_400_000_000, 200, 0)]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn newly_busy_tail_still_marks_the_run_busy() {
        let state = tail_fixture(20, 0, 160, 0);
        assert!(state.windows[0].busy());
        assert_eq!(state.windows[0].end_ns, 1_400_000_000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_stat_line_parses_past_any_command_name() {
        // pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime cutime cstime ...
        let line = "4242 (a) b) (c) S 17 4242 4242 0 -1 4194560 100 0 0 0 7 3 20 10 20 0 1 0 5 0 0";
        assert_eq!(imp::parse_stat(line), Some((17, false, 40)));
        let kernel = "2 (kthreadd) S 0 0 0 0 -1 2129984 0 0 0 0 1 2 0 0 20 0 1 0 5 0 0";
        assert_eq!(imp::parse_stat(kernel), Some((0, true, 3)));
        assert_eq!(imp::parse_stat("7 (cut short"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn other_programs_leave_out_our_tree_and_kernel_threads() {
        let procs: std::collections::HashMap<u32, (u32, bool, u64)> = [
            (1, (0, false, 1000)),   // init, with what it has reaped
            (2, (0, true, 50)),      // kthreadd
            (30, (2, true, 70)),     // a kernel worker
            (100, (1, false, 5)),    // a shell
            (200, (100, false, 9)),  // us
            (201, (200, false, 11)), // our child
            (202, (201, false, 13)), // its child
            (300, (1, false, 17)),   // another program
            (301, (300, false, 19)), // its child
        ]
        .into_iter()
        .collect();
        assert_eq!(imp::sum_other(200, &procs), 1000 + 5 + 17 + 19);
        // A parent loop (a table read mid-change) ends the walk.
        let looped: std::collections::HashMap<u32, (u32, bool, u64)> = [(5, (6, false, 1)), (6, (5, false, 2))].into_iter().collect();
        assert_eq!(imp::sum_other(200, &looped), 3);
    }

    /// Who counts as another program, asked of /proc directly: no load is
    /// measured, so tests running beside this one cannot sway it, and its
    /// processes sleep, so it sways none of theirs.
    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_and_what_it_starts_are_ours_and_others_are_not() {
        assert!(!imp::counts_as_other(std::process::id()), "this process");
        assert!(!imp::counts_as_other(2), "kthreadd, a kernel thread");
        // A child, however started, and a grandchild under it.
        let mut child = std::process::Command::new("sh").args(["-c", "sleep 30 & wait"]).spawn().expect("sh");
        let grandchild = loop {
            let children = std::fs::read_to_string(format!("/proc/{0}/task/{0}/children", child.id())).unwrap_or_default();
            if let Some(pid) = children.split_whitespace().next() {
                break pid.parse::<u32>().expect("a pid");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert!(!imp::counts_as_other(child.id()), "a child");
        assert!(!imp::counts_as_other(grandchild), "a grandchild");
        // An orphan: its shell exits at once, so an ancestor outside this
        // process's tree (init, or a subreaper) adopts it.
        let out = std::process::Command::new("sh").args(["-c", "sleep 30 > /dev/null 2>&1 < /dev/null & echo $!"]).output().expect("sh");
        let orphan: u32 = String::from_utf8(out.stdout).unwrap().trim().parse().expect("a pid");
        let counted = imp::counts_as_other(orphan);
        for pid in [orphan, grandchild] {
            std::process::Command::new("kill").arg(pid.to_string()).status().ok();
        }
        child.wait().ok();
        assert!(counted, "a program this process did not start is another's");
    }

    /// Our own children's work, and the kernel's on their behalf, is no
    /// other program's: a child hashing on every CPU for a window leaves
    /// that window quiet.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "spawns a CPU-heavy child; run alone: cargo test --release -- --ignored own_children"]
    fn own_children_are_no_other_programs() {
        let _turn = exclusive();
        let before = Reading::take().unwrap();
        let started = std::time::Instant::now();
        while started.elapsed() < std::time::Duration::from_millis(1200) {
            let status = std::process::Command::new("sh").args(["-c", "i=0; while [ $i -lt 20000 ]; do i=$((i+1)); done"]).status().unwrap();
            assert!(status.success());
        }
        let after = Reading::take().unwrap();
        let window = between(before, after);
        assert!(window.other_milli_cpus < 200, "our children read as other programs: {window:?}");
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn apple_counters_wrap_one_by_one() {
        let from = imp::CpuTimes([u32::MAX - 9, 0, 0, 5]);
        let to = imp::CpuTimes([10, 20, 1000, 5]);
        let (busy, _) = imp::busy_and_steal_ns(from, to);
        assert_eq!(busy, 40 * 1_000_000_000 / 100);
    }

    #[cfg(any(target_os = "linux", target_vendor = "apple"))]
    #[test]
    fn ticks_close_a_window_after_a_window_of_time() {
        let _turn = exclusive();
        tick();
        let started = now_ns();
        while now_ns() - started < WINDOW_NS + WINDOW_NS / 10 {
            crate::busy_work(10_000_000);
            tick();
        }
        let ended = now_ns();
        let windows = windows();
        assert!(!windows.is_empty(), "a window closed");
        for pair in windows.windows(2) {
            assert_eq!(pair[0].end_ns, pair[1].start_ns, "windows follow one another");
        }
        assert!(windows.iter().all(|w| w.end_ns - w.start_ns >= WINDOW_NS / 2), "each window lasts half a window or more");
        // The tail under half a window joined the last window: it reaches the call.
        let last = *windows.last().unwrap();
        assert!(last.busy() || last.end_ns >= ended, "the tail is covered or the last busy finding is preserved");
        // Once: a later call joins nothing more to it.
        let tail = now_ns();
        while now_ns() - tail < WINDOW_NS / 10 {
            crate::busy_work(10_000_000);
        }
        assert_eq!(super::windows().last().copied(), Some(last), "a second call joins nothing more");
    }
}

/// What a machine reading costs (`cargo test --release -- --ignored
/// --nocapture reading_cost`): measured through this crate, printed.
#[cfg(test)]
#[test]
#[ignore]
fn reading_cost() {
    for batch in crate::measure(9, 20_000_000, || {
        std::hint::black_box(Reading::take());
    }) {
        println!("a machine reading: {}", batch.show());
    }
    for batch in crate::measure(9, 20_000_000, tick) {
        println!("a tick between readings: {}", batch.show());
    }
}
