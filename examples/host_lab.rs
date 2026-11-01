//! Probe 3 (branch probe/mac-load): can this account read every other
//! process's CPU time on macOS (proc_pid_rusage), as clocks::load would
//! need to count other programs' system time? Lists every pid, reads each,
//! and counts what it could read, by owner (this user or not); then one
//! second of every readable process's system time beside the machine's.
unsafe extern "C" {
    fn proc_listallpids(buffer: *mut i32, size: i32) -> i32;
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn proc_pidinfo(pid: i32, flavor: i32, arg: u64, buffer: *mut u8, size: i32) -> i32;
    fn getuid() -> u32;
    fn mach_host_self() -> u32;
    fn host_statistics(host: u32, flavor: i32, info: *mut u32, count: *mut u32) -> i32;
    fn mach_timebase_info(info: *mut [u32; 2]) -> i32;
}

fn pids() -> Vec<i32> {
    let mut buf = vec![0i32; 8192];
    let n = unsafe { proc_listallpids(buf.as_mut_ptr(), (buf.len() * 4) as i32) };
    assert!(n > 0, "proc_listallpids");
    buf.truncate(n as usize);
    buf
}

/// (user, system) mach time units, or the errno.
fn rusage(pid: i32) -> Result<(u64, u64), i32> {
    let mut w = [0u64; 64];
    if unsafe { proc_pid_rusage(pid, 0, w.as_mut_ptr()) } == 0 { Ok((w[2], w[3])) } else { Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)) }
}

/// The owner's uid (PROC_PIDTBSDINFO's pbi_uid at byte offset 20), if readable.
fn owner(pid: i32) -> Option<u32> {
    let mut b = [0u8; 136];
    let n = unsafe { proc_pidinfo(pid, 3, 0, b.as_mut_ptr(), b.len() as i32) };
    (n as usize >= 24).then(|| u32::from_ne_bytes(b[20..24].try_into().unwrap()))
}

fn machine_system_ticks() -> u64 {
    let mut t = [0u32; 4];
    let mut n = 4u32;
    assert_eq!(unsafe { host_statistics(mach_host_self(), 3, t.as_mut_ptr(), &mut n) }, 0);
    u64::from(t[1])
}

fn main() {
    let me = unsafe { getuid() };
    let (mut ok_mine, mut ok_other, mut err_mine, mut err_other, mut err_unknown) = (0, 0, 0, 0, 0);
    let mut errnos = std::collections::BTreeMap::new();
    let list = pids();
    for &pid in &list {
        let own = owner(pid);
        match (rusage(pid), own) {
            (Ok(_), Some(u)) if u == me => ok_mine += 1,
            (Ok(_), _) => ok_other += 1,
            (Err(e), Some(u)) if u == me => { err_mine += 1; *errnos.entry(e).or_insert(0) += 1 }
            (Err(e), Some(_)) => { err_other += 1; *errnos.entry(e).or_insert(0) += 1 }
            (Err(e), None) => { err_unknown += 1; *errnos.entry(e).or_insert(0) += 1 }
        }
    }
    println!("# probe/mac-load 3: uid {me}, {} pids: read {ok_mine} of this user's and {ok_other} of others'; failed {err_mine} of this user's, {err_other} of others', {err_unknown} of unknown owner; errnos {errnos:?}", list.len());
    let mut tb = [0u32; 2];
    unsafe { mach_timebase_info(&mut tb) };
    let ns = |t: u64| t * u64::from(tb[0]) / u64::from(tb[1]);
    let sum = || pids().iter().filter_map(|&p| rusage(p).ok()).map(|(_, s)| s).sum::<u64>();
    for w in 0..3 {
        let (s0, m0) = (sum(), machine_system_ticks());
        let t = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let (s1, m1) = (sum(), machine_system_ticks());
        let wall = t.elapsed().as_nanos() as u64;
        println!("window {w}: readable processes' system time {} milli-CPUs; the machine's {}", ns(s1.saturating_sub(s0)) * 1000 / wall, (m1 - m0) * 10_000_000 * 1000 / wall);
    }
}
