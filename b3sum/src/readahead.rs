//! Many files at once (Linux): b3sum keeps up to DEPTH files read from
//! storage in flight across its file threads, where it kept one a thread.
//!
//! Each small file (a regular file reporting under `SMALL` bytes) is
//! opened, its reported size taken as a hint for the buffer's first length,
//! then read with preadv2(RWF_NOWAIT), which succeeds only from the page
//! cache: a cached file is read and hashed at once. A file that misses is
//! advised (posix_fadvise WILLNEED, which starts its reads from storage and
//! returns) and set aside; once the thread holds its batch of such files,
//! it reads and hashes each in turn, by which time the reads have arrived
//! or are on the way. Every file is read to its end, whatever size it
//! reports (/proc reports 0 bytes, /sys 4096, and a file may grow while it
//! is read): the size affects how b3sum reads, never what it hashes.
//! Larger files and other kinds of file go to `hash_path`, as before.
//!
//! Measured (bench-hashes-validation/b3sum-uring): from storage, 1000 files
//! of 16 KiB with one thread took 0.27 of today's time, with 20 threads 0.86
//! (there the drive's random reads set the pace, about 110,000 a second);
//! cached files cost what they did. Tried and dropped: io_uring with 4 reads
//! in flight a thread (as fast as the hint with a ring and a dependency), a
//! stream a thread with Stream::submit_reader (slower cached: x1.18-1.35),
//! O_DIRECT for files that miss (x1.27-1.85 from storage: one read a thread
//! at a time, and no readahead), advising every file without RWF_NOWAIT
//! first (WILLNEED walks a cached file's pages: x1.07-1.17 for one thread).

use crate::Args;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Files below this go through the batch; larger ones are mapped or read as
/// before (`hash_path`; MAP_LEN).
const SMALL: u64 = crate::MAP_LEN as u64;

/// Files read from storage in flight across the file threads. From
/// storage, 1000 files of 16 KiB with one thread took x0.51 of today's
/// time at 4 in flight, x0.34 at 16, x0.27 at 64; with 20 threads, 2 to 16
/// a thread took alike. At most DEPTH files below SMALL wait in memory
/// (32 MiB at most), or MIN_BATCH a thread when there are more threads.
const DEPTH: usize = 64;

/// Files that missed the cache a thread holds at least.
const MIN_BATCH: usize = 4;

/// Whether batches serve this run: several files, none of them stdin.
pub(crate) fn applies(args: &Args) -> bool {
    args.file_args.len() > 1 && args.file_args.iter().all(|p| p != Path::new("-"))
}

/// Hash every file of the run, printing in order: the files that failed.
pub(crate) fn hash_inputs(args: &Args) -> u64 {
    let threads = crate::threads().get().min(args.file_args.len());
    let batch = DEPTH.div_ceil(threads).max(MIN_BATCH);
    grow_descriptor_table(threads * batch + 64);
    crate::hash_inputs_with(args, threads, |args: &Args, next: &AtomicUsize, done: &mut dyn FnMut(usize, anyhow::Result<blake3::OutputReader>) -> bool| hash_some(args, next, done, batch), crate::print_output)
}

/// Room for `descriptors` open descriptors, made while this process has
/// one thread. The kernel's table starts at 64 and doubles as descriptors
/// need it; in a process whose threads share it, each growth waits for an
/// RCU grace period (expand_fdtable's synchronize_rcu): 10-13 ms a growth,
/// measured (openat of descriptor 64 and 128, i7-12700K, Linux 7.0). With
/// one thread, the kernel frees the old table at once. The table never
/// shrinks, so the room lasts the run.
fn grow_descriptor_table(descriptors: usize) {
    let target = descriptors.next_power_of_two() as i32 - 1;
    // Sound: plain descriptor calls on descriptors this function owns.
    unsafe {
        let fd = libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
        if fd >= 0 {
            if libc::dup2(fd, target) == target {
                libc::close(target);
            }
            libc::close(fd);
        }
    }
}

/// A file that missed the cache: its index, the open file, and its bytes
/// so far (`at` of them, in a buffer with room for its reported size and
/// one byte more).
struct Missed {
    index: usize,
    file: std::fs::File,
    bytes: Vec<u8>,
    at: usize,
}

/// Hash the files whose turn `next` gives, holding up to `batch` that missed
/// the cache, giving each result to `done` (false once nobody listens).
fn hash_some(args: &Args, next: &AtomicUsize, done: &mut dyn FnMut(usize, anyhow::Result<blake3::OutputReader>) -> bool, batch: usize) {
    let paths = &args.file_args;
    let mut buffer = Vec::new();
    let mut more = true;
    while more {
        let mut missed = Vec::with_capacity(batch);
        while missed.len() < batch {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= paths.len() {
                more = false;
                break;
            }
            let small = std::fs::File::open(&paths[i]).and_then(|file| Ok((file.metadata()?, file)));
            let (size, file) = match small {
                Ok((opened, file)) if opened.is_file() && opened.len() < SMALL => (opened.len() as usize, file),
                // Not for the batch (or an error, which hash_path reports).
                _ => {
                    if !done(i, crate::hash_path(args, &paths[i])) {
                        return;
                    }
                    continue;
                }
            };
            // Room for the size and one byte more; grown, never cleared: the
            // bytes past what a read returns are never hashed.
            if buffer.len() < size + 1 {
                buffer.resize(size + 1, 0);
            }
            let output = match read_to_end(&file, &mut buffer, 0, true) {
                Ok((n, true)) => Ok(hashed(args, &buffer[..n])),
                Ok((at, false)) => {
                    // Sound: advice on an open descriptor.
                    unsafe { libc::posix_fadvise(file.as_raw_fd(), at as i64, 0, libc::POSIX_FADV_WILLNEED) };
                    // What was read, and room for the rest (the size and one
                    // byte more, or more where the content already ran past).
                    let mut bytes = buffer[..at].to_vec();
                    bytes.resize(at.max(size) + 1, 0);
                    missed.push(Missed { index: i, file, bytes, at });
                    continue;
                }
                Err(e) => Err(e.into()),
            };
            if !done(i, output) {
                return;
            }
        }
        for Missed { index, file, mut bytes, at } in missed {
            let output = match read_to_end(&file, &mut bytes, at, false) {
                Ok((n, _)) => Ok(hashed(args, &bytes[..n])),
                Err(e) => Err(e.into()),
            };
            if !done(index, output) {
                return;
            }
        }
    }
}

/// Read `file` from byte `at` to its end into `buffer`, growing it in place
/// when the content runs past it: the size a file reports is a hint for the
/// buffer's first length (its size and one byte more), never for how much
/// is read. With
/// `cached_only`, preadv2's RWF_NOWAIT reads only what the page cache holds.
/// The bytes read and whether the end was reached, which only a read of 0
/// bytes says (false: a cached-only read stopped short, to be finished by a
/// plain read). The size never decides where a file ends: a file appended
/// to while it is read is read to its end.
fn read_to_end(file: &std::fs::File, buffer: &mut Vec<u8>, mut at: usize, cached_only: bool) -> std::io::Result<(usize, bool)> {
    let flags = if cached_only { libc::RWF_NOWAIT } else { 0 };
    loop {
        if at == buffer.len() {
            buffer.resize((buffer.len() * 2).max(4096), 0);
        }
        let iov = libc::iovec { iov_base: buffer[at..].as_mut_ptr().cast(), iov_len: buffer.len() - at };
        // Sound: one iovec over the buffer's unread part.
        let n = unsafe { libc::preadv2(file.as_raw_fd(), &iov, 1, at as i64, flags) };
        if n == 0 {
            return Ok((at, true));
        }
        if n < 0 {
            let e = std::io::Error::last_os_error();
            match e.kind() {
                std::io::ErrorKind::Interrupted => continue,
                // A read that would wait for storage, or a filesystem without
                // RWF_NOWAIT (/proc, some network ones): read it plainly,
                // which reports any real error.
                _ if cached_only => return Ok((at, false)),
                _ => return Err(e),
            }
        }
        at += n as usize;
    }
}

/// `bytes`, a file's content, hashed.
fn hashed(args: &Args, bytes: &[u8]) -> blake3::OutputReader {
    let mut hasher = args.base_hasher.clone();
    hasher.update(bytes);
    let mut output = hasher.finalize_xof();
    output.set_position(args.seek());
    output
}

#[cfg(test)]
mod test {
    /// The reported size is a hint: a buffer smaller than the content (a
    /// file that grew, or one that reports less) still reads all of it, and
    /// a larger one stops at the end; from any starting offset; cached-only
    /// then plain, as the batch reads a file that misses.
    #[test]
    fn read_to_end_reads_all_whatever_the_hint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        let content: Vec<u8> = (0..100_000u32).map(|i| (i * 7) as u8).collect();
        std::fs::write(&path, &content).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        for hint in [0, 1, 4095, 4096, 99_999, 100_000, 100_001, 300_000] {
            for cached_only in [false, true] {
                let mut buffer = vec![0u8; hint + 1];
                let (mut n, end) = super::read_to_end(&file, &mut buffer, 0, cached_only).unwrap();
                if !end {
                    // Cached-only stopped short (here, a filesystem without
                    // RWF_NOWAIT): a plain read finishes it, as the batch does.
                    assert!(cached_only, "a plain read reads to the end");
                    n = super::read_to_end(&file, &mut buffer, n, false).unwrap().0;
                }
                assert_eq!(&buffer[..n], &content[..], "hint {hint}, cached_only {cached_only}");
            }
        }
        let mut buffer = vec![0u8; 10];
        buffer[..3].copy_from_slice(&content[..3]);
        let (n, _) = super::read_to_end(&file, &mut buffer, 3, false).unwrap();
        assert_eq!(&buffer[..n], &content[..], "from an offset");
    }
}
