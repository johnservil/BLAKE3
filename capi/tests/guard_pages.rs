//! Every C call against guard pages: each input, output, key, context and
//! hasher flush against an inaccessible page (after it, then before it), so
//! a call that reads or writes one byte outside the caller's memory stops
//! the test with a fault. The C layer turns pointers and lengths into the
//! library's slices; these tests hold it to the lengths the header states.

#![cfg(any(target_os = "linux", target_os = "macos"))]

use b3_servil::*;
use core::ffi::{c_char, c_void};

/// `len` bytes with an inaccessible page before and after them, the bytes
/// flush against the page after (`at_end`) or before.
struct Guarded {
    base: *mut u8,
    total: usize,
    start: usize,
    len: usize,
}

impl Guarded {
    fn new(len: usize, at_end: bool) -> Self {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let body = len.div_ceil(page).max(1) * page;
        let total = body + 2 * page;
        let base = unsafe { libc::mmap(core::ptr::null_mut(), total, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        assert_ne!(base, libc::MAP_FAILED, "mmap");
        let base = base as *mut u8;
        unsafe {
            assert_eq!(libc::mprotect(base as *mut _, page, libc::PROT_NONE), 0);
            assert_eq!(libc::mprotect(base.add(page + body) as *mut _, page, libc::PROT_NONE), 0);
        }
        let start = if at_end { page + body - len } else { page };
        Guarded { base, total, start, len }
    }

    fn with(bytes: &[u8], at_end: bool) -> Self {
        let mut g = Guarded::new(bytes.len(), at_end);
        g.bytes().copy_from_slice(bytes);
        g
    }

    fn ptr(&self) -> *mut u8 {
        unsafe { self.base.add(self.start) }
    }

    fn bytes(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr(), self.len) }
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base as *mut _, self.total) };
    }
}

#[test]
fn every_call_inside_guard_pages() {
    let lens = [0, 1, 63, 64, 65, 1023, 1024, 1025, 2049, 3839, 4470, 7935, 16385, 65543, 524291, (1 << 20) + 5];
    for at_end in [true, false] {
        let key = Guarded::with(&[7u8; 32], at_end);
        let context = Guarded::with(b"guard pages 2026-10-25 the C calls\0", at_end);
        let modes =
            unsafe { [("plain", b3_mode_plain()), ("keyed", b3_mode_keyed(key.ptr())), ("derive key", b3_mode_derive_key(context.ptr() as *const c_char))] };
        for (name, mode) in &modes {
            for len in lens {
                let what = format!("{name}, {len} B, at end {at_end}");
                let message: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
                let want = unsafe { b3_hash(mode, message.as_ptr() as *const c_void, len) }.bytes;
                let mut xof_want = vec![0u8; 100];
                let mut flush = Guarded::with(&message, at_end);
                unsafe {
                    let input = flush.ptr() as *const c_void;
                    assert_eq!(b3_hash(mode, input, len).bytes, want, "b3_hash, {what}");
                    // A hasher whose state sits flush against a page too.
                    let state = Guarded::new(core::mem::size_of::<B3Hasher>(), at_end);
                    let hasher = state.ptr() as *mut B3Hasher;
                    b3_hasher_new(hasher, mode, 0, 0);
                    b3_hasher_update(hasher, input, len.min(1000));
                    b3_hasher_update(hasher, flush.bytes()[len.min(1000)..].as_ptr() as *const c_void, len - len.min(1000));
                    assert_eq!(b3_hasher_finalize(hasher).bytes, want, "b3_hasher_finalize, {what}");
                    let mut plain: Box<B3Hasher> = Box::new(core::mem::zeroed());
                    b3_hasher_new(&mut *plain, mode, 0, 0);
                    b3_hasher_update(&mut *plain, message.as_ptr() as *const c_void, len);
                    b3_hasher_finalize_xof(&*plain, 7, xof_want.as_mut_ptr() as *mut c_void, 100);
                    let mut xof = Guarded::new(100, !at_end);
                    b3_hasher_finalize_xof(hasher, 7, xof.ptr() as *mut c_void, 100);
                    assert_eq!(xof.bytes(), &xof_want[..], "b3_hasher_finalize_xof, {what}");
                }
            }
        }
    }
}
