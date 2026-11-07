//! The C API (`b3.h`): a thin layer over the `blake3-servil` crate.
//! Hashing entry points initialize the crate; later initialization calls
//! check its completed state. Detected invalid arguments panic here and
//! stop the process across `extern "C"`. Raw-memory obligations remain
//! the caller's, as the header states.
//!
//! Every unsafe entry additionally requires disjoint writable argument
//! ranges, valid storage/lifetimes, and exclusive use of a mutable handle.
//! The C API is the convenience API: one-shot and incremental hashing on
//! the calling thread (BLAKE3 #267, Zooko, November 6); batches, other
//! cores and streams are the Rust throughput API's.
// Output-storage initialization remains a separate BLAKE3 #155 review item.

use blake3_servil::convenience::Hasher;
use blake3_servil::{Hash, Mode, OUT_LEN};
use core::ffi::{CStr, c_char, c_void};

/// `b3_digest`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Digest {
    pub bytes: [u8; OUT_LEN],
}

/// `b3_mode`: `kind` 0 plain, 1 keyed (`key` the key), 2 key derivation
/// (`key` the hashed context).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct B3Mode {
    kind: u8,
    key: [u8; 32],
}

/// `b3_hasher`: the caller's memory, holding a `Hasher`.
#[repr(C, align(64))]
pub struct B3Hasher {
    opaque: [u8; HASHER_SIZE],
}

const HASHER_SIZE: usize = 18432;
const _: () = assert!(size_of::<Hasher>() == HASHER_SIZE && align_of::<Hasher>() <= 64, "b3.h's B3_HASHER_SIZE and alignment hold a Hasher");

impl B3Mode {
    fn mode(&self) -> Mode<'_> {
        match self.kind {
            0 => Mode::Plain,
            1 => Mode::Keyed(&self.key),
            2 => Mode::DeriveKeyWith(&self.key),
            kind => panic!("b3: a b3_mode made by b3_mode_plain, b3_mode_keyed, or b3_mode_derive_key (kind {kind})"),
        }
    }
}

fn digest(hash: Hash) -> Digest {
    Digest { bytes: *hash.as_bytes() }
}

/// Local checks only: they cannot establish allocation ownership or lifetime.
fn check_range(ptr: *const c_void, len: usize, what: &str) {
    assert!(len <= isize::MAX as usize, "b3: {what} length exceeds isize::MAX");
    if len != 0 {
        assert!(!ptr.is_null(), "b3: {what} is NULL with a length of {len}");
        assert!(ptr.addr().checked_add(len).is_some(), "b3: {what} address range overflows");
    }
}

/// Compare known ranges without forming pointers past their ends.
fn check_disjoint(a: *const c_void, a_len: usize, b: *const c_void, b_len: usize) {
    if a_len == 0 || b_len == 0 {
        return;
    }
    let (a, b) = (a.addr(), b.addr());
    let overlaps = if a <= b { b - a < a_len } else { a - b < b_len };
    assert!(!overlaps, "b3: input and output ranges overlap");
}

fn check_hasher(ptr: *const B3Hasher) {
    check_range(ptr.cast(), size_of::<B3Hasher>(), "the hasher");
    assert!(ptr.is_aligned(), "b3: the hasher is not 64-byte aligned");
}

/// The bytes at `ptr`, `len` of them; NULL only with `len` 0.
unsafe fn bytes<'a>(ptr: *const c_void, len: usize, what: &str) -> &'a [u8] {
    check_range(ptr, len, what);
    if len == 0 {
        return &[];
    }
    // Sound by the contract: `len` readable bytes at `ptr`.
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len) }
}

unsafe fn mode<'a>(mode: *const B3Mode) -> &'a B3Mode {
    check_range(mode.cast(), size_of::<B3Mode>(), "the mode");
    unsafe { &*mode }
}

unsafe fn hasher<'a>(hasher: *const B3Hasher) -> &'a Hasher {
    check_hasher(hasher);
    unsafe { &*(hasher as *const Hasher) }
}

unsafe fn hasher_mut<'a>(hasher: *mut B3Hasher) -> &'a mut Hasher {
    check_hasher(hasher);
    unsafe { &mut *(hasher as *mut Hasher) }
}

/// `b3_version`: the library's version, as `b3.h`'s `B3_VERSION`.
#[unsafe(no_mangle)]
pub extern "C" fn b3_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn b3_mode_plain() -> B3Mode {
    B3Mode { kind: 0, key: [0; 32] }
}

/// # Safety
/// `key` points at 32 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_mode_keyed(key: *const u8) -> B3Mode {
    check_range(key.cast(), 32, "the key");
    B3Mode { kind: 1, key: unsafe { *(key as *const [u8; 32]) } }
}

/// # Safety
/// `context` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_mode_derive_key(context: *const c_char) -> B3Mode {
    blake3_servil::initialize();
    assert!(!context.is_null(), "b3: the context is NULL");
    let context = unsafe { CStr::from_ptr(context) }.to_str().expect("b3: the context is UTF-8");
    B3Mode { kind: 2, key: blake3_servil::throughput::context_key(context) }
}

/// # Safety
/// `mode` is a mode; `input` holds `len` bytes (NULL only with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hash(mode: *const B3Mode, input: *const c_void, len: usize) -> Digest {
    blake3_servil::initialize();
    let mode = unsafe { self::mode(mode) };
    digest(blake3_servil::convenience::hash_with(mode.mode(), unsafe { bytes(input, len, "the input") }))
}

/// # Safety
/// `hasher` points at a `b3_hasher`'s memory; `mode` is a mode. The hint
/// is advisory (b3.h): `length_hint` counts only with `has_length_hint`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_new(hasher: *mut B3Hasher, mode: *const B3Mode, has_length_hint: core::ffi::c_int, length_hint: u64) {
    blake3_servil::initialize();
    check_hasher(hasher);
    check_disjoint(hasher.cast(), size_of::<B3Hasher>(), mode.cast(), size_of::<B3Mode>());
    let mode = unsafe { self::mode(mode) };
    let new = Hasher::new(mode.mode(), (has_length_hint != 0).then_some(length_hint));
    // Sound: the memory is the caller's, sized and aligned for a Hasher,
    // which owns nothing to drop.
    unsafe { (hasher as *mut Hasher).write(new) };
}

/// # Safety
/// `hasher` was initialized by `b3_hasher_new`; `input` holds `len`
/// bytes (NULL only with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_update(hasher: *mut B3Hasher, input: *const c_void, len: usize) {
    check_disjoint(hasher.cast(), size_of::<B3Hasher>(), input, len);
    let hasher = unsafe { hasher_mut(hasher) };
    hasher.update(unsafe { bytes(input, len, "the input") });
}

/// # Safety
/// `hasher` was initialized by `b3_hasher_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_finalize(hasher: *const B3Hasher) -> Digest {
    digest(unsafe { self::hasher(hasher) }.finalize())
}

/// # Safety
/// `hasher` was initialized by `b3_hasher_new`; `out` has room for `len`
/// bytes (NULL only with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_finalize_xof(hasher: *const B3Hasher, offset: u64, out: *mut c_void, len: usize) {
    check_range(out, len, "the output");
    check_disjoint(hasher.cast(), size_of::<B3Hasher>(), out, len);
    let hasher = unsafe { self::hasher(hasher) };
    if len == 0 {
        return;
    }
    let mut reader = hasher.finalize_xof();
    reader.set_position(offset);
    // Sound by the contract: room for `len` bytes at `out`, zeroed first,
    // since the caller's storage may be uninitialized (#155).
    unsafe { core::ptr::write_bytes(out.cast::<u8>(), 0, len) };
    reader.fill(unsafe { core::slice::from_raw_parts_mut(out as *mut u8, len) });
}

#[cfg(test)]
mod test {
    include!("../../tools/require_nextest.rs");
}
