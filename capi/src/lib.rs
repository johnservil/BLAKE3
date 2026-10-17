//! The C API (`b3.h`): a thin layer over the `blake3-servil` crate, its
//! contracts checked at the boundary (a panic here stops the program, as
//! every `extern "C"` panic does). Each call initializes the crate first
//! (a load once it has).

use blake3_servil::{Hash, Hasher, Mode, OUT_LEN};
use core::ffi::{c_char, c_void, CStr};

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
const _: () = assert!(
    size_of::<Hasher>() == HASHER_SIZE && align_of::<Hasher>() <= 64,
    "b3.h's B3_HASHER_SIZE and alignment hold a Hasher"
);

impl B3Mode {
    fn mode(&self) -> Mode<'_> {
        match self.kind {
            0 => Mode::Hash,
            1 => Mode::Keyed(&self.key),
            2 => Mode::DeriveKeyMaterial(&self.key),
            kind => panic!("b3: a b3_mode made by b3_plain, b3_keyed, or b3_derive_key (kind {kind})"),
        }
    }
}

fn digest(hash: Hash) -> Digest {
    Digest { bytes: *hash.as_bytes() }
}

/// The bytes at `ptr`, `len` of them; NULL only with `len` 0.
unsafe fn bytes<'a>(ptr: *const c_void, len: usize, what: &str) -> &'a [u8] {
    if len == 0 {
        return &[];
    }
    assert!(!ptr.is_null(), "b3: {what} is NULL with a length of {len}");
    // Sound by the contract: `len` readable bytes at `ptr`.
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len) }
}

unsafe fn mode<'a>(mode: *const B3Mode) -> &'a B3Mode {
    assert!(!mode.is_null(), "b3: the mode is NULL");
    unsafe { &*mode }
}

unsafe fn hasher<'a>(hasher: *const B3Hasher) -> &'a Hasher {
    assert!(!hasher.is_null(), "b3: the hasher is NULL");
    unsafe { &*(hasher as *const Hasher) }
}

unsafe fn hasher_mut<'a>(hasher: *mut B3Hasher) -> &'a mut Hasher {
    assert!(!hasher.is_null(), "b3: the hasher is NULL");
    unsafe { &mut *(hasher as *mut Hasher) }
}

#[unsafe(no_mangle)]
pub extern "C" fn b3_plain() -> B3Mode {
    B3Mode { kind: 0, key: [0; 32] }
}

/// # Safety
/// `key` points at 32 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_keyed(key: *const u8) -> B3Mode {
    assert!(!key.is_null(), "b3: the key is NULL");
    B3Mode { kind: 1, key: unsafe { *(key as *const [u8; 32]) } }
}

/// # Safety
/// `context` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_derive_key(context: *const c_char) -> B3Mode {
    blake3_servil::initialize();
    assert!(!context.is_null(), "b3: the context is NULL");
    let context = unsafe { CStr::from_ptr(context) }.to_str().expect("b3: the context is UTF-8");
    B3Mode { kind: 2, key: blake3_servil::hazmat::hash_derive_key_context(context) }
}

/// # Safety
/// `mode` is a mode; `input` holds `len` bytes (NULL only with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hash(mode: *const B3Mode, input: *const c_void, len: usize) -> Digest {
    blake3_servil::initialize();
    let mode = unsafe { self::mode(mode) };
    digest(blake3_servil::hash_with(mode.mode(), unsafe { bytes(input, len, "the input") }))
}

/// # Safety
/// As `b3_hash`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hash_multithreaded(mode: *const B3Mode, input: *const c_void, len: usize) -> Digest {
    blake3_servil::initialize_multithreaded(None);
    let mode = unsafe { self::mode(mode) };
    digest(blake3_servil::hash_multithreaded_with(mode.mode(), unsafe { bytes(input, len, "the input") }))
}

/// The batch's input and digests, as slices.
unsafe fn batch<'a>(input: *const c_void, message_len: usize, count: usize, out: *mut Digest) -> (&'a [u8], &'a mut [[u8; OUT_LEN]]) {
    let stride = if message_len == 0 { 64 } else { message_len.div_ceil(64) * 64 };
    let total = stride.checked_mul(count).expect("b3: the batch's length fits a size_t");
    let input = unsafe { bytes(input, total, "the batch") };
    if count == 0 {
        return (input, &mut []);
    }
    assert!(!out.is_null(), "b3: the digests are NULL with a count of {count}");
    // Sound by the contract: room for `count` digests at `out`; a Digest is
    // 32 bytes, as [u8; 32] is.
    (input, unsafe { core::slice::from_raw_parts_mut(out as *mut [u8; OUT_LEN], count) })
}

/// # Safety
/// `mode` is a mode; `input` holds `count` strides of `message_len`
/// (NULL only with `count` 0), the bytes past each message zero; `out`
/// has room for `count` digests.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hash_many(mode: *const B3Mode, input: *const c_void, message_len: usize, count: usize, out: *mut Digest) {
    blake3_servil::initialize();
    let mode = unsafe { self::mode(mode) };
    let (input, out) = unsafe { batch(input, message_len, count, out) };
    blake3_servil::hash_many_with(mode.mode(), input, message_len, out);
}

/// # Safety
/// As `b3_hash_many`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hash_many_multithreaded(mode: *const B3Mode, input: *const c_void, message_len: usize, count: usize, out: *mut Digest) {
    blake3_servil::initialize_multithreaded(None);
    let mode = unsafe { self::mode(mode) };
    let (input, out) = unsafe { batch(input, message_len, count, out) };
    blake3_servil::hash_many_multithreaded_with(mode.mode(), input, message_len, out);
}

/// # Safety
/// `hasher` points at a `b3_hasher`'s memory; `mode` is a mode.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_init(hasher: *mut B3Hasher, mode: *const B3Mode) {
    blake3_servil::initialize();
    assert!(!hasher.is_null(), "b3: the hasher is NULL");
    let mode = unsafe { self::mode(mode) };
    let new = match mode.mode() {
        Mode::Hash => Hasher::new(),
        Mode::Keyed(key) => Hasher::new_keyed(key),
        Mode::DeriveKeyMaterial(context_key) => blake3_servil::hazmat::HasherExt::new_from_context_key(context_key),
        Mode::DeriveKey(_) => unreachable!("b3_mode holds a hashed context"),
    };
    // Sound: the memory is the caller's, sized and aligned for a Hasher,
    // which owns nothing to drop.
    unsafe { (hasher as *mut Hasher).write(new) };
}

/// # Safety
/// `hasher` was initialized by `b3_hasher_init`; `input` holds `len`
/// bytes (NULL only with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_update(hasher: *mut B3Hasher, input: *const c_void, len: usize) {
    let hasher = unsafe { hasher_mut(hasher) };
    hasher.update(unsafe { bytes(input, len, "the input") });
}

/// # Safety
/// As `b3_hasher_update`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_update_multithreaded(hasher: *mut B3Hasher, input: *const c_void, len: usize) {
    blake3_servil::initialize_multithreaded(None);
    let hasher = unsafe { hasher_mut(hasher) };
    hasher.update_multithreaded(unsafe { bytes(input, len, "the input") });
}

/// # Safety
/// `hasher` was initialized by `b3_hasher_init`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_finalize(hasher: *const B3Hasher) -> Digest {
    digest(unsafe { self::hasher(hasher) }.finalize())
}

/// # Safety
/// `hasher` was initialized by `b3_hasher_init`; `out` has room for `len`
/// bytes (NULL only with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn b3_hasher_output(hasher: *const B3Hasher, offset: u64, out: *mut c_void, len: usize) {
    let hasher = unsafe { self::hasher(hasher) };
    if len == 0 {
        return;
    }
    assert!(!out.is_null(), "b3: the output is NULL with a length of {len}");
    let mut reader = hasher.finalize_xof();
    reader.set_position(offset);
    // Sound by the contract: room for `len` bytes at `out`.
    reader.fill(unsafe { core::slice::from_raw_parts_mut(out as *mut u8, len) });
}
