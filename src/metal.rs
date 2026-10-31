//! Probe branch: Apple Silicon Metal hashing, unkeyed 32-byte hashes, for
//! the measurements in `examples/host_lab.rs`. The bridge (`c/metal.mm`)
//! and the shader (`c/blake3.metal`) are kobizk's from #170, with a no-copy
//! call added; the CPU side of a split uses the crate's pool.

use crate::{CVBytes, Hash, IV};
use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn blake3_servil_metal_create(error: *mut c_char) -> *mut c_void;
    fn blake3_servil_metal_destroy(ctx: *mut c_void);
    fn blake3_servil_metal_stream_reserve(ctx: *mut c_void, slot: usize, capacity: usize, error: *mut c_char) -> *mut u8;
    fn blake3_servil_metal_stream_submit(ctx: *mut c_void, slot: usize, length: usize, error: *mut c_char) -> bool;
    fn blake3_servil_metal_stream_done(ctx: *mut c_void, slot: usize) -> bool;
    fn blake3_servil_metal_stream_wait(ctx: *mut c_void, slot: usize, out: *mut u8, error: *mut c_char) -> bool;
    fn blake3_servil_metal_hash(
        ctx: *mut c_void,
        data: *const u8,
        length: usize,
        cpu_groups: usize,
        group_chunks: usize,
        cpu: extern "C" fn(*mut c_void, *mut u8),
        context: *mut c_void,
        out: *mut u8,
        error: *mut c_char,
    ) -> bool;
    fn blake3_servil_metal_hash_nocopy(ctx: *mut c_void, data: *const u8, length: usize, out: *mut u8, error: *mut c_char) -> bool;
    fn blake3_servil_metal_wrap_only(ctx: *mut c_void, data: *const u8, length: usize, error: *mut c_char) -> bool;
}

fn message(error: &[c_char; 512]) -> String {
    // The bridge writes a NUL-terminated message into this zeroed buffer.
    unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy().into_owned()
}

/// The GPU's pipelines, queue and buffers, reused across calls.
pub struct Hasher {
    context: NonNull<c_void>,
}

impl Hasher {
    /// Create the pipelines (compiling the shader the first time in the
    /// process) and a hasher's buffers. Requires `initialize_multithreaded`.
    pub fn new() -> Result<Self, String> {
        crate::lanes::require();
        let mut error = [0; 512];
        let context = unsafe { blake3_servil_metal_create(error.as_mut_ptr()) };
        NonNull::new(context).map(|context| Self { context }).ok_or_else(|| message(&error))
    }

    /// Hash `input`, the CPU pool taking `cpu_percent` (0..=99) of its
    /// subtrees, the GPU the rest from a copy in its own buffer. A split
    /// (`cpu_percent` > 0) requires at least 2 MiB.
    pub fn hash(&mut self, input: &[u8], cpu_percent: u8) -> Hash {
        assert!(cpu_percent < 100, "CPU percentage 0..=99");
        assert!(cpu_percent == 0 || input.len() >= 4 * crate::lanes::MIN_SPLIT_LEN, "a split needs 2 MiB");
        let group_len = if cpu_percent == 0 { crate::CHUNK_LEN } else { 1usize << (usize::BITS - 1 - (input.len() / 4).leading_zeros()) };
        let groups = input.len().div_ceil(group_len);
        let cpu_groups = if cpu_percent == 0 { 0 } else { (groups * usize::from(cpu_percent) / 100).clamp(1, groups - 1) };
        let mut prefix = (&input[..cpu_groups * group_len], group_len);
        let mut out = [0; 32];
        let mut error = [0; 512];
        // The bridge borrows every pointer until it returns, and calls cpu
        // once, synchronously, while the GPU works.
        let ok = unsafe {
            blake3_servil_metal_hash(
                self.context.as_ptr(),
                input.as_ptr(),
                input.len(),
                cpu_groups,
                group_len / crate::CHUNK_LEN,
                cpu,
                (&mut prefix as *mut (&[u8], usize)).cast(),
                out.as_mut_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(ok, "{}", message(&error));
        Hash::from(out)
    }

    /// Hash `input` whole on the GPU, read in place: a buffer wraps its
    /// pages for the call. Requires a page-aligned input whose last page is
    /// the caller's to the page's end.
    pub fn hash_nocopy(&mut self, input: &[u8]) -> Hash {
        let mut out = [0; 32];
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_metal_hash_nocopy(self.context.as_ptr(), input.as_ptr(), input.len(), out.as_mut_ptr(), error.as_mut_ptr()) };
        assert!(ok, "{}", message(&error));
        Hash::from(out)
    }

    /// Wrap `input`'s pages in a GPU buffer and release it: the no-copy
    /// call's driver cost alone. Requirements as for [`Hasher::hash_nocopy`].
    pub fn wrap_only(&mut self, input: &[u8]) {
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_metal_wrap_only(self.context.as_ptr(), input.as_ptr(), input.len(), error.as_mut_ptr()) };
        assert!(ok, "{}", message(&error));
    }

    /// Slot `slot`'s GPU memory, at least `capacity` bytes, for the program
    /// to write a message into. The slot has no command in flight.
    pub fn slot(&mut self, slot: usize, capacity: usize) -> &mut [u8] {
        let mut error = [0; 512];
        let room = unsafe { blake3_servil_metal_stream_reserve(self.context.as_ptr(), slot, capacity, error.as_mut_ptr()) };
        assert!(!room.is_null(), "{}", message(&error));
        // Sound: the slot's buffer holds capacity bytes and no command reads it.
        unsafe { core::slice::from_raw_parts_mut(room, capacity) }
    }

    /// Start hashing the slot's first `len` bytes on the GPU, without waiting.
    pub fn submit(&mut self, slot: usize, len: usize) {
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_metal_stream_submit(self.context.as_ptr(), slot, len, error.as_mut_ptr()) };
        assert!(ok, "{}", message(&error));
    }

    /// Whether the slot's command has completed.
    pub fn done(&self, slot: usize) -> bool {
        unsafe { blake3_servil_metal_stream_done(self.context.as_ptr(), slot) }
    }

    /// Wait for the slot's command, and return its digest; the slot is free after.
    pub fn wait(&mut self, slot: usize) -> Hash {
        let mut digest = [0; 32];
        let mut error = [0; 512];
        let ok = unsafe { blake3_servil_metal_stream_wait(self.context.as_ptr(), slot, digest.as_mut_ptr(), error.as_mut_ptr()) };
        assert!(ok, "{}", message(&error));
        Hash::from(digest)
    }
}

extern "C" fn cpu(context: *mut c_void, out: *mut u8) {
    // Called only with a nonempty prefix of whole groups of at least
    // MIN_SPLIT_LEN, each a power of two, and a CV slot for each.
    let (prefix, group_len) = unsafe { *context.cast::<(&[u8], usize)>() };
    let groups = prefix.len() / group_len;
    let cvs = unsafe { std::slice::from_raw_parts_mut(out.cast::<CVBytes>(), groups) };
    for (i, group) in prefix.chunks_exact(group_len).enumerate() {
        let children = crate::lanes::subtree_children(group, IV, (i * group_len / crate::CHUNK_LEN) as u64, 0);
        cvs[i] = crate::hasher::parent_output(
            children[..32].try_into().unwrap(),
            children[32..].try_into().unwrap(),
            IV,
            0,
            crate::platform::Platform::detect(),
        )
        .chaining_value();
    }
}

impl Drop for Hasher {
    fn drop(&mut self) {
        // The unique owner; every call waits for its commands.
        unsafe { blake3_servil_metal_destroy(self.context.as_ptr()) };
    }
}
