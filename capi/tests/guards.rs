//! Simple C guards diagnose bad arguments before borrowing their storage.
use b3_servil::*;
use core::mem::{MaybeUninit, size_of};

const CHILD: &str = "B3_CAPI_GUARD_CASE";
const CASES: &[(&str, &str)] = &[
    ("hasher-init-alignment", "the hasher is not 64-byte aligned"),
    ("hasher-read-alignment", "the hasher is not 64-byte aligned"),
    ("hasher-mode-overlap", "input and output ranges overlap"),
    ("input-null", "the input is NULL"),
    ("input-length", "the input length exceeds isize::MAX"),
    ("input-wrap", "the input address range overflows"),
    ("hasher-update-overlap", "input and output ranges overlap"),
    ("hasher-output-overlap", "input and output ranges overlap"),
    ("output-length", "the output length exceeds isize::MAX"),
];

#[repr(align(64))]
struct Arena([u8; size_of::<B3Hasher>() + 64]);

unsafe fn bad_call(case: &str) {
    let mode = b3_mode_plain();
    let mut arena = Arena([42; size_of::<B3Hasher>() + 64]);
    let pointer = arena.0.as_mut_ptr();
    let mut output = [Digest { bytes: [0; 32] }; 2];
    let mut hasher = MaybeUninit::<B3Hasher>::uninit();
    unsafe {
        match case {
            "hasher-init-alignment" => b3_hasher_new(pointer.add(1).cast(), &mode, 0, 0),
            "hasher-read-alignment" => {
                b3_hasher_new(pointer.cast(), &mode, 0, 0);
                b3_hasher_finalize(pointer.add(1).cast());
            }
            "hasher-mode-overlap" => {
                pointer.cast::<B3Mode>().write(mode);
                b3_hasher_new(pointer.cast(), pointer.cast(), 0, 0);
            }
            "input-null" => {
                b3_hash(&mode, core::ptr::null(), 1);
            }
            "input-length" => {
                b3_hash(&mode, pointer.cast(), isize::MAX as usize + 1);
            }
            "input-wrap" => {
                b3_hash(&mode, core::ptr::without_provenance(usize::MAX - 7), 16);
            }
            "hasher-update-overlap" => {
                b3_hasher_new(hasher.as_mut_ptr(), &mode, 0, 0);
                b3_hasher_update(hasher.as_mut_ptr(), hasher.as_ptr().cast(), 1);
            }
            "hasher-output-overlap" => {
                b3_hasher_new(hasher.as_mut_ptr(), &mode, 0, 0);
                b3_hasher_finalize_xof(hasher.as_ptr(), 0, hasher.as_mut_ptr().cast(), 32);
            }
            "output-length" => {
                b3_hasher_new(hasher.as_mut_ptr(), &mode, 0, 0);
                b3_hasher_finalize_xof(hasher.as_ptr(), 0, output.as_mut_ptr().cast(), isize::MAX as usize + 1);
            }
            _ => panic!("unknown guard case"),
        }
    }
    panic!("guard did not stop {case}");
}

#[test]
fn bad_calls_stop_at_the_guard() {
    if let Ok(case) = std::env::var(CHILD) {
        unsafe { bad_call(&case) };
    }
    for &(case, message) in CASES {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "bad_calls_stop_at_the_guard", "--nocapture", "--test-threads", "1"])
            .env(CHILD, case)
            .env("RUST_BACKTRACE", "0")
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{case}: the child returned");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(result.status.signal(), Some(6), "{case}: expected SIGABRT, got {:?}: {error}", result.status);
        }
        assert!(error.contains(message), "{case}: missing guard diagnostic {message}: {error}");
        eprintln!("{case}: guard diagnostic and abort verified");
    }
}

#[test]
fn adjacent_ranges_and_empty_null_ranges_are_allowed() {
    let mode = b3_mode_plain();
    unsafe {
        // A hasher's state and its input exactly adjacent, in both address
        // orders, with initialized storage.
        for (hasher_at, input_at) in [(0, size_of::<B3Hasher>()), (64, 0)] {
            let mut arena = Arena([42; size_of::<B3Hasher>() + 64]);
            let pointer = arena.0.as_mut_ptr();
            let hasher = pointer.add(hasher_at).cast::<B3Hasher>();
            let input = pointer.add(input_at);
            b3_hasher_new(hasher, &mode, 0, 0);
            b3_hasher_update(hasher, input.cast(), 64);
            assert_eq!(b3_hasher_finalize(hasher), b3_hash(&mode, input.cast(), 64));
        }
        let mut hasher = MaybeUninit::<B3Hasher>::uninit();
        b3_hasher_new(hasher.as_mut_ptr(), &mode, 0, 0);
        b3_hasher_update(hasher.as_mut_ptr(), core::ptr::null(), 0);
        b3_hasher_finalize_xof(hasher.as_ptr(), 0, core::ptr::null_mut(), 0);
    }
}
