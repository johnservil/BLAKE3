//! The C API against the official test vectors (test_vectors.json), through
//! every call: one-shot, and a hasher fed in uneven pieces, its digest and
//! its extended output at an offset.

use b3_servil::*;
use core::ffi::c_void;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_call_against_the_test_vectors() {
    let json: serde_json::Value = serde_json::from_str(include_str!("../../test_vectors/test_vectors.json")).unwrap();
    let key: [u8; 32] = json["key"].as_str().unwrap().as_bytes().try_into().unwrap();
    let context = std::ffi::CString::new(json["context_string"].as_str().unwrap()).unwrap();
    let modes = unsafe { [("hash", b3_mode_plain()), ("keyed_hash", b3_mode_keyed(key.as_ptr())), ("derive_key", b3_mode_derive_key(context.as_ptr()))] };
    let mut hasher: Box<B3Hasher> = unsafe { Box::new(core::mem::zeroed()) };
    for case in json["cases"].as_array().unwrap() {
        let len = case["input_len"].as_u64().unwrap() as usize;
        let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        for (name, mode) in &modes {
            let expected = case[*name].as_str().unwrap();
            let what = format!("{name}, {len} bytes");
            unsafe {
                assert_eq!(hex(&b3_hash(mode, input.as_ptr() as *const c_void, len).bytes), expected[..64], "b3_hash, {what}");
                // A hasher fed in pieces of 1, 1000, and the rest.
                b3_hasher_new(&mut *hasher, mode, 0, 0);
                let (a, rest) = input.split_at(len.min(1));
                let (b, c) = rest.split_at(rest.len().min(1000));
                b3_hasher_update(&mut *hasher, a.as_ptr() as *const c_void, a.len());
                b3_hasher_update(&mut *hasher, b.as_ptr() as *const c_void, b.len());
                b3_hasher_update(&mut *hasher, c.as_ptr() as *const c_void, c.len());
                assert_eq!(hex(&b3_hasher_finalize(&*hasher).bytes), expected[..64], "b3_hasher_finalize, {what}");
                let full = expected.len() / 2;
                let mut output = vec![0u8; full - 7];
                b3_hasher_finalize_xof(&*hasher, 7, output.as_mut_ptr() as *mut c_void, output.len());
                assert_eq!(hex(&output), expected[14..], "b3_hasher_finalize_xof from byte 7, {what}");
            }
        }
    }
}

/// The empty input through NULL pointers, as the contract allows.
#[test]
fn null_with_nothing() {
    let mode = b3_mode_plain();
    let empty = unsafe { b3_hash(&mode, core::ptr::null(), 0) };
    assert_eq!(empty.bytes, *blake3_servil::convenience::hash(b"").as_bytes());
}
