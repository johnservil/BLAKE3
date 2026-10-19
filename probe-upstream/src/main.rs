// blake3 1.8.7's AVX2 hash_many, from safe code, with each input flush
// against an inaccessible page: a read past an input's end faults.
use blake3::platform::Platform;

fn guarded(len: usize) -> &'static [u8] {
    unsafe {
        let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let base = libc::mmap(core::ptr::null_mut(), 2 * page, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) as *mut u8;
        assert!(base as isize != -1);
        assert_eq!(libc::mprotect(base.add(page) as *mut _, page, libc::PROT_NONE), 0);
        let start = base.add(page - len);
        for i in 0..len {
            *start.add(i) = i as u8;
        }
        core::slice::from_raw_parts(start, len)
    }
}

fn main() {
    let platform = Platform::detect();
    println!("blake3 1.8.7, platform {platform:?}");
    for count in 1..=16 {
        let inputs: Vec<&[u8; 64]> = (0..count).map(|_| guarded(64).try_into().unwrap()).collect();
        let mut out = vec![0u8; 32 * count];
        println!("{count} inputs of one block, each ending at a guard page");
        platform.hash_many(&inputs, &[0u32; 8], 0, blake3::IncrementCounter::No, 0, 1, 2, &mut out);
    }
    println!("no fault");
}
