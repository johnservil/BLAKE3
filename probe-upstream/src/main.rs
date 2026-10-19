use blake3::platform::Platform;

fn guarded(len: usize) -> &'static [u8] {
    unsafe {
        let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let base = libc::mmap(core::ptr::null_mut(), 2 * page, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) as *mut u8;
        assert!(base as isize != -1);
        assert_eq!(libc::mprotect(base.add(page) as *mut _, page, libc::PROT_NONE), 0);
        let start = base.add(page - len);
        for i in 0..len { *start.add(i) = i as u8; }
        core::slice::from_raw_parts(start, len)
    }
}

fn main() {
    let platform = Platform::detect();
    println!("platform {platform:?}");
    for blocks in [1usize, 2, 16] {
        let n = blocks * 64;
        for count in 1..=17 {
            println!("{platform:?}: {count} inputs x {blocks} blocks");
            match blocks {
                1 => run::<64>(platform, count),
                2 => run::<128>(platform, count),
                _ => run::<1024>(platform, count),
            }
        }
    }
    println!("no fault");
}

fn run<const N: usize>(platform: Platform, count: usize) {
    let inputs: Vec<&[u8; N]> = (0..count).map(|_| guarded(N).try_into().unwrap()).collect();
    let mut out = vec![0u8; 32 * count];
    platform.hash_many(&inputs, &[0u32; 8], 0, blake3::IncrementCounter::No, 0, 1, 2, &mut out);
}
