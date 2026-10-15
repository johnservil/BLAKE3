// Hashes one message of LEN bytes REPS times, one call after another, as
// SUPERCOP's measure does; run under qemu's instruction counter.
fn main() {
    blake3_servil::initialize();
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (len, reps) = (a[0], a[1]);
    let input: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
    let mut acc = 0u8;
    for _ in 0..reps {
        acc ^= blake3_servil::hash(std::hint::black_box(&input)).as_bytes()[0];
    }
    std::process::exit((acc == 0x55) as i32);
}
