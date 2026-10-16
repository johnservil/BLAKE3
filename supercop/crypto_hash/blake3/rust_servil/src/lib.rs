// SUPERCOP's crypto_hash is this crate's only entry point, with no setup
// hook, so the first call makes blake3-servil's required initialize() call
// (its startup self-test, before anything is timed); later calls pay only
// the Once's check.
static INITIALIZE: std::sync::Once = std::sync::Once::new();

pub fn crypto_hash(out: &mut [u8; 32], m: &[u8]) {
    INITIALIZE.call_once(blake3_servil::initialize);
    out.copy_from_slice(blake3_servil::hash(m).as_bytes())
}
