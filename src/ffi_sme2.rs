//! AArch64 SME2 implementation, 512-bit streaming vector length (Apple M4
//! and later). The kernels live in c/blake3_sme2_aarch64.S.
//!
//! The assembly handles exactly one shape: groups of sixteen whole inputs,
//! either 1024-byte chunks (sixteen blocks each, counter incrementing) or
//! 64-byte parent blocks (one block each, counter fixed). This wrapper feeds
//! it every full group and hands anything else to the NEON backend: a
//! remainder of fewer than sixteen inputs, chunk hashing with
//! `IncrementCounter::No`, or parent hashing with `IncrementCounter::Yes`.
//!
//! The kernels return the streaming vector length in 32-bit lanes and do
//! work only when it is 16. `Platform::detect()` checks that once and
//! selects NEON otherwise, so a return value other than 16 here is a
//! contract violation and stops the program.

use crate::{BLOCK_LEN, CHUNK_LEN, CVWords, IncrementCounter, OUT_LEN};

/// Lanes per kernel group: the hardware's real SIMD width.
pub const GROUP: usize = 16;

/// The degree `Platform::SME2` reports. Every `hash_many` call carries up to
/// this many inputs, so the kernel hashes `DEGREE / GROUP` groups per entry
/// into streaming mode. Entering and leaving streaming mode costs about half
/// a microsecond on Apple M4, more than a whole group of parent compressions
/// (~170 ns); eight groups per call amortize it. Measured single-thread
/// `blake3::hash` on a 64 MiB input: degree 16 → 226 ps/B, 64 → 182, 128 → 167
/// (NEON: 424). 128 also keeps `compress_subtree_wide`'s stack arrays at 8 KiB.
pub const DEGREE: usize = 128;

/// Chunk-group and parent-group entry points share one signature.
type Kernel = unsafe extern "C" fn(*const u8, *const u32, u64, u32, *mut u8, u64) -> u64;

// Unsafe because this may only be called on CPUs that report SME2 with a
// 512-bit streaming vector length, and on which NEON is also available.
pub unsafe fn hash_many<const N: usize>(
    inputs: &[&[u8; N]],
    key: &CVWords,
    counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8],
) {
    assert!(out.len() >= inputs.len() * OUT_LEN);

    let full_groups = inputs.len() / GROUP;
    // The chunk kernel takes flags, flags_start, and flags_end packed into
    // one word; the parent kernel applies flags | flags_start | flags_end to
    // its single block, which is what hash_many means for one-block inputs.
    let (kernel, packed_flags): (Option<Kernel>, u32) = if N == CHUNK_LEN && increment_counter.yes()
    {
        (
            Some(ffi::blake3_sme2_hash16_chunks_512),
            flags as u32 | (flags_start as u32) << 8 | (flags_end as u32) << 16,
        )
    } else if N == BLOCK_LEN && !increment_counter.yes() {
        (
            Some(ffi::blake3_sme2_hash16_parents_512),
            (flags | flags_start | flags_end) as u32,
        )
    } else {
        (None, 0)
    };

    let mut done = 0;
    let kind = if N == CHUNK_LEN { pace::CHUNKS } else { pace::PARENTS };
    if let (Some(kernel), true) = (kernel, full_groups > 0 && pace::sme2_allowed()) {
        let count = full_groups * GROUP;
        let lanes = if N == BLOCK_LEN {
            // The parent kernel reads its sixteen 64-byte blocks from one
            // contiguous run. Every caller in this crate already stores
            // parent inputs that way; anyone else gets a copy.
            let base = inputs[0].as_ptr();
            let contiguous = inputs[..count]
                .iter()
                .enumerate()
                .all(|(i, input)| input.as_ptr() == unsafe { base.add(i * BLOCK_LEN) });
            if contiguous {
                pace::timed(kind, full_groups, || unsafe {
                    kernel(
                        base,
                        key.as_ptr(),
                        counter,
                        packed_flags,
                        out.as_mut_ptr(),
                        full_groups as u64,
                    )
                })
            } else {
                // Gather up to GATHER groups per kernel call, so one entry
                // into streaming mode covers them all.
                const GATHER: usize = DEGREE / GROUP;
                let mut buf = [0u8; GATHER * GROUP * BLOCK_LEN];
                let mut group = 0;
                while group < full_groups {
                    let groups = (full_groups - group).min(GATHER);
                    for (i, input) in inputs[group * GROUP..][..groups * GROUP].iter().enumerate() {
                        buf[i * BLOCK_LEN..][..BLOCK_LEN].copy_from_slice(&input[..]);
                    }
                    let lanes = pace::timed(kind, groups, || unsafe {
                        kernel(
                            buf.as_ptr(),
                            key.as_ptr(),
                            counter,
                            packed_flags,
                            out[group * GROUP * OUT_LEN..].as_mut_ptr(),
                            groups as u64,
                        )
                    });
                    assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
                    group += groups;
                }
                16
            }
        } else {
            pace::timed(kind, full_groups, || unsafe {
                kernel(
                    inputs.as_ptr() as *const u8,
                    key.as_ptr(),
                    counter,
                    packed_flags,
                    out.as_mut_ptr(),
                    full_groups as u64,
                )
            })
        };
        assert_eq!(lanes, 16, "SME2 streaming vector length changed under us");
        done = count;
    }

    if done < inputs.len() {
        let rest_counter = if increment_counter.yes() {
            counter + done as u64
        } else {
            counter
        };
        // The remainder is fewer than sixteen inputs: the NEON kernels
        // (integer + vector hybrids, see neon_hybrid.rs) are the fastest
        // path there. They need the SHA-3 extension; the C kernel is the
        // fallback without it.
        if crate::neon_hybrid::sha3_detected() {
            unsafe {
                crate::neon_hybrid::hash_many(
                    &inputs[done..],
                    key,
                    rest_counter,
                    increment_counter,
                    flags,
                    flags_start,
                    flags_end,
                    &mut out[done * OUT_LEN..],
                );
            }
            return;
        }
        unsafe {
            crate::neon::hash_many(
                &inputs[done..],
                key,
                rest_counter,
                increment_counter,
                flags,
                flags_start,
                flags_end,
                &mut out[done * OUT_LEN..],
            );
        }
    }
}

/*
 * Pacing: SME2 only while it runs faster than NEON.
 *
 * An SME unit serves a whole cluster of cores. When another thread uses
 * it, a thread's SME2 calls slow down several times over (16-vCPU VM, 16
 * threads each hashing 1 MiB at once: 0.87 ns/B per thread on SME2, 0.33
 * on NEON). Other work on the other cores costs SME2 little (one SME2
 * thread beside fifteen hashing on NEON: 0.185 against 0.170 alone), and
 * the NEON hybrids run at nearly the same speed whatever runs beside them.
 * So the process times one full-size NEON call of each kernel kind once,
 * and each thread compares its full-size SME2 calls (eight groups, one
 * entry into streaming mode) with it: after STREAK calls in a row that
 * each took a quarter longer than NEON's best, the thread runs NEON for
 * BACKOFF, then tries SME2 again. The quarter covers what the comparison
 * leaves out: NEON's best is a tight loop's, and each switch costs a
 * transition. A lone slow call (an interrupt, a page fault) moves nothing.
 */
mod pace {
    /// Kernel kinds.
    pub const CHUNKS: usize = 0;
    pub const PARENTS: usize = 1;

    /// Consecutive slow calls that send a thread to NEON.
    #[cfg(feature = "std")]
    const STREAK: u32 = 4;

    /// How long, in thousandths of a second, a thread stays on NEON.
    #[cfg(feature = "std")]
    const BACKOFF_MS: u64 = 1;

    /// This thread's state: SME2 with the count of slow calls in a row, or
    /// NEON until a counter value.
    #[cfg(feature = "std")]
    #[derive(Clone, Copy)]
    enum State {
        Sme2(u32),
        NeonUntil(u64),
    }

    #[cfg(feature = "std")]
    std::thread_local! {
        static STATE: core::cell::Cell<State> = const { core::cell::Cell::new(State::Sme2(0)) };
    }

    /// The virtual counter (CNTVCT_EL0): one scalar instruction, so reading
    /// it right after an SME2 kernel costs no transition, as a clock call
    /// through the C library could.
    #[cfg(feature = "std")]
    #[inline(always)]
    fn ticks() -> u64 {
        let t: u64;
        unsafe { core::arch::asm!("mrs {t}, cntvct_el0", t = out(reg) t, options(nostack, nomem, preserves_flags)) };
        t
    }

    /// Counter ticks per second (CNTFRQ_EL0).
    #[cfg(feature = "std")]
    fn ticks_per_second() -> u64 {
        let f: u64;
        unsafe { core::arch::asm!("mrs {f}, cntfrq_el0", f = out(reg) f, options(nostack, nomem, preserves_flags)) };
        f
    }

    /// Ticks one full-size NEON call of each kind takes, the best of five,
    /// measured once per process on the first full-size SME2 call (about
    /// 0.2 ms on the VM).
    #[cfg(feature = "std")]
    fn neon_ticks() -> &'static [u64; 2] {
        use crate::{BLOCK_LEN, CHUNK_LEN, IncrementCounter, OUT_LEN};
        static NEON_TICKS: std::sync::OnceLock<[u64; 2]> = std::sync::OnceLock::new();
        NEON_TICKS.get_or_init(|| {
            let n = super::DEGREE;
            let input = std::vec![0x5au8; n * CHUNK_LEN];
            let chunks: std::vec::Vec<&[u8; CHUNK_LEN]> = input.chunks_exact(CHUNK_LEN).map(|c| c.try_into().unwrap()).collect();
            let parents: std::vec::Vec<&[u8; BLOCK_LEN]> = input.chunks_exact(BLOCK_LEN).take(n).map(|c| c.try_into().unwrap()).collect();
            let mut out = std::vec![0u8; n * OUT_LEN];
            let mut best = |f: &mut dyn FnMut(&mut [u8])| {
                (0..5).map(|_| { let t = ticks(); f(&mut out); ticks() - t }).min().unwrap().max(1)
            };
            let key = crate::IV;
            // Sound: the SME2 platform is only selected where NEON with
            // the SHA-3 extension is present (see sme2::hash_many's remainder).
            let chunk = best(&mut |o| unsafe { crate::neon_hybrid::hash_many(&chunks, key, 0, IncrementCounter::Yes, 0, crate::CHUNK_START, crate::CHUNK_END, o) });
            let parent = best(&mut |o| unsafe { crate::neon_hybrid::hash_many(&parents, key, 0, IncrementCounter::No, crate::PARENT, 0, 0, o) });
            [chunk, parent]
        })
    }

    /// Send this thread to NEON until `ticks` (tests: the paced-out path).
    #[cfg(all(test, feature = "std"))]
    pub fn neon_until(until: u64) {
        STATE.with(|state| state.set(if until == 0 { State::Sme2(0) } else { State::NeonUntil(until) }));
    }

    /// Whether this thread may run the SME2 kernels now.
    #[inline]
    pub fn sme2_allowed() -> bool {
        #[cfg(feature = "std")]
        return STATE.with(|state| match state.get() {
            State::Sme2(_) => true,
            State::NeonUntil(until) => {
                if ticks() < until {
                    return false;
                }
                state.set(State::Sme2(0));
                true
            }
        });
        #[cfg(not(feature = "std"))]
        true
    }

    /// Run one kernel call of `groups` groups, judging it against NEON when
    /// it is a full-size call.
    #[inline]
    pub fn timed(kind: usize, groups: usize, call: impl FnOnce() -> u64) -> u64 {
        #[cfg(feature = "std")]
        if groups == super::DEGREE / super::GROUP {
            let started = ticks();
            let lanes = call();
            let took = ticks() - started;
            STATE.with(|state| {
                let State::Sme2(slow) = state.get() else { return };
                let slow = if took * 4 > neon_ticks()[kind] * 5 { slow + 1 } else { 0 };
                state.set(if slow >= STREAK {
                    State::NeonUntil(ticks() + ticks_per_second() * BACKOFF_MS / 1000)
                } else {
                    State::Sme2(slow)
                });
            });
            return lanes;
        }
        let _ = (kind, groups);
        call()
    }
}

pub mod ffi {
    unsafe extern "C" {
        /// Sixteen whole 1024-byte chunks per group. `inputs` is a table of
        /// `16 * groups` chunk pointers; `counter` applies to the first chunk
        /// and increments per chunk; `flags` packs `flags | flags_start << 8
        /// | flags_end << 16`. Writes `16 * groups` 32-byte chaining values
        /// to `out`. Returns the streaming vector length in 32-bit lanes.
        pub fn blake3_sme2_hash16_chunks_512(
            inputs: *const u8,
            key: *const u32,
            counter: u64,
            flags: u32,
            out: *mut u8,
            groups: u64,
        ) -> u64;

        /// Sixteen parent compressions per group. `pairs` points at
        /// `16 * groups` adjacent 64-byte parent blocks; `counter` is used
        /// unchanged for every block; `flags` are the parent flags. Writes
        /// `16 * groups` 32-byte chaining values to `out`. Returns the
        /// streaming vector length in 32-bit lanes.
        pub fn blake3_sme2_hash16_parents_512(
            pairs: *const u8,
            key: *const u32,
            counter: u64,
            flags: u32,
            out: *mut u8,
            groups: u64,
        ) -> u64;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// A thread paced out to NEON gets the same digests from the SME2
    /// platform, tree and batch alike, and returns to SME2 after.
    #[test]
    #[cfg(feature = "std")]
    fn test_paced_out_matches() {
        if !crate::platform::sme2_detected() {
            return;
        }
        let mut input = std::vec![0u8; 1 << 20];
        crate::test::paint_test_input(&mut input);
        let messages: std::vec::Vec<&[u8]> = input.chunks_exact(BLOCK_LEN).take(1000).collect();
        let run = || {
            let mut digests = std::vec![crate::Hash::from_bytes([0; OUT_LEN]); messages.len()];
            crate::hash_many(&messages, &mut digests);
            (crate::hash(&input), digests)
        };
        let on_sme2 = run();
        pace::neon_until(u64::MAX);
        assert!(!pace::sme2_allowed());
        let on_neon = run();
        pace::neon_until(0);
        assert!(pace::sme2_allowed());
        assert!(on_sme2 == on_neon);
        assert_eq!(on_neon.0, crate::hash_all_at_once::<crate::join::SerialJoin>(&input, crate::IV, 0, 0, crate::platform::Platform::NEON).root_hash());
    }

    #[test]
    fn test_hash_many() {
        if !crate::platform::sme2_detected() {
            return;
        }
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }
}
