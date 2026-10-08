//! The single-thread tree walk, in safe Rust over its kernels: the walk
//! `tools/verify/tree` proves computes the C2SP specification's tree.

use crate::platform::{Platform, MAX_SIMD_DEGREE_OR_2};
use crate::{CVWords, CHUNK_LEN, OUT_LEN};

/// A chaining value.
pub(crate) type Cv = [u8; OUT_LEN];

/// The most values a walk returns: the largest degree, at least 2.
const MAX: usize = MAX_SIMD_DEGREE_OR_2;

include!("tree_core.rs");

/// The library's kernels for one key and flags on one platform.
pub(crate) struct PlatformKernels<'a> {
    pub key: &'a CVWords,
    pub flags: u8,
    pub platform: Platform,
}

impl Kernels for PlatformKernels<'_> {
    #[inline(always)]
    fn degree(&self) -> usize {
        self.platform.simd_degree()
    }

    #[inline(always)]
    fn chunks(&self, input: &[u8], counter: u64, out: &mut [Cv]) -> usize {
        crate::compress_chunks_parallel(input, self.key, counter, self.flags, self.platform, out.as_flattened_mut())
    }

    #[inline(always)]
    fn parents(&self, children: &[Cv], out: &mut [Cv]) -> usize {
        crate::compress_parents_parallel(children.as_flattened(), self.key, self.flags, self.platform, out.as_flattened_mut())
    }

    #[inline(always)]
    #[allow(unused_variables)]
    fn subtree(&self, input: &[u8], ahead: usize, counter: u64, out: &mut [Cv]) -> Option<usize> {
        // Whole subtrees of 32 KiB to 1 MiB on SME2: the flat walk, SME2
        // alone (see sme2::compress_subtree_flat).
        #[cfg(blake3_sme2)]
        if matches!(self.platform, Platform::SME2) && crate::sme2::flat_takes(input.len()) {
            // Safe: the SME2 platform is selected only where the CPU has it,
            // and `out` holds simd_degree() values, which is sme2::DEGREE.
            return Some(unsafe {
                crate::sme2::compress_subtree_flat(input, ahead, self.key, counter, self.flags, out.as_flattened_mut())
            });
        }
        None
    }
}
