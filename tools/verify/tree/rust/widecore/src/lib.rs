#![no_std]
//! The library's single-thread tree walk (src/tree_core.rs, included here
//! as it is), for Aeneas: its chunk length, chaining-value type, and the
//! largest buffer SME2's degree needs.

pub const CHUNK_LEN: usize = 1024;
pub const OUT_LEN: usize = 32;
pub const MAX: usize = 128;
pub type Cv = [u8; 32];

include!("../../../../../../src/tree_core.rs");
