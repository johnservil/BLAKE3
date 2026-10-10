#![no_std]
//! The Hasher's stack (src/stack_core.rs, included here as it is), for
//! Aeneas: its chaining-value type and depth.

pub type Cv = [u8; 32];
pub const DEPTH: usize = 55;

include!("../../../../../../src/stack_core.rs");
