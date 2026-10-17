//! Every interleaving of a caller's one-shot job beside two workers. Run
//! with LOOM_MAX_PREEMPTIONS=3 (about a minute; 2 in seconds):
//!     RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 cargo test --release
#![cfg(loom)]
use loom_scheduler::{job, Broken};

#[test]
fn a_job_of_two_pieces() {
    job(2, 2, Broken::Nothing);
}

#[test]
fn a_job_of_three_pieces() {
    job(2, 3, Broken::Nothing);
}

/// The model can fail. Without its look after setting its bit, a parking
/// worker misses work published meanwhile, and every thread blocks: Loom
/// reports "deadlock" and aborts (a deadlock is no panic a test can
/// catch), so run it alone to see it:
///     RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=2 cargo test --release -- --ignored without_the_recheck
#[test]
#[ignore = "fails by design, aborting: run alone"]
fn without_the_recheck_a_wake_is_lost() {
    job(2, 2, Broken::NoRecheck);
}

/// Without the fences that stand for the code's SeqCst accesses (Loom
/// models those as AcqRel), Loom finds lost wake-ups the code's SeqCst
/// order rules out: the store-buffering pairs need SeqCst. Run alone:
///     RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=2 cargo test --release -- --ignored without_seqcst
#[test]
#[ignore = "fails by design, aborting: run alone"]
fn without_seqcst_a_wake_is_lost() {
    job(2, 2, Broken::NoFences);
}

/// A stream of three tasks through a ring of two slots (one slot reused)
/// beside two workers: each task runs once, and the ring's slots are never
/// written while read (Loom checks the UnsafeCell accesses).
#[test]
fn a_stream_reusing_its_ring() {
    loom_scheduler::stream(2, 3, 2, Broken::Nothing);
}

/// The ring's room must follow finished tasks, not claims: a worker claims
/// a slot's number by CAS and reads the slot after, so a producer counting
/// claims can overwrite the slot first, and Loom reports the race.
#[test]
#[should_panic(expected = "Causality violation")]
fn room_counted_by_claims_races() {
    loom_scheduler::stream(2, 3, 2, Broken::RoomByClaims);
}
