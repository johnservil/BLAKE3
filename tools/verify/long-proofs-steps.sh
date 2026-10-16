# The long proofs' steps, sourced by tools/verify/long-proofs.sh in the
# commit's tree (each `run NAME COMMAND...` logs to NAME.txt).
# Every crate the builds need, first: a registry outage then ends the run
# in its first minute (the lockfiles this writes keep the later builds
# off the network).
run fetch sh -c 'cargo fetch -q && cargo fetch -q --manifest-path tools/verify/rust/Cargo.toml'
run lean_spec python3 c2sp/BLAKE3/check.py
run lean_emit sh -c 'pypy3 tools/verify/lean/emit.py && git diff --exit-code tools/verify/lean/compress.json tools/verify/lean/schedules.json tools/verify/lean/Generic.lean'
run generated sh -c 'pypy3 tools/gen_neon_hybrid.py | cmp - c/blake3_neon_hybrid_aarch64.S'
run helpers pypy3 -m unittest discover -s tools/verify -p test_helpers.py
run prove_hybrid pypy3 tools/verify/prove_hybrid.py
run prove_sme2 pypy3 tools/verify/prove_sme2.py chunks chunks_at messages parents xof chunks_x2 every parents_every x2_every xof_every messages_every
run prove_flat pypy3 tools/verify/prove_flat.py
run cross_check pypy3 tools/verify/cross_check.py
run mutants pypy3 tools/verify/mutants.py
# The library's tests against plausible bugs, every survivor known
# (tools/mutants/check.py; cargo-mutants under /root/tools when absent).
export PATH=/root/tools/bin:$PATH
command -v cargo-mutants >/dev/null || CARGO_INSTALL_ROOT=/root/tools cargo install -q cargo-mutants --locked
run test_mutants pypy3 tools/mutants/check.py
