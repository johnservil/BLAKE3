"""Prove the instruction models equal to Arm's own specification.

For each register form the proofs use (cross_check.py's list), Isla runs the
instruction on Arm's architecture model (Sail's Armv9.4, as an Isla
snapshot) and gives every path through it: the registers it reads, the
values it writes, and the path's conditions, in SMT. Each trace becomes one
Z3 query: its definitions and conditions, and Isla's written value
differing from the model's on the same inputs (aarch64.py, vector.py). Every
query unsatisfiable proves the model equal to Arm's specification for every
input on which the instruction completes (a path that takes an exception,
such as a trap the system configures, runs no instruction). Expressions Z3
cannot read (the system state's enumerations and structs) are left out of
the conditions, which can only make a query harder to prove; an unreadable
written value fails the form. A vector register is compared in its 128
bits, the V register the NEON instructions write; a flag in PSTATE's
1-bit field, and a condition the inputs leave open (csel, ccmp, cset) on
each of its two sides.

    python3 tools/verify/isla_check.py

Needs Isla (github.com/rems-project/isla, `isla-footprint`), the Armv9.4
snapshot (isla-snapshots, `armv9p4.ir`), and Isla's source (its
configurations), named by ISLA_FOOTPRINT, ISLA_SNAPSHOT, and ISLA; and
LLVM's assembler (LLVM_MC).

The snapshot runs with Sail's later fix of one library function applied
(`write_snapshot`). Isla gives no trace for tbl (an index read from data):
that form stays checked against the CPU alone (cross_check.py).

The SME2 kernels' streaming register forms (cross_check_sme.py's list)
run in streaming mode with ZA enabled (`write_streaming_config`): all
sixteen lanes of a Z register, and ZA's tiles cell by cell, each ZA form
at the slice base its register holds in the kernels. The predicated add
and cmphi (Isla runs a path per predicate, or none), and the ZA loads and
stores (memory forms), are left to cross_check_sme.py. The four moves into
tile 1's vertical slices need more than 50 GB of memory in Isla, and fail
on a machine with less.
"""

import multiprocessing
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from z3 import BitVec, BitVecSort, BitVecVal, BoolSort, Const, Extract, Not, Or, Solver, is_true, parse_smt2_string, sat, unsat
import aarch64

FOOTPRINT = os.environ.get("ISLA_FOOTPRINT", "/tmp/target/release/isla-footprint")
ORIGINAL_SNAPSHOT = os.environ.get("ISLA_SNAPSHOT", "/root/isla/armv9p4.ir")
SNAPSHOT = "/tmp/isla-armv9p4-sailfix.ir"
ISLA = os.environ.get("ISLA", "/root/isla/isla")
CONFIG = "/tmp/isla-blake3.toml"


def write_snapshot():
    """The Armv9.4 snapshot with Sail's fix of `unsigned_subrange`
    (rems-project/sail 1f8f173, for sail-arm#32): the snapshot (isla-snapshots
    9591cbb) was compiled before it, and shifts the selected bits right by
    their high index where the fix shifts by the low one, so DUP (element)
    read imm5[4] as its index (isla#107)."""
    s = open(ORIGINAL_SNAPSHOT).read()
    start = s.index("fn zunsigned_subrange(zxs, zi, zj) {")
    end = s.index("\n}\n", start)
    old = "zz40 = zsail_shiftright(zz46, zi)"
    assert s[start:end].count(old) == 1, "the snapshot's unsigned_subrange has changed: check whether it has Sail's fix"
    with open(SNAPSHOT, "w") as f:
        f.write(s[:start] + s[start:end].replace(old, "zz40 = zsail_shiftright(zz46, zj)") + s[end:])


def write_config():
    """Isla's Armv9.4 configuration, with what the kernels run on: floating
    point and SIMD enabled at EL0 and EL1 and untrapped at EL2, a 512-bit
    vector length, and the SHA-3 instructions (xar)."""
    s = open(os.path.join(ISLA, "configs", "armv9p4.toml")).read()
    edits = [
        ('"ZCR_EL3" = "{ bits = 0x0000000000000000 }"', '"ZCR_EL3" = "{ bits = 0x0000000000000003 }"'),
        ('"FEAT_SHA3_IMPLEMENTED" = false', '"FEAT_SHA3_IMPLEMENTED" = true'),
        ('"VTCR_EL2" = "{ bits = 0x0000000000000000 }"\n',
         '"VTCR_EL2" = "{ bits = 0x0000000000000000 }"\n'
         '"CPACR_EL1" = "{ bits = 0x0000000000300000 }"\n'
         '"CPTR_EL2" = "{ bits = 0x00000000000032ff }"\n'
         '"ZCR_EL1" = "{ bits = 0x0000000000000003 }"\n'
         '"ZCR_EL2" = "{ bits = 0x0000000000000003 }"\n'),
    ]
    for old, new in edits:
        assert s.count(old) >= 1, f"Isla's configuration has changed: {old}"
        s = s.replace(old, new, 1)
    with open(CONFIG, "w") as f:
        f.write(s)


STREAMING_CONFIG = "/tmp/isla-blake3-streaming.toml"


def write_streaming_config():
    """write_config's configuration in streaming mode with ZA enabled, as
    the SME2 kernels run: SME implemented, enabled at EL0 and EL1 and
    untrapped at EL2 and EL3, a 512-bit streaming vector length."""
    s = open(CONFIG).read()
    edits = [
        ('"FEAT_SME_IMPLEMENTED" = false', '"FEAT_SME_IMPLEMENTED" = true'),
        ('"CPACR_EL1" = "{ bits = 0x0000000000300000 }"', '"CPACR_EL1" = "{ bits = 0x0000000003300000 }"'),
        ('"CPTR_EL2" = "{ bits = 0x00000000000032ff }"', '"CPTR_EL2" = "{ bits = 0x00000000000022ff }"'),
        ('"CPTR_EL3" = "{ bits = 0x0000000000000000 }"', '"CPTR_EL3" = "{ bits = 0x0000000000001100 }"'),
        ('"SMCR_EL3" = "{ bits = 0x0000000000000000 }"', '"SMCR_EL3" = "{ bits = 0x0000000000000003 }"'),
        ('"ZCR_EL2" = "{ bits = 0x0000000000000003 }"\n',
         '"ZCR_EL2" = "{ bits = 0x0000000000000003 }"\n'
         '"SMCR_EL1" = "{ bits = 0x0000000000000003 }"\n'
         '"SMCR_EL2" = "{ bits = 0x0000000000000003 }"\n'
         '"PSTATE.SM" = "0b1"\n'
         '"PSTATE.ZA" = "0b1"\n'),
    ]
    for old, new in edits:
        assert s.count(old) >= 1, f"Isla's configuration has changed: {old}"
        s = s.replace(old, new, 1)
    with open(STREAMING_CONFIG, "w") as f:
        f.write(s)


MC = os.environ.get("LLVM_MC", "/usr/lib/llvm-19/bin/llvm-mc")


def opcode(instruction):
    """The instruction's encoding, as hex, from LLVM's assembler (Isla's own
    lacks some of the extensions the kernels use)."""
    out = subprocess.run([MC, "-triple=aarch64", "-mattr=+sha3,+sve2,+sme2", "-show-encoding"],
                         input=instruction, capture_output=True, text=True, check=True).stdout
    b = re.search(r"encoding: \[(.*)\]", out).group(1).split(",")
    return "".join(x[2:] for x in b)  # in memory order, as Isla reads it


def traces(instruction, config=CONFIG, initial=None):
    """Isla's traces of `instruction`, `initial` {x register: value} set."""
    given = [a for r, v in (initial or {}).items() for a in ("-I", f"R{r}=0x{v:016x}")]
    out = subprocess.run([FOOTPRINT, "-A", SNAPSHOT, "-C", config, "-x", "-i", opcode(instruction)] + given,
                         capture_output=True, text=True, timeout=1200, check=True).stdout
    return [t for t in re.split(r"^\(trace\b", out, flags=re.M)[1:]]


def sexprs(text):
    """The trace's top-level events, as strings."""
    events, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                events.append(text[start:i + 1])
                start = None
    return events


SORT = re.compile(r"\(_ BitVec (\d+)\)|Bool")


FLAGS = "NZCV"


def query(trace, model_outputs, lanes=4, extra=lambda xin: []):
    """Why the trace's written registers or flags differ from the model's
    for some input on the trace's path, or None when they agree on every
    one (each query unsatisfiable). A vector register is Arm's Z register's
    low 512 bits, the model's sixteen 32-bit lanes; a flag is PSTATE's 1-bit
    field."""
    decls, conds, xin, xout, zin, zout = {}, [], {}, {}, None, None
    fin, fout, zain, zaout = {}, {}, None, None

    def term(expr):
        f = parse_smt2_string(f"(assert (= {expr} {expr}))", decls=decls)
        return f[0].arg(0)

    def field(ev, name):
        m = re.search(r"\(\|" + name + r"\| (v\d+|#b[01])\)", ev)
        return term(m.group(1)) if m else None

    for ev in sexprs(trace):
        if ev.startswith("(read-mem") or ev.startswith("(write-mem"):
            return "accesses memory (the memory forms are to come)"
        m = re.match(r"\(declare-const (\S+) ((?:\(_ BitVec \d+\))|Bool)\)", ev)
        if m:
            name, sort = m.groups()
            s = BoolSort() if sort == "Bool" else BitVecSort(int(SORT.match(sort).group(1)))
            decls[name] = Const(name, s)
            continue
        m = re.match(r"\(define-const (\S+) (.*)\)$", ev, re.S)
        if m:
            name, expr = m.groups()
            try:
                t = term(expr)
            except Exception:
                continue        # an enumeration or struct: left out
            decls[name] = Const(name, t.sort())
            conds.append(decls[name] == t)
            continue
        m = re.match(r"\(assert (.*)\)$", ev, re.S)
        if m:
            try:
                conds.append(term(m.group(1)))
            except Exception:
                pass
            continue
        m = re.match(r"\(read-reg \|R(\d+)\| nil (v\d+)\)", ev)
        if m and m.group(2) in decls and int(m.group(1)) not in xout:
            xin.setdefault(int(m.group(1)), decls[m.group(2)])
            continue
        m = re.match(r"\(write-reg \|R(\d+)\| nil (.*)\)$", ev, re.S)
        if m:
            xout[int(m.group(1))] = term(m.group(2))
            continue
        m = re.match(r"\((read|write)-reg \|PSTATE\| \(\(_ field \|([NZCV])\|\)\) ", ev)
        if m:
            value = field(ev, m.group(2))
            if value is None:
                return f"a flag {m.group(2)} this check cannot read"
            if m.group(1) == "read" and m.group(2) not in fout:
                fin.setdefault(m.group(2), value)
            elif m.group(1) == "write":
                fout[m.group(2)] = value
            continue
        m = re.match(r"\((read|write)-reg \|_Z\| nil \(_ vec (.*)\)\)$", ev, re.S)
        if m:
            elems = m.group(2).split()
            if len(elems) != 32 or not all(re.fullmatch(r"v\d+", e) for e in elems):
                return "a vector register Z's value this check cannot read"
            if m.group(1) == "read" and zin is None:
                zin = elems
            elif m.group(1) == "write":
                zout = elems
            continue
        m = re.match(r"\((read|write)-reg \|_ZA\| nil \(_ vec (.*)\)\)$", ev, re.S)
        if m:
            elems = m.group(2).split()
            if len(elems) != 256 or not all(re.fullmatch(r"v\d+", e) for e in elems):
                return "ZA's value this check cannot read"
            if m.group(1) == "read" and zain is None:
                zain = elems
            elif m.group(1) == "write":
                zaout = elems
    vin = {}
    if zin is not None:
        for n, e in enumerate(zin):
            z = decls[e]
            vin[n] = [Extract(32 * l + 31, 32 * l, z) for l in range(16)]
    wantv = {}
    if zout is not None:
        for n, e in enumerate(zout):
            if zin is None or e != zin[n]:
                z = decls[e]
                wantv[n] = [Extract(32 * l + 31, 32 * l, z) for l in range(16)]
    def tiles(rows):
        """ZA's four .S tiles from its rows: slice i of tile t is row 4 i + t."""
        return [[[Extract(32 * j + 31, 32 * j, decls[rows[4 * i + t]]) for j in range(16)] for i in range(16)]
                for t in range(4)]
    zain_tiles = tiles(zain) if zain is not None else None
    wantza = tiles(zaout) if zaout is not None and zaout != zain else None
    conds += extra(xin)
    s = Solver()
    s.add(*conds)
    if s.check() != sat:
        return "the parsed path conditions have no established satisfying input"
    try:
        branches = model_outputs(xin, vin, fin, zain_tiles)
    except aarch64.Unproved as e:
        return f"the model: {e}"
    for assumed, gotx, gotv, gotf, gotza in branches:
        s.push()
        s.add(*assumed)
        if s.check() == unsat:
            s.pop()
            continue                # the model's branch lies off this path
        if set(gotx) != set(xout) or set(gotv) != set(wantv) or (gotf is None) != (not fout) or \
                (gotza is None) != (wantza is None):
            s.pop()
            return (f"writes {sorted(xout)}, vectors {sorted(wantv)}, flags {sorted(fout)} by Arm's model; "
                    f"{sorted(gotx)}, vectors {sorted(gotv)}, flags {'NZCV' if gotf else '[]'} by ours")
        pairs = [(f"R{r}", xout[r], gotx[r]) for r in xout]
        # Lanes 0 to 3 (the 128-bit V register); Arm's model leaves the rest
        # to the vector length, which the model fixes at 512 bits.
        pairs += [(f"V{n} lane {l}", wantv[n][l], gotv[n][l]) for n in wantv for l in range(lanes)]
        if wantza is not None:
            pairs += [(f"ZA{t} slice {i} element {j}", wantza[t][i][j], gotza[t][i][j])
                      for t in range(4) for i in range(16) for j in range(16)]
        if gotf is not None:
            if set(fout) != set(FLAGS):
                s.pop()
                return f"Arm's model writes flags {sorted(fout)}, ours all four"
            pairs += [(f"flag {f}", fout[f] == 1, g) for f, g in zip(FLAGS, gotf)]
        # One query: does any written value differ? Its model names one.
        s.push()
        s.add(Or([want != got for _, want, got in pairs]) if pairs else False)
        result = s.check()
        if result != unsat:
            names = [what for what, want, got in pairs
                     if result == sat and is_true(s.model().eval(want != got, model_completion=True))]
            s.pop()
            s.pop()
            return f"{names[0] if names else 'a written value'} differs on some input"
        s.pop()
        s.pop()
    return None


def model(mnem, ops, streaming=False, fixed=None):
    """The model's writes for the inputs Arm's model read, one branch per
    outcome of each condition the inputs leave open: a list of (the
    branch's assumptions, {x register: value}, {vector register: sixteen
    lanes}, the four flags or None when they keep their value, ZA's tiles
    or None when it keeps its value). Every input counts as a length, so
    the model computes flags and conditions as terms (aarch64.decide
    splits on them). `streaming`: in streaming mode with ZA enabled;
    `fixed`: x registers given these values (a ZA slice base)."""
    def run(xin, vin, fin, zain):
        def machine(assumed):
            m = aarch64.Machine({0: (mnem, ops)}, {})
            for i in range(31):
                m.x[i] = xin.get(i, BitVec(f"model_x{i}", 64))
            for n in range(32):
                m.v[n] = vin.get(n, [BitVec(f"model_v{n}_{l}", 32) for l in range(16)])
            m.x["sp"] = aarch64.Ptr("stack", 0)
            for r, value in (fixed or {}).items():
                m.x[r] = BitVecVal(value, 64)
            if streaming:
                m.streaming = True
                m.p = {i: None for i in range(16)}
                m.za = [[list(row) for row in tile] for tile in zain] if zain is not None else \
                    [[[BitVec(f"model_za{t}_{i}_{j}", 32) for j in range(16)] for i in range(16)] for t in range(4)]
            m.nzcv = tuple((fin[f] if f in fin else BitVec(f"model_{f}", 1)) == 1 for f in FLAGS)
            m.assumptions = list(assumed)
            names = set()
            for t in list(m.x.values()) + [l for ls in m.v.values() for l in ls] + list(m.nzcv):
                if not isinstance(t, aarch64.Ptr):
                    names |= aarch64.term_symbols(t)
            m.lengths = names
            written = set()
            setter = m.set
            def record(name, val):
                if name[0] in "wx" and name not in ("wzr", "xzr"):
                    written.add(int(name[1:]))
                return setter(name, val)
            m.set = record
            return m, written
        out, work = [], [[]]
        while work:
            assumed = work.pop()
            m, written = machine(assumed)
            vs, flags = {n: list(ls) for n, ls in m.v.items()}, m.nzcv
            za = [[list(row) for row in tile] for tile in m.za] if streaming else None
            try:
                aarch64.step(m, 0, mnem, ops)
            except aarch64.Undecided as e:
                work += [assumed + [e.cond], assumed + [Not(e.cond)]]
                continue
            gotx = {i: m.x[i] for i in written}
            gotv = {n: m.v[n] for n in range(32) if any(not a.eq(b) for a, b in zip(m.v[n], vs[n]))}
            # A vector write of an unchanged value is still a write.
            dest = re.match(r"([vqdsbz])(\d+)", ops[0])
            if dest and not gotx and not gotv and m.nzcv is flags:
                gotv[int(dest.group(2))] = m.v[int(dest.group(2))]
            gotza = None
            if streaming and any(not a.eq(b) for t0, t1 in zip(za, m.za) for r0, r1 in zip(t0, t1) for a, b in zip(r0, r1)):
                gotza = m.za
            out.append((assumed, gotx, gotv, None if m.nzcv is flags else m.nzcv, gotza))
        return out
    return run


def check_form(form):
    mnem, text = form
    instruction = f"{mnem} {text}"
    ops = aarch64.split_operands(text)
    try:
        ts = traces(instruction)
    except subprocess.TimeoutExpired:
        return instruction, "Isla timed out"
    except subprocess.CalledProcessError as e:
        return instruction, f"Isla or LLVM failed (exit {e.returncode})"
    if not ts:
        return instruction, "Isla gave no trace"
    # A path that takes an exception (a trap the system's configuration
    # leaves open, such as SIMD in streaming mode) runs no instruction: the
    # model describes the paths that complete.
    done = [t for t in ts if not re.search(r"write-reg \|ESR_EL\d\|", t.split("(cycle)")[-1])]
    if not done:
        return instruction, f"every one of Isla's {len(ts)} paths takes an exception"
    for t in done:
        try:
            why = query(t, model(mnem, ops))
        except Exception as e:
            why = f"this check failed: {type(e).__name__}: {e}"
        if why:
            return instruction, why
    return instruction, f"equal to Arm's specification on all {len(done)} paths that complete"


def completed(ts):
    """The traces whose instruction completes (no exception after it starts)."""
    return [t for t in ts if not re.search(r"write-reg \|ESR_EL\d\|", t.split("(cycle)")[-1])]


# The kernels' ZA slice bases (c/blake3_sme2_aarch64.S): w12-w15 hold
# 0, 4, 8, and 12 throughout.
SLICE_BASES = {12: 0, 13: 4, 14: 8, 15: 12}


def check_streaming_form(form):
    """A streaming SVE or SME2 form, as check_form a NEON one, at all sixteen
    lanes, ZA included. A ZA form runs at the slice base its register holds
    in the kernels (a symbolic base makes Arm's model select among ZA's 256
    rows, past what the solver settles in hours)."""
    mnem, text = form
    instruction = f"{mnem} {text}"
    ops = aarch64.split_operands(text)
    base = re.search(r"\[w(\d+),", text)
    fixed = {int(base.group(1)): SLICE_BASES[int(base.group(1))]} if base else None
    try:
        ts = completed(traces(instruction, STREAMING_CONFIG, fixed))
    except subprocess.TimeoutExpired:
        return instruction, "Isla timed out"
    except subprocess.CalledProcessError as e:
        return instruction, f"Isla or LLVM failed (exit {e.returncode})"
    if not ts:
        return instruction, "Isla gave no completing path"
    for t in ts:
        try:
            why = query(t, model(mnem, ops, streaming=True, fixed=fixed), lanes=16)
        except Exception as e:
            why = f"this check failed: {type(e).__name__}: {e}"
        if why:
            return instruction, why
    at = f", the slice base {fixed[int(base.group(1))]}" if base else ""
    return instruction, f"equal to Arm's specification on all {len(ts)} paths that complete{at}"


def streaming_forms():
    """The streaming register forms the SME2 kernels use (cross_check_sme.py's
    list), but those Isla cannot run within reason: the predicated add (a
    path per predicate) and cmphi (no path); and the ZA loads and stores
    (memory forms, to come)."""
    import cross_check_sme
    import prove_sme2
    return [(m, t) for m, t in cross_check_sme.forms(prove_sme2.assemble())
            if m not in ("ld1w", "st1w", "cmphi") and not re.search(r"\bp\d+/m", t)]


def all_forms():
    import cross_check
    import prove_hybrid
    import prove_rust
    prove_rust.main(exit=False)
    obj = prove_hybrid.assemble()
    forms = cross_check.forms(list(aarch64.disassemble(obj, 0, 1 << 30).values()) + sorted(aarch64.EXECUTED))
    return forms


def mutant(i):
    """Whether planted model error `i` of MUTANTS is rejected: (instruction,
    the error, how to plant it (a function returning the function that
    undoes it), and whether the form runs in streaming mode)."""
    (mnem, text), what, plant, streaming = MUTANTS[i]
    ops = aarch64.split_operands(text)
    base = re.search(r"\[w(\d+),", text)
    fixed = {int(base.group(1)): SLICE_BASES[int(base.group(1))]} if base else None
    ts = completed(traces(f"{mnem} {text}", STREAMING_CONFIG if streaming else CONFIG, fixed))
    undo = plant()
    try:
        rejected = any(query(t, model(mnem, ops, streaming=streaming, fixed=fixed), lanes=16 if streaming else 4)
                       for t in ts)
    finally:
        undo()
    return f"{mnem} {text}, {what}", rejected


def plant_carry_gt():
    from z3 import UGT
    original = aarch64.symbolic_flags
    def wrong(m, a, b, w):
        original(m, a, b, w)
        x = Extract(w - 1, 0, a) if a.size() > w else a
        y = Extract(w - 1, 0, b) if b.size() > w else b
        n, z, _, v = m.nzcv
        m.nzcv = (n, z, UGT(x, y), v)
    aarch64.symbolic_flags = wrong
    return lambda: setattr(aarch64, "symbolic_flags", original)


def plant_hi_without_z():
    original = aarch64.CONDS["hi"]
    aarch64.CONDS["hi"] = lambda n, z, c, v: c
    return lambda: aarch64.CONDS.__setitem__("hi", original)


def plant_operands(mnem, wrong_text):
    """The model runs `wrong_text`'s operands in place of the form's."""
    def plant():
        original = aarch64.step
        def wrong(m, pc, mn, ops):
            return original(m, pc, mn, aarch64.split_operands(wrong_text) if mn == mnem else ops)
        aarch64.step = wrong
        return lambda: setattr(aarch64, "step", original)
    return plant


MUTANTS = [
    (("cmp", "x9, x10"), "the carry as > in place of >=", plant_carry_gt, False),
    (("cset", "w9, hi"), "hi ignoring Z", plant_hi_without_z, False),
    (("xar", "v0.2d, v1.2d, v2.2d, #0x20"), "rotated by 24", plant_operands("xar", "v0.2d, v1.2d, v2.2d, #0x18"), False),
    (("dup", "v0.4s, v1.s[3]"), "lane 2", plant_operands("dup", "v0.4s, v1.s[2]"), False),
    (("eor", "w9, w10, w11, ror #7"), "rotated by 8", plant_operands("eor", "w9, w10, w11, ror #8"), False),
    (("xar", "z0.s, z0.s, z1.s, #0x7"), "rotated by 8", plant_operands("xar", "z0.s, z0.s, z1.s, #0x8"), True),
    (("mov", "za2h.s[w12, 0x0:0x3], { z0.s - z3.s }"), "into tile 1",
     plant_operands("mov", "za1h.s[w12, 0x0:0x3], { z0.s - z3.s }"), True),
]


def main():
    write_config()
    write_snapshot()
    forms = all_forms()
    bad = 0
    with multiprocessing.Pool(min(6, multiprocessing.cpu_count()), maxtasksperchild=1) as pool:
        for instruction, verdict in pool.imap_unordered(check_form, forms):
            print(f"{instruction}: {verdict}", flush=True)
            bad += not verdict.startswith("equal")
    print(f"{len(forms) - bad} of {len(forms)} instruction forms proved equal to Arm's specification")
    write_streaming_config()
    streaming = streaming_forms()
    sbad = 0
    with multiprocessing.Pool(min(6, multiprocessing.cpu_count()), maxtasksperchild=1) as pool:
        for instruction, verdict in pool.imap_unordered(check_streaming_form, streaming):
            print(f"{instruction}: {verdict}", flush=True)
            sbad += not verdict.startswith("equal")
    print(f"{len(streaming) - sbad} of {len(streaming)} streaming forms proved equal to Arm's specification")
    bad += sbad
    # The check rejects wrong models.
    accepted = 0
    with multiprocessing.Pool(min(6, multiprocessing.cpu_count()), maxtasksperchild=1) as pool:
        for what, rejected in pool.imap_unordered(mutant, range(len(MUTANTS))):
            print(f"wrong model, {what}: {'rejected' if rejected else 'ACCEPTED'}", flush=True)
            accepted += not rejected
    print(f"{len(MUTANTS) - accepted} of {len(MUTANTS)} wrong models rejected")
    sys.exit(1 if bad or accepted else 0)


if __name__ == "__main__":
    main()
