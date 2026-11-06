"""Fault controls for the last-block proof harness, each in a fresh process.

Both an unmodified positive and the planted fault are checked for each
control. Only the expected Unproved result counts as rejection; a parser
error or unsupported instruction is an error, not a successful control.
"""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from aarch64 import Unproved
import prove_last
import prove_x86
import workers

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = [
    ("root bit removed", (16,3,5),
     ".Lmixed16_endflags:\n    .long " + ", ".join(["10"]*16),
     ".Lmixed16_endflags:\n    .long " + ", ".join(["2"]*16), "differs"),
    ("completed lanes overwritten", (16,3,5),
     "    vmovdqu32 ZMMWORD PTR [r15 + 0]{k1}, zmm0",
     "    vmovdqu32 ZMMWORD PTR [r15 + 0], zmm0", "differs"),
    ("AVX2 snapshot selection reversed", (8,3,5),
     "        vpblendvb ymm8, ymm8, ymm0, ymm15",
     "        vpblendvb ymm8, ymm0, ymm8, ymm15", "differs"),
]


def control(item):
    label, case, before, after, reason = item
    source = (ROOT / "c/blake3_last_x86-64_unix.S").read_text()
    assert source.count(before) == 1, (label,source.count(before))
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="last-control-", dir=scratch) as d:
        c = Path(d) / "c"
        c.mkdir()
        path = c / "blake3_last_x86-64_unix.S"
        prove_x86.ROOT_DIR = d
        path.write_text(source)
        prove_x86.OBJS.clear()
        prove_last.prove(case)
        path.write_text(source.replace(before,after))
        prove_x86.OBJS.clear()
        try:
            prove_last.prove(case)
        except Unproved as e:
            if reason not in str(e):
                raise AssertionError(f"{label}: unexpected rejection: {e}") from e
        else:
            raise AssertionError(f"{label}: fault was accepted")
    return "rejected with positive control: " + label


if __name__ == "__main__":
    for result in workers.each(control, CONTROLS, 3):
        print(result, flush=True)
