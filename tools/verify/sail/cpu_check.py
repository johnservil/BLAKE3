"""The Lean model, compiled (Sail's Lean output without --lean-noncomputable,
with CpuCheck.lean as its main), against the CPU (cpu_check.c) on random
register values: add, eor, eor with a rotated operand, and extr.

    python3 cpu_check.py LEAN_CHECK_EXE CPU_EXE
"""
import random
import subprocess
import sys

lean_exe, cpu_exe = sys.argv[1], sys.argv[2]
random.seed(7)
lines = [f"{op} {random.getrandbits(64)} {random.getrandbits(64)} {random.randrange(32)}"
         for op in ("add", "eor", "eorror", "ror") for _ in range(2000)]
inp = "\n".join(lines) + "\n"
cpu = subprocess.run([cpu_exe], input=inp, capture_output=True, text=True, check=True).stdout.split()
lean = subprocess.run([lean_exe], input=inp, capture_output=True, text=True, check=True).stdout.split()
bad = [line for line, c, e in zip(lines, cpu, lean) if c != e]
print(f"{len(lines)} cases, {len(bad)} differ")
sys.exit(1 if bad or len(cpu) != len(lines) or len(lean) != len(lines) else 0)
