"""Instructions of one hash() call (x86-64, qemu's execlog plugin): from
the second entry into `hash` to the third (one call plus the probe loop's
own few), grouped by function: where a difference in count.sh's totals comes
from. Run count.sh first (it builds the binaries and the plugins).
Usage: pypy3 one_call.py BINARY HASH_SYMBOL LEN"""
import bisect, collections, os, subprocess, sys
PLUGINS = os.path.join(os.environ.get("TMPDIR", "/tmp"), "qemu-plugins")
binary, sym, length = sys.argv[1:4]
BASE = 0x4000000000
syms = []
for line in subprocess.run(["x86_64-linux-gnu-nm", "-C", "--defined-only", binary], capture_output=True, text=True).stdout.splitlines():
    p = line.split(" ", 2)
    if len(p) == 3 and p[1] in "tTwW":
        syms.append((int(p[0], 16), p[2]))
syms.sort(); addrs = [a for a, _ in syms]
entry = BASE + next(a for a, n in syms if n == sym)
subprocess.run(["qemu-x86_64", "-cpu", "max", "-L", "/usr/x86_64-linux-gnu", "-plugin", os.path.join(PLUGINS, "libexeclog.so"),
                "-d", "plugin", "-D", os.path.join(PLUGINS, "exec.log"), binary, length, "3"], check=False)
pcs = []
for line in open(os.path.join(PLUGINS, "exec.log")):
    parts = line.split(",")
    if len(parts) >= 3:
        try: pcs.append(int(parts[1], 16))
        except ValueError: pass
hits = [i for i, pc in enumerate(pcs) if pc == entry]
one = pcs[hits[1]:hits[2]]
by = collections.Counter()
for pc in one:
    if BASE <= pc < BASE + 0x1000000:
        i = bisect.bisect_right(addrs, pc - BASE) - 1
        by[syms[i][1][:110]] += 1
    else:
        by["(libc)"] += 1
for n, c in by.most_common(): print(f"{c:6d}  {n}")
print(f"{len(one):6d}  total, entries into hash: {len(hits)}")
