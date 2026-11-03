"""The BLAKE3 hash of what an ELF executable loads: every section the
loader maps (SHF_ALLOC), its name, address, size and bytes (a NOBITS
section's size alone), but the notes, whose build ID hashes the whole
file, debug info and line tables included. Two executables with the
same hash run the same: the machine code, its data and its relocations
are byte for byte the same. A mutant whose test binaries all hash as the
unmutated build's changes nothing on this machine (test-runner,
check.py's "same code"); one that moves a line moves a panic's location
in the read-only data, so it hashes differently and its tests run.

The hash is official BLAKE3's b3sum (MUTANTS_B3SUM, which check.py
installs): the identity of the code under test never rests on that
code's own hashing.

    MUTANTS_B3SUM=/path/to/b3sum python3 tools/mutants/code_hash.py EXECUTABLE
"""

import os
import struct
import subprocess
import sys

SHF_ALLOC = 0x2
SHT_NOBITS = 8
SHT_NOTE = 7


def loaded_bytes(path):
    """What the loader maps, as code_hash hashes it."""
    data = open(path, "rb").read()
    assert data[:4] == b"\x7fELF", f"{path}: not an ELF file"
    assert data[4] == 2 and data[5] == 1, f"{path}: not a 64-bit little-endian ELF file"
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    assert shentsize == 64 and shnum > 0, f"{path}: unexpected section headers"

    def header(i):
        return struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)

    strtab = header(shstrndx)
    names = data[strtab[4]:strtab[4] + strtab[5]]
    out = []
    for i in range(shnum):
        name_off, kind, flags, addr, offset, size = header(i)[:6]
        if not flags & SHF_ALLOC or kind == SHT_NOTE:
            continue
        name = names[name_off:names.index(b"\0", name_off)]
        out.append(struct.pack("<QQQ", len(name), addr, size) + name)
        if kind != SHT_NOBITS:
            out.append(data[offset:offset + size])
    return b"".join(out)


def code_hash(path, b3sum=None):
    """The hash of what the loader maps of `path`, by official b3sum (`b3sum`,
    or MUTANTS_B3SUM: test-runner's, whose environment check.py sets)."""
    b3sum = b3sum or os.environ["MUTANTS_B3SUM"]
    out = subprocess.run([b3sum, "--no-names"], input=loaded_bytes(path), capture_output=True, check=True).stdout
    digest = out.decode().strip()
    assert len(digest) == 64, f"b3sum gave {digest!r}"
    return digest


def verdict(state):
    """The build's verdict, for test-runner, which holds the state's lock:
    "baseline" for the shard's first build (whose hashes it records),
    "same" when every listed binary hashes as that build's, else
    "differs". The library's own test binary first: most mutants change
    it, so most builds cost one hash."""
    binaries = sorted(f[:-5] for f in os.listdir(state) if f.endswith(".path"))
    binaries.sort(key=lambda name: not name.startswith("blake3_servil-"))
    assert binaries, f"{state}: no binary listed"
    if not os.path.exists(os.path.join(state, binaries[0] + ".base")):
        for name in binaries:
            path = open(os.path.join(state, name + ".path")).read().strip()
            open(os.path.join(state, name + ".base"), "w").write(open(os.path.join(state, name + ".stat")).read().strip() + " " + code_hash(path) + "\n")
        return "baseline"
    for name in binaries:
        stat = open(os.path.join(state, name + ".stat")).read().strip()
        base_stat, _, base_hash = open(os.path.join(state, name + ".base")).read().strip().rpartition(" ")
        if stat == base_stat:
            return "baseline"  # not relinked: the baseline's own file
        if code_hash(open(os.path.join(state, name + ".path")).read().strip()) != base_hash:
            return "differs"
    return "same"


if __name__ == "__main__":
    if sys.argv[1] == "--verdict":
        print(verdict(sys.argv[2]))
    else:
        print(code_hash(sys.argv[1]))
