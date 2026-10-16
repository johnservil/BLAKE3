"""The instruction forms a passing run of the Rust proofs (prove_rust.py)
or the whole-call proofs (prove_timing.py) ran, kept between runs so that
cross_check.py and isla_check.py, which check those forms' models, need
not run the proofs again. A record is named by the hash of the built
library and of the verify tools that made it, so a record for other code
or other tools is never found; a missing record means: run the proofs.
"""

import json
import os
import subprocess

DIR = "/tmp/target/verify-results"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B3SUM = "/tmp/target/release/b3sum"


def blake3(paths=(), data=None):
    """BLAKE3 hashes, from this repository's b3sum (built when absent): of
    each file in `paths`, or of `data`."""
    if not os.path.exists(B3SUM):
        subprocess.run(["cargo", "build", "-q", "--release", "--manifest-path", os.path.join(ROOT, "b3sum", "Cargo.toml")],
                       env={**os.environ, "CARGO_TARGET_DIR": "/tmp/target"}, check=True)
    out = subprocess.run([B3SUM, "--no-names", *paths], input=data, capture_output=True, check=True)
    return out.stdout.decode().split()


def assembled(command, source, suffix):
    """`source`, assembly text, built by `command` (a compiler and its
    flags; the input and `-o` follow) into a file named for the BLAKE3 of
    both, in /tmp: each content built once and reused, so runs leave a
    file per content, not per call."""
    key = blake3(data="\0".join([*command, source]).encode())[0][:32]
    out = f"/tmp/verify-build/{key}{suffix}"
    if not os.path.exists(out):
        os.makedirs("/tmp/verify-build", exist_ok=True)
        part = f"{out}.{os.getpid()}"
        with open(part + ".S", "w") as f:
            f.write(source)
        subprocess.run([*command, part + ".S", "-o", part], check=True)
        os.replace(part, out)
    return out


def _key(library):
    """BLAKE3 of the library's and the verify tools' hashes, in order."""
    tools = sorted(os.path.join(HERE, n) for n in os.listdir(HERE) if n.endswith(".py"))
    files = [library] + tools + [os.path.join(HERE, "rust", "src", "lib.rs")]
    return blake3(data="\n".join(blake3(files)).encode())[0]


def save_forms(kind, library, forms):
    """The forms a passing run of `kind` ("rust", "timing") ran on `library`."""
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, f"forms-{kind}-{_key(library)}.json"), "w") as f:
        json.dump(sorted([m, list(o)] for m, o in forms), f)


def load_forms(kind, library):
    """The forms of a passing run of `kind` on `library` with these tools, or None."""
    p = os.path.join(DIR, f"forms-{kind}-{_key(library)}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {(m, tuple(o)) for m, o in json.load(f)}
