#!/usr/bin/env python3
"""Make every caller of this crate call `initialize()` first, as the crate's
contract requires: each doc example (a hidden line), and each `main`,
`#[test]` function of the integration tests, examples, b3sum, and
test_vectors. A doc example that makes a
multithreaded call or uses a `stream` calls `initialize_multithreaded(None)`
instead (its hidden `initialize()` line becomes that). A function that
already calls an `initialize` is left as it is, so the script can run again
after a merge (take either side's version of a conflicting file, then run
this); a function's multithreaded calls are checked by its test run.

    pypy3 tools/call_initialize.py
"""
import glob, os, re

ROOT = os.path.join(os.path.dirname(__file__), "..")

def doc_examples(path):
    lines = open(path).read().split("\n")
    out, i, changed = [], 0, False
    fence = re.compile(r"^(\s*//[/!]\s*)```([\w,]*)\s*$")
    while i < len(lines):
        m = fence.match(lines[i])
        out.append(lines[i])
        if not m:
            i += 1
            continue
        prefix, lang = m.group(1), m.group(2)
        j = i + 1
        while not fence.match(lines[j]):
            j += 1
        block = lines[i + 1:j]
        rust = set(lang.split(",")) <= {"", "rust", "no_run", "should_panic", "compile_fail", "edition2021", "edition2024"}
        threads = any(re.search(r"multithreaded|\bstream\(", b) for b in block)
        call = "initialize_multithreaded" if threads else "initialize"
        if rust and "compile_fail" not in lang and not any("initialize" in b for b in block):
            at = next((k + 1 for k, b in enumerate(block) if re.search(r"\bfn main\b", b)), 0)
            block = block[:at] + [prefix + f"# blake3_servil::{call}();"] + block[at:]
            changed = True
        elif rust and threads and any(b.endswith("# blake3_servil::initialize();") for b in block):
            block = [b.replace("# blake3_servil::initialize();", "# blake3_servil::initialize_multithreaded(None);") for b in block]
            changed = True
        out += block + [lines[j]]
        i = j + 1
    if changed:
        open(path, "w").write("\n".join(out))

def first_statements(path, crate):
    text = open(path).read()
    pattern = re.compile(r"((?:#\[(?:test|bench)\][^\n]*\n\s*)(?:#\[[^\n]*\n\s*)*(?:pub )?fn \w+|fn main)\s*\([^)]*\)[^{;]*\{\n")
    out, last = [], 0
    for m in pattern.finditer(text):
        body = text[m.end():]
        indent = re.match(r"[ \t]*", body).group(0) or "    "
        out.append(text[last:m.end()])
        if "initialize" not in body.split("\n", 1)[0]:
            out.append(f"{indent}{crate}::initialize();\n")
        last = m.end()
    out.append(text[last:])
    new = "".join(out)
    if new != text:
        open(path, "w").write(new)

for path in sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.rs"), recursive=True)):
    doc_examples(path)
for pattern, crate in [("tests/*.rs", "blake3_servil"), ("examples/*.rs", "blake3_servil"),
                       ("b3sum/src/*.rs", "blake3"), ("b3sum/tests/*.rs", "blake3"), ("test_vectors/src/**/*.rs", "blake3")]:
    for path in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
        first_statements(path, crate)
