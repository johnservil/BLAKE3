"""Writes one llvm-mca input per loop of a disassembly (llvm-objdump -d
--no-show-raw-insn): the loop's body, from its label to the branch back to
it, branches dropped (llvm-mca models straight-line code, repeated, which a
loop running many times approximates). A label with no branch back to it
runs to the next label.
Usage: pypy3 extract.py DISASSEMBLY OUT_DIR LABEL..."""
import re, sys
dis, out_dir, names = sys.argv[1], sys.argv[2], sys.argv[3:]
BRANCH = re.compile(r'(b|b\.\w+|bl|br|blr|ret|cbz|cbnz|tbz|tbnz)\b')
cur, body, closed = None, {}, set()
for line in open(dis):
    m = re.match(r'^[0-9a-f]+ <(\w+)>:', line)
    if m:
        cur = m.group(1); body[cur] = []; continue
    m = re.match(r'^\s+[0-9a-f]+:\s+(.*)$', line)
    if cur and cur not in closed and m:
        text = m.group(1)
        ins = text.split('//')[0].split('<')[0].strip()
        if BRANCH.match(ins):
            if f'<{cur}>' in text:      # the branch back: the loop's end
                closed.add(cur)
            continue
        body[cur].append(ins)
for n in names:
    open(f'{out_dir}/{n}.s', 'w').write('\n'.join(body[n]) + '\n')
    print(n, len(body[n]), "instructions")
