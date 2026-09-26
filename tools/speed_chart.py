#!/usr/bin/env python3
"""The README's speed chart, drawn from a bench-hashes record.

    python3 tools/speed_chart.py bench-hashes/benchmark-results/AppleM4Max.darwin25 > media/speed.svg

Reads the record's text report (bench-hashes.result.txt): the solo table
"One input at a time (ns/B)", the row for SIZE (16 KiB), and draws one bar
per single-threaded contender, in GB/s (10^9 bytes per second), fastest
first. Each value is the record's median, in integer picoseconds per byte
until the drawing. A two-speed cell (a|b) stops the script: a bar has one
length. The footer, in pale type, names the commits the record measured
and its date.
"""
import re
import sys
from pathlib import Path

SIZE = "16 KiB"
# Report column label -> (name on the chart, colour: bench-hashes' Algorithm::color).
CONTENDERS = {
    "B3 servil st": ("BLAKE3 servil (this fork)", "#7c3aed"),
    "B3 official": ("BLAKE3 official crate", "#3b82f6"),
    "ab-blake3": ("ab-blake3", "#c026d3"),
    "SHA-256 ring": ("SHA-256 (ring)", "#c2410c"),
    "SHA-256 CC": ("SHA-256 (Apple CommonCrypto)", "#0e9aa7"),
    "SHA-256": ("SHA-256 (sha2)", "#e07a45"),
    "SHA3-256": ("SHA3-256 (sha3)", "#db2777"),
    "SHA-1DC": ("SHA-1 with collision detection", "#8a7a1e"),
}


def solo_row(text):
    """{column label: picoseconds per byte} for SIZE in the solo one-input table."""
    table = text.split("SOLO:", 1)[1].split("One input at a time (ns/B)", 1)[1]
    lines = table.splitlines()
    header = next(l for l in lines if l.strip().startswith("size"))
    # Labels are separated by two or more spaces; values by whitespace.
    labels = re.split(r"\s{2,}", header.strip())[1:]
    row = next(l for l in lines if l.startswith(f"  {SIZE} "))
    values = row.split()[len(SIZE.split()):]
    assert len(values) == len(labels), f"{len(values)} values under {len(labels)} labels"
    out = {}
    for label, value in zip(labels, values):
        value = value.rstrip("~")
        assert "|" not in value or label not in CONTENDERS, f"{label} at {SIZE} ran at two speeds ({value}): a bar has one length"
        if label in CONTENDERS:
            whole, frac = value.split(".")
            assert len(frac) == 3, f"{label}: {value} is not ns/B to three decimals"
            out[label] = int(whole) * 1000 + int(frac)
    return out


def main():
    record = Path(sys.argv[1])
    text = (record / "bench-hashes.result.txt").read_text()
    ps = solo_row(text)
    missing = set(CONTENDERS) - set(ps)
    assert not missing, f"the record lacks {sorted(missing)}"
    machine = text.splitlines()[0].split(" on ", 1)[1].split(" (", 1)[0]
    date = re.search(r"(\d{4}-\d{2}-\d{2}) \d", text.splitlines()[0]).group(1)
    fork = re.search(r"BLAKE3 servil st: blake3-servil [^\n]*?; commit ([0-9a-f]{7})", text).group(1)
    bench = re.search(r"https://github.com/johnservil/bench-hashes, commit ([0-9a-f]{7})", text).group(1)

    bars = sorted(ps.items(), key=lambda kv: kv[1])
    # Hundredths of GB/s: 10^5 / (ps per byte), rounded once.
    speed = {label: (100_000 + p // 2) // p for label, p in bars}
    top = max(speed.values())

    width, left, bar_h, gap, head = 720, 230, 26, 10, 78
    plot_w = width - left - 90
    height = head + len(bars) * (bar_h + gap) + 44
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="28" font-size="17" font-weight="600" fill="#111827">Hashing one 16 KiB input on one thread</text>',
           f'<text x="{left}" y="50" font-size="13" fill="#4b5563">{machine}, GB/s</text>',
           f'<text x="{width - 20}" y="50" font-size="12" fill="#4b5563" text-anchor="end">faster \u2192</text>']
    for i, (label, _) in enumerate(bars):
        name, color = CONTENDERS[label]
        y = head + i * (bar_h + gap)
        w = plot_w * speed[label] / top
        weight = ' font-weight="600"' if label == "B3 servil st" else ""
        out.append(f'<text x="{left - 10}" y="{y + bar_h * 0.68:.1f}" font-size="13" fill="#111827" text-anchor="end"{weight}>{name}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
        out.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h * 0.68:.1f}" font-size="13" fill="#111827"{weight}>'
                   f'{speed[label] // 100}.{speed[label] % 100:02d}</text>')
    out.append(f'<text x="{left}" y="{height - 16}" font-size="11" fill="#9ca3af">bench-hashes {bench} \u00b7 blake3-servil {fork} \u00b7 {date}</text>')
    out.append("</svg>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
