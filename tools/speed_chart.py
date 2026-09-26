#!/usr/bin/env python3
"""The README's speed charts, drawn from a bench-hashes record.

    python3 tools/speed_chart.py RECORD one-thread > media/speed.svg
    python3 tools/speed_chart.py RECORD every-core > media/speed-every-core.svg

RECORD is a record directory, such as
bench-hashes/benchmark-results/AppleM4Max.darwin25. The script reads its
text report (bench-hashes.result.txt), from the solo table "One input at a
time (ns/B)" and the row for the chart's input size, and draws one bar for
BLAKE3 servil and one for the fastest implementation of each other family
(SHA-256, SHA3-256, SHA-1DC) at that size, in GB/s (10^9 bytes per second),
fastest first.

Values stay in integer picoseconds per byte until the drawing. The report
gives three decimals of ns/B, so speeds of 10 GB/s and more are drawn as
whole numbers, and slower ones with two decimals. A two-speed cell (a|b)
stops the script, because a bar has one length. The footer, in pale type,
names the commits the record measured and its date.
"""
import re
import sys
from pathlib import Path

CHARTS = {
    "one-thread": dict(size="16 KiB", servil="B3 servil st", servil_name="BLAKE3 servil",
                       title="Hashing one 16 KiB input on one thread", note=None),
    "every-core": dict(size="1 MiB", servil="B3 servil mt", servil_name="BLAKE3 servil, every core",
                       title="Hashing one 1 MiB input",
                       note="BLAKE3's tree spreads one input over every core; the others use one."),
}
# Each other family's implementations, as the report labels them.
FAMILIES = [["SHA-256", "SHA-256 ring", "SHA-256 CC"], ["SHA3-256"], ["SHA-1DC"]]
# Report label -> (name on the chart, colour: bench-hashes' Algorithm::color).
LOOK = {
    "B3 servil st": (None, "#7c3aed"),
    "B3 servil mt": (None, "#4c1d95"),
    "SHA-256": ("SHA-256 (sha2)", "#e07a45"),
    "SHA-256 ring": ("SHA-256 (ring)", "#c2410c"),
    "SHA-256 CC": ("SHA-256 (Apple CommonCrypto)", "#0e9aa7"),
    "SHA3-256": ("SHA3-256 (sha3)", "#db2777"),
    "SHA-1DC": ("SHA-1 with collision detection", "#8a7a1e"),
}


def solo_row(text, size):
    """{column label: report cell} for `size` in the solo one-input table."""
    table = text.split("SOLO:", 1)[1].split("One input at a time (ns/B)", 1)[1]
    lines = table.splitlines()
    header = next(l for l in lines if l.strip().startswith("size"))
    # Labels are separated by two or more spaces; values by whitespace.
    labels = re.split(r"\s{2,}", header.strip())[1:]
    row = next(l for l in lines if l.startswith(f"  {size} "))
    values = row.split()[len(size.split()):]
    assert len(values) == len(labels), f"{len(values)} values under {len(labels)} labels"
    return dict(zip(labels, values))


def picoseconds(label, cell):
    """A report cell in ns/B, three decimals, as integer ps/B."""
    value = cell.rstrip("~")
    assert "|" not in value, f"{label} ran at two speeds ({value}): a bar has one length"
    whole, frac = value.split(".")
    assert len(frac) == 3, f"{label}: {value} is not ns/B to three decimals"
    return int(whole) * 1000 + int(frac)


def shown(centi):
    """Hundredths of GB/s as drawn: whole numbers from 10 GB/s."""
    return f"{(centi + 50) // 100}" if centi >= 1000 else f"{centi // 100}.{centi % 100:02d}"


def main():
    record, which = Path(sys.argv[1]), sys.argv[2]
    chart = CHARTS[which]
    text = (record / "bench-hashes.result.txt").read_text()
    cells = solo_row(text, chart["size"])
    ps = {chart["servil"]: picoseconds(chart["servil"], cells[chart["servil"]])}
    for family in FAMILIES:
        present = [label for label in family if label in cells]
        assert present, f"the record lacks {family}"
        best = min(present, key=lambda label: picoseconds(label, cells[label]))
        ps[best] = picoseconds(best, cells[best])
    first = text.splitlines()[0]
    machine = first.split(" on ", 1)[1].split(" (", 1)[0]
    date = re.search(r"(\d{4}-\d{2}-\d{2}) \d", first).group(1)
    fork = re.search(r"BLAKE3 servil st: blake3-servil [^\n]*?; commit ([0-9a-f]{7})", text).group(1)
    bench = re.search(r"https://github.com/johnservil/bench-hashes, commit ([0-9a-f]{7})", text).group(1)

    bars = sorted(ps, key=ps.get)
    # Hundredths of GB/s: 10^5 / (ps per byte), rounded once.
    speed = {label: (100_000 + ps[label] // 2) // ps[label] for label in bars}
    top = max(speed.values())

    width, left, bar_h, gap = 720, 230, 26, 10
    head = 78 + (22 if chart["note"] else 0)
    plot_w = width - left - 60
    height = head + len(bars) * (bar_h + gap) + 44
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="28" font-size="17" font-weight="600" fill="#111827">{chart["title"]}</text>',
           f'<text x="{left}" y="50" font-size="13" fill="#4b5563">{machine}, GB/s</text>',
           f'<text x="{width - 20}" y="50" font-size="12" fill="#4b5563" text-anchor="end">faster \u2192</text>']
    if chart["note"]:
        out.append(f'<text x="{left}" y="72" font-size="13" fill="#4b5563">{chart["note"]}</text>')
    for i, label in enumerate(bars):
        name, color = LOOK[label]
        servil = label == chart["servil"]
        name = chart["servil_name"] if servil else name
        y = head + i * (bar_h + gap)
        w = plot_w * speed[label] / top
        weight = ' font-weight="600"' if servil else ""
        out.append(f'<text x="{left - 10}" y="{y + bar_h * 0.68:.1f}" font-size="13" fill="#111827" text-anchor="end"{weight}>{name}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
        out.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h * 0.68:.1f}" font-size="13" fill="#111827"{weight}>{shown(speed[label])}</text>')
    out.append(f'<text x="{left}" y="{height - 16}" font-size="11" fill="#9ca3af">'
               f'bench-hashes {bench} \u00b7 blake3-servil {fork} \u00b7 {date}</text>')
    out.append("</svg>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
