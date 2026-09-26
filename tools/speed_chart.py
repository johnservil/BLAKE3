#!/usr/bin/env python3
"""The README's speed charts and the page that says how they were made.

    python3 tools/speed_chart.py bench-hashes/benchmark-results/AppleM4Max.darwin25 media

reads a bench-hashes record's text report (bench-hashes.result.txt) and
writes into the given directory:

- speed.svg: one 16 KiB input on one thread;
- speed-every-core.svg: one 1 MiB input, BLAKE3 on every core and on one;
- speed-charts.md: how the charts were made, from the same record.

Each chart has a bar for BLAKE3 servil and one for the fastest
implementation of each other family (SHA-256, SHA3-256, SHA-1DC) in the
record's solo table "One input at a time (ns/B)", in GB/s (10^9 bytes per
second), fastest first. Both charts share one scale, and each family keeps
its colour in both. Values stay in integer picoseconds per byte until the
drawing; the report gives three decimals of ns/B, so speeds of 10 GB/s and
more are drawn as whole numbers and slower ones with two decimals. A
two-speed cell (a|b) stops the script, because a bar has one length.
"""
import re
import sys
from pathlib import Path

# Each chart: its file, input size, title, note, and its BLAKE3 bars with their names.
CHARTS = [
    dict(file="speed.svg", size="16 KiB", title="Hashing one 16 KiB input on one thread", note=None,
         blake3={"B3 servil st": "BLAKE3"}),
    dict(file="speed-every-core.svg", size="1 MiB", title="Hashing one 1 MiB input",
         note="BLAKE3's tree spreads one input over every core; the others use one.",
         blake3={"B3 servil mt": "BLAKE3, every core", "B3 servil st": "BLAKE3, one thread"}),
]
# Each other family: its name on the charts, and its implementations as the report's table
# labels them, with the names its provenance lines use.
FAMILIES = [
    ("SHA-256", {"SHA-256": "SHA-256", "SHA-256 ring": "SHA-256 ring", "SHA-256 CC": "SHA-256 CommonCrypto"}),
    ("SHA3-256", {"SHA3-256": "SHA3-256"}),
    ("SHA-1", {"SHA-1DC": "SHA-1DC"}),
]
# One colour per family (bench-hashes' colours for BLAKE3 servil, SHA-256
# ring, SHA3-256, and SHA-1DC).
COLORS = {"BLAKE3": "#7c3aed", "SHA-256": "#c2410c", "SHA3-256": "#db2777", "SHA-1": "#8a7a1e"}

WIDTH, LEFT, BAR, GAP = 720, 170, 26, 10
PLOT = WIDTH - LEFT - 60
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def family_of(label):
    return "BLAKE3" if label.startswith("B3 ") else next(name for name, members in FAMILIES if label in members)


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


def speed(label, cell):
    """A report cell in ns/B (three decimals) as hundredths of GB/s."""
    value = cell.rstrip("~")
    assert "|" not in value, f"{label} ran at two speeds ({value}): a bar has one length"
    whole, frac = value.split(".")
    assert len(frac) == 3, f"{label}: {value} is not ns/B to three decimals"
    ps = int(whole) * 1000 + int(frac)
    # 10^5 / (ps per byte), rounded once.
    return (100_000 + ps // 2) // ps


def shown(centi):
    """Hundredths of GB/s as drawn: whole numbers from 10 GB/s."""
    return f"{(centi + 50) // 100}" if centi >= 1000 else f"{centi // 100}.{centi % 100:02d}"


def bars_of(text, chart):
    """[(report label, name on the chart, hundredths of GB/s)], fastest first."""
    cells = solo_row(text, chart["size"])
    bars = [(label, name, speed(label, cells[label])) for label, name in chart["blake3"].items()]
    for name, members in FAMILIES:
        present = [label for label in members if label in cells]
        assert present, f"the record lacks {sorted(members)}"
        best = max(present, key=lambda label: speed(label, cells[label]))
        bars.append((best, name, speed(best, cells[best])))
    return sorted(bars, key=lambda bar: -bar[2])


def draw(chart, bars, top, subtitle):
    """One chart as SVG text; `top` (hundredths of GB/s) spans the full bar width."""
    head = 78 + (22 if chart["note"] else 0)
    height = head + len(bars) * (BAR + GAP) + 14
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" font-family="{FONT}">',
           f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
           f'<text x="{LEFT}" y="28" font-size="17" font-weight="600" fill="#111827">{chart["title"]}</text>',
           f'<text x="{WIDTH - 20}" y="28" font-size="12" fill="#4b5563" text-anchor="end">faster \u2192</text>',
           f'<text x="{LEFT}" y="50" font-size="13" fill="#4b5563">{subtitle}</text>']
    if chart["note"]:
        out.append(f'<text x="{LEFT}" y="72" font-size="13" fill="#4b5563">{chart["note"]}</text>')
    for i, (label, name, centi) in enumerate(bars):
        y = head + i * (BAR + GAP)
        w = PLOT * centi / top
        bold = ' font-weight="600"' if family_of(label) == "BLAKE3" else ""
        text_y = f"{y + BAR * 0.68:.1f}"
        out += [f'<text x="{LEFT - 10}" y="{text_y}" font-size="13" fill="#111827" text-anchor="end"{bold}>{name}</text>',
                f'<rect x="{LEFT}" y="{y}" width="{w:.1f}" height="{BAR}" rx="3" fill="{COLORS[family_of(label)]}"/>',
                f'<text x="{LEFT + w + 6:.1f}" y="{text_y}" font-size="13" fill="#111827"{bold}>{shown(centi)}</text>']
    return "\n".join(out + ["</svg>", ""])


def provenance(text, name):
    """The record's provenance line for a contender, after its name."""
    return re.search(rf"^  {re.escape(name)}: (.*)$", text, re.M).group(1)


def page(text, record, charts):
    """speed-charts.md: how the charts were made, from the record."""
    first = text.splitlines()[0]
    date = re.search(r"(\d{4}-\d{2}-\d{2}) \d", first).group(1)
    bench = re.search(r"https://github.com/johnservil/bench-hashes, commit ([0-9a-f]{40})", text).group(1)
    fork = re.search(r"^  BLAKE3 servil st: .*?; commit ([0-9a-f]{40})", text, re.M).group(1)
    rustc = re.search(r"^  (rustc [^;]*); target ([^;]*);", text, re.M)
    load = re.search(r"^  load during the run: (.*)$", text, re.M).group(1)
    lines = [
        "# How the speed charts were made",
        "",
        f"The README's two speed charts come from one run of [bench-hashes](https://github.com/johnservil/bench-hashes) "
        f"on {date}, on an {machine_of(text, split=True)}. `tools/speed_chart.py` draws them, and this page, from the run's report, "
        f"[`{record}/bench-hashes.result.txt`](https://github.com/johnservil/bench-hashes/blob/main/{record}/bench-hashes.result.txt), "
        "whose methodology is in bench-hashes' [METHODOLOGY.md](https://github.com/johnservil/bench-hashes/blob/main/METHODOLOGY.md). "
        "Each bar is the median time for one input of that size, taken over the run's rounds, as a speed. "
        "For SHA-256 each chart shows the fastest of three implementations in the run (sha2, ring, and Apple's CommonCrypto).",
        "",
        "| chart | bar | implementation | GB/s |",
        "|---|---|---|---:|",
    ]
    for chart, bars in charts:
        for label, name, centi in bars:
            if family_of(label) == "BLAKE3":
                call = "`hash_multithreaded`" if label == "B3 servil mt" else "`hash`"
                impl = f"blake3-servil (this repository) at [{fork[:7]}](https://github.com/johnservil/BLAKE3/commit/{fork}), {call}"
            else:
                impl = provenance(text, next(m for n, m in FAMILIES if label in m)[label]).split(";")[0]
                impl = {"CommonCrypto": "Apple CommonCrypto, from the running macOS"}.get(impl.split()[0], impl)
                if label == "SHA-1DC":
                    impl += " (SHA-1 with the collision detection git uses)"
            lines.append(f"| {chart['size']} | {name} | {impl} | {shown(centi)} |")
    lines += [
        "",
        f"- bench-hashes: commit [{bench[:7]}](https://github.com/johnservil/bench-hashes/commit/{bench})",
        f"- compiler: {rustc.group(1)}, target {rustc.group(2)}",
        f"- load during the run: {load}",
        "",
        "To draw the charts again from a newer record, with bench-hashes cloned inside this repository:",
        "",
        "```sh",
        f"python3 tools/speed_chart.py bench-hashes/{record} media",
        "```",
        "",
    ]
    return "\n".join(lines)


def machine_of(text, split=False):
    """The machine and its core count (with `split`, by kind), from the record."""
    first = text.splitlines()[0]
    machine = first.split(" on ", 1)[1].split(" (", 1)[0]
    cores = re.search(r"hw\.perflevel0\.physicalcpu: (\d+) · hw\.perflevel1\.physicalcpu: (\d+)", text)
    count = int(cores.group(1)) + int(cores.group(2)) if cores else int(re.search(r"(\d+) CPUs", first).group(1))
    if split and cores:
        return f"{machine} with {count} cores ({cores.group(1)} performance, {cores.group(2)} efficiency)"
    return f"{machine}, {count} cores"


def main():
    record_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    text = (record_dir / "bench-hashes.result.txt").read_text()
    record = "benchmark-results/" + record_dir.name
    charts = [(chart, bars_of(text, chart)) for chart in CHARTS]
    top = max(bars[0][2] for _, bars in charts)
    for chart, bars in charts:
        (out_dir / chart["file"]).write_text(draw(chart, bars, top, f"{machine_of(text)}; GB/s"))
    (out_dir / "speed-charts.md").write_text(page(text, record, charts))


if __name__ == "__main__":
    main()
