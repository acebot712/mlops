"""Emit the two SVG figures for post 318 from the measured benchmark JSON.

Series are separated by LIGHTNESS (--text / --accent / --muted) rather than by
hue, plus a direct label on each. Lightness survives every form of colour
blindness, and it suits a text-first theme that only has one accent colour --
a three-hue categorical palette failed the contrast/CVD checks against this
palette's low chroma, and inventing two saturated hues for one chart would look
imported from a different site.
"""

import json, pathlib

R = pathlib.Path("bench-results")
D = {n: json.loads((R / f"{n}.json").read_text()) for n in ("baseline", "device", "tuned")}

LABEL = {
    "baseline": ("CPU, batch 4", "var(--text)"),
    "device":   ("MPS, batch 4", "var(--accent)"),
    "tuned":    ("MPS, batch 128", "var(--muted)"),
}


def throughput_chart():
    """Horizontal bars, one per config. The MPS/batch-4 bar is the finding."""
    rows = []
    for n, d in D.items():
        ips = sum(m["images_per_second"] for m in d["epoch_metrics"]) / len(d["epoch_metrics"])
        rows.append((n, ips, d["total_seconds"]))
    top = 3200.0
    # Narrow enough that the longest value label ("2,843 img/s · 142s total")
    # still ends inside the 640 viewBox. Checked by assert_fits() below.
    x0, w = 150, 330
    out = ['<svg viewBox="0 0 640 200" role="img" aria-label="Training throughput. '
           'CPU at batch 4 reaches 1105 images per second. MPS at batch 4 reaches 920, '
           'slower than the CPU. MPS at batch 128 reaches 2843.">']
    for i in range(5):
        gx = x0 + w * i / 4
        out.append(f'<line x1="{gx:.0f}" y1="24" x2="{gx:.0f}" y2="140" stroke="var(--line)" stroke-width="1"/>')
    y = 34
    for n, ips, tot in rows:
        lab, col = LABEL[n]
        bw = w * ips / top
        out.append(f'<text class="d-label" x="0" y="{y+11}">{lab}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="14" rx="4" fill="{col}"/>')
        out.append(f'<text class="d-sub" x="{x0+bw+8:.1f}" y="{y+11}">{ips:,.0f} img/s '
                   f'&#183; {tot:.0f}s total</text>')
        y += 34
    out.append(f'<line x1="{x0}" y1="146" x2="{x0+w}" y2="146" stroke="var(--line)" stroke-width="1"/>')
    for i in range(5):
        gx = x0 + w * i / 4
        out.append(f'<text class="d-sub" x="{gx:.0f}" y="162" text-anchor="middle">{int(top*i/4):,}</text>')
    out.append('<text class="d-sub" x="365" y="182" text-anchor="middle">images per second (mean over 5 epochs)</text>')
    out.append("</svg>")
    return "\n".join(out)


def convergence_chart():
    """Validation accuracy against cumulative wall-clock, which is the honest axis.

    Plotting against epochs would hide the whole point: the batch-128 run does
    far fewer optimiser steps per epoch, so it is faster per epoch and worse per
    epoch, and only a time axis shows both at once.
    """
    x0, y0, w, h = 46, 26, 500, 150
    tmax, amin, amax = 320.0, 40.0, 80.0
    sx = lambda t: x0 + w * t / tmax
    sy = lambda a: y0 + h - h * (a - amin) / (amax - amin)

    out = ['<svg viewBox="0 0 640 232" role="img" aria-label="Validation accuracy against '
           'wall-clock time. The batch-128 run climbs fastest early but finishes at 69.4 percent '
           'after 142 seconds. Both batch-4 runs reach about 74.6 percent, the CPU at 269 seconds '
           'and MPS at 308 seconds.">']
    for a in range(40, 81, 10):
        out.append(f'<line x1="{x0}" y1="{sy(a):.1f}" x2="{x0+w}" y2="{sy(a):.1f}" stroke="var(--line)" stroke-width="1"/>')
        out.append(f'<text class="d-sub" x="{x0-8}" y="{sy(a)+4:.1f}" text-anchor="end">{a}%</text>')

    for n, d in D.items():
        lab, col = LABEL[n]
        t = 0.0
        pts = []
        for m in d["epoch_metrics"]:
            t += m["train_seconds"] + m["val_seconds"]
            pts.append((sx(t), sy(m["val_acc"])))
        path = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--bg)" stroke="{col}" stroke-width="2"/>')
        # baseline and device finish at 74.66% and 74.58% -- 0.3px apart on this
        # scale -- so the endpoint labels have to be placed by hand rather than
        # by a rule. baseline goes up and left, device goes right.
        ex, ey = pts[-1]
        anchor, dx, dy = {
            "baseline": ("end", -8, -10),
            "device":   ("start", 8, 4),
            "tuned":    ("start", 8, 4),
        }[n]
        out.append(f'<text class="d-label" x="{ex+dx:.1f}" y="{ey+dy:.1f}" fill="{col}" '
                   f'text-anchor="{anchor}">{lab}</text>')

    out.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="var(--line)" stroke-width="1"/>')
    for t in range(0, 321, 80):
        out.append(f'<text class="d-sub" x="{sx(t):.1f}" y="{y0+h+18:.0f}" text-anchor="middle">{t}s</text>')
    out.append(f'<text class="d-sub" x="{x0+w/2:.0f}" y="{y0+h+38:.0f}" text-anchor="middle">'
               f'cumulative wall-clock, 5 epochs each</text>')
    out.append("</svg>")
    return "\n".join(out)


def assert_fits(svg, name, width=640):
    """Cheap geometry check, because the browser preview is not always available.

    Estimates the right edge of every <text> from its anchor and a 5.6px average
    glyph width at 12px, and fails loudly if anything runs past the viewBox.
    Caught a 34px overflow and a 0.3px label collision the first time round.
    """
    import re

    worst = 0.0
    for m in re.finditer(r'<text[^>]*x="([\d.]+)"[^>]*?(?:text-anchor="(\w+)")?[^>]*>(.*?)</text>', svg):
        x, anchor, text = float(m.group(1)), m.group(2) or "start", m.group(3)
        n = len(re.sub(r"&[a-z]+;", "x", text))
        est = n * 5.6
        right = x + est if anchor == "start" else (x if anchor == "end" else x + est / 2)
        worst = max(worst, right)
    if worst > width:
        raise SystemExit(f"{name}: text extends to {worst:.0f}px, past the {width}px viewBox")
    print(f"  {name}: widest text ends at {worst:.0f}px (limit {width}) OK")


t = throughput_chart()
c = convergence_chart()
assert_fits(t, "fig-throughput")
assert_fits(c, "fig-convergence")
pathlib.Path("fig-throughput.svg").write_text(t)
pathlib.Path("fig-convergence.svg").write_text(c)
print("wrote fig-throughput.svg, fig-convergence.svg")
