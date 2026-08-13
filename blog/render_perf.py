"""Render the performance bar charts for blog/post1.md.

The numbers live in blog/perf-data.md, as the markdown tables they came from -
the post embeds the rendered charts, and that file keeps the exact figures. Re-run
after editing a table there:

    python blog/render_perf.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

DATA = Path(__file__).with_name("perf-data.md")
OUTDIR = Path(__file__).with_name("figures")

INK = "#1f2933"
MUTED = "#6b7684"
GRID = "#e3e7ec"
# cuBLAS first, in neutral grey: it is the reference the other two are read
# against, not a third competitor. The two configurations under discussion keep
# saturated colours so they stand out from it.
SERIES = [
    ("torch.matmul (cuBLAS)", "#b7bfc9"),
    ("BK=64", "#7f9dbb"),
    ("BK=128", "#4f7a52"),
]
# column index in the parsed rows for each series above
SERIES_COL = [3, 1, 2]

# The splitacc table also sweeps GSM, so it needs its own series list: cuBLAS
# first in grey as always, then each BK shaded light to dark by GSM depth, so
# the two families stay distinguishable by hue and GSM by lightness.
SERIES_GSM = [
    ("torch.matmul (cuBLAS)", "#b7bfc9", 9),
    ("BK=64  GSM=8", "#c3d3e2", 1),
    ("BK=64  GSM=12", "#9ab4cd", 2),
    ("BK=64  GSM=16", "#6d90b0", 3),
    ("BK=64  GSM=20", "#456682", 4),
    ("BK=128  GSM=8", "#b3d2b7", 5),
    ("BK=128  GSM=12", "#87b28d", 6),
    ("BK=128  GSM=16", "#5b8a61", 7),
    ("BK=128  GSM=20", "#375a3c", 8),
]


def read_rows(heading, ncols=3):
    """Rows under a heading as (shape, v1, ..., vn).

    ncols=3 is the delta-carrying three-series table; ncols=7 is the splitacc
    table, which has no delta column because there is no single pair to compare.
    """
    text = DATA.read_text()
    body = r"\s*([\d.]+)\s*\|" * ncols
    if ncols == 3:
        body = (r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[+-][\d.]+%\s*\|"
                r"\s*([\d.]+)\s*\|")
    rows = []
    for line in text[text.index(heading):].split("\n"):
        m = re.match(r"\|\s*(\d+)³\s*\|" + body, line)
        if m:
            rows.append(tuple([int(m.group(1))]
                              + [float(g) for g in m.groups()[1:]]))
        elif rows and not line.startswith("|"):
            break
    if not rows:
        raise SystemExit(f"no table rows found under {heading!r} in {DATA}")
    return sorted(rows)


def bar_chart(rows, title, path, note=None, series=None, figsize=(12.4, 4.9)):
    shapes = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=figsize)
    series = series or [(l, c, k) for (l, c), k in zip(SERIES, SERIES_COL)]
    n = len(series)
    width = 0.86 / n
    for i, (label, colour, col) in enumerate(series):
        vals = [r[col] for r in rows]
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(rows))]
        ax.bar(xs, vals, width=width * 0.94, label=label, color=colour,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{s // 1024}K" for s in shapes], fontsize=9)
    ax.set_xlabel("square shape (M = N = K)", fontsize=10, color=MUTED,
                  labelpad=8)
    ax.set_ylabel("TFLOP/s", fontsize=10, color=MUTED)
    # start above zero: every bar here is >1100, so a zero baseline would hide
    # the differences the surrounding text is about
    lo = min(min(r[1:]) for r in rows)
    hi = max(max(r[1:]) for r in rows)
    ax.set_ylim(max(0.0, lo - 90), hi + 70)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left",
                 pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED)
    ax.legend(frameon=False, fontsize=9.5, ncol=min(len(series), 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    if note:
        ax.text(0.0, -0.30, note, transform=ax.transAxes, ha="left",
                va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for heading, title, stem in (
            ("## BN=256", "BN=256 · GSM sweep, both BK settings", "perf-bn256"),
            ("## BN=512", "BN=512 · GSM sweep, both BK settings", "perf-bn512")):
        bar_chart(read_rows(heading, ncols=9), title, OUTDIR / stem,
                  series=SERIES_GSM, figsize=(16.0, 5.6))
    bar_chart(read_rows("## BN=512 splitacc", ncols=9),
              "BN=512 splitacc · GSM sweep, both BK settings",
              OUTDIR / "perf-bn512-splitacc",
              series=SERIES_GSM, figsize=(16.0, 5.6))


if __name__ == "__main__":
    main()
