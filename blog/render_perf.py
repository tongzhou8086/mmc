"""Render the performance bar charts for blog/post1.md.

The numbers live in blog/perf-data.tsv, which is the single source for them -
the post embeds the rendered charts rather than a table. Re-run after editing
the data:

    python blog/render_perf.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

DATA = Path(__file__).with_name("perf-data.tsv")
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


def read_rows(design):
    """(shape, bk64, bk128, torch) rows for one design, from perf-data.tsv."""
    rows = []
    for line in DATA.read_text().split("\n"):
        if not line or line.startswith("#"):
            continue
        name, shape, a, b, t = line.split("\t")
        if name == design:
            rows.append((int(shape), float(a), float(b), float(t)))
    if not rows:
        raise SystemExit(f"no rows for {design!r} in {DATA}")
    return sorted(rows)


def bar_chart(all_rows, title, path, note=None):
    # 2048 sits far below everything else; keeping it would flatten the range
    # the surrounding text is about, and an empty column reads worse than a
    # sentence. It is called out in the caption instead.
    rows = [r for r in all_rows if r[0] >= 3072]
    shapes = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    n = len(SERIES)
    width = 0.82 / n
    for i, (label, colour) in enumerate(SERIES):
        vals = [r[SERIES_COL[i]] for r in rows]
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(rows))]
        ax.bar(xs, vals, width=width * 0.94, label=label, color=colour,
               edgecolor="white", linewidth=0.6, zorder=3)

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
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
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
    bn256 = read_rows("bn256")
    bn512 = read_rows("bn512")
    small = next(r for r in bn256 if r[0] == 2048)
    bar_chart(bn256, "BN=256 · cuBLAS vs BK=64 vs BK=128",
              OUTDIR / "perf-bn256",
              note=f"2048³ omitted: {small[1]:.0f} / {small[2]:.0f} / "
                   f"{small[3]:.0f} TFLOP/s, far below the rest")
    small = next(r for r in bn512 if r[0] == 2048)
    bar_chart(bn512, "BN=512 · cuBLAS vs BK=64 vs BK=128",
              OUTDIR / "perf-bn512",
              note=f"2048³ omitted: {small[1]:.0f} / {small[2]:.0f} / "
                   f"{small[3]:.0f} TFLOP/s, far below the rest")


if __name__ == "__main__":
    main()
