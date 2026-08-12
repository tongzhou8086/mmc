"""Render the performance bar charts for blog/post1.md.

The numbers are parsed out of the post's own tables, so the charts cannot drift
away from the text around them. Re-run after editing a table:

    python blog/render_perf.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

POST = Path(__file__).with_name("post1.md")
OUTDIR = Path(__file__).with_name("figures")

INK = "#1f2933"
MUTED = "#6b7684"
GRID = "#e3e7ec"
SERIES = [
    ("BK=64", "#7f9dbb"),
    ("BK=128", "#4f7a52"),
    ("torch.matmul (cuBLAS)", "#c9a668"),
]


def parse_table(heading):
    """Pull (shape, bk64, bk128, torch) rows from the table under a heading."""
    text = POST.read_text()
    start = text.index(heading)
    rows = []
    for line in text[start:].split("\n"):
        m = re.match(r"\|\s*(\d+)³\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                     r"\s*[+-][\d.]+%\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)),
                         float(m.group(3)), float(m.group(4))))
        elif rows and not line.startswith("|"):
            break
    if not rows:
        raise SystemExit(f"no table rows found under {heading!r}")
    return rows


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
        vals = [r[i + 1] for r in rows]
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
    bn256 = parse_table("| Shape | BK=64（6 个 TMA buffer）")
    bn512 = parse_table("| Shape | BK=64（4 个 TMA buffer）")
    small = next(r for r in bn256 if r[0] == 2048)
    bar_chart(bn256, "BN=256 · BK=64 vs BK=128 vs cuBLAS",
              OUTDIR / "perf-bn256",
              note=f"2048³ omitted: {small[1]:.0f} / {small[2]:.0f} / "
                   f"{small[3]:.0f} TFLOP/s, far below the rest")
    small = next(r for r in bn512 if r[0] == 2048)
    bar_chart(bn512, "BN=512 · BK=64 vs BK=128 vs cuBLAS",
              OUTDIR / "perf-bn512",
              note=f"2048³ omitted: {small[1]:.0f} / {small[2]:.0f} / "
                   f"{small[3]:.0f} TFLOP/s, far below the rest")


if __name__ == "__main__":
    main()
