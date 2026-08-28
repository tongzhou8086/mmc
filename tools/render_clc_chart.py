"""Chart the cluster-launch-control sweep.

    python tools/render_clc_chart.py clc-sweep.log

Reads the markdown tables benchmarks/sweep_clc.py prints and writes
blog/figures/perf-clc.{png,svg}: per shape, cuBLAS against the best config of
design 4 and the best config of design 5 (design 4 + CLC). Same visual language
as blog/render_perf.py, so the figure can drop straight into the series.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

OUTDIR = Path(__file__).resolve().parent.parent / "blog" / "figures"

INK = "#1f2933"
MUTED = "#6b7684"
GRID = "#e3e7ec"
# cuBLAS in neutral grey - it is the reference, not a third competitor. The two
# designs keep the blue/green pair the rest of the series uses.
SERIES = [
    ("torch.matmul (cuBLAS)", "#b7bfc9"),
    ("design 4 (static persistent)", "#7f9dbb"),
    ("design 5 (+ cluster launch control)", "#4f7a52"),
]


def parse(path):
    """{shape: {config: tflops}} for both designs, plus cuBLAS."""
    text = Path(path).read_text()
    out = {}
    for tag in ("clc", "2round"):
        m = re.search(rf"^## {tag}\n(.*?)(?=\n## |\Z)", text,
                      re.S | re.M)
        if not m:
            raise SystemExit(f"no '## {tag}' table in {path}")
        header = None
        for line in m.group(1).strip().split("\n"):
            if line.startswith("| Shape"):
                header = [c.strip() for c in line.strip("|").split("|")][1:]
                continue
            row = re.match(r"\|\s*(\d+)³\s*\|(.*)\|", line)
            if not row or header is None:
                continue
            shape = int(row.group(1))
            vals = [float(c) for c in row.group(2).split("|")]
            slot = out.setdefault(shape, {})
            for name, v in zip(header, vals):
                slot[("cublas" if name.startswith("torch")
                      else f"{tag} {name}")] = v
    return out


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "clc-sweep.log"
    data = parse(log)
    shapes = sorted(data)

    def best(shape, tag):
        return max(v for k, v in data[shape].items() if k.startswith(tag))

    rows = [(s, data[s]["cublas"], best(s, "2round"), best(s, "clc"))
            for s in shapes]

    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    n = len(SERIES)
    width = 0.82 / n
    for i, (label, colour) in enumerate(SERIES):
        vals = [r[i + 1] for r in rows]
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(rows))]
        ax.bar(xs, vals, width=width * 0.94, label=label, color=colour,
               edgecolor="white", linewidth=0.5, zorder=3)
        if i == 2:
            # only the CLC bars carry a number: the delta against design 4 is
            # the one quantity the figure exists to show
            for x, v, r in zip(xs, vals, rows):
                d = (v - r[2]) / r[2]
                ax.text(x, v + 6, f"{d:+.1%}", ha="center", va="bottom",
                        fontsize=8, color=INK if d > 0 else MUTED,
                        fontweight="bold" if d > 0.005 else "normal")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{s}³" for s, *_ in rows], fontsize=9)
    ax.set_xlabel("square shape (M = N = K)", fontsize=10, color=MUTED,
                  labelpad=8)
    ax.set_ylabel("TFLOP/s", fontsize=10, color=MUTED)
    lo = min(min(r[1:]) for r in rows)
    hi = max(max(r[1:]) for r in rows)
    ax.set_ylim(max(0.0, lo - 90), hi + 90)
    ax.set_title("Cluster launch control on the fourth design · best config per shape",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED)
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    ax.text(0.0, -0.28,
            "best of BK 64/128 x GSM 8/12/16 for each design; labels are "
            "design 5 against design 4, same node, same process",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=MUTED)

    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        out = OUTDIR / f"perf-clc.{suffix}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
