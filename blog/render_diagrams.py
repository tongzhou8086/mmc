"""Render the explanatory diagrams for blog/post1.md.

    python blog/render_diagrams.py

Writes PNG + SVG into blog/figures/. Same visual language as render_perf.py:
slate ink, muted labels, a light-grey grid, and the blue/green pair used for
the two operand families.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUTDIR = Path(__file__).with_name("figures")

INK = "#1f2933"
MUTED = "#6b7684"
EDGE = "#8d97a3"
BASE = "#eef1f6"          # untouched part of a matrix
PANEL = "#a9cbb4"         # the A row-panel and B column-panel being multiplied
TILE = "#e9b787"          # the C tile they produce

# tile counts, in units of one BM x BN block. N is deliberately much larger
# than K: it keeps the figure a wide rectangle like the charts, and it is also
# the honest shape - K here stands for a single BK step, not the whole K.
NM, NN = 4, 10            # C is NN blocks wide, NM tall
K = 1.8                   # A is K wide, B is K tall
NBK = 4                   # dashed BK subdivisions across K
GAP = 1.05
ROW, COL = 1, 4           # which block row / column is highlighted


def _grid(ax, x, y, w, h, ncols, nrows, **kw):
    """A filled block with interior rules."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=BASE, edgecolor=EDGE,
                           linewidth=1.2, zorder=2, **kw))
    for c in range(1, ncols):
        ax.plot([x + w * c / ncols] * 2, [y, y + h], color=EDGE,
                linewidth=0.8, zorder=3)
    for r in range(1, nrows):
        ax.plot([x, x + w], [y + h * r / nrows] * 2, color=EDGE,
                linewidth=0.8, zorder=3)


def _dim(ax, x0, y0, x1, y1, label, offset=(0, 0), fontsize=12):
    """A double-headed dimension arrow with a label beside it."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>",
                                 mutation_scale=11, linewidth=1.4,
                                 color=INK, zorder=6, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2 + offset[0], (y0 + y1) / 2 + offset[1], label,
            ha="center", va="center", fontsize=fontsize, style="italic",
            color=INK, zorder=6)


def tiled_gemm(path):
    w, h = NN * 1.0, NM * 1.0          # C block, one unit per BM x BN tile
    # C bottom-left at the origin; A sits to its left, B directly above it
    cx, cy = 0.0, 0.0
    ax_, ay = -(K + GAP), 0.0
    bx, by = 0.0, h + GAP

    fig, ax = plt.subplots(figsize=(12.0, 7.2))

    _grid(ax, ax_, ay, K, h, 1, NM)                    # A [M x K]
    _grid(ax, bx, by, w, K, NN, 1)                     # B [K x N]
    _grid(ax, cx, cy, w, h, NN, NM)                    # C [M x N]

    # the row of A and the column of B that produce one C tile
    row_y = ay + h - (ROW + 1) * (h / NM)
    col_x = cx + COL * (w / NN)
    ax.add_patch(Rectangle((ax_, row_y), K, h / NM, facecolor=PANEL,
                           edgecolor=EDGE, linewidth=1.2, zorder=4))
    ax.add_patch(Rectangle((col_x, by), w / NN, K, facecolor=PANEL,
                           edgecolor=EDGE, linewidth=1.2, zorder=4))
    ax.add_patch(Rectangle((col_x, row_y), w / NN, h / NM, facecolor=TILE,
                           edgecolor=INK, linewidth=2.2, zorder=5))

    # BK subdivisions, dashed, only inside the highlighted panels - they are
    # what a single MMA step consumes
    for i in range(1, NBK):
        x = ax_ + K * i / NBK
        ax.plot([x, x], [row_y, row_y + h / NM], color=MUTED, linewidth=0.9,
                linestyle=(0, (4, 3)), zorder=6)
        y = by + K * i / NBK
        ax.plot([col_x, col_x + w / NN], [y, y], color=MUTED, linewidth=0.9,
                linestyle=(0, (4, 3)), zorder=6)

    # matrix names
    ax.text(ax_ - 0.55, ay + h / 2, "A  [M × K]", rotation=90, ha="center",
            va="center", fontsize=13, style="italic", color=INK)
    ax.text(bx + w / 2, by + K + 0.35, "B  [K × N]", ha="center", va="bottom",
            fontsize=13, style="italic", color=INK)
    ax.text(cx + w / 2, cy - 0.45, "C  [M × N]", ha="center", va="top",
            fontsize=13, style="italic", color=INK)

    # tile dimensions: BK across one dashed chunk, BN across the C tile, BM down it
    # both horizontal dimensions sit on the same line, in the gap between the
    # top row and the matrices below it
    _dim(ax, ax_, ay + h + 0.32, ax_ + K / NBK, ay + h + 0.32, "BK",
         offset=(0, 0.34))
    _dim(ax, col_x, cy + h + 0.32, col_x + w / NN, cy + h + 0.32, "BN",
         offset=(0, 0.34))
    _dim(ax, col_x + w / NN + 0.3, row_y, col_x + w / NN + 0.3,
         row_y + h / NM, "BM", offset=(0.42, 0))

    ax.set_xlim(ax_ - 1.2, cx + w + 0.3)
    ax.set_ylim(cy - 0.95, by + K + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


# 2-CTA MMA. Ownership is the whole point of the figure, so colour means "which
# CTA holds this operand" - the same blue / amber pair the persistent-grid
# figure uses for CTA identity - while the accumulator keeps the MMA-buffer
# green it has in the timelines.
CTA0_FILL = "#a9c4de"     # CTA 0's operands, in SMEM
CTA1_FILL = "#e9c79a"     # CTA 1's operands, in SMEM
ACC_FILL = "#a9d3b0"      # the C tile, in TMEM

TC_BM, TC_BN, TC_K, TC_GAP = 1.0, 3.0, 0.55, 0.18


def _tc_block(ax, x, y, w, h, fill, label, fontsize=13):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE,
                           linewidth=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, style="italic", color=INK, zorder=4)


def two_cta_mma(path):
    """2-CTA MMA off vs on: the same output, half the B traffic.

    Left: two independent CTAs, each with its own full B tile - B crosses the
    memory bus twice. Right: one cluster computing a 2BM x BN output, with B
    split BN/2 per CTA and the MMA reading across the cluster, so B crosses
    once.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    BM, BN, KW, G = TC_BM, TC_BN, TC_K, TC_GAP

    def cta_label(ax, y, text):
        ax.text(-KW - G - 0.30, y + BM / 2, text, ha="right", va="center",
                fontsize=11, fontweight="bold", color=INK)

    def frame(ax, title, caption):
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                     loc="left", pad=12)
        ax.text((BN - KW - G) / 2, -2.55, caption, ha="center", va="top",
                fontsize=11, color=INK)
        ax.set_xlim(-1.95, BN + 0.75)
        ax.set_ylim(-3.05, 2.75)
        ax.set_aspect("equal")
        ax.axis("off")

    # ---- left: 2-CTA off ------------------------------------------------
    ax = axes[0]
    for y0, fill, name in ((0.55, CTA0_FILL, "CTA 0"),
                           (-1.85, CTA1_FILL, "CTA 1")):
        _tc_block(ax, 0, y0, BN, BM, ACC_FILL, "C")
        _tc_block(ax, -KW - G, y0, KW, BM, fill, "A", fontsize=11)
        _tc_block(ax, 0, y0 + BM + G, BN, KW, fill, "B")
        cta_label(ax, y0, name)
    _dim(ax, 0, -2.25, BN, -2.25, "BN", offset=(0, 0.22), fontsize=11)
    frame(ax, "2-CTA MMA off:  two independent CTAs",
          "each CTA loads the whole B tile — B crosses the bus 2×")

    # ---- right: 2-CTA on -------------------------------------------------
    ax = axes[1]
    _tc_block(ax, 0, 0, BN, BM, ACC_FILL, "C")
    _tc_block(ax, 0, -BM, BN, BM, ACC_FILL, "C")
    _tc_block(ax, -KW - G, 0, KW, BM, CTA0_FILL, "A", fontsize=11)
    _tc_block(ax, -KW - G, -BM, KW, BM, CTA1_FILL, "A", fontsize=11)
    for x0, fill in ((0, CTA0_FILL), (BN / 2, CTA1_FILL)):
        _tc_block(ax, x0, BM + G, BN / 2, KW, fill, "")
        ax.text(x0 + BN / 4, BM + G + KW / 2 + 0.09, "B", ha="center",
                va="center", fontsize=13, style="italic", color=INK, zorder=4)
        ax.text(x0 + BN / 4, BM + G + KW / 2 - 0.16, "BN/2", ha="center",
                va="center", fontsize=9, style="italic", color=MUTED,
                zorder=4)
    cta_label(ax, 0, "CTA 0")
    cta_label(ax, -BM, "CTA 1")

    # the MMA of the cluster reads both halves of B, so each CTA reaches
    # across to the other one's half - the arc says exactly that
    y_arc = BM + G + KW
    ax.add_patch(FancyArrowPatch((BN / 4, y_arc), (3 * BN / 4, y_arc),
                                 connectionstyle="arc3,rad=-0.45",
                                 arrowstyle="<|-|>", mutation_scale=11,
                                 linewidth=1.4, linestyle=(0, (4, 3)),
                                 color=MUTED, shrinkA=3, shrinkB=3, zorder=5))
    ax.text(BN / 2, y_arc + 0.72, "MMA reads both halves across the cluster",
            ha="center", va="bottom", fontsize=10, color=MUTED)

    _dim(ax, BN + 0.30, -BM, BN + 0.30, BM, "2BM", offset=(0.42, 0),
         fontsize=11)
    frame(ax, "2-CTA MMA on:  one cluster, one 2BM × BN output",
          "B is split BN/2 per CTA — B crosses the bus 1×")

    handles = [Rectangle((0, 0), 1, 1, facecolor=CTA0_FILL, edgecolor=EDGE,
                         linewidth=1.0),
               Rectangle((0, 0), 1, 1, facecolor=CTA1_FILL, edgecolor=EDGE,
                         linewidth=1.0),
               Rectangle((0, 0), 1, 1, facecolor=ACC_FILL, edgecolor=EDGE,
                         linewidth=1.0)]
    fig.legend(handles,
               ["CTA 0 operands (SMEM)", "CTA 1 operands (SMEM)",
                "output C (TMEM)"],
               loc="lower center", ncol=3, frameon=False, fontsize=10.5,
               labelcolor=MUTED, handlelength=1.1, handleheight=1.0,
               columnspacing=2.4, bbox_to_anchor=(0.5, 0.015))

    fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tiled_gemm(OUTDIR / "tiled-gemm")
    two_cta_mma(OUTDIR / "two-cta-mma")
    cta_assignment(OUTDIR / "cta-assignment")
    cta_swizzle(OUTDIR / "cta-swizzle")
    persistent_swizzle(OUTDIR / "persistent-swizzle")
    single_buffer_timeline(OUTDIR / "single-buffer-timeline")
    two_tma_buffer_timeline(OUTDIR / "two-tma-buffer-timeline")
    two_accumulator_timeline(OUTDIR / "two-accumulator-timeline")




# One distinct hue per CTA. Deliberately not two families shaded by row: that
# encodes a hierarchy the assignment does not have, and invites the reader to
# look for meaning in the shade. Mid-tone so the label reads on every tile.
CTA_COLOURS = ["#8fb0d1",   # blue
               "#8bc09a",   # green
               "#e8bd83",   # amber
               "#b7a6d8",   # purple
               "#7fc7cb",   # teal
               "#dfa0a8",   # rose
               "#cbc57e",   # olive
               "#c0a68f"]   # tan
CTA_TEXT = [INK] * 8

GRID_M, GRID_N, NCTA = 4, 8, 8


def cta_assignment(path):
    """The output grid,每个 tile 标上负责它的 CTA.

    tile (m, n) -> CTA (n % 2) * GRID_M + m: walk down a column of GRID_M
    tiles, then the next column, then start over at CTA0. With 8 CTAs and 4
    rows the pattern has a period of two tile columns, which is what the
    repetition in the figure shows.
    """
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    w = h = 1.0

    for n in range(GRID_N):
        for m in range(GRID_M):
            cta = (n % 2) * GRID_M + m
            x, y = n * w, (GRID_M - 1 - m) * h
            ax.add_patch(Rectangle((x, y), w, h, facecolor=CTA_COLOURS[cta],
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
            ax.text(x + w / 2, y + h / 2, f"CTA{cta}", ha="center",
                    va="center", fontsize=11, fontweight="bold",
                    color=CTA_TEXT[cta], zorder=3)

    ax.add_patch(Rectangle((0, 0), GRID_N * w, GRID_M * h, facecolor="none",
                           edgecolor=EDGE, linewidth=1.2, zorder=4))

    # tile coordinates, so the (m, n) in the text has something to point at
    for n in range(GRID_N):
        ax.text(n * w + w / 2, GRID_M * h + 0.15, str(n), ha="center",
                va="bottom", fontsize=10, color=MUTED)
    for m in range(GRID_M):
        ax.text(-0.20, (GRID_M - 1 - m) * h + h / 2, str(m), ha="right",
                va="center", fontsize=10, color=MUTED)
    ax.text(GRID_N * w / 2, GRID_M * h + 0.50, "output tile column  n",
            ha="center", va="bottom", fontsize=11, color=INK)
    ax.text(-0.76, GRID_M * h / 2, "output tile row  m", rotation=90,
            ha="center", va="center", fontsize=11, color=INK)

    ax.text(GRID_N * w / 2, -0.36,
            "8 CTAs, assigned down one tile column then the next; "
            "the pattern repeats every two columns",
            ha="center", va="top", fontsize=10, color=MUTED)

    ax.set_xlim(-1.08, GRID_N * w + 0.28)
    ax.set_ylim(-0.50, GRID_M * h + 0.72)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)



# One row per operation, time left to right. The row already says which
# operation a bar is, so colour is free to carry something the rows cannot:
# WHICH COPY of the buffer that step writes into. Single buffering is then one
# flat colour, and every doubling shows up as alternation - which is the whole
# point being made and was invisible when colour meant operation identity.
SB_ROWS = ["TMA load", "MMA", "tcgen05.ld", "stage", "TMA store"]
# hue = which resource, shade = which copy of it. A doubling is then a shade
# alternation inside one family, and no row ever changes what it stands for.
TMA_COLOUR = ["#a9c4de", "#5b82ab"]         # TMA buffer copy 0, copy 1
ACC_COLOUR = ["#a9d3b0", "#4f8a5c"]         # MMA buffer copy 0, copy 1
PLAIN = "#c7cfd8"                            # buffers that are never doubled
ROW_FAMILY = [TMA_COLOUR, ACC_COLOUR, None, None, None]

# (row, t0, t1, label, copy) where copy is the buffer copy this step is using:
# for a load, the copy it fills; for an MMA, the TMA-buffer copy it reads, or
# once the accumulator is doubled, the accumulator copy it writes.
# K = 2*BK throughout.
BARS_1BUF = [(0, 0, 1, "k=0", 0), (1, 1, 2, "k=0", 0),
             (0, 2, 3, "k=1", 0), (1, 3, 4, "k=1", 0),
             (2, 4, 5, "", 0), (3, 5, 6, "", 0), (4, 6, 7, "", 0),
             (0, 4, 5, "k=0", 0), (1, 5, 6, "k=0", 0),
             (0, 6, 7, "k=1", 0), (1, 7, 8, "k=1", 0),
             (2, 8, 9, "", 0), (3, 9, 10, "", 0), (4, 10, 11, "", 0)]
BRACKETS_1BUF = [(0, 7, "output tile 0"), (4, 11, "output tile 1")]

# Two TMA buffers: the loads alternate between the two copies, which is why the
# k=1 load can run while MMA k=0 still reads the k=0 copy. Only one accumulator,
# so every MMA still writes copy 0.
BARS_2BUF = [(0, 0, 1, "k=0", 0), (0, 1, 2, "k=1", 1),
             (1, 1, 2, "k=0", 0), (1, 2, 3, "k=1", 0),
             (2, 3, 4, "", 0), (3, 4, 5, "", 0), (4, 5, 6, "", 0),
             (0, 2, 3, "k=0", 0), (0, 3, 4, "k=1", 1),
             (1, 4, 5, "k=0", 0), (1, 5, 6, "k=1", 0),
             (2, 6, 7, "", 0), (3, 7, 8, "", 0), (4, 8, 9, "", 0)]
BRACKETS_2BUF = [(0, 6, "output tile 0"), (2, 9, "output tile 1")]

# Two TMA buffers and two accumulators: the loads alternate as before, and now
# the MMAs do too - tile 0 accumulates into copy 0, tile 1 into copy 1, so
# tile 1 no longer waits for tile 0's drain.
BARS_2ACC = [(0, 0, 1, "k=0", 0), (0, 1, 2, "k=1", 1),
             (1, 1, 2, "k=0", 0), (1, 2, 3, "k=1", 0),
             (2, 3, 4, "", 0), (3, 4, 5, "", 0), (4, 5, 6, "", 0),
             (0, 2, 3, "k=0", 0), (0, 3, 4, "k=1", 1),
             (1, 3, 4, "k=0", 1), (1, 4, 5, "k=1", 1),
             (2, 5, 6, "", 0), (3, 6, 7, "", 0), (4, 7, 8, "", 0)]
BRACKETS_2ACC = [(0, 6, "output tile 0"), (2, 8, "output tile 1")]


def _op_timeline(path, bars, brackets, title, subtitle, xmax, legend):
    H, DY = 0.66, 1.0
    NR = len(SB_ROWS)
    row_y = lambda r: (NR - 1 - r) * DY

    fig, ax = plt.subplots(figsize=(1.55 * xmax + 3.0, 5.2))

    def tile_bracket(t0, t1, y, text):
        ax.plot([t0, t1], [y, y], color=MUTED, linewidth=1.3)
        for x in (t0, t1):
            ax.plot([x, x], [y - 0.09, y + 0.09], color=MUTED, linewidth=1.3)
        ax.text((t0 + t1) / 2, y + 0.14, text, ha="center", va="bottom",
                fontsize=9.5, color=MUTED)

    for r, name in enumerate(SB_ROWS):
        y = row_y(r)
        ax.text(-0.20, y + H / 2, name, ha="right", va="center", fontsize=11,
                fontweight="bold", color=INK)
        ax.plot([0, xmax], [y + H / 2] * 2, color="#e3e7ec", linewidth=1.0,
                zorder=1)

    # both output tiles are equally real, so nothing distinguishes them except
    # the brackets above; the fill says which buffer copy this step writes
    for r, t0, t1, label, copy in bars:
        family = ROW_FAMILY[r]
        y, colour = row_y(r), PLAIN if family is None else family[copy]
        ax.add_patch(Rectangle((t0 + 0.025, y), t1 - t0 - 0.05, H,
                               facecolor=colour, edgecolor="white",
                               linewidth=1.2, zorder=3))
        if label:
            ax.text((t0 + t1) / 2, y + H / 2, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=INK, zorder=4)

    for k, (t0, t1, text) in enumerate(brackets):
        tile_bracket(t0, t1, NR * DY + 0.16 + 0.58 * k, text)

    base = -0.78
    ax.add_patch(FancyArrowPatch((0, base), (xmax + 0.1, base),
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.5, color=MUTED, shrinkA=0,
                                 shrinkB=0))
    ax.text(0, base + 0.22, "time", ha="left", va="bottom", fontsize=10,
            color=MUTED)

    # colour key in the empty margin at the lower left. It used to sit at the
    # bottom right, on the same line as the time arrow, where it read as one
    # more row of the timeline.
    for k, (colour, text) in enumerate(legend):
        ky = base - 0.42 - k * 0.36
        ax.add_patch(Rectangle((-2.35, ky), 0.30, 0.24, facecolor=colour,
                               edgecolor="white", linewidth=1.0))
        ax.text(-1.93, ky + 0.02, text, ha="left", va="bottom", fontsize=9.5,
                color=MUTED)

    ax.text(-2.35, NR * DY + 1.72, title, ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(-2.35, NR * DY + 1.32, subtitle, ha="left", va="center",
            fontsize=10.5, color=MUTED)

    ax.set_xlim(-2.40, xmax + 0.30)
    ax.set_ylim(base - 0.52 - 0.36 * max(len(legend) - 1, 0) - 0.30,
                NR * DY + 2.10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def single_buffer_timeline(path):
    _op_timeline(path, BARS_1BUF, BRACKETS_1BUF, "One buffer of each kind",
                 "every consecutive pair shares a buffer, so within one tile "
                 "nothing overlaps at all",
                 11, [])


def two_tma_buffer_timeline(path):
    _op_timeline(path, BARS_2BUF, BRACKETS_2BUF, "Two TMA buffers",
                 "the loads alternate between the two copies, so the k=1 load "
                 "runs while MMA k=0 still reads the k=0 copy",
                 9, [(TMA_COLOUR[0], "TMA buffer 0"),
                     (TMA_COLOUR[1], "TMA buffer 1")])


def two_accumulator_timeline(path):
    _op_timeline(path, BARS_2ACC, BRACKETS_2ACC,
                 "Two TMA buffers and two MMA buffers",
                 "now the accumulators alternate too, so tile 1 never waits "
                 "for tile 0's drain - the MMA row has no gaps left",
                 8, [(TMA_COLOUR[0], "TMA buffer 0"),
                     (TMA_COLOUR[1], "TMA buffer 1"),
                     (ACC_COLOUR[0], "MMA buffer 0"),
                     (ACC_COLOUR[1], "MMA buffer 1")])




# Execution order of output tiles, row-major against grouped (CTA swizzle).
# The point is not the order itself but how much of A and B one wave of CTAs
# has to touch: the highlighted tiles are the first 8 to run, and the strips
# along the edges are the A row-panels and B column-panels they need.
SWZ_N, SWZ_M, SWZ_WAVE, SWZ_GROUP = 8, 8, 8, 4


def cta_swizzle(path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    W = 1.0

    def panel(ax, grouped, title):
        order = {}
        for i in range(SWZ_M * SWZ_N):
            if grouped:
                per = SWZ_GROUP * SWZ_N
                g, r = divmod(i, per)
                m = g * SWZ_GROUP + r % SWZ_GROUP
                n = r // SWZ_GROUP
            else:
                m, n = divmod(i, SWZ_N)
            order[(m, n)] = i

        hot = {c for c, i in order.items() if i < SWZ_WAVE}
        rows_hit = sorted({m for m, _ in hot})
        cols_hit = sorted({n for _, n in hot})

        for (m, n), i in order.items():
            x, y = n * W, (SWZ_M - 1 - m) * W
            live = i < SWZ_WAVE
            ax.add_patch(Rectangle((x, y), W, W,
                                   facecolor="#8fb0d1" if live else BASE,
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
            ax.text(x + W / 2, y + W / 2, str(i), ha="center", va="center",
                    fontsize=8.5, color=INK if live else MUTED,
                    fontweight="bold" if live else "normal", zorder=3)

        # A row-panels on the left, B column-panels on top
        for m in range(SWZ_M):
            ax.add_patch(Rectangle((-0.75, (SWZ_M - 1 - m) * W), 0.5, W,
                                   facecolor="#8bc09a" if m in rows_hit else BASE,
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
        for n in range(SWZ_N):
            ax.add_patch(Rectangle((n * W, SWZ_M * W + 0.25), W, 0.5,
                                   facecolor="#e8bd83" if n in cols_hit else BASE,
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
        ax.text(-0.5, SWZ_M * W + 0.5, "A", ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(-1.05, SWZ_M * W / 2, "A row panels", rotation=90, ha="center",
                va="center", fontsize=10, color=MUTED)
        ax.text(SWZ_N * W / 2, SWZ_M * W + 1.05, "B column panels", ha="center",
                va="bottom", fontsize=10, color=MUTED)

        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                     loc="left", pad=10)
        ax.text(SWZ_N * W / 2, -0.75,
                f"the first {SWZ_WAVE} tiles touch {len(rows_hit)} A panels "
                f"+ {len(cols_hit)} B panels = {len(rows_hit) + len(cols_hit)}",
                ha="center", va="top", fontsize=11, color=INK)
        ax.set_xlim(-1.35, SWZ_N * W + 0.2)
        ax.set_ylim(-1.35, SWZ_M * W + 1.45)
        ax.set_aspect("equal")
        ax.axis("off")

    panel(axes[0], False, "row-major order")
    panel(axes[1], True, f"grouped order  (CTA swizzle, GROUP_SIZE_M={SWZ_GROUP})")
    fig.tight_layout(pad=0.4)
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


# One colour per CTA for the persistent-grid figure. Four hues x two shades,
# taken from the palette the rest of the article already uses (the blue and
# green are literally TMA_COLOUR / ACC_COLOUR, the amber is TILE). Hue is what
# separates neighbours; the shade splits each hue into a light and a dark CTA,
# so eight CTAs stay tellable apart without inventing eight new colours.
PERSIST_LIGHT = ["#a9c4de",   # blue, light
                 "#a9d3b0",   # green, light
                 "#e9c79a",   # amber, light
                 "#c3b3e0"]   # purple, light
PERSIST_DARK = ["#4a6f96",    # blue, dark
                "#47825a",    # green, dark
                "#b07d31",    # amber, dark
                "#6a58a0"]    # purple, dark
PERSIST_COLOURS = PERSIST_LIGHT + PERSIST_DARK
PERSIST_TEXT = [INK] * 4 + ["white"] * 4


def _plural(n, noun):
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _tile_order(grouped):
    """(m, n) -> position in the traversal order, for the SWZ_M x SWZ_N grid."""
    order = {}
    for i in range(SWZ_M * SWZ_N):
        if grouped:
            per = SWZ_GROUP * SWZ_N
            g, r = divmod(i, per)
            m = g * SWZ_GROUP + r % SWZ_GROUP
            n = r // SWZ_GROUP
        else:
            m, n = divmod(i, SWZ_N)
        order[(m, n)] = i
    return order


def persistent_swizzle(path):
    """Persistent grid x CTA swizzle: same 8x8 grid, who computes what.

    Same base as cta_swizzle - the traversal order is identical - but here the
    tile is painted with the CTA that owns it (tile i -> CTA i % NCTA), which
    is what the persistent grid adds on top of the order. Wave 0 (the first
    NCTA tiles, one per CTA) keeps the bold outline and the A/B panel strips,
    so the L2 argument from the previous figure carries over unchanged.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.8))
    W = 1.0
    nwaves = SWZ_M * SWZ_N // NCTA

    def panel(ax, grouped, title):
        order = _tile_order(grouped)
        hot = {c for c, i in order.items() if i < NCTA}
        rows_hit = sorted({m for m, _ in hot})
        cols_hit = sorted({n for _, n in hot})

        for (m, n), i in order.items():
            cta = i % NCTA
            x, y = n * W, (SWZ_M - 1 - m) * W
            ax.add_patch(Rectangle((x, y), W, W,
                                   facecolor=PERSIST_COLOURS[cta],
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
            ax.text(x + W / 2, y + W / 2 - 0.04, f"CTA{cta}", ha="center",
                    va="center", fontsize=9, fontweight="bold",
                    color=PERSIST_TEXT[cta], zorder=5)
            # the position in the traversal order - the same number the
            # previous figure labelled its tiles with
            ax.text(x + 0.07, y + W - 0.07, str(i), ha="left", va="top",
                    fontsize=6.5, color=PERSIST_TEXT[cta], alpha=0.75,
                    zorder=5)

        # wave 0 outlines in a second pass: drawn over every fill, so a
        # neighbouring tile can never paint over half a border
        for (m, n), i in order.items():
            if i >= NCTA:
                continue
            ax.add_patch(Rectangle((n * W, (SWZ_M - 1 - m) * W), W, W,
                                   facecolor="none", edgecolor=INK,
                                   linewidth=2.4, zorder=6))

        # A row-panels on the left, B column-panels on top - highlighted for
        # the tiles of wave 0, exactly as in the CTA swizzle figure
        for m in range(SWZ_M):
            ax.add_patch(Rectangle((-0.75, (SWZ_M - 1 - m) * W), 0.5, W,
                                   facecolor="#8bc09a" if m in rows_hit else BASE,
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
        for n in range(SWZ_N):
            ax.add_patch(Rectangle((n * W, SWZ_M * W + 0.25), W, 0.5,
                                   facecolor="#e8bd83" if n in cols_hit else BASE,
                                   edgecolor=EDGE, linewidth=0.8, zorder=2))
        ax.text(-0.5, SWZ_M * W + 0.5, "A", ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(-1.05, SWZ_M * W / 2, "A row panels", rotation=90, ha="center",
                va="center", fontsize=10, color=MUTED)
        ax.text(SWZ_N * W / 2, SWZ_M * W + 1.05, "B column panels", ha="center",
                va="bottom", fontsize=10, color=MUTED)

        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                     loc="left", pad=10)
        ax.text(SWZ_N * W / 2, -0.55,
                f"wave 0 = tiles 0-{NCTA - 1} (bold outline): "
                f"{_plural(len(rows_hit), 'A panel')} + "
                f"{_plural(len(cols_hit), 'B panel')}",
                ha="center", va="top", fontsize=11, color=INK)
        ax.text(SWZ_N * W / 2, -1.05,
                f"each CTA then walks its own {nwaves} tiles, "
                f"one per outer-loop iteration",
                ha="center", va="top", fontsize=9.5, color=MUTED)
        ax.set_xlim(-1.35, SWZ_N * W + 0.2)
        ax.set_ylim(-1.75, SWZ_M * W + 1.45)
        ax.set_aspect("equal")
        ax.axis("off")

    panel(axes[0], False, "persistent grid + row-major order")
    panel(axes[1], True,
          f"persistent grid + grouped order  (GROUP_SIZE_M={SWZ_GROUP})")
    fig.tight_layout(pad=0.4)
    for suffix in ("png", "svg"):
        out = path.with_suffix("." + suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
