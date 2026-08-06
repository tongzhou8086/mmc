"""Render the data-flow model figures.

The model has three kinds of buffer - TMA (input tiles in SMEM), MMA (the TMEM
accumulator) and store (the epilogue's SMEM staging) - drawn as boxes with two
ports each: an input port on top and an output port on the bottom.

An operation is an edge from one buffer's output port to another buffer's input
port. It may fire only when both of those ports are green:

    output port green -> the data in this buffer is ready to be consumed
    input port green  -> this buffer is free to be overwritten

A port carries its state twice over: colour, plus a glyph for the state of the
path through it - a vertical bar when data may pass, a horizontal bar when it is
barred. The glyph is what keeps the figures readable in grayscale and for
red-green colourblind readers, and it describes the path rather than the
buffer's contents, which matters because green means opposite occupancy on the
two ports: a green input port is an empty buffer, a green output port a full one.

So synchronization is nothing but flipping port states: every mbarrier in the
kernel toggles one port, and a stalled operation is one whose two ports are not
both green yet.

Usage:  python data-flow-models/render.py [--outdir DIR]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


READY = "#2f9e56"       # green: data ready / free to overwrite
NOT_READY = "#d1495b"   # red: data not ready / still in use
INK = "#1f2933"
MUTED = "#6b7684"
BOX_FACE = "#f7f9fb"
BOX_EDGE = "#8c98a4"

BOX_W = 3.0
BOX_H = 1.15
PORT_R = 0.135


def draw_port(ax, x, y, state, radius=PORT_R):
    """Draw one port: a coloured disc plus a glyph for the state of the path.

    Vertical bar - aligned with the direction data flows - means the path is
    open; horizontal bar means it is barred.
    """
    ax.add_patch(Circle((x, y), radius, facecolor=state, edgecolor=INK,
                        linewidth=1.2, zorder=5))
    arm = radius * 0.52
    if state == READY:
        ax.plot([x, x], [y - arm, y + arm], color="white", zorder=6,
                linewidth=radius * 15, solid_capstyle="round")
    else:
        ax.plot([x - arm, x + arm], [y, y], color="white", zorder=6,
                linewidth=radius * 15, solid_capstyle="round")


def draw_buffer(ax, cx, cy, title, subtitle, input_state, output_state):
    """Draw one buffer box and return its port coordinates."""
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=BOX_EDGE, facecolor=BOX_FACE, zorder=2,
    ))
    ax.text(cx, cy + 0.16, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK, zorder=4)
    ax.text(cx, cy - 0.21, subtitle, ha="center", va="center",
            fontsize=9, color=MUTED, zorder=4)

    top = (cx, cy + BOX_H / 2)
    bottom = (cx, cy - BOX_H / 2)
    draw_port(ax, *top, input_state)
    draw_port(ax, *bottom, output_state)
    return {"in": top, "out": bottom}


def draw_operation(ax, src_port, dst_port, label, note=None):
    """Draw an operation edge from a source output port to a dest input port."""
    x0, y0 = src_port
    x1, y1 = dst_port
    ax.add_patch(FancyArrowPatch(
        (x0, y0 - PORT_R - 0.04), (x1, y1 + PORT_R + 0.04),
        arrowstyle="-|>", mutation_scale=17, linewidth=1.8,
        color=INK, shrinkA=0, shrinkB=0, zorder=3,
    ))
    mid_y = (y0 + y1) / 2
    ax.text(x0 + 0.24, mid_y + 0.07, label, ha="left", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=4)
    if note:
        ax.text(x0 + 0.24, mid_y - 0.18, note, ha="left", va="center",
                fontsize=8.5, color=MUTED, zorder=4)


def legend(ax, x, y, title="Port states", columns=1, col_width=3.3):
    """Explain what the two port states mean in each port position.

    One column by default, so it can sit beside a tall diagram; pass columns=2
    to spread it under a wide one.
    """
    rows = [
        (READY, "input port", "buffer is free to be overwritten"),
        (NOT_READY, "input port", "buffer still holds data in use"),
        (READY, "output port", "data is ready to be consumed"),
        (NOT_READY, "output port", "data is not ready yet"),
    ]
    ax.text(x - 0.02, y + 0.52, title, ha="left", va="center",
            fontsize=10, fontweight="bold", color=INK)
    per_col = -(-len(rows) // columns)
    for i, (state, role, meaning) in enumerate(rows):
        xx = x + (i // per_col) * col_width
        yy = y - (i % per_col) * 0.46
        draw_port(ax, xx, yy, state, radius=PORT_R * 0.85)
        ax.text(xx + 0.26, yy + 0.11, role, ha="left", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.text(xx + 0.26, yy - 0.13, meaning, ha="left", va="center",
                fontsize=8.5, color=MUTED)


def per_slot_initial_state():
    """One buffer of each kind, in the state the pipeline starts in.

    Every buffer is empty: free to be overwritten (input green) but with no data
    to hand on (output red). Neither operation can fire, because each needs its
    source's output port to turn green first.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    tma = draw_buffer(ax, 2.35, 4.85, "TMA buffer",
                      "SMEM  ·  A / B input tiles", READY, NOT_READY)
    mma = draw_buffer(ax, 2.35, 3.00, "MMA buffer",
                      "TMEM  ·  accumulator", READY, NOT_READY)
    store = draw_buffer(ax, 2.35, 1.15, "Store buffer",
                        "SMEM  ·  epilogue staging", READY, NOT_READY)

    draw_operation(ax, tma["out"], mma["in"], "MMA",
                   "input tiles  →  accumulator")
    draw_operation(ax, mma["out"], store["in"], "Epilogue",
                   "accumulator  →  staged output")

    ax.text(0.30, 6.30, "Per-slot data flow", ha="left", va="center",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 5.93, "initial state — one buffer of each kind",
            ha="left", va="center", fontsize=10.5, color=MUTED)

    legend(ax, 5.15, 4.35)
    ax.text(0.30, 0.06,
            "An operation may fire only when its source output port and its "
            "destination input port are both green.\n"
            "Synchronization is therefore just the flipping of port states: "
            "each mbarrier toggles one port.",
            ha="left", va="center", fontsize=9, color=INK, linespacing=1.5)

    ax.set_xlim(0.0, 9.1)
    ax.set_ylim(-0.35, 6.60)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def buffer_kinds():
    """The three kinds of buffer, side by side, with no operations between them.

    This figure introduces the vocabulary only: what buffers exist, where they
    live, and that each one has an input port on top and an output port on the
    bottom. The initial state is drawn - every buffer empty, so every input port
    green and every output port red.
    """
    fig, ax = plt.subplots(figsize=(10.6, 4.3))

    kinds = [
        ("TMA buffer", "SMEM  ·  A / B input tiles",
         "TMA writes  ·  MMA reads"),
        ("MMA buffer", "TMEM  ·  accumulator",
         "MMA writes  ·  epilogue reads"),
        ("Store buffer", "SMEM  ·  epilogue staging",
         "epilogue writes  ·  TMA store reads"),
    ]
    for i, (title, subtitle, note) in enumerate(kinds):
        cx = 1.85 + i * 3.5
        draw_buffer(ax, cx, 3.05, title, subtitle, READY, NOT_READY)
        ax.text(cx, 2.10, note, ha="center", va="center",
                fontsize=8.5, color=MUTED)

    ax.text(0.30, 4.55, "The three kinds of buffer", ha="left", va="center",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 4.18, "initial state — every buffer empty, no operations drawn yet",
            ha="left", va="center", fontsize=10.5, color=MUTED)

    legend(ax, 0.30, 1.30, columns=2, col_width=4.6)
    ax.text(0.30, -0.10,
            "Every buffer has an input port on top and an output port on the "
            "bottom, and each port is a switch.\n"
            "The glyph shows the state of the path through it: a vertical bar "
            "where data may pass, a horizontal bar where it is barred.",
            ha="left", va="center", fontsize=9, color=INK, linespacing=1.5)

    ax.set_xlim(0.0, 10.7)
    ax.set_ylim(-0.55, 4.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


FIGURES = {
    "buffer-kinds": buffer_kinds,
    "per-slot-initial-state": per_slot_initial_state,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(Path(__file__).parent / "figures"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES.items():
        fig = builder()
        for suffix in ("svg", "png"):
            path = outdir / f"{name}.{suffix}"
            fig.savefig(path, format=suffix, dpi=200,
                        bbox_inches="tight", facecolor="white")
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
