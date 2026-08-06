"""Render the data-flow model figures.

The model has four kinds of buffer - TMA (input tiles in SMEM), MMA (the TMEM
accumulator), tcgen05.ld (its results in registers) and store (the
epilogue's SMEM staging) - drawn as boxes with two ports each: an input port on
top and an output port on the bottom.

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
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image


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


def draw_buffer(ax, cx, cy, title, subtitle, input_state, output_state,
                width=BOX_W):
    """Draw one buffer box and return its port coordinates."""
    ax.add_patch(FancyBboxPatch(
        (cx - width / 2, cy - BOX_H / 2), width, BOX_H,
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


PIPE_FACE = "#e3edf7"
PIPE_EDGE = "#7f9dbb"
PIPE_W = 0.62


def draw_pipe(ax, src_port, dst_port, label, note=None, label_dx=0.62):
    """Draw an operation as a pipe joining two ports.

    The pipe runs from the source buffer's output port to the destination
    buffer's input port. Data can move through it only when both ports are
    green; the pipe itself is just the connection, it carries no state.
    """
    x0, y0 = src_port
    x1, y1 = dst_port
    top, bottom = max(y0, y1), min(y0, y1)
    ax.add_patch(FancyBboxPatch(
        (x0 - PIPE_W / 2, bottom), PIPE_W, top - bottom,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=1.5, edgecolor=PIPE_EDGE, facecolor=PIPE_FACE, zorder=1,
    ))
    mid = (top + bottom) / 2
    ax.add_patch(FancyArrowPatch(
        (x0, mid + 0.30), (x1, mid - 0.30),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
        color=PIPE_EDGE, shrinkA=0, shrinkB=0, zorder=3,
    ))
    ax.text(x0 + label_dx, mid + 0.13, label, ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=INK, zorder=4)
    if note:
        ax.text(x0 + label_dx, mid - 0.15, note, ha="left", va="center",
                fontsize=8.5, color=MUTED, zorder=4)


SIGNAL = "#c8781a"


def draw_signal(ax, target, label, reach=1.75, rise=0.75):
    """Draw a signal: a dashed arrow delivered to a buffer.

    A signal is the only thing that changes a port's state, and it is addressed
    to the buffer rather than to one port - one signal flips both of them.
    """
    tx, ty = target
    sx, sy = tx + reach, ty + rise
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (tx + 0.06, ty),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
        linestyle=(0, (4, 2.4)), color=SIGNAL, zorder=6,
        shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.25",
    ))
    ax.text(sx + 0.06, sy + 0.04, label, ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=SIGNAL, zorder=6)


def highlight_port(ax, x, y):
    """Ring a port that a signal has just flipped."""
    ax.add_patch(Circle((x, y), PORT_R + 0.11, facecolor="none",
                        edgecolor=SIGNAL, linewidth=1.6,
                        linestyle=(0, (2.4, 1.8)), zorder=7))


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
    """The four kinds of buffer in a 2x2 grid, with no operations between them.

    Laid out so the data flow runs clockwise - TMA to MMA across the top, down
    to the tcgen05.ld buffer, then left to the store buffer - which is the shape
    the operation edges will take once they are drawn.

    This figure introduces the vocabulary only: what buffers exist, where they
    live, and that each one has an input port on top and an output port on the
    bottom. The initial state is drawn - every buffer empty, so every input port
    green and every output port red.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    kinds = [
        (0, 0, "TMA buffer", "SMEM  ·  A / B input tiles",
         "TMA writes  ·  MMA reads"),
        (1, 0, "MMA buffer", "TMEM  ·  accumulator",
         "MMA writes  ·  tcgen05.ld reads"),
        (0, 1, "Store buffer", "SMEM  ·  epilogue staging",
         "epilogue writes  ·  TMA store reads"),
        (1, 1, "tcgen05.ld buffer", "RMEM  ·  tcgen05.ld results",
         "tcgen05.ld writes  ·  epilogue reads"),
    ]
    for col, row, title, subtitle, note in kinds:
        cx = 2.05 + col * 4.30
        cy = 3.70 - row * 2.45
        draw_buffer(ax, cx, cy, title, subtitle, READY, NOT_READY, width=3.15)
        ax.text(cx, cy - 0.95, note, ha="center", va="center",
                fontsize=8.5, color=MUTED)

    ax.text(0.30, 5.45, "The four kinds of buffer", ha="left", va="center",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 5.08, "initial state — every buffer empty, no operations drawn yet",
            ha="left", va="center", fontsize=10.5, color=MUTED)

    ax.text(0.30, -0.30,
            "Every buffer has an input port on top and an output port on the "
            "bottom.\n"
            "A vertical bar is a path data may pass through, a horizontal bar "
            "one that is barred.",
            ha="left", va="center", fontsize=9, color=INK, linespacing=1.5)

    ax.set_xlim(0.0, 8.9)
    ax.set_ylim(-0.70, 5.70)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def operation_as_pipe():
    """How an operation is modelled: a pipe joining two ports.

    Drawn in a state where the MMA may actually start - the TMA buffer is full
    (input red, output green) and the MMA buffer is empty (input green, output
    red), which is exactly both-green across the pipe.
    """
    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    cx = 2.35
    tma = draw_buffer(ax, cx, 4.00, "TMA buffer",
                      "SMEM  ·  A / B input tiles", NOT_READY, READY)
    mma = draw_buffer(ax, cx, 1.45, "MMA buffer",
                      "TMEM  ·  accumulator", READY, NOT_READY)

    draw_pipe(ax, tma["out"], mma["in"], "MMA",
              "input tiles  →  accumulator")

    # One short leader to the pipe, rather than two long ones to the ports:
    # the pipe spans both ports, and crossing leaders fought with the label.
    mid = (tma["out"][1] + mma["in"][1]) / 2
    lead_y = mid + 0.44   # above the MMA label, so the leader crosses nothing
    ax.annotate("", xy=(cx + PIPE_W / 2 + 0.04, lead_y), xytext=(5.26, lead_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.2, color=MUTED,
                                shrinkA=2, shrinkB=0))
    ax.text(5.40, lead_y + 0.17, "both ports green", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=READY)
    ax.text(5.40, lead_y - 0.15, "the MMA may start", ha="left", va="center",
            fontsize=9.5, color=MUTED)

    ax.text(0.30, 5.55, "An operation is a pipe between two ports",
            ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 5.18,
            "from the source buffer's output port to the destination buffer's "
            "input port",
            ha="left", va="center", fontsize=10.5, color=MUTED)

    legend(ax, 5.40, 1.62)
    ax.text(0.30, -0.42,
            "The pipe is only the connection — it holds no state of its own. "
            "The operation may start only when the\n"
            "output port feeding it and the input port it feeds are both "
            "green, and it is the ports that then flip.",
            ha="left", va="center", fontsize=9, color=INK, linespacing=1.5)

    ax.set_xlim(0.0, 9.5)
    ax.set_ylim(-0.85, 5.80)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _signal_panel(ax, x0, caption, tma_states, mma_states, signal=False):
    """One TMA -> MMA pair, used twice side by side to show a signal landing."""
    cx = x0 + 1.55
    tma = draw_buffer(ax, cx, 3.55, "TMA buffer", "SMEM  ·  A / B input tiles",
                      *tma_states, width=2.95)
    mma = draw_buffer(ax, cx, 1.30, "MMA buffer", "TMEM  ·  accumulator",
                      *mma_states, width=2.95)
    draw_pipe(ax, tma["out"], mma["in"], "MMA", label_dx=0.58)
    ax.text(cx, 4.62, caption, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    if signal:
        draw_signal(ax, (cx + 2.95 / 2, 3.55), "buffer free")
        highlight_port(ax, *tma["in"])
        highlight_port(ax, *tma["out"])
    return tma, mma


def signal_flips_a_port():
    """What a signal is: the only thing that changes a port's state.

    Before and after one MMA completes. The MMA consumed the TMA buffer, so a
    buffer-free signal lands on that buffer: its output port closes (the data is
    gone) and its input port opens (it may be refilled).

    The MMA buffer deliberately does not flip - it accumulates over every k-tile,
    so its data-ready signal fires only after the last one. That is explained in
    the README rather than on the canvas, which stays uncluttered.
    """
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    _signal_panel(ax, 0.30, "while the MMA runs",
                  (NOT_READY, READY), (READY, NOT_READY))
    ax.annotate("", xy=(4.62, 2.60), xytext=(3.94, 2.60),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=16,
                                linewidth=1.6, color=MUTED))
    ax.text(4.28, 2.92, "MMA\ncompletes", ha="center", va="center",
            fontsize=9, color=MUTED, linespacing=1.4)
    _signal_panel(ax, 4.95, "after it completes",
                  (READY, NOT_READY), (READY, NOT_READY), signal=True)

    ax.text(0.30, 5.72, "A signal is what flips a port",
            ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 5.35,
            "each buffer takes two: buffer free opens its input and closes its "
            "output, data ready does the opposite",
            ha="left", va="center", fontsize=10.5, color=MUTED)

    ax.set_xlim(0.0, 11.6)
    ax.set_ylim(0.30, 5.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _signal_scene(tma_states, mma_states, step, signal=False, ringed=False):
    """One frame of the signal animation: a single TMA -> MMA pair."""
    fig, ax = plt.subplots(figsize=(7.3, 5.55))
    cx = 2.05
    tma = draw_buffer(ax, cx, 3.55, "TMA buffer", "SMEM  ·  A / B input tiles",
                      *tma_states, width=2.95)
    mma = draw_buffer(ax, cx, 1.30, "MMA buffer", "TMEM  ·  accumulator",
                      *mma_states, width=2.95)
    draw_pipe(ax, tma["out"], mma["in"], "MMA", label_dx=0.58)
    if signal:
        draw_signal(ax, (cx + 2.95 / 2, 3.55), "buffer free")
    if ringed:
        highlight_port(ax, *tma["in"])
        highlight_port(ax, *tma["out"])

    ax.text(0.30, 5.10, "A signal is what flips a port", ha="left",
            va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.30, 0.06, step, ha="left", va="center", fontsize=10.5,
            color=INK)
    ax.set_xlim(0.0, 7.3)
    ax.set_ylim(-0.20, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    # Frames must all come out the same size, so the margins are pinned here
    # rather than cropped per frame by bbox_inches="tight".
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig


# (states, caption, signal drawn, ports ringed, hold in ms)
SIGNAL_FRAMES = [
    ((NOT_READY, READY), (READY, NOT_READY),
     "both ports green — the MMA may start", False, False, 1500),
    ((NOT_READY, READY), (READY, NOT_READY),
     "the MMA runs, draining the TMA buffer", False, False, 1200),
    ((NOT_READY, READY), (READY, NOT_READY),
     "it completes, and sends a buffer-free signal", True, False, 1500),
    ((READY, NOT_READY), (READY, NOT_READY),
     "the signal flips both ports of that buffer", True, True, 1800),
    ((READY, NOT_READY), (READY, NOT_READY),
     "output closed, input open — TMA may refill it", False, False, 1800),
]


def signal_animation(path, dpi=110):
    """Write the signal figure as a GIF, for posts that can show one."""
    frames, durations = [], []
    for tma_states, mma_states, step, signal, ringed, hold in SIGNAL_FRAMES:
        fig = _signal_scene(tma_states, mma_states, step, signal, ringed)
        buf = io.BytesIO()
        # No bbox_inches="tight": every frame must come out the same size.
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        durations.append(hold)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    return path


FIGURES = {
    "buffer-kinds": buffer_kinds,
    "operation-as-pipe": operation_as_pipe,
    "signal-flips-a-port": signal_flips_a_port,
    "per-slot-initial-state": per_slot_initial_state,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(Path(__file__).parent / "figures"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"wrote {signal_animation(outdir / 'signal-flips-a-port.gif')}")
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
