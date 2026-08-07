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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
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


# One 32 KB buffer is this wide; every other box is scaled from it, so box area
# is proportional to the memory the buffer actually occupies.
KB32_W = 1.15
SLOT_H = 0.90


def draw_slot(ax, cx, cy, kb, label, input_state, output_state, active=False):
    """Draw one buffer sized by its capacity, with a port at each end.

    `active` tints the box in the pipe colour, marking the buffers an operation
    is touching in the state being drawn.
    """
    w = KB32_W * kb / 32.0
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - SLOT_H / 2), w, SLOT_H,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=2.2 if active else 1.4,
        edgecolor=PIPE_EDGE if active else BOX_EDGE,
        facecolor=PIPE_FACE if active else BOX_FACE, zorder=2,
    ))
    if label:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK, zorder=4)
    top, bottom = (cx, cy + SLOT_H / 2), (cx, cy - SLOT_H / 2)
    draw_port(ax, *top, input_state, radius=0.105)
    draw_port(ax, *bottom, output_state, radius=0.105)
    return {"in": top, "out": bottom, "w": w}


def draw_row(ax, x0, cy, count, kb, labels, input_state, output_state, gap=0.22):
    """Lay out a row of identically sized buffers, left-aligned at x0."""
    w = KB32_W * kb / 32.0
    slots = []
    for i in range(count):
        cx = x0 + w / 2 + i * (w + gap)
        slots.append(draw_slot(ax, cx, cy, kb, labels[i] if labels else None,
                               input_state, output_state))
    return slots


def draw_pipe_between(ax, src_port, dst_port, label, label_dx=0.30):
    """A pipe joining two ports that are not vertically aligned.

    Drawn as a slanted conduit rather than a dog-leg: the ports of adjacent rows
    rarely line up once boxes are sized by capacity.
    """
    import math
    x0, y0 = src_port
    x1, y1 = dst_port
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    px, py = -uy * PIPE_W / 2, ux * PIPE_W / 2
    ax.add_patch(Polygon(
        [(x0 + px, y0 + py), (x1 + px, y1 + py),
         (x1 - px, y1 - py), (x0 - px, y0 - py)],
        closed=True, linewidth=1.5, edgecolor=PIPE_EDGE,
        facecolor=PIPE_FACE, zorder=1,
    ))
    ax.add_patch(FancyArrowPatch(
        (x0 + ux * 0.26, y0 + uy * 0.26), (x1 - ux * 0.26, y1 - uy * 0.26),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
        color=PIPE_EDGE, shrinkA=0, shrinkB=0, zorder=3,
    ))
    ax.text((x0 + x1) / 2 + PIPE_W / 2 + label_dx, (y0 + y1) / 2, label,
            ha="left", va="center", fontsize=11, fontweight="bold",
            color=INK, zorder=4)


def draw_source_pipe(ax, dst_port, label, length=1.15):
    """A pipe with no source buffer - data arriving from memory.

    Global memory is deliberately not a buffer in this model, so a load from it
    is drawn as a pipe that simply begins.
    """
    x, y = dst_port
    top = y + length
    ax.add_patch(FancyBboxPatch(
        (x - PIPE_W / 2, y), PIPE_W, length,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=1.5, edgecolor=PIPE_EDGE, facecolor=PIPE_FACE, zorder=1,
    ))
    ax.add_patch(FancyArrowPatch(
        (x, top - 0.28), (x, y + 0.22), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.6, color=PIPE_EDGE, shrinkA=0, shrinkB=0, zorder=3,
    ))
    ax.text(x + PIPE_W / 2 + 0.18, top - 0.30, label, ha="left", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=4)



def _signal_panel(ax, x0, caption, tma_states, mma_states,
                  pipe=True, ringed=False):
    """One TMA -> MMA pair, used three times across to show a handshake."""
    cx = x0 + 1.55
    tma = draw_buffer(ax, cx, 3.55, "TMA buffer", "SMEM  ·  A / B input tiles",
                      *tma_states, width=2.95)
    mma = draw_buffer(ax, cx, 1.30, "MMA buffer", "TMEM  ·  accumulator",
                      *mma_states, width=2.95)
    if pipe:
        draw_pipe(ax, tma["out"], mma["in"], "MMA", label_dx=0.58)
    if ringed:
        highlight_port(ax, *tma["in"])
        highlight_port(ax, *tma["out"])
    ax.text(cx, 4.62, caption, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    return tma, mma


def _signal_transition(ax, x_from, x_to, y, label, note):
    """A dashed arrow between panels, named for the signal that causes it."""
    ax.add_patch(FancyArrowPatch(
        (x_from, y), (x_to, y), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.8, linestyle=(0, (4, 2.4)), color=SIGNAL,
        shrinkA=0, shrinkB=0, zorder=4,
    ))
    mx = (x_from + x_to) / 2
    ax.text(mx, y + 0.34, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=SIGNAL)
    ax.text(mx, y - 0.36, note, ha="center", va="center",
            fontsize=8.5, color=MUTED)


def signal_flips_a_port():
    """What a signal is: the only thing that changes a port's state.

    One full turn of the handshake. Data ready opens the TMA buffer's output
    port, at which point both ends of the MMA pipe are green and the pipe
    appears; buffer free closes it again once the MMA has drained the buffer.

    The pipe is drawn only while it could actually carry data, which is the
    firing rule made visible. The MMA buffer never flips here - it accumulates
    over every k-tile, so its data-ready signal fires only after the last one.
    That reading lives in the README rather than on the canvas.
    """
    fig, ax = plt.subplots(figsize=(12.8, 5.7))

    _signal_panel(ax, 0.30, "both buffers empty",
                  (READY, NOT_READY), (READY, NOT_READY), pipe=False)
    _signal_transition(ax, 3.50, 4.72, 2.42, "data ready",
                       "TMA has filled it")
    _signal_panel(ax, 4.80, "output open — the MMA runs",
                  (NOT_READY, READY), (READY, NOT_READY), ringed=True)
    _signal_transition(ax, 8.00, 9.22, 2.42, "buffer free",
                       "the MMA has drained it")
    _signal_panel(ax, 9.30, "flipped back — TMA may refill",
                  (READY, NOT_READY), (READY, NOT_READY),
                  pipe=False, ringed=True)

    ax.text(0.30, 5.72, "A signal is what flips a port",
            ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 5.35,
            "each buffer takes two: data ready opens its output and closes its "
            "input, buffer free does the opposite",
            ha="left", va="center", fontsize=10.5, color=MUTED)

    ax.set_xlim(0.0, 12.7)
    ax.set_ylim(0.30, 5.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _signal_scene(tma_states, mma_states, step, signal=None, ringed=False,
                  pipe=True):
    """One frame of the signal animation: a single TMA -> MMA pair.

    The pipe is drawn only while it could actually carry data - that is, while
    the TMA buffer's output port and the MMA buffer's input port are both green.
    Seeing it appear and vanish is the firing rule made visible.
    """
    fig, ax = plt.subplots(figsize=(7.3, 5.55))
    cx = 2.05
    tma = draw_buffer(ax, cx, 3.55, "TMA buffer", "SMEM  ·  A / B input tiles",
                      *tma_states, width=2.95)
    mma = draw_buffer(ax, cx, 1.30, "MMA buffer", "TMEM  ·  accumulator",
                      *mma_states, width=2.95)
    if pipe:
        draw_pipe(ax, tma["out"], mma["in"], "MMA", label_dx=0.58)
    if signal:
        draw_signal(ax, (cx + 2.95 / 2, 3.55), signal)
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


# One full turn of the handshake: data ready opens the TMA buffer's output and
# the pipe appears; buffer free closes it again once the MMA has drained it.
# (tma states, mma states, caption, signal label, ports ringed, pipe, hold ms)
SIGNAL_FRAMES = [
    ((READY, NOT_READY), (READY, NOT_READY),
     "both buffers empty — no data for the MMA to read",
     None, False, False, 1600),
    ((READY, NOT_READY), (READY, NOT_READY),
     "TMA finishes its load and signals data ready",
     "data ready", False, False, 1600),
    ((NOT_READY, READY), (READY, NOT_READY),
     "the signal flips both ports: the output opens",
     "data ready", True, True, 1900),
    ((NOT_READY, READY), (READY, NOT_READY),
     "both ports green — the MMA runs",
     None, False, True, 1500),
    ((NOT_READY, READY), (READY, NOT_READY),
     "it completes and signals buffer free",
     "buffer free", False, True, 1600),
    ((READY, NOT_READY), (READY, NOT_READY),
     "flipped back — the data is gone, TMA may refill",
     "buffer free", True, False, 1900),
]


def signal_animation(path, dpi=110):
    """Write the signal figure as a GIF, for posts that can show one."""
    frames, durations = [], []
    for tma_states, mma_states, step, signal, ringed, pipe, hold in SIGNAL_FRAMES:
        fig = _signal_scene(tma_states, mma_states, step, signal, ringed, pipe)
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

# (row label, count, KB each, row centre y, per-box labels)
BN256_ROWS = [
    ("TMA", 6, 32, 5.90, [str(i) for i in range(6)]),
    ("MMA", 2, 128, 4.00, ["0", "1"]),
    ("tcgen05.ld", 1, 32, 2.50, [""]),
    ("Store", 2, 16, 1.05, ["0", "1"]),
]
BN256_SIZES = [
    (5.90, "6 x 32 KB", "SMEM  ·  192 KB"),
    (4.00, "2 x 128 KB", "TMEM  ·  256 KB"),
    (2.50, "1 x 32 KB", "RMEM"),
    (1.05, "2 x 16 KB", "SMEM  ·  32 KB"),
]


def _bn256_layout(ax, states=None, active=(), x0=2.30):
    """Draw every buffer of the BN=256 pipeline, sized by capacity.

    `states` maps (row, index) to (input, output); anything absent is an empty
    buffer, i.e. free to overwrite with nothing to hand on. `active` marks the
    buffers an operation is touching.
    """
    states = states or {}
    placed = {}
    for ri, (name, count, kb, cy, labels) in enumerate(BN256_ROWS):
        w = KB32_W * kb / 32.0
        for i in range(count):
            cx = x0 + w / 2 + i * (w + 0.22)
            ins, outs = states.get((ri, i), (READY, NOT_READY))
            placed[(ri, i)] = draw_slot(ax, cx, cy, kb, labels[i], ins, outs,
                                        active=(ri, i) in active)
        ax.text(x0 - 0.30, cy, name, ha="right", va="center",
                fontsize=11, fontweight="bold", color=INK)
    return placed


def _bn256_chrome(ax, state, subtitle, note):
    """Title, size legend and footnote shared by every BN=256 state figure."""
    ax.text(0.30, 8.30,
            "BM=128 · BN=256 · BK=64 pipeline — K=128 for illustration "
            f"— State {state}",
            ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.30, 7.91, subtitle, ha="left", va="center",
            fontsize=10.5, color=MUTED)
    lx = 12.05
    ax.text(lx, 6.75, "Buffer sizes", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    for cy, size, where in BN256_SIZES:
        ax.text(lx, cy + 0.15, size, ha="left", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(lx, cy - 0.16, where, ha="left", va="center",
                fontsize=8.5, color=MUTED)
    ax.text(0.30, 0.28, note, ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlim(0.0, 14.9)
    ax.set_ylim(0.05, 8.50)
    ax.set_aspect("equal")
    ax.axis("off")


def bn256_state1_tma_load():
    """State 1 of the BM=128 / BN=256 / BK=64 pipeline: only the TMA load runs.

    K is 128 throughout this sequence, so an output tile takes exactly two
    k-tiles - few enough to show a whole one.

    Nothing has produced anything yet, so every output port is red and no
    operation between buffers can fire. The one thing that can run is the load
    from memory, drawn as a pipe with no source buffer: global memory is not a
    buffer in this model.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    placed = _bn256_layout(ax, active={(0, 0)})
    draw_source_pipe(ax, placed[(0, 0)]["in"], "TMA load")
    _bn256_chrome(
        ax, 1, "only the TMA load is running; every buffer is still empty",
        "Box area is proportional to capacity, so the two accumulators "
        "outweigh all six input slots put together.")
    fig.tight_layout()
    return fig


def bn256_state2_tma_and_mma():
    """State 2: the TMA load and the first MMA run at once.

    With K=128 this MMA is k-tile 0 of 2 for the first output tile.

    Slot 0 is full, so its output port is green and the MMA can read it; the TMA
    load has moved on to slot 1. That overlap is the whole point of the multi-
    stage ring - the load for the next k-tile runs while the MMA consumes the
    last one.

    The MMA buffer's ports do not change while it accumulates: its input stays
    green because the MMA keeps writing to it, and its output stays red until
    the last k-tile of the tile is in.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    states = {
        (0, 0): (NOT_READY, READY),   # slot 0 full, being consumed by the MMA
    }
    placed = _bn256_layout(ax, states, active={(0, 0), (0, 1), (1, 0)})
    draw_source_pipe(ax, placed[(0, 1)]["in"], "TMA load")
    draw_pipe_between(ax, placed[(0, 0)]["out"], placed[(1, 0)]["in"], "MMA",
                      label_dx=0.34)
    _bn256_chrome(
        ax, 2, "the TMA load and the first MMA run at the same time",
        "Slot 0 is full so the MMA can read it — this is k-tile 0 of 2 — while "
        "the load has moved on to slot 1, the overlap the ring exists for.")
    fig.tight_layout()
    return fig



def bn256_state3_ring_advances():
    """State 3: the ring has advanced one slot.

    Slot 0 has been consumed, so a buffer-free signal returned it to empty -
    input green, output red - and the load has moved on to slot 2 while the MMA
    reads slot 1.

    With K=128 slot 1 is the *last* k-tile of this output tile, so the
    accumulator is about to be complete; its output is still red only because
    this MMA has not finished yet. Slot 2 is therefore already the next output
    tile's first k-tile - the ring runs ahead across the tile boundary.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    states = {
        (0, 1): (NOT_READY, READY),   # slot 1 full, being consumed
    }
    placed = _bn256_layout(ax, states, active={(0, 1), (0, 2), (1, 0)})
    draw_source_pipe(ax, placed[(0, 2)]["in"], "TMA load")
    draw_pipe_between(ax, placed[(0, 1)]["out"], placed[(1, 0)]["in"], "MMA",
                      label_dx=0.34)
    _bn256_chrome(
        ax, 3, "the ring advances: the load is on slot 2, the MMA on slot 1",
        "Slot 1 is the last of this tile's two k-tiles, so slot 2 already holds "
        "the next output tile's first — the ring runs ahead across the boundary.")
    fig.tight_layout()
    return fig


def bn256_state4_data_ready_and_ld():
    """State 4: three operations at once, on both accumulators.

    The last k-tile of the first output tile finished, so a data-ready signal
    fired on MMA buffer 0: its output port is green and its input red - it is
    full, and nothing may overwrite it until the drain is done. That is what
    lets tcgen05.ld run, pulling the accumulator into registers.

    Meanwhile the next output tile's MMA has already started, on MMA buffer 1,
    reading slot 2 - and the load has moved on to slot 3. This is the reason
    there are two accumulators: one drains while the other fills.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    states = {
        (0, 2): (NOT_READY, READY),   # slot 2 full, feeding the next tile's MMA
        (1, 0): (NOT_READY, READY),   # accumulator 0 complete: data ready fired
    }
    placed = _bn256_layout(ax, states,
                           active={(0, 2), (0, 3), (1, 0), (1, 1), (2, 0)})
    draw_source_pipe(ax, placed[(0, 3)]["in"], "TMA load")
    draw_pipe_between(ax, placed[(1, 0)]["out"], placed[(2, 0)]["in"],
                      "tcgen05.ld", label_dx=0.34)
    draw_pipe_between(ax, placed[(0, 2)]["out"], placed[(1, 1)]["in"], "MMA",
                      label_dx=0.34)
    _bn256_chrome(
        ax, 4, "three operations at once — load, MMA, and the accumulator drain",
        "MMA buffer 0 took a data-ready signal, so its output opened and its "
        "input closed, and tcgen05.ld drains it. The next output tile's MMA is "
        "already running on MMA buffer 1 — the reason there are two.")
    fig.tight_layout()
    return fig


FIGURES = {
    "buffer-kinds": buffer_kinds,
    "bn256-state1-tma-load": bn256_state1_tma_load,
    "bn256-state2-tma-and-mma": bn256_state2_tma_and_mma,
    "bn256-state3-ring-advances": bn256_state3_ring_advances,
    "bn256-state4-data-ready-and-ld": bn256_state4_data_ready_and_ld,
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
