# Data-flow models

Figures for reasoning about GEMM pipeline scheduling on B200 as a data-flow
problem. `render.py` generates every figure into `figures/` as both SVG and PNG:

```bash
python data-flow-models/render.py
```

## The model

Four kinds of buffer:

| Buffer | Lives in | Holds | Written by | Read by |
|---|---|---|---|---|
| TMA buffer | SMEM | A / B input tiles | TMA | MMA |
| MMA buffer | TMEM | the accumulator | MMA | `tcgen05.ld` |
| `tcgen05.ld` buffer | RMEM | `tcgen05.ld` results | `tcgen05.ld` | epilogue |
| Store buffer | SMEM | epilogue output staged for the write-back | epilogue | TMA store |

Each buffer is a box with two **ports**: an input port on top and an output port
on the bottom. An **operation** is an edge from one buffer's output port to
another buffer's input port — `MMA` runs from the TMA buffer to the MMA buffer,
`Epilogue` from the MMA buffer to the store buffer.

A port is a switch with two states:

| | green, vertical bar | red, horizontal bar |
|---|---|---|
| **input port** | buffer is free to be overwritten | buffer still holds data in use |
| **output port** | data is ready to be consumed | data is not ready yet |

Each port carries its state twice over: colour, plus a glyph for the state of the
path through it — a vertical bar where data may pass, a horizontal bar where it
is barred. The glyph keeps the figures readable in grayscale and for red-green
colourblind readers, and it describes the *path* rather than the buffer's
contents, which matters because green means opposite occupancy on the two ports:
a green input port is an empty buffer, a green output port a full one.

(The prose deliberately avoids "open" and "closed" for these states — an open
gate suggests flow but a closed circuit also means flow, and readers split on
which one is permissive.)

**An operation may fire only when its source's output port and its destination's
input port are both green.**

A **signal** is the only thing that changes a port's state, and it is addressed
to a buffer rather than to one port — one signal flips both. Each buffer takes
two kinds:

| Signal | Effect on the buffer it is sent to |
|---|---|
| **buffer free** | opens the input port, closes the output port — the data has been consumed, so it may be refilled |
| **data ready** | opens the output port, closes the input port — the buffer has been filled, so it may be read |

That is the whole model, and it is what makes it useful: synchronization is
nothing more than flipping port states. Every mbarrier in a kernel toggles one
port, and every stall is an operation waiting for one of its two ports to turn
green. Scheduling questions — how deep to make a ring, when to release an
accumulator, whether to split a barrier — become questions about which port
flips when.

## Figures

### `buffer-kinds`

The four kinds of buffer in a 2x2 grid, with no operations between them. They
are placed so the data flow runs clockwise — TMA to MMA across the top, down to
the `tcgen05.ld` buffer, then left to the store buffer — which is the shape the
operation edges will take once they are drawn. This figure introduces the
vocabulary only: what buffers exist, where they live, and that
each has an input port on top and an output port on the bottom. Drawn in the
initial state — every buffer empty, so every input port green and every output
port red.

It carries no port-state legend: the glyphs and the one-line caption say enough,
and four legend rows next to four buffers was more text than the figure needed.

![The four kinds of buffer](figures/buffer-kinds.png)

### `operation-as-pipe`

How an operation is modelled: a pipe joining the source buffer's output port to
the destination buffer's input port, drawn here for the MMA between the TMA
buffer and the MMA buffer. The pipe is only the connection — it holds no state
of its own; the ports do.

The state shown is one where the MMA may actually start, rather than a contrived
all-green one: the TMA buffer is full (input red, output green) and the MMA
buffer is empty (input green, output red), which is exactly both-green across
the pipe.

![An operation is a pipe between two ports](figures/operation-as-pipe.png)

### `signal-flips-a-port`

What a signal is, drawn on the same TMA → MMA pair as the pipe figure: before
and after one MMA completes. The MMA consumed the TMA buffer, so a buffer-free
signal is delivered there and flips both of its ports (ringed) — the output
closes because the data is gone, the input opens so TMA may refill it.

The MMA buffer deliberately does *not* flip: it accumulates over every k-tile,
and its data-ready signal fires only after the last one.

![A signal is what flips a port](figures/signal-flips-a-port.png)

The same sequence is also generated as a GIF, `signal-flips-a-port.gif`, for
posts that can show one. The static figure is the primary — it survives print,
PDF and feed readers — and the two are generated from the same primitives, so
they cannot drift apart.

![A signal is what flips a port, animated](figures/signal-flips-a-port.gif)

### `per-slot-initial-state`

One buffer of each kind, in the state the pipeline starts in: every buffer is
empty, so every input port is green (free to overwrite) and every output port is
red (nothing to hand on). Neither operation can fire yet — each is waiting for
its source's output port to turn green, which only the upstream producer can do.

![Per-slot data flow, initial state](figures/per-slot-initial-state.png)
