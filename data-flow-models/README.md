# Data-flow models

Figures for reasoning about GEMM pipeline scheduling on B200 as a data-flow
problem. `render.py` generates every figure into `figures/` as both SVG and PNG:

```bash
python data-flow-models/render.py
```

## The model

Three kinds of buffer:

| Buffer | Lives in | Holds |
|---|---|---|
| TMA buffer | SMEM | A / B input tiles fetched by TMA |
| MMA buffer | TMEM | the MMA accumulator |
| Store buffer | SMEM | epilogue output staged for the write-back to HBM |

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

That is the whole model, and it is what makes it useful: synchronization is
nothing more than flipping port states. Every mbarrier in a kernel toggles one
port, and every stall is an operation waiting for one of its two ports to turn
green. Scheduling questions — how deep to make a ring, when to release an
accumulator, whether to split a barrier — become questions about which port
flips when.

## Figures

### `buffer-kinds`

The three kinds of buffer side by side, with no operations between them. This
one introduces the vocabulary only: what buffers exist, where they live, and
that each has an input port on top and an output port on the bottom. Drawn in
the initial state — every buffer empty, so every input port green and every
output port red.

![The three kinds of buffer](figures/buffer-kinds.png)

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

### `per-slot-initial-state`

One buffer of each kind, in the state the pipeline starts in: every buffer is
empty, so every input port is green (free to overwrite) and every output port is
red (nothing to hand on). Neither operation can fire yet — each is waiting for
its source's output port to turn green, which only the upstream producer can do.

![Per-slot data flow, initial state](figures/per-slot-initial-state.png)
