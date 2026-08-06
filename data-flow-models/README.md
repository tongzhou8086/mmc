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

| | green | red |
|---|---|---|
| **input port** | buffer is free to be overwritten | buffer still holds data in use |
| **output port** | data is ready to be consumed | data is not ready yet |

**An operation may fire only when its source's output port and its destination's
input port are both green.**

That is the whole model, and it is what makes it useful: synchronization is
nothing more than flipping port colors. Every mbarrier in a kernel toggles one
port, and every stall is an operation waiting for one of its two ports to turn
green. Scheduling questions — how deep to make a ring, when to release an
accumulator, whether to split a barrier — become questions about which port
flips when.

## Figures

### `per-slot-initial-state`

One buffer of each kind, in the state the pipeline starts in: every buffer is
empty, so every input port is green (free to overwrite) and every output port is
red (nothing to hand on). Neither operation can fire yet — each is waiting for
its source's output port to turn green, which only the upstream producer can do.

![Per-slot data flow, initial state](figures/per-slot-initial-state.png)
