"""Where does MMA issue block, and for how long?

    python benchmarks/profile_mma_waits.py 8192
    python benchmarks/profile_mma_waits.py 8192 --cluster 3 --out /tmp/t.png

The MMA warp blocks on exactly two things, and they are the two conditions the
data-flow model says any operation needs:

  * **data-ready**  — the TMA buffer it is about to read is not filled yet
  * **panel-free**  — the accumulator panel it is about to write is still being
    drained by the epilogue

The instrumented kernel probes each barrier with a non-blocking
`mbarrier.test_wait` first. A wait that is already satisfied records nothing;
only a genuine block reads the clock, waits, and reads it again. So the
instrumentation cannot manufacture the stalls it measures.

Output: a per-cluster timeline showing every blocked interval on a single time
axis, from the MMA warp's first instruction to its last, plus a text summary of
where the time went.
"""

import argparse
import sys
from pathlib import Path

import torch

import mmc
from mmc._kernels import KernelSpec
from mmc._runtime import runtime_for

KERNEL = "bf16-single-ns2-store2-bk128-bn512-load256-w8-splitacc-prof"
SPEC = KernelSpec(KERNEL, 128, 384, 230400, m_multiple=256, n_multiple=512)

PROF_MAX_EVENTS = 8192
PROF_SLICE_U64 = 2 + 4 * PROF_MAX_EVENTS

DATA_READY, PANEL_FREE, SENTINEL = 0, 1, 0xFFFF
KIND_NAME = {DATA_READY: "data-ready (TMA buffer not filled)",
             PANEL_FREE: "panel-free (accumulator still draining)"}
KIND_COLOUR = {DATA_READY: "#c98b3f", PANEL_FREE: "#8f4f7d"}


def decode(slice_u64):
    """-> (t_base, t_end, [(kind, detail, ti, k, t0, t1)])."""
    n = int(slice_u64[0])
    t_base = int(slice_u64[1])
    events, t_end = [], None
    for i in range(n):
        e = slice_u64[2 + 4 * i: 6 + 4 * i]
        meta = int(e[0])
        kind = (meta >> 48) & 0xFFFF
        if kind == SENTINEL:
            t_end = int(e[1])
            continue
        events.append((kind, (meta >> 40) & 0xFF, (meta >> 16) & 0xFFFFFF,
                       meta & 0xFFFF, int(e[1]), int(e[2])))
    return t_base, t_end, events


def render(t_base, t_end, events, path, shape, cluster, ghz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    INK, MUTED, GRID = "#1f2933", "#6b7684", "#e3e7ec"
    to_us = lambda c: (c - t_base) / (ghz * 1e3)
    span = to_us(t_end)

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 5.4),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.95})

    blocked = sum(t1 - t0 for _, _, _, _, t0, t1 in events)
    total = t_end - t_base

    # a window in the steady state, wide enough to show a few k-tiles
    zoom_lo = span * 0.50
    zoom_hi = min(zoom_lo + 6.0, span)

    for ax, (lo, hi, title) in zip(axes, [
            (0.0, span,
             f"MMA issue timeline · {shape}³ · cluster {cluster} · "
             f"{span:.0f} µs total, {100 * blocked / total:.1f}% blocked"),
            (zoom_lo, zoom_hi,
             f"zoom · {zoom_lo:.0f}–{zoom_hi:.0f} µs of the same run")]):
        width = hi - lo
        ax.add_patch(Rectangle((lo, 0.35), width, 0.30, facecolor="#dfe7ef",
                               edgecolor="none", zorder=2))
        for kind, _detail, _ti, _k, t0, t1 in events:
            a0, a1 = to_us(t0), to_us(t1)
            if a1 < lo or a0 > hi:
                continue
            ax.add_patch(Rectangle((a0, 0.35), max(a1 - a0, width * 3e-4), 0.30,
                                   facecolor=KIND_COLOUR.get(kind, "#999"),
                                   edgecolor="none", zorder=3))
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK,
                     loc="left", pad=10)
        ax.set_xlim(lo - width * 0.005, hi + width * 0.005)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("time since the MMA warp started (µs)", fontsize=10,
                      color=MUTED, labelpad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", length=0, colors=MUTED)

    axes[1].legend(handles=[Patch(facecolor="#dfe7ef", label="issuing"),
                            *[Patch(facecolor=KIND_COLOUR[k], label=KIND_NAME[k])
                              for k in (DATA_READY, PANEL_FREE)]],
                   frameon=False, fontsize=9.5, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shape", type=int, help="square shape (M = N = K)")
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--out", default="docs/figures/mma-wait-timeline.png")
    ap.add_argument("--clock-ghz", type=float, default=1.965,
                    help="SM clock, for converting cycles to microseconds")
    args = ap.parse_args()

    runtime = runtime_for(torch.cuda.current_device())
    n_clusters = (runtime.sm_count - runtime.sm_count % 2) // 2

    m = n = k = args.shape
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    prof = torch.zeros(n_clusters * PROF_SLICE_U64, dtype=torch.int64,
                       device="cuda")

    runtime.launch_bf16_prof(SPEC, a, b, out, prof)
    torch.cuda.synchronize()

    err = (out.float() - torch.matmul(a, b).float()).abs().max().item()
    print(f"max abs error vs torch.matmul: {err:.4g}")
    if err > 1.0:
        print("instrumented kernel is wrong; refusing to report a timeline",
              file=sys.stderr)
        sys.exit(1)

    host = prof.cpu().numpy()
    print(f"\n{'cluster':>8} {'events':>7} {'span µs':>9} {'blocked':>8} "
          f"{'data-ready':>11} {'panel-free':>11}")
    totals = {DATA_READY: 0, PANEL_FREE: 0}
    for c in range(n_clusters):
        t_base, t_end, ev = decode(host[c * PROF_SLICE_U64:(c + 1) * PROF_SLICE_U64])
        if t_end is None or not ev:
            continue
        span = (t_end - t_base) / (args.clock_ghz * 1e3)
        by = {DATA_READY: 0, PANEL_FREE: 0}
        for kind, _d, _ti, _k, t0, t1 in ev:
            by[kind] = by.get(kind, 0) + (t1 - t0)
        for kk in totals:
            totals[kk] += by.get(kk, 0)
        if c < 8 or c == args.cluster:
            blocked = sum(by.values())
            print(f"{c:>8} {len(ev):>7} {span:>8.0f}  "
                  f"{100 * blocked / (t_end - t_base):>6.1f}% "
                  f"{100 * by[DATA_READY] / (t_end - t_base):>10.1f}% "
                  f"{100 * by[PANEL_FREE] / (t_end - t_base):>10.1f}%")
    grand = sum(totals.values()) or 1
    print(f"\nacross all clusters, blocked time splits: "
          f"data-ready {100 * totals[DATA_READY] / grand:.1f}%, "
          f"panel-free {100 * totals[PANEL_FREE] / grand:.1f}%")

    t_base, t_end, ev = decode(
        host[args.cluster * PROF_SLICE_U64:(args.cluster + 1) * PROF_SLICE_U64])
    if t_end is None or not ev:
        print(f"cluster {args.cluster} recorded nothing", file=sys.stderr)
        sys.exit(1)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    render(t_base, t_end, ev, outp, args.shape, args.cluster, args.clock_ghz)


if __name__ == "__main__":
    main()
