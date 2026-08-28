"""CLC vs the static persistent partition, on the shapes where it should matter.

    python benchmarks/sweep_clc.py

Design 5 is design 4 (BN=512 accumulator, BN=256-sized slots, two rounds per
slot) with cluster launch control: the grid is one cluster per output tile and a
cluster takes its next tile by cancelling one that has not launched yet, instead
of walking a static stride through the tile list. Both designs are measured in
the same process against the same cuBLAS control, because absolute TFLOP/s vary
by a few percent between nodes - only a paired comparison is sound.

Small shapes first: that is where the tile count is a small multiple of the
cluster count, so the ragged last wave costs the most.
"""
import argparse, random, statistics, sys
import torch
from triton.testing import do_bench
import mmc
from mmc._kernels import KernelSpec
from mmc._runtime import runtime_for

GSMS = (8, 12, 16)
BKS = (64, 128)
DESIGNS = [
    # claim depth is how many tiles a cluster runs ahead of its consumers.
    # Shallow wins: a deep claim hoards tiles at the tail of the grid.
    ("clc1", {64: "bf16-double-ns6-store2-bk64-bn512-2round-clc1",
              128: "bf16-double-ns3-store2-bk128-bn512-2round-clc1"}, True),
    ("clc2", {64: "bf16-double-ns6-store2-bk64-bn512-2round-clc2",
              128: "bf16-double-ns3-store2-bk128-bn512-2round-clc2"}, True),
    ("clc3", {64: "bf16-double-ns6-store2-bk64-bn512-2round-clc3",
              128: "bf16-double-ns3-store2-bk128-bn512-2round-clc3"}, True),
    # The design-4 kernel launched with the CLC grid. Its static stride then
    # equals the cluster count, so every cluster runs exactly one tile: this is
    # "full grid, no persistence, no CLC", the control that says how much of any
    # CLC delta is stealing and how much is just the grid shape.
    ("fullgrid", {64: "bf16-double-ns6-store2-bk64-bn512-2round",
                  128: "bf16-double-ns3-store2-bk128-bn512-2round"}, True),
    ("2round", {64: "bf16-double-ns6-store2-bk64-bn512-2round",
                128: "bf16-double-ns3-store2-bk128-bn512-2round"}, False),
]


def spec(stem, bk, gsm, clc):
    name = stem if gsm == 8 else f"{stem}-gsm{gsm}"
    return KernelSpec(name, bk, 384, 230400, m_multiple=256, n_multiple=512,
                      clc=clc)


def tile_counts(s, sm_count):
    """Clusters of work, and how they land on the device."""
    clusters = (s // 256) * (s // 512)
    slots = sm_count // 2
    waves = clusters / slots
    return clusters, slots, waves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, nargs="*",
                    default=list(range(3072, 8193, 1024)))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    rt = runtime_for(torch.cuda.current_device())
    sm_count = rt.sm_count
    rng = random.Random(0)
    rows = {}

    print(f"# SMs={sm_count}")
    for S in args.shapes:
        clusters, slots, waves = tile_counts(S, sm_count)
        a = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(S, S, device="cuda", dtype=torch.bfloat16)
        entries = [(f"{tag}/bk{bk}/gsm{g}",
                    (lambda s=spec(stems[bk], bk, g, clc): rt.launch_bf16(s, a, b, out)))
                   for tag, stems, clc in DESIGNS for bk in BKS for g in GSMS]
        entries.append(("torch", lambda: torch.matmul(a, b, out=out)))

        ref = torch.matmul(a, b)
        scale = 0.05 * ref.float().abs().max().item()
        for label, fn in entries[:-1]:
            out.zero_()
            fn(); torch.cuda.synchronize()
            err = (out.float() - ref.float()).abs().max().item()
            if err > scale:
                print(f"FAILED {S} {label}: max abs err {err}", file=sys.stderr)
                sys.exit(1)
        del ref

        samples = {l: [] for l, _ in entries}
        for _ in range(args.reps):
            order = entries[:]; rng.shuffle(order)
            for l, fn in order:
                samples[l].append(do_bench(fn, warmup=1000, rep=1000))
        rows[S] = {l: 2 * S**3 / (statistics.median(v) * 1e-3) / 1e12
                   for l, v in samples.items()}
        print(f"# {S} clusters={clusters} slots={slots} waves={waves:.2f} "
              + " ".join(f"{l}={rows[S][l]:.0f}" for l, _ in entries), flush=True)
        del a, b, out; torch.cuda.empty_cache()

    cols = [(bk, g) for bk in BKS for g in GSMS]
    for tag, _, _ in DESIGNS:
        print(f"\n## {tag}\n")
        print("| Shape | " + " | ".join(f"BK={bk} GSM={g}" for bk, g in cols)
              + " | torch.matmul |")
        print("|---:" * (len(cols) + 2) + "|")
        for S in args.shapes:
            cells = " | ".join(f"{rows[S][f'{tag}/bk{bk}/gsm{g}']:.0f}"
                               for bk, g in cols)
            print(f"| {S}³ | {cells} | {rows[S]['torch']:.0f} |")

    tags = [t for t, _, _ in DESIGNS]
    print("\n## best of each design, and the wave picture\n")
    print("| Shape | clusters | waves | " + " | ".join(tags)
          + " | cuBLAS | best clc vs 2round |")
    print("|---:" * (len(tags) + 5) + "|")
    for S in args.shapes:
        clusters, slots, waves = tile_counts(S, sm_count)
        best = {t: max(rows[S][f"{t}/bk{bk}/gsm{g}"] for bk, g in cols)
                for t in tags}
        r, t = best["2round"], rows[S]["torch"]
        bestclc = max(best[k] for k in tags if k.startswith("clc"))
        print(f"| {S}³ | {clusters} | {waves:.2f} | "
              + " | ".join(f"{best[k]:.0f}" for k in tags)
              + f" | {t:.0f} | {(bestclc - r) / r:+.1%} |")

    print("\n## clc minus 2round, paired within each config (%)\n")
    print("| Config | " + " | ".join(f"{S}³" for S in args.shapes) + " |")
    print("|---:" * (len(args.shapes) + 1) + "|")
    for tag in [t for t in tags if t != "2round"]:
        for bk, g in cols:
            d = [(rows[S][f"{tag}/bk{bk}/gsm{g}"] / rows[S][f"2round/bk{bk}/gsm{g}"] - 1)
                 for S in args.shapes]
            print(f"| {tag} BK={bk} GSM={g} | " + " | ".join(f"{x:+.1%}" for x in d) + " |")


if __name__ == "__main__":
    main()
