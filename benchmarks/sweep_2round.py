"""GSM/BK sweep for the fourth design: BN=512 accumulator, BN=256-sized slots.

    python benchmarks/sweep_2round.py

Each TMA slot holds A plus one accumulator panel's B, so it is 32 KB and six
fit where four 48 KB slots did. A slot is filled twice per visit: round 1 writes
A and panel 0's B, round 2 overwrites only B for panel 1 and reuses the resident
A. That gives BN=512's arithmetic intensity with BN=256's ring depth: six slots
at BK=64, and three at BK=128 where a slot is twice as large.

Emits a markdown table in the shape blog/perf-data.md expects, and checks every
kernel against torch.matmul at every shape first.
"""
import random, statistics, sys
import torch
from triton.testing import do_bench
import mmc
from mmc._kernels import KernelSpec
from mmc._runtime import runtime_for

SHAPES = list(range(3072, 20481, 1024))
GSMS = (8, 12, 16)
# design 4 and design 3 are measured in the same process: absolute TFLOP/s
# vary by a few percent between nodes, so the only sound comparison is a paired
# one. The cuBLAS column doubles as the control.
BKS = (64, 128)
DESIGNS = [
    ("2round", {64: "bf16-double-ns6-store2-bk64-bn512-2round",
                128: "bf16-double-ns3-store2-bk128-bn512-2round"}),
    ("splitacc", {64: "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc",
                  128: "bf16-single-ns2-store2-bk128-bn512-load256-w8-splitacc"}),
]
CONFIGS = [(tag, bk, g) for tag, _ in DESIGNS for bk in BKS for g in GSMS]


def spec(stem, bk, gsm):
    name = stem if gsm == 8 else f"{stem}-gsm{gsm}"
    return KernelSpec(name, bk, 384, 230400, m_multiple=256, n_multiple=512)


def main():
    rt = runtime_for(torch.cuda.current_device())
    rng = random.Random(0)
    rows = {}
    for S in SHAPES:
        a = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(S, S, device="cuda", dtype=torch.bfloat16)
        entries = [(f"{tag}/bk{bk}/gsm{g}",
                    (lambda s=spec(stems[bk], bk, g): rt.launch_bf16(s, a, b, out)))
                   for tag, stems in DESIGNS for bk in BKS for g in GSMS]
        entries.append(("torch", lambda: torch.matmul(a, b, out=out)))

        ref = torch.matmul(a, b)
        for label, fn in entries[:-1]:
            fn(); torch.cuda.synchronize()
            err = (out.float() - ref.float()).abs().max().item()
            if err > 0.05 * ref.float().abs().max().item():
                print(f"FAILED {S} {label}: {err}", file=sys.stderr); sys.exit(1)
        del ref

        samples = {l: [] for l, _ in entries}
        for _ in range(3):
            order = entries[:]; rng.shuffle(order)
            for l, fn in order:
                samples[l].append(do_bench(fn, warmup=1000, rep=1000))
        rows[S] = {l: 2 * S**3 / (statistics.median(v) * 1e-3) / 1e12
                   for l, v in samples.items()}
        print(f"# {S} " + " ".join(f"{l}={rows[S][l]:.0f}" for l, _ in entries),
              flush=True)
        del a, b, out; torch.cuda.empty_cache()

    cols = [(bk, g) for bk in BKS for g in GSMS]
    for tag, _ in DESIGNS:
        print(f"\n## {tag}\n")
        print("| Shape | " + " | ".join(f"BK={bk} GSM={g}" for bk, g in cols)
              + " | torch.matmul |")
        print("|---:" * (len(cols) + 2) + "|")
        for S in SHAPES:
            cells = " | ".join(f"{rows[S][f'{tag}/bk{bk}/gsm{g}']:.0f}"
                               for bk, g in cols)
            print(f"| {S}³ | {cells} | {rows[S]['torch']:.0f} |")

    print("\n## 2round minus splitacc, best config of each (%)\n")
    print("| Shape | 2round | splitacc | delta |")
    print("|---:|---:|---:|---:|")
    for S in SHAPES:
        best = {t: max(rows[S][f"{t}/bk{bk}/gsm{g}"] for bk, g in cols)
                for t, _ in DESIGNS}
        a4, a3 = best["2round"], best["splitacc"]
        print(f"| {S}³ | {a4:.0f} | {a3:.0f} | {(a4 - a3) / a3:+.1%} |")

    print("\n## 2round minus splitacc, paired within each config (%)\n")
    print("| Config | " + " | ".join(f"{S}³" for S in SHAPES) + " |")
    print("|---:" * (len(SHAPES) + 1) + "|")
    for bk, g in cols:
        d = [(rows[S][f"2round/bk{bk}/gsm{g}"] / rows[S][f"splitacc/bk{bk}/gsm{g}"] - 1)
             for S in SHAPES]
        print(f"| BK={bk} GSM={g} | " + " | ".join(f"{x:+.1%}" for x in d) + " |")


if __name__ == "__main__":
    main()
