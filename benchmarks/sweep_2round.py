"""GSM sweep for the fourth design: BN=512 accumulator, BN=256-sized slots.

    python benchmarks/sweep_2round.py

Each TMA slot holds A plus one accumulator panel's B, so it is 32 KB and six
fit where four 48 KB slots did. A slot is filled twice per visit: round 1 writes
A and panel 0's B, round 2 overwrites only B for panel 1 and reuses the resident
A. That gives BN=512's arithmetic intensity with BN=256's ring depth.

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
DESIGNS = [("2round", "bf16-double-ns6-store2-bk64-bn512-2round"),
           ("splitacc", "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc")]


def spec(stem, gsm):
    name = stem if gsm == 8 else f"{stem}-gsm{gsm}"
    return KernelSpec(name, 64, 384, 230400, m_multiple=256, n_multiple=512)


def main():
    rt = runtime_for(torch.cuda.current_device())
    rng = random.Random(0)
    rows = {}
    for S in SHAPES:
        a = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(S, S, device="cuda", dtype=torch.bfloat16)
        entries = [(f"{tag}/gsm{g}",
                    (lambda s=spec(stem, g): rt.launch_bf16(s, a, b, out)))
                   for tag, stem in DESIGNS for g in GSMS]
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

    for tag, _ in DESIGNS:
        print(f"\n## {tag}\n")
        print("| Shape | " + " | ".join(f"BK=64 GSM={g}" for g in GSMS)
              + " | torch.matmul |")
        print("|---:" * (len(GSMS) + 2) + "|")
        for S in SHAPES:
            cells = " | ".join(f"{rows[S][f'{tag}/gsm{g}']:.0f}" for g in GSMS)
            print(f"| {S}³ | {cells} | {rows[S]['torch']:.0f} |")

    print("\n## 2round minus splitacc, best GSM of each (%)\n")
    print("| Shape | 2round | splitacc | delta |")
    print("|---:|---:|---:|---:|")
    for S in SHAPES:
        a4 = max(rows[S][f"2round/gsm{g}"] for g in GSMS)
        a3 = max(rows[S][f"splitacc/gsm{g}"] for g in GSMS)
        print(f"| {S}³ | {a4:.0f} | {a3:.0f} | {(a4 - a3) / a3:+.1%} |")


if __name__ == "__main__":
    main()
