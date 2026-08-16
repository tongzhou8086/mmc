"""Does an L2 evict_first hint on the epilogue TMA store help?

    python benchmarks/sweep_store_hint.py           # 18 square shapes

C is written once and never read again, so with the default policy its lines
sit in L2 competing with the A/B panels neighbouring CTAs are about to reuse -
the working set GSM exists to protect. Tagging the store
`cp.async.bulk.tensor...L2::cache_hint` with a `createpolicy.fractional.
L2::evict_first` descriptor makes those lines the first eviction candidates.

Base and hinted kernels are measured in the same process, interleaved in the
same shuffled passes, so each pair can be subtracted directly. Only the store
instruction differs between them.

None of these kernels is registered as an autotune candidate; the specs are
synthesized here, so mmc/_kernels.py is untouched.
"""

import argparse
import random
import statistics
import sys

import torch
from triton.testing import do_bench

import mmc
from mmc._kernels import KernelSpec
from mmc._runtime import runtime_for

SHAPES = list(range(3072, 20481, 1024))
CONFIGS = [(bk, gsm) for bk in (64, 128) for gsm in (8, 12, 16)]

STEM = {64: "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc",
        128: "bf16-single-ns2-store2-bk128-bn512-load256-w8-splitacc"}


def spec_for(bk, gsm, hint):
    name = STEM[bk] + ("" if gsm == 8 else f"-gsm{gsm}") + ("-storehint" if hint else "")
    return KernelSpec(name, bk, 384, 230400, m_multiple=256, n_multiple=512)


def tflops(m, n, k, ms):
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shapes", nargs="*", type=int, default=SHAPES)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    runtime = runtime_for(torch.cuda.current_device())
    rng = random.Random(0)
    rows = {}

    for shape in args.shapes:
        m = n = k = shape
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        entries = []
        for bk, gsm in CONFIGS:
            for hint in (False, True):
                spec = spec_for(bk, gsm, hint)
                label = f"{bk}/{gsm}{'h' if hint else ''}"
                entries.append((label, spec,
                                (lambda s=spec: runtime.launch_bf16(s, a, b, out))))
        entries.append(("torch", None, lambda: torch.matmul(a, b, out=out)))

        # the hint must not change the result
        reference = torch.matmul(a, b)
        for label, spec, fn in entries:
            if spec is None:
                continue
            fn()
            torch.cuda.synchronize()
            err = (out.float() - reference.float()).abs().max().item()
            if err > 0.05 * reference.float().abs().max().item():
                print(f"FAILED at {shape}, {label}: max abs error {err}",
                      file=sys.stderr)
                sys.exit(1)
        del reference

        samples = {label: [] for label, _, _ in entries}
        for _ in range(args.passes):
            order = entries[:]
            rng.shuffle(order)
            for label, _spec, fn in order:
                samples[label].append(do_bench(fn, warmup=1000, rep=1000))
        rows[shape] = {label: tflops(m, n, k, statistics.median(v))
                       for label, v in samples.items()}
        print(f"# {shape} done", flush=True)
        del a, b, out
        torch.cuda.empty_cache()

    header = ("| Shape | " + " | ".join(
        f"BK={bk} GSM={gsm} {t}" for bk, gsm in CONFIGS for t in ("base", "hint"))
        + " | torch.matmul |")
    print("\n## BN=512 splitacc, epilogue store L2 policy\n")
    print(header)
    print("|---:" * (2 * len(CONFIGS) + 2) + "|")
    for shape in args.shapes:
        r = rows[shape]
        cells = " | ".join(f"{r[f'{bk}/{gsm}{s}']:.0f}"
                           for bk, gsm in CONFIGS for s in ("", "h"))
        print(f"| {shape}³ | {cells} | {r['torch']:.0f} |")

    print("\n## hint minus base, per configuration (%)\n")
    print("| Shape | " + " | ".join(f"BK={bk} GSM={gsm}" for bk, gsm in CONFIGS)
          + " |")
    print("|---:" * (len(CONFIGS) + 1) + "|")
    deltas = {c: [] for c in CONFIGS}
    for shape in args.shapes:
        r = rows[shape]
        cells = []
        for bk, gsm in CONFIGS:
            base, hint = r[f"{bk}/{gsm}"], r[f"{bk}/{gsm}h"]
            d = (hint - base) / base
            deltas[(bk, gsm)].append(d)
            cells.append(f"{d:+.1%}")
        print(f"| {shape}³ | " + " | ".join(cells) + " |")
    print("| **mean** | " + " | ".join(
        f"**{sum(v) / len(v):+.1%}**" for v in deltas.values()) + " |")


if __name__ == "__main__":
    main()
