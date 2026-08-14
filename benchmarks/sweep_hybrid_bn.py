"""Sweep the hybrid BN schedule: BN=512 splitacc bulk + a narrower tail.

    python benchmarks/sweep_hybrid_bn.py            # 18 square shapes
    python benchmarks/sweep_hybrid_bn.py 7168 11264

Prints a markdown table in the shape blog/perf-data.md expects: BK 64/128 x
GSM 8/12/16, plus torch.matmul as the cross-node control.

Wave quantization: the kernels are persistent, so 74 clusters are resident and
runtime is `ceil(T/74)` tile-times. Splitting N yields complete, independent
output tiles, so the partial last wave can be re-tiled by simply launching a
narrower kernel over the remaining columns - no partial accumulators, no
workspace, no reduction. This measures the whole idea end to end.

None of these kernels is registered as an autotune candidate; the specs are
synthesized here, so mmc/_kernels.py is untouched.
"""

import argparse
import math
import random
import statistics
import sys

import torch
from triton.testing import do_bench

import mmc
from mmc._kernels import KernelSpec
from mmc._runtime import runtime_for

RESIDENT = 74
BM_CLUSTER = 256
SHAPES = list(range(3072, 20481, 1024))
CONFIGS = [(bk, gsm) for bk in (64, 128) for gsm in (8, 12, 16)]

# relative per-tile efficiency, measured in docs/bn128-wave-quantization.md
EFFICIENCY = {512: 1.00, 256: 0.97, 128: 0.835}

# (bn, bk) -> (stem, threads, shared_bytes). GSM is appended as a suffix, with
# GSM=8 being the unsuffixed parent for the kernels that ship at GSM=8.
FAMILY = {
    (512, 64): ("bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc", 384, 230400),
    (512, 128): ("bf16-single-ns2-store2-bk128-bn512-load256-w8-splitacc", 384, 230400),
    (256, 64): ("bf16-double-ns6-store2-bk64", 256, 230400),
    (256, 128): ("bf16-double-ns3-store2-bk128", 256, 230400),
    (128, 64): ("bf16-double-ns8-store2-bk64-bn128", 256, 230400),
    (128, 128): ("bf16-double-ns4-store2-bk128-bn128", 256, 230400),
}
# families whose GSM=8 build carries an explicit -gsm8 suffix
ALWAYS_SUFFIXED = {("bf16-double-ns4-store2-bk128-bn128")}


def spec_for(bn, bk, gsm):
    stem, threads, shared = FAMILY[(bn, bk)]
    name = stem if (gsm == 8 and stem not in ALWAYS_SUFFIXED) else f"{stem}-gsm{gsm}"
    return KernelSpec(name, bk, threads, shared,
                      m_multiple=256, n_multiple=bn)


def waves(tiles):
    return math.ceil(tiles / RESIDENT)


def plan(m, n, bulk_bn=512):
    """Best (n_a, tail_bn) column split, or None when no split helps.

    Depends only on the shape and the tile widths, so it is the same for every
    BK/GSM configuration.
    """
    tm = math.ceil(m / BM_CLUSTER)
    base = waves(tm * math.ceil(n / bulk_bn))
    best, chosen = base, None
    for tail_bn in (256, 128):
        for a in range(n // bulk_bn + 1):
            n_a, n_b = bulk_bn * a, n - bulk_bn * a
            if n_b % tail_bn:
                continue
            cost = waves(tm * a) if n_a else 0
            if n_b:
                cost += (waves(tm * (n_b // tail_bn)) * (tail_bn / bulk_bn)
                         / EFFICIENCY[tail_bn])
            if cost < best - 1e-9:
                best, chosen = cost, (n_a, tail_bn)
    return None if chosen is None else (chosen[0], chosen[1], base, best)


def make_runner(runtime, m, n, k, bk, gsm, split, a, b, out):
    bulk = spec_for(512, bk, gsm)
    if split is None:
        return lambda: runtime.launch_bf16(bulk, a, b, out)
    n_a, tail_bn = split[0], split[1]
    tail = spec_for(tail_bn, bk, gsm)

    def run():
        if n_a:
            runtime.launch_bf16_slice(bulk, a, b, out, 0, n_a)
        runtime.launch_bf16_slice(tail, a, b, out, n_a, n - n_a)
    return run


def tflops(m, n, k, ms):
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shapes", nargs="*", type=int, default=SHAPES)
    ap.add_argument("--passes", type=int, default=3,
                    help="shuffled measurement passes; the median is reported")
    args = ap.parse_args()

    runtime = runtime_for(torch.cuda.current_device())
    rng = random.Random(0)
    rows, plans = {}, {}

    for shape in args.shapes:
        m = n = k = shape
        split = plan(m, n)
        plans[shape] = split
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        entries = [(f"{bk}/{gsm}",
                    make_runner(runtime, m, n, k, bk, gsm, split, a, b, out))
                   for bk, gsm in CONFIGS]
        # the uniform BN=512 baseline, measured in the same process so the
        # hybrid is compared against it directly rather than across runs. When
        # no split was planned the hybrid IS the uniform kernel, so skip it.
        if split is not None:
            entries += [(f"u{bk}/{gsm}",
                         make_runner(runtime, m, n, k, bk, gsm, None, a, b, out))
                        for bk, gsm in CONFIGS]
        entries.append(("torch", lambda: torch.matmul(a, b, out=out)))

        # correctness once per shape, on the first configuration
        entries[0][1]()
        torch.cuda.synchronize()
        reference = torch.matmul(a, b)
        err = (out.float() - reference.float()).abs().max().item()
        if err > 0.05 * reference.float().abs().max().item():
            print(f"FAILED correctness at {shape}: max abs error {err}",
                  file=sys.stderr)
            sys.exit(1)
        del reference

        samples = {label: [] for label, _ in entries}
        for _ in range(args.passes):
            order = entries[:]
            rng.shuffle(order)
            for label, fn in order:
                samples[label].append(do_bench(fn, warmup=1000, rep=1000))
        rows[shape] = {label: tflops(m, n, k, statistics.median(v))
                       for label, v in samples.items()}
        print(f"# {shape}: split={split} "
              + " ".join(f"{lab}={rows[shape][lab]:.0f}"
                         for lab, _ in entries), flush=True)
        del a, b, out
        torch.cuda.empty_cache()

    print("\n## BN=512 splitacc + adaptive tail\n")
    header = "| Shape | " + " | ".join(
        f"BK={bk} GSM={gsm}" for bk, gsm in CONFIGS) + " | torch.matmul |"
    print(header)
    print("|---:" * (len(CONFIGS) + 2) + "|")
    for shape in args.shapes:
        cells = " | ".join(f"{rows[shape][f'{bk}/{gsm}']:.0f}"
                           for bk, gsm in CONFIGS)
        print(f"| {shape}³ | {cells} | {rows[shape]['torch']:.0f} |")

    print("\n## Uniform BN=512 splitacc, same run\n")
    print(header)
    print("|---:" * (len(CONFIGS) + 2) + "|")
    for shape in args.shapes:
        r = rows[shape]
        cells = " | ".join(f"{r.get(f'u{bk}/{gsm}', r[f'{bk}/{gsm}']):.0f}"
                           for bk, gsm in CONFIGS)
        print(f"| {shape}³ | {cells} | {r['torch']:.0f} |")

    print("\n<!-- splits chosen per shape")
    for shape in args.shapes:
        s = plans[shape]
        print(f"  {shape}: " + ("uniform BN=512" if s is None else
              f"BN=512 over N[0:{s[0]}], BN={s[1]} over N[{s[0]}:{shape}] "
              f"({s[2]:.2f} -> {s[3]:.2f} tile-times)"))
    print("-->")


if __name__ == "__main__":
    main()
