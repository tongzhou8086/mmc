"""Prototype: cover one N range with a wide kernel and the tail with a narrow one.

    python benchmarks/prototype_adaptive_bn.py 7168 [12288 ...]

The BF16 kernels are persistent, so 74 clusters are resident and runtime is
`ceil(T / 74)` tile-times for `T = ceil(M/256) * ceil(N/BN)` tiles. The last
wave is partial and its idle clusters are pure loss. Splitting N (unlike
splitting K) yields complete, independent output tiles, so the tail can simply
be a second launch of a narrower kernel over the remaining columns - no partial
accumulators, no workspace, no reduction.

This measures whether the modelled gain survives a real second launch. It is a
benchmark-only path: nothing here is wired into mmc.matmul's dispatch.

See docs/bn128-wave-quantization.md for the model and the measured per-tile
penalty each narrower BN pays.
"""

import argparse
import math
import sys

import torch
from triton.testing import do_bench

import mmc
from mmc._kernels import BF16_KERNELS, KernelSpec
from mmc._runtime import runtime_for

BM_CLUSTER = 256                 # one cluster owns a 256 x BN output tile
RESIDENT = 74                    # 148-CTA persistent grid / 2 CTAs per cluster

# per-tile efficiency relative to BN=512, measured in docs/bn128-wave-quantization.md
EFFICIENCY = {512: 1.00, 256: 0.97, 128: 0.835}

# BK=64, GSM=8 - the plain candidates, all registered in _kernels.py
KERNELS = {
    512: "bf16-single-ns4-store2-bk64-bn512",
    256: "bf16-double-ns6-store2-bk64",
    128: "bf16-double-ns8-store2-bk64-bn128",
}

# BK=128, GSM=16 - the configuration that wins the blog sweep. These are not
# registered as autotune candidates (the GSM candidate set is not settled), so
# their specs are synthesized from the parent kernel below.
TUNED_KERNELS = {
    512: "bf16-single-ns2-store2-bk128-bn512-gsm16",
    256: "bf16-double-ns3-store2-bk128-gsm16",
    128: "bf16-double-ns4-store2-bk128-bn128-gsm16",
}

# name -> (bk, threads, shared_bytes) for kernels that have a cubin but no
# KernelSpec. The runtime loads cubins by name, so an ad-hoc spec is enough.
UNREGISTERED = {
    "bf16-single-ns2-store2-bk128-bn512-gsm16": (128, 256, 230400),
    "bf16-double-ns3-store2-bk128-gsm16": (128, 256, 230400),
    "bf16-double-ns4-store2-bk128-bn128-gsm16": (128, 256, 230400),
}


def spec_for(bn, tuned=False):
    name = (TUNED_KERNELS if tuned else KERNELS)[bn]
    for spec in BF16_KERNELS:
        if spec.name == name:
            return spec
    if name in UNREGISTERED:
        bk, threads, shared = UNREGISTERED[name]
        return KernelSpec(name, bk, threads, shared,
                          m_multiple=256, n_multiple=bn)
    raise SystemExit(f"kernel {name} is neither registered nor known here")


def waves(tiles):
    return math.ceil(tiles / RESIDENT)


def plan(m, n, bulk_bn):
    """Best (n_a, tail_bn) rectangle split, in BN=512 tile-time units.

    Returns None when no split beats running the bulk kernel over all of N.
    """
    tm = math.ceil(m / BM_CLUSTER)
    base = waves(tm * math.ceil(n / bulk_bn))
    best, chosen = base, None
    for tail_bn in (256, 128):
        if tail_bn >= bulk_bn:
            continue
        for a in range(n // bulk_bn + 1):
            n_a = bulk_bn * a
            n_b = n - n_a
            if n_b % tail_bn:
                continue
            cost = waves(tm * a) if n_a else 0
            if n_b:
                tiles_b = tm * (n_b // tail_bn)
                cost += (waves(tiles_b) * (tail_bn / bulk_bn)
                         / EFFICIENCY[tail_bn])
            if cost < best - 1e-9:
                best, chosen = cost, (n_a, tail_bn)
    if chosen is None:
        return None
    return chosen[0], chosen[1], base, best


def run_split(runtime, bulk_spec, tail_spec, a, b, out, n_a):
    n = b.shape[1]
    if n_a:
        runtime.launch_bf16_slice(bulk_spec, a, b, out, 0, n_a)
    if n - n_a:
        runtime.launch_bf16_slice(tail_spec, a, b, out, n_a, n - n_a)


def tflops(m, n, k, ms):
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shapes", nargs="+", type=int, metavar="N",
                    help="square shape (M = N = K)")
    ap.add_argument("--bulk-bn", type=int, default=512, choices=(512, 256))
    ap.add_argument("--compare", default="",
                    help="also bench these registered kernels uniformly, "
                         "comma-separated, as extra baselines")
    ap.add_argument("--tuned", action="store_true",
                    help="use the BK=128 / GSM=16 kernels instead of BK=64 / GSM=8")
    args = ap.parse_args()

    runtime = runtime_for(torch.cuda.current_device())
    print(f"device sm_count={runtime.sm_count}, resident clusters={RESIDENT}")

    for shape in args.shapes:
        m = n = k = shape
        found = plan(m, n, args.bulk_bn)
        print(f"\n=== {m}x{n}x{k}, bulk BN={args.bulk_bn} ===")
        if found is None:
            print("  no rectangle split beats the uniform kernel; skipping")
            continue
        n_a, tail_bn, base, modelled = found
        print(f"  split: BN={args.bulk_bn} over N[0:{n_a}], "
              f"BN={tail_bn} over N[{n_a}:{n}]")
        print(f"  model: {base:.2f} -> {modelled:.2f} tile-times "
              f"({(base - modelled) / base:+.1%})")

        bulk_spec = spec_for(args.bulk_bn, args.tuned)
        tail_spec = spec_for(tail_bn, args.tuned)
        print(f"  kernels: {bulk_spec.name} + {tail_spec.name}")
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        # correctness: the two launches together must reproduce the whole GEMM
        run_split(runtime, bulk_spec, tail_spec, a, b, out, n_a)
        torch.cuda.synchronize()
        reference = torch.matmul(a, b)
        max_err = (out.float() - reference.float()).abs().max().item()
        scale = reference.float().abs().max().item()
        print(f"  max abs error {max_err:.4g} (reference max {scale:.4g})")
        if max_err > 0.05 * scale:
            print("  FAILED: split result does not match torch.matmul")
            sys.exit(1)

        uniform = torch.empty_like(out)
        ms_uniform = do_bench(
            lambda: runtime.launch_bf16(bulk_spec, a, b, uniform),
            warmup=1000, rep=1000)
        ms_split = do_bench(
            lambda: run_split(runtime, bulk_spec, tail_spec, a, b, out, n_a),
            warmup=1000, rep=1000)
        ms_torch = do_bench(lambda: torch.matmul(a, b),
                                   warmup=1000, rep=1000)

        print(f"  uniform BN={args.bulk_bn}: {tflops(m, n, k, ms_uniform):8.1f} "
              f"TFLOP/s ({ms_uniform * 1e3:.1f} us)")
        print(f"  two-launch split:   {tflops(m, n, k, ms_split):8.1f} "
              f"TFLOP/s ({ms_split * 1e3:.1f} us)")
        print(f"  torch.matmul:       {tflops(m, n, k, ms_torch):8.1f} "
              f"TFLOP/s ({ms_torch * 1e3:.1f} us)")
        measured = (ms_uniform - ms_split) / ms_uniform
        print(f"  measured {measured:+.1%} vs modelled "
              f"{(base - modelled) / base:+.1%}")

        # extra whole-N baselines, so the split is compared against the best
        # known kernels on this node rather than across runs
        for name in [x for x in args.compare.split(",") if x]:
            other = next((k for k in BF16_KERNELS if k.name == name), None)
            if other is None:
                print(f"  {name}: not registered, skipped")
                continue
            if n % other.n_multiple or m % other.m_multiple:
                print(f"  {name}: shape not aligned, skipped")
                continue
            scratch = torch.empty_like(out)
            ms = do_bench(lambda: runtime.launch_bf16(other, a, b, scratch),
                          warmup=1000, rep=1000)
            print(f"  {name}: {tflops(m, n, k, ms):8.1f} TFLOP/s "
                  f"(split is {(ms - ms_split) / ms:+.1%} vs this)")


if __name__ == "__main__":
    main()
