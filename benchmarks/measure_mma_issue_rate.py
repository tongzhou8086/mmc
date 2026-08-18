"""How often is an MMA issued, and what does that buy per unit time?

    python benchmarks/measure_mma_issue_rate.py 4096 8192 16384

Throughput decomposes into three terms:

    throughput = A x f x duty

  A     compute produced by one tcgen05.mma issue. Exact arithmetic, no
        measurement needed: 2 * M * N * K_inst FLOP.
  f     how often that issue happens inside the k loop.
  duty  the fraction of time spent in the k loop at all, the rest being the
        pause between output tiles while the accumulator drains.

The instrumented kernels record two timestamps per output tile in the MMA
warp - one before the tile's first MMA issue, one after its last - which gives
`duty` directly as a clock-free ratio. Combining that with the benchmarked
throughput yields A and f as measurements rather than fits.

Note that A is identical for the two designs compared here: tcgen05.mma caps N
at 256, so BN=512 is issued as two N=256 MMAs per k-tile rather than one wider
one. Any throughput difference between them is therefore a difference in rate,
not in compute per issue.
"""

import argparse
import statistics

import torch
from triton.testing import do_bench

import mmc
from mmc._kernels import KernelSpec
from mmc._runtime import runtime_for

BM, CG, MMA_K, N_INST = 128, 2, 16, 256
A_FLOP = 2 * (CG * BM) * N_INST * MMA_K          # one tcgen05.mma issue

RATE_MAX_TILES = 4096
RATE_SLICE_U64 = 2 + 2 * RATE_MAX_TILES

DESIGNS = [
    ("design 1  BN=256", "bf16-double-ns6-store2-bk64-mmarate",
     "bf16-double-ns6-store2-bk64", 256, 64, 256),
    ("design 2  BN=512", "bf16-single-ns4-store2-bk64-bn512-mmarate",
     "bf16-single-ns4-store2-bk64-bn512", 512, 64, 512),
]


def spec(name, bk, n_multiple):
    return KernelSpec(name, bk, 256, 230400, m_multiple=256,
                      n_multiple=n_multiple)


def phases(host, clusters):
    """-> (mean compute cycles, mean pause cycles) per output tile."""
    comp, pause = [], []
    for c in range(clusters):
        sl = host[c * RATE_SLICE_U64:(c + 1) * RATE_SLICE_U64]
        n = int(sl[0])
        stamps = [(int(sl[2 + 2 * i]), int(sl[3 + 2 * i])) for i in range(n)]
        for i, (b, e) in enumerate(stamps):
            if b == 0:
                continue                       # tile whose first stamp was lost
            comp.append(e - b)
            if i + 1 < len(stamps) and stamps[i + 1][0]:
                pause.append(stamps[i + 1][0] - e)
    return statistics.median(comp), statistics.median(pause)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shapes", nargs="+", type=int)
    args = ap.parse_args()

    runtime = runtime_for(torch.cuda.current_device())
    clusters = (runtime.sm_count - runtime.sm_count % 2) // 2
    print(f"A = 2 * {CG * BM} * {N_INST} * {MMA_K} = {A_FLOP / 1e6:.3f} MFLOP "
          f"per tcgen05.mma issue (identical for both designs)\n")

    hdr = (f"{'design':<18} {'shape':>6} {'measured':>9} {'duty':>7} "
           f"{'k-loop rate':>12} {'f':>12} {'issue every':>12}")
    print(hdr)
    print("-" * len(hdr))
    for shape in args.shapes:
        m = n = k = shape
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        for label, prof_name, plain_name, BN, BK, n_mult in DESIGNS:
            if n % n_mult or m % 256:
                continue
            ps, plain = spec(prof_name, BK, n_mult), spec(plain_name, BK, n_mult)

            buf = torch.zeros(clusters * RATE_SLICE_U64, dtype=torch.int64,
                              device="cuda")
            runtime.launch_bf16_prof(ps, a, b, out, buf)
            torch.cuda.synchronize()
            err = (out.float() - torch.matmul(a, b).float()).abs().max().item()
            assert err < 1.0, f"{label} at {shape}: wrong result ({err})"

            comp, pause = phases(buf.cpu().numpy(), clusters)
            duty = comp / (comp + pause)

            ms = do_bench(lambda: runtime.launch_bf16(plain, a, b, out),
                          warmup=500, rep=500)
            meas = 2 * m * n * k / (ms * 1e-3) / 1e12

            # measured = k-loop rate x duty, so the k-loop rate follows without
            # needing to know the clock at all
            kloop = meas / duty
            f = kloop * 1e12 / A_FLOP / clusters       # issues/s per cluster
            print(f"{label:<18} {shape:>6} {meas:>8.0f}  {duty:>6.1%} "
                  f"{kloop:>11.0f}  {f / 1e6:>9.2f} M/s {1e9 / f:>9.0f} ns")
            del buf
        del a, b, out
        torch.cuda.empty_cache()

    print("\nk-loop rate is TFLOP/s while inside the k loop; f is MMA issues "
          "per second\nper cluster. A x f x clusters reproduces the k-loop "
          "rate by construction.")


if __name__ == "__main__":
    main()
