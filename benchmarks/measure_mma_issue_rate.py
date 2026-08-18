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

# (label, BN, BK, GSM) -> kernel stem; the -mmarate build is the same kernel
# plus two clock reads per output tile
def variants():
    out = [("BN=256 BK=64  GSM=8", "bf16-double-ns6-store2-bk64", 256, 64)]
    for BK, stem in ((64, "bf16-single-ns4-store2-bk64-bn512"),
                     (128, "bf16-single-ns2-store2-bk128-bn512")):
        for gsm in (8, 12, 16):
            name = stem if gsm == 8 else f"{stem}-gsm{gsm}"
            out.append((f"BN=512 BK={BK:<3} GSM={gsm:<2}", name, 512, BK))
    return out


def spec(name, bk, n_multiple):
    return KernelSpec(name, bk, 256, 230400, m_multiple=256,
                      n_multiple=n_multiple)


def phases(host, clusters):
    """-> (compute cycles, pause cycles) per output tile, cluster-averaged.

    Totals, not medians. A median tile is blind to a skewed tail, and the tail
    is exactly what swizzle and cache behaviour move: the slow tiles are the
    ones that change, so a per-tile median reports no effect where the
    benchmark sees several percent.
    """
    comp_tot = pause_tot = tiles = 0
    for c in range(clusters):
        sl = host[c * RATE_SLICE_U64:(c + 1) * RATE_SLICE_U64]
        n = int(sl[0])
        stamps = [(int(sl[2 + 2 * i]), int(sl[3 + 2 * i])) for i in range(n)]
        stamps = [(b, e) for b, e in stamps if b]
        if len(stamps) < 2:
            continue
        span = stamps[-1][1] - stamps[0][0]
        comp = sum(e - b for b, e in stamps)
        comp_tot += comp
        pause_tot += span - comp
        tiles += len(stamps)
    return comp_tot / tiles, pause_tot / tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shapes", nargs="+", type=int)
    args = ap.parse_args()

    runtime = runtime_for(torch.cuda.current_device())
    clusters = (runtime.sm_count - runtime.sm_count % 2) // 2
    print(f"A = {A_FLOP / 1e6:.3f} MFLOP per tcgen05.mma issue, identical everywhere\n")

    rows = []
    for shape in args.shapes:
        m = n = k = shape
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        for label, stem, BN, BK in variants():
            if n % BN or m % 256:
                continue
            ps = spec(stem + "-mmarate", BK, BN)
            plain = spec(stem, BK, BN)
            buf = torch.zeros(clusters * RATE_SLICE_U64, dtype=torch.int64,
                              device="cuda")
            # reach the same steady state do_bench measures: warm L2 and a
            # settled clock. A single cold launch samples a different regime,
            # and the instrumentation then disagrees with the benchmark for
            # reasons that have nothing to do with the model.
            for _ in range(30):
                runtime.launch_bf16_prof(ps, a, b, out, buf)
            torch.cuda.synchronize()
            err = (out.float() - torch.matmul(a, b).float()).abs().max().item()
            assert err < 1.0, f"{label} at {shape}: wrong result ({err})"
            comp, pause = phases(buf.cpu().numpy(), clusters)
            del buf

            ms = do_bench(lambda: runtime.launch_bf16(plain, a, b, out),
                          warmup=500, rep=500)
            meas = 2 * m * n * k / (ms * 1e-3) / 1e12

            # everything below is measured, nothing is fitted per row:
            # FLOPs per output tile is exact, compute and pause are cycle counts
            flops_tile = 2 * (CG * BM) * BN * k
            issues_tile = (k // BK) * (BK // MMA_K) * (BN // N_INST)
            rows.append(dict(shape=shape, label=label, meas=meas,
                             comp=comp, pause=pause, flops=flops_tile,
                             issues=issues_tile))
        del a, b, out
        torch.cuda.empty_cache()

    # one global constant for the whole table: the SM clock. Everything else
    # comes from the instrumentation, so this is a genuine prediction rather
    # than a per-row fit.
    def predict(r, ghz):
        return r["flops"] / (r["comp"] + r["pause"]) * ghz * 1e9 * clusters / 1e12
    best = min(((sum((predict(r, g / 1000) - r["meas"]) ** 2 for r in rows), g / 1000)
                for g in range(800, 2600, 5)))
    ghz = best[1]
    print(f"one global clock fitted over all {len(rows)} rows: {ghz:.3f} GHz\n")

    hdr = (f"{'shape':>6} {'config':<20} {'cyc/issue':>10} {'duty':>7} "
           f"{'predicted':>10} {'measured':>9} {'error':>7}")
    print(hdr); print("-" * len(hdr))
    worst = 0.0
    for r in rows:
        cyc = r["comp"] / r["issues"]
        duty = r["comp"] / (r["comp"] + r["pause"])
        pred = predict(r, ghz)
        e = pred / r["meas"] - 1
        worst = max(worst, abs(e))
        print(f"{r['shape']:>6} {r['label']:<20} {cyc:>10.1f} {duty:>6.1%} "
              f"{pred:>10.0f} {r['meas']:>9.0f} {e:>+6.1%}")
    print(f"\nlargest error {worst:.1%}. cyc/issue is the inverse issue rate "
          f"inside the k loop;\nduty is the share of time not spent waiting on "
          f"the drain between tiles.")


if __name__ == "__main__":
    main()
