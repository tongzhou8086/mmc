import argparse

import torch
from triton.testing import do_bench

import mmc


def parse_shape(value):
    parts = value.lower().split("x")
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "shape must be N or MxNxK, for example 1024 or 1024x2048x4096"
        )

    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def benchmark_shape(m, n, k, b_transposed=False):
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b_shape = (n, k) if b_transposed else (k, n)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device="cuda")
    aq, bq, sfa, sfb = mmc.quantize_to_mxfp8(
        a, b, b_transposed=b_transposed
    )

    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    mmc.matmul_mxfp8_out(aq, bq, sfa, sfb, out)
    latency_ms = do_bench(
        lambda: mmc.matmul_mxfp8_out(aq, bq, sfa, sfb, out),
        warmup=1000,
        rep=3000,
        return_mode="median",
    )
    tflops = 2 * m * n * k / latency_ms / 1e9
    return latency_ms, tflops


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MMC MXFP8 GEMM for one or more MxNxK shapes."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        type=parse_shape,
        metavar="SHAPE",
        help="N for a square GEMM, or MxNxK",
    )
    parser.add_argument(
        "--b-transposed",
        action="store_true",
        dest="b_transposed",
        help="allocate raw B as row-major [N,K] and pass b_transposed=True",
    )
    args = parser.parse_args()

    print(f"{'M':>8} {'N':>8} {'K':>8} {'ms':>10} {'TFLOP/s':>12}")
    for m, n, k in args.shapes:
        latency_ms, tflops = benchmark_shape(
            m, n, k, b_transposed=args.b_transposed
        )
        print(f"{m:8d} {n:8d} {k:8d} {latency_ms:10.4f} {tflops:12.2f}")


if __name__ == "__main__":
    main()
