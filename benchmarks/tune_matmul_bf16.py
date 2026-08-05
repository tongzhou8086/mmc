import argparse

import torch

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


def tune_shape(m, n, k, tuning_window=1):
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")

    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    mmc.matmul_bf16_out(
        a,
        b,
        out,
        retune=True,
        print_tuning=True,
        tuning_window=tuning_window,
    )
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Print MMC BF16 autotuning results for one or more MxNxK shapes."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        type=parse_shape,
        metavar="SHAPE",
        help="N for a square GEMM, or MxNxK",
    )
    parser.add_argument(
        "--tuning-window",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="autotuning window: 1=500/500 ms, 2=1000/1000 ms, 3=1000/2000 ms",
    )
    args = parser.parse_args()

    for index, (m, n, k) in enumerate(args.shapes):
        if index:
            print()
        tune_shape(m, n, k, tuning_window=args.tuning_window)


if __name__ == "__main__":
    main()
