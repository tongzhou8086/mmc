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


def tune_shape(m, n, k, b_transposed=False):
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    if b_transposed:
        b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    else:
        b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
    aq, bq, sfa, sfb = mmc.quantize_to_mxfp8(
        a, b, b_transposed=b_transposed
    )

    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    mmc.matmul_mxfp8_out(
        aq,
        bq,
        sfa,
        sfb,
        out,
        retune=True,
        print_tuning=True,
    )
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Print MMC MXFP8 autotuning results for one or more MxNxK shapes."
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
        help="allocate raw B as row-major [N,K] and pass b_transposed=True",
    )
    args = parser.parse_args()

    for index, (m, n, k) in enumerate(args.shapes):
        if index:
            print()
        tune_shape(m, n, k, b_transposed=args.b_transposed)


if __name__ == "__main__":
    main()
