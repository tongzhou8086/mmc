# MMC

MMC is a small collection of GEMM-family kernels. The initial release provides autotuned MXFP8 GEMM for
NVIDIA B200 GPUs.

```python
import torch
import mmc

A = torch.randn((8192, 1024), dtype=torch.bfloat16, device="cuda")
B = torch.randn((1024, 8192), dtype=torch.bfloat16, device="cuda")

Aq, Bq, SFA, SFB = mmc.quantize_to_mxfp8(A, B)
C = mmc.matmul_mxfp8(Aq, Bq, SFA, SFB)
```

## Layout contract

The public inputs use the conventional row-major layouts `A[M,K]` and
`B[K,N]`. Quantization returns:

- `Aq[M,K]`, row-major E4M3;
- `Bq[N,K]`, transposed row-major E4M3 for the kernels' ABt convention;
- `SFA[M/128,K/128,32,16]`, packed E8M0;
- `SFB[N/128,K/128,32,16]`, packed E8M0.

The result is row-major BF16 `C[M,N]`.

Current shape constraints are `M % 256 == 0`, `N % 256 == 0`, and
`K % 128 == 0`.

## Autotuning

The first `matmul_mxfp8` call for a shape benchmarks all compatible bundled
kernels with `triton.testing.do_bench`. Its warmup and measurement windows
default to 200 ms and 300 ms:

```python
C = mmc.matmul_mxfp8(
    Aq, Bq, SFA, SFB, warmup_ms=200, rep_ms=300
)
```

The winner is stored in:

```text
~/.cache/mmc/autotune.json
```

Set `MMC_CACHE_DIR` to override the cache directory. The cache key includes the
shape, GPU properties, CUDA driver version, and kernel-set version.

## Binary distribution

SM100a cubins are included in the Python package. Runtime execution requires:

- an NVIDIA B200 GPU;
- an installed CUDA driver;
- PyTorch, NumPy, and cuda-python.

NVCC and the CUDA toolkit are not required. The corresponding `.cu` sources
are included for review but are never compiled at runtime.

Install locally with:

```bash
pip install -e .
```
