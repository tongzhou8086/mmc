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

If `B` is already stored as row-major `[N,K]`, pass `b_transposed=True`:

```python
Aq, Bq, SFA, SFB = mmc.quantize_to_mxfp8(A, B_t, b_transposed=True)
```

Use `matmul_mxfp8_out` to reuse an existing output allocation:

```python
C = torch.empty((8192, 8192), dtype=torch.bfloat16, device="cuda")
mmc.matmul_mxfp8_out(Aq, Bq, SFA, SFB, C)
```

## Layout contract

The public quantizer input uses row-major `A[M,K]`. By default, it expects
row-major `B[K,N]`; pass `b_transposed=True` when `B` is already row-major
`[N,K]`. Quantization returns:

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
are internally fixed at 200 ms and 300 ms. MMC runs three independently
shuffled benchmark passes and selects the kernel with the lowest median time.
Pass `retune=True` to ignore a cached winner and repeat autotuning:

```python
C = mmc.matmul_mxfp8(Aq, Bq, SFA, SFB, retune=True)
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
