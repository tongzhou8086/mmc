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

## BF16

`matmul_bf16` computes `C[M,N] = A[M,K] @ B[K,N]` for BF16 operands. `B` is
conventional row-major `[K,N]`, unlike the MXFP8 path, which takes the transposed
RHS for its ABt kernels:

```python
A = torch.randn((8192, 1024), dtype=torch.bfloat16, device="cuda")
B = torch.randn((1024, 8192), dtype=torch.bfloat16, device="cuda")

C = mmc.matmul_bf16(A, B)
```

`matmul_bf16_out` reuses an existing output allocation, and both take the same
`retune`, `print_tuning`, and `tuning_window` options as their MXFP8
counterparts. To inspect BF16 tuning results from the command line:

```bash
python benchmarks/tune_matmul_bf16.py 4096 8192x8192x512
```

The BF16 candidate set holds custom CUDA kernels plus a `torch.matmul` passthrough.
Each CUDA kernel declares the shape alignment it needs — all of them want
`M % 256 == 0` and `K % 64 == 0`, and `N % 256 == 0` or `N % 512 == 0` depending on
its tile width. Shapes that do not meet a kernel's requirement are tuned over the
remaining candidates, so `torch.matmul` always applies and `matmul_bf16` accepts
any shape.

Autotuning is per data type: each kernel set has its own version and the cache key
carries it, so MXFP8 and BF16 winners for the same shape never collide and bumping
one set does not invalidate the other.

## Autotuning

The first `matmul_mxfp8` call for a shape benchmarks all compatible bundled
kernels with `triton.testing.do_bench`. MMC runs three independently shuffled
benchmark passes and selects the kernel with the lowest median time.
Pass `retune=True` to ignore a cached winner and repeat autotuning:

```python
C = mmc.matmul_mxfp8(Aq, Bq, SFA, SFB, retune=True)
```

Add `print_tuning=True` with `retune=True` to print the measured TFLOP/s for
each compatible bundled kernel:

```python
C = mmc.matmul_mxfp8(Aq, Bq, SFA, SFB, retune=True, print_tuning=True)
```

To inspect tuning results from the command line:

```bash
python benchmarks/tune_matmul_mxfp8.py 4096 8192x8192x8192
```

To tune over only some of the bundled kernels, pass `tuning_include` a list of
names, or `--include` on either tuning script (repeat it, or comma-separate):

```python
C = mmc.matmul_bf16(A, B, retune=True, print_tuning=True,
                    tuning_include=["bf16-double-ns6-store2-bk64",
                                    "bf16-double-ns3-store2-bk128"])
```

```bash
python benchmarks/tune_matmul_bf16.py 8192 --include bf16-double-ns6-store2-bk64,torch.matmul
```

The winner of a subset is not the winner of the full set, so this mode neither
reads nor writes the autotune cache — it is for comparing a chosen few, not for
selecting what to run.

The winner is stored in:

```text
~/.cache/mmc/autotune.json
```

Set `MMC_CACHE_DIR` to override the cache directory. The cache key includes the
shape, GPU properties, CUDA driver version, and kernel-set version.

## Binary distribution

SM100a binary artifacts are included in the Python package. Runtime execution requires:

- an NVIDIA B200 GPU;
- an installed CUDA driver;
- PyTorch, NumPy, and cuda-python.

NVCC and the CUDA toolkit are not required. The corresponding `.cu` sources and
their `.cuh` header are included for review but are never compiled at runtime.

The CUDA kernel sources share `mmc/kernels/cuda-mxfp8.cuh`, which holds the
SMEM/TMEM tile descriptors, tcgen05 fences and TMEM loads, mbarrier and cluster
primitives, the 128B-swizzled TMA load/store wrappers, and the MXFP8 scale-atom
and MMA helpers. Each kernel `.cu` keeps only its own tile shape, pipeline
constants, MMA schedule and kernel body.

Install locally with:

```bash
pip install -e .
```
