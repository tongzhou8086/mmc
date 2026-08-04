import json
import os
import random
from pathlib import Path
from statistics import median

import torch
from triton.testing import do_bench

from ._kernels import (
    BF16_KERNEL_BY_NAME,
    BF16_KERNEL_SET_VERSION,
    BF16_KERNELS,
    BM,
    BN,
    MXFP8_KERNEL_BY_NAME,
    MXFP8_KERNEL_SET_VERSION,
    MXFP8_KERNELS,
)
from ._runtime import runtime_for


MXFP8_MAX = 448.0
E8M0_BIAS = 127.0
TUNING_WINDOWS = {
    1: (500, 500),
    2: (1000, 1000),
    3: (1000, 2000),
}


def _quantize_rows(tensor):
    rows, cols = tensor.shape
    values = tensor.float()
    amax = values.abs().view(rows, cols // 32, 32).amax(dim=-1)
    exponent = torch.ceil(torch.log2(amax / MXFP8_MAX)).clamp(-127, 127)
    exponent = torch.where(amax == 0, 0, exponent)
    scales = (exponent + E8M0_BIAS).to(torch.uint8)
    scale_values = torch.exp2(exponent).repeat_interleave(32, dim=-1)
    quantized = (values / scale_values).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scales.contiguous()


def _pack_scales(scales):
    rows, k_blocks = scales.shape
    packed = scales.reshape(rows // 128, 128, k_blocks // 4, 4)
    packed = packed.transpose(1, 2).reshape(
        rows // 128, k_blocks // 4, 4, 32, 4
    )
    packed = packed.transpose(-2, -3).reshape(
        rows // 128, k_blocks // 4, 32, 16
    )
    return packed.contiguous()


def quantize_to_mxfp8(a, b, b_transposed=False):
    """Quantize row-major A[M,K] and B for MMC.

    By default, B is conventional row-major [K,N]. Pass b_transposed=True
    when B is already row-major [N,K].

    Returns Aq[M,K], Bq[N,K], packed SFA, and packed SFB.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same CUDA device")
    b_k = b.shape[1] if b_transposed else b.shape[0]
    if a.shape[1] != b_k:
        raise ValueError(f"incompatible shapes: A{tuple(a.shape)}, B{tuple(b.shape)}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous row-major tensors")

    m, k = a.shape
    n = b.shape[0] if b_transposed else b.shape[1]
    if m % 256 or n % 256 or k % 128:
        raise ValueError("MMC currently requires M%256 == N%256 == 0 and K%128 == 0")

    aq, sfa = _quantize_rows(a)
    if b_transposed:
        bq, sfb = _quantize_rows(b)
    else:
        bq, sfb = _quantize_rows(b.t().contiguous())
    return aq, bq, _pack_scales(sfa), _pack_scales(sfb)


def _validate_quantized(a, b, sfa, sfb):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("quantized A and B must be rank-2 tensors")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("quantized A and B must have dtype torch.float8_e4m3fn")
    if a.device.type != "cuda" or any(x.device != a.device for x in (b, sfa, sfb)):
        raise ValueError("A, B, SFA, and SFB must be on the same CUDA device")
    if not all(x.is_contiguous() for x in (a, b, sfa, sfb)):
        raise ValueError("A, B, SFA, and SFB must be contiguous")

    m, k = a.shape
    n, b_k = b.shape
    if b_k != k:
        raise ValueError(f"incompatible quantized shapes: A{tuple(a.shape)}, B{tuple(b.shape)}")
    if m % 256 or n % 256 or k % 128:
        raise ValueError("MMC currently requires M%256 == N%256 == 0 and K%128 == 0")
    expected_a = (m // 128, k // 128, 32, 16)
    expected_b = (n // 128, k // 128, 32, 16)
    if sfa.dtype != torch.uint8 or tuple(sfa.shape) != expected_a:
        raise ValueError(f"SFA must be uint8 with shape {expected_a}")
    if sfb.dtype != torch.uint8 or tuple(sfb.shape) != expected_b:
        raise ValueError(f"SFB must be uint8 with shape {expected_b}")
    return m, n, k


def _validate_bf16(a, b):
    """Check BF16 operands and return (M, N, K) for A[M,K] @ B[K,N]."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("A and B must have dtype torch.bfloat16")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same CUDA device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous")

    m, k = a.shape
    b_k, n = b.shape
    if b_k != k:
        raise ValueError(f"incompatible shapes: A{tuple(a.shape)}, B{tuple(b.shape)}")
    return m, n, k


def _cache_path():
    root = Path(os.environ.get("MMC_CACHE_DIR", Path.home() / ".cache" / "mmc"))
    return root / "autotune.json"


def _read_cache():
    try:
        return json.loads(_cache_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_cache(cache):
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cache_key(runtime, kernel_set_version, m, n, k):
    properties = torch.cuda.get_device_properties(runtime.device_index)
    machine = {
        "name": properties.name,
        "sms": properties.multi_processor_count,
        "memory": properties.total_memory,
        "driver": runtime.driver_version,
    }
    return json.dumps(
        {
            "kernels": kernel_set_version,
            "machine": machine,
            "shape": [m, n, k],
        },
        sort_keys=True,
    )


def _benchmark(run, warmup_ms, rep_ms):
    return do_bench(
        run,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode="median",
    )


def _tuning_window_ms(tuning_window):
    try:
        return TUNING_WINDOWS[tuning_window]
    except (KeyError, TypeError) as error:
        raise ValueError("tuning_window must be 1, 2, or 3") from error


def _print_tuning_results(median_timings, m, n, k):
    rows = sorted(
        (
            (name, 2 * m * n * k / timing_ms / 1e9)
            for name, timing_ms in median_timings.items()
        ),
        key=lambda row: row[1],
        reverse=True,
    )
    print(f"MMC tuning results for M={m}, N={n}, K={k}")
    print(f"{'config':<40} {'TFLOP/s':>10}")
    print(f"{'-' * 40} {'-' * 10}")
    for name, tflops in rows:
        print(f"{name:<40} {tflops:10.2f}")


def _select_kernel(
    runtime,
    kernels,
    kernel_by_name,
    dtype,
    kernel_set_version,
    make_run,
    m,
    n,
    k,
    benchmark_runs=3,
    retune=False,
    print_tuning=False,
    tuning_window=1,
):
    """Return the fastest compatible spec for a shape, caching the winner.

    Data-type agnostic: the caller supplies its own candidate set and a
    make_run(spec) factory returning a no-argument callable that launches that
    spec on the caller's operands. Each data type has its own kernel-set version
    and the cache key carries it, so the MXFP8 and BF16 winners for one shape
    never collide. dtype is used only in error messages.
    """
    if benchmark_runs < 1:
        raise ValueError("benchmark_runs must be positive")
    warmup_ms, rep_ms = _tuning_window_ms(tuning_window)

    cache = _read_cache()
    key = _cache_key(runtime, kernel_set_version, m, n, k)
    cached = cache.get(key)
    if not retune and cached in kernel_by_name:
        spec = kernel_by_name[cached]
        if (
            k % spec.bk == 0
            and m % spec.m_multiple == 0
            and n % spec.n_multiple == 0
        ):
            return spec

    candidates = [
        spec
        for spec in kernels
        if k % spec.bk == 0
        and m % spec.m_multiple == 0
        and n % spec.n_multiple == 0
    ]
    if not candidates:
        raise ValueError(f"no bundled {dtype} kernel is compatible with K={k}")
    timings = {spec.name: [] for spec in candidates}
    for _ in range(benchmark_runs):
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        for spec in shuffled:
            timing = _benchmark(
                make_run(spec), warmup_ms=warmup_ms, rep_ms=rep_ms
            )
            timings[spec.name].append(timing)

    median_timings = {
        name: median(samples) for name, samples in timings.items()
    }
    if print_tuning:
        _print_tuning_results(median_timings, m, n, k)
    winner = min(candidates, key=lambda spec: median_timings[spec.name])
    cache[key] = winner.name
    _write_cache(cache)
    return winner


def _select_mxfp8_kernel(runtime, a, b, sfa, sfb, out, m, n, k, **kwargs):
    def make_run(spec):
        return lambda: runtime.launch_mxfp8(spec, a, b, sfa, sfb, out)

    return _select_kernel(
        runtime,
        MXFP8_KERNELS,
        MXFP8_KERNEL_BY_NAME,
        "mxfp8",
        MXFP8_KERNEL_SET_VERSION,
        make_run,
        m,
        n,
        k,
        **kwargs,
    )


def _select_bf16_kernel(runtime, a, b, out, m, n, k, **kwargs):
    def make_run(spec):
        return lambda: runtime.launch_bf16(spec, a, b, out)

    return _select_kernel(
        runtime,
        BF16_KERNELS,
        BF16_KERNEL_BY_NAME,
        "bf16",
        BF16_KERNEL_SET_VERSION,
        make_run,
        m,
        n,
        k,
        **kwargs,
    )


def matmul_mxfp8_out(
    a,
    b,
    sfa,
    sfb,
    out,
    retune=False,
    print_tuning=False,
    tuning_window=1,
):
    """Compute into a reusable BF16 output tensor.

    The first call for a shape benchmarks valid bundled kernels and stores the
    winner in ~/.cache/mmc/autotune.json. Later calls launch the cached winner
    unless retune is True. Autotuning uses Triton's do_bench internally. Pass
    retune=True and print_tuning=True to print per-kernel tuning TFLOP/s.
    tuning_window selects 500/500, 1000/1000, or 1000/2000 ms windows.
    """
    m, n, k = _validate_quantized(a, b, sfa, sfb)
    if out.device != a.device:
        raise ValueError("out must be on the same CUDA device as the inputs")
    if out.dtype != torch.bfloat16:
        raise TypeError("out must have dtype torch.bfloat16")
    if tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(f"out must be contiguous with shape {(m, n)}")

    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        runtime = runtime_for(device_index)
        spec = _select_mxfp8_kernel(
            runtime,
            a,
            b,
            sfa,
            sfb,
            out,
            m,
            n,
            k,
            retune=retune,
            print_tuning=print_tuning,
            tuning_window=tuning_window,
        )
        runtime.launch_mxfp8(spec, a, b, sfa, sfb, out)
        return out


def matmul_mxfp8(
    a,
    b,
    sfa,
    sfb,
    retune=False,
    print_tuning=False,
    tuning_window=1,
):
    """Allocate and return C[M,N] for quantized A[M,K] and B[N,K]."""
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    return matmul_mxfp8_out(
        a,
        b,
        sfa,
        sfb,
        out,
        retune=retune,
        print_tuning=print_tuning,
        tuning_window=tuning_window,
    )


def matmul_bf16_out(
    a,
    b,
    out,
    retune=False,
    print_tuning=False,
    tuning_window=1,
):
    """Compute out[M,N] = A[M,K] @ B[K,N] for BF16 operands.

    B is conventional row-major [K,N], unlike the MXFP8 path, which takes the
    transposed RHS for its ABt kernels. Selection and caching work exactly as
    they do for MXFP8, against the BF16 candidate set; the cache key records the
    data type so the two never collide. The only bundled BF16 candidate so far is
    a torch.matmul passthrough.
    """
    m, n, k = _validate_bf16(a, b)
    if out.device != a.device:
        raise ValueError("out must be on the same CUDA device as the inputs")
    if out.dtype != torch.bfloat16:
        raise TypeError("out must have dtype torch.bfloat16")
    if tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(f"out must be contiguous with shape {(m, n)}")

    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        runtime = runtime_for(device_index)
        spec = _select_bf16_kernel(
            runtime,
            a,
            b,
            out,
            m,
            n,
            k,
            retune=retune,
            print_tuning=print_tuning,
            tuning_window=tuning_window,
        )
        runtime.launch_bf16(spec, a, b, out)
        return out


def matmul_bf16(
    a,
    b,
    retune=False,
    print_tuning=False,
    tuning_window=1,
):
    """Allocate and return C[M,N] for BF16 A[M,K] and B[K,N]."""
    m, n = a.shape[0], b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    return matmul_bf16_out(
        a,
        b,
        out,
        retune=retune,
        print_tuning=print_tuning,
        tuning_window=tuning_window,
    )
