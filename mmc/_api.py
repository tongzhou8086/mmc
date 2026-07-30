import json
import os
from pathlib import Path

import torch
from triton.testing import do_bench

from ._kernels import (
    BM,
    BN,
    KERNEL_BY_NAME,
    KERNEL_SET_VERSION,
    KERNELS,
)
from ._runtime import runtime_for


MXFP8_MAX = 448.0
E8M0_BIAS = 127.0


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


def quantize_to_mxfp8(a, b):
    """Quantize conventional row-major A[M,K] and B[K,N] for MMC.

    Returns Aq[M,K], Bq[N,K], packed SFA, and packed SFB. Bq is transposed
    because Blackwell's ABt MXFP8 kernels consume B as row-major [N,K].
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same CUDA device")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: A{tuple(a.shape)}, B{tuple(b.shape)}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous row-major tensors")

    m, k = a.shape
    n = b.shape[1]
    if m % 256 or n % 256 or k % 128:
        raise ValueError("MMC currently requires M%256 == N%256 == 0 and K%128 == 0")

    aq, sfa = _quantize_rows(a)
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


def _cache_key(runtime, m, n, k):
    properties = torch.cuda.get_device_properties(runtime.device_index)
    machine = {
        "name": properties.name,
        "sms": properties.multi_processor_count,
        "memory": properties.total_memory,
        "driver": runtime.driver_version,
    }
    return json.dumps(
        {
            "kernels": KERNEL_SET_VERSION,
            "machine": machine,
            "shape": [m, n, k],
        },
        sort_keys=True,
    )


def _benchmark(run, warmup_ms, rep_ms):
    return do_bench(run, warmup=warmup_ms, rep=rep_ms)


def _select_kernel(
    runtime, a, b, sfa, sfb, out, m, n, k, warmup_ms, rep_ms
):
    cache = _read_cache()
    key = _cache_key(runtime, m, n, k)
    cached = cache.get(key)
    if cached in KERNEL_BY_NAME and k % KERNEL_BY_NAME[cached].bk == 0:
        return KERNEL_BY_NAME[cached]

    candidates = [spec for spec in KERNELS if k % spec.bk == 0]
    timings = {}
    for spec in candidates:
        run = runtime.prepare(spec, a, b, sfa, sfb, out)
        timings[spec.name] = _benchmark(run, warmup_ms, rep_ms)
    winner = min(candidates, key=lambda spec: timings[spec.name])
    cache[key] = winner.name
    _write_cache(cache)
    return winner


def matmul_mxfp8(a, b, sfa, sfb, *, warmup_ms=200, rep_ms=300):
    """Compute C[M,N] from MMC's quantized A[M,K] and B[N,K].

    The first call for a shape benchmarks valid bundled kernels and stores the
    winner in ~/.cache/mmc/autotune.json. Later calls launch the cached winner.
    Autotuning uses Triton's do_bench with 200 ms warmup and 300 ms measurement
    by default.
    """
    if warmup_ms <= 0 or rep_ms <= 0:
        raise ValueError("warmup_ms and rep_ms must be positive")

    m, n, k = _validate_quantized(a, b, sfa, sfb)
    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        runtime = runtime_for(device_index)
        spec = _select_kernel(
            runtime, a, b, sfa, sfb, out, m, n, k, warmup_ms, rep_ms
        )
        runtime.prepare(spec, a, b, sfa, sfb, out)()
        return out
