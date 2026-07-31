import json
from pathlib import Path

import torch

import mmc
from mmc import _api
from mmc._api import E8M0_BIAS, _quantize_rows
from mmc._kernels import KERNELS


def _dequantize(values, scales):
    exponent = scales.float() - E8M0_BIAS
    return values.float() * torch.exp2(exponent).repeat_interleave(32, dim=-1)


def test_quantize_and_matmul(tmp_path, monkeypatch):
    monkeypatch.setenv("MMC_CACHE_DIR", str(tmp_path))
    torch.manual_seed(0)
    a = torch.randn((2048, 1024), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((1024, 2048), dtype=torch.bfloat16, device="cuda")

    aq, bq, sfa, sfb = mmc.quantize_to_mxfp8(a, b)
    out = mmc.matmul_mxfp8(aq, bq, sfa, sfb)
    out_buffer = torch.empty_like(out)
    returned = mmc.matmul_mxfp8_out(aq, bq, sfa, sfb, out_buffer)
    assert returned is out_buffer

    runtime = _api.runtime_for(torch.cuda.current_device())
    cached_launches = len(runtime._launch_cache)
    mmc.matmul_mxfp8_out(aq, bq, sfa, sfb, out_buffer)
    torch.cuda.synchronize()
    assert len(runtime._launch_cache) == cached_launches

    _, sfa_unpacked = _quantize_rows(a)
    _, sfb_unpacked = _quantize_rows(b.t().contiguous())
    reference = (
        _dequantize(aq, sfa_unpacked)
        @ _dequantize(bq, sfb_unpacked).t()
    ).bfloat16()
    for result in (out, out_buffer):
        error = (result.float() - reference.float()).abs().max()
        assert error / reference.float().abs().max() < 0.05

    cache_path = Path(tmp_path) / "autotune.json"
    cache = json.loads(cache_path.read_text())
    assert len(cache) == 1

    previous = cache_path.stat().st_mtime_ns
    mmc.matmul_mxfp8(aq, bq, sfa, sfb)
    torch.cuda.synchronize()
    assert cache_path.stat().st_mtime_ns == previous


def test_retune_bypasses_cached_winner(tmp_path, monkeypatch):
    monkeypatch.setenv("MMC_CACHE_DIR", str(tmp_path))
    a = torch.randn((2048, 1024), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((1024, 2048), dtype=torch.bfloat16, device="cuda")
    aq, bq, sfa, sfb = mmc.quantize_to_mxfp8(a, b)

    m, k = aq.shape
    n = bq.shape[0]
    runtime = _api.runtime_for(torch.cuda.current_device())
    key = _api._cache_key(runtime, m, n, k)
    cache_path = Path(tmp_path) / "autotune.json"
    cache_path.write_text(json.dumps({key: KERNELS[0].name}))

    benchmark_calls = 0

    def fake_benchmark(run, warmup_ms, rep_ms):
        nonlocal benchmark_calls
        benchmark_calls += 1
        return 1.0

    monkeypatch.setattr(_api, "_benchmark", fake_benchmark)

    mmc.matmul_mxfp8(aq, bq, sfa, sfb)
    assert benchmark_calls == 0

    mmc.matmul_mxfp8(aq, bq, sfa, sfb, retune=True)
    torch.cuda.synchronize()
    compatible = [spec for spec in KERNELS if k % spec.bk == 0]
    assert benchmark_calls == 3 * len(compatible)
