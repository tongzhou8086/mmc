import json
from pathlib import Path

import pytest
import torch

import mmc
from mmc import _api
from mmc._kernels import (
    BF16_KERNEL_SET_VERSION,
    BF16_KERNELS,
    MXFP8_KERNEL_SET_VERSION,
)


def test_matmul_bf16(tmp_path, monkeypatch):
    monkeypatch.setenv("MMC_CACHE_DIR", str(tmp_path))
    torch.manual_seed(0)
    a = torch.randn((512, 1024), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((1024, 768), dtype=torch.bfloat16, device="cuda")

    out = mmc.matmul_bf16(a, b)
    out_buffer = torch.empty_like(out)
    returned = mmc.matmul_bf16_out(a, b, out_buffer)
    assert returned is out_buffer

    reference = (a.float() @ b.float()).bfloat16()
    for result in (out, out_buffer):
        assert result.shape == (512, 768)
        assert result.dtype == torch.bfloat16
        error = (result.float() - reference.float()).abs().max()
        assert error / reference.float().abs().max() < 0.02

    cache = json.loads((Path(tmp_path) / "autotune.json").read_text())
    assert len(cache) == 1
    assert next(iter(cache.values())) in {spec.name for spec in BF16_KERNELS}
    assert json.loads(next(iter(cache)))["kernels"] == BF16_KERNEL_SET_VERSION


def test_bf16_and_mxfp8_caches_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setenv("MMC_CACHE_DIR", str(tmp_path))
    torch.manual_seed(0)
    a = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")

    mmc.matmul_bf16(a, b)
    aq, bq, sfa, sfb = mmc.quantize_to_mxfp8(a, b)
    mmc.matmul_mxfp8(aq, bq, sfa, sfb)

    cache = json.loads((Path(tmp_path) / "autotune.json").read_text())
    # Same M, N, K for both data types, so a shared key would have collapsed
    # these into one entry. The per-data-type kernel-set version is what keeps
    # them apart.
    assert MXFP8_KERNEL_SET_VERSION != BF16_KERNEL_SET_VERSION
    assert len(cache) == 2
    versions = sorted(json.loads(key)["kernels"] for key in cache)
    assert versions == sorted([BF16_KERNEL_SET_VERSION, MXFP8_KERNEL_SET_VERSION])


def test_matmul_bf16_rejects_bad_operands():
    a = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError):
        mmc.matmul_bf16(a.float(), b.float())
    with pytest.raises(ValueError):
        mmc.matmul_bf16(a, torch.randn((64, 256), dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError):
        mmc.matmul_bf16(a.t().contiguous().t(), b)
    with pytest.raises(TypeError):
        mmc.matmul_bf16_out(a, b, torch.empty((256, 256), dtype=torch.float32, device="cuda"))


def test_bf16_retune_reports_every_candidate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MMC_CACHE_DIR", str(tmp_path))
    # 512 divides every candidate's N requirement, so all of them are eligible.
    a = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")

    mmc.matmul_bf16(a, b, retune=True, print_tuning=True)
    printed = capsys.readouterr().out
    for spec in BF16_KERNELS:
        assert spec.name in printed, spec.name


def test_bf16_cuda_kernels_match_torch():
    runtime = _api.runtime_for(torch.cuda.current_device())
    torch.manual_seed(0)
    cuda_specs = [spec for spec in BF16_KERNELS if spec.backend == "cuda"]
    assert cuda_specs
    for spec in cuda_specs:
        for m, k, n in [(256, 64, 1024), (512, 1024, 512), (4096, 1024, 8192)]:
            if m % spec.m_multiple or n % spec.n_multiple or k % spec.bk:
                continue
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
            out = torch.full(
                (m, n), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            runtime.launch_bf16(spec, a, b, out)
            torch.cuda.synchronize()

            reference = a.float() @ b.float()
            assert not torch.isnan(out.float()).any(), (spec.name, m, k, n)
            error = (out.float() - reference).abs().max() / reference.abs().max()
            assert error < 0.02, (spec.name, m, k, n, float(error))


def test_bf16_unaligned_shape_falls_back_to_torch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MMC_CACHE_DIR", str(tmp_path))
    # N is not a multiple of 256, so the CUDA kernel is not a candidate.
    a = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((128, 320), dtype=torch.bfloat16, device="cuda")

    result = mmc.matmul_bf16(a, b, retune=True, print_tuning=True)
    printed = capsys.readouterr().out
    assert "torch.matmul" in printed
    for spec in BF16_KERNELS:
        if spec.backend == "cuda":
            assert spec.name not in printed

    cache = json.loads((Path(tmp_path) / "autotune.json").read_text())
    assert next(iter(cache.values())) == "torch.matmul"

    reference = (a.float() @ b.float())
    error = (result.float() - reference).abs().max() / reference.abs().max()
    assert error < 0.02
