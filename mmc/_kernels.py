from dataclasses import dataclass


BM = 128
BN = 256
BN_LOCAL = 128
STORE_N = 64


@dataclass(frozen=True)
class KernelSpec:
    name: str
    bk: int
    threads: int = 0
    shared_bytes: int = 0
    backend: str = "cuda"
    # BN=384 kernels take a second B descriptor with this many rows per box.
    # Both of their MMAs are cta_group::2 and split N across the CTA pair, so
    # each CTA holds 128 B rows for the N=256 MMA plus this many for the N=128
    # MMA, from a different global N offset.
    bn_local_tail: int = 0


# These are the retained candidates from mxfp8-gemm-study/autotune.
KERNELS = (
    KernelSpec("single-ns5-store3-bk128-load256", 128, 384, 224768),
    KernelSpec("single-ns6-store1-bk128-load256", 128, 384, 226304),
    KernelSpec("single-ns3-store1-bk256-load256", 256, 384, 226304),
    # `-earlysc` variants move the accumulator-buffer wait past the scale
    # copies, so the MMA warp starts copying the next tile's scales into TMEM
    # before the epilogue has drained the accumulator. Only the single-TMEM-
    # buffered kernels keep the scales in dedicated TMEM columns, so only they
    # can do this.
    KernelSpec("single-ns5-store3-bk128-load256-earlysc", 128, 384, 224768),
    KernelSpec("single-ns6-store1-bk128-load256-earlysc", 128, 384, 226304),
    KernelSpec("single-ns3-store1-bk256-load256-earlysc", 256, 384, 226304),
    # BN=384: one N=256 MMA plus one N=128 MMA per K step, for higher arithmetic
    # intensity per output tile. The wider B tile and the third scale atom per
    # slot cost shared memory, so this runs 4 pipeline stages instead of 5.
    KernelSpec(
        "single-ns4-store3-bk128-bn384-earlysc", 128, 384, 224768,
        bn_local_tail=64,
    ),
    # `-splitacc2` uses one accumulator-free barrier per MMA N group, with the
    # epilogue's register loads aligned to those groups (128 columns then 256),
    # so each group's accumulator columns are released before any of that
    # group's SMEM staging or TMA stores and the next output tile's MMA for that
    # group overlaps all of it.
    KernelSpec(
        "single-ns4-store3-bk128-bn384-splitacc2", 128, 384, 224768,
        bn_local_tail=64,
    ),
    KernelSpec("double-ns5-store3-bk128", 128, 256, 224768),
    KernelSpec("double-ns6-store1-bk128", 128, 256, 226304),
    KernelSpec("double-ns3-store1-bk256", 256, 256, 226304),
    KernelSpec("double-ns6-store1-bk128-load128", 128, 256, 226304),
    KernelSpec("double-ns3-store1-bk256-load128", 256, 256, 226304),
    KernelSpec("tk-1024", 128, backend="tk"),
    KernelSpec("tk-2048", 128, backend="tk"),
    KernelSpec("tk-4096", 128, backend="tk"),
    KernelSpec("tk-8192", 128, backend="tk"),
    KernelSpec("tk-16384", 128, backend="tk"),
    # The tcgen05_kernels F4 kernel (fc2_mxfp8_add_matrix_fwd): MXFP8 ABt with an
    # optional BF16 residual add. MMC has no residual input, so this runs the
    # HAS_C=false specialization, which elides the residual load and add.
    KernelSpec("tk-fc2add", 128, backend="tk"),
)

KERNEL_BY_NAME = {kernel.name: kernel for kernel in KERNELS}
KERNEL_SET_VERSION = "sm100a-mxfp8-x32-v6"
