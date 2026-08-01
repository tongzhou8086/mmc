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


# These are the eight retained candidates from mxfp8-gemm-study/autotune.
KERNELS = (
    KernelSpec("single-ns5-store3-bk128-load256", 128, 384, 224768),
    KernelSpec("single-ns6-store1-bk128-load256", 128, 384, 226304),
    KernelSpec("single-ns3-store1-bk256-load256", 256, 384, 226304),
    KernelSpec("double-ns5-store3-bk128", 128, 256, 224768),
    KernelSpec("double-ns6-store1-bk128", 128, 256, 226304),
    KernelSpec("double-ns3-store1-bk256", 256, 256, 226304),
    KernelSpec("double-ns6-store1-bk128-load128", 128, 256, 226304),
    KernelSpec("double-ns3-store1-bk256-load128", 256, 256, 226304),
    KernelSpec("tk-4096", 128, backend="tk"),
)

KERNEL_BY_NAME = {kernel.name: kernel for kernel in KERNELS}
KERNEL_SET_VERSION = "sm100a-mxfp8-x32-v2"
