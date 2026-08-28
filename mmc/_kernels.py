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
    # Shape alignment the kernel requires, beyond K % bk == 0. Candidates that
    # do not divide a shape are filtered out of autotuning for it.
    m_multiple: int = 1
    n_multiple: int = 1
    # BN=384 kernels take a second B descriptor with this many rows per box.
    # Both of their MMAs are cta_group::2 and split N across the CTA pair, so
    # each CTA holds 128 B rows for the N=256 MMA plus this many for the N=128
    # MMA, from a different global N offset.
    bn_local_tail: int = 0
    # Cluster-launch-control kernels are launched with one cluster per output
    # tile rather than one per SM pair: the hardware, not a static stride,
    # decides which cluster runs which tile.
    clc: bool = False


# These are the retained MXFP8 candidates from mxfp8-gemm-study/autotune.
MXFP8_KERNELS = (
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
)

# BF16 candidates. torch.matmul is the always-eligible fallback (bk=1 and no
# alignment requirement), so every shape has at least one candidate.
BF16_KERNELS = (
    # C[M,N] = A[M,K] @ B[K,N], 2-CTA cluster MMA, double-buffered TMEM, chunked
    # TMA-store epilogue. Needs M % 256 == 0, N % 256 == 0 and K % 64 == 0.
    KernelSpec(
        "bf16-double-ns6-store2-bk64", 64, 256, 230400,
        m_multiple=256, n_multiple=256,
    ),
    # BN=512 with a single TMEM accumulator: the epilogue drain is synchronized
    # before the accumulator is reused, which fits twice the output columns in
    # the same TMEM budget. Needs M % 256 == 0, N % 512 == 0 and K % 64 == 0.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512", 64, 256, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # Same kernel with a two-level epilogue drain: 128 accumulator columns come
    # out of TMEM per outer iteration (4 trips instead of 8), stored 64 columns
    # at a time by the inner loop.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load128", 64, 256, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # Same two-level drain widened to 256 columns per outer iteration, which
    # takes the whole BN=512 tile out of TMEM in two trips. 8 epilogue warps
    # split the columns into two groups, so a lane still holds 128 floats;
    # launch width grows to 384 threads.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8", 64, 384, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # Same 8-warp/256-column drain with one accumulator-free barrier per MMA
    # panel: the epilogue releases a panel as soon as it is in registers, so the
    # next tile's first-panel MMAs overlap the second panel's drain and stores.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc", 64, 384, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # Same split-panel barriers, but at k == 0 the MMA warp runs panel 0 ahead by
    # LEAD_K=2 k-tiles instead of stalling on panel 1's free barrier; panel 1
    # replays those k-tiles from the same pinned SMEM slots.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc-lead2",
        64, 384, 230400, m_multiple=256, n_multiple=512,
    ),
    # Lead depth 3. With NS=4 this pins three of the four SMEM slots during the
    # k == 0 block, leaving the TMA warp one slot of prefetch.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc-lead3",
        64, 384, 230400, m_multiple=256, n_multiple=512,
    ),
    # Split data-ready as well as split accumulator-free: panel 0's data-ready
    # fires right after its last-k MMAs, before panel 1's are issued, so the
    # epilogue drains panel 0 while panel 1 is still accumulating.
    # Fourth design: BN=512's arithmetic intensity with BN=256's ring depth.
    # A slot holds A plus one accumulator panel's B, so it is 32 KB and six fit
    # where four 48 KB slots did. Each slot is filled twice per visit - round 1
    # writes A and panel 0's B, round 2 overwrites only B for panel 1, reusing
    # the resident A - so A is fetched once per two panels.
    KernelSpec(
        "bf16-double-ns6-store2-bk64-bn512-2round", 64, 384, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # BK=128 form of the same idea: the slot doubles to 64 KB so the ring is
    # three deep rather than six, still deeper than design 3's two at BK=128.
    KernelSpec(
        "bf16-double-ns3-store2-bk128-bn512-2round", 128, 384, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # Fifth design: design four plus cluster launch control. Same pipeline,
    # different work distribution - the grid is one cluster per output tile and
    # a cluster takes its next tile by cancelling one that has not launched,
    # which removes the ragged last wave of the static persistent partition.
    # GSM 8 / 12 / 16 as usual; the swizzle still orders the grid, CLC only
    # decides who runs what.
    KernelSpec(
        "bf16-double-ns6-store2-bk64-bn512-2round-clc", 64, 384, 230400,
        m_multiple=256, n_multiple=512, clc=True,
    ),
    KernelSpec(
        "bf16-double-ns6-store2-bk64-bn512-2round-clc-gsm12", 64, 384, 230400,
        m_multiple=256, n_multiple=512, clc=True,
    ),
    KernelSpec(
        "bf16-double-ns6-store2-bk64-bn512-2round-clc-gsm16", 64, 384, 230400,
        m_multiple=256, n_multiple=512, clc=True,
    ),
    KernelSpec(
        "bf16-double-ns3-store2-bk128-bn512-2round-clc", 128, 384, 230400,
        m_multiple=256, n_multiple=512, clc=True,
    ),
    KernelSpec(
        "bf16-double-ns3-store2-bk128-bn512-2round-clc-gsm12", 128, 384, 230400,
        m_multiple=256, n_multiple=512, clc=True,
    ),
    KernelSpec(
        "bf16-double-ns3-store2-bk128-bn512-2round-clc-gsm16", 128, 384, 230400,
        m_multiple=256, n_multiple=512, clc=True,
    ),
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc-splitdr",
        64, 384, 230400, m_multiple=256, n_multiple=512,
    ),
    # The same split data-ready applied to the two run-ahead variants.
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc-lead2-splitdr",
        64, 384, 230400, m_multiple=256, n_multiple=512,
    ),
    KernelSpec(
        "bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc-lead3-splitdr",
        64, 384, 230400, m_multiple=256, n_multiple=512,
    ),
    # BK=128 with a 3-deep ring: same 192 KB of staging as the BK=64 kernel's
    # 6 slots, but half as many k iterations and half the run-ahead depth.
    KernelSpec(
        "bf16-double-ns3-store2-bk128", 128, 256, 230400,
        m_multiple=256, n_multiple=256,
    ),
    # BK=128 with a 2-deep ring: same 192 KB of staging as the BK=64 kernels'
    # 4 slots, but half as many k iterations and half the run-ahead depth.
    KernelSpec(
        "bf16-single-ns2-store2-bk128-bn512", 128, 256, 230400,
        m_multiple=256, n_multiple=512,
    ),
    KernelSpec(
        "bf16-single-ns2-store2-bk128-bn512-load128", 128, 256, 230400,
        m_multiple=256, n_multiple=512,
    ),
    KernelSpec(
        "bf16-single-ns2-store2-bk128-bn512-load256-w8", 128, 384, 230400,
        m_multiple=256, n_multiple=512,
    ),
    # BK=128 on the split-accumulator-barrier kernel: each MMA the freed panel
    # starts on covers twice the K, so there is twice as much accumulation to
    # overlap with the other panel's drain.
    KernelSpec(
        "bf16-single-ns2-store2-bk128-bn512-load256-w8-splitacc",
        128, 384, 230400, m_multiple=256, n_multiple=512,
    ),
    KernelSpec("torch.matmul", 1, backend="torch"),
)

MXFP8_KERNEL_BY_NAME = {kernel.name: kernel for kernel in MXFP8_KERNELS}
BF16_KERNEL_BY_NAME = {kernel.name: kernel for kernel in BF16_KERNELS}
# One version per kernel set. The autotune cache key carries it, so bumping one
# set's version invalidates only that set's cached winners, and the two sets'
# winners for the same shape cannot collide. Keep these distinct.
MXFP8_KERNEL_SET_VERSION = "sm100a-mxfp8-x32-v6"
BF16_KERNEL_SET_VERSION = "sm100a-bf16-v12"
assert MXFP8_KERNEL_SET_VERSION != BF16_KERNEL_SET_VERSION
