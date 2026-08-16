#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// BF16 GEMM: C[M,N] = A[M,K] @ B[K,N], BN=512 with a single TMEM accumulator,
// 2-CTA cluster MMA and a chunked TMA-store epilogue. The epilogue drain is
// synchronized before the accumulator is reused, which is what lets BN double
// from 256 to 512 within the 512-column TMEM budget. Like the other BF16 kernel
// this is an AB (not ABt) kernel: B is conventional row-major [K,N], and the TMA
// descriptors are rank 2 rather than rank 5.
//
// This variant runs 8 epilogue warps instead of 4 and drains 256 accumulator
// columns per outer loop iteration, so the whole BN=512 tile comes out of TMEM
// in two trips and the second one releases the accumulator. The extra warps
// split the columns into two groups (COL_GROUPS=2), so each lane still holds
// only 256/2 = 128 floats - the same register footprint as the 4-warp,
// 128-column variant. Launch width goes from 256 to 384 threads.
//
// This variant additionally splits the single accumulator-free barrier into one
// per 256-column MMA panel. The epilogue releases a panel as soon as it has
// pulled that panel out of TMEM, so the next output tile's first-panel MMAs can
// start while the epilogue is still draining and storing the second panel. On
// the MMA side only the k == 0 iteration waits, panel by panel, interleaved with
// that panel's MMA issue. Same idea as single-ns4-store3-bk128-bn384-splitacc2.
//
// This variant splits the data-ready signal the same way the accumulator-free
// barrier is already split: one per 256-column MMA panel. The MMA warp signals
// panel 0's data-ready right after panel 0's last-k MMAs, before it issues
// panel 1's, so the epilogue starts draining panel 0 without waiting for panel
// 1 to finish accumulating. The epilogue waits per panel, immediately before
// the outer-loop iteration that drains that panel.
//
// The cost is one branch inside the k loop, on a k == num_k - 1 test that is
// cheap and perfectly predicted.
//
// Adapted from bf16-single-ns4-store2-bk64-bn512-load256-w8-splitacc.cu, which
// is otherwise identical. It does not yet
// use cuda-mxfp8.cuh; the data-type-agnostic helpers will be split out of that
// header and shared with the BF16 kernels separately.
//
// Launch contract, mirrored by Runtime.launch_bf16:
//   grid    = sm_count - sm_count % CTA_GROUP   (persistent, EPILOGUE_OVERLAP)
//   block   = LAUNCH_THREADS = (NUM_WARPS + 4) * 32 = 384
//   dynamic SMEM = NS * SLOT_BYTES + EPI_BYTES + 1024 = 230400
//   args    = A_tmap, B_tmap, C_tmap, C_ptr, M, N, K
// and requires M % (CTA_GROUP * BM) == 0, N % BN == 0, K % BK == 0, so
// M % 256 == 0, N % 512 == 0 and K % 64 == 0.
//
// These sources are shipped for review only. MMC loads the prebuilt cubins at
// runtime and never compiles a kernel.

// ── User-tunable constants (the webui substitutes these) ────────────
constexpr int BM           = 128;
constexpr int BN           = 512;
constexpr int BK           = 64;
constexpr int NS           = 4;       // multi-stage SMEM ring depth
constexpr int GROUP_SIZE_M = 16;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 8;       // total warps per CTA
constexpr int TCGEN05_LD_WIDTH = 8;  // TMEM->reg epilogue load width: 8 or 16 (32-bit elems per lane)
constexpr int EPILOGUE_OVERLAP = 1;  // 1 = persistent 2-CTA cluster + epilogue/K-loop overlap
constexpr int EPILOGUE_SPLIT   = 0;  // 1 = split overlapped int4 writeback into two half-BN passes
constexpr int EPILOGUE_TMA_PIPELINED = 1;  // 1 = chunked staged TMA-store overlap epilogue
constexpr int SINGLE_TMEM_ACCUM = 1;  // 1 = overlap path synchronizes epilogue drain before reusing one TMEM accumulator
constexpr int SEGMENTED_PANELS = 0;  // 1 = BN512 segmented panel schedule (SEG = NS k-tiles per segment)
constexpr int TWO_CTA          = 1;  // 1 = 2-CTA cluster MMA (cta_group::2); 0 = single-CTA

// ── Derived constants (do not edit) ─────────────────────────────────
constexpr int MMA_K     = 16;
constexpr int BF16_BYTES = 2;
constexpr int K_MMAS    = BK / MMA_K;        // 4

constexpr int CTA_GROUP        = TWO_CTA ? 2 : 1;    // 2-CTA cluster vs single-CTA
constexpr int BN_LOCAL         = BN / CTA_GROUP;     // per-CTA N width of B (=BN single-CTA)
constexpr int SWIZZLE_ROW_BYTES = 128;               // one 128B-swizzle atom row
constexpr int STORE_N          = 64;                 // TMA-store chunk width
constexpr int TMA_STORE_STAGES = 2;                  // TMA-store SMEM buffers

// Per-stage SMEM per CTA: A = BM*BK*2 = 16 KB; B = BN_LOCAL*BK*2 = 16 KB.
// Total 32 KB / stage / CTA — half of ch07's 48 KB / stage / CTA.
constexpr int A_SLOT_BYTES = BM       * BK * BF16_BYTES;       // 16 KB
constexpr int B_SLOT_BYTES = BN_LOCAL * BK * BF16_BYTES;       // 16 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;      // 32 KB / slot


// ── Important: dynamic SMEM is used in TWO non-overlapping phases ──
//
// 1.  During the K-loop, the kernel uses `NS * SLOT_BYTES` bytes —
//     NS slots × (A + B) per slot — as the multi-stage ring buffer.
// 2.  During the epilogue, the same dynamic SMEM is REINTERPRETED as
//     a `[BM][BN+8]` BF16 staging buffer for the coalesced writeback
//     (see ch07).  Its size is `EPILOGUE_STAGING_BYTES` below.
//
// The two phases never overlap in time (`all_mmas_done` separates
// them), so SMEM can be reused.  But the launcher MUST size the
// dynamic SMEM allocation to the MAX of the two phases:
//
//     shared_bytes = max(NS * SLOT_BYTES, EPILOGUE_STAGING_BYTES)
//                  + padding for __align__(1024)
//
// In ch07 (single-CTA) the K-loop term always dominated, so we never
// had to think about this.  In ch08, the per-CTA B-slot SMEM cost
// drops from 32 KB to 16 KB (cluster splits B), which means at low
// NS the K-loop SMEM can fall *below* the staging buffer's needs.
// Specifically, at NS=2, `NS * SLOT_BYTES = 64 KB < 67584 B` and the
// staging dominates.  Allocate too little dynamic SMEM and the
// epilogue scribbles past it → CUDA_ERROR_ILLEGAL_ADDRESS.
//
// See `shared_for()` in `main.py` for the launcher-side computation,
// and the README's "Sizing the dynamic SMEM" subsection for the
// full discussion.
constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;  // epilogue worker threads
constexpr int LAUNCH_THREADS = (NUM_WARPS + 4) * WARP_SIZE;


// ── helpers ─────────────────────────────────────────────────────────
// ---- elementwise epilogue (EDL): per-element fp32 map before bf16 store ----
__device__ __forceinline__ float mmc_epi(float x) { return x; }

__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

// TMA load — `.cta_group::2` is the key new modifier.  The tx-count
// is bookkept against a cluster-wide mbar (the peer-CTA arrival is
// what makes both CTAs' arrivals count toward CTA 0's SMEM-compute-full
// mbar).  Without it, peer-CTA loads silently fail to advance the
// mbar and the kernel deadlocks.
__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}


// A's descriptor — MN-major, unchanged from earlier chapters.
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);   // SWIZZLE_128B
}

// K-major B descriptor (same as ch06/07).
__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// idesc with M = CTA_GROUP * BM (cluster spans both CTAs in M),
// bit 16 = 1 (B is K-major).
__device__ __forceinline__ uint32_t make_idesc_bf16_cluster(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (1u << 16);                                   // B is K-major
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);          // n_dim
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);          // m_dim
    return d;
}


// ── tcgen05 MMA wrappers (cta_group::2 cluster / cta_group::1 single) ─
// Same names + signatures under both TWO_CTA arms so the call sites (and the
// MMA_ISSUE macro) are identical — TWO_CTA=1 renders byte-for-byte as the
// cluster tier; TWO_CTA=0 swaps in the single-CTA cta_group::1 instructions.
__device__ __forceinline__ void tcgen05_alloc_g2(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc_g2(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma_g2(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
// Multicast commit: arrives on the supplied mbar in every CTA whose bit is set in
// the mask.  cta_mask = (1 << CTA_GROUP) - 1 = 0b11 → both CTAs.
__device__ __forceinline__ void signal_on_mma_completion(uint32_t smem_bar, int16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        :: "r"(smem_bar), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// ── tcgen05.ld width helpers (building block) ───────────────────────
// mvp_core splices these at the TCGEN05_LD marker in every tier, so the
// TMEM->register load width (TCGEN05_LD_WIDTH = 8/16 32-bit elems per lane)
// is one knob with the asm in a single place.  Wider = fewer ld + fewer
// wait_ld syncs (more registers, but we're SMEM-occupancy-bound so it's free).
// The epilogue picks the variant via `#if` (resolved at generation time).

__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        :
          "=f"(out[0]),  "=f"(out[1]),  "=f"(out[2]),  "=f"(out[3]),
          "=f"(out[4]),  "=f"(out[5]),  "=f"(out[6]),  "=f"(out[7]),
          "=f"(out[8]),  "=f"(out[9]),  "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15]),
          "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]),
          "=f"(out[20]), "=f"(out[21]), "=f"(out[22]), "=f"(out[23]),
          "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
          "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
          "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
        : "r"(taddr));
}


// ── mbarrier helpers ────────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster(uint32_t mb) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                 :: "r"(mb) : "memory");
}
// Cluster-scope address of an mbarrier as owned by CTA 0: clearing bit 24 of the
// distributed-SMEM address selects CTA 0 of the pair.
__device__ __forceinline__ uint32_t mbarrier_addr_in_cta0(uint64_t& mbarrier) {
    return ((uint32_t)__cvta_generic_to_shared(&mbarrier)) & 0xFEFFFFFFu;
}
__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster_cta0(uint64_t& mbarrier) {
    mbarrier_arrive_no_tx_cluster(mbarrier_addr_in_cta0(mbarrier));
}
__device__ __forceinline__ void signal_on_bytes_loaded(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel (NS, GROUP_SIZE_M are file-level constexpr knobs) ────────
// ── TMA store helpers (pipelined TMA-store epilogue) ────────────────
__device__ __forceinline__ void tma_2d_store(
    const void* tmap, uint32_t smem_src, int x, int y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_src) : "memory");
}
__device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

// MMA-issue building block (shared fragment).  The cluster tier uses the
// g2 (cta_group::2) MMA instruction.
#define MMA_ISSUE(t, a, b, i, e) tcgen05_mma_g2((t), (a), (b), (i), (e))
// ── MMA-issue chain (building block) ────────────────────────────────
// Issues the K_MMAS tcgen05 MMAs for one K-tile (slot) into the
// accumulator at `taddr`.  mvp_core stitches this into every tier at the
// MMA-chain marker, so the descriptor math + K-step loop live in exactly
// one place.  The only per-tier variation is the MMA instruction
// itself, supplied just before the marker as:
//   MMA_ISSUE(taddr, a_desc, b_desc, idesc, enable_d)
// → tcgen05_mma (single-CTA) or tcgen05_mma_g2 (2-CTA cluster).
__device__ __forceinline__ void issue_mma_chain(
    uint32_t taddr, uint32_t a_base_slot, uint32_t b_base_slot,
    uint32_t idesc, bool first_k_tile)
{
    #pragma unroll
    for (int kk = 0; kk < K_MMAS; kk++) {
        const uint64_t a_desc = make_desc(a_base_slot + kk * MMA_K * BF16_BYTES);
        const uint64_t b_desc = make_desc_K_major(
            b_base_slot + kk * MMA_K * SWIZZLE_ROW_BYTES, BK * SWIZZLE_ROW_BYTES);
        const bool first_ever = first_k_tile && (kk == 0);
        MMA_ISSUE(taddr, a_desc, b_desc, idesc, !first_ever);
    }
}
#undef MMA_ISSUE

__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    const CUtensorMap* C_tmap_ptr,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
    // ── Per-cluster + per-CTA tile coords ───────────────────────────
    //
    // Grid is ceil(M / (CTA_GROUP*BM)) * ceil(N / BN) flat CTA ids.  Each
    // *pair* of CTAs forms one cluster; cta_rank picks which CTA in
    // the pair owns which half.  Ragged edge tiles are clipped by TMA.
    //
    // bid (the cluster id derived from blockIdx.x / CTA_GROUP) is what
    // we'd normally call the grid coordinate; the cluster handles a
    // 2*BM × BN output tile.
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Tile coords (the GSM chunked-walk swizzle) are computed PER-TILE
    // inside each path's persistent loop below — both the overlap and the
    // non-overlap branch derive (cluster_m, cluster_n) from their own
    // cluster id, so there are no tile-specific coords at this scope.

    // ── SMEM (per CTA — B is now half-width) ────────────────────────
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t mbar_compute_data_ready[NS];
    __shared__ uint64_t mbar_compute_buffer_free[NS];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    {
        // Persistent cluster pipeline: both CTAs stream A/B, CTA 0 issues
        // cta_group::2 MMA into a two-buffer TMEM accumulator, and every CTA
        // drains its own BM x BN output half while the next cluster tile runs.
        __shared__ uint64_t mbar_tmem_data_ready[2];
        // One accumulator-free barrier per 256-column MMA panel.
        __shared__ uint64_t mbar_tmem_panel_free[2];
        // Pipelined TMA-store mode keeps the K-loop ring intact and reserves
        // compact 128B-swizzled SMEM buffers for chunked TMA stores.
        constexpr int STORE_BUF_BYTES = BM * STORE_N * BF16_BYTES;
        const uint32_t STORE_SMEM_BASE = SMEM_BASE + NS * SLOT_BYTES;
        auto C_store = reinterpret_cast<__nv_bfloat16*>(smem + NS * SLOT_BYTES);

        if (warp_id == 0) {
            tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
        }
        if (warp_id == 0 && elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]));
            }
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[b]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_panel_free[b]), CTA_GROUP);
                mbarrier_arrive_no_tx_cluster_cta0(mbar_tmem_panel_free[b]);
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }

        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");

        const uint32_t taddr = tmem_addr_holder[0];
        constexpr int BN_PANEL = 256;
        constexpr int BN_PANEL_LOCAL = BN_PANEL / CTA_GROUP;
        const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN_PANEL);
        const int num_k = K / BK;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

        // ceil-div tile counts: a ragged M/N launches partial edge tiles whose
        // TMA box is clipped out of bounds (zero-fill on load, masked on store).
        const int grid_m_clusters = (M + CTA_GROUP * BM - 1) / (CTA_GROUP * BM);
        const int grid_n          = (N + BN - 1) / BN;
        const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
        const int num_clusters = grid_m_clusters * grid_n;
        const int cluster_pid = (int)blockIdx.x / CTA_GROUP;
        const int cluster_stride = (int)gridDim.x / CTA_GROUP;
        const int num_my = (cluster_pid >= num_clusters) ? 0
                         : (num_clusters - cluster_pid + cluster_stride - 1) / cluster_stride;

        auto map_off = [&](int ti, int& base_m, int& base_n, int& local_m, int& local_n) {
            int tile = cluster_pid + ti * cluster_stride;
            int group = tile / num_cluster_in_group;
            int first = group * GROUP_SIZE_M;
            int gsm_i = min(grid_m_clusters - first, GROUP_SIZE_M);
            int cm = first + (tile % gsm_i);
            int cn = (tile % num_cluster_in_group) / gsm_i;
            base_m = cm * (CTA_GROUP * BM);
            base_n = cn * BN;
            local_m = base_m + cta_rank * BM;
            local_n = base_n + cta_rank * BN_LOCAL;
        };

        if (warp_id == 0 && elect_sync()) {
            uint32_t compute_buffer_free_phase[NS] = {};
            long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                map_off(ti, base_m, base_n, local_m, local_n);
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    uint32_t compute_buffer_free_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);
                    uint32_t compute_data_ready_cta0 =
                        mbarrier_addr_in_cta0(mbar_compute_data_ready[slot]);
                    wait_phase(compute_buffer_free_addr, compute_buffer_free_phase[slot]);
                    tma_2d_load_g2(A_base(slot), A_tmap, k * BK, local_m, compute_data_ready_cta0);
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        #pragma unroll
                        for (int n = 0; n < BN_PANEL_LOCAL; n += 64) {
                            tma_2d_load_g2(
                                B_base(slot) + (panel * BN_PANEL_LOCAL + n) * BK * BF16_BYTES,
                                B_tmap,
                                base_n + panel * BN_PANEL + cta_rank * BN_PANEL_LOCAL + n,
                                k * BK,
                                compute_data_ready_cta0);
                        }
                    }
                    signal_on_bytes_loaded(compute_data_ready_cta0, SLOT_BYTES);
                    compute_buffer_free_phase[slot] ^= 1;
                    gk++;
                }
            }
        } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
            uint32_t compute_data_ready_phase[NS] = {};
            uint32_t tmem_panel_free_phase[2] = {};
            long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                uint32_t d_tmem = taddr;

                auto take_slot = [&](int s) {
                    wait_phase((uint32_t)__cvta_generic_to_shared(
                                   &mbar_compute_data_ready[s]),
                               compute_data_ready_phase[s]);
                    compute_data_ready_phase[s] ^= 1;
                };
                auto free_slot = [&](int s) {
                    signal_on_mma_completion(
                        (uint32_t)__cvta_generic_to_shared(
                            &mbar_compute_buffer_free[s]), cta_mask);
                };
                auto wait_panel = [&](int p) {
                    wait_phase((uint32_t)__cvta_generic_to_shared(
                                   &mbar_tmem_panel_free[p]),
                               tmem_panel_free_phase[p]);
                    tmem_panel_free_phase[p] ^= 1;
                };
                auto mma0 = [&](int s, bool first) {
                    tcgen05_fence_after_thread_sync();
                    issue_mma_chain(d_tmem, A_base(s), B_base(s), idesc, first);
                };
                auto mma1 = [&](int s, bool first) {
                    tcgen05_fence_after_thread_sync();
                    issue_mma_chain(d_tmem + BN_PANEL,
                                    A_base(s),
                                    B_base(s) + BN_PANEL_LOCAL * BK * BF16_BYTES,
                                    idesc, first);
                };
                auto panel0_ready = [&]() {
                    signal_on_mma_completion(
                        (uint32_t)__cvta_generic_to_shared(
                            &mbar_tmem_data_ready[0]), cta_mask);
                };

                // Panel 0 runs one k-tile ahead of panel 1 at both ends of the
                // k loop. tcgen05.commit signals on completion of everything
                // issued before it, so moving panel 0's last MMA two issues
                // earlier moves its data-ready a whole k-tile earlier, and the
                // epilogue starts draining panel 0 that much sooner. The cost
                // is slot lifetime: at each end two TMA slots stay live until
                // panel 1 has consumed them, instead of one.
                if (num_k >= 4) {
                    // head: panel 0 takes k = 0 and 1 before panel 1 starts,
                    // so panel 1's free-barrier wait does not delay panel 0
                    const int s0 = (int)(gk % NS), s1 = (int)((gk + 1) % NS);
                    take_slot(s0);
                    wait_panel(0);
                    mma0(s0, /*first_k_tile=*/ true);
                    take_slot(s1);
                    mma0(s1, /*first_k_tile=*/ false);
                    wait_panel(1);
                    mma1(s0, /*first_k_tile=*/ true);
                    free_slot(s0);
                    mma1(s1, /*first_k_tile=*/ false);
                    free_slot(s1);
                    gk += 2;

                    // steady state: both panels on the same k-tile, one slot live
                    for (int k = 2; k < num_k - 2; k++) {
                        const int s = (int)(gk % NS);
                        take_slot(s);
                        mma0(s, /*first_k_tile=*/ false);
                        mma1(s, /*first_k_tile=*/ false);
                        free_slot(s);
                        gk++;
                    }

                    // tail: panel 0 takes the last two k-tiles first, so its
                    // data-ready fires a whole k-tile before panel 1's
                    const int sa = (int)(gk % NS), sb = (int)((gk + 1) % NS);
                    take_slot(sa);
                    mma0(sa, /*first_k_tile=*/ false);
                    take_slot(sb);
                    mma0(sb, /*first_k_tile=*/ false);
                    panel0_ready();
                    mma1(sa, /*first_k_tile=*/ false);
                    free_slot(sa);
                    mma1(sb, /*first_k_tile=*/ false);
                    free_slot(sb);
                    gk += 2;
                } else {
                    // too few k-tiles to stagger: the original interleaved order
                    for (int k = 0; k < num_k; k++) {
                        const int s = (int)(gk % NS);
                        take_slot(s);
                        if (k == 0) {
                            wait_panel(0);
                            mma0(s, true);
                            wait_panel(1);
                        } else {
                            mma0(s, false);
                        }
                        if (k == num_k - 1) panel0_ready();
                        mma1(s, k == 0);
                        free_slot(s);
                        gk++;
                    }
                }
                signal_on_mma_completion(
                    (uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[1]),
                    cta_mask);
            }
        } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {
            // Contract for the shared overlap-drain fragment: cluster tier writes
            // this CTA's BM x BN output half (local_m / base_n) and releases the
            // TMEM buffer with a CTA-0-masked cluster arrive.
#define EPI_OUT_ROW                 local_m
#define EPI_OUT_COL_BASE            base_n
            constexpr int ROW_STRIPS    = BM / 32;
            constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
            constexpr int COLS_PER_WARP = BN / COL_GROUPS;
            constexpr int EPI_THREADS   = NUM_WARPS * 32;
            const int ew = warp_id - 4;
            const int row_warp = ew % ROW_STRIPS;
            const int col_warp = ew / ROW_STRIPS;
            const int my_row = row_warp * 32 + lane;
            const int col_base = col_warp * COLS_PER_WARP;
            const int etid = ew * 32 + lane;
            uint32_t full[2] = {};
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                const int buf = 0;
                map_off(ti, base_m, base_n, local_m, local_n);
                const uint32_t trow =
                    (taddr + buf * BN) + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
                constexpr int LDW = TCGEN05_LD_WIDTH;
                {
                    // Two-level drain. LOAD_N accumulator columns come out of
                    // TMEM per outer iteration, then the inner loop stages and
                    // stores them STORE_N columns at a time. With two column
                    // warp groups a lane's share of a LOAD_N group is
                    // LOAD_N / COL_GROUPS columns, and one tcgen05.ld.x32
                    // covers exactly one group's share of one STORE_N chunk -
                    // so there is one load per chunk and no inner load index.
                    constexpr int BN_PANEL_EPI    = 256;                 // must match the MMA panel width
                    constexpr int LOAD_N          = 256;                 // columns per outer load
                    constexpr int LD_REGS         = 32;                  // floats per tcgen05.ld.x32
                    constexpr int NUM_LOADS       = BN / LOAD_N;         // 2 TMEM->reg trips
                    constexpr int CHUNKS_PER_LOAD = LOAD_N / STORE_N;    // 4 stores per trip
                    constexpr int PACKS_PER_LD    = LD_REGS / 8;         // int4 writes per load
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(COL_GROUPS == 2, "load256 drain assumes two column groups");
                    static_assert(STORE_N / COL_GROUPS == LD_REGS,
                                  "one x32 load must cover one group's share of a chunk");
                    static_assert(NUM_LOADS * LOAD_N == BN, "BN must divide into LOAD_N groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int load = 0; load < NUM_LOADS; load++) {
                        // Each panel has its own data-ready signal, so wait for
                        // just this panel: panel 0 drains while panel 1 is still
                        // accumulating.
                        wait_phase(
                            (uint32_t)__cvta_generic_to_shared(
                                &mbar_tmem_data_ready[load]),
                            full[load]);
                        full[load] ^= 1;
                        tcgen05_fence_after_thread_sync();

                        // t[c] holds this column group's 32 columns of chunk c.
                        float t[CHUNKS_PER_LOAD][LD_REGS];
                        #pragma unroll
                        for (int c = 0; c < CHUNKS_PER_LOAD; c++)
                            tcgen05_ld_32x32b_x32(
                                trow + (uint32_t)(load * LOAD_N + c * STORE_N +
                                                  col_warp * LD_REGS),
                                t[c]);
                        tcgen05_wait_ld();

                        // Release this panel's accumulator columns now that they
                        // are in registers: one outer iteration drains exactly
                        // one MMA panel, so the next output tile's MMAs for this
                        // panel can start while the stores below are still
                        // running. Nothing here depends on a free store slot, so
                        // the arrive stays ahead of the tma_wait_group.
                        //
                        // tcgen05_wait_ld above covers only the issuing warp, so
                        // the bar.sync is what lets one warp's arrive speak for
                        // all eight. It cannot be dropped or moved after the
                        // arrive: the MMA would then be free to overwrite TMEM
                        // while the other warps are still reading it.
                        static_assert(LOAD_N == BN_PANEL_EPI,
                                      "one outer load must drain exactly one MMA panel");
                        tcgen05_fence_before_thread_sync();
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
                        if (ew == 0 && elect_sync())
                            mbarrier_arrive_no_tx_cluster_cta0(
                                mbar_tmem_panel_free[load]);

                        #pragma unroll
                        for (int c = 0; c < CHUNKS_PER_LOAD; c++) {
                            // The TMEM->reg loads above don't touch the store
                            // buffer, so the free-store-slot wait sits here,
                            // just before the buffer write, and stays before the
                            // bar.sync so every warp observes the ew==0 wait.
                            if (ew == 0)
                                tma_wait_group<TMA_STORE_STAGES - 1>();

                            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                            #pragma unroll
                            for (int g = 0; g < PACKS_PER_LD; g++) {
                                __nv_bfloat162 pk[4];
                                #pragma unroll
                                for (int i = 0; i < 4; i++)
                                    pk[i] = __floats2bfloat162_rn(
                                        mmc_epi(t[c][g * 8 + 2 * i]),
                                        mmc_epi(t[c][g * 8 + 2 * i + 1]));
                                const int local_n = col_warp * PACKS_PER_LD + g;
                                const int swizzled_n = local_n ^ (my_row & 7);
                                __nv_bfloat16* write_ptr =
                                    C_store + store_stage * BM * STORE_N +
                                    my_row * STORE_N + swizzled_n * 8;
                                *reinterpret_cast<int4*>(write_ptr) =
                                    *reinterpret_cast<int4*>(pk);
                            }

                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                            if (ew == 0 && elect_sync()) {
                                const uint32_t src =
                                    STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                                tma_2d_store(C_tmap_ptr, src,
                                             EPI_OUT_COL_BASE +
                                                 (load * CHUNKS_PER_LOAD + c) * STORE_N,
                                             EPI_OUT_ROW);
                                tma_commit_group();
                            }

                            store_stage ^= 1;
                        }
                    }
                }
            }
            if (ew == 0)
                tma_wait_group<0>();
            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
#undef EPI_OUT_ROW
#undef EPI_OUT_COL_BASE
        }

        __syncthreads();
        if (warp_id == 0 && elect_sync()) {
            tcgen05_dealloc_g2(taddr, BN);
        }
        return;
    }
}


// ── Single entry symbol — NS and GROUP_SIZE_M are baked in from the
// constexpr knobs at the top of the file (the webui substitutes them).
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(LAUNCH_THREADS, 1)
void matmul_cluster(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    __nv_bfloat16* C_ptr, int M, int N, int K
)
{
    matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K
    );
}
