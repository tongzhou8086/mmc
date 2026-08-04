// Helpers shared by the MMC CUDA MXFP8 kernels: SMEM/TMEM tile descriptors, warp
// election, tcgen05 fences and TMEM loads, mbarrier and cluster primitives, the
// 128B-swizzled TMA load/store wrappers, the dynamic-SMEM allocator, and the
// MXFP8 pieces - the packed E8M0 scale atom with its TMA and tcgen05.cp paths,
// the K-major FP8 operand descriptors, and the tcgen05.mma.kind::mxf8f6f4
// instruction descriptor and issue wrapper.
//
// Each kernel .cu keeps only what is per-variant: its tile shape and pipeline
// constants, its tile typedefs, its MMA schedule and the kernel body.
//
// Most of this is not MXFP8-specific and will be split into a data-type-agnostic
// header when BF16 kernels are added.
//
// These sources are shipped for review only. MMC loads the prebuilt cubins at
// runtime and never compiles a kernel.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

template <typename T, int Rows_, int Cols_, bool Swizzle_, int SwizzleBytes_>
struct alignas(128) SmemTile {
    using dtype = T;
    static constexpr int rows = Rows_;
    static constexpr int cols = Cols_;
    static constexpr bool swizzle = Swizzle_;
    static constexpr int swizzle_bytes = SwizzleBytes_;

    T data[Rows_ * Cols_];
};

template <typename T, int Rows_, int Cols_>
struct TmemTile {
    using dtype = T;
    static constexpr int rows = Rows_;
    static constexpr int cols = Cols_;

    uint32_t addr;

    __device__ explicit TmemTile(uint32_t tmem_addr) : addr(tmem_addr) {}

    template <int SubCols>
    __device__ __forceinline__ TmemTile<T, Rows_, SubCols> subtile(int col_offset) const {
        constexpr uint32_t columns_per_tmem_addr = 4 / static_cast<uint32_t>(sizeof(T));
        return TmemTile<T, Rows_, SubCols>(addr + col_offset / columns_per_tmem_addr);
    }
};

static_assert(sizeof(CUtensorMap) == 128);

__device__ __forceinline__ float mmc_epi(float x) { return x; }

__device__ __forceinline__ bool elect_sync() {
    // Use the warp-level election primitive, not a lane-0 predicate.  The
    // explicit lane check is logically similar here but generates slower
    // control flow in the persistent pipeline.
    uint32_t elected = 0;
    asm volatile(
        "{.reg .pred P;\n"
        " elect.sync _|P, %1;\n"
        " selp.u32 %0, 1, 0, P;}\n"
        : "+r"(elected)
        : "r"(0xFFFFFFFF));
    return static_cast<bool>(elected);
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
          "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15]),
          "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]),
          "=f"(out[20]), "=f"(out[21]), "=f"(out[22]), "=f"(out[23]),
          "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
          "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31])
        : "r"(taddr));
}

__device__ __forceinline__ uint32_t shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t shared_u32_in_cta(const void* ptr, int cta_rank) {
    uint32_t local_addr = shared_u32(ptr);
    uint32_t mapped_addr;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped_addr)
        : "r"(local_addr), "r"(cta_rank));
    return mapped_addr;
}

__device__ __forceinline__ uint32_t mbarrier_addr(uint64_t& mbarrier) {
    return shared_u32(&mbarrier);
}

__device__ __forceinline__ uint32_t mbarrier_addr_in_cta(uint64_t& mbarrier, int cta_rank) {
    return shared_u32_in_cta(&mbarrier, cta_rank);
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__ int cluster_cta_rank() {
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cta_rank));
    return static_cast<int>(cta_rank);
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor(const CUtensorMap* tensor_map) {
    asm volatile(
        "{prefetch.tensormap [%0];}"
        :: "l"(reinterpret_cast<uint64_t>(tensor_map))
        : "memory");
}

template <typename Tile>
__device__ __forceinline__ void load_fp8_operand_tile_from_gmem_to_smem(
    Tile& dst,
    const CUtensorMap* operand_tensor_map,
    int row_start,
    int k_tile,
    uint32_t completion_mbarrier_addr,
    uint16_t destination_cta_mask
) {
    static_assert(Tile::swizzle, "FP8 operand TMA tiles must use the 128B swizzled layout");
    static_assert(Tile::swizzle_bytes == 128, "This wrapper mirrors TK's 128B swizzled TMA path");
    constexpr int swizzle_elements =
        Tile::swizzle_bytes / sizeof(typename Tile::dtype);
    static_assert(Tile::cols % swizzle_elements == 0);
    constexpr int k_tma_step = Tile::cols / swizzle_elements;

    const uint32_t dst_addr = shared_u32(&dst);
    const uint64_t tma_addr = reinterpret_cast<uint64_t>(operand_tensor_map);
    const int k_swizzle_tile = k_tile * k_tma_step;

    // Mirrors the TK TMA wrapper for a 128B-swizzled ST tile
    // with axis=ROW.  The CUtensorMap has innermost-first dimensions:
    //   [128B swizzle group, logical rows, logical K / 128B, depth, batch].
    // So a logical tile at rows row_start..row_start+rows-1 and K stage k
    // is requested with TMA coords {0, row_start, k_swizzle_tile, 0, 0}.
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
        " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
        :
        : "r"(dst_addr), "l"(tma_addr), "r"(completion_mbarrier_addr),
          "n"(0), "r"(row_start), "r"(k_swizzle_tile), "r"(0), "r"(0),
          "h"(destination_cta_mask)
        : "memory");
}

template <typename Tile>
__device__ __forceinline__ void store_bf16_tile_from_smem_to_gmem(
    const CUtensorMap* output_tensor_map,
    const Tile& src,
    int row_start,
    int col_tile
) {
    static_assert(Tile::swizzle, "BF16 store tile must use the 128B swizzled layout");
    static_assert(Tile::swizzle_bytes == 128, "This wrapper mirrors TK's 128B swizzled TMA store path");
    constexpr int swizzle_elements =
        Tile::swizzle_bytes / sizeof(typename Tile::dtype);
    static_assert(Tile::cols % swizzle_elements == 0);
    constexpr int col_tma_step = Tile::cols / swizzle_elements;

    const uint64_t tma_addr = reinterpret_cast<uint64_t>(output_tensor_map);
    const uint32_t src_addr = shared_u32(&src);
    const int col_swizzle_tile = col_tile * col_tma_step;

    fence_proxy_async_shared_cta();
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4, %5, %6}], [%1];"
        :
        : "l"(tma_addr), "r"(src_addr),
          "n"(0), "r"(row_start), "r"(col_swizzle_tile), "r"(0), "r"(0)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void commit_mma_arrive_cta_group_2(
    uint32_t mbarrier_addr,
    uint16_t cta_mask
) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :: "r"(mbarrier_addr), "h"(cta_mask));
}

__device__ __forceinline__ void tcgen05_alloc_g2(uint32_t& tmem_addr, int columns) {
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :: "l"(reinterpret_cast<uint64_t>(&tmem_addr)), "r"(columns));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
}

__device__ __forceinline__ void tcgen05_dealloc_g2(uint32_t tmem_addr, int columns) {
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
        :: "r"(tmem_addr), "r"(columns));
}

template <int BarrierId, int Threads>
__device__ __forceinline__ void bar_sync() {
    asm volatile("bar.sync %0, %1;" :: "n"(BarrierId), "n"(Threads));
}

__device__ __forceinline__ void mbarrier_init(uint64_t& mbarrier, int count) {
    const uint32_t mbar_addr = mbarrier_addr(mbarrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_no_tx(uint64_t& mbarrier) {
    const uint32_t mbar_addr = mbarrier_addr(mbarrier);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :: "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster_cta0(uint64_t& mbarrier) {
    const uint32_t mbar_addr = mbarrier_addr_in_cta(mbarrier, 0);
    asm volatile(
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(1)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cta0(uint64_t& mbarrier, int bytes) {
    const uint32_t mbar_addr = mbarrier_addr_in_cta(mbarrier, 0);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(bytes));
}

__device__ __forceinline__ void tcgen05_mma_commit_arrive(
    uint64_t& mbarrier,
    uint16_t cta_mask
) {
    commit_mma_arrive_cta_group_2(mbarrier_addr(mbarrier), cta_mask);
}

__device__ __forceinline__ void mbarrier_wait_phase(uint64_t& mbarrier, uint32_t phase) {
    const uint32_t mbar_addr = mbarrier_addr(mbarrier);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_addr), "r"(phase));
}

template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

struct SharedMemoryAllocator1024 {
    int* ptr;

    __device__ explicit SharedMemoryAllocator1024(int* base) : ptr(base) {}

    __device__ __forceinline__ void align_ptr() {
        uint64_t p = reinterpret_cast<uint64_t>(ptr);
        constexpr uint64_t alignment = 1024;
        if (p % alignment != 0) {
            ptr = reinterpret_cast<int*>(p + (alignment - (p % alignment)));
        }
    }

    template <typename A>
    __device__ __forceinline__ A& allocate() {
        align_ptr();
        A* out = reinterpret_cast<A*>(ptr);
        ptr += sizeof(A) / sizeof(int);
        return *out;
    }
};

struct alignas(16) ScaleAtom {
    uint8_t data[32][16];
};

static_assert(sizeof(ScaleAtom) == 32 * 16);

__device__ __forceinline__ uint64_t make_scale_smem_desc(const void* smem_ptr) {
    // Matches the descriptor TK builds for tcgen05.cp on a non-swizzled
    // 32x16 E8M0 source tile.  The 128 offsets are the ISA descriptor
    // units used by the 32x128b copy shape.
    uint64_t desc = matrix_descriptor_encode(reinterpret_cast<uint64_t>(smem_ptr));
    desc |= 1ull << 46;  // Blackwell shared-memory descriptor bit.
    desc |= matrix_descriptor_encode(128ull) << 16;
    desc |= matrix_descriptor_encode(128ull) << 32;
    return desc;
}

__device__ __forceinline__ void load_scale_atom_from_gmem_to_smem(
    ScaleAtom& dst,
    const CUtensorMap* scale_tensor_map,
    int outer_tile,
    int k_tile,
    uint32_t completion_mbarrier_addr,
    uint16_t destination_cta_mask
) {
    const uint32_t dst_addr = shared_u32(&dst.data[0][0]);
    const uint64_t tma_addr = reinterpret_cast<uint64_t>(scale_tensor_map);

    // The scale CUtensorMap is 4D with innermost-first dimensions:
    //   [16, 32, K/128, outer].
    // Therefore the requested tile coordinate is:
    //   {col=0, row=0, k_tile, outer_tile}.
    //
    // On SM100, the cta_group::2 form is what lets the transaction complete
    // into CTA 0's mbarrier even when CTA 1 issues the load.  The destination
    // mask controls whether the payload lands in one CTA or is multicast.
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
        " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :
        : "r"(dst_addr), "l"(tma_addr), "r"(completion_mbarrier_addr),
          "r"(0), "r"(0), "r"(k_tile), "r"(outer_tile), "h"(destination_cta_mask)
        : "memory");
}

__device__ __forceinline__ void copy_scale_atom_from_smem_to_tmem(
    uint32_t tmem_addr,
    const ScaleAtom& src
) {
    uint64_t st_desc = make_scale_smem_desc(&src.data[0][0]);
    asm volatile(
        "{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}"
        :: "r"(tmem_addr), "l"(st_desc));
}

template <typename Tile>
__device__ __forceinline__ uint64_t make_k_major_operand_smem_desc(
    const Tile& tile
) {
    static_assert(Tile::swizzle, "tcgen05 operand tiles must be swizzled");
    static_assert(Tile::swizzle_bytes == 128, "This kernel uses 128B swizzled FP8 operand tiles");
    static_assert(sizeof(typename Tile::dtype) == 1, "This helper is specialized for FP8 operands");

    uint64_t desc = matrix_descriptor_encode(reinterpret_cast<uint64_t>(&tile.data[0]));
    desc |= 1ull << 46;  // Blackwell shared-memory descriptor bit.
    desc |= matrix_descriptor_encode(16ull) << 16;    // ignored in K-major mode
    desc |= matrix_descriptor_encode(1024ull) << 32;  // 128B swizzle stride
    desc |= 1ull << 62;  // 128B swizzle mode
    return desc;
}

template <typename Tile>
__device__ __forceinline__ uint64_t k_major_operand_chunk_desc(
    uint64_t base_desc,
    int chunk_idx
) {
    static_assert(Tile::swizzle_bytes == 128);
    constexpr int tile_row_dim = 16;
    const int byte_offset =
        (chunk_idx % 4) * 32 +
        (chunk_idx / 4) * (Tile::rows / tile_row_dim) * 2048;
    return base_desc + matrix_descriptor_encode(static_cast<uint64_t>(byte_offset));
}

// MXFP8 dense ABt instruction descriptor for tcgen05.mma.kind::mxf8f6f4 with
// E4M3 operands, E8M0 scales and scale_vec::1X. MmaM/MmaN are the MMA shape the
// caller issues; with cta_group::2 MmaM is the CTA pair's row count.
template <int ScaleFactorId, int MmaM, int MmaN>
__device__ __forceinline__ constexpr uint32_t mxfp8_mma2_abt_desc() {
    static_assert(ScaleFactorId >= 0 && ScaleFactorId < 4);
    static_assert(MmaM % 128 == 0 && MmaM <= 256);
    static_assert(MmaN > 0 && MmaN <= 256 && MmaN % 8 == 0);

    uint32_t desc = 0;
    desc |= 0b00 << 0;                 // dense, no sparsity
    desc |= 0b0 << 2;                  // dense
    desc |= 0b0 << 3;                  // no saturate
    desc |= ScaleFactorId << 4;        // B scale-factor id
    desc |= 0b000 << 7;                // A is E4M3
    desc |= 0b000 << 10;               // B is E4M3
    desc |= 0b0 << 13;                 // do not negate A
    desc |= 0b0 << 14;                 // do not negate B
    desc |= 0b0 << 15;                 // block-scaled MMA: no transpose bits
    desc |= 0b0 << 16;
    desc |= (MmaN >> 3) << 17;         // N dimension
    desc |= 0b1 << 23;                 // E8M0 scales
    desc |= 0b000 << 24;
    desc |= (MmaM >> 7) << 27;         // M dimension
    desc |= ScaleFactorId << 29;       // A scale-factor id
    desc |= 0b0u << 31;                // MXFP8 K chunk is 32B
    return desc;
}

template <int Accumulate>
__device__ __forceinline__ void issue_mxfp8_mma2_abt_chunk(
    uint32_t dst_tmem_addr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_tmem_addr,
    uint32_t b_scale_tmem_addr,
    uint32_t instruction_desc
) {
    static_assert(Accumulate == 0 || Accumulate == 1);
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;}\n"
        :
        : "r"(dst_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(instruction_desc),
          "r"(a_scale_tmem_addr), "r"(b_scale_tmem_addr), "n"(Accumulate));
}
