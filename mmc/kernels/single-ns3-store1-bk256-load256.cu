#include "cuda-mxfp8.cuh"

// Single-TMEM early-release BK=256 variant. Eight epilogue warps load all 256
// accumulator columns into registers, release TMEM, then drain four
// 64-column SMEM/TMA store tiles.
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 256;
constexpr int NS           = 3;
constexpr int GROUP_SIZE_M = 16;
constexpr int NUM_WARPS    = 8;
constexpr int CTA_GROUP    = 2;
constexpr int BN_LOCAL     = BN / CTA_GROUP;
constexpr int STORE_N      = 64;
constexpr int TMA_STORE_STAGES = 1;
constexpr int K_SCALE_ATOMS = BK / 128;

constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int LAUNCH_THREADS = (NUM_WARPS + 4) * WARP_SIZE;

using a_tile = SmemTile<__nv_fp8_e4m3, BM, BK, true, 128>;
using b_tile = SmemTile<__nv_fp8_e4m3, BN_LOCAL, BK, true, 128>;
using d_tile = SmemTile<__nv_bfloat16, BM, STORE_N, true, 128>;

template <
    int Accumulate,
    typename AccumTensor,
    typename AScaleTensor0,
    typename BScaleTensor0,
    typename AScaleTensor1,
    typename BScaleTensor1>
__device__ __forceinline__ void issue_mxfp8_mma2_abt_128x256x256(
    const AccumTensor& dst,
    const a_tile& a,
    const b_tile& b,
    const AScaleTensor0& a_scale_0,
    const BScaleTensor0& b_scale_0,
    const AScaleTensor1& a_scale_1,
    const BScaleTensor1& b_scale_1,
    uint32_t smem_buffer_free_mbarrier_addr
) {
    static_assert(BM == 128 && BN == 256 && BK == 256);
    static_assert(CTA_GROUP == 2 && BN_LOCAL == 128);
    static_assert(K_SCALE_ATOMS == 2);

    constexpr uint32_t idescs[4] = {
        mxfp8_mma2_abt_desc<0, CTA_GROUP * BM, BN>(),
        mxfp8_mma2_abt_desc<1, CTA_GROUP * BM, BN>(),
        mxfp8_mma2_abt_desc<2, CTA_GROUP * BM, BN>(),
        mxfp8_mma2_abt_desc<3, CTA_GROUP * BM, BN>(),
    };
    const uint64_t a_base = make_k_major_operand_smem_desc(a);
    const uint64_t b_base = make_k_major_operand_smem_desc(b);

    issue_mxfp8_mma2_abt_chunk<Accumulate>(
        dst.addr,
        k_major_operand_chunk_desc<a_tile>(a_base, 0),
        k_major_operand_chunk_desc<b_tile>(b_base, 0),
        a_scale_0.addr, b_scale_0.addr, idescs[0]);

    #pragma unroll
    for (int sfid = 1; sfid < 4; ++sfid) {
        issue_mxfp8_mma2_abt_chunk<1>(
            dst.addr,
            k_major_operand_chunk_desc<a_tile>(a_base, sfid),
            k_major_operand_chunk_desc<b_tile>(b_base, sfid),
            a_scale_0.addr, b_scale_0.addr, idescs[sfid]);
    }

    #pragma unroll
    for (int sfid = 0; sfid < 4; ++sfid) {
        issue_mxfp8_mma2_abt_chunk<1>(
            dst.addr,
            k_major_operand_chunk_desc<a_tile>(a_base, 4 + sfid),
            k_major_operand_chunk_desc<b_tile>(b_base, 4 + sfid),
            a_scale_1.addr, b_scale_1.addr, idescs[sfid]);
    }

    commit_mma_arrive_cta_group_2(
        smem_buffer_free_mbarrier_addr,
        static_cast<uint16_t>((1 << CTA_GROUP) - 1));
}

__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap& A_tmap,
    const CUtensorMap& A_sc_tmap,
    const CUtensorMap& B_tmap,
    const CUtensorMap& B_sc_tmap,
    const CUtensorMap& D_tmap,
    __nv_bfloat16*,
    int M, int N, int K
) {
    int cta_rank = cluster_cta_rank();

    if (threadIdx.x == 0) {
        // Optional hint: prefetch the 128B TMA descriptors, not tensor data.
        prefetch_tma_descriptor(&A_tmap);
        prefetch_tma_descriptor(&A_sc_tmap);
        prefetch_tma_descriptor(&B_tmap);
        prefetch_tma_descriptor(&B_sc_tmap);
        prefetch_tma_descriptor(&D_tmap);
    }

    extern __shared__ int __shm[];
    SharedMemoryAllocator1024 al((int*)&__shm[0]);

    a_tile (&a_smem)[NS] = al.allocate<a_tile[NS]>();
    b_tile (&b_smem)[NS] = al.allocate<b_tile[NS]>();
    ScaleAtom (&a_sc_smem)[NS][K_SCALE_ATOMS] =
        al.allocate<ScaleAtom[NS][K_SCALE_ATOMS]>();
    ScaleAtom (&b_sc_smem)[NS][CTA_GROUP][K_SCALE_ATOMS] =
        al.allocate<ScaleAtom[NS][CTA_GROUP][K_SCALE_ATOMS]>();
    d_tile (&d_smem)[TMA_STORE_STAGES] = al.allocate<d_tile[TMA_STORE_STAGES]>();

    __shared__ uint64_t mbar_compute_data_ready[NS];
    __shared__ uint64_t mbar_compute_buffer_free[NS];
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    __shared__ uint64_t mbar_tmem_data_ready;
    __shared__ uint64_t mbar_tmem_buffer_free;

    constexpr int TMEM_ALLOC_COLS = 512;
    constexpr int TMEM_OUT_COL = 0;
    constexpr int TMEM_A_SCALE_COL = 256;
    constexpr int TMEM_B_SCALE_COL = 384;

    if (warp_id == 0) {
        tcgen05_alloc_g2(tmem_addr_holder[0], TMEM_ALLOC_COLS);
    }
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            mbarrier_init(mbar_compute_data_ready[s], CTA_GROUP);
            mbarrier_init(mbar_compute_buffer_free[s], 1);
            mbarrier_arrive_no_tx(mbar_compute_buffer_free[s]);
        }
        mbarrier_init(mbar_tmem_data_ready, 1);
        mbarrier_init(mbar_tmem_buffer_free, CTA_GROUP);
        mbarrier_arrive_no_tx_cluster_cta0(mbar_tmem_buffer_free);
        fence_mbarrier_init_release_cluster();
    }

    cluster_sync();

    const uint32_t taddr = tmem_addr_holder[0];
    const int num_k = K / BK;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

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
        int remaining_m_groups = grid_m_clusters - first;
        int gsm_i = remaining_m_groups < GROUP_SIZE_M ? remaining_m_groups : GROUP_SIZE_M;
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
                mbarrier_wait_phase(
                    mbar_compute_buffer_free[slot],
                    compute_buffer_free_phase[slot]);

                uint32_t compute_data_ready_cta0 =
                    mbarrier_addr_in_cta(mbar_compute_data_ready[slot], 0);

                load_fp8_operand_tile_from_gmem_to_smem(
                    a_smem[slot],
                    &A_tmap,
                    local_m, k,
                    compute_data_ready_cta0,
                    static_cast<uint16_t>(1 << cta_rank));
                load_fp8_operand_tile_from_gmem_to_smem(
                    b_smem[slot],
                    &B_tmap,
                    local_n, k,
                    compute_data_ready_cta0,
                    static_cast<uint16_t>(1 << cta_rank));
                #pragma unroll
                for (int k_half = 0; k_half < K_SCALE_ATOMS; ++k_half) {
                    const int scale_k = k * K_SCALE_ATOMS + k_half;
                    load_scale_atom_from_gmem_to_smem(
                        a_sc_smem[slot][k_half],
                        &A_sc_tmap,
                        local_m / BM, scale_k,
                        compute_data_ready_cta0,
                        static_cast<uint16_t>(1 << cta_rank));

                    load_scale_atom_from_gmem_to_smem(
                        b_sc_smem[slot][cta_rank][k_half],
                        &B_sc_tmap,
                        local_n / BN_LOCAL, scale_k,
                        compute_data_ready_cta0,
                        static_cast<uint16_t>(cta_mask));
                }

                constexpr int slot_bytes =
                    sizeof(a_tile) + sizeof(b_tile) +
                    sizeof(ScaleAtom) * K_SCALE_ATOMS * (1 + CTA_GROUP);
                mbarrier_arrive_expect_tx_cta0(
                    mbar_compute_data_ready[slot], slot_bytes);
                compute_buffer_free_phase[slot] ^= 1;
                gk++;
            }
        }
    } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        uint32_t compute_data_ready_phase[NS] = {};
        uint32_t tmem_buffer_free_phase = 0;
        long gk = 0;

        TmemTile<float, 128, BN> out_tm(taddr + TMEM_OUT_COL);
        TmemTile<__nv_fp8_e8m0, 128, 16 * K_SCALE_ATOMS * NS>
            a_sc_tm(taddr + TMEM_A_SCALE_COL);
        TmemTile<__nv_fp8_e8m0, 128, 32 * K_SCALE_ATOMS * NS>
            b_sc_tm(taddr + TMEM_B_SCALE_COL);

        for (int ti = 0; ti < num_my; ti++) {
            mbarrier_wait_phase(mbar_tmem_buffer_free, tmem_buffer_free_phase);
            tmem_buffer_free_phase ^= 1;
            tcgen05_fence_after_thread_sync();

            for (int k = 0; k < num_k; k++) {
                int slot = gk % NS;
                constexpr int A_SCALE_COLS_PER_SLOT = 16 * K_SCALE_ATOMS;
                constexpr int B_SCALE_COLS_PER_SLOT = 32 * K_SCALE_ATOMS;
                const int a_scale_slot_col = slot * A_SCALE_COLS_PER_SLOT;
                const int b_scale_slot_col = slot * B_SCALE_COLS_PER_SLOT;
                auto a_sc_stage_0 =
                    a_sc_tm.template subtile<16>(a_scale_slot_col);
                auto a_sc_stage_1 =
                    a_sc_tm.template subtile<16>(a_scale_slot_col + 16);
                auto b_sc_atom_00 =
                    b_sc_tm.template subtile<16>(b_scale_slot_col);
                auto b_sc_atom_01 =
                    b_sc_tm.template subtile<16>(b_scale_slot_col + 16);
                auto b_sc_atom_10 =
                    b_sc_tm.template subtile<16>(b_scale_slot_col + 32);
                auto b_sc_atom_11 =
                    b_sc_tm.template subtile<16>(b_scale_slot_col + 48);
                auto b_sc_stage_0 =
                    b_sc_tm.template subtile<32>(b_scale_slot_col);
                auto b_sc_stage_1 =
                    b_sc_tm.template subtile<32>(b_scale_slot_col + 32);
                uint32_t compute_buffer_free_addr =
                    mbarrier_addr(mbar_compute_buffer_free[slot]);

                mbarrier_wait_phase(
                    mbar_compute_data_ready[slot],
                    compute_data_ready_phase[slot]);
                copy_scale_atom_from_smem_to_tmem(
                    a_sc_stage_0.addr, a_sc_smem[slot][0]);
                copy_scale_atom_from_smem_to_tmem(
                    a_sc_stage_1.addr, a_sc_smem[slot][1]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_00.addr, b_sc_smem[slot][0][0]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_01.addr, b_sc_smem[slot][1][0]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_10.addr, b_sc_smem[slot][0][1]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_11.addr, b_sc_smem[slot][1][1]);

                if (k == 0) {
                    issue_mxfp8_mma2_abt_128x256x256<0>(
                        out_tm, a_smem[slot], b_smem[slot],
                        a_sc_stage_0, b_sc_stage_0,
                        a_sc_stage_1, b_sc_stage_1,
                        compute_buffer_free_addr);
                } else {
                    issue_mxfp8_mma2_abt_128x256x256<1>(
                        out_tm, a_smem[slot], b_smem[slot],
                        a_sc_stage_0, b_sc_stage_0,
                        a_sc_stage_1, b_sc_stage_1,
                        compute_buffer_free_addr);
                }

                compute_data_ready_phase[slot] ^= 1;
                gk++;
            }
            tcgen05_mma_commit_arrive(mbar_tmem_data_ready, cta_mask);
        }
    } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {
        constexpr int ROW_STRIPS    = BM / 32;
        constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
        constexpr int COLS_PER_WARP = BN / COL_GROUPS;
        constexpr int EPI_THREADS   = NUM_WARPS * 32;
        const int ew = warp_id - 4;
        const int row_warp = ew % ROW_STRIPS;
        const int col_warp = ew / ROW_STRIPS;
        const int my_row = row_warp * 32 + lane;
        uint32_t full = 0;
        int store_stage = 0;

        for (int ti = 0; ti < num_my; ti++) {
            int base_m, base_n, local_m, local_n;
            map_off(ti, base_m, base_n, local_m, local_n);
            mbarrier_wait_phase(mbar_tmem_data_ready, full);
            full ^= 1;
            tcgen05_fence_after_thread_sync();

            const uint32_t trow = taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
            constexpr int NUM_STORE_TILES = BN / STORE_N;
            constexpr int LOAD_WIDTH = 32;
            constexpr int LOADS_PER_STORE_TILE = STORE_N / LOAD_WIDTH;
            constexpr int LOADS_PER_WARP =
                LOADS_PER_STORE_TILE / COL_GROUPS;
            static_assert(STORE_N == 64);
            static_assert(COL_GROUPS == 2);
            static_assert(
                LOADS_PER_WARP * COL_GROUPS == LOADS_PER_STORE_TILE);

            float t[NUM_STORE_TILES][LOADS_PER_WARP][LOAD_WIDTH];
            #pragma unroll
            for (int store_tile = 0; store_tile < NUM_STORE_TILES; store_tile++) {
                #pragma unroll
                for (int n = 0; n < LOADS_PER_WARP; n++) {
                    const int local_n = col_warp * LOADS_PER_WARP + n;
                    tcgen05_ld_32x32b_x32(
                        trow + (uint32_t)(
                            store_tile * STORE_N + local_n * LOAD_WIDTH),
                        t[store_tile][n]);
                }
            }
            tcgen05_wait_ld();

            tcgen05_fence_before_thread_sync();
            if (ew == 0 && elect_sync()) {
                mbarrier_arrive_no_tx_cluster_cta0(mbar_tmem_buffer_free);
            }

            #pragma unroll
            for (int store_tile = 0; store_tile < NUM_STORE_TILES; store_tile++) {
                if (ew == 0) {
                    tma_wait_group<TMA_STORE_STAGES - 1>();
                }
                bar_sync<1, EPI_THREADS>();

                #pragma unroll
                for (int n = 0; n < LOADS_PER_WARP; n++) {
                    const int local_n = col_warp * LOADS_PER_WARP + n;
                    #pragma unroll
                    for (int sub = 0; sub < LOAD_WIDTH / 8; sub++) {
                        __nv_bfloat162 pk[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int element = sub * 8 + 2 * i;
                            pk[i] = __floats2bfloat162_rn(
                                mmc_epi(t[store_tile][n][element]),
                                mmc_epi(t[store_tile][n][element + 1]));
                        }
                        const int chunk8 = local_n * (LOAD_WIDTH / 8) + sub;
                        const int swizzled_n = chunk8 ^ (my_row & 7);
                        __nv_bfloat16* write_ptr =
                            &d_smem[store_stage].data[0] +
                            my_row * STORE_N + swizzled_n * 8;
                        *reinterpret_cast<int4*>(write_ptr) =
                            *reinterpret_cast<int4*>(pk);
                    }
                }

                fence_proxy_async_shared_cta();
                bar_sync<1, EPI_THREADS>();

                if (ew == 0 && elect_sync()) {
                    store_bf16_tile_from_smem_to_gmem(
                        &D_tmap, d_smem[store_stage],
                        local_m,
                        (base_n + store_tile * STORE_N) / STORE_N);
                }
                store_stage = (store_stage + 1 == TMA_STORE_STAGES)
                    ? 0
                    : store_stage + 1;
            }
        }
        if (ew == 0) {
            tma_wait_group<0>();
        }
        bar_sync<1, EPI_THREADS>();
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc_g2(taddr, TMEM_ALLOC_COLS);
    }
}

extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(LAUNCH_THREADS, 1)
void matmul_cluster(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap A_sc_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap B_sc_tmap,
    const __grid_constant__ CUtensorMap D_tmap,
    __nv_bfloat16* C_ptr, int M, int N, int K
) {
    matmul_cluster_impl(
        A_tmap, A_sc_tmap, B_tmap, B_sc_tmap, D_tmap,
        C_ptr, M, N, K);
}
