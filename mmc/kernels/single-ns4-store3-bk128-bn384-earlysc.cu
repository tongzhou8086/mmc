#include "cuda-mxfp8.cuh"

// Single-TMEM BN=384 variant, for higher arithmetic intensity per output tile:
// a 128x384 tile does 96 MACs per operand byte loaded against 85.3 for 128x256.
// Built on the early-scale-copy variant, so the MMA warp also starts the next
// tile's scale copies before the epilogue releases the accumulator buffer.
//
// A tcgen05 MMA takes N <= 256, so each K step issues an N=256 MMA followed by
// an N=128 MMA into the next 128 accumulator columns. Both MMAs are cta_group::2
// and split their N across the CTA pair, so B arrives as two cooperative loads:
//
//   the N=256 tile, 128 rows per CTA from global row base_n + rank*128;
//   the N=128 tile,  64 rows per CTA from global row base_n + 256 + rank*64.
//
// The 64-row piece has to be its own tile rather than the tail of a contiguous
// 192-row load: a scale atom covers a fixed 128 N values, and only with this
// split do the N=128 MMA's 128 N values form exactly one atom (CTA 0 supplying
// its low 64 lanes, CTA 1 its high 64). Hence the extra B_tail_tmap argument.
//
// Accumulator columns still map to global N contiguously - the N=256 MMA covers
// [base_n, +256) and the N=128 MMA covers [base_n+256, +384) - so the epilogue
// drains six 64-column store tiles in order with no permutation.
//
// N does not have to be a multiple of BN. The B, B tail and B scale descriptors
// are all encoded with OOB fill NONE, so a partial last N tile reads zeros, and
// the TMA store drops the OOB columns it would write.
//
// The 384-column accumulator does not fit in registers in one pass: 384 columns
// over the 256 epilogue threads is 192 floats/thread, and the per-SM register
// file caps this kernel at 65536/384 = 168 registers/thread. So the epilogue
// loads 192 columns at a time - t[3][1][32] = 96 floats - drains those three
// 64-column store tiles, then repeats, releasing TMEM after the second load.
constexpr int BM           = 128;
constexpr int BN           = 384;
constexpr int BK           = 128;
constexpr int NS           = 4;
constexpr int GROUP_SIZE_M = 16;
constexpr int NUM_WARPS    = 8;
constexpr int CTA_GROUP    = 2;
constexpr int BN_MAIN      = 256;        // N of the first MMA
constexpr int BN_TAIL      = BN - BN_MAIN;
constexpr int BN_LOCAL     = BN_MAIN / CTA_GROUP;
constexpr int BN_LOCAL_TAIL = BN_TAIL / CTA_GROUP;
constexpr int N_SCALE_ATOMS = BN / 128;  // 128 N values per scale atom
constexpr int STORE_N      = 64;
constexpr int LOAD_N       = 192;        // accumulator columns per register load
constexpr int TMA_STORE_STAGES = 3;

constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int LAUNCH_THREADS = (NUM_WARPS + 4) * WARP_SIZE;

using a_tile = SmemTile<__nv_fp8_e4m3, BM, BK, true, 128>;
using b_tile = SmemTile<__nv_fp8_e4m3, BN_LOCAL, BK, true, 128>;
using b_tail_tile = SmemTile<__nv_fp8_e4m3, BN_LOCAL_TAIL, BK, true, 128>;
using d_tile = SmemTile<__nv_bfloat16, BM, STORE_N, true, 128>;

template <
    int Accumulate,
    typename AccumTensor,
    typename AScaleTensor,
    typename BScaleTensor,
    typename BTailScaleTensor>
__device__ __forceinline__ void issue_mxfp8_mma2_abt_128x384x128(
    const AccumTensor& dst,
    const a_tile& a,
    const b_tile& b,
    const b_tail_tile& b_tail,
    const AScaleTensor& a_scale,
    const BScaleTensor& b_scale,
    const BTailScaleTensor& b_tail_scale,
    uint32_t smem_buffer_free_mbarrier_addr
) {
    static_assert(BM == 128 && BN == 384 && BK == 128);
    static_assert(CTA_GROUP == 2 && BN_LOCAL == 128 && BN_LOCAL_TAIL == 64);

    constexpr uint32_t main_idescs[4] = {
        mxfp8_mma2_abt_desc<0, CTA_GROUP * BM, BN_MAIN>(),
        mxfp8_mma2_abt_desc<1, CTA_GROUP * BM, BN_MAIN>(),
        mxfp8_mma2_abt_desc<2, CTA_GROUP * BM, BN_MAIN>(),
        mxfp8_mma2_abt_desc<3, CTA_GROUP * BM, BN_MAIN>(),
    };
    constexpr uint32_t tail_idescs[4] = {
        mxfp8_mma2_abt_desc<0, CTA_GROUP * BM, BN_TAIL>(),
        mxfp8_mma2_abt_desc<1, CTA_GROUP * BM, BN_TAIL>(),
        mxfp8_mma2_abt_desc<2, CTA_GROUP * BM, BN_TAIL>(),
        mxfp8_mma2_abt_desc<3, CTA_GROUP * BM, BN_TAIL>(),
    };
    const uint64_t a_base = make_k_major_operand_smem_desc(a);
    const uint64_t b_base = make_k_major_operand_smem_desc(b);
    const uint64_t b_tail_base = make_k_major_operand_smem_desc(b_tail);
    const uint32_t tail_dst = dst.addr + BN_MAIN;

    // The two N groups write disjoint accumulator columns, so each one needs
    // its own non-accumulating first chunk on the first K step.
    issue_mxfp8_mma2_abt_chunk<Accumulate>(
        dst.addr,
        k_major_operand_chunk_desc<a_tile>(a_base, 0),
        k_major_operand_chunk_desc<b_tile>(b_base, 0),
        a_scale.addr, b_scale.addr, main_idescs[0]);

    #pragma unroll
    for (int sfid = 1; sfid < 4; ++sfid) {
        issue_mxfp8_mma2_abt_chunk<1>(
            dst.addr,
            k_major_operand_chunk_desc<a_tile>(a_base, sfid),
            k_major_operand_chunk_desc<b_tile>(b_base, sfid),
            a_scale.addr, b_scale.addr, main_idescs[sfid]);
    }

    issue_mxfp8_mma2_abt_chunk<Accumulate>(
        tail_dst,
        k_major_operand_chunk_desc<a_tile>(a_base, 0),
        k_major_operand_chunk_desc<b_tail_tile>(b_tail_base, 0),
        a_scale.addr, b_tail_scale.addr, tail_idescs[0]);

    #pragma unroll
    for (int sfid = 1; sfid < 4; ++sfid) {
        issue_mxfp8_mma2_abt_chunk<1>(
            tail_dst,
            k_major_operand_chunk_desc<a_tile>(a_base, sfid),
            k_major_operand_chunk_desc<b_tail_tile>(b_tail_base, sfid),
            a_scale.addr, b_tail_scale.addr, tail_idescs[sfid]);
    }

    commit_mma_arrive_cta_group_2(
        smem_buffer_free_mbarrier_addr,
        static_cast<uint16_t>((1 << CTA_GROUP) - 1));
}

__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap& A_tmap,
    const CUtensorMap& A_sc_tmap,
    const CUtensorMap& B_tmap,
    const CUtensorMap& B_tail_tmap,
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
        prefetch_tma_descriptor(&B_tail_tmap);
        prefetch_tma_descriptor(&B_sc_tmap);
        prefetch_tma_descriptor(&D_tmap);
    }

    extern __shared__ int __shm[];
    SharedMemoryAllocator1024 al((int*)&__shm[0]);

    a_tile (&a_smem)[NS] = al.allocate<a_tile[NS]>();
    b_tile (&b_smem)[NS] = al.allocate<b_tile[NS]>();
    b_tail_tile (&b_tail_smem)[NS] = al.allocate<b_tail_tile[NS]>();
    ScaleAtom (&a_sc_smem)[NS] = al.allocate<ScaleAtom[NS]>();
    ScaleAtom (&b_sc_smem)[NS][N_SCALE_ATOMS] =
        al.allocate<ScaleAtom[NS][N_SCALE_ATOMS]>();
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
    constexpr int TMEM_E8M0_PER_PHYS_COL = 4;
    constexpr int A_SCALE_LOGICAL_COLS_PER_SLOT = 16;
    constexpr int B_SCALE_LOGICAL_COLS_PER_SLOT = 16 * N_SCALE_ATOMS;
    constexpr int TMEM_A_SCALE_COL = BN;
    constexpr int TMEM_B_SCALE_COL =
        TMEM_A_SCALE_COL +
        NS * A_SCALE_LOGICAL_COLS_PER_SLOT / TMEM_E8M0_PER_PHYS_COL;
    static_assert(
        TMEM_B_SCALE_COL +
            NS * B_SCALE_LOGICAL_COLS_PER_SLOT / TMEM_E8M0_PER_PHYS_COL <=
        TMEM_ALLOC_COLS);

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

    auto map_off = [&](int ti, int& base_m, int& base_n, int& local_m,
                       int& local_n, int& local_n_tail) {
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
        local_n_tail = base_n + BN_MAIN + cta_rank * BN_LOCAL_TAIL;
    };

    if (warp_id == 0 && elect_sync()) {
        uint32_t compute_buffer_free_phase[NS] = {};
        long gk = 0;
        for (int ti = 0; ti < num_my; ti++) {
            int base_m, base_n, local_m, local_n, local_n_tail;
            map_off(ti, base_m, base_n, local_m, local_n, local_n_tail);
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
                load_fp8_operand_tile_from_gmem_to_smem(
                    b_tail_smem[slot],
                    &B_tail_tmap,
                    local_n_tail, k,
                    compute_data_ready_cta0,
                    static_cast<uint16_t>(1 << cta_rank));
                load_scale_atom_from_gmem_to_smem(
                    a_sc_smem[slot],
                    &A_sc_tmap,
                    local_m / BM, k,
                    compute_data_ready_cta0,
                    static_cast<uint16_t>(1 << cta_rank));

                // Every CTA of the pair runs tcgen05.cp against its own SMEM,
                // so each one needs all N_SCALE_ATOMS B scale atoms locally.
                #pragma unroll
                for (int atom = 0; atom < N_SCALE_ATOMS; ++atom) {
                    load_scale_atom_from_gmem_to_smem(
                        b_sc_smem[slot][atom],
                        &B_sc_tmap,
                        base_n / 128 + atom, k,
                        compute_data_ready_cta0,
                        static_cast<uint16_t>(1 << cta_rank));
                }

                constexpr int slot_bytes =
                    sizeof(a_tile) + sizeof(b_tile) + sizeof(b_tail_tile) +
                    sizeof(ScaleAtom) * (1 + N_SCALE_ATOMS);
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
        TmemTile<__nv_fp8_e8m0, 128, A_SCALE_LOGICAL_COLS_PER_SLOT * NS>
            a_sc_tm(taddr + TMEM_A_SCALE_COL);
        TmemTile<__nv_fp8_e8m0, 128, B_SCALE_LOGICAL_COLS_PER_SLOT * NS>
            b_sc_tm(taddr + TMEM_B_SCALE_COL);

        for (int ti = 0; ti < num_my; ti++) {
            for (int k = 0; k < num_k; k++) {
                int slot = gk % NS;
                constexpr int B_SC_COLS = B_SCALE_LOGICAL_COLS_PER_SLOT;
                const int b_sc_slot_col = slot * B_SC_COLS;
                auto a_sc_stage = a_sc_tm.template subtile<16>(slot * 16);
                auto b_sc_atom_0 =
                    b_sc_tm.template subtile<16>(b_sc_slot_col);
                auto b_sc_atom_1 =
                    b_sc_tm.template subtile<16>(b_sc_slot_col + 16);
                auto b_sc_atom_2 =
                    b_sc_tm.template subtile<16>(b_sc_slot_col + 32);
                // The N=256 MMA reads a 2-atom (256 N) scale block; the N=128
                // MMA reads the third atom on its own.
                auto b_sc_stage = b_sc_tm.template subtile<32>(b_sc_slot_col);
                auto b_sc_tail_stage = b_sc_atom_2;
                uint32_t compute_buffer_free_addr =
                    mbarrier_addr(mbar_compute_buffer_free[slot]);

                mbarrier_wait_phase(
                    mbar_compute_data_ready[slot],
                    compute_data_ready_phase[slot]);
                copy_scale_atom_from_smem_to_tmem(
                    a_sc_stage.addr, a_sc_smem[slot]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_0.addr, b_sc_smem[slot][0]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_1.addr, b_sc_smem[slot][1]);
                copy_scale_atom_from_smem_to_tmem(
                    b_sc_atom_2.addr, b_sc_smem[slot][2]);

                if (k == 0) {
                    // The scale copies above only write the dedicated scale
                    // columns, so they are already in flight by the time the
                    // epilogue releases the accumulator buffer.
                    mbarrier_wait_phase(
                        mbar_tmem_buffer_free, tmem_buffer_free_phase);
                    tmem_buffer_free_phase ^= 1;
                    tcgen05_fence_after_thread_sync();
                    issue_mxfp8_mma2_abt_128x384x128<0>(
                        out_tm, a_smem[slot], b_smem[slot], b_tail_smem[slot],
                        a_sc_stage, b_sc_stage, b_sc_tail_stage,
                        compute_buffer_free_addr);
                } else {
                    issue_mxfp8_mma2_abt_128x384x128<1>(
                        out_tm, a_smem[slot], b_smem[slot], b_tail_smem[slot],
                        a_sc_stage, b_sc_stage, b_sc_tail_stage,
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
            int base_m, base_n, local_m, local_n, local_n_tail;
            map_off(ti, base_m, base_n, local_m, local_n, local_n_tail);
            mbarrier_wait_phase(mbar_tmem_data_ready, full);
            full ^= 1;
            tcgen05_fence_after_thread_sync();

            const uint32_t trow = taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
            constexpr int NUM_STORE_TILES = BN / STORE_N;
            constexpr int LOAD_WIDTH = 32;
            constexpr int LOADS_PER_STORE_TILE = STORE_N / LOAD_WIDTH;
            constexpr int LOADS_PER_WARP =
                LOADS_PER_STORE_TILE / COL_GROUPS;
            constexpr int STORE_TILES_PER_LOAD = LOAD_N / STORE_N;
            constexpr int NUM_LOADS = BN / LOAD_N;
            static_assert(STORE_N == 64);
            static_assert(COL_GROUPS == 2);
            static_assert(NUM_LOADS * STORE_TILES_PER_LOAD == NUM_STORE_TILES);
            static_assert(
                LOADS_PER_WARP * COL_GROUPS == LOADS_PER_STORE_TILE);

            #pragma unroll
            for (int load = 0; load < NUM_LOADS; load++) {
                float t[STORE_TILES_PER_LOAD][LOADS_PER_WARP][LOAD_WIDTH];
                #pragma unroll
                for (int tile = 0; tile < STORE_TILES_PER_LOAD; tile++) {
                    #pragma unroll
                    for (int n = 0; n < LOADS_PER_WARP; n++) {
                        const int local_n = col_warp * LOADS_PER_WARP + n;
                        const int column = load * LOAD_N + tile * STORE_N +
                            local_n * LOAD_WIDTH;
                        tcgen05_ld_32x32b_x32(
                            trow + (uint32_t)column, t[tile][n]);
                    }
                }
                tcgen05_wait_ld();

                if (load == NUM_LOADS - 1) {
                    tcgen05_fence_before_thread_sync();
                    if (ew == 0 && elect_sync()) {
                        mbarrier_arrive_no_tx_cluster_cta0(mbar_tmem_buffer_free);
                    }
                }

                #pragma unroll
                for (int tile = 0; tile < STORE_TILES_PER_LOAD; tile++) {
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
                                    mmc_epi(t[tile][n][element]),
                                    mmc_epi(t[tile][n][element + 1]));
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
                        const int store_tile =
                            load * STORE_TILES_PER_LOAD + tile;
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
    const __grid_constant__ CUtensorMap B_tail_tmap,
    const __grid_constant__ CUtensorMap B_sc_tmap,
    const __grid_constant__ CUtensorMap D_tmap,
    __nv_bfloat16* C_ptr, int M, int N, int K
) {
    matmul_cluster_impl(
        A_tmap, A_sc_tmap, B_tmap, B_tail_tmap, B_sc_tmap, D_tmap,
        C_ptr, M, N, K);
}
