"""Derive the cluster-launch-control kernels from the design-4 (2round) sources.

    python tools/make_clc_kernels.py

Design 4 hands each persistent cluster a fixed slice of the tile list
(`cluster_pid + ti * cluster_stride`). This script rewrites that static
partition into a dynamic one: the grid is launched with one cluster per output
tile, every cluster starts on its own blockIdx tile, and a scheduler warp then
asks the hardware to cancel a not-yet-launched cluster and inherit its tile
(`clusterlaunchcontrol.try_cancel`). The tile ids come back in launch order, so
the GSM swizzle - and the L2 locality it buys - is untouched.

Writes {bk64 NS=6, bk128 NS=3} x claim depth {1, 2, 3} x GSM {8, 12, 16},
named `...-2round-clcD[-gsmNN].cu`, and leaves the originals alone. Depth 1 is
the one to use: see benchmarks/clc-results.md.
"""

from pathlib import Path

KERNELS = Path(__file__).resolve().parent.parent / "mmc" / "kernels"
BASES = {
    64: "bf16-double-ns6-store2-bk64-bn512-2round",
    128: "bf16-double-ns3-store2-bk128-bn512-2round",
}
GSMS = (8, 12, 16)
# try_cancel does not merely fetch a tile id, it *claims* the tile, so the ring
# depth is how many tiles a cluster claims beyond the one it is running. Deep
# rings hide the response latency but let the clusters that start first grab
# work the others will sit idle for - which is exactly the imbalance CLC exists
# to remove. Depth is therefore a swept knob, not a constant.
DEPTHS = (1, 2, 3)


# ── the CLC device helpers, inserted after the mbarrier ones ────────────
HELPERS = '''
// ── Cluster launch control (Blackwell) ──────────────────────────────
//
// try_cancel asks the hardware to cancel one cluster that has not been
// launched yet; if it succeeds, the response carries that cluster's first
// CTA id and this cluster runs its tile. The response is 16 bytes, written by
// the async proxy into SMEM and multicast to every CTA of the cluster, and its
// arrival completes the transaction count of an mbarrier - exactly like a TMA
// load, so the same arrive/expect_tx idiom applies.
__device__ __forceinline__ void clc_try_cancel(uint32_t resp_smem, uint32_t mb) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta"
        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
        :: "r"(resp_smem), "r"(mb) : "memory");
}

// Decodes a response that has already been waited on. Returns the cancelled
// cluster's first CTA id, or -1 when the grid is drained.
__device__ __forceinline__ int clc_first_ctaid(uint32_t resp_smem) {
    uint32_t ok, x;
    asm volatile(
        "{\\n\\t"
        ".reg .pred P;\\n\\t"
        ".reg .b128 R;\\n\\t"
        ".reg .b32 cy, cz;\\n\\t"
        "ld.shared.b128 R, [%2];\\n\\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P, R;\\n\\t"
        "selp.u32 %0, 1, 0, P;\\n\\t"
        "mov.u32 %1, 0;\\n\\t"
        "@P clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
        "{%1, cy, cz, _}, R;\\n\\t"
        "}"
        : "=r"(ok), "=r"(x) : "r"(resp_smem) : "memory");
    return ok ? (int)x : -1;
}
'''

# ── knobs ───────────────────────────────────────────────────────────────
OLD_KNOB = "constexpr int TWO_CTA          = 1;  // 1 = 2-CTA cluster MMA (cta_group::2); 0 = single-CTA"
NEW_KNOB = OLD_KNOB + """
constexpr int CLC_DEPTH        = 3;  // tiles claimed ahead (see DEPTHS)"""

# ── launch-contract comment ─────────────────────────────────────────────
OLD_CONTRACT = "//   grid    = sm_count - sm_count % CTA_GROUP   (persistent, EPILOGUE_OVERLAP)"
NEW_CONTRACT = """//   grid    = CTA_GROUP * ceil(M/(CTA_GROUP*BM)) * ceil(N/BN)   (one cluster
//             per output tile; cluster launch control redistributes them)"""

# ── static partition -> home tile ───────────────────────────────────────
OLD_PARTITION = """        const int num_clusters = grid_m_clusters * grid_n;
        const int cluster_pid = (int)blockIdx.x / CTA_GROUP;
        const int cluster_stride = (int)gridDim.x / CTA_GROUP;
        const int num_my = (cluster_pid >= num_clusters) ? 0
                         : (num_clusters - cluster_pid + cluster_stride - 1) / cluster_stride;

        auto map_off = [&](int ti, int& base_m, int& base_n, int& local_m, int& local_n) {
            int tile = cluster_pid + ti * cluster_stride;
            int group = tile / num_cluster_in_group;"""
NEW_PARTITION = """        // One cluster per output tile, so blockIdx alone is the first tile;
        // every later tile is one this cluster cancelled out of the grid.
        const int home_tile = (int)blockIdx.x / CTA_GROUP;

        auto map_off = [&](int tile, int& base_m, int& base_n, int& local_m, int& local_n) {
            int group = tile / num_cluster_in_group;"""

# ── the CLC ring, declared next to the other cluster barriers ───────────
OLD_BARRIERS = """        __shared__ uint64_t mbar_tmem_data_ready[2];
        // One accumulator-free barrier per 256-column MMA panel.
        __shared__ uint64_t mbar_tmem_panel_free[2];"""
NEW_BARRIERS = """        __shared__ uint64_t mbar_tmem_data_ready[2];
        // One accumulator-free barrier per 256-column MMA panel.
        __shared__ uint64_t mbar_tmem_panel_free[2];
        // CLC ring. mbar_clc_full is completed by the try_cancel response,
        // mbar_clc_tile_ready publishes the decoded tile id to this CTA's
        // consumers, and mbar_clc_slot_free - which lives on CTA 0 and collects
        // arrivals from both CTAs - paces the scheduler so a new response never
        // lands on a slot a peer is still reading.
        __shared__ __align__(16) uint32_t clc_resp[CLC_DEPTH][4];
        __shared__ uint64_t mbar_clc_full[CLC_DEPTH];
        __shared__ uint64_t mbar_clc_tile_ready[CLC_DEPTH];
        __shared__ uint64_t mbar_clc_slot_free[CLC_DEPTH];
        __shared__ int clc_tile[CLC_DEPTH];
        // per CTA: the TMA warp plus one leader per epilogue warp, and the MMA
        // warp on CTA 0 only
        constexpr int CLC_CONSUMERS = 2 * (1 + NUM_WARPS) + 1;"""

# ── init + pre-arm ──────────────────────────────────────────────────────
OLD_INIT = """            #pragma unroll
            for (int b = 0; b < 2; b++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[b]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_panel_free[b]), CTA_GROUP);
                mbarrier_arrive_no_tx_cluster_cta0(mbar_tmem_panel_free[b]);
            }"""
NEW_INIT = OLD_INIT + """
            #pragma unroll
            for (int i = 0; i < CLC_DEPTH; i++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_clc_full[i]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_clc_tile_ready[i]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_clc_slot_free[i]),
                              CLC_CONSUMERS);
                // arm the first CLC_DEPTH responses: the arrive satisfies the
                // count, the response completes the 16 transaction bytes
                signal_on_bytes_loaded(
                    (uint32_t)__cvta_generic_to_shared(&mbar_clc_full[i]), 16);
            }"""

# ── consumer helper, next to map_off ────────────────────────────────────
OLD_MAPOFF_TAIL = """            local_m = base_m + cta_rank * BM;
            local_n = base_n + cta_rank * BN_LOCAL;
        };
"""
NEW_MAPOFF_TAIL = OLD_MAPOFF_TAIL + """
        // Consume work item `item` of the CLC ring: wait for the scheduler to
        // publish it, read the tile id, then release the slot. Returns -1 when
        // the grid is drained, which is every consumer's loop exit.
        // `leader_only` is for the epilogue warps, where all 32 lanes read the
        // id but only the leader arrives.
        auto clc_take = [&](int item, bool warp_wide) -> int {
            const int slot = item % CLC_DEPTH;
            wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_clc_tile_ready[slot]),
                       (uint32_t)((item / CLC_DEPTH) & 1));
            const int t = clc_tile[slot];
            if (warp_wide) {
                __syncwarp();
                if (lane == 0) mbarrier_arrive_no_tx_cluster_cta0(mbar_clc_slot_free[slot]);
            } else {
                mbarrier_arrive_no_tx_cluster_cta0(mbar_clc_slot_free[slot]);
            }
            return t;
        };
"""

# ── the three consumer loops ────────────────────────────────────────────
OLD_TMA_LOOP = """            uint32_t compute_buffer_free_phase[NS] = {};
            int slot_base = 0;          // continues across output tiles
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                map_off(ti, base_m, base_n, local_m, local_n);"""
NEW_TMA_LOOP = """            uint32_t compute_buffer_free_phase[NS] = {};
            int slot_base = 0;          // continues across output tiles
            int my_tile = home_tile;
            for (int ti = 0; ; ti++) {
                if (ti > 0) {
                    my_tile = clc_take(ti - 1, /*warp_wide=*/false);
                    if (my_tile < 0) break;
                }
                int base_m, base_n, local_m, local_n;
                map_off(my_tile, base_m, base_n, local_m, local_n);"""

OLD_MMA_LOOP = """            int slot_base = 0;          // continues across output tiles
            for (int ti = 0; ti < num_my; ti++) {
                uint32_t d_tmem = taddr;"""
NEW_MMA_LOOP = """            int slot_base = 0;          // continues across output tiles
            // The MMA warp never needs the tile coordinates - only the count -
            // but it still walks the ring so its slot releases stay in step.
            for (int ti = 0; ; ti++) {
                if (ti > 0 && clc_take(ti - 1, /*warp_wide=*/false) < 0) break;
                uint32_t d_tmem = taddr;"""

OLD_EPI_LOOP = """            uint32_t full[2] = {};
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                const int buf = 0;
                map_off(ti, base_m, base_n, local_m, local_n);"""
NEW_EPI_LOOP = """            uint32_t full[2] = {};
            int my_tile = home_tile;
            for (int ti = 0; ; ti++) {
                if (ti > 0) {
                    my_tile = clc_take(ti - 1, /*warp_wide=*/true);
                    if (my_tile < 0) break;
                }
                int base_m, base_n, local_m, local_n;
                const int buf = 0;
                map_off(my_tile, base_m, base_n, local_m, local_n);"""

# ── the scheduler warp, spliced in front of the epilogue branch ─────────
OLD_EPI_BRANCH = """        } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {"""
NEW_EPI_BRANCH = """        } else if (warp_id == 2 && elect_sync()) {
            // CLC scheduler. Both CTAs re-arm their own mbar_clc_full and
            // decode the multicast response; only CTA 0 issues try_cancel, and
            // only it paces on mbar_clc_slot_free, which carries arrivals from
            // the whole cluster. Both ranks re-arm BEFORE publishing the tile,
            // so a consumer arrival - and therefore CTA 0's next try_cancel -
            // can never overtake CTA 1's expect_tx for that slot.
            for (int item = 0; ; item++) {
                const int slot = item % CLC_DEPTH;
                const uint32_t full_mb =
                    (uint32_t)__cvta_generic_to_shared(&mbar_clc_full[slot]);
                if (cta_rank == 0) {
                    if (item >= CLC_DEPTH)
                        wait_phase(
                            (uint32_t)__cvta_generic_to_shared(&mbar_clc_slot_free[slot]),
                            (uint32_t)(((item - CLC_DEPTH) / CLC_DEPTH) & 1));
                    clc_try_cancel(
                        (uint32_t)__cvta_generic_to_shared(&clc_resp[slot][0]), full_mb);
                }
                wait_phase(full_mb, (uint32_t)((item / CLC_DEPTH) & 1));
                const int ctaid = clc_first_ctaid(
                    (uint32_t)__cvta_generic_to_shared(&clc_resp[slot][0]));
                signal_on_bytes_loaded(full_mb, 16);       // arm use item+CLC_DEPTH
                clc_tile[slot] = (ctaid < 0) ? -1 : (ctaid / CTA_GROUP);
                mbarrier_arrive_no_tx(
                    (uint32_t)__cvta_generic_to_shared(&mbar_clc_tile_ready[slot]));
                if (ctaid < 0) break;
            }
        } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {"""

# ── the file-header note ────────────────────────────────────────────────
HEADER_ANCHOR = "// ── User-tunable constants (the webui substitutes these) ────────────"
HEADER_NOTE = """// ── Cluster launch control ──────────────────────────────────────────
//
// Same pipeline as the -2round design; only the work distribution differs.
// That design launches one persistent cluster per SM pair and gives each a
// static stride through the tile list, so the last wave is ragged whenever the
// tile count is not a multiple of the cluster count, and a cluster that runs
// long holds tiles a free SM could have taken. Here the grid is the full tile
// grid and each cluster, after its own blockIdx tile, calls
// clusterlaunchcontrol.try_cancel to take a tile from a cluster that has not
// launched yet - dynamic scheduling in hardware, in launch order, so the GSM
// swizzle still decides which tiles are co-resident.
//
""" + HEADER_ANCHOR

EDITS = [
    (HEADER_ANCHOR, HEADER_NOTE),
    (OLD_CONTRACT, NEW_CONTRACT),
    (OLD_KNOB, NEW_KNOB),
    (OLD_BARRIERS, NEW_BARRIERS),
    (OLD_INIT, NEW_INIT),
    (OLD_PARTITION, NEW_PARTITION),
    (OLD_MAPOFF_TAIL, NEW_MAPOFF_TAIL),
    (OLD_TMA_LOOP, NEW_TMA_LOOP),
    (OLD_MMA_LOOP, NEW_MMA_LOOP),
    (OLD_EPI_LOOP, NEW_EPI_LOOP),
    (OLD_EPI_BRANCH, NEW_EPI_BRANCH),
]


def clc_source(base_text):
    text = base_text
    # helpers go right after the last mbarrier helper
    anchor = "__device__ __forceinline__ void wait_phase(uint32_t mb, uint32_t phase) {"
    assert anchor in text
    text = text.replace(anchor, HELPERS.lstrip("\n") + "\n" + anchor, 1)
    for old, new in EDITS:
        assert old in text, old[:60]
        assert text.count(old) == 1, f"ambiguous anchor: {old[:60]}"
        text = text.replace(old, new, 1)
    return text


def main():
    written = []
    for bk, stem in BASES.items():
        base = (KERNELS / f"{stem}.cu").read_text()
        clc = clc_source(base)
        for depth in DEPTHS:
            for gsm in GSMS:
                body = clc.replace(
                    "constexpr int GROUP_SIZE_M = 8;",
                    f"constexpr int GROUP_SIZE_M = {gsm};", 1)
                body = body.replace(
                    "constexpr int CLC_DEPTH        = 3;",
                    f"constexpr int CLC_DEPTH        = {depth};", 1)
                tag = f"-clc{depth}"
                name = f"{stem}{tag}" + ("" if gsm == 8 else f"-gsm{gsm}")
                (KERNELS / f"{name}.cu").write_text(body)
                written.append(name)
    for name in written:
        print(f"wrote mmc/kernels/{name}.cu")


if __name__ == "__main__":
    main()
