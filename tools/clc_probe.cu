// Does clusterlaunchcontrol actually hand out work? A full-grid launch is
// correct whether or not try_cancel ever succeeds - every tile has a home
// cluster - so the mechanism has to be checked on its own.
//
//   nvcc -arch=sm_100a -o clc_probe tools/clc_probe.cu && ./clc_probe
//
// Prints how many clusters the device actually launched and the distribution of
// tiles per cluster. If CLC works, launched << grid and clusters take several
// tiles each; if it silently fails, launched == grid and every count is 1.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int CTA_GROUP = 2;
__device__ __constant__ int CLC_DEPTH_D;
static int CLC_DEPTH_H = 3;
constexpr int MAX_DEPTH = 8;

__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "elect.sync _|P, 0xffffffff;\n\t"
                 "selp.u32 %0, 1, 0, P;\n\t}"
                 : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void signal_on_bytes_loaded(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                 "@P bra.uni DONE_%=;\n\t"
                 "bra.uni WAIT_%=;\n\t"
                 "DONE_%=:\n\t}" :: "r"(mb), "r"(phase) : "memory");
}
__device__ __forceinline__ void clc_try_cancel(uint32_t resp_smem, uint32_t mb) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta"
        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
        :: "r"(resp_smem), "r"(mb) : "memory");
}
__device__ __forceinline__ int clc_first_ctaid(uint32_t resp_smem) {
    uint32_t ok, x;
    asm volatile(
        "{\n\t.reg .pred P;\n\t.reg .b128 R;\n\t.reg .b32 cy, cz;\n\t"
        "ld.shared.b128 R, [%2];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P, R;\n\t"
        "selp.u32 %0, 1, 0, P;\n\t"
        "mov.u32 %1, 0;\n\t"
        "@P clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
        "{%1, cy, cz, _}, R;\n\t}"
        : "=r"(ok), "=r"(x) : "r"(resp_smem) : "memory");
    return ok ? (int)x : -1;
}

// tiles_per_cluster[c] = how many tiles the cluster whose home tile is c ran.
// owner[t] = which home cluster ended up running tile t (each must be hit once).
__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(128, 1)
void probe(int *tiles_per_cluster, int *owner, int *launched) {
    __shared__ __align__(16) uint32_t clc_resp[MAX_DEPTH][4];
    __shared__ uint64_t mbar_full[MAX_DEPTH];
    const int depth = CLC_DEPTH_D;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int home = (int)blockIdx.x / CTA_GROUP;

    if (threadIdx.x == 0) {
        for (int i = 0; i < depth; i++) {
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_full[i]), 1);
            signal_on_bytes_loaded(
                (uint32_t)__cvta_generic_to_shared(&mbar_full[i]), 16);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (cta_rank == 0 && threadIdx.x == 0) atomicAdd(launched, 1);

    if (threadIdx.x == 0 && elect_sync()) {
        int tile = home, count = 0;
        for (int item = 0; ; item++) {
            if (cta_rank == 0) {
                atomicAdd(&owner[tile], 1);
                count++;
            }
            // simulate a tile's worth of work so clusters finish at different times
            __nanosleep(1000 * (1 + (tile % 7)));

            const int slot = item % depth;
            const uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar_full[slot]);
            if (cta_rank == 0)
                clc_try_cancel(
                    (uint32_t)__cvta_generic_to_shared(&clc_resp[slot][0]), mb);
            wait_phase(mb, (uint32_t)((item / depth) & 1));
            const int ctaid = clc_first_ctaid(
                (uint32_t)__cvta_generic_to_shared(&clc_resp[slot][0]));
            signal_on_bytes_loaded(mb, 16);
            if (ctaid < 0) break;
            tile = ctaid / CTA_GROUP;
        }
        if (cta_rank == 0) tiles_per_cluster[home] = count;
    }
}

int main(int argc, char **argv) {
    const int tiles_arg = argc > 1 ? atoi(argv[1]) : 1000;
    CLC_DEPTH_H = argc > 2 ? atoi(argv[2]) : 3;
    cudaMemcpyToSymbol(CLC_DEPTH_D, &CLC_DEPTH_H, sizeof(int));
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int tiles = tiles_arg;                  // pretend output tiles
    int *d_counts, *d_owner, *d_launched;
    cudaMalloc(&d_counts, tiles * sizeof(int));
    cudaMalloc(&d_owner, tiles * sizeof(int));
    cudaMalloc(&d_launched, sizeof(int));
    cudaMemset(d_counts, 0, tiles * sizeof(int));
    cudaMemset(d_owner, 0, tiles * sizeof(int));
    cudaMemset(d_launched, 0, sizeof(int));

    probe<<<tiles * CTA_GROUP, 128>>>(d_counts, d_owner, d_launched);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int *counts = new int[tiles], *owner = new int[tiles], launched = 0;
    cudaMemcpy(counts, d_counts, tiles * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(owner, d_owner, tiles * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&launched, d_launched, sizeof(int), cudaMemcpyDeviceToHost);

    int total = 0, active = 0, maxc = 0, missing = 0, dup = 0;
    for (int i = 0; i < tiles; i++) {
        total += counts[i];
        if (counts[i] > 0) active++;
        if (counts[i] > maxc) maxc = counts[i];
        if (owner[i] == 0) missing++;
        if (owner[i] > 1) dup++;
    }
    printf("depth               %d\n", CLC_DEPTH_H);
    printf("SMs                 %d  (=> %d cluster slots)\n", sm_count, sm_count / 2);
    printf("grid                %d clusters\n", tiles);
    printf("clusters launched   %d\n", launched);
    printf("clusters that ran   %d\n", active);
    printf("tiles executed      %d  (expected %d)\n", total, tiles);
    printf("max tiles/cluster   %d\n", maxc);
    int hist[16] = {0};
    for (int i = 0; i < tiles; i++)
        if (counts[i] > 0) hist[counts[i] < 15 ? counts[i] : 15]++;
    printf("tiles/cluster hist  ");
    for (int i = 1; i <= maxc && i < 16; i++) printf("%dx%d ", hist[i], i);
    printf("\n");
    printf("tiles never run     %d\n", missing);
    printf("tiles run twice+    %d\n", dup);
    printf("%s\n", (total == tiles && missing == 0 && dup == 0)
           ? "PASS: every tile ran exactly once"
           : "FAIL: tile coverage is wrong");
    return 0;
}
