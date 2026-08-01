#include "tk-common.cuh"

using tk_2048 = mxfp8_gemm::config<256, 5, 8, 12, 2, true>;

extern "C" void* tk_create_2048(
    void* A,
    void* A_sc,
    void* B,
    void* B_sc,
    void* D,
    size_t M,
    size_t N,
    size_t K
) {
    return create_tk_launch_state<tk_2048>(A, A_sc, B, B_sc, D, M, N, K);
}

extern "C" int tk_launch_2048(void* launch_state, void* stream) {
    return launch_tk_state<tk_2048>(launch_state, stream);
}

extern "C" void tk_destroy_2048(void* launch_state) {
    destroy_tk_launch_state<tk_2048>(launch_state);
}
