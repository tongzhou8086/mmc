#include "tk-common.cuh"

using tk_16384 = mxfp8_gemm::config<256, 4, 8, 8, 2, false>;

extern "C" void* tk_create_16384(
    void* A,
    void* A_sc,
    void* B,
    void* B_sc,
    void* D,
    size_t M,
    size_t N,
    size_t K
) {
    return create_tk_launch_state<tk_16384>(A, A_sc, B, B_sc, D, M, N, K);
}

extern "C" int tk_launch_16384(void* launch_state, void* stream) {
    return launch_tk_state<tk_16384>(launch_state, stream);
}

extern "C" void tk_destroy_16384(void* launch_state) {
    destroy_tk_launch_state<tk_16384>(launch_state);
}
