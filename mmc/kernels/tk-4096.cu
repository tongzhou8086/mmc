// Host adapter around ThunderKittens' official B200 MXFP8 GEMM config.
// TK constructs its own globals, TMA descriptors, launch attributes, and
// dynamic-SMEM configuration. Python creates persistent launch state and
// invokes it through this C ABI.

#define main thunderkittens_mxfp8_standalone_main
#include "mxfp8_b200_gemm.cu"
#undef main

using tk_4096 = mxfp8_gemm::config<256, 5, 8, 8, 2, false>;

template <typename C>
struct TkLaunchState {
    using G = mxfp8_gemm::globals<C>;
    G globals;

    TkLaunchState(
        __nv_fp8_e4m3* A,
        __nv_fp8_e8m0* A_sc,
        __nv_fp8_e4m3* B,
        __nv_fp8_e8m0* B_sc,
        __nv_bfloat16* D,
        size_t M,
        size_t N,
        size_t K
    ) : globals{
        typename G::A_gl{A, nullptr, nullptr, M, K},
        typename G::A_sc_gl{A_sc, M / 128, K / 128, nullptr, nullptr},
        typename G::B_gl{B, nullptr, nullptr, N, K},
        typename G::B_sc_gl{B_sc, N / 128, K / 128, nullptr, nullptr},
        typename G::D_gl{D, nullptr, nullptr, M, N}
    } {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_entrypoint<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            globals.dynamic_shared_memory()
        ));
    }

    cudaError_t launch(cudaStream_t stream) {
        LaunchConfig<true, true> config(
            globals.grid(),
            globals.block(),
            globals.dynamic_shared_memory(),
            stream,
            C::CLUSTER_SIZE
        );
        return cudaLaunchKernelEx(config, kernel_entrypoint<C>, globals);
    }
};

extern "C" void* tk_create_4096(
    void* A,
    void* A_sc,
    void* B,
    void* B_sc,
    void* D,
    size_t M,
    size_t N,
    size_t K
) {
    return new TkLaunchState<tk_4096>(
        static_cast<__nv_fp8_e4m3*>(A),
        static_cast<__nv_fp8_e8m0*>(A_sc),
        static_cast<__nv_fp8_e4m3*>(B),
        static_cast<__nv_fp8_e8m0*>(B_sc),
        static_cast<__nv_bfloat16*>(D),
        M,
        N,
        K
    );
}

extern "C" int tk_launch_4096(void* launch_state, void* stream) {
    return static_cast<int>(
        static_cast<TkLaunchState<tk_4096>*>(launch_state)->launch(
            static_cast<cudaStream_t>(stream)
        )
    );
}

extern "C" void tk_destroy_4096(void* launch_state) {
    delete static_cast<TkLaunchState<tk_4096>*>(launch_state);
}
