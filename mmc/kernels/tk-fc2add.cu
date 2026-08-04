// Host adapter around the meshy-research tcgen05_kernels F4 kernel
// `fc2_mxfp8_add_matrix_fwd` (Y = A @ B.T + C, MXFP8 ABt with an optional BF16
// residual add). MMC uses the no-residual specialization, so C is always null
// here and the HAS_C=false kernel elides the residual load and add entirely.
//
// The kernel's operand contract already matches MMC's: A[M,K] E4M3 row-major,
// B[N,K] E4M3 (the transposed RHS for ABt), scale factors [M/128, K/128, 32, 16]
// and [N/128, K/128, 32, 16] in the NVIDIA tcgen05 swizzle, BF16 [M,N] output.
// The kernel names those dimensions F=K and D=N.
//
// The upstream ABI (tk_launch_fc2_mxfp8_add) is stateless and takes ten
// arguments, so it is not directly callable through MMC's TK backend, which
// expects the create/launch/destroy triple with prebound pointers. This file
// bridges the two: it holds the pointers and extents in launch state and calls
// the kernel's internal launcher, which lets MMC's existing `backend="tk"` path
// in _runtime.py drive it unchanged.
//
// Build (TCGEN05 = the tcgen05_kernels package root, TK = its fetched
// ThunderKittens checkout under build/*/_deps/thunderkittens-src):
//
//   nvcc -shared -std=c++20 -O3 --use_fast_math --expt-extended-lambda \
//     --expt-relaxed-constexpr -forward-unknown-to-host-compiler \
//     -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
//     -Xcompiler=-fPIC -Xptxas=--warn-on-spills -DNDEBUG \
//     -DKITTENS_SM100 -gencode arch=compute_100a,code=sm_100a \
//     -I $TK/include -I $TK/prototype \
//     -I $TCGEN05/csrc -I $TCGEN05/csrc/kernels \
//     -I $CUDA_HOME/include/cccl \
//     mmc/kernels/tk-fc2add.cu \
//     -L $CUDA_HOME/lib64/stubs -lcudart -lcuda \
//     -o mmc/cubins/tk-fc2add.so

#include <cassert>
#include <stdexcept>

#include "kittens.cuh"

#include "kernels/tk_fc2_mxfp8_add.cuh"

namespace {

struct Fc2AddLaunchState {
    const __nv_fp8_e4m3* A;
    const __nv_fp8_e8m0* A_sc;
    const __nv_fp8_e4m3* B;
    const __nv_fp8_e8m0* B_sc;
    __nv_bfloat16* D;
    int m;
    int n;
    int k;
};

}  // namespace

extern "C" void* tk_create_fc2add(
    void* A,
    void* A_sc,
    void* B,
    void* B_sc,
    void* D,
    size_t M,
    size_t N,
    size_t K
) {
    // The kernel requires K%128 == 0 and N%128 == 0; MMC's public shape contract
    // (M%256 == N%256 == 0, K%128 == 0) is strictly stronger.
    if (K % 128 != 0 || N % 128 != 0 || M == 0) {
        return nullptr;
    }
    return new Fc2AddLaunchState{
        static_cast<const __nv_fp8_e4m3*>(A),
        static_cast<const __nv_fp8_e8m0*>(A_sc),
        static_cast<const __nv_fp8_e4m3*>(B),
        static_cast<const __nv_fp8_e8m0*>(B_sc),
        static_cast<__nv_bfloat16*>(D),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
    };
}

extern "C" int tk_launch_fc2add(void* launch_state, void* stream) {
    if (launch_state == nullptr) {
        return -1;
    }
    const auto& s = *static_cast<Fc2AddLaunchState*>(launch_state);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    using namespace fc2_mxfp8_add;
    try {
        // Nb=256 whenever N allows it, matching the upstream dispatch; C is null
        // so only the HAS_C=false specializations are instantiated.
        if (s.n % 256 == 0) {
            launch_fc2_mxfp8_add<cfg_nb256, false>(
                s.A, s.A_sc, s.B, s.B_sc, nullptr, s.D,
                s.m, s.m, s.n, s.k, cuda_stream);
        } else {
            launch_fc2_mxfp8_add<cfg_nb128, false>(
                s.A, s.A_sc, s.B, s.B_sc, nullptr, s.D,
                s.m, s.m, s.n, s.k, cuda_stream);
        }
    } catch (const std::exception&) {
        return -5;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" void tk_destroy_fc2add(void* launch_state) {
    delete static_cast<Fc2AddLaunchState*>(launch_state);
}
