#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <algorithm>   // std::min (the grid y cap, 2026-10-31)
#include <cstdint>     // uintptr_t (the alignment check, 2026-10-31)
#include "device/CudaKernelScope.h"
#include "profiler/RecordFunction.h"

// =================================================================================
// SM120 (Blackwell RTX: RTX 50-series, RTX 6000 Blackwell) Optimized SwiGLU
//
// Layout matches the library convention used by launch_swiglu_generic:
//   input  : [rows, 2 * hidden]  (gate half | up half, contiguous per row)
//   output : [rows, hidden]
//   out[r, c] = swish(in[r, c]) * in[r, hidden + c]
//   swish(x) = x * sigmoid(x)
//
// vs generic kernel:
//   1. float4 vectorized fp32 path (4 outputs / thread)
//   2. 2D launch (col tiling along hidden, row tiling along rows)
//   3. __launch_bounds__(256, 8) occupancy hint for Blackwell
//   4. hidden % 4 != 0 or a pointer not 16-byte aligned: the scalar kernel for the whole row (2026-10-31; before,
//      the float4 kernel + a scalar tail kernel, which gave wrong rows: the float4 row stride was 2 * (hidden / 4))
//   5. fp16 / bf16 paths upcast to fp32 for the swish math
//
// Forward only. Backward is a placeholder stub that the dispatcher does not call
// yet (it falls through to launch_swiglu_backward_generic). Fill in later.
// =================================================================================

namespace OwnTensor {
namespace cuda {

// ── Type helpers ──
template<typename T> __device__ __forceinline__ float to_f_sw120(T v);
template<> __device__ __forceinline__ float to_f_sw120(float v) { return v; }
template<> __device__ __forceinline__ float to_f_sw120(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_f_sw120(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_f_sw120(float v);
template<> __device__ __forceinline__ float from_f_sw120(float v) { return v; }
template<> __device__ __forceinline__ __half from_f_sw120(float v) { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f_sw120(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float sigmoid_sw120(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float swish_sw120(float x) {
    return x * sigmoid_sw120(x);
}

// =================================================================================
// Forward — float4 vectorized fp32 main kernel
//   X : float4 view of input  [rows, 2*hidden]
//   Y : float4 view of output [rows, hidden]
//   cols4 = hidden / 4; used only when hidden % 4 == 0 and both pointers are 16-byte aligned (2026-10-31)
// =================================================================================
__global__ void __launch_bounds__(256, 8)
swiglu_sm120_kernel(const float4* __restrict__ X, float4* __restrict__ Y,
                    int64_t rows, int cols4) {
    // 2026-10-31: rows were in grid y, which holds at most 65,535 blocks (x 4 rows = 262,140 rows per call); now the
    // launcher caps grid y at 65,535 and each thread loops over its rows (stride gridDim.y * blockDim.y).
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= cols4) return;

    for (int64_t row = (int64_t)blockDim.y * blockIdx.y + threadIdx.y; row < rows; row += (int64_t)gridDim.y * blockDim.y) {
        int64_t row_in4  = row * (2 * cols4);
        int64_t row_out4 = row * cols4;

        float4 a = __ldg(&X[row_in4 + col]);
        float4 b = __ldg(&X[row_in4 + cols4 + col]);

        float4 y;
        y.x = swish_sw120(a.x) * b.x;
        y.y = swish_sw120(a.y) * b.y;
        y.z = swish_sw120(a.z) * b.z;
        y.w = swish_sw120(a.w) * b.w;

        Y[row_out4 + col] = y;
    }
}

// Scalar typed kernel for fp16 / bf16 (whole row, upcast math)
template<typename T>
__global__ void __launch_bounds__(256, 8)
swiglu_sm120_typed_kernel(const T* __restrict__ X, T* __restrict__ Y,
                          int64_t rows, int hidden) {
    // 2026-10-31: rows were in grid y, which holds at most 65,535 blocks (x 4 rows = 262,140 rows per call); now the
    // launcher caps grid y at 65,535 and each thread loops over its rows (stride gridDim.y * blockDim.y).
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= hidden) return;

    for (int64_t row = (int64_t)blockDim.y * blockIdx.y + threadIdx.y; row < rows; row += (int64_t)gridDim.y * blockDim.y) {
        float a = to_f_sw120(X[row * 2 * hidden + col]);
        float b = to_f_sw120(X[row * 2 * hidden + hidden + col]);
        Y[row * hidden + col] = from_f_sw120<T>(swish_sw120(a) * b);
    }
}

// =================================================================================
// Launchers
// =================================================================================
void swiglu_sm120(const float* in, float* out, int64_t rows, int64_t hidden, cudaStream_t stream) {
    CUDA_KERNEL_SCOPE("swiglu_sm120", stream);

    dim3 block(32, 4);

    // 2026-10-31: float4 only when every row is whole float4s and both pointers are 16-byte aligned (PyTorch checks
    // can_vectorize_up_to before its vectorized loop; TransformerEngine has an aligned and an unaligned launch);
    // otherwise the scalar kernel for the whole row. Before, hidden % 4 != 0 ran the float4 kernel with the row
    // strides 2 * (hidden / 4) and hidden / 4 (every row after row 0 wrong) plus a scalar tail kernel. Chosen by
    // test with the grid y cap below (SFT work doc 45.123 / 45.124): exact in every case, no slowdown at our sizes.
    const bool vec = hidden % 4 == 0 && reinterpret_cast<uintptr_t>(in) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(out) % 16 == 0;
    if (vec) {
        const int cols4 = (int)(hidden / 4);
        dim3 grid((unsigned)((cols4 + block.x - 1) / block.x),
                  (unsigned)std::min<int64_t>((rows + block.y - 1) / block.y, 65535));   // the kernel loops over rows
        swiglu_sm120_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            rows, cols4);
    } else {
        dim3 grid((unsigned)((hidden + block.x - 1) / block.x),
                  (unsigned)std::min<int64_t>((rows + block.y - 1) / block.y, 65535));   // the kernel loops over rows
        swiglu_sm120_typed_kernel<float><<<grid, block, 0, stream>>>(in, out, rows, (int)hidden);
    }
}

void swiglu_sm120(const __half* in, __half* out, int64_t rows, int64_t hidden, cudaStream_t stream) {
    CUDA_KERNEL_SCOPE("swiglu_sm120", stream);
    dim3 block(32, 4);
    dim3 grid((unsigned)((hidden + block.x - 1) / block.x),   // 2026-10-31: grid y capped at 65,535, the kernel loops over rows
              (unsigned)std::min<int64_t>((rows + block.y - 1) / block.y, 65535));
    swiglu_sm120_typed_kernel<__half><<<grid, block, 0, stream>>>(in, out, rows, (int)hidden);
}

void swiglu_sm120(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t rows, int64_t hidden, cudaStream_t stream) {
    CUDA_KERNEL_SCOPE("swiglu_sm120", stream);
    dim3 block(32, 4);
    dim3 grid((unsigned)((hidden + block.x - 1) / block.x),   // 2026-10-31: grid y capped at 65,535, the kernel loops over rows
              (unsigned)std::min<int64_t>((rows + block.y - 1) / block.y, 65535));
    swiglu_sm120_typed_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(in, out, rows, (int)hidden);
}

// =================================================================================
// PLACEHOLDER — SwiGLU backward sm120 (not yet implemented)
//
// Signature mirrors launch_swiglu_backward_generic. The dispatcher in
// ActivationKernels.cu does NOT call these yet; backward still routes to the
// generic launcher. Implement the optimized backward here, then add the
// get_arch() == ArchFamily::Blackwell_RTX branch in swiglu_backward_cuda.
// =================================================================================
template<typename T>
void swiglu_backward_sm120(const T* /*grad_out*/, const T* /*input*/, T* /*grad_input*/,
                           int64_t /*rows*/, int64_t /*hidden*/, cudaStream_t /*stream*/) {
    CUDA_KERNEL_SCOPE("swiglu_backward_sm120_placeholder", /*stream*/ 0);
    // TODO(sm120): implement optimized SwiGLU backward.
}
template void swiglu_backward_sm120<float>(const float*, const float*, float*, int64_t, int64_t, cudaStream_t);
template void swiglu_backward_sm120<__half>(const __half*, const __half*, __half*, int64_t, int64_t, cudaStream_t);
template void swiglu_backward_sm120<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, cudaStream_t);

} // namespace cuda
} // namespace OwnTensor
