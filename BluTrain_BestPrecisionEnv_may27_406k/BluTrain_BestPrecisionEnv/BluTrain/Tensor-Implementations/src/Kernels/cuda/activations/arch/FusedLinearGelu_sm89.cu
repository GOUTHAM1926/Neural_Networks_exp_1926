// =============================================================================
// fusedGelu.cu
//
// Fused Linear + GELU using cuBLASLt epilogue API.
//
// Ada (sm_89) fast path:
//   cuBLASLt matmul descriptor with CUBLASLT_EPILOGUE_GELU (+ bias when provided).
//   The epilogue is fused into the GEMM output stage so no intermediate tensor
//   is written to global memory.  NVIDIA's epilogue internally uses PTX
//   tanh.approx.f32 — same approximation as the standalone GELUKernel.cu.
//
// Generic fallback (sm_86 and earlier):
//   Standard cublasGemmEx (row-major, alpha=1, beta=0) followed by the
//   standalone fused_gelu_cuda / gelu_forward_typed kernel.
//
// Layout convention (all row-major / no transpose on the host side):
//   input   : [M, K]
//   weight  : [N, K]   → the kernel transposes B internally (CUBLAS_OP_T)
//   output  : [M, N]
//   bias    : [N]
//
// cuBLASLt uses column-major by default.  We feed it the transposed problem
// to keep row-major semantics:
//   C = A  × B^T   in row-major
//   ≡ C^T = B × A^T in col-major
// So the cuBLASLt call is:
//   matmul( opA=N, opB=N,
//           A = weight [N×K col-major],
//           B = input  [K×M col-major] (which is our [M×K] row-major pointer),
//           C = output [N×M col-major] (which is our [M×N] row-major pointer))
//   with lda=K, ldb=K, ldc=N.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <mutex>
#include <array>
#include <cstring>
#include "dtype/Dtype.h"

#include "ops/helpers/ActivationKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "dtype/Dtype.h"

namespace OwnTensor {
namespace cuda {

// =============================================================================
// cuBLASLt handle — one per device, lazy init, thread-safe
// =============================================================================

static std::mutex                        s_lt_mutex;
static std::array<cublasLtHandle_t, 8>  s_lt_handles{};
static std::array<bool, 8>              s_lt_ready{};

static cublasLtHandle_t get_cublaslt_handle(int device) {
    if (!s_lt_ready[device]) {
        std::lock_guard<std::mutex> lock(s_lt_mutex);
        if (!s_lt_ready[device]) {
            cudaSetDevice(device);
            cublasLtCreate(&s_lt_handles[device]);
            s_lt_ready[device] = true;
        }
    }
    return s_lt_handles[device];
}

// =============================================================================
// Workspace — 32 MiB per device, allocated on first use
// =============================================================================

static constexpr size_t WORKSPACE_SIZE = 32ull * 1024 * 1024; // 32 MiB

static std::array<void*, 8>  s_workspace{};
static std::array<bool, 8>   s_ws_ready{};
static std::mutex            s_ws_mutex;

static void* get_workspace(int device) {
    if (!s_ws_ready[device]) {
        std::lock_guard<std::mutex> lock(s_ws_mutex);
        if (!s_ws_ready[device]) {
            cudaSetDevice(device);
            cudaMalloc(&s_workspace[device], WORKSPACE_SIZE);
            s_ws_ready[device] = true;
        }
    }
    return s_workspace[device];
}

// =============================================================================
// Ada fast path: cuBLASLt with GELU epilogue (float32)
// =============================================================================

static void run_cublaslt_gelu_f32(
    cublasLtHandle_t lt,
    const float* A,   // weight  [N, K] row-major
    const float* B,   // input   [M, K] row-major
    const float* bias,// [N] or nullptr
    float*       C,   // output  [M, N] row-major
    int64_t M, int64_t N, int64_t K,
    void* workspace, size_t ws_size)
{
    // -------------------------------------------------------------------------
    // Operation descriptor
    // -------------------------------------------------------------------------
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Both A and B are presented without extra transpose because we already
    // handle row→col reinterpretation via the leading-dimension trick.
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Choose epilogue: GELU with or without bias
    cublasLtEpilogue_t epilogue = (bias != nullptr)
        ? CUBLASLT_EPILOGUE_GELU_BIAS
        : CUBLASLT_EPILOGUE_GELU;
    cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

    if (bias != nullptr) {
        cublasLtMatmulDescSetAttribute(op_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }

    // -------------------------------------------------------------------------
    // Matrix layouts (column-major view of row-major buffers)
    //
    // Row-major [M, K] == col-major [K, M] with leading dim K.
    // We compute C(col) = A(col) × B(col)
    //   A = weight [N×K] row-major → presented as [K×N] col-major, ld=K
    //   B = input  [M×K] row-major → presented as [K×M] col-major, ld=K
    //   C = output [M×N] row-major → presented as [N×M] col-major, ld=N
    // -------------------------------------------------------------------------
    cublasLtMatrixLayout_t layout_A = nullptr, layout_B = nullptr, layout_C = nullptr;
    cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_32F, K, N, K); // A: K rows, N cols
    cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_32F, K, M, K); // B: K rows, M cols
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32F, N, M, N); // C: N rows, M cols

    // -------------------------------------------------------------------------
    // Heuristic search — pick the best algorithm for this shape
    // -------------------------------------------------------------------------
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op_desc,
        layout_A, layout_B, layout_C, layout_C,
        pref, 1, &heuristic, &returned);

    // -------------------------------------------------------------------------
    // Execute
    // -------------------------------------------------------------------------
    const float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(lt, op_desc,
        &alpha,
        A, layout_A,
        B, layout_B,
        &beta,
        C, layout_C,
        C, layout_C,
        (returned > 0) ? &heuristic.algo : nullptr,
        workspace, ws_size,
        /*stream=*/nullptr);

    // -------------------------------------------------------------------------
    // Cleanup descriptors (lightweight, no GPU memory)
    // -------------------------------------------------------------------------
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_C);
    cublasLtMatrixLayoutDestroy(layout_B);
    cublasLtMatrixLayoutDestroy(layout_A);
    cublasLtMatmulDescDestroy(op_desc);
}

// =============================================================================
// Ada fast path: cuBLASLt with GELU epilogue (float16 / bfloat16)
// cuBLASLt GELU epilogue is supported for FP16 and BF16 on sm_89.
// =============================================================================

static void run_cublaslt_gelu_fp16(
    cublasLtHandle_t lt,
    const __half* A,    // weight  [N, K] row-major
    const __half* B,    // input   [M, K] row-major
    const __half* bias, // [N] or nullptr
    __half*       C,    // output  [M, N] row-major
    int64_t M, int64_t N, int64_t K,
    void* workspace, size_t ws_size)
{
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);

    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    cublasLtEpilogue_t epilogue = (bias != nullptr)
        ? CUBLASLT_EPILOGUE_GELU_BIAS
        : CUBLASLT_EPILOGUE_GELU;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (bias != nullptr) {
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }

    cublasLtMatrixLayout_t layout_A = nullptr, layout_B = nullptr, layout_C = nullptr;
    cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_16F, K, N, K);
    cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op_desc,
        layout_A, layout_B, layout_C, layout_C,
        pref, 1, &heuristic, &returned);

    const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    cublasLtMatmul(lt, op_desc,
        &alpha_h, A, layout_A, B, layout_B,
        &beta_h,  C, layout_C, C, layout_C,
        (returned > 0) ? &heuristic.algo : nullptr,
        workspace, ws_size, nullptr);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_C);
    cublasLtMatrixLayoutDestroy(layout_B);
    cublasLtMatrixLayoutDestroy(layout_A);
    cublasLtMatmulDescDestroy(op_desc);
}

static void run_cublaslt_gelu_bf16(
    cublasLtHandle_t lt,
    const __nv_bfloat16* A,    // weight  [N, K] row-major
    const __nv_bfloat16* B,    // input   [M, K] row-major
    const __nv_bfloat16* bias, // [N] or nullptr
    __nv_bfloat16*       C,    // output  [M, N] row-major
    int64_t M, int64_t N, int64_t K,
    void* workspace, size_t ws_size)
{
    cublasLtMatmulDesc_t op_desc = nullptr;
    // BF16 accumulation via FP32 for numerical stability
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    cublasLtEpilogue_t epilogue = (bias != nullptr)
        ? CUBLASLT_EPILOGUE_GELU_BIAS
        : CUBLASLT_EPILOGUE_GELU;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (bias != nullptr) {
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }

    cublasLtMatrixLayout_t layout_A = nullptr, layout_B = nullptr, layout_C = nullptr;
    cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_16BF, N, M, N);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op_desc,
        layout_A, layout_B, layout_C, layout_C,
        pref, 1, &heuristic, &returned);

    const float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(lt, op_desc,
        &alpha, A, layout_A, B, layout_B,
        &beta,  C, layout_C, C, layout_C,
        (returned > 0) ? &heuristic.algo : nullptr,
        workspace, ws_size, nullptr);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_C);
    cublasLtMatrixLayoutDestroy(layout_B);
    cublasLtMatrixLayoutDestroy(layout_A);
    cublasLtMatmulDescDestroy(op_desc);
}

// =============================================================================
// Generic fallback: cuBLAS GemmEx + standalone GELU kernel
// =============================================================================

static std::mutex                       s_cublas_mutex;
static std::array<cublasHandle_t, 8>    s_cublas_handles{};
static std::array<bool, 8>              s_cublas_ready{};

static cublasHandle_t get_cublas_handle(int device) {
    if (!s_cublas_ready[device]) {
        std::lock_guard<std::mutex> lock(s_cublas_mutex);
        if (!s_cublas_ready[device]) {
            cudaSetDevice(device);
            cublasCreate(&s_cublas_handles[device]);
            cublasSetMathMode(s_cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH);
            s_cublas_ready[device] = true;
        }
    }
    return s_cublas_handles[device];
}

// Bias add kernel (element-wise, broadcasts bias across rows)
__global__ void add_bias_kernel(float* output, const float* bias,
                                int64_t M, int64_t N) {
    int64_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total  = M * N;
    for (int64_t i = idx; i < total; i += stride)
        output[i] += bias[i % N];
}

static void run_cublas_gelu_fallback_f32(
    cublasHandle_t handle,
    const float* A,    // weight  [N, K] row-major
    const float* B,    // input   [M, K] row-major
    const float* bias, // [N] or nullptr
    float*       C,    // output  [M, N] row-major
    int64_t M, int64_t N, int64_t K)
{
    // Row-major C = B * A^T  ↔  col-major C^T = A * B^T
    // cuBLAS (col-major): C[N×M] = A[N×K] * B^T[K×M]
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        A, (int)K,   // A  [K × N] col-major (our [N × K] row-major)
        B, (int)K,   // B  [K × M] col-major (our [M × K] row-major)
        &beta,
        C, (int)N);  // C  [N × M] col-major (our [M × N] row-major)

    if (bias != nullptr) {
        int threads = 256;
        int64_t total = M * N;
        int blocks = (int)std::min((total + threads - 1) / threads, (int64_t)65535);
        add_bias_kernel<<<blocks, threads>>>(C, bias, M, N);
    }

    // Standalone GELU over the [M*N] output elements
    fused_gelu_cuda(C, C, M * N);
}

// =============================================================================
// Public entry points
// =============================================================================

void fused_linear_gelu_forward(
    const void*  input,
    const void*  weight,
    const void*  bias,
    void*        output,
    int64_t      M,
    int64_t      N,
    int64_t      K,
    Dtype        dtype,
    int          device_idx)
{
    if (device_idx < 0) cudaGetDevice(&device_idx);

    ArchFamily arch = get_arch(device_idx);
    void*  ws      = get_workspace(device_idx);

    if (dtype == Dtype::Float32) {
        const float* A  = static_cast<const float*>(weight);
        const float* B  = static_cast<const float*>(input);
        const float* b  = static_cast<const float*>(bias);
        float*       C  = static_cast<float*>(output);

        if (arch == ArchFamily::Ada) {
            run_cublaslt_gelu_f32(get_cublaslt_handle(device_idx),
                A, B, b, C, M, N, K, ws, WORKSPACE_SIZE);
        } else {
            run_cublas_gelu_fallback_f32(get_cublas_handle(device_idx),
                A, B, b, C, M, N, K);
        }
    }
    else if (dtype == Dtype::Float16) {
        const __half* A = static_cast<const __half*>(weight);
        const __half* B = static_cast<const __half*>(input);
        const __half* b = static_cast<const __half*>(bias);
        __half*       C = static_cast<__half*>(output);

        if (arch == ArchFamily::Ada) {
            run_cublaslt_gelu_fp16(get_cublaslt_handle(device_idx),
                A, B, b, C, M, N, K, ws, WORKSPACE_SIZE);
        } else {
            // Fallback: FP16 GemmEx + standalone typed GELU
            cublasHandle_t h = get_cublas_handle(device_idx);
            const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
            cublasHgemm(h,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)N, (int)M, (int)K,
                &alpha_h, A, (int)K, B, (int)K,
                &beta_h,  C, (int)N);
                if (dtype == Dtype::Float16) {
                    fused_gelu_cuda<float16_t>(reinterpret_cast<const float16_t*>(C), reinterpret_cast<float16_t*>(C), M * N);
                }
        }
    }
    else if (dtype == Dtype::Bfloat16) {
        const __nv_bfloat16* A = static_cast<const __nv_bfloat16*>(weight);
        const __nv_bfloat16* B = static_cast<const __nv_bfloat16*>(input);
        const __nv_bfloat16* b = static_cast<const __nv_bfloat16*>(bias);
        __nv_bfloat16*       C = static_cast<__nv_bfloat16*>(output);

        if (arch == ArchFamily::Ada) {
            run_cublaslt_gelu_bf16(get_cublaslt_handle(device_idx),
                A, B, b, C, M, N, K, ws, WORKSPACE_SIZE);
        } else {
            // Fallback: BF16 GemmEx + standalone typed GELU
            cublasHandle_t h = get_cublas_handle(device_idx);
            const float alpha = 1.0f, beta = 0.0f;
            cublasGemmEx(h,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)N, (int)M, (int)K,
                &alpha,
                A, CUDA_R_16BF, (int)K,
                B, CUDA_R_16BF, (int)K,
                &beta,
                C, CUDA_R_16BF, (int)N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (dtype == Dtype::Bfloat16) {
                fused_gelu_cuda<bfloat16_t>(reinterpret_cast<const bfloat16_t*>(C), reinterpret_cast<bfloat16_t*>(C), M * N);
            }

        }
    }
    else {
        throw std::runtime_error("fused_linear_gelu_forward: unsupported dtype");
    }
}

void fused_linear_gelu_forward_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int64_t      M,
    int64_t      N,
    int64_t      K)
{
    fused_linear_gelu_forward(input, weight, bias, output,
                              M, N, K, Dtype::Float32, -1);
}

} // namespace cuda
} // namespace OwnTensor
