#include "ops/UnaryOps/Activations.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/helpers/ActivationKernels.h"
#include "ops/helpers/Vectorized.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/FusedKernels.cuh"
#include "device/DeviceCore.h"
#include "dtype/CudaTraits.h"
#include "dtype/DtypeTraits.h"
#include "utils/Profiler.h"
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace OwnTensor {

// =================================================================
// relu_forward: CPU/GPU dispatch
// relu(x) = (x + |x|) * 0.5   — NaN-propagating, branch-free
// =================================================================
Tensor relu_forward(const Tensor& input) {
    // ── GPU dispatch (fp32, fp16, bf16) ──
    if (input.device().is_cuda()) {
        Tensor output(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));
        {
            AUTO_PROFILE_CUDA("Forward::ReLU_Forward");
            if (input.dtype() == Dtype::Float32) {
                cuda::relu_forward_cuda(input.data<float>(), output.data<float>(), input.numel());
            } else {
                throw std::runtime_error("ReLU CUDA: only float32 supported");
            }
        }
        return output;
    }

    // ── CPU: fused AVX2 kernel using (x + |x|) * 0.5 ──
    if (input.dtype() == Dtype::Float32) {
        const int64_t numel = input.numel();
        Tensor output(input.shape(), TensorOptions().with_dtype(Dtype::Float32).with_device(input.device()));

        const float* in_ptr  = input.data<float>();
        float*       out_ptr = output.data<float>();

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();
        const Vec kHalf(0.5f);

        const int64_t omp_threshold = 16384;
        const int num_threads = (numel >= omp_threshold) ? omp_get_max_threads() : 1;

        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            const int64_t chunk = (numel + nthreads - 1) / nthreads;
            const int64_t start = tid * chunk;
            const int64_t end   = std::min(start + chunk, numel);
            const int64_t len   = end - start;

            const float* src = in_ptr  + start;
            float*       dst = out_ptr + start;

            int64_t i = 0;
            for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
                Vec x = Vec::loadu(src + i);
                Vec result = (x + x.abs()) * kHalf;
                result.storeu(dst + i);
            }
            for (; i < len; i++) {
                float x = src[i];
                dst[i] = (x + fabsf(x)) * 0.5f;
            }
        }
        return output;
    }

    // Other dtypes: tensor-level fallback (still NaN-propagating via arithmetic)
    return (input + OwnTensor::abs(input)) * 0.5f;
}

// =================================================================
// gelu_forward: CPU/GPU dispatch
// =================================================================
Tensor gelu_forward(const Tensor& input) {
    if (!is_float(input.dtype())) {
        throw std::runtime_error("GeLU: only float dtypes supported!");
    }

    if (input.device().is_cuda()) {
        Tensor output(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));
        {
            AUTO_PROFILE_CUDA("Forward::GeLU_Forward");
            switch (input.dtype()) {
                case Dtype::Float32:
                    cuda::fused_gelu_cuda(input.data<float>(), output.data<float>(), input.numel());
                    break;
                case Dtype::Float16: {
                    using CudaF16 = detail::CudaNativeType<float16_t>;
                    cuda::fused_gelu_cuda(
                        reinterpret_cast<const CudaF16*>(input.data<float16_t>()),
                        reinterpret_cast<CudaF16*>(output.data<float16_t>()),
                        input.numel());
                    break;
                }
                case Dtype::Bfloat16: {
                    using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
                    cuda::fused_gelu_cuda(
                        reinterpret_cast<const CudaBF16*>(input.data<bfloat16_t>()),
                        reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
                        input.numel());
                    break;
                }
                default:
                    throw std::runtime_error("GeLU CUDA: unsupported dtype");
            }
        }
        return output;
    }

    // ── CPU: fused single-pass AVX2 kernel ──────────────────────────
    // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    //
    // One read, all math in registers, one write. No temporaries.
    // Uses: Vectorized<float>::tanh(), fmadd, 2x loop unrolling, OpenMP.
    // ───────────────────────────────────────────────────────────────────

    // ── fp16/bf16 CPU: load→float32→compute→store using F16C ──────
    // All optimizations: single-pass, AVX2, vectorized tanh, FMA, 2x unroll, OpenMP
    if (input.dtype() == Dtype::Float16 || input.dtype() == Dtype::Bfloat16) {
        const int64_t numel = input.numel();
        Tensor output(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();  // 8

        const Vec kHalf(0.5f);
        const Vec kOne(1.0f);
        const Vec kBeta(std::sqrt(2.0f / M_PI));
        const Vec kKappa(0.044715f);

        const bool is_fp16 = (input.dtype() == Dtype::Float16);
        constexpr int elem_bytes = 2;

        const char* in_base  = reinterpret_cast<const char*>(input.data());
        char*       out_base = reinterpret_cast<char*>(output.data());

        auto load_vec = [&](const void* ptr) -> Vec {
            return is_fp16 ? vec::load_fp16_as_float(ptr) : vec::load_bf16_as_float(ptr);
        };
        auto store_vec = [&](void* ptr, const Vec& v) {
            is_fp16 ? vec::store_float_as_fp16(ptr, v) : vec::store_float_as_bf16(ptr, v);
        };

        const int64_t omp_threshold = 16384;
        const int num_threads = (numel >= omp_threshold) ? omp_get_max_threads() : 1;

        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            const int64_t chunk = (numel + nthreads - 1) / nthreads;
            const int64_t start = tid * chunk;
            const int64_t end   = std::min(start + chunk, numel);
            const int64_t len   = end - start;

            const char* src = in_base  + start * elem_bytes;
            char*       dst = out_base + start * elem_bytes;

            int64_t i = 0;

            // ── Main loop: 2x unrolled (16 elements per iteration) ──
            for (; i + 2 * VEC_SIZE <= len; i += 2 * VEC_SIZE) {
                Vec x0 = load_vec(src + i * elem_bytes);
                Vec x1 = load_vec(src + (i + VEC_SIZE) * elem_bytes);

                Vec x0_sq = x0 * x0;  Vec x1_sq = x1 * x1;
                Vec x0_cube = x0_sq * x0;  Vec x1_cube = x1_sq * x1;

                Vec inner0 = kBeta * Vec::fmadd(kKappa, x0_cube, x0);
                Vec inner1 = kBeta * Vec::fmadd(kKappa, x1_cube, x1);

                Vec out0 = kHalf * x0 * (kOne + inner0.tanh());
                Vec out1 = kHalf * x1 * (kOne + inner1.tanh());

                store_vec(dst + i * elem_bytes, out0);
                store_vec(dst + (i + VEC_SIZE) * elem_bytes, out1);
            }

            // ── Tail: single vector ──
            for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
                Vec x = load_vec(src + i * elem_bytes);
                Vec x_cube = x * x * x;
                Vec inner = kBeta * Vec::fmadd(kKappa, x_cube, x);
                Vec result = kHalf * x * (kOne + inner.tanh());
                store_vec(dst + i * elem_bytes, result);
            }

            // ── Remainder: scalar ──
            const float beta_s = std::sqrt(2.0f / M_PI);
            for (; i < len; i++) {
                float x;
                if (is_fp16) {
                    uint16_t raw; std::memcpy(&raw, src + i * elem_bytes, 2);
                    x = _cvtsh_ss(raw);
                } else {
                    uint16_t raw; std::memcpy(&raw, src + i * elem_bytes, 2);
                    uint32_t bits = static_cast<uint32_t>(raw) << 16;
                    std::memcpy(&x, &bits, 4);
                }
                float x3 = x * x * x;
                float inner = beta_s * (x + 0.044715f * x3);
                float result = 0.5f * x * (1.0f + std::tanh(inner));
                if (is_fp16) {
                    uint16_t out = _cvtss_sh(result, _MM_FROUND_TO_NEAREST_INT);
                    std::memcpy(dst + i * elem_bytes, &out, 2);
                } else {
                    uint32_t bits; std::memcpy(&bits, &result, 4);
                    uint16_t out = static_cast<uint16_t>(bits >> 16);
                    std::memcpy(dst + i * elem_bytes, &out, 2);
                }
            }
        }
        return output;
    }

    if (input.dtype() != Dtype::Float32) {
        // Other float dtypes: tensor ops fallback
        const float sqrt_2_over_pi_val = std::sqrt(2.0f / M_PI);
        Tensor half_x = 0.5f * input;
        Tensor x_cubed = input * input * input;
        Tensor tanh_inp = sqrt_2_over_pi_val * (input + 0.044715f * x_cubed);
        Tensor inner_output = 1.0f + tanh(tanh_inp);
        return half_x * inner_output;
    }

    // Float32 CPU: fused AVX2 kernel
    const int64_t numel = input.numel();
    Tensor output(input.shape(), TensorOptions().with_dtype(Dtype::Float32).with_device(input.device()));

    const float* in_ptr  = input.data<float>();
    float*       out_ptr = output.data<float>();

    using Vec = vec::Vectorized<float>;
    constexpr int VEC_SIZE = Vec::size();  // 8 for AVX2

    const Vec kHalf(0.5f);
    const Vec kOne(1.0f);
    const Vec kBeta(std::sqrt(2.0f / M_PI));   // sqrt(2/pi) ≈ 0.7978845608
    const Vec kKappa(0.044715f);

    // OpenMP for large tensors (threshold: 16384 elements, like PyTorch)
    const int64_t omp_threshold = 16384;
    const int num_threads = (numel >= omp_threshold) ? omp_get_max_threads() : 1;

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int64_t chunk = (numel + nthreads - 1) / nthreads;
        const int64_t start = tid * chunk;
        const int64_t end   = std::min(start + chunk, numel);
        const int64_t len   = end - start;

        const float* src = in_ptr  + start;
        float*       dst = out_ptr + start;

        int64_t i = 0;

        // ── Main loop: 2x unrolled (16 floats per iteration) ──
        for (; i + 2 * VEC_SIZE <= len; i += 2 * VEC_SIZE) {
            Vec x0 = Vec::loadu(src + i);
            Vec x1 = Vec::loadu(src + i + VEC_SIZE);

            // x^3
            Vec x0_sq = x0 * x0;
            Vec x1_sq = x1 * x1;
            Vec x0_cube = x0_sq * x0;
            Vec x1_cube = x1_sq * x1;

            // inner = beta * (x + kappa * x^3)  →  beta * fmadd(kappa, x^3, x)
            Vec inner0 = kBeta * Vec::fmadd(kKappa, x0_cube, x0);
            Vec inner1 = kBeta * Vec::fmadd(kKappa, x1_cube, x1);

            // tanh(inner)
            Vec tanh0 = inner0.tanh();
            Vec tanh1 = inner1.tanh();

            // 0.5 * x * (1 + tanh(inner))
            Vec out0 = kHalf * x0 * (kOne + tanh0);
            Vec out1 = kHalf * x1 * (kOne + tanh1);

            out0.storeu(dst + i);
            out1.storeu(dst + i + VEC_SIZE);
        }

        // ── Tail: single vector ──
        for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
            Vec x = Vec::loadu(src + i);
            Vec x_cube = x * x * x;
            Vec inner = kBeta * Vec::fmadd(kKappa, x_cube, x);
            Vec result = kHalf * x * (kOne + inner.tanh());
            result.storeu(dst + i);
        }

        // ── Remainder: scalar ──
        const float beta_s  = std::sqrt(2.0f / M_PI);
        const float kappa_s = 0.044715f;
        for (; i < len; i++) {
            float x = src[i];
            float x3 = x * x * x;
            float inner = beta_s * (x + kappa_s * x3);
            dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    return output;
}

// =================================================================
// sigmoid_forward: CPU/GPU dispatch
// =================================================================
Tensor sigmoid_forward(const Tensor& input) {
    if (!is_float(input.dtype())) {
        throw std::runtime_error("Sigmoid: only float dtypes supported!");
    }

    if (input.device().is_cuda()) {
        Tensor output(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));
        {
            AUTO_PROFILE_CUDA("Forward::Sigmoid_Forward");
            switch (input.dtype()) {
                case Dtype::Float32:
                    cuda::sigmoid_forward_cuda(input.data<float>(), output.data<float>(), input.numel());
                    break;
                case Dtype::Float16: {
                    using CudaF16 = detail::CudaNativeType<float16_t>;
                    cuda::sigmoid_forward_cuda(
                        reinterpret_cast<const CudaF16*>(input.data<float16_t>()),
                        reinterpret_cast<CudaF16*>(output.data<float16_t>()),
                        input.numel());
                    break;
                }
                case Dtype::Bfloat16: {
                    using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
                    cuda::sigmoid_forward_cuda(
                        reinterpret_cast<const CudaBF16*>(input.data<bfloat16_t>()),
                        reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
                        input.numel());
                    break;
                }
                default:
                    throw std::runtime_error("Sigmoid CUDA: unsupported dtype");
            }
        }
        return output;
    }

    // CPU fallback
    Tensor exp_input = exp(input);
    Tensor denom = 1.0f + exp_input;
    Tensor output = exp_input / denom;
    return output;
}

// =================================================================
// softmax_forward: CPU/GPU dispatch
// =================================================================
Tensor softmax_forward(const Tensor& input, int64_t dim) {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;

    if (!is_float(input.dtype())) {
        throw std::runtime_error("Softmax: only float dtypes supported!");
    }

    if (input.device().is_cuda() && dim == ndim - 1) {
        Tensor output(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));

        int64_t cols = input.shape().dims.back();
        int64_t rows = input.numel() / cols;
        {
            AUTO_PROFILE_CUDA("Forward::Softmax_Forward");
            switch (input.dtype()) {
                case Dtype::Float32:
                    cuda::softmax_forward_cuda(input.data<float>(), output.data<float>(), rows, cols);
                    break;
                case Dtype::Float16: {
                    using CudaF16 = detail::CudaNativeType<float16_t>;
                    cuda::softmax_forward_cuda_typed(
                        reinterpret_cast<const CudaF16*>(input.data<float16_t>()),
                        reinterpret_cast<CudaF16*>(output.data<float16_t>()),
                        rows, cols);
                    break;
                }
                case Dtype::Bfloat16: {
                    using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
                    cuda::softmax_forward_cuda_typed(
                        reinterpret_cast<const CudaBF16*>(input.data<bfloat16_t>()),
                        reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
                        rows, cols);
                    break;
                }
                default:
                    throw std::runtime_error("Softmax CUDA: unsupported dtype");
            }
        }
        return output;
    }

    // CPU fallback (also handles non-last-dim CUDA softmax)
    Tensor max_val = reduce_max(input, {dim}, true);
    Tensor shifted = input - max_val;
    Tensor exp_x = exp(shifted);
    Tensor sum_exp = reduce_sum(exp_x, {dim}, true);
    Tensor output = exp_x / sum_exp;
    return output;
}

// =================================================================
// swiglu_forward: CPU/GPU dispatch
// =================================================================
Tensor swiglu_forward(const Tensor& input) {
    int64_t last = input.shape().dims.back();
    if (last % 2 != 0) {
        throw std::runtime_error("SwiGLU expects last dim divisible by 2");
    }

    int64_t hidden = last / 2;
    Shape out_shape = input.shape();
    out_shape.dims.back() = hidden;

    if (!is_float(input.dtype())) {
        throw std::runtime_error("SwiGLU: only float dtypes supported!");
    }

    if (input.device().is_cuda()) {
        Tensor output(out_shape, input.opts());

        int64_t rows = input.numel() / last;
        {
            AUTO_PROFILE_CUDA("Forward::SwiGLU_Forward");
            switch (input.dtype()) {
                case Dtype::Float32:
                    cuda::swiglu_forward_cuda(input.data<float>(), output.data<float>(), rows, hidden);
                    break;
                case Dtype::Float16: {
                    using CudaF16 = detail::CudaNativeType<float16_t>;
                    cuda::swiglu_forward_cuda(
                        reinterpret_cast<const CudaF16*>(input.data<float16_t>()),
                        reinterpret_cast<CudaF16*>(output.data<float16_t>()),
                        rows, hidden);
                    break;
                }
                case Dtype::Bfloat16: {
                    using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
                    cuda::swiglu_forward_cuda(
                        reinterpret_cast<const CudaBF16*>(input.data<bfloat16_t>()),
                        reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
                        rows, hidden);
                    break;
                }
                default:
                    throw std::runtime_error("SwiGLU CUDA: unsupported dtype");
            }
        }
        return output;
    }

    // CPU fallback: swiglu(x) = swish(A) * B
    // Split input along last dim into A (first half) and B (second half)
    int64_t ndim = input.ndim();
    Tensor& input_ref = const_cast<Tensor&>(input);
    Tensor A = input_ref.narrow(ndim - 1, 0, hidden);
    Tensor B = input_ref.narrow(ndim - 1, hidden, hidden);

    // swish(A) = A * sigmoid(A)
    Tensor sig_A = sigmoid_forward(A);
    Tensor swish_A = A * sig_A;

    return swish_A * B;
}

// =================================================================
// fused_bias_gelu_forward: CPU/GPU dispatch
// =================================================================
Tensor fused_bias_gelu_forward(const Tensor& input, const Tensor& bias) {
    if (input.device().is_cuda()) {
        int64_t hidden_dim = input.shape().dims.back();
        int64_t batch_size = input.numel() / hidden_dim;

        Tensor output(input.shape(), input.opts());
        {
            AUTO_PROFILE_CUDA("Forward::FusedBiasGeLU_Forward");
            switch (input.dtype()) {
                case Dtype::Float32:
                    cuda::fused_bias_gelu_cuda(input.data<float>(), bias.data<float>(),
                                               output.data<float>(), batch_size, hidden_dim);
                    break;
                default:
                    throw std::runtime_error("FusedBiasGeLU CUDA: only float32 supported");
            }
        }
        return output;
    }

    // ── CPU: fused single-pass AVX2 kernel ──────────────────────────
    // output = gelu(input + bias)
    // Fuses bias addition + GeLU into one pass. No temporary tensor.
    // Bias is broadcast along the last dimension.
    // ───────────────────────────────────────────────────────────────────

    // ── fp16/bf16 CPU: fused bias+gelu single-pass with F16C ──────
    if (input.dtype() == Dtype::Float16 || input.dtype() == Dtype::Bfloat16) {
        const int64_t numel = input.numel();
        const int64_t hidden_dim = input.shape().dims.back();
        const int64_t num_rows = numel / hidden_dim;
        Tensor output(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();
        constexpr int elem_bytes = 2;

        const Vec kHalf(0.5f);
        const Vec kOne(1.0f);
        const Vec kBeta(std::sqrt(2.0f / M_PI));
        const Vec kKappa(0.044715f);

        const bool is_fp16 = (input.dtype() == Dtype::Float16);
        const char* in_base   = reinterpret_cast<const char*>(input.data());
        const char* bias_base = reinterpret_cast<const char*>(bias.data());
        char*       out_base  = reinterpret_cast<char*>(output.data());

        auto load_vec = [&](const void* ptr) -> Vec {
            return is_fp16 ? vec::load_fp16_as_float(ptr) : vec::load_bf16_as_float(ptr);
        };
        auto store_vec = [&](void* ptr, const Vec& v) {
            is_fp16 ? vec::store_float_as_fp16(ptr, v) : vec::store_float_as_bf16(ptr, v);
        };

        const int64_t omp_threshold = 16384;
        const int num_threads = (numel >= omp_threshold) ? omp_get_max_threads() : 1;

        #pragma omp parallel for num_threads(num_threads)
        for (int64_t row = 0; row < num_rows; row++) {
            const char* src  = in_base   + row * hidden_dim * elem_bytes;
            char*       dst  = out_base  + row * hidden_dim * elem_bytes;

            int64_t j = 0;

            // ── Main loop: 2x unrolled ──
            for (; j + 2 * VEC_SIZE <= hidden_dim; j += 2 * VEC_SIZE) {
                Vec x0 = load_vec(src + j * elem_bytes)              + load_vec(bias_base + j * elem_bytes);
                Vec x1 = load_vec(src + (j + VEC_SIZE) * elem_bytes) + load_vec(bias_base + (j + VEC_SIZE) * elem_bytes);

                Vec x0_sq = x0 * x0;  Vec x1_sq = x1 * x1;
                Vec x0_cube = x0_sq * x0;  Vec x1_cube = x1_sq * x1;

                Vec inner0 = kBeta * Vec::fmadd(kKappa, x0_cube, x0);
                Vec inner1 = kBeta * Vec::fmadd(kKappa, x1_cube, x1);

                Vec out0 = kHalf * x0 * (kOne + inner0.tanh());
                Vec out1 = kHalf * x1 * (kOne + inner1.tanh());

                store_vec(dst + j * elem_bytes, out0);
                store_vec(dst + (j + VEC_SIZE) * elem_bytes, out1);
            }

            // ── Tail: single vector ──
            for (; j + VEC_SIZE <= hidden_dim; j += VEC_SIZE) {
                Vec x = load_vec(src + j * elem_bytes) + load_vec(bias_base + j * elem_bytes);
                Vec x_cube = x * x * x;
                Vec inner = kBeta * Vec::fmadd(kKappa, x_cube, x);
                Vec result = kHalf * x * (kOne + inner.tanh());
                store_vec(dst + j * elem_bytes, result);
            }

            // ── Remainder: scalar ──
            const float beta_s = std::sqrt(2.0f / M_PI);
            for (; j < hidden_dim; j++) {
                float xf;
                if (is_fp16) {
                    uint16_t ri, rb;
                    std::memcpy(&ri, src + j * elem_bytes, 2);
                    std::memcpy(&rb, bias_base + j * elem_bytes, 2);
                    xf = _cvtsh_ss(ri) + _cvtsh_ss(rb);
                } else {
                    uint16_t ri, rb;
                    std::memcpy(&ri, src + j * elem_bytes, 2);
                    std::memcpy(&rb, bias_base + j * elem_bytes, 2);
                    uint32_t bi = static_cast<uint32_t>(ri) << 16;
                    uint32_t bb = static_cast<uint32_t>(rb) << 16;
                    float fi, fb;
                    std::memcpy(&fi, &bi, 4); std::memcpy(&fb, &bb, 4);
                    xf = fi + fb;
                }
                float x3 = xf * xf * xf;
                float inner = beta_s * (xf + 0.044715f * x3);
                float result = 0.5f * xf * (1.0f + std::tanh(inner));
                if (is_fp16) {
                    uint16_t out = _cvtss_sh(result, _MM_FROUND_TO_NEAREST_INT);
                    std::memcpy(dst + j * elem_bytes, &out, 2);
                } else {
                    uint32_t bits; std::memcpy(&bits, &result, 4);
                    uint16_t out = static_cast<uint16_t>(bits >> 16);
                    std::memcpy(dst + j * elem_bytes, &out, 2);
                }
            }
        }
        return output;
    }

    if (input.dtype() != Dtype::Float32) {
        // Other float dtypes: tensor ops fallback
        Tensor biased = input + bias;
        return gelu_forward(biased);
    }

    // ── Float32 CPU: fused single-pass AVX2 kernel ──────────────────
    const int64_t numel = input.numel();
    const int64_t hidden_dim = input.shape().dims.back();
    Tensor output(input.shape(), input.opts());

    const float* in_ptr   = input.data<float>();
    const float* bias_ptr = bias.data<float>();
    float*       out_ptr  = output.data<float>();

    using Vec = vec::Vectorized<float>;
    constexpr int VEC_SIZE = Vec::size();

    const Vec kHalf(0.5f);
    const Vec kOne(1.0f);
    const Vec kBeta(std::sqrt(2.0f / M_PI));
    const Vec kKappa(0.044715f);

    const int64_t num_rows = numel / hidden_dim;
    const int64_t omp_threshold = 16384;
    const int num_threads = (numel >= omp_threshold) ? omp_get_max_threads() : 1;

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t row = 0; row < num_rows; row++) {
        const float* src  = in_ptr   + row * hidden_dim;
        float*       dst  = out_ptr  + row * hidden_dim;

        int64_t j = 0;

        // ── Main loop: 2x unrolled ──
        for (; j + 2 * VEC_SIZE <= hidden_dim; j += 2 * VEC_SIZE) {
            Vec x0 = Vec::loadu(src + j)              + Vec::loadu(bias_ptr + j);
            Vec x1 = Vec::loadu(src + j + VEC_SIZE)   + Vec::loadu(bias_ptr + j + VEC_SIZE);

            Vec x0_sq = x0 * x0;  Vec x1_sq = x1 * x1;
            Vec x0_cube = x0_sq * x0;  Vec x1_cube = x1_sq * x1;

            Vec inner0 = kBeta * Vec::fmadd(kKappa, x0_cube, x0);
            Vec inner1 = kBeta * Vec::fmadd(kKappa, x1_cube, x1);

            Vec out0 = kHalf * x0 * (kOne + inner0.tanh());
            Vec out1 = kHalf * x1 * (kOne + inner1.tanh());

            out0.storeu(dst + j);
            out1.storeu(dst + j + VEC_SIZE);
        }

        // ── Tail: single vector ──
        for (; j + VEC_SIZE <= hidden_dim; j += VEC_SIZE) {
            Vec x = Vec::loadu(src + j) + Vec::loadu(bias_ptr + j);
            Vec x_cube = x * x * x;
            Vec inner = kBeta * Vec::fmadd(kKappa, x_cube, x);
            Vec result = kHalf * x * (kOne + inner.tanh());
            result.storeu(dst + j);
        }

        // ── Remainder: scalar ──
        const float beta_s = std::sqrt(2.0f / M_PI);
        for (; j < hidden_dim; j++) {
            float x = src[j] + bias_ptr[j];
            float x3 = x * x * x;
            float inner = beta_s * (x + 0.044715f * x3);
            dst[j] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    return output;
}

// =================================================================
// dropout_forward: CPU/GPU dispatch
// =================================================================
DropoutForwardResult dropout_forward(const Tensor& input, float p) {
    if (!is_float(input.dtype())) {
        throw std::runtime_error("Dropout: only float dtypes supported!");
    }

    if (p == 0.0f) {
        Tensor ones = Tensor::ones(input.shape(), TensorOptions()
                                        .with_dtype(input.dtype())
                                        .with_device(input.device()));
        return {input, ones};
    }

    // Delegate to existing OwnTensor::dropout which handles both CPU and CUDA
    DropoutResult result = OwnTensor::dropout(input, p);
    return {result.output, result.mask};
}

// =================================================================
// fused_tril_softmax_forward: CPU/GPU dispatch
// =================================================================
Tensor fused_tril_softmax_forward(const Tensor& input, int64_t diagonal, double value) {
    if (!is_float(input.dtype())) {
        throw std::runtime_error("FusedTrilSoftmax: only float dtypes supported!");
    }

    if (input.device().is_cuda()) {
        // GPU: delegate to existing fused kernel in FusedKernels.cu
        // Note: OwnTensor::fused_tril_softmax takes non-const ref
        Tensor& input_mut = const_cast<Tensor&>(input);
        return OwnTensor::fused_tril_softmax(input_mut, diagonal, value);
    }

    // CPU fallback: compose tril + softmax separately
    Tensor triled = tril(input, diagonal, value);
    int64_t ndim = input.ndim();
    int64_t last_dim = ndim - 1;
    return softmax_forward(triled, last_dim);
}

// =================================================================
//
//                      BACKWARD PASS
//
// =================================================================

// =================================================================
// relu_backward: CPU/GPU dispatch
// grad_input = grad_output * ((input + |input|) > 0 ? 1 : 0)
// =================================================================
Tensor relu_backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.device().is_cuda()) {
        Tensor grad_input(input.shape(), grad_output.opts());
        device::set_cuda_device(grad_output.device().index);
        switch (grad_output.dtype()) {
            case Dtype::Float32:
                cuda::relu_backward_cuda(grad_output.data<float>(), input.data<float>(), grad_input.data<float>(), input.numel());
                break;
            case Dtype::Float16:
                cuda::relu_backward_cuda(grad_output.data<float16_t>(), input.data<float16_t>(), grad_input.data<float16_t>(), input.numel());
                break;
            case Dtype::Bfloat16:
                cuda::relu_backward_cuda(grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), input.numel());
                break;
            default:
                return grad_output * ((input + OwnTensor::abs(input)) > 0.0f);
        }
        return grad_input;
    }

    // CPU: (x+|x|) > 0 gives 1 where x>0, 0 elsewhere, NaN where NaN
    if (input.dtype() == Dtype::Float32) {
        const int64_t numel = input.numel();
        Tensor grad_input(input.shape(), TensorOptions().with_dtype(Dtype::Float32).with_device(input.device()));
        const float* g_ptr = grad_output.data<float>();
        const float* x_ptr = input.data<float>();
        float* out_ptr = grad_input.data<float>();

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();
        const Vec kZero(0.0f);

        const int num_threads = (numel >= 16384) ? omp_get_max_threads() : 1;
        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            const int64_t chunk = (numel + nthreads - 1) / nthreads;
            const int64_t start = tid * chunk;
            const int64_t end = std::min(start + chunk, numel);
            int64_t i = 0;
            const int64_t len = end - start;
            const float* g = g_ptr + start;
            const float* x = x_ptr + start;
            float* o = out_ptr + start;

            for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
                Vec xv = Vec::loadu(x + i);
                Vec gv = Vec::loadu(g + i);
                // mask: (x + |x|) > 0 → all positive x pass, NaN → NaN
                Vec abs_x = xv.abs();
                Vec sum = xv + abs_x;
                // Compare: sum > 0 → mask bits set
                __m256 mask = _mm256_cmp_ps(sum.values, kZero.values, _CMP_GT_OQ);
                Vec result = Vec(_mm256_and_ps(mask, gv.values));
                result.storeu(o + i);
            }
            for (; i < len; i++) {
                float xv = x[i];
                o[i] = (xv + fabsf(xv)) > 0.0f ? g[i] : 0.0f;
            }
        }
        return grad_input;
    }

    // Fallback
    Tensor mask = (input + OwnTensor::abs(input)) > 0.0f;
    return grad_output * mask;
}

// =================================================================
// gelu_backward: CPU/GPU dispatch
// gelu'(x) = 0.5*(1+tanh(u)) + 0.5*x*sech²(u)*du/dx
// =================================================================
Tensor gelu_backward(const Tensor& grad_output, const Tensor& input) {
    if (input.device().is_cuda()) {
        Tensor grad_input(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));
        device::set_cuda_device(input.device().index);
        switch (input.dtype()) {
            case Dtype::Float32:
                cuda::fused_gelu_backward_cuda(grad_output.data<float>(), input.data<float>(), grad_input.data<float>(), input.numel());
                break;
            case Dtype::Float16:
                cuda::fused_gelu_backward_cuda(grad_output.data<float16_t>(), input.data<float16_t>(), grad_input.data<float16_t>(), input.numel());
                break;
            case Dtype::Bfloat16:
                cuda::fused_gelu_backward_cuda(grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), input.numel());
                break;
            default:
                throw std::runtime_error("GeLU backward CUDA: unsupported dtype");
        }
        return grad_input;
    }

    // CPU fp32: fused AVX2 kernel
    if (input.dtype() == Dtype::Float32) {
        const int64_t numel = input.numel();
        Tensor grad_input(input.shape(), TensorOptions().with_dtype(Dtype::Float32).with_device(input.device()));
        const float* g_ptr = grad_output.data<float>();
        const float* x_ptr = input.data<float>();
        float* out_ptr = grad_input.data<float>();

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();
        const Vec kHalf(0.5f);
        const Vec kOne(1.0f);
        const Vec kBeta(std::sqrt(2.0f / M_PI));
        const Vec kKappa(0.044715f);
        const Vec kThreeKappa(3.0f * 0.044715f);

        const int num_threads = (numel >= 16384) ? omp_get_max_threads() : 1;
        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            const int64_t chunk = (numel + nthreads - 1) / nthreads;
            const int64_t start = tid * chunk;
            const int64_t end = std::min(start + chunk, numel);
            int64_t i = 0;
            const int64_t len = end - start;
            const float* g = g_ptr + start;
            const float* x = x_ptr + start;
            float* o = out_ptr + start;

            for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
                Vec xv = Vec::loadu(x + i);
                Vec gv = Vec::loadu(g + i);

                Vec x_sq = xv * xv;
                Vec x_cube = x_sq * xv;
                Vec u = kBeta * Vec::fmadd(kKappa, x_cube, xv);
                Vec tanh_u = u.tanh();
                Vec sech2_u = kOne - tanh_u * tanh_u;
                Vec du_dx = kBeta * Vec::fmadd(kThreeKappa, x_sq, kOne);
                Vec gelu_grad = kHalf * (kOne + tanh_u) + kHalf * xv * sech2_u * du_dx;
                Vec result = gv * gelu_grad;
                result.storeu(o + i);
            }

            const float beta_s = std::sqrt(2.0f / M_PI);
            const float c = 0.044715f;
            for (; i < len; i++) {
                float xv = x[i];
                float x2 = xv * xv;
                float x3 = x2 * xv;
                float u = beta_s * (xv + c * x3);
                float th = std::tanh(u);
                float sech2 = 1.0f - th * th;
                float du = beta_s * (1.0f + 3.0f * c * x2);
                o[i] = g[i] * (0.5f * (1.0f + th) + 0.5f * xv * sech2 * du);
            }
        }
        return grad_input;
    }

    // CPU fp16/bf16: F16C load → fp32 compute → F16C store, with AVX2+OpenMP
    if (input.dtype() == Dtype::Float16 || input.dtype() == Dtype::Bfloat16) {
        const int64_t numel = input.numel();
        Tensor grad_input(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));
        const bool is_fp16 = (input.dtype() == Dtype::Float16);
        constexpr int elem_bytes = 2;

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();
        const Vec kHalf(0.5f), kOne(1.0f);
        const Vec kBeta(std::sqrt(2.0f / M_PI)), kKappa(0.044715f), kThreeKappa(3.0f * 0.044715f);

        const char* g_base = reinterpret_cast<const char*>(grad_output.data());
        const char* x_base = reinterpret_cast<const char*>(input.data());
        char* o_base = reinterpret_cast<char*>(grad_input.data());

        auto load_vec = [&](const void* p) -> Vec {
            return is_fp16 ? vec::load_fp16_as_float(p) : vec::load_bf16_as_float(p);
        };
        auto store_vec = [&](void* p, const Vec& v) {
            is_fp16 ? vec::store_float_as_fp16(p, v) : vec::store_float_as_bf16(p, v);
        };

        const int num_threads = (numel >= 16384) ? omp_get_max_threads() : 1;
        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            const int64_t chunk = (numel + nthreads - 1) / nthreads;
            const int64_t start = tid * chunk;
            const int64_t end = std::min(start + chunk, numel);
            const int64_t len = end - start;
            const char* gp = g_base + start * elem_bytes;
            const char* xp = x_base + start * elem_bytes;
            char* op = o_base + start * elem_bytes;

            int64_t i = 0;
            for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
                Vec xv = load_vec(xp + i * elem_bytes);
                Vec gv = load_vec(gp + i * elem_bytes);
                Vec x_sq = xv * xv;
                Vec x_cube = x_sq * xv;
                Vec u = kBeta * Vec::fmadd(kKappa, x_cube, xv);
                Vec tanh_u = u.tanh();
                Vec sech2_u = kOne - tanh_u * tanh_u;
                Vec du_dx = kBeta * Vec::fmadd(kThreeKappa, x_sq, kOne);
                Vec gelu_grad = kHalf * (kOne + tanh_u) + kHalf * xv * sech2_u * du_dx;
                store_vec(op + i * elem_bytes, gv * gelu_grad);
            }
            const float beta_s = std::sqrt(2.0f / M_PI);
            const float c = 0.044715f;
            for (; i < len; i++) {
                float xv, gvf;
                if (is_fp16) {
                    uint16_t rx, rg;
                    std::memcpy(&rx, xp + i * 2, 2); std::memcpy(&rg, gp + i * 2, 2);
                    xv = _cvtsh_ss(rx); gvf = _cvtsh_ss(rg);
                } else {
                    uint16_t rx, rg;
                    std::memcpy(&rx, xp + i * 2, 2); std::memcpy(&rg, gp + i * 2, 2);
                    uint32_t bx = (uint32_t)rx << 16, bg = (uint32_t)rg << 16;
                    std::memcpy(&xv, &bx, 4); std::memcpy(&gvf, &bg, 4);
                }
                float x2 = xv*xv, x3 = x2*xv;
                float u = beta_s * (xv + c * x3);
                float th = std::tanh(u);
                float result = gvf * (0.5f*(1.0f+th) + 0.5f*xv*(1.0f-th*th)*beta_s*(1.0f+3.0f*c*x2));
                if (is_fp16) {
                    uint16_t out = _cvtss_sh(result, _MM_FROUND_TO_NEAREST_INT);
                    std::memcpy(op + i * 2, &out, 2);
                } else {
                    uint32_t bits; std::memcpy(&bits, &result, 4);
                    uint16_t out = (uint16_t)(bits >> 16);
                    std::memcpy(op + i * 2, &out, 2);
                }
            }
        }
        return grad_input;
    }

    // CPU fallback: tensor ops (float64 etc)
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float c = 0.044715f;
    Tensor x_sq = input * input;
    Tensor x_cubed = x_sq * input;
    Tensor u = sqrt_2_over_pi * (input + c * x_cubed);
    Tensor tanh_u = tanh(u);
    Tensor sech2_u = 1.0f - tanh_u * tanh_u;
    Tensor du_dx = sqrt_2_over_pi * (1.0f + 3.0f * c * x_sq);
    Tensor grad_x = 0.5f * (1.0f + tanh_u) + 0.5f * input * sech2_u * du_dx;
    return grad_output * grad_x;
}

// =================================================================
// sigmoid_backward: CPU/GPU dispatch
// grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
// =================================================================
Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
    if (grad_output.device().is_cuda()) {
        Tensor grad_input(output.shape(), grad_output.opts());
        device::set_cuda_device(grad_output.device().index);
        switch (grad_output.dtype()) {
            case Dtype::Float32:
                cuda::sigmoid_backward_cuda(grad_output.data<float>(), output.data<float>(), grad_input.data<float>(), output.numel());
                break;
            case Dtype::Float16:
                cuda::sigmoid_backward_cuda(grad_output.data<float16_t>(), output.data<float16_t>(), grad_input.data<float16_t>(), output.numel());
                break;
            case Dtype::Bfloat16:
                cuda::sigmoid_backward_cuda(grad_output.data<bfloat16_t>(), output.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), output.numel());
                break;
            default:
                return grad_output * output * (1.0f - output);
        }
        return grad_input;
    }
    return grad_output * output * (1.0f - output);
}

// =================================================================
// softmax_backward: CPU/GPU dispatch
// grad_input = s * (grad_output - sum(grad_output * s, dim))
// =================================================================
Tensor softmax_backward(const Tensor& grad_output, const Tensor& output, int64_t dim) {
    int64_t ndim = output.ndim();
    int64_t d = dim < 0 ? dim + ndim : dim;

    if (grad_output.device().is_cuda() && d == ndim - 1) {
        Tensor grad_input(output.shape(), grad_output.opts());
        int64_t cols = output.shape().dims.back();
        int64_t rows = output.numel() / cols;
        device::set_cuda_device(grad_output.device().index);
        switch (grad_output.dtype()) {
            case Dtype::Float32:
                cuda::softmax_backward_cuda(grad_output.data<float>(), output.data<float>(), grad_input.data<float>(), rows, cols);
                break;
            case Dtype::Float16:
                cuda::softmax_backward_cuda(grad_output.data<float16_t>(), output.data<float16_t>(), grad_input.data<float16_t>(), rows, cols);
                break;
            case Dtype::Bfloat16:
                cuda::softmax_backward_cuda(grad_output.data<bfloat16_t>(), output.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), rows, cols);
                break;
            default: {
                Tensor gs = grad_output * output;
                Tensor sum_gs = reduce_sum(gs, {dim}, true);
                return output * (grad_output - sum_gs);
            }
        }
        return grad_input;
    }

    // CPU fallback
    Tensor gs = grad_output * output;
    Tensor sum_gs = reduce_sum(gs, {dim}, true);
    return output * (grad_output - sum_gs);
}

// =================================================================
// dropout_backward: pure tensor ops (same on CPU and GPU)
// =================================================================
Tensor dropout_backward(const Tensor& grad_output, const Tensor& mask, float scale) {
    return grad_output * mask * scale;
}

// =================================================================
// swiglu_backward: CPU/GPU dispatch
// =================================================================
Tensor swiglu_backward(const Tensor& grad_output, const Tensor& input) {
    int64_t last = input.shape().dims.back();
    int64_t hidden = last / 2;
    int64_t rows = input.numel() / last;

    if (input.device().is_cuda()) {
        Tensor grad_input(input.shape(), input.opts());
        switch (input.dtype()) {
            case Dtype::Float32:
                cuda::swiglu_backward_cuda(grad_output.data<float>(), input.data<float>(), grad_input.data<float>(), rows, hidden);
                break;
            case Dtype::Float16:
                cuda::swiglu_backward_cuda(grad_output.data<float16_t>(), input.data<float16_t>(), grad_input.data<float16_t>(), rows, hidden);
                break;
            case Dtype::Bfloat16:
                cuda::swiglu_backward_cuda(grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), rows, hidden);
                break;
            default:
                throw std::runtime_error("SwiGLU backward: unsupported dtype");
        }
        return grad_input;
    }

    // CPU fallback
    int64_t ndim = input.ndim();
    Tensor& input_ref = const_cast<Tensor&>(input);
    Tensor A = input_ref.narrow(ndim - 1, 0, hidden);
    Tensor B = input_ref.narrow(ndim - 1, hidden, hidden);
    Tensor sig_A = sigmoid_forward(A);
    Tensor swish_A = A * sig_A;

    // dA = grad * B * (sig + A * sig * (1 - sig))
    Tensor dA = grad_output * B * (sig_A + A * sig_A * (1.0f - sig_A));
    // dB = grad * swish(A)
    Tensor dB = grad_output * swish_A;

    // Concatenate dA and dB along last dim
    // For now, write into grad_input directly
    Tensor grad_input(input.shape(), input.opts());
    // TODO: use a proper concat or narrow-write
    // For now tensor ops fallback
    throw std::runtime_error("SwiGLU backward CPU: not yet implemented (GPU only)");
}

// =================================================================
// fused_bias_gelu_backward: CPU/GPU dispatch
// =================================================================
FusedBiasGeLUBackwardResult fused_bias_gelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& bias) {
    int64_t hidden_dim = input.shape().dims.back();
    int64_t batch_size = input.numel() / hidden_dim;

    if (input.device().is_cuda() && input.dtype() == Dtype::Float32) {
        Tensor grad_input(input.shape(), input.opts());
        Tensor grad_bias = Tensor::zeros(bias.shape(), bias.opts());
        device::set_cuda_device(input.device().index);
        cuda::fused_bias_gelu_backward_cuda(
            grad_output.data<float>(), input.data<float>(), bias.data<float>(),
            grad_input.data<float>(), grad_bias.data<float>(),
            batch_size, hidden_dim);
        return {grad_input, grad_bias};
    }

    // CPU fp32: fused AVX2 kernel — compute grad_input and accumulate grad_bias in one pass
    if (input.dtype() == Dtype::Float32) {
        Tensor grad_input(input.shape(), input.opts());
        Tensor grad_bias = Tensor::zeros(bias.shape(), bias.opts());

        const float* g_ptr = grad_output.data<float>();
        const float* x_ptr = input.data<float>();
        const float* b_ptr = bias.data<float>();
        float* gi_ptr = grad_input.data<float>();
        float* gb_ptr = grad_bias.data<float>();

        using Vec = vec::Vectorized<float>;
        constexpr int VEC_SIZE = Vec::size();
        const Vec kHalf(0.5f), kOne(1.0f);
        const Vec kBeta(std::sqrt(2.0f / M_PI)), kKappa(0.044715f), kThreeKappa(3.0f * 0.044715f);

        // Thread-local bias accumulators to avoid contention
        const int num_threads = (batch_size * hidden_dim >= 16384) ? omp_get_max_threads() : 1;
        std::vector<std::vector<float>> local_gb(num_threads, std::vector<float>(hidden_dim, 0.0f));

        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            float* my_gb = local_gb[tid].data();

            #pragma omp for
            for (int64_t row = 0; row < batch_size; row++) {
                const float* g = g_ptr + row * hidden_dim;
                const float* x = x_ptr + row * hidden_dim;
                float* gi = gi_ptr + row * hidden_dim;

                int64_t j = 0;
                for (; j + VEC_SIZE <= hidden_dim; j += VEC_SIZE) {
                    Vec xv = Vec::loadu(x + j) + Vec::loadu(b_ptr + j);
                    Vec gv = Vec::loadu(g + j);

                    Vec x_sq = xv * xv;
                    Vec x_cube = x_sq * xv;
                    Vec u = kBeta * Vec::fmadd(kKappa, x_cube, xv);
                    Vec tanh_u = u.tanh();
                    Vec sech2_u = kOne - tanh_u * tanh_u;
                    Vec du_dx = kBeta * Vec::fmadd(kThreeKappa, x_sq, kOne);
                    Vec gelu_grad = kHalf * (kOne + tanh_u) + kHalf * xv * sech2_u * du_dx;
                    Vec gi_v = gv * gelu_grad;

                    gi_v.storeu(gi + j);
                    // Accumulate into thread-local bias grad
                    Vec acc = Vec::loadu(my_gb + j) + gi_v;
                    acc.storeu(my_gb + j);
                }

                const float beta_s = std::sqrt(2.0f / M_PI);
                const float c = 0.044715f;
                for (; j < hidden_dim; j++) {
                    float xv = x[j] + b_ptr[j];
                    float x2 = xv*xv, x3 = x2*xv;
                    float u = beta_s * (xv + c * x3);
                    float th = std::tanh(u);
                    float gg = 0.5f*(1.0f+th) + 0.5f*xv*(1.0f-th*th)*beta_s*(1.0f+3.0f*c*x2);
                    float gi_val = g[j] * gg;
                    gi[j] = gi_val;
                    my_gb[j] += gi_val;
                }
            }
        }

        // Merge thread-local accumulators
        for (int t = 0; t < num_threads; t++) {
            for (int64_t j = 0; j < hidden_dim; j++) {
                gb_ptr[j] += local_gb[t][j];
            }
        }

        return {grad_input, grad_bias};
    }

    // CPU fallback: tensor ops
    Tensor biased = input + bias;
    Tensor grad_input = gelu_backward(grad_output, biased);
    // Reduce all dims except last for bias grad
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < input.ndim() - 1; d++) reduce_dims.push_back(d);
    Tensor grad_bias = reduce_sum(grad_input, reduce_dims, false);
    return {grad_input, grad_bias};
}

} // namespace OwnTensor
