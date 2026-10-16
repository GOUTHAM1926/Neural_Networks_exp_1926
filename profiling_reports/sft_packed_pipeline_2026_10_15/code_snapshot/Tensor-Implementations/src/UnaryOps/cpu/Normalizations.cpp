#include "ops/UnaryOps/Normalizations.h"
#include "ops/helpers/LayerNormKernels.h"
#include "ops/helpers/FusedRMSRoPEKernels.h"
#include "ops/helpers/Vectorized.h"
#include "device/DeviceCore.h"
#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"
#include "core/TensorDataManip.h"
#include <cmath>
#include <omp.h>
#include <string>      // std::to_string in fused_rms_rope_positions_ptr (v2 positions check)
#include <vector>
#include <algorithm>
#include "profiler/RecordFunction.h"

namespace OwnTensor {

// =================================================================
// CPU Kernels — LayerNorm Forward (Welford one-pass + AVX2)
// =================================================================

// Templated CPU forward: works for float, float16_t, bfloat16_t
// Welford one-pass: computes mean + variance in a single data pass
// Then a second pass for normalize + gamma/beta application
// Total: 2 passes (was 3 before — mean, variance, normalize)
template<typename T>
static void cpu_layer_norm_forward_impl(
    const T* x, const T* gamma, const T* beta,
    T* y, float* mean_out, float* rstd_out,
    int64_t rows, int64_t cols, float eps)
{
    PROFILER_SCOPE("cpu_layer_norm_forward_impl");
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();  // 8 for AVX2 float

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const T* row_x = x + i * cols;
        T*       row_y = y + i * cols;

        // ── PASS 1: Welford one-pass mean + variance (AVX2 vectorized) ──
        // Upcast to float for accumulation, then reduce
        Vec v_mean(0.0f);
        Vec v_m2(0.0f);
        Vec v_count(0.0f);
        Vec v_one(1.0f);

        float s_mean = 0.0f, s_m2 = 0.0f;
        int64_t s_count = 0;

        int64_t j = 0;
        for (; j + VEC <= cols; j += VEC) {
            // Load and upcast to float
            float tmp[8];
            for (int k = 0; k < VEC; ++k)
                tmp[k] = static_cast<float>(row_x[j + k]);
            Vec val = Vec::loadu(tmp);

            // Welford update: vectorized across 8 lanes independently
            // Each lane tracks its own running mean/m2/count
            v_count = v_count + v_one;
            Vec delta = val - v_mean;
            v_mean = v_mean + delta / v_count;
            Vec delta2 = val - v_mean;
            v_m2 = v_m2 + delta * delta2;
        }
        // Scalar tail
        for (; j < cols; ++j) {
            float val = static_cast<float>(row_x[j]);
            s_count++;
            float delta = val - s_mean;
            s_mean += delta / s_count;
            float delta2 = val - s_mean;
            s_m2 += delta * delta2;
        }

        // Reduce 8 SIMD lanes into scalar using Welford merge
        // Extract lane values
        float lane_mean[8], lane_m2[8], lane_count[8];
        v_mean.storeu(lane_mean);
        v_m2.storeu(lane_m2);
        v_count.storeu(lane_count);

        float final_n = static_cast<float>(s_count);
        float final_mean = s_mean;
        float final_m2 = s_m2;

        for (int k = 0; k < VEC; ++k) {
            if (lane_count[k] == 0.0f) continue;
            float nb = lane_count[k];
            float total_n = final_n + nb;
            float delta = lane_mean[k] - final_mean;
            final_mean = final_mean + delta * (nb / total_n);
            final_m2 = final_m2 + lane_m2[k] + delta * delta * (final_n * nb / total_n);
            final_n = total_n;
        }

        float mu = final_mean;
        float var = final_m2 / cols;
        float rs = 1.0f / std::sqrt(var + eps);
        mean_out[i] = mu;
        rstd_out[i] = rs;

        // ── PASS 2: Normalize + gamma/beta (AVX2 vectorized) ──
        Vec v_mu(mu);
        Vec v_rs(rs);

        j = 0;
        for (; j + VEC <= cols; j += VEC) {
            // Load x
            float xtmp[8];
            for (int k = 0; k < VEC; ++k)
                xtmp[k] = static_cast<float>(row_x[j + k]);
            Vec vx = Vec::loadu(xtmp);

            // Normalize
            Vec norm = (vx - v_mu) * v_rs;

            // Load gamma/beta
            float gtmp[8], btmp[8];
            if (gamma) {
                for (int k = 0; k < VEC; ++k) gtmp[k] = static_cast<float>(gamma[j + k]);
            }
            if (beta) {
                for (int k = 0; k < VEC; ++k) btmp[k] = static_cast<float>(beta[j + k]);
            }

            Vec vg = gamma ? Vec::loadu(gtmp) : Vec(1.0f);
            Vec vb = beta  ? Vec::loadu(btmp) : Vec(0.0f);

            // result = norm * gamma + beta
            Vec result = Vec::fmadd(norm, vg, vb);

            // Store back (downcast from float)
            float out[8];
            result.storeu(out);
            for (int k = 0; k < VEC; ++k)
                row_y[j + k] = static_cast<T>(out[k]);
        }
        // Scalar tail
        for (; j < cols; ++j) {
            float val = (static_cast<float>(row_x[j]) - mu) * rs;
            float g = gamma ? static_cast<float>(gamma[j]) : 1.0f;
            float b = beta  ? static_cast<float>(beta[j])  : 0.0f;
            row_y[j] = static_cast<T>(val * g + b);
        }
    }
}

// Float32 specialization: fully vectorized load/store (no upcast needed)
template<>
void cpu_layer_norm_forward_impl<float>(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean_out, float* rstd_out,
    int64_t rows, int64_t cols, float eps)
{
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const float* row_x = x + i * cols;
        float*       row_y = y + i * cols;

        // ── PASS 1: Welford one-pass (AVX2, no upcast needed) ──
        Vec v_mean(0.0f), v_m2(0.0f), v_count(0.0f), v_one(1.0f);
        float s_mean = 0.0f, s_m2 = 0.0f;
        int64_t s_count = 0;

        int64_t j = 0;
        for (; j + VEC <= cols; j += VEC) {
            Vec val = Vec::loadu(row_x + j);
            v_count = v_count + v_one;
            Vec delta = val - v_mean;
            v_mean = v_mean + delta / v_count;
            Vec delta2 = val - v_mean;
            v_m2 = v_m2 + delta * delta2;
        }
        for (; j < cols; ++j) {
            float val = row_x[j];
            s_count++;
            float delta = val - s_mean;
            s_mean += delta / s_count;
            float delta2 = val - s_mean;
            s_m2 += delta * delta2;
        }

        // Merge SIMD lanes
        float lane_mean[8], lane_m2[8], lane_count[8];
        v_mean.storeu(lane_mean);
        v_m2.storeu(lane_m2);
        v_count.storeu(lane_count);

        float final_n = (float)s_count, final_mean = s_mean, final_m2 = s_m2;
        for (int k = 0; k < VEC; ++k) {
            if (lane_count[k] == 0.0f) continue;
            float nb = lane_count[k];
            float total = final_n + nb;
            float d = lane_mean[k] - final_mean;
            final_mean += d * (nb / total);
            final_m2 += lane_m2[k] + d * d * (final_n * nb / total);
            final_n = total;
        }

        float mu = final_mean;
        float rs = 1.0f / std::sqrt(final_m2 / cols + eps);
        mean_out[i] = mu;
        rstd_out[i] = rs;

        // ── PASS 2: Normalize + gamma/beta (direct AVX2) ──
        Vec v_mu(mu), v_rs(rs);
        j = 0;
        for (; j + VEC <= cols; j += VEC) {
            Vec vx = Vec::loadu(row_x + j);
            Vec norm = (vx - v_mu) * v_rs;
            Vec vg = gamma ? Vec::loadu(gamma + j) : Vec(1.0f);
            Vec vb = beta  ? Vec::loadu(beta + j)  : Vec(0.0f);
            Vec::fmadd(norm, vg, vb).storeu(row_y + j);
        }
        for (; j < cols; ++j) {
            float val = (row_x[j] - mu) * rs;
            float g = gamma ? gamma[j] : 1.0f;
            float b = beta  ? beta[j]  : 0.0f;
            row_y[j] = val * g + b;
        }
    }
}

// =================================================================
// CPU Kernels — LayerNorm Backward (AVX2 + OMP parallel reduction)
// =================================================================

template<typename T>
static void cpu_layer_norm_backward_impl(
    const T* grad_y, const T* x, const float* mean_ptr, const float* rstd_ptr,
    const T* gamma,
    T* grad_x, float* grad_weight, float* grad_bias,
    int64_t rows, int64_t cols)
{
    PROFILER_SCOPE("cpu_layer_norm_backward_impl");
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();
    bool need_weight_grad = (grad_weight != nullptr || grad_bias != nullptr);

    // ── PASS 1: grad_gamma/grad_beta — SKIP if not needed (like PyTorch grad_input_mask) ──
    const int max_threads = omp_get_max_threads();
    std::vector<float> gw_all, gb_all;
    if (need_weight_grad) {
        gw_all.resize(max_threads * cols, 0.0f);
        gb_all.resize(max_threads * cols, 0.0f);
    }

    if (need_weight_grad) {
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* gw_local = gw_all.data() + tid * cols;
        float* gb_local = gb_all.data() + tid * cols;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const T* row_gy = grad_y + i * cols;
            const T* row_x  = x + i * cols;
            float mu = mean_ptr[i];
            float rs = rstd_ptr[i];

            int64_t j = 0;
            Vec v_mu(mu), v_rs(rs);
            for (; j + VEC <= cols; j += VEC) {
                // Load and upcast
                float gy_tmp[8], x_tmp[8];
                for (int k = 0; k < VEC; ++k) {
                    gy_tmp[k] = static_cast<float>(row_gy[j + k]);
                    x_tmp[k]  = static_cast<float>(row_x[j + k]);
                }
                Vec vgy = Vec::loadu(gy_tmp);
                Vec vx  = Vec::loadu(x_tmp);
                Vec norm = (vx - v_mu) * v_rs;

                // Accumulate
                Vec gw_cur = Vec::loadu(gw_local + j);
                Vec gb_cur = Vec::loadu(gb_local + j);
                (gw_cur + vgy * norm).storeu(gw_local + j);
                (gb_cur + vgy).storeu(gb_local + j);
            }
            for (; j < cols; ++j) {
                float gy = static_cast<float>(row_gy[j]);
                float val = (static_cast<float>(row_x[j]) - mu) * rs;
                gw_local[j] += gy * val;
                gb_local[j] += gy;
            }
        }
    }

    // Reduce thread-local buffers into final grad_weight/grad_bias
    for (int64_t j = 0; j < cols; j += VEC) {
        int count = std::min((int64_t)VEC, cols - j);
        Vec sum_gw(0.0f), sum_gb(0.0f);
        for (int t = 0; t < max_threads; ++t) {
            sum_gw = sum_gw + Vec::loadu(gw_all.data() + t * cols + j, count);
            sum_gb = sum_gb + Vec::loadu(gb_all.data() + t * cols + j, count);
        }
        if (grad_weight) sum_gw.storeu(grad_weight + j, count);
        if (grad_bias) sum_gb.storeu(grad_bias + j, count);
    }
    } // end if (need_weight_grad)

    // ── PASS 2: grad_input — OMP parallel over rows, AVX2 inner ──
    if (grad_x == nullptr) return;  // skip if not needed
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const T* row_gy = grad_y + i * cols;
        const T* row_x  = x + i * cols;
        T*       row_gx = grad_x + i * cols;
        float mu = mean_ptr[i];
        float rs = rstd_ptr[i];

        // Sub-pass A: compute sum1 = sum(dy*gamma), sum2 = sum(dy*gamma*norm_x)
        float sum1 = 0.0f, sum2 = 0.0f;
        Vec v_mu(mu), v_rs(rs);
        Vec v_sum1(0.0f), v_sum2(0.0f);

        int64_t j = 0;
        for (; j + VEC <= cols; j += VEC) {
            float gy_tmp[8], x_tmp[8], g_tmp[8];
            for (int k = 0; k < VEC; ++k) {
                gy_tmp[k] = static_cast<float>(row_gy[j + k]);
                x_tmp[k]  = static_cast<float>(row_x[j + k]);
                g_tmp[k]  = gamma ? static_cast<float>(gamma[j + k]) : 1.0f;
            }
            Vec vgy = Vec::loadu(gy_tmp);
            Vec vx  = Vec::loadu(x_tmp);
            Vec vg  = Vec::loadu(g_tmp);
            Vec norm = (vx - v_mu) * v_rs;
            Vec dy_g = vgy * vg;
            v_sum1 = v_sum1 + dy_g;
            v_sum2 = v_sum2 + dy_g * norm;
        }
        sum1 = v_sum1.reduce_add();
        sum2 = v_sum2.reduce_add();
        for (; j < cols; ++j) {
            float gy = static_cast<float>(row_gy[j]);
            float g = gamma ? static_cast<float>(gamma[j]) : 1.0f;
            float norm = (static_cast<float>(row_x[j]) - mu) * rs;
            sum1 += gy * g;
            sum2 += gy * g * norm;
        }

        // Sub-pass B: compute grad_x
        float inv_cols = 1.0f / cols;
        Vec v_s1(sum1), v_s2(sum2), v_inv(inv_cols);

        j = 0;
        for (; j + VEC <= cols; j += VEC) {
            float gy_tmp[8], x_tmp[8], g_tmp[8];
            for (int k = 0; k < VEC; ++k) {
                gy_tmp[k] = static_cast<float>(row_gy[j + k]);
                x_tmp[k]  = static_cast<float>(row_x[j + k]);
                g_tmp[k]  = gamma ? static_cast<float>(gamma[j + k]) : 1.0f;
            }
            Vec vgy = Vec::loadu(gy_tmp);
            Vec vx  = Vec::loadu(x_tmp);
            Vec vg  = Vec::loadu(g_tmp);
            Vec norm = (vx - v_mu) * v_rs;
            Vec result = v_rs * (vgy * vg - (v_s1 + norm * v_s2) * v_inv);

            float out[8];
            result.storeu(out);
            for (int k = 0; k < VEC; ++k)
                row_gx[j + k] = static_cast<T>(out[k]);
        }
        for (; j < cols; ++j) {
            float gy = static_cast<float>(row_gy[j]);
            float g = gamma ? static_cast<float>(gamma[j]) : 1.0f;
            float norm = (static_cast<float>(row_x[j]) - mu) * rs;
            row_gx[j] = static_cast<T>(rs * (gy * g - (sum1 + norm * sum2) * inv_cols));
        }
    }
}

// Float32 specialization — direct AVX2 load/store without upcast
template<>
void cpu_layer_norm_backward_impl<float>(
    const float* grad_y, const float* x, const float* mean_ptr, const float* rstd_ptr,
    const float* gamma,
    float* grad_x, float* grad_weight, float* grad_bias,
    int64_t rows, int64_t cols)
{
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();
    bool need_weight_grad = (grad_weight != nullptr || grad_bias != nullptr);

    const int max_threads = omp_get_max_threads();
    std::vector<float> gw_all, gb_all;
    if (need_weight_grad) {
        gw_all.resize(max_threads * cols, 0.0f);
        gb_all.resize(max_threads * cols, 0.0f);
    }

    if (need_weight_grad) {
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* gw_local = gw_all.data() + tid * cols;
        float* gb_local = gb_all.data() + tid * cols;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const float* row_gy = grad_y + i * cols;
            const float* row_x  = x + i * cols;
            float mu = mean_ptr[i];
            float rs = rstd_ptr[i];
            Vec v_mu(mu), v_rs(rs);

            int64_t j = 0;
            for (; j + VEC <= cols; j += VEC) {
                Vec vgy = Vec::loadu(row_gy + j);
                Vec vx  = Vec::loadu(row_x + j);
                Vec norm = (vx - v_mu) * v_rs;
                Vec gw_cur = Vec::loadu(gw_local + j);
                Vec gb_cur = Vec::loadu(gb_local + j);
                (gw_cur + vgy * norm).storeu(gw_local + j);
                (gb_cur + vgy).storeu(gb_local + j);
            }
            for (; j < cols; ++j) {
                float norm = (row_x[j] - mu) * rs;
                gw_local[j] += row_gy[j] * norm;
                gb_local[j] += row_gy[j];
            }
        }
    }

    // Reduce
    for (int64_t j = 0; j < cols; j += VEC) {
        int count = std::min((int64_t)VEC, cols - j);
        Vec sum_gw(0.0f), sum_gb(0.0f);
        for (int t = 0; t < max_threads; ++t) {
            sum_gw = sum_gw + Vec::loadu(gw_all.data() + t * cols + j, count);
            sum_gb = sum_gb + Vec::loadu(gb_all.data() + t * cols + j, count);
        }
        if (grad_weight) sum_gw.storeu(grad_weight + j, count);
        if (grad_bias) sum_gb.storeu(grad_bias + j, count);
    }
    } // end if (need_weight_grad)

    // grad_input
    if (grad_x == nullptr) return;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const float* row_gy = grad_y + i * cols;
        const float* row_x  = x + i * cols;
        float* row_gx = grad_x + i * cols;
        float mu = mean_ptr[i], rs = rstd_ptr[i];
        Vec v_mu(mu), v_rs(rs), v_sum1(0.0f), v_sum2(0.0f);

        int64_t j = 0;
        float sum1 = 0.0f, sum2 = 0.0f;
        for (; j + VEC <= cols; j += VEC) {
            Vec vgy = Vec::loadu(row_gy + j);
            Vec vx  = Vec::loadu(row_x + j);
            Vec vg  = gamma ? Vec::loadu(gamma + j) : Vec(1.0f);
            Vec norm = (vx - v_mu) * v_rs;
            Vec dy_g = vgy * vg;
            v_sum1 = v_sum1 + dy_g;
            v_sum2 = v_sum2 + dy_g * norm;
        }
        sum1 = v_sum1.reduce_add();
        sum2 = v_sum2.reduce_add();
        for (; j < cols; ++j) {
            float g = gamma ? gamma[j] : 1.0f;
            float norm = (row_x[j] - mu) * rs;
            sum1 += row_gy[j] * g;
            sum2 += row_gy[j] * g * norm;
        }

        float inv_cols = 1.0f / cols;
        Vec v_s1(sum1), v_s2(sum2), v_inv(inv_cols);
        j = 0;
        for (; j + VEC <= cols; j += VEC) {
            Vec vgy = Vec::loadu(row_gy + j);
            Vec vx  = Vec::loadu(row_x + j);
            Vec vg  = gamma ? Vec::loadu(gamma + j) : Vec(1.0f);
            Vec norm = (vx - v_mu) * v_rs;
            Vec result = v_rs * (vgy * vg - (v_s1 + norm * v_s2) * v_inv);
            result.storeu(row_gx + j);
        }
        for (; j < cols; ++j) {
            float g = gamma ? gamma[j] : 1.0f;
            float norm = (row_x[j] - mu) * rs;
            row_gx[j] = rs * (row_gy[j] * g - (sum1 + norm * sum2) * inv_cols);
        }
    }
}

// =================================================================
// CPU Kernels — RMSNorm Forward (one-pass sum-of-squares + AVX2)
// =================================================================

template<typename T>
static void cpu_rms_norm_forward_impl(
    const T* x, const T* gamma,
    T* y, float* rstd_out,
    int64_t rows, int64_t cols, float eps)
{
    PROFILER_SCOPE("cpu_rms_norm_forward_impl");
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const T* row_x = x + i * cols;
        T*       row_y = y + i * cols;

        // ── PASS 1: one-pass sum of squares (AVX2) ──
        Vec v_sq_sum(0.0f);
        float s_sq_sum = 0.0f;

        int64_t j = 0;
        for (; j + VEC <= cols; j += VEC) {
            float tmp[8];
            for (int k = 0; k < VEC; ++k)
                tmp[k] = static_cast<float>(row_x[j + k]);
            Vec val = Vec::loadu(tmp);
            v_sq_sum = Vec::fmadd(val, val, v_sq_sum);
        }
        s_sq_sum = v_sq_sum.reduce_add();
        for (; j < cols; ++j) {
            float v = static_cast<float>(row_x[j]);
            s_sq_sum += v * v;
        }

        float rs = 1.0f / std::sqrt(s_sq_sum / cols + eps);
        rstd_out[i] = rs;

        // ── PASS 2: Normalize (x * rstd * gamma, no mean subtraction, no beta) ──
        Vec v_rs(rs);
        j = 0;
        for (; j + VEC <= cols; j += VEC) {
            float xtmp[8], gtmp[8];
            for (int k = 0; k < VEC; ++k) {
                xtmp[k] = static_cast<float>(row_x[j + k]);
                gtmp[k] = gamma ? static_cast<float>(gamma[j + k]) : 1.0f;
            }
            Vec vx = Vec::loadu(xtmp);
            Vec vg = Vec::loadu(gtmp);
            Vec result = vx * v_rs * vg;
            float out[8];
            result.storeu(out);
            for (int k = 0; k < VEC; ++k)
                row_y[j + k] = static_cast<T>(out[k]);
        }
        for (; j < cols; ++j) {
            float v = static_cast<float>(row_x[j]);
            float g = gamma ? static_cast<float>(gamma[j]) : 1.0f;
            row_y[j] = static_cast<T>(v * rs * g);
        }
    }
}

// Float32 specialization
template<>
void cpu_rms_norm_forward_impl<float>(
    const float* x, const float* gamma,
    float* y, float* rstd_out,
    int64_t rows, int64_t cols, float eps)
{
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const float* row_x = x + i * cols;
        float*       row_y = y + i * cols;

        Vec v_sq(0.0f);
        float s_sq = 0.0f;
        int64_t j = 0;
        for (; j + VEC <= cols; j += VEC) {
            Vec val = Vec::loadu(row_x + j);
            v_sq = Vec::fmadd(val, val, v_sq);
        }
        s_sq = v_sq.reduce_add();
        for (; j < cols; ++j) s_sq += row_x[j] * row_x[j];

        float rs = 1.0f / std::sqrt(s_sq / cols + eps);
        rstd_out[i] = rs;

        Vec v_rs(rs);
        j = 0;
        for (; j + VEC <= cols; j += VEC) {
            Vec vx = Vec::loadu(row_x + j);
            Vec vg = gamma ? Vec::loadu(gamma + j) : Vec(1.0f);
            (vx * v_rs * vg).storeu(row_y + j);
        }
        for (; j < cols; ++j) {
            float g = gamma ? gamma[j] : 1.0f;
            row_y[j] = row_x[j] * rs * g;
        }
    }
}

// =================================================================
// CPU Kernels — RMSNorm Backward
// =================================================================

template<typename T>
static void cpu_rms_norm_backward_impl(
    const T* grad_y, const T* x, const float* rstd_ptr,
    const T* gamma,
    T* grad_x, float* grad_weight,
    int64_t rows, int64_t cols)
{
    PROFILER_SCOPE("cpu_rms_norm_backward_impl");
    using Vec = vec::Vectorized<float>;
    constexpr int VEC = Vec::size();

    // grad_weight: OMP parallel reduction
    const int max_threads = omp_get_max_threads();
    std::vector<float> gw_all(max_threads * cols, 0.0f);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* gw_local = gw_all.data() + tid * cols;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const T* row_gy = grad_y + i * cols;
            const T* row_x  = x + i * cols;
            float rs = rstd_ptr[i];

            for (int64_t j = 0; j < cols; ++j) {
                float gy = static_cast<float>(row_gy[j]);
                float v  = static_cast<float>(row_x[j]);
                gw_local[j] += gy * v * rs;  // d_gamma = dy * (x * rstd)
            }
        }
    }

    for (int64_t j = 0; j < cols; j += VEC) {
        int count = std::min((int64_t)VEC, cols - j);
        Vec sum_gw(0.0f);
        for (int t = 0; t < max_threads; ++t)
            sum_gw = sum_gw + Vec::loadu(gw_all.data() + t * cols + j, count);
        sum_gw.storeu(grad_weight + j, count);
    }

    // grad_input: RMSNorm backward formula
    // dx = rstd * (dy*gamma - x * rstd^2 * sum(dy*gamma*x) / cols)
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const T* row_gy = grad_y + i * cols;
        const T* row_x  = x + i * cols;
        T* row_gx = grad_x + i * cols;
        float rs = rstd_ptr[i];

        // Sum(dy * gamma * x)
        float dot = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float gy = static_cast<float>(row_gy[j]);
            float g  = gamma ? static_cast<float>(gamma[j]) : 1.0f;
            float v  = static_cast<float>(row_x[j]);
            dot += gy * g * v;
        }

        float inv_cols = 1.0f / cols;
        float rs3 = rs * rs * rs;  // rstd^3
        for (int64_t j = 0; j < cols; ++j) {
            float gy = static_cast<float>(row_gy[j]);
            float g  = gamma ? static_cast<float>(gamma[j]) : 1.0f;
            float v  = static_cast<float>(row_x[j]);
            row_gx[j] = static_cast<T>(rs * gy * g - rs3 * v * dot * inv_cols);
        }
    }
}

// =================================================================
// Public API — LayerNorm Forward (CPU/GPU dispatch)
// =================================================================
LayerNormForwardResult layer_norm_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int normalized_shape,
    float eps)
{
    PROFILER_SCOPE("layer_norm_forward");
    Shape x_shape = input.shape();
    Tensor output = Tensor(x_shape, input.opts());

    int64_t total_ele = input.numel();
    int64_t cols = normalized_shape;
    int64_t rows = total_ele / cols;

    if (x_shape.dims.back() != cols) {
        throw std::runtime_error("LayerNorm: last dim must match normalized_shape");
    }

    TensorOptions stat_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(input.device())
        .with_req_grad(false);
    Tensor mean = Tensor(Shape{{rows}}, stat_opts);
    Tensor rstd = Tensor(Shape{{rows}}, stat_opts);

    if (input.device().is_cuda()) {
        device::set_cuda_device(input.device().index);

        if (input.dtype() == Dtype::Float16) {
            const float16_t* gp = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            const float16_t* bp = bias.is_valid() ? bias.data<float16_t>() : nullptr;
            cuda::layer_norm_forward_cuda(
                input.data<float16_t>(), gp, bp,
                output.data<float16_t>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        } else if (input.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gp = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            const bfloat16_t* bp = bias.is_valid() ? bias.data<bfloat16_t>() : nullptr;
            cuda::layer_norm_forward_cuda(
                input.data<bfloat16_t>(), gp, bp,
                output.data<bfloat16_t>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        } else {
            const float* gp = weight.is_valid() ? weight.data<float>() : nullptr;
            const float* bp = bias.is_valid() ? bias.data<float>() : nullptr;
            cuda::layer_norm_forward_cuda(
                input.data<float>(), gp, bp,
                output.data<float>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        }
    } else {
        if (input.dtype() == Dtype::Float16) {
            const float16_t* gp = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            const float16_t* bp = bias.is_valid() ? bias.data<float16_t>() : nullptr;
            cpu_layer_norm_forward_impl(input.data<float16_t>(), gp, bp,
                output.data<float16_t>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        } else if (input.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gp = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            const bfloat16_t* bp = bias.is_valid() ? bias.data<bfloat16_t>() : nullptr;
            cpu_layer_norm_forward_impl(input.data<bfloat16_t>(), gp, bp,
                output.data<bfloat16_t>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        } else {
            const float* gp = weight.is_valid() ? weight.data<float>() : nullptr;
            const float* bp = bias.is_valid() ? bias.data<float>() : nullptr;
            cpu_layer_norm_forward_impl(input.data<float>(), gp, bp,
                output.data<float>(), mean.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        }
    }

    return {output, mean, rstd};
}

// =================================================================
// Public API — LayerNorm Backward (CPU/GPU dispatch)
// =================================================================
LayerNormBackwardResult layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int normalized_shape,
    float eps,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias)
{
    PROFILER_SCOPE("layer_norm_backward");
    (void)eps;
    int64_t total_ele = input.numel();
    int64_t cols = normalized_shape;
    int64_t rows = total_ele / cols;

    Tensor grad_input;
    if (need_grad_input)
        grad_input = Tensor::empty(input.shape(), input.opts());

    TensorOptions wg_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(input.device());
    Tensor grad_weight, grad_bias;
    if (need_grad_weight)
        grad_weight = weight.is_valid()
            ? Tensor::empty(weight.shape(), wg_opts)
            : Tensor::empty(Shape{{cols}}, wg_opts);
    if (need_grad_bias)
        grad_bias = weight.is_valid()
            ? Tensor::empty(weight.shape(), wg_opts)
            : Tensor::empty(Shape{{cols}}, wg_opts);

    if (grad_output.device().is_cuda()) {
        // Pass nullptr for gradients we don't need — CUDA launcher skips kernel
        float* gw_ptr = need_grad_weight ? grad_weight.data<float>() : nullptr;
        float* gb_ptr = need_grad_bias ? grad_bias.data<float>() : nullptr;
        float* gx_ptr_f = nullptr; float16_t* gx_ptr_h = nullptr; bfloat16_t* gx_ptr_b = nullptr;

        if (grad_output.dtype() == Dtype::Float32) {
            const float* gamma = weight.is_valid() ? weight.data<float>() : nullptr;
            cuda::layer_norm_backward_cuda(
                grad_output.data<float>(), input.data<float>(),
                mean.data<float>(), rstd.data<float>(), gamma,
                need_grad_input ? grad_input.data<float>() : nullptr,
                gw_ptr, gb_ptr, rows, cols);
        } else if (grad_output.dtype() == Dtype::Float16) {
            const float16_t* gamma = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            cuda::layer_norm_backward_cuda(
                grad_output.data<float16_t>(), input.data<float16_t>(),
                mean.data<float>(), rstd.data<float>(), gamma,
                need_grad_input ? grad_input.data<float16_t>() : nullptr,
                gw_ptr, gb_ptr, rows, cols);
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gamma = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            cuda::layer_norm_backward_cuda(
                grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(),
                mean.data<float>(), rstd.data<float>(), gamma,
                need_grad_input ? grad_input.data<bfloat16_t>() : nullptr,
                gw_ptr, gb_ptr, rows, cols);
        }
    } else {
        // CPU: pass nullptr for grads we don't need, kernel skips that work
        float* gw = (need_grad_weight && grad_weight.is_valid()) ? grad_weight.data<float>() : nullptr;
        float* gb = (need_grad_bias && grad_bias.is_valid()) ? grad_bias.data<float>() : nullptr;

        if (grad_output.dtype() == Dtype::Float32) {
            const float* gamma = weight.is_valid() ? weight.data<float>() : nullptr;
            cpu_layer_norm_backward_impl(
                grad_output.data<float>(), input.data<float>(),
                mean.data<float>(), rstd.data<float>(), gamma,
                need_grad_input ? grad_input.data<float>() : nullptr, gw, gb,
                rows, cols);
        } else if (grad_output.dtype() == Dtype::Float16) {
            const float16_t* gamma = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            cpu_layer_norm_backward_impl(
                grad_output.data<float16_t>(), input.data<float16_t>(),
                mean.data<float>(), rstd.data<float>(), gamma,
                need_grad_input ? grad_input.data<float16_t>() : nullptr, gw, gb,
                rows, cols);
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gamma = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            cpu_layer_norm_backward_impl(
                grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(),
                mean.data<float>(), rstd.data<float>(), gamma,
                need_grad_input ? grad_input.data<bfloat16_t>() : nullptr, gw, gb,
                rows, cols);
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

// =================================================================
// Public API — RMSNorm Forward (CPU/GPU dispatch)
// =================================================================
RMSNormForwardResult rms_norm_forward(
    const Tensor& input,
    const Tensor& weight,
    int normalized_shape,
    float eps)
{
    PROFILER_SCOPE("rms_norm_forward");
    Shape x_shape = input.shape();
    Tensor output = Tensor(x_shape, input.opts());

    int64_t cols = normalized_shape;
    int64_t rows = input.numel() / cols;

    if (x_shape.dims.back() != cols) {
        throw std::runtime_error("RMSNorm: last dim must match normalized_shape");
    }

    TensorOptions stat_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(input.device())
        .with_req_grad(false);
    Tensor rstd = Tensor(Shape{{rows}}, stat_opts);

    if (input.device().is_cuda()) {
        device::set_cuda_device(input.device().index);

        if (input.dtype() == Dtype::Float16) {
            const float16_t* gp = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            cuda::rms_norm_forward_cuda(
                input.data<float16_t>(), gp,
                output.data<float16_t>(), rstd.data<float>(),
                rows, cols, eps);
        } else if (input.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gp = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            cuda::rms_norm_forward_cuda(
                input.data<bfloat16_t>(), gp,
                output.data<bfloat16_t>(), rstd.data<float>(),
                rows, cols, eps);
        } else {
            const float* gp = weight.is_valid() ? weight.data<float>() : nullptr;
            cuda::rms_norm_forward_cuda(
                input.data<float>(), gp,
                output.data<float>(), rstd.data<float>(),
                rows, cols, eps);
        }
    } else {
        if (input.dtype() == Dtype::Float16) {
            const float16_t* gp = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            cpu_rms_norm_forward_impl(input.data<float16_t>(), gp,
                output.data<float16_t>(), rstd.data<float>(), rows, cols, eps);
        } else if (input.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gp = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            cpu_rms_norm_forward_impl(input.data<bfloat16_t>(), gp,
                output.data<bfloat16_t>(), rstd.data<float>(), rows, cols, eps);
        } else {
            const float* gp = weight.is_valid() ? weight.data<float>() : nullptr;
            cpu_rms_norm_forward_impl(input.data<float>(), gp,
                output.data<float>(), rstd.data<float>(), rows, cols, eps);
        }
    }

    return {output, rstd};
}

// =================================================================
// Public API — RMSNorm Backward (CPU/GPU dispatch)
// =================================================================
RMSNormBackwardResult rms_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& rstd,
    const Tensor& weight,
    int normalized_shape,
    float eps,
    bool need_grad_input,
    bool need_grad_weight)
{
    PROFILER_SCOPE("rms_norm_backward");
    (void)eps;
    int64_t cols = normalized_shape;
    int64_t rows = input.numel() / cols;

    Tensor grad_input;
    if (need_grad_input)
        grad_input = Tensor::empty(input.shape(), input.opts());
    TensorOptions wg_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(input.device());
    Tensor grad_weight = weight.is_valid()
        ? Tensor::empty(weight.shape(), wg_opts)
        : Tensor::empty(Shape{{cols}}, wg_opts);

    if (grad_output.device().is_cuda()) {
        float* gw_ptr = grad_weight.data<float>();

        if (grad_output.dtype() == Dtype::Float32) {
            const float* gamma = weight.is_valid() ? weight.data<float>() : nullptr;
            cuda::rms_norm_backward_cuda(
                grad_output.data<float>(), input.data<float>(),
                rstd.data<float>(), gamma,
                grad_input.data<float>(), gw_ptr, rows, cols);
        } else if (grad_output.dtype() == Dtype::Float16) {
            const float16_t* gamma = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            cuda::rms_norm_backward_cuda(
                grad_output.data<float16_t>(), input.data<float16_t>(),
                rstd.data<float>(), gamma,
                grad_input.data<float16_t>(), gw_ptr, rows, cols);
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gamma = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            cuda::rms_norm_backward_cuda(
                grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(),
                rstd.data<float>(), gamma,
                grad_input.data<bfloat16_t>(), gw_ptr, rows, cols);
        }
    } else {
        if (grad_output.dtype() == Dtype::Float32) {
            const float* gamma = weight.is_valid() ? weight.data<float>() : nullptr;
            cpu_rms_norm_backward_impl(
                grad_output.data<float>(), input.data<float>(),
                rstd.data<float>(), gamma,
                grad_input.data<float>(), grad_weight.data<float>(), rows, cols);
        } else if (grad_output.dtype() == Dtype::Float16) {
            const float16_t* gamma = weight.is_valid() ? weight.data<float16_t>() : nullptr;
            cpu_rms_norm_backward_impl(
                grad_output.data<float16_t>(), input.data<float16_t>(),
                rstd.data<float>(), gamma,
                grad_input.data<float16_t>(), grad_weight.data<float>(), rows, cols);
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            const bfloat16_t* gamma = weight.is_valid() ? weight.data<bfloat16_t>() : nullptr;
            cpu_rms_norm_backward_impl(
                grad_output.data<bfloat16_t>(), input.data<bfloat16_t>(),
                rstd.data<float>(), gamma,
                grad_input.data<bfloat16_t>(), grad_weight.data<float>(), rows, cols);
        }
    }

    return {grad_input, grad_weight};
}

// v2 packing (added 2026-10-10): validates the optional per-token positions tensor and
// returns its device pointer, or nullptr when it is not given (= the original behaviour,
// position = t + pos_offset). Rules: int32, CUDA, same device as q, contiguous [B, T], and
// pos_offset == 0 (positions and an offset together would be ambiguous). Value range
// (0 <= p < cache length) is NOT checked here: that would need a device->host copy every
// call; the loader checks it once on the host (same split as torchtitan's async check,
// models/common/rope.py:25-38).
static const int32_t* fused_rms_rope_positions_ptr(const Tensor& positions, const Tensor& q,
                                                   int B, int T, int64_t pos_offset,
                                                   const char* who) {
    if (!positions.is_valid()) return nullptr;
    const std::string w(who);
    if (!positions.device().is_cuda() || positions.device().index != q.device().index)
        throw std::runtime_error(w + ": positions must be a CUDA tensor on q's device");
    if (positions.dtype() != Dtype::Int32)
        throw std::runtime_error(w + ": positions must be int32");
    if (positions.ndim() != 2 || positions.shape().dims[0] != B || positions.shape().dims[1] != T)
        throw std::runtime_error(w + ": positions must have shape [B, T] = [" +
                                 std::to_string(B) + ", " + std::to_string(T) + "]");
    if (!positions.is_contiguous())
        throw std::runtime_error(w + ": positions must be contiguous");
    if (pos_offset != 0)
        throw std::runtime_error(w + ": pass either positions or a nonzero pos_offset, not both");
    return positions.data<int32_t>();
}

// =================================================================
// Public API — Fused RMSNorm + RoPE Forward
// =================================================================
FusedRMSRoPEForwardResult fused_rms_rope_forward(
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int64_t pos_offset,
    float eps,
    bool interleaved,
    const Tensor& positions)
{
    PROFILER_SCOPE("fused_rms_rope_forward");
    if (!q.device().is_cuda() || !k.device().is_cuda()) {
        throw std::runtime_error("fused_rms_rope_forward: q and k must be CUDA tensors");
    }
    if (q.ndim() != 4 || k.ndim() != 4) {
        throw std::runtime_error("fused_rms_rope_forward: q and k must be 4D tensors [B, H, T, hd]");
    }

    int B   = static_cast<int>(q.shape().dims[0]);
    int Hq  = static_cast<int>(q.shape().dims[1]);
    int T   = static_cast<int>(q.shape().dims[2]);
    int hd  = static_cast<int>(q.shape().dims[3]);
    int Hkv = static_cast<int>(k.shape().dims[1]);

    if (k.shape().dims[0] != B || k.shape().dims[2] != T || k.shape().dims[3] != hd) {
        throw std::runtime_error("fused_rms_rope_forward: shape mismatch between q and k");
    }
    if (!q_gamma.is_valid() || q_gamma.shape().dims[0] != hd || !k_gamma.is_valid() || k_gamma.shape().dims[0] != hd) {
        throw std::runtime_error("fused_rms_rope_forward: q_gamma and k_gamma must have shape [hd]");
    }

    Tensor q_out = Tensor::empty(q.shape(), q.opts());
    Tensor k_out = Tensor::empty(k.shape(), k.opts());

    TensorOptions stat_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(q.device())
        .with_req_grad(false);

    Tensor q_rstd = Tensor::empty(Shape{{B, Hq, T}}, stat_opts);
    Tensor k_rstd = Tensor::empty(Shape{{B, Hkv, T}}, stat_opts);

    // v2: nullptr unless a positions tensor is given (then validated here).
    const int32_t* pos_ptr =
        fused_rms_rope_positions_ptr(positions, q, B, T, pos_offset, "fused_rms_rope_forward");

    device::set_cuda_device(q.device().index);
    cuda::fused_qk_norm_rope_forward(
        q.data<float>(), k.data<float>(),
        q_out.data<float>(), k_out.data<float>(),
        q_gamma.data<float>(), k_gamma.data<float>(),
        cos_sin_cache.data<float>(),
        q_rstd.data<float>(), k_rstd.data<float>(),
        B, Hq, Hkv, T, hd,
        static_cast<int>(pos_offset), eps, interleaved,
        /*stream=*/0, pos_ptr);

    return {q_out, k_out, q_rstd, k_rstd};
}

// =================================================================
// Public API — Fused RMSNorm + RoPE Backward
// =================================================================
FusedRMSRoPEBackwardResult fused_rms_rope_backward(
    const Tensor& grad_q,
    const Tensor& grad_k,
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    const Tensor& q_rstd,
    const Tensor& k_rstd,
    int64_t pos_offset,
    bool interleaved,
    const Tensor& positions)
{
    PROFILER_SCOPE("fused_rms_rope_backward");
    int B   = static_cast<int>(q.shape().dims[0]);
    int Hq  = static_cast<int>(q.shape().dims[1]);
    int T   = static_cast<int>(q.shape().dims[2]);
    int hd  = static_cast<int>(q.shape().dims[3]);
    int Hkv = static_cast<int>(k.shape().dims[1]);

    // v2: must be the same positions the forward used (FusedRMSRoPEBackward saves them).
    const int32_t* pos_ptr =
        fused_rms_rope_positions_ptr(positions, q, B, T, pos_offset, "fused_rms_rope_backward");

    Tensor grad_q_in = Tensor::empty(q.shape(), q.opts());
    Tensor grad_k_in = Tensor::empty(k.shape(), k.opts());

    TensorOptions weight_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(q.device());

    Tensor grad_q_gamma = Tensor::empty(q_gamma.shape(), weight_opts);
    Tensor grad_k_gamma = Tensor::empty(k_gamma.shape(), weight_opts);

    device::set_cuda_device(q.device().index);
    cuda::fused_qk_norm_rope_backward(
        grad_q.data<float>(), grad_k.data<float>(),
        q.data<float>(), k.data<float>(),
        q_gamma.data<float>(), k_gamma.data<float>(),
        cos_sin_cache.data<float>(),
        q_rstd.data<float>(), k_rstd.data<float>(),
        grad_q_in.data<float>(), grad_k_in.data<float>(),
        grad_q_gamma.data<float>(), grad_k_gamma.data<float>(),
        B, Hq, Hkv, T, hd,
        static_cast<int>(pos_offset), interleaved,
        /*stream=*/0, pos_ptr);

    return {grad_q_in, grad_k_in, grad_q_gamma, grad_k_gamma};
}


} // namespace OwnTensor

