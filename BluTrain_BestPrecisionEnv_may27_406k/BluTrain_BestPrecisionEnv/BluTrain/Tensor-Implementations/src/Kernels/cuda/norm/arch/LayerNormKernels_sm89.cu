#include "ops/helpers/LayerNormKernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// =================================================================================
// SM89 (Ada) Optimized LayerNorm Kernels
//
// Differences from generic (sm86) kernel:
//   1. 512 threads per block (vs 256) — better occupancy on Ada's larger register file
//   2. __launch_bounds__(512) — helps compiler optimize register allocation for Ada
//   3. Same Welford one-pass algorithm + float4 vectorization as generic
//   4. Gamma/beta backward: wider grid to fill 142 SMs (vs 28 on 3060)
//
// Why NOT separate algorithms: LayerNorm uses float math + warp shuffles.
// These are identical instructions on sm86 and sm89. No PTX differences.
// PyTorch also uses one kernel for all archs — only adapts launch config.
// =================================================================================

namespace OwnTensor {
namespace cuda {

// ── Helpers ──
template<typename T> __device__ __forceinline__ float to_float_sm89(T val);
template<> __device__ __forceinline__ float to_float_sm89(float val) { return val; }
template<> __device__ __forceinline__ float to_float_sm89(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float_sm89(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float_sm89(float val);
template<> __device__ __forceinline__ float from_float_sm89(float val) { return val; }
template<> __device__ __forceinline__ __half from_float_sm89(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float_sm89(float val) { return __float2bfloat16(val); }

template<typename T>
__inline__ __device__ T warpReduceSum_sm89(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename AccT>
struct WelfordData_sm89 { AccT n, mu, m2; };

template<typename AccT>
__device__ __inline__ WelfordData_sm89<AccT> welford_merge_sm89(
    WelfordData_sm89<AccT> a, WelfordData_sm89<AccT> b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordData_sm89<AccT> res;
    res.n  = a.n + b.n;
    AccT delta = b.mu - a.mu;
    res.mu = a.mu + delta * (b.n / res.n);
    res.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / res.n);
    return res;
}

// =================================================================================
// Forward: Welford one-pass + float4 vectorized — 512 threads for Ada
// =================================================================================
template<typename T, typename AccT>
__global__ __launch_bounds__(512)
void layer_norm_forward_sm89_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ y, AccT* __restrict__ mean_out, AccT* __restrict__ rstd_out,
    int cols, AccT eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row_x = x + row * cols;
    T* row_y = y + row * cols;

    AccT local_sum = 0, local_sq_sum = 0;
    int local_count = 0;

    if constexpr (std::is_same_v<T, float>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 v = x_vec[i];
            local_sum += v.x + v.y + v.z + v.w;
            local_sq_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            local_count += 4;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            AccT val = row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __half22float2(h[k]);
                local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x*(AccT)f.x + (AccT)f.y*(AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __bfloat1622float2(h[k]);
                local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x*(AccT)f.x + (AccT)f.y*(AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    }

    // Welford state
    AccT n = (AccT)local_count;
    WelfordData_sm89<AccT> state = {
        n,
        (n > 0) ? local_sum / n : (AccT)0,
        (n > 0) ? (local_sq_sum - local_sum * local_sum / n) : (AccT)0
    };

    // Warp reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        WelfordData_sm89<AccT> other;
        other.n  = __shfl_down_sync(0xffffffff, state.n,  offset);
        other.mu = __shfl_down_sync(0xffffffff, state.mu, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        state = welford_merge_sm89(state, other);
    }

    // Block reduction — 512/32 = 16 warps
    __shared__ WelfordData_sm89<AccT> s_welford[16];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) s_welford[warp_id] = state;
    __syncthreads();

    if (tid == 0) {
        WelfordData_sm89<AccT> fs = s_welford[0];
        for (int i = 1; i < 16; ++i) fs = welford_merge_sm89(fs, s_welford[i]);
        s_welford[0] = fs;
    }
    __syncthreads();

    const AccT mu   = s_welford[0].mu;
    const AccT rstd = rsqrtf(s_welford[0].m2 / cols + eps);
    if (tid == 0) { if (mean_out) mean_out[row] = mu; if (rstd_out) rstd_out[row] = rstd; }

    // Vectorized normalize
    if constexpr (std::is_same_v<T, float>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xv = x_vec[i];
            float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1,1,1,1);
            float4 bv = beta  ? reinterpret_cast<const float4*>(beta)[i]  : make_float4(0,0,0,0);
            y_vec[i] = {(xv.x-mu)*rstd*gv.x+bv.x, (xv.y-mu)*rstd*gv.y+bv.y,
                        (xv.z-mu)*rstd*gv.z+bv.z, (xv.w-mu)*rstd*gv.w+bv.w};
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b);
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __half2* xh = reinterpret_cast<const __half2*>(&xraw);
            float4 graw, braw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            if (beta)  braw = reinterpret_cast<const float4*>(beta)[i];
            const __half2* gh = gamma ? reinterpret_cast<const __half2*>(&graw) : nullptr;
            const __half2* bh = beta  ? reinterpret_cast<const __half2*>(&braw) : nullptr;
            float4 yraw; __half2* yh = reinterpret_cast<__half2*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __half22float2(xh[k]);
                xf.x = (xf.x-mu)*rstd; xf.y = (xf.y-mu)*rstd;
                if (gh) { float2 gf = __half22float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if (bh) { float2 bf = __half22float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                yh[k] = __float22half2_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b);
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xraw);
            float4 graw, braw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            if (beta)  braw = reinterpret_cast<const float4*>(beta)[i];
            const __nv_bfloat162* gh = gamma ? reinterpret_cast<const __nv_bfloat162*>(&graw) : nullptr;
            const __nv_bfloat162* bh = beta  ? reinterpret_cast<const __nv_bfloat162*>(&braw) : nullptr;
            float4 yraw; __nv_bfloat162* yh = reinterpret_cast<__nv_bfloat162*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xh[k]);
                xf.x = (xf.x-mu)*rstd; xf.y = (xf.y-mu)*rstd;
                if (gh) { float2 gf = __bfloat1622float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if (bh) { float2 bf = __bfloat1622float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                yh[k] = __float22bfloat162_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b);
        }
    }
}

// =================================================================================
// Backward A: Gamma/Beta — templated for all dtypes, wider grid for Ada
// =================================================================================
template<typename T>
__global__ __launch_bounds__(256)
void ln_backward_gamma_beta_sm89_kernel(
    const T* __restrict__ grad_y, const T* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    float* __restrict__ grad_gamma, float* __restrict__ grad_beta,
    int rows, int cols)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float s_dg[8][32], s_db[8][32];

    #pragma unroll 4
    for (int cb = blockIdx.x * 32; cb < cols; cb += gridDim.x * 32) {
        int col = cb + tx;
        float dg = 0, db = 0;
        if (col < cols) {
            for (int r = blockIdx.y * 8 + ty; r < rows; r += gridDim.y * 8) {
                float gy = to_float_sm89(grad_y[r*cols+col]);
                float v  = to_float_sm89(x[r*cols+col]);
                float nx = (v - mean[r]) * rstd[r];
                db += gy; dg += gy * nx;
            }
        }
        s_dg[ty][tx] = dg; s_db[ty][tx] = db;
        __syncthreads();
        if (ty == 0 && col < cols) {
            float fg = 0, fb = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) { fg += s_dg[i][tx]; fb += s_db[i][tx]; }
            atomicAdd(&grad_gamma[col], fg);
            atomicAdd(&grad_beta[col], fb);
        }
        __syncthreads();
    }
}

// =================================================================================
// Backward B: Input grad — float4 vectorized, 512 threads, all dtypes
// =================================================================================
template<typename T>
__global__ __launch_bounds__(512)
void ln_backward_input_sm89_kernel(
    const T* __restrict__ grad_y, const T* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const T* __restrict__ gamma, T* __restrict__ grad_x, int cols)
{
    int row = blockIdx.x, tid = threadIdx.x;
    const T* dy_row = grad_y + row*cols;
    const T* x_row = x + row*cols;
    T* dx_row = grad_x + row*cols;
    float m = mean[row], rs = rstd[row];

    float sum1 = 0, sum2 = 0;
    if constexpr (std::is_same_v<T, float>) {
        const float4* dyv = reinterpret_cast<const float4*>(dy_row);
        const float4* xv  = reinterpret_cast<const float4*>(x_row);
        const float4* gv  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 d = dyv[i], xi = xv[i], g = gv ? gv[i] : make_float4(1,1,1,1);
            float nx=(xi.x-m)*rs, ny=(xi.y-m)*rs, nz=(xi.z-m)*rs, nw=(xi.w-m)*rs;
            float a=d.x*g.x, b=d.y*g.y, c=d.z*g.z, e=d.w*g.w;
            sum1 += a+b+c+e; sum2 += a*nx+b*ny+c*nz+e*nw;
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f, dy = dy_row[i], nx = (x_row[i]-m)*rs;
            sum1 += dy*g; sum2 += dy*g*nx;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float_sm89(gamma[i]) : 1.0f;
            float dy = to_float_sm89(dy_row[i]), nx = (to_float_sm89(x_row[i])-m)*rs;
            sum1 += dy*g; sum2 += dy*g*nx;
        }
    }

    sum1 = warpReduceSum_sm89(sum1); sum2 = warpReduceSum_sm89(sum2);
    __shared__ float s1, s2;
    if (tid == 0) { s1 = 0; s2 = 0; }
    __syncthreads();
    if (tid % warpSize == 0) { atomicAdd(&s1, sum1); atomicAdd(&s2, sum2); }
    __syncthreads();
    float t1 = s1, t2 = s2, ic = 1.0f/cols;

    if constexpr (std::is_same_v<T, float>) {
        float4* dxv = reinterpret_cast<float4*>(dx_row);
        const float4* dyv = reinterpret_cast<const float4*>(dy_row);
        const float4* xv  = reinterpret_cast<const float4*>(x_row);
        const float4* gv  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 d = dyv[i], xi = xv[i], g = gv ? gv[i] : make_float4(1,1,1,1);
            float nx=(xi.x-m)*rs, ny=(xi.y-m)*rs, nz=(xi.z-m)*rs, nw=(xi.w-m)*rs;
            dxv[i] = {rs*(d.x*g.x-(t1+nx*t2)*ic), rs*(d.y*g.y-(t1+ny*t2)*ic),
                      rs*(d.z*g.z-(t1+nz*t2)*ic), rs*(d.w*g.w-(t1+nw*t2)*ic)};
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f, nx = (x_row[i]-m)*rs;
            dx_row[i] = rs*(dy_row[i]*g-(t1+nx*t2)*ic);
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float_sm89(gamma[i]) : 1.0f;
            float dy = to_float_sm89(dy_row[i]), nx = (to_float_sm89(x_row[i])-m)*rs;
            dx_row[i] = from_float_sm89<T>(rs*(dy*g-(t1+nx*t2)*ic));
        }
    }
}

// =================================================================================
// Launchers
// =================================================================================
void layer_norm_forward_sm89_cuda(const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd, int rows, int cols, float eps) {
    layer_norm_forward_sm89_kernel<float, float><<<rows, 512>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}
void layer_norm_forward_sm89_cuda(const __half* x, const __half* gamma, const __half* beta,
    __half* y, float* mean, float* rstd, int rows, int cols, float eps) {
    layer_norm_forward_sm89_kernel<__half, float><<<rows, 512>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}
void layer_norm_forward_sm89_cuda(const __nv_bfloat16* x, const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    __nv_bfloat16* y, float* mean, float* rstd, int rows, int cols, float eps) {
    layer_norm_forward_sm89_kernel<__nv_bfloat16, float><<<rows, 512>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

void layer_norm_backward_sm89_cuda(
    const float* grad_y, const float* x, const float* mean, const float* rstd, const float* gamma,
    float* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols)
{
    if (grad_gamma != nullptr || grad_beta != nullptr) {
        cudaMemset(grad_gamma, 0, cols * sizeof(float));
        cudaMemset(grad_beta, 0, cols * sizeof(float));
        dim3 threads(32, 8);
        int bx = (cols + 31) / 32;
        int by = std::min(64, std::max(1, 512 / bx));
        ln_backward_gamma_beta_sm89_kernel<float><<<dim3(bx, by), threads>>>(
            grad_y, x, mean, rstd, grad_gamma, grad_beta, rows, cols);
    }
    if (grad_x != nullptr) {
        ln_backward_input_sm89_kernel<float><<<rows, 512>>>(
            grad_y, x, mean, rstd, gamma, grad_x, cols);
    }
}

} // namespace cuda
} // namespace OwnTensor
