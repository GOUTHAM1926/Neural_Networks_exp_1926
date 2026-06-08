#include "dtype/Types.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ops/helpers/LayerNormKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

// SM89 forward declarations
void layer_norm_forward_sm89_cuda(const float* x, const float* gamma, const float* beta, float* y, float* mean, float* rstd, int rows, int cols, float eps);
void layer_norm_forward_sm89_cuda(const __half* x, const __half* gamma, const __half* beta, __half* y, float* mean, float* rstd, int rows, int cols, float eps);
void layer_norm_forward_sm89_cuda(const __nv_bfloat16* x, const __nv_bfloat16* gamma, const __nv_bfloat16* beta, __nv_bfloat16* y, float* mean, float* rstd, int rows, int cols, float eps);
void layer_norm_backward_sm89_cuda(const float* grad_y, const float* x, const float* mean, const float* rstd, const float* gamma, float* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);

// =================================================================================
// Helpers
// =================================================================================
template<typename T> __device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float(float val) { return val; }
template<> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float(float val) { return val; }
template<> __device__ __forceinline__ __half from_float(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename AccT>
struct WelfordData { AccT n, mu, m2; };

template<typename AccT>
__device__ __inline__ WelfordData<AccT> welford_merge(WelfordData<AccT> a, WelfordData<AccT> b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordData<AccT> res;
    res.n = a.n + b.n;
    AccT delta = b.mu - a.mu;
    res.mu = a.mu + delta * (b.n / res.n);
    res.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / res.n);
    return res;
}

// =================================================================================
// Fused Forward — LayerNorm + RMSNorm via bool template (zero runtime overhead)
// =================================================================================
template<typename T, typename AccT, bool rms_norm>
__global__ void norm_forward_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ y, AccT* __restrict__ mean_out, AccT* __restrict__ rstd_out,
    int cols, AccT eps)
{
    const int row = blockIdx.x, tid = threadIdx.x;
    const T* row_x = x + row * cols;
    T* row_y = y + row * cols;

    AccT local_sum = 0, local_sq_sum = 0;
    int local_count = 0;

    if constexpr (std::is_same_v<T, float>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 v = x_vec[i];
            if constexpr (!rms_norm) local_sum += v.x + v.y + v.z + v.w;
            local_sq_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            local_count += 4;
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            AccT val = row_x[i];
            if constexpr (!rms_norm) local_sum += val;
            local_sq_sum += val*val; local_count++;
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __half22float2(h[k]);
                if constexpr (!rms_norm) local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x*(AccT)f.x + (AccT)f.y*(AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            if constexpr (!rms_norm) local_sum += val;
            local_sq_sum += val*val; local_count++;
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __bfloat1622float2(h[k]);
                if constexpr (!rms_norm) local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x*(AccT)f.x + (AccT)f.y*(AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            if constexpr (!rms_norm) local_sum += val;
            local_sq_sum += val*val; local_count++;
        }
    }

    // Welford state
    AccT n = (AccT)local_count;
    WelfordData<AccT> state;
    if constexpr (!rms_norm) {
        state = { n, (n>0) ? local_sum/n : (AccT)0, (n>0) ? (local_sq_sum - local_sum*local_sum/n) : (AccT)0 };
    } else {
        state = { n, (AccT)0, local_sq_sum };
    }

    // Warp reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        WelfordData<AccT> other;
        other.n = __shfl_down_sync(0xffffffff, state.n, offset);
        other.mu = __shfl_down_sync(0xffffffff, state.mu, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        if constexpr (!rms_norm) state = welford_merge(state, other);
        else { state.m2 += other.m2; state.n += other.n; }
    }

    // Block reduction
    __shared__ WelfordData<AccT> s_welford[32];
    const int warp_id = tid/32, lane_id = tid%32;
    if (lane_id == 0) s_welford[warp_id] = state;
    __syncthreads();
    if (tid == 0) {
        WelfordData<AccT> fs = s_welford[0];
        for (int i = 1; i < (int)(blockDim.x/32); ++i) {
            if constexpr (!rms_norm) fs = welford_merge(fs, s_welford[i]);
            else { fs.m2 += s_welford[i].m2; fs.n += s_welford[i].n; }
        }
        s_welford[0] = fs;
    }
    __syncthreads();

    AccT mu, rstd;
    if constexpr (!rms_norm) { mu = s_welford[0].mu; rstd = rsqrtf(s_welford[0].m2/cols + eps); }
    else { mu = (AccT)0; rstd = rsqrtf(s_welford[0].m2/cols + eps); }

    if (tid == 0) {
        if constexpr (!rms_norm) { if (mean_out) mean_out[row] = mu; }
        if (rstd_out) rstd_out[row] = rstd;
    }

    // Vectorized normalize
    if constexpr (std::is_same_v<T, float>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xv = x_vec[i];
            float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1,1,1,1);
            if constexpr (!rms_norm) {
                float4 bv = beta ? reinterpret_cast<const float4*>(beta)[i] : make_float4(0,0,0,0);
                y_vec[i] = {(xv.x-mu)*rstd*gv.x+bv.x, (xv.y-mu)*rstd*gv.y+bv.y, (xv.z-mu)*rstd*gv.z+bv.z, (xv.w-mu)*rstd*gv.w+bv.w};
            } else {
                y_vec[i] = {xv.x*rstd*gv.x, xv.y*rstd*gv.y, xv.z*rstd*gv.z, xv.w*rstd*gv.w};
            }
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            if constexpr (!rms_norm) { AccT b = beta ? (AccT)beta[i] : (AccT)0; row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b); }
            else { row_y[i] = (T)((AccT)row_x[i]*rstd*g); }
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols/8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xraw = x_vec[i]; const __half2* xh = reinterpret_cast<const __half2*>(&xraw);
            float4 graw; if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            const __half2* gh = gamma ? reinterpret_cast<const __half2*>(&graw) : nullptr;
            float4 braw; if constexpr (!rms_norm) { if (beta) braw = reinterpret_cast<const float4*>(beta)[i]; }
            const __half2* bh = (!rms_norm && beta) ? reinterpret_cast<const __half2*>(&braw) : nullptr;
            float4 yraw; __half2* yh = reinterpret_cast<__half2*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __half22float2(xh[k]);
                if constexpr (!rms_norm) { xf.x = (xf.x-mu)*rstd; xf.y = (xf.y-mu)*rstd; }
                else { xf.x *= rstd; xf.y *= rstd; }
                if (gh) { float2 gf = __half22float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if constexpr (!rms_norm) { if (bh) { float2 bf = __half22float2(bh[k]); xf.x += bf.x; xf.y += bf.y; } }
                yh[k] = __float22half2_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            if constexpr (!rms_norm) { AccT b = beta ? (AccT)beta[i] : (AccT)0; row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b); }
            else { row_y[i] = (T)((AccT)row_x[i]*rstd*g); }
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols/8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xraw = x_vec[i]; const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xraw);
            float4 graw; if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            const __nv_bfloat162* gh = gamma ? reinterpret_cast<const __nv_bfloat162*>(&graw) : nullptr;
            float4 braw; if constexpr (!rms_norm) { if (beta) braw = reinterpret_cast<const float4*>(beta)[i]; }
            const __nv_bfloat162* bh = (!rms_norm && beta) ? reinterpret_cast<const __nv_bfloat162*>(&braw) : nullptr;
            float4 yraw; __nv_bfloat162* yh = reinterpret_cast<__nv_bfloat162*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xh[k]);
                if constexpr (!rms_norm) { xf.x = (xf.x-mu)*rstd; xf.y = (xf.y-mu)*rstd; }
                else { xf.x *= rstd; xf.y *= rstd; }
                if (gh) { float2 gf = __bfloat1622float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if constexpr (!rms_norm) { if (bh) { float2 bf = __bfloat1622float2(bh[k]); xf.x += bf.x; xf.y += bf.y; } }
                yh[k] = __float22bfloat162_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            if constexpr (!rms_norm) { AccT b = beta ? (AccT)beta[i] : (AccT)0; row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b); }
            else { row_y[i] = (T)((AccT)row_x[i]*rstd*g); }
        }
    }
}

// =================================================================================
// LayerNorm Forward Launchers (with get_arch() dispatch)
// =================================================================================
void layer_norm_forward_cuda(const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd, int rows, int cols, float eps) {
    if (get_arch() == ArchFamily::Ada) { layer_norm_forward_sm89_cuda(x, gamma, beta, y, mean, rstd, rows, cols, eps); return; }
    norm_forward_kernel<float, float, false><<<rows, 256>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}
void layer_norm_forward_cuda(const float16_t* x, const float16_t* gamma, const float16_t* beta,
    float16_t* y, float* mean, float* rstd, int rows, int cols, float eps) {
    if (get_arch() == ArchFamily::Ada) {
        layer_norm_forward_sm89_cuda(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(gamma),
            reinterpret_cast<const __half*>(beta), reinterpret_cast<__half*>(y), mean, rstd, rows, cols, eps); return;
    }
    norm_forward_kernel<__half, float, false><<<rows, 256>>>(reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(gamma), reinterpret_cast<const __half*>(beta),
        reinterpret_cast<__half*>(y), mean, rstd, cols, eps);
}
void layer_norm_forward_cuda(const bfloat16_t* x, const bfloat16_t* gamma, const bfloat16_t* beta,
    bfloat16_t* y, float* mean, float* rstd, int rows, int cols, float eps) {
    if (get_arch() == ArchFamily::Ada) {
        layer_norm_forward_sm89_cuda(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(gamma),
            reinterpret_cast<const __nv_bfloat16*>(beta), reinterpret_cast<__nv_bfloat16*>(y), mean, rstd, rows, cols, eps); return;
    }
    norm_forward_kernel<__nv_bfloat16, float, false><<<rows, 256>>>(reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gamma), reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_bfloat16*>(y), mean, rstd, cols, eps);
}

// =================================================================================
// RMSNorm Forward Launchers — same kernel with rms_norm=true
// =================================================================================
void rms_norm_forward_cuda(const float* x, const float* gamma, float* y, float* rstd, int rows, int cols, float eps) {
    norm_forward_kernel<float, float, true><<<rows, 256>>>(x, gamma, nullptr, y, nullptr, rstd, cols, eps);
}
void rms_norm_forward_cuda(const float16_t* x, const float16_t* gamma, float16_t* y, float* rstd, int rows, int cols, float eps) {
    norm_forward_kernel<__half, float, true><<<rows, 256>>>(reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(gamma), nullptr, reinterpret_cast<__half*>(y), nullptr, rstd, cols, eps);
}
void rms_norm_forward_cuda(const bfloat16_t* x, const bfloat16_t* gamma, bfloat16_t* y, float* rstd, int rows, int cols, float eps) {
    norm_forward_kernel<__nv_bfloat16, float, true><<<rows, 256>>>(reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gamma), nullptr, reinterpret_cast<__nv_bfloat16*>(y), nullptr, rstd, cols, eps);
}

// =================================================================================
// Backward: Gamma/Beta kernel (column-wise reduction)
// =================================================================================
template<typename T>
__global__ void ln_backward_gamma_beta_kernel(
    const T* __restrict__ grad_y, const T* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    float* __restrict__ grad_gamma, float* __restrict__ grad_beta,
    int rows, int cols) {
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float s_dg[8][32], s_db[8][32];
    #pragma unroll 4
    for (int cb = blockIdx.x*32; cb < cols; cb += gridDim.x*32) {
        int col = cb + tx;
        float dg = 0, db = 0;
        if (col < cols) {
            for (int r = blockIdx.y*8+ty; r < rows; r += gridDim.y*8) {
                float gy = to_float(grad_y[r*cols+col]);
                float v = to_float(x[r*cols+col]);
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
// Backward: Input kernel (per-row, float4 vectorized for fp32)
// =================================================================================
template<typename T>
__global__ void ln_backward_input_kernel(
    const T* __restrict__ grad_y, const T* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const T* __restrict__ gamma, T* __restrict__ grad_x, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    const T* dy = grad_y + row*cols;
    const T* xr = x + row*cols;
    T* dx = grad_x + row*cols;
    float m = mean[row], rs = rstd[row];
    float sum1 = 0, sum2 = 0;

    if constexpr (std::is_same_v<T, float>) {
        const float4* dyv = reinterpret_cast<const float4*>(dy);
        const float4* xv = reinterpret_cast<const float4*>(xr);
        const float4* gv = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 d = dyv[i], xi = xv[i], g = gv ? gv[i] : make_float4(1,1,1,1);
            float nx=(xi.x-m)*rs, ny=(xi.y-m)*rs, nz=(xi.z-m)*rs, nw=(xi.w-m)*rs;
            float a=d.x*g.x, b=d.y*g.y, c=d.z*g.z, e=d.w*g.w;
            sum1 += a+b+c+e; sum2 += a*nx+b*ny+c*nz+e*nw;
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f, dv = dy[i], nx = (xr[i]-m)*rs;
            sum1 += dv*g; sum2 += dv*g*nx;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float(gamma[i]) : 1.0f;
            float dv = to_float(dy[i]), nx = (to_float(xr[i])-m)*rs;
            sum1 += dv*g; sum2 += dv*g*nx;
        }
    }

    sum1 = warpReduceSum(sum1); sum2 = warpReduceSum(sum2);
    __shared__ float s1, s2;
    if (tid == 0) { s1 = 0; s2 = 0; } __syncthreads();
    if (tid % warpSize == 0) { atomicAdd(&s1, sum1); atomicAdd(&s2, sum2); } __syncthreads();
    float t1 = s1, t2 = s2, ic = 1.0f/cols;

    if constexpr (std::is_same_v<T, float>) {
        float4* dxv = reinterpret_cast<float4*>(dx);
        const float4* dyv = reinterpret_cast<const float4*>(dy);
        const float4* xv = reinterpret_cast<const float4*>(xr);
        const float4* gv = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 d = dyv[i], xi = xv[i], g = gv ? gv[i] : make_float4(1,1,1,1);
            float nx=(xi.x-m)*rs, ny=(xi.y-m)*rs, nz=(xi.z-m)*rs, nw=(xi.w-m)*rs;
            dxv[i] = {rs*(d.x*g.x-(t1+nx*t2)*ic), rs*(d.y*g.y-(t1+ny*t2)*ic),
                      rs*(d.z*g.z-(t1+nz*t2)*ic), rs*(d.w*g.w-(t1+nw*t2)*ic)};
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f, nx = (xr[i]-m)*rs;
            dx[i] = rs*(dy[i]*g-(t1+nx*t2)*ic);
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float(gamma[i]) : 1.0f;
            float dv = to_float(dy[i]), nx = (to_float(xr[i])-m)*rs;
            dx[i] = from_float<T>(rs*(dv*g-(t1+nx*t2)*ic));
        }
    }
}

// =================================================================================
// Backward Launch Helper
// =================================================================================
template<typename T>
static void launch_layer_norm_backward(
    const T* grad_y, const T* x, const float* mean, const float* rstd, const T* gamma,
    T* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols) {
    if (grad_gamma != nullptr || grad_beta != nullptr) {
        cudaMemset(grad_gamma, 0, cols * sizeof(float));
        cudaMemset(grad_beta, 0, cols * sizeof(float));
        dim3 threads(32, 8);
        int bx = (cols+31)/32, by = std::max(1, std::min(32, 128/bx));
        ln_backward_gamma_beta_kernel<T><<<dim3(bx,by), threads>>>(grad_y, x, mean, rstd, grad_gamma, grad_beta, rows, cols);
    }
    if (grad_x != nullptr) {
        int threads = (cols > 256) ? 512 : 256;
        ln_backward_input_kernel<T><<<rows, threads>>>(grad_y, x, mean, rstd, gamma, grad_x, cols);
    }
}

// =================================================================================
// LayerNorm Backward Launchers (with get_arch() dispatch)
// =================================================================================
void layer_norm_backward_cuda(const float* gy, const float* x, const float* mean, const float* rstd, const float* gamma,
    float* gx, float* gg, float* gb, int rows, int cols) {
    if (get_arch() == ArchFamily::Ada) { layer_norm_backward_sm89_cuda(gy, x, mean, rstd, gamma, gx, gg, gb, rows, cols); return; }
    launch_layer_norm_backward<float>(gy, x, mean, rstd, gamma, gx, gg, gb, rows, cols);
}
void layer_norm_backward_cuda(const float16_t* gy, const float16_t* x, const float* mean, const float* rstd, const float16_t* gamma,
    float16_t* gx, float* gg, float* gb, int rows, int cols) {
    launch_layer_norm_backward<__half>(reinterpret_cast<const __half*>(gy), reinterpret_cast<const __half*>(x),
        mean, rstd, reinterpret_cast<const __half*>(gamma), reinterpret_cast<__half*>(gx), gg, gb, rows, cols);
}
void layer_norm_backward_cuda(const bfloat16_t* gy, const bfloat16_t* x, const float* mean, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* gx, float* gg, float* gb, int rows, int cols) {
    launch_layer_norm_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(gy), reinterpret_cast<const __nv_bfloat16*>(x),
        mean, rstd, reinterpret_cast<const __nv_bfloat16*>(gamma), reinterpret_cast<__nv_bfloat16*>(gx), gg, gb, rows, cols);
}

// =================================================================================
// RMSNorm Backward Kernels
// =================================================================================
template<typename T>
__global__ void rms_backward_gamma_kernel(const T* __restrict__ gy, const T* __restrict__ x,
    const float* __restrict__ rstd, float* __restrict__ grad_gamma, int rows, int cols) {
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float s_dg[8][32];
    #pragma unroll 4
    for (int cb = blockIdx.x*32; cb < cols; cb += gridDim.x*32) {
        int col = cb + tx; float acc = 0;
        if (col < cols) {
            for (int r = blockIdx.y*8+ty; r < rows; r += gridDim.y*8)
                acc += to_float(gy[r*cols+col]) * to_float(x[r*cols+col]) * rstd[r];
        }
        s_dg[ty][tx] = acc; __syncthreads();
        if (ty == 0 && col < cols) {
            float fg = 0; for (int i = 0; i < 8; i++) fg += s_dg[i][tx];
            atomicAdd(&grad_gamma[col], fg);
        } __syncthreads();
    }
}

template<typename T>
__global__ void rms_backward_input_kernel(const T* __restrict__ gy, const T* __restrict__ x,
    const float* __restrict__ rstd, const T* __restrict__ gamma, T* __restrict__ gx, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    const T* dy = gy+row*cols; const T* xr = x+row*cols; T* dx = gx+row*cols;
    float rs = rstd[row], dot = 0;
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = gamma ? to_float(gamma[i]) : 1.0f;
        dot += to_float(dy[i]) * g * to_float(xr[i]);
    }
    dot = warpReduceSum(dot);
    __shared__ float s_dot; if (tid == 0) s_dot = 0; __syncthreads();
    if (tid % warpSize == 0) atomicAdd(&s_dot, dot); __syncthreads();
    float td = s_dot, ic = 1.0f/cols, rs3 = rs*rs*rs;
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = gamma ? to_float(gamma[i]) : 1.0f;
        dx[i] = from_float<T>(rs*to_float(dy[i])*g - rs3*to_float(xr[i])*td*ic);
    }
}

template<typename T>
static void launch_rms_norm_backward(const T* gy, const T* x, const float* rstd, const T* gamma,
    T* gx, float* gg, int rows, int cols) {
    if (gg) {
        cudaMemset(gg, 0, cols*sizeof(float));
        dim3 threads(32,8); int bx = (cols+31)/32, by = std::max(1, std::min(64, 512/bx));
        rms_backward_gamma_kernel<T><<<dim3(bx,by), threads>>>(gy, x, rstd, gg, rows, cols);
    }
    if (gx) {
        int threads = (cols > 256) ? 512 : 256;
        rms_backward_input_kernel<T><<<rows, threads>>>(gy, x, rstd, gamma, gx, cols);
    }
}

// RMSNorm Backward Launchers
void rms_norm_backward_cuda(const float* gy, const float* x, const float* rstd, const float* gamma,
    float* gx, float* gg, int rows, int cols) { launch_rms_norm_backward<float>(gy, x, rstd, gamma, gx, gg, rows, cols); }
void rms_norm_backward_cuda(const float16_t* gy, const float16_t* x, const float* rstd, const float16_t* gamma,
    float16_t* gx, float* gg, int rows, int cols) {
    launch_rms_norm_backward<__half>(reinterpret_cast<const __half*>(gy), reinterpret_cast<const __half*>(x),
        rstd, reinterpret_cast<const __half*>(gamma), reinterpret_cast<__half*>(gx), gg, rows, cols);
}
void rms_norm_backward_cuda(const bfloat16_t* gy, const bfloat16_t* x, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* gx, float* gg, int rows, int cols) {
    launch_rms_norm_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(gy), reinterpret_cast<const __nv_bfloat16*>(x),
        rstd, reinterpret_cast<const __nv_bfloat16*>(gamma), reinterpret_cast<__nv_bfloat16*>(gx), gg, rows, cols);
}

} // namespace cuda
} // namespace OwnTensor
