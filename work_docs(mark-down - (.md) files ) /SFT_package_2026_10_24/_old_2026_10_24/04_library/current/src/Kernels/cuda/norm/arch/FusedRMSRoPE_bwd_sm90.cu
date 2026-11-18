#include <cuda_runtime.h>
#include "ops/helpers/FusedRMSRoPEKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/helpers/KernelPolicy.h"

namespace OwnTensor {
namespace cuda {

// Hopper path, defined in FusedRMSRoPE_bwd_sm90_kernel.cu. Atomic-free and
// bit-reproducible; owns its own dgamma-partial workspace. Every other arch
// falls back to the atomicAdd kernel below.
void fused_qk_norm_rope_backward_sm90(
    const float* dz_q, const float* dz_k,
    const float* q_in, const float* k_in,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    const float* q_rstd, const float* k_rstd,
    float*       dq_in, float* dk_in,
    float*       dq_gamma, float* dk_gamma,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions);   // v2: per-token positions [B, T] or nullptr

// HAS_POS (v2 packing, added 2026-10-10): false = original kernel (cache row t + pos_offset),
// true = cache row positions[b*T + t], matching the forward. Template flag so the false
// instantiation is unchanged; positions is appended last.
template <bool IS_NEOX, bool HAS_POS>
__global__ void fused_qknorm_rope_backward(
    const float* __restrict__ dz_q,      // [B, Hq,  T, hd]  upstream grad for Q
    const float* __restrict__ dz_k,      // [B, Hkv, T, hd]  upstream grad for K
    const float* __restrict__ q_in,      // [B, Hq,  T, hd]  saved pre-norm Q
    const float* __restrict__ k_in,      // [B, Hkv, T, hd]  saved pre-norm K
    const float* __restrict__ q_gamma,   // [hd]
    const float* __restrict__ k_gamma,   // [hd]
    const float* __restrict__ cos_sin,   // [cache_seq_len, hd]
    const float* __restrict__ q_rstd,    // [B, Hq,  T]  saved 1/rms
    const float* __restrict__ k_rstd,    // [B, Hkv, T]  saved 1/rms
    float*       __restrict__ dq_in,      // [B, Hq,  T, hd]  grad w.r.t. Q input
    float*       __restrict__ dk_in,      // [B, Hkv, T, hd]  grad w.r.t. K input
    float*       __restrict__ dq_gamma,   // [hd]  accumulated (zero before launch)
    float*       __restrict__ dk_gamma,   // [hd]  accumulated (zero before launch)
    int Hq, int Hkv, int T, int hd,
    int pos_offset,
    const int32_t* __restrict__ positions)   // [B, T], read only when HAS_POS
{
    const int t    = blockIdx.x;
    const int hh   = blockIdx.y;
    const int b    = blockIdx.z;
    const int lane = threadIdx.x;

    const bool is_q = (hh < Hq);
    const int  h    = is_q ? hh : (hh - Hq);
    const int  H    = is_q ? Hq : Hkv;

    const float* dz_base = is_q ? dz_q    : dz_k;
    const float* in_base = is_q ? q_in    : k_in;
    const float* gamma   = is_q ? q_gamma : k_gamma;
    const float* rstd_in = is_q ? q_rstd  : k_rstd;
    float*       dx_base = is_q ? dq_in    : dk_in;
    float*       dgamma  = is_q ? dq_gamma : dk_gamma;

    const long head_off = (((long)b * H + h) * T + t) * hd;
    const float* dz = dz_base + head_off;
    const float* x  = in_base + head_off;
    float*       dx = dx_base + head_off;
    const float  rstd = rstd_in[((long)b * H + h) * T + t];  // saved by the forward

    const int  half      = hd / 2;
    // v2: same cache row as the forward used for this token (was always t + pos_offset;
    // HAS_POS=false keeps exactly that).
    const long cache_row = HAS_POS ? (long)positions[(long)b * T + t] * hd
                                   : (long)(t + pos_offset) * hd;

    // ---- Pass A: dy = rope_inverse(dz); S = (1/hd) sum_j g[j]*xhat[j] ----
    float partial = 0.f;
    for (int i = lane; i < half; i += 32) {
        int ia, ib;
        if (IS_NEOX) { ia = i;     ib = i + half; }
        else         { ia = 2 * i; ib = 2 * i + 1; }

        float cs = cos_sin[cache_row + i];
        float sn = cos_sin[cache_row + i + half];

        float dza = dz[ia], dzb = dz[ib];
        float dya =  dza * cs + dzb * sn;      // rope backward: R^T (sin negated)
        float dyb = -dza * sn + dzb * cs;

        float xha = x[ia] * rstd, xhb = x[ib] * rstd;      // xhat
        float ga  = dya * gamma[ia], gb = dyb * gamma[ib]; // g = dy * gamma

        partial += ga * xha + gb * xhb;
    }
    for (int o = 16; o > 0; o >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, o);
    float S = __shfl_sync(0xffffffff, partial, 0) / (float)hd;

    // ---- Pass B: dx = rstd*(g - xhat*S); dgamma += dy*xhat ----
    for (int i = lane; i < half; i += 32) {
        int ia, ib;
        if (IS_NEOX) { ia = i;     ib = i + half; }
        else         { ia = 2 * i; ib = 2 * i + 1; }

        float cs = cos_sin[cache_row + i];
        float sn = cos_sin[cache_row + i + half];

        float dza = dz[ia], dzb = dz[ib];
        float dya =  dza * cs + dzb * sn;
        float dyb = -dza * sn + dzb * cs;

        float xha = x[ia] * rstd, xhb = x[ib] * rstd;
        float ga  = dya * gamma[ia], gb = dyb * gamma[ib];

        dx[ia] = rstd * (ga - xha * S);
        dx[ib] = rstd * (gb - xhb * S);

        atomicAdd(&dgamma[ia], dya * xha);
        atomicAdd(&dgamma[ib], dyb * xhb);
    }
}

// Host launcher: zeros dgamma, then launches one warp per head vector.
void fused_qk_norm_rope_backward(
    const float* dz_q, const float* dz_k,
    const float* q_in, const float* k_in,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    const float* q_rstd, const float* k_rstd,
    float*       dq_in, float* dk_in,
    float*       dq_gamma, float* dk_gamma,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions)
{
    // Hopper dispatch gate; the kernel below is the fallback for all other archs.
    // The Hopper path writes dq_gamma/dk_gamma outright, so it must be taken
    // before the memsets below and needs no pre-zeroing.
    // BLUTRAIN_NO_SM90_QKNORM_ROPE forces the generic path. The gate stays above
    // the memsets below on purpose: the generic path accumulates dgamma with
    // atomicAdd and needs the pre-zeroing, so the toggle must fall *into* the
    // memsets, never skip past them.
    if (!disable_sm90_qknorm_rope() && get_arch() == ArchFamily::Hopper) {
        fused_qk_norm_rope_backward_sm90(
            dz_q, dz_k, q_in, k_in, q_gamma, k_gamma, cos_sin,
            q_rstd, k_rstd, dq_in, dk_in, dq_gamma, dk_gamma,
            B, Hq, Hkv, T, hd, pos_offset, interleaved, stream, positions);
        return;
    }

    // dgamma is accumulated with atomicAdd, so it must start at zero.
    cudaMemsetAsync(dq_gamma, 0, sizeof(float) * hd, stream);
    cudaMemsetAsync(dk_gamma, 0, sizeof(float) * hd, stream);

    dim3 block(32);
    dim3 grid(T, Hq + Hkv, B);

    // v2: positions == nullptr launches the original <.., false> instantiation.
    if (!interleaved) {   // NeoX (bluscript default)
        if (positions == nullptr)
            fused_qknorm_rope_backward<true, false><<<grid, block, 0, stream>>>(
                dz_q, dz_k, q_in, k_in, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, dq_in, dk_in, dq_gamma, dk_gamma,
                Hq, Hkv, T, hd, pos_offset, positions);
        else
            fused_qknorm_rope_backward<true, true><<<grid, block, 0, stream>>>(
                dz_q, dz_k, q_in, k_in, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, dq_in, dk_in, dq_gamma, dk_gamma,
                Hq, Hkv, T, hd, pos_offset, positions);
    } else {              // GPT-J interleaved
        if (positions == nullptr)
            fused_qknorm_rope_backward<false, false><<<grid, block, 0, stream>>>(
                dz_q, dz_k, q_in, k_in, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, dq_in, dk_in, dq_gamma, dk_gamma,
                Hq, Hkv, T, hd, pos_offset, positions);
        else
            fused_qknorm_rope_backward<false, true><<<grid, block, 0, stream>>>(
                dz_q, dz_k, q_in, k_in, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, dq_in, dk_in, dq_gamma, dk_gamma,
                Hq, Hkv, T, hd, pos_offset, positions);
    }
}

} // namespace cuda
} // namespace OwnTensor

