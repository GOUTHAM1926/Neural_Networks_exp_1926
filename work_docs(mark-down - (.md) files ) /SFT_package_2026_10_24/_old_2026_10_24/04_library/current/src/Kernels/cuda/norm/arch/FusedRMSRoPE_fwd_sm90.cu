#include <cassert>
#include <cuda_runtime.h>
#include "ops/helpers/FusedRMSRoPEKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/helpers/KernelPolicy.h"

namespace OwnTensor {
namespace cuda {

// Hopper path, defined in FusedRMSRoPE_fwd_sm90_kernel.cu. Every other arch
// falls back to the portable kernel below.
void fused_qk_norm_rope_forward_sm90(
    const float* q_in, const float* k_in,
    float*       q_out, float* k_out,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    float*       q_rstd, float* k_rstd,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, float eps, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions);   // v2: per-token positions [B, T] or nullptr

// HAS_POS (v2 packing, added 2026-10-10): false = original kernel (position = t + pos_offset),
// true = position = positions[b*T + t]. Template flag so the false instantiation is unchanged;
// positions is appended last. See the note in FusedRMSRoPEKernels.h.
template <bool IS_NEOX, bool HAS_POS>
__global__ void fused_qknorm_rope_forward(
    const float* __restrict__ q_in,          // [B, Hq,  T, hd]  pre-norm Q
    const float* __restrict__ k_in,          // [B, Hkv, T, hd]  pre-norm K
    float*       __restrict__ q_out,         // [B, Hq,  T, hd]  normed + roped Q
    float*       __restrict__ k_out,         // [B, Hkv, T, hd]  normed + roped K
    const float* __restrict__ q_gamma,       // [hd]
    const float* __restrict__ k_gamma,       // [hd]
    const float* __restrict__ cos_sin_cache, // [cache_seq_len, hd]
    float*       __restrict__ q_rstd,        // [B, Hq,  T]  1/rms to save for backward
    float*       __restrict__ k_rstd,        // [B, Hkv, T]  1/rms to save for backward
    int Hq, int Hkv, int T, int Hd, int pos_offset, float eps,
    const int32_t* __restrict__ positions) {  // [B, T], read only when HAS_POS
    
    const int tok_pos = blockIdx.x;
    const int head_ind = blockIdx.y;
    const int batch = blockIdx.z;
    const int lane = threadIdx.x;

    const bool is_query = (head_ind < Hq);
    const int h = is_query ? head_ind : (head_ind - Hq);
    const int H = is_query ? Hq : Hkv;
    
    const float* in_base  = is_query ? q_in    : k_in;
    float*       out_base = is_query ? q_out   : k_out;
    const float* gamma    = is_query ? q_gamma : k_gamma;
    float*       rstd_out = is_query ? q_rstd  : k_rstd;
    

    const long head_offset = (((long)batch * H + h) * T + tok_pos) * Hd;
    const float* x = in_base + head_offset;
    float* y = out_base + head_offset;
    const long rstd_offset = ((long)batch * H + h) * T + tok_pos;

    // Sum of squares over head dim, accumulated in fp32.
    float part = 0.0f;
    for (int j = lane; j < Hd; j+=32) {
        float v = x[j];
        part += v * v;
    }
    for (int  o = 16; o > 0; o >>= 1) {
        part += __shfl_down_sync(0xffffffff, part, o);
    }

    float rstd = rsqrtf(part / float(Hd) + eps);
    rstd = __shfl_sync(0xffffffff, rstd, 0);
    if (lane == 0) rstd_out[rstd_offset] = rstd;
    
    // Norm + RoPE. One lane owns whole pairs, so both halves stay in one thread.
    const int half = Hd/2;
    // v2: with HAS_POS the cache row is the token's position inside its chat
    // (was always tok_pos + pos_offset; HAS_POS=false keeps exactly that).
    const long cache_row = HAS_POS ? (long)positions[(long)batch * T + tok_pos] * Hd
                                   : long(tok_pos + pos_offset) * Hd;

    for (int i = lane; i < half; i += 32) {
        int ia, ib;
        if (IS_NEOX) { ia = i; ib = i + half; }
        else { ia =  2 * i; ib = 2 * i + 1; }

        float ya = x[ia] * rstd * gamma[ia];
        float yb = x[ib] * rstd * gamma[ib];

        float cs = cos_sin_cache[cache_row + i];
        float sn = cos_sin_cache[cache_row + i + half];

        y[ia] = ya * cs - yb * sn;
        y[ib] = ya * sn + yb * cs;
    }
}

void fused_qk_norm_rope_forward(
    const float* q_in, const float* k_in,
    float*       q_out, float* k_out,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    float*       q_rstd, float* k_rstd,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, float eps, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions)
{
    // Hopper dispatch gate; the kernel below is the fallback for all other archs.
    if (!disable_sm90_qknorm_rope() && get_arch() == ArchFamily::Hopper) {
        fused_qk_norm_rope_forward_sm90(
            q_in, k_in, q_out, k_out, q_gamma, k_gamma, cos_sin,
            q_rstd, k_rstd, B, Hq, Hkv, T, hd, pos_offset, eps, interleaved, stream,
            positions);
        return;
    }

    dim3 block(32);
    dim3 grid(T, Hq + Hkv, B);

    // v2: positions == nullptr launches the original <.., false> instantiation.
    if (!interleaved) {   // NeoX (bluscript default)
        if (positions == nullptr)
            fused_qknorm_rope_forward<true, false><<<grid, block, 0, stream>>>(
                q_in, k_in, q_out, k_out, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, Hq, Hkv, T, hd, pos_offset, eps, positions);
        else
            fused_qknorm_rope_forward<true, true><<<grid, block, 0, stream>>>(
                q_in, k_in, q_out, k_out, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, Hq, Hkv, T, hd, pos_offset, eps, positions);
    } else {              // GPT-J interleaved
        if (positions == nullptr)
            fused_qknorm_rope_forward<false, false><<<grid, block, 0, stream>>>(
                q_in, k_in, q_out, k_out, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, Hq, Hkv, T, hd, pos_offset, eps, positions);
        else
            fused_qknorm_rope_forward<false, true><<<grid, block, 0, stream>>>(
                q_in, k_in, q_out, k_out, q_gamma, k_gamma, cos_sin,
                q_rstd, k_rstd, Hq, Hkv, T, hd, pos_offset, eps, positions);
    }
}

} // namespace cuda
} // namespace OwnTensor


