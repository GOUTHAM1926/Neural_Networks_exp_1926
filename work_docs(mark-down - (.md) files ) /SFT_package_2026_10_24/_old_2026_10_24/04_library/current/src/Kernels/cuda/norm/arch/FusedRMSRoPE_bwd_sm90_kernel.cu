#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "ops/helpers/FusedRMSRoPEKernels.h"

// SM90 (Hopper) fused QK-norm + RoPE backward. Gated at runtime behind
// ArchFamily::Hopper by the dispatcher in FusedRMSRoPE_bwd_sm90.cu.
//
// dgamma carries no atomics: grid.x is BWD_T_CHUNKS fixed chunks, each block
// accumulates dgamma in registers over the tokens it strides across and writes
// ONE private row of a [B*H*BWD_T_CHUNKS, hd] partial buffer, which a second
// kernel reduces in fixed index order. Keep that structure: it is what makes
// dgamma bit-reproducible (rows keyed by launch shape, not arrival order).

namespace OwnTensor {
namespace cuda {

#ifndef BWD_T_CHUNKS
#define BWD_T_CHUNKS 32
#endif

// HAS_POS (v2 packing, added 2026-10-10): false = original kernel (cache row t + pos_offset),
// true = cache row positions[b*T + t] (several chats per row, each restarting at 0). It must
// match the forward exactly: the backward undoes the SAME rotation. Template flag so the false
// instantiation is the pre-change kernel; positions is appended last. dgamma partials keep their
// fixed (t-chunk, head, batch) order, so they stay bit-reproducible with positions too.
template <bool IS_NEOX, int HD, bool HAS_POS>
__global__ void fused_qk_norm_rope_bwd_kernel_sm90(
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
    float*       __restrict__ dq_gpart,   // [B*Hq*BWD_T_CHUNKS,  hd]  Q dgamma partials
    float*       __restrict__ dk_gpart,   // [B*Hkv*BWD_T_CHUNKS, hd]  K dgamma partials
    int Hq, int Hkv, int T,
    int pos_offset,
    const int32_t* __restrict__ positions)   // [B, T], read only when HAS_POS
{
    // HD must stay a template parameter: it makes HALF, PAIRS and every ia/ib
    // compile-time constants, which is what keeps the per-lane arrays below in
    // registers. With a runtime hd, ptxas spills them to local memory.
    constexpr int HALF  = HD / 2;
    constexpr int PAIRS = (HALF + 31) / 32;   // hd64 -> 1, hd128 -> 2

    const int lane = threadIdx.x & 31;

    const int t0   = blockIdx.x;          // token chunk id, strides by BWD_T_CHUNKS
    const int hh   = blockIdx.y;
    const int b    = blockIdx.z;

    const bool is_q = (hh < Hq);
    const int  h    = is_q ? hh : (hh - Hq);
    const int  H    = is_q ? Hq : Hkv;

    const float* dz_base = is_q ? dz_q    : dz_k;
    const float* in_base = is_q ? q_in    : k_in;
    const float* gamma   = is_q ? q_gamma : k_gamma;
    const float* rstd_in = is_q ? q_rstd  : k_rstd;
    float*       dx_base = is_q ? dq_in    : dk_in;

    // Pair indices owned by this lane: i = lane, lane+32, ...
    int ia_r[PAIRS], ib_r[PAIRS];
    bool act[PAIRS];                      // whether this lane owns the pair slot
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        const int i = lane + p * 32;
        act[p] = (HALF % 32 == 0) || (i < HALF);
        const int ii = act[p] ? i : 0;    // clamp so the address is always valid
        ia_r[p] = IS_NEOX ? ii          : (2 * ii);
        ib_r[p] = IS_NEOX ? (ii + HALF) : (2 * ii + 1);
    }

    // dgamma accumulator, private to this thread: summed over every token this
    // block visits, then written once. No atomics anywhere.
    float dg_a[PAIRS] = {}, dg_b[PAIRS] = {};

    // gamma is the one load worth hoisting: unlike dz/x it is identical across
    // every token this block visits.
    float ga_r[PAIRS], gb_r[PAIRS];
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        ga_r[p] = gamma[ia_r[p]];
        gb_r[p] = gamma[ib_r[p]];
    }

    for (int t = t0; t < T; t += BWD_T_CHUNKS) {
        const long head_off = (((long)b * H + h) * T + t) * HD;
        const float* dz = dz_base + head_off;
        const float* x  = in_base + head_off;
        float*       dx = dx_base + head_off;
        const float  rstd = rstd_in[((long)b * H + h) * T + t];
        // v2: same cache row as the forward used for this token (was always t + pos_offset;
        // HAS_POS=false keeps exactly that).
        const float* cs_row = HAS_POS
            ? cos_sin + (long)positions[(long)b * T + t] * HD
            : cos_sin + (long)(t + pos_offset) * HD;

        // Pass A must finish S before any dx is written, so the two passes
        // cannot merge. Instead xhat, dy and g are held in registers across
        // both so pass B re-reads no global memory. Do not drop g and recompute
        // it from dy: pass B uses both.
        float xh_a[PAIRS], xh_b[PAIRS];
        float dy_a[PAIRS], dy_b[PAIRS];
        float g_a[PAIRS],  g_b[PAIRS];

        float partial = 0.f;
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const int i  = lane + p * 32;
            const int ia = ia_r[p], ib = ib_r[p];

            const float cs = cs_row[act[p] ? i        : 0];
            const float sn = cs_row[act[p] ? i + HALF : HALF];

            const float dza = dz[ia], dzb = dz[ib];
            dy_a[p] =  dza * cs + dzb * sn;          // rope backward: R^T (sin negated)
            dy_b[p] = -dza * sn + dzb * cs;

            xh_a[p] = x[ia] * rstd;                  // xhat
            xh_b[p] = x[ib] * rstd;
            g_a[p]  = dy_a[p] * ga_r[p];             // g = dy * gamma
            g_b[p]  = dy_b[p] * gb_r[p];

            if (act[p]) partial += g_a[p] * xh_a[p] + g_b[p] * xh_b[p];
        }
        // XOR-shuffle leaves the full sum in EVERY lane -> no broadcast needed.
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, o);
        const float S = partial / (float)HD;

        // Pass B: dx = rstd*(g - xhat*S); dgamma += dy*xhat. Register math only.
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            if (!act[p]) continue;
            dx[ia_r[p]] = rstd * (g_a[p] - xh_a[p] * S);
            dx[ib_r[p]] = rstd * (g_b[p] - xh_b[p] * S);

            dg_a[p] += dy_a[p] * xh_a[p];
            dg_b[p] += dy_b[p] * xh_b[p];
        }
    }

    // Each (t_chunk, head, batch) block owns exactly one row, so no two blocks
    // touch the same address and no row is left unwritten. Q and K must index
    // their own buffers with their OWN head count (H = Hq or Hkv).
    const int row = ((b * H) + h) * BWD_T_CHUNKS + t0;
    float* dst = (is_q ? dq_gpart : dk_gpart) + (long)row * HD;
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        if (!act[p]) continue;
        dst[ia_r[p]] = dg_a[p];
        dst[ib_r[p]] = dg_b[p];
    }
}

// Reduction of the dgamma partials. Rows must be summed in strictly increasing
// order; that fixed order is what makes the result bit-identical across runs.
__global__ void reduce_dgamma_kernel(const float* __restrict__ part,
                                     float* __restrict__ dgamma,
                                     int n_rows, int hd)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hd) return;
    float acc = 0.f;
    for (int r = 0; r < n_rows; ++r)
        acc += part[(long)r * hd + d];
    dgamma[d] = acc;
}

// Partial-buffer size in floats. Q and K differ (Hq vs Hkv), so it must be
// called once per side.
static int fused_qk_norm_rope_bwd_workspace_floats(int B, int H, int hd) {
    return B * H * BWD_T_CHUNKS * hd;
}

// File-local fp32 workspace for the dgamma partials. The public signature
// (FusedRMSRoPEKernels.h) exposes only dq_gamma/dk_gamma, so the
// [B*H*BWD_T_CHUNKS, hd] partial buffers are owned here: grown on demand, kept
// for the process lifetime, same pattern as d_gamma_workspace in
// LayerNormKernels.cu. Never memset them: every row is fully written by its
// owning block before the reduction reads it.
static float* d_sm90_qgamma_part = nullptr;
static size_t sm90_qgamma_part_size = 0;   // in floats
static float* d_sm90_kgamma_part = nullptr;
static size_t sm90_kgamma_part_size = 0;   // in floats

static void ensure_sm90_gamma_part(float** buf, size_t* cap, size_t need) {
    if (need > *cap) {
        if (*buf) cudaFree(*buf);
        cudaMalloc(buf, need * sizeof(float));
        *cap = need;
    }
}

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
    const int32_t* positions)
{
    // Q and K size their partial buffers with their OWN head count (Hq vs Hkv).
    ensure_sm90_gamma_part(&d_sm90_qgamma_part, &sm90_qgamma_part_size,
                           (size_t)fused_qk_norm_rope_bwd_workspace_floats(B, Hq,  hd));
    ensure_sm90_gamma_part(&d_sm90_kgamma_part, &sm90_kgamma_part_size,
                           (size_t)fused_qk_norm_rope_bwd_workspace_floats(B, Hkv, hd));
    float* dq_gamma_part = d_sm90_qgamma_part;
    float* dk_gamma_part = d_sm90_kgamma_part;

    // No memset needed: every partial row is fully written by its owning block.
    dim3 block(32);
    dim3 grid(BWD_T_CHUNKS, Hq + Hkv, B);

    // hd is specialized to the shipped shapes (64, 128), matching the forward.
    // Keep abort() rather than assert(): assert compiles out under -DNDEBUG and
    // an unsupported hd would then silently leave dq/dk untouched.
    // v2: HASP selects the per-token-positions instantiation (positions != nullptr).
#define LAUNCH_BWD(NEOX, HDV, HASP)                                            \
    fused_qk_norm_rope_bwd_kernel_sm90<NEOX, HDV, HASP><<<grid, block, 0, stream>>>( \
        dz_q, dz_k, q_in, k_in, q_gamma, k_gamma, cos_sin,                     \
        q_rstd, k_rstd, dq_in, dk_in, dq_gamma_part, dk_gamma_part,            \
        Hq, Hkv, T, pos_offset, positions)

#define DISPATCH_BWD_HD(NEOX, HASP)                                            \
    switch (hd) {                                                              \
        case  64: LAUNCH_BWD(NEOX,  64, HASP);  break;                         \
        case 128: LAUNCH_BWD(NEOX, 128, HASP);  break;                         \
        default :                                                              \
            fprintf(stderr, "fused_qk_norm_rope_backward: unsupported hd=%d "  \
                            "(expected 64 or 128)\n", hd);                     \
            abort();                                                           \
    }

    if (positions == nullptr) {   // original path: cache row = t + pos_offset
        if (!interleaved) {   // NeoX (bluscript default)
            DISPATCH_BWD_HD(true, false);
        } else {              // GPT-J interleaved
            DISPATCH_BWD_HD(false, false);
        }
    } else {                      // v2: per-token positions[b*T + t]
        if (!interleaved) {
            DISPATCH_BWD_HD(true, true);
        } else {
            DISPATCH_BWD_HD(false, true);
        }
    }

#undef DISPATCH_BWD_HD
#undef LAUNCH_BWD

    // Each side reduces only its own rows: Q has B*Hq*BWD_T_CHUNKS, K has
    // B*Hkv*BWD_T_CHUNKS. Passing the wrong count pulls in uninitialized rows.
    const int thr = 128;
    const int blk = (hd + thr - 1) / thr;
    reduce_dgamma_kernel<<<blk, thr, 0, stream>>>(
        dq_gamma_part, dq_gamma, B * Hq * BWD_T_CHUNKS, hd);
    reduce_dgamma_kernel<<<blk, thr, 0, stream>>>(
        dk_gamma_part, dk_gamma, B * Hkv * BWD_T_CHUNKS, hd);
}

} // namespace cuda
} // namespace OwnTensor
