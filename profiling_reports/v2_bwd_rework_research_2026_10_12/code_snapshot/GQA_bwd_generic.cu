// Generic (non-Blackwell) GQA BACKWARD path + the library-facing dispatch
// wrappers. NOT Blackwell-marked, so the Makefile compiles it for the MACHINE
// arch (sm_86 / sm_90a / sm_103a). The portable WMMA backward kernels live here
// so they have an image for whatever GPU actually runs; the tcgen05 backward
// stays in GQA_bwd_sm103.cu (compiled compute_103a) and is reached via the
// extern entry gqa_backward_v17_kv_launch_d64(), taken only on a B300.
//
// NOTE: the WMMA backward (launch_gqa_backward) is NON-CAUSAL. On non-Blackwell
// GPUs the causal path falls back to it, which is numerically correct ONLY for
// full (bidirectional) attention. Causal-correct training on non-B300 needs the
// memory-efficient backend instead.

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

#include "ops/helpers/AttentionKernels.h"
#include "dtype/Types.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/helpers/KernelPolicy.h"
using namespace nvcuda;

namespace OwnTensor {
namespace cuda {

// Defined in GQA_bwd_sm103.cu (compiled for compute_103a). Only *called* when
// the running GPU is a B300, so a non-Blackwell build that lacks its device
// image never launches it.
void gqa_backward_v17_kv_launch_d64(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Nq_heads, int Nkv_heads, int G, int T, float scale);

// Defined in GQA_bwd_sm90.cu (kernel v44, wgmma/TMA + TMA-reduce dQ, compiled
// for the machine arch). Causal + head_dim 128 only; taken ONLY when
// get_arch()==Hopper. This is the CAUSAL-CORRECT backward on H200 -- the WMMA
// fallback below is non-causal.
void gqa_v44_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale);

// Defined in GQA_bwd_vj1d_sm90.cu (kernel Vj1d: per-hq one-writer bf16 partial
// dK/dV + a fixed-order, atomic-free greduce over the G query heads). dK/dV are
// BIT-IDENTICAL run-to-run, matching cuDNN's one-writer reproducibility; dQ is
// still K-tile reduced and non-deterministic (~3e-5), exactly as cuDNN's is.
// This is the DEFAULT Hopper backward. seg_start (v2 design B, added 2026-10-10):
// packed rows, or nullptr for plain causal; only this kernel accepts it.
void gqa_vj1d_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* seg_start, const int32_t* chat_tiles, int n_chat_tiles);

// Defined in GQA_bwd_vj1_sm90.cu (kernel Vj1: same design as Vj1d but the
// greduce is replaced by an atomic TMA-reduce). Faster, and dK/dV then wobble
// ~1 bf16 ULP run-to-run. Opt in with USE_FLASH=1.
void gqa_vj1_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale);

// Defined in GQA_bwd_vj2_sm90.cu (kernel Vj2, added 2026-10-12: FA3-structured -- 128 keys per block, a
// loader warpgroup with a dQ-writer warp, 2 compute warpgroups, P/dS in registers for the dV/dK GEMMs, dS
// from fp32 P). Same outputs and determinism as Vj1d (one-writer dK/dV partials + fixed-order head sum).
// Opt in with USE_VJ2=1 while it is validated; dense causal only (packed rows stay on Vj1d).
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale);

// v44 / Vj1 / Vj1d all tile at Br=Bc=64 over the KV axis, so S must be a
// multiple of 64. Vj1d's greduce grid is S/8, which 64 already covers.
static inline bool gqa_v44_shape_ok(int T, int hd) {
    return hd == 128 && (T % 64 == 0);
}

// ---------------------------------------------------------------------------
// Hopper backward kernel selection.
//
//   default          Vj1d  -- bit-identical dK/dV (cuDNN-matching)
//   USE_FLASH=1      Vj1   -- fastest, fully non-deterministic
//   USE_DETERMINISTIC_DQ=1
//                    v44   -- bit-identical dQ (per-CTA dq slices + fixed-order
//                             reduce), the slower reproducible-dQ path
//   USE_VJ2=1        Vj2   -- FA3-structured (2026-10-12), being validated; dense
//                             causal only, same determinism as Vj1d
//
// Setting BOTH opt-in flags is ambiguous: they select different kernels and are
// not composable. Warn once and fall back to the default rather than silently
// honouring one, which would make an A/B measure something other than its label.
// Read once into a function-local static: the value is cached for the process,
// so a setenv after the first backward launch has no effect.
// ---------------------------------------------------------------------------
enum class GqaBwdMode { Vj1d_Default, Vj1_Flash, V44_DetDQ, Vj2_Fa3 };

static inline GqaBwdMode gqa_bwd_mode() {
    static const GqaBwdMode mode = [] {
        auto on = [](const char* n) {
            const char* e = std::getenv(n);
            return e && e[0] == '1';
        };
        const bool flash  = on("USE_FLASH");
        const bool det_dq = on("USE_DETERMINISTIC_DQ");
        const bool vj2    = on("USE_VJ2");
        if ((int)flash + (int)det_dq + (int)vj2 > 1) {
            fprintf(stderr,
                    "[GQA bwd] WARNING: more than one of USE_FLASH=1, USE_DETERMINISTIC_DQ=1, USE_VJ2=1 is set. "
                    "They select different backward kernels and are not composable. "
                    "Falling back to the default Vj1d (bit-identical dK/dV).\n");
            return GqaBwdMode::Vj1d_Default;
        }
        if (flash)  return GqaBwdMode::Vj1_Flash;
        if (det_dq) return GqaBwdMode::V44_DetDQ;
        if (vj2)    return GqaBwdMode::Vj2_Fa3;
        return GqaBwdMode::Vj1d_Default;
    }();
    return mode;
}

// seg_start reaches only the Vj1d case: the callers refuse a non-null seg_start
// unless gqa_bwd_mode() is Vj1d_Default, so Vj1 / v44 never see packed rows.
static inline void gqa_hopper_causal_bwd_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* seg_start, const GqaChatPlan* chats) {
    switch (gqa_bwd_mode()) {
        case GqaBwdMode::Vj1_Flash:
            gqa_vj1_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale);
            return;
        case GqaBwdMode::V44_DetDQ:
            gqa_v44_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale);
            return;
        case GqaBwdMode::Vj2_Fa3:
            // Vj2 takes 128 keys per block. S % 128 == 64 stays on Vj1d: the forward's fast path needs S % 256 == 0
            // (gqa_v62_shape_ok), so training never sends such an S, and the tests (forward + backward) cannot reach
            // Vj2's half-size last block -- it is not run until it can be tested.
            if (S % 128 == 0) {
                gqa_vj2_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale);
                return;
            }
            gqa_vj1d_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,nullptr,nullptr,0);
            return;
        case GqaBwdMode::Vj1d_Default:
        default:
            gqa_vj1d_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,seg_start,
                                            chats ? chats->bwd_tiles : nullptr, chats ? chats->n_bwd_tiles : 0);
            return;
    }
}

namespace {  // internal linkage: portable WMMA backward kernels + launcher



// =============================================================================
// Legacy WMMA backward (gqa_backward_dQ / gqa_backward_dKdV / launch_gqa_backward)
// — kept as the NON-CAUSAL fallback path, exactly as the sibling forward keeps
// gqa_v2. The SM103 causal kernel below (V17) serves is_causal && head_dim==64;
// everything else routes here.
// =============================================================================
// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — dQ
// Grid  : (B, Hq, S/Br)  — each block owns one Q tile exclusively
// Block : 32 threads (one warp)
//
// Q, dO, O stay in smem throughout.
// K, V are reloaded each K-tile.
// dQ_acc lives in registers, written once to d_dQ at the end.
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
__global__ void gqa_backward_dQ(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO,
    const float         *d_LSE,
          __nv_bfloat16 *d_dQ,
    int B, int Hq, int Hkv, int G, int S, float scale
){
    const int b       = blockIdx.x;
    const int hq      = blockIdx.y;
    const int q_tile  = blockIdx.z;
    const int hkv     = hq / G;
    const int lane    = threadIdx.x;
    const int q_row0  = q_tile * Br;
    const int nKTiles = S / Bc;

    const long qBase  = ((long)(b * Hq  + hq)  * S + q_row0) * D;
    const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
    const long lBase  =  (long)(b * Hq  + hq)  * S + q_row0;

    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];   // also reused as store tmp
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sdS [Br * Bc];
    __shared__ float sLSE[Br];
    __shared__ float sD  [Br];

    // ── Load Q, dO, O — stay in smem for all K-tile iterations ───────────────
    for(int i = lane; i < Br * D; i += 32){
        sQ [i] = d_Q [qBase + i];
        sdO[i] = d_dO[qBase + i];
        sO [i] = d_O [qBase + i];
    }
    if(lane < Br) sLSE[lane] = d_LSE[lBase + lane];
    __syncwarp();

    // ── D[i] = rowsum(dO[i] ⊙ O[i])  (one scalar per query row) ─────────────
    if(lane < Br){
        float d = 0.0f;
        for(int j = 0; j < D; j++)
            d += __bfloat162float(sdO[lane * D + j]) *
                 __bfloat162float(sO [lane * D + j]);
        sD[lane] = d;
    }
    __syncwarp();

    // ── dQ accumulator: [Br, D] split into D/16 tiles of [16,16] in fp32 ─────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[D / 16];
    for(int t = 0; t < D / 16; t++) wmma::fill_fragment(dQ_acc[t], 0.0f);

    // ── Main loop: stream over all K tiles ────────────────────────────────────
    for(int kc = 0; kc < nKTiles; kc++){
        const long kBase = kvBase + (long)kc * Bc * D;

        for(int i = lane; i < Bc * D; i += 32){
            sK[i] = d_K[kBase + i];
            sV[i] = d_V[kBase + i];
        }
        __syncwarp();

        // S = Q @ K^T * scale → sS [Br, Bc]
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for(int nt = 0; nt < Bc / 16; nt++){
                wmma::fill_fragment(acc, 0.0f);
                for(int kt = 0; kt < D / 16; kt++){
                    wmma::load_matrix_sync(qf, sQ + kt * 16,            D);
                    wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, qf, kf, acc);
                }
                for(int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncwarp();

        // P = exp(S - LSE) → sP bf16
        if(lane < Br){
            float lse = sLSE[lane];
            for(int j = 0; j < Bc; j++)
                sP[lane * Bc + j] = __float2bfloat16(expf(sS[lane * Bc + j] - lse));
        }
        __syncwarp();

        // dP = dO @ V^T → sdP [Br, Bc] fp32
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for(int nt = 0; nt < Bc / 16; nt++){
                wmma::fill_fragment(acc, 0.0f);
                for(int kt = 0; kt < D / 16; kt++){
                    wmma::load_matrix_sync(dof, sdO + kt * 16,             D);
                    wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, dof, vf, acc);
                }
                wmma::store_matrix_sync(sdP + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncwarp();

        // dS = P ⊙ (dP - D) → sdS bf16
        if(lane < Br){
            float d = sD[lane];
            for(int j = 0; j < Bc; j++){
                float p  = __bfloat162float(sP [lane * Bc + j]);
                float dp = sdP[lane * Bc + j];
                sdS[lane * Bc + j] = __float2bfloat16(p * (dp - d));
            }
        }
        __syncwarp();

        // dQ_acc += dS @ K   (scale applied once after loop)
        // dS [Br,Bc] row_major,  K [Bc,D] row_major
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dsf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> kf;
            for(int nt = 0; nt < D / 16; nt++){
                for(int kt = 0; kt < Bc / 16; kt++){
                    wmma::load_matrix_sync(dsf, sdS + kt * 16,             Bc);
                    wmma::load_matrix_sync(kf,  sK  + kt * 16 * D + nt * 16, D);
                    wmma::mma_sync(dQ_acc[nt], dsf, kf, dQ_acc[nt]);
                }
            }
        }
        __syncwarp();
    }

    // ── Scale and write dQ_acc → d_dQ as bf16 ────────────────────────────────
    // Reuse sdP (Br*Bc fp32) as a [16,16] temp store buffer per tile.
    for(int nt = 0; nt < D / 16; nt++){
        for(int t = 0; t < dQ_acc[nt].num_elements; t++)
            dQ_acc[nt].x[t] *= scale;
        wmma::store_matrix_sync(sdP, dQ_acc[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for(int i = lane; i < Br * 16; i += 32){
            int r = i / 16, c = i % 16;
            d_dQ[qBase + r * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
        }
        __syncwarp();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — dK, dV
// Grid  : (B, Hkv, S/Bc)  — each block owns one K tile exclusively
// Block : 32 threads (one warp)
//
// K, V stay in smem throughout.
// Outer loop: G query heads; inner loop: S/Br Q tiles.
// dK_acc, dV_acc live in registers, written once at the end.
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
__global__ void gqa_backward_dKdV(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO,
    const float         *d_LSE,
          __nv_bfloat16 *d_dK,
          __nv_bfloat16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
){
    const int b      = blockIdx.x;
    const int hkv    = blockIdx.y;
    const int k_tile = blockIdx.z;
    const int lane   = threadIdx.x;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long kvBase = ((long)(b * Hkv + hkv) * S + k_row0) * D;

    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];   // stays all kernel
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];   // stays all kernel
    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];   // reloaded per Q tile
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];   // reloaded per Q tile
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];   // reloaded per Q tile
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];  // also reused as store tmp
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sdS [Br * Bc];
    __shared__ float sLSE[Br];
    __shared__ float sD  [Br];

    // ── Load K, V — fixed for this block ─────────────────────────────────────
    for(int i = lane; i < Bc * D; i += 32){
        sK[i] = d_K[kvBase + i];
        sV[i] = d_V[kvBase + i];
    }
    __syncwarp();

    // ── dK, dV accumulators: [Bc, D] = (Bc/16) x (D/16) tiles ───────────────
    constexpr int nAccTiles = (Bc / 16) * (D / 16);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[nAccTiles];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[nAccTiles];
    for(int t = 0; t < nAccTiles; t++){
        wmma::fill_fragment(dV_acc[t], 0.0f);
        wmma::fill_fragment(dK_acc[t], 0.0f);
    }

    // ── Outer: G query heads; Inner: S/Br Q tiles ────────────────────────────
    for(int g = 0; g < G; g++){
        const int hq = hkv * G + g;
        for(int qc = 0; qc < nQTiles; qc++){
            const int  q_row0 = qc * Br;
            const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
            const long lBase  =  (long)(b * Hq + hq) * S + q_row0;

            // Load Q, dO, O, LSE
            for(int i = lane; i < Br * D; i += 32){
                sQ [i] = d_Q [qBase + i];
                sdO[i] = d_dO[qBase + i];
                sO [i] = d_O [qBase + i];
            }
            if(lane < Br) sLSE[lane] = d_LSE[lBase + lane];
            __syncwarp();

            // D[i] = rowsum(dO[i] ⊙ O[i])
            if(lane < Br){
                float d = 0.0f;
                for(int j = 0; j < D; j++)
                    d += __bfloat162float(sdO[lane * D + j]) *
                         __bfloat162float(sO [lane * D + j]);
                sD[lane] = d;
            }
            __syncwarp();

            // S = Q @ K^T * scale → sS [Br, Bc]
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for(int nt = 0; nt < Bc / 16; nt++){
                    wmma::fill_fragment(acc, 0.0f);
                    for(int kt = 0; kt < D / 16; kt++){
                        wmma::load_matrix_sync(qf, sQ + kt * 16,            D);
                        wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, qf, kf, acc);
                    }
                    for(int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                    wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // P = exp(S - LSE) → sP bf16
            if(lane < Br){
                float lse = sLSE[lane];
                for(int j = 0; j < Bc; j++)
                    sP[lane * Bc + j] = __float2bfloat16(expf(sS[lane * Bc + j] - lse));
            }
            __syncwarp();

            // dV_acc += P^T @ dO
            // P  [Br,Bc] col_major trick gives P^T [Bc,Br]  — matrix_a
            // dO [Br,D]  row_major                           — matrix_b
            // Output [Bc,D] = (Bc/16) x (D/16) tiles, inner Br/16=1
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                for(int vt = 0; vt < Bc / 16; vt++){
                    for(int nt = 0; nt < D / 16; nt++){
                        for(int kt = 0; kt < Br / 16; kt++){
                            // P^T tile (vt,kt): load P tile (kt rows, vt cols) col_major
                            wmma::load_matrix_sync(pf,  sP  + kt * 16 * Bc + vt * 16, Bc);
                            wmma::load_matrix_sync(dof, sdO + kt * 16 * D  + nt * 16, D);
                            wmma::mma_sync(dV_acc[vt * (D/16) + nt], pf, dof,
                                           dV_acc[vt * (D/16) + nt]);
                        }
                    }
                }
            }
            __syncwarp();

            // dP = dO @ V^T → sdP [Br, Bc] fp32
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for(int nt = 0; nt < Bc / 16; nt++){
                    wmma::fill_fragment(acc, 0.0f);
                    for(int kt = 0; kt < D / 16; kt++){
                        wmma::load_matrix_sync(dof, sdO + kt * 16,             D);
                        wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, dof, vf, acc);
                    }
                    wmma::store_matrix_sync(sdP + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // dS = P ⊙ (dP - D) → sdS bf16
            if(lane < Br){
                float d = sD[lane];
                for(int j = 0; j < Bc; j++){
                    float p  = __bfloat162float(sP [lane * Bc + j]);
                    float dp = sdP[lane * Bc + j];
                    sdS[lane * Bc + j] = __float2bfloat16(p * (dp - d));
                }
            }
            __syncwarp();

            // dK_acc += dS^T @ Q   (scale applied once after loop)
            // dS [Br,Bc] col_major trick gives dS^T [Bc,Br]  — matrix_a
            // Q  [Br,D]  row_major                            — matrix_b
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> dsf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                for(int vt = 0; vt < Bc / 16; vt++){
                    for(int nt = 0; nt < D / 16; nt++){
                        for(int kt = 0; kt < Br / 16; kt++){
                            // dS^T tile (vt,kt): load dS tile (kt rows, vt cols) col_major
                            wmma::load_matrix_sync(dsf, sdS + kt * 16 * Bc + vt * 16, Bc);
                            wmma::load_matrix_sync(qf,  sQ  + kt * 16 * D  + nt * 16, D);
                            wmma::mma_sync(dK_acc[vt * (D/16) + nt], dsf, qf,
                                           dK_acc[vt * (D/16) + nt]);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }

    // ── Write dV_acc and dK_acc → global as bf16 ─────────────────────────────
    // Reuse sdP as [16,16] fp32 temp, one tile at a time.
    for(int vt = 0; vt < Bc / 16; vt++){
        for(int nt = 0; nt < D / 16; nt++){
            // dV — no scale needed
            wmma::store_matrix_sync(sdP, dV_acc[vt * (D/16) + nt], 16, wmma::mem_row_major);
            __syncwarp();
            for(int i = lane; i < 16 * 16; i += 32){
                int r = i / 16, c = i % 16;
                d_dV[kvBase + (vt * 16 + r) * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
            }
            __syncwarp();

            // dK — apply scale
            for(int t = 0; t < dK_acc[vt * (D/16) + nt].num_elements; t++)
                dK_acc[vt * (D/16) + nt].x[t] *= scale;
            wmma::store_matrix_sync(sdP, dK_acc[vt * (D/16) + nt], 16, wmma::mem_row_major);
            __syncwarp();
            for(int i = lane; i < 16 * 16; i += 32){
                int r = i / 16, c = i % 16;
                d_dK[kvBase + (vt * 16 + r) * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
            }
            __syncwarp();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed launcher
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
void launch_gqa_backward(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO,
    const float         *d_LSE,
          __nv_bfloat16 *d_dQ,
          __nv_bfloat16 *d_dK,
          __nv_bfloat16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
){
    // WMMA does 16x16x16 tiles, so every tiled dimension must be a multiple of 16.
    static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
    static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
    static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");
    // The score GEMMs only emit one 16-row WMMA tile (no loop over Br/16),
    // so these kernels process exactly 16 query rows per Q tile.
    static_assert(Br == 16, "gqa backward processes exactly 16 query rows per Q tile");

    dim3 BLOCK(32);

    // Kernel 1: dQ  — one block per Q tile
    dim3 GRID1(B, Hq, S / Br);
    gqa_backward_dQ<Br, Bc, D><<<GRID1, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ,
        B, Hq, Hkv, G, S, scale
    );

    // Kernel 2: dK, dV — one block per K tile
    dim3 GRID2(B, Hkv, S / Bc);
    gqa_backward_dKdV<Br, Bc, D><<<GRID2, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV,
        B, Hq, Hkv, G, S, scale
    );
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Packed-QKV backward wrapper. Causal hd==64: B300 -> tcgen05 (v17), else the
// portable WMMA backward (NON-causal; see file header note).
// ---------------------------------------------------------------------------
void gqa_flash_attn_backward(
    const bfloat16_t* qkv,
    const bfloat16_t* output,
    const bfloat16_t* grad_output,
    const float*      lse,
    bfloat16_t*       grad_qkv,
    float*            D_buf,
    int B, int Nq_heads, int Nkv_heads, int T, int hd,
    bool is_causal)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_backward: Nq_heads must be a positive multiple of Nkv_heads");
    }
    (void)D_buf;

    const int   G     = Nq_heads / Nkv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(qkv);
    const long qElems = (long)B * Nq_heads  * T * hd;
    const long kElems = (long)B * Nkv_heads * T * hd;

    const __nv_bfloat16* dQ_in = in;
    const __nv_bfloat16* dK_in = in + qElems;
    const __nv_bfloat16* dV_in = in + qElems + kElems;
    const __nv_bfloat16* dO  = reinterpret_cast<const __nv_bfloat16*>(output);
    const __nv_bfloat16* ddO = reinterpret_cast<const __nv_bfloat16*>(grad_output);

    __nv_bfloat16* gradBase = reinterpret_cast<__nv_bfloat16*>(grad_qkv);
    __nv_bfloat16* dQ_out   = gradBase;
    __nv_bfloat16* dK_out   = gradBase + qElems;
    __nv_bfloat16* dV_out   = gradBase + qElems + kElems;

    // Hopper (H100/H200) causal hd==128 -> v44 (wgmma/TMA, causal-correct).
    // BLUTRAIN_NO_SM90_GQA forces the generic path below. Disabling this also
    // removes compute_drowsum_v22_sm90 and convert_dq_accum_to_bf16_v5_sm90,
    // which are fixed sub-steps of the v44 launch. As on the forward side, there
    // is no generic causal kernel for hd==128, so that config throws.
    if (is_causal && !disable_sm90_gqa() &&
        get_arch() == ArchFamily::Hopper && gqa_v44_shape_ok(T, hd)) {
        gqa_hopper_causal_bwd_d128(
            dQ_in, dK_in, dV_in, dO, ddO, lse,
            dQ_out, dK_out, dV_out,
            B, Nq_heads, Nkv_heads, G, T, scale, /*seg_start=*/nullptr, /*chats=*/nullptr);
        return;
    }

    if (is_causal && hd == 64) {
        if (T % 128 != 0) {
            throw std::runtime_error(
                "gqa_flash_attn_backward: causal path requires T to be a multiple of 128");
        }
        if (get_arch() == ArchFamily::Blackwell_B300) {
            gqa_backward_v17_kv_launch_d64(
                dQ_in, dK_in, dV_in, dO, ddO, lse,
                dQ_out, dK_out, dV_out,
                B, Nq_heads, Nkv_heads, G, T, scale);
            return;
        }
        // Non-B300: portable WMMA backward (NON-causal fallback).
        launch_gqa_backward<16, 32, 64>(
            dQ_in, dK_in, dV_in, dO, ddO, lse,
            dQ_out, dK_out, dV_out,
            B, Nq_heads, Nkv_heads, G, T, scale);
        return;
    }

    if (is_causal) {
        throw std::runtime_error(
            "gqa_flash_attn_backward: causal attention is only implemented for head_dim 64");
    }

    if (T % 32 != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_backward: T must be a multiple of 32 (tile size)");
    }

    switch (hd) {
        case 64:
            launch_gqa_backward<16, 32, 64>(
                dQ_in, dK_in, dV_in, dO, ddO, lse, dQ_out, dK_out, dV_out,
                B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        case 128:
            launch_gqa_backward<16, 32, 128>(
                dQ_in, dK_in, dV_in, dO, ddO, lse, dQ_out, dK_out, dV_out,
                B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        default:
            throw std::runtime_error(
                "gqa_flash_attn_backward: only head_dim 64 or 128 are compiled");
    }
}

// ---------------------------------------------------------------------------
// Unpacked-QKV backward wrapper. Same dispatch rule.
// ---------------------------------------------------------------------------
void gqa_flash_attn_backward_qkv(
    const bfloat16_t* q,
    const bfloat16_t* k,
    const bfloat16_t* v,
    const bfloat16_t* output,
    const bfloat16_t* grad_output,
    const float*      lse,
    bfloat16_t*       grad_q,
    bfloat16_t*       grad_k,
    bfloat16_t*       grad_v,
    int B, int Nq_heads, int Nkv_heads, int T, int hd,
    bool is_causal, const int32_t* seg_start, const GqaChatPlan* chats)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_backward_qkv: Nq_heads must be a positive multiple of Nkv_heads");
    }

    const int   G     = Nq_heads / Nkv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    const __nv_bfloat16* dQ_in = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* dK_in = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* dV_in = reinterpret_cast<const __nv_bfloat16*>(v);
    const __nv_bfloat16* dO    = reinterpret_cast<const __nv_bfloat16*>(output);
    const __nv_bfloat16* ddO   = reinterpret_cast<const __nv_bfloat16*>(grad_output);

    __nv_bfloat16* dQ_out = reinterpret_cast<__nv_bfloat16*>(grad_q);
    __nv_bfloat16* dK_out = reinterpret_cast<__nv_bfloat16*>(grad_k);
    __nv_bfloat16* dV_out = reinterpret_cast<__nv_bfloat16*>(grad_v);

    const bool hopper_ok = is_causal && !disable_sm90_gqa() &&
                           get_arch() == ArchFamily::Hopper && gqa_v44_shape_ok(T, hd);

    // v2 packed rows (design B, 2026-10-10): only the Hopper Vj1d kernel knows seg_start. Vj1
    // (USE_FLASH=1), v44 (USE_DETERMINISTIC_DQ=1) and every non-Hopper path below would IGNORE it and
    // silently let the chats of a row see each other, so a seg_start that cannot be honoured is
    // refused here instead (same rule as gqa_flash_attn_forward_qkv).
    if (seg_start != nullptr && !(hopper_ok && gqa_bwd_mode() == GqaBwdMode::Vj1d_Default)) {
        throw std::runtime_error(
            "gqa_flash_attn_backward_qkv: seg_start (packed rows) needs the Hopper Vj1d path: "
            "causal, head_dim 128, T % 64 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset, "
            "and none of USE_FLASH=1, USE_DETERMINISTIC_DQ=1, USE_VJ2=1");
    }
    // v2 design A: the same rule for a plan, which must be the one the forward got (same shape).
    if (chats != nullptr) {
        if (seg_start != nullptr)
            throw std::runtime_error("gqa_flash_attn_backward_qkv: pass seg_start (design B) OR chats "
                                     "(design A), not both");
        if (!(hopper_ok && gqa_bwd_mode() == GqaBwdMode::Vj1d_Default))
            throw std::runtime_error(
                "gqa_flash_attn_backward_qkv: chats (packed rows, design A) needs the Hopper Vj1d path: "
                "causal, head_dim 128, T % 64 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset, "
                "and none of USE_FLASH=1, USE_DETERMINISTIC_DQ=1, USE_VJ2=1");
        if (chats->B != B || chats->S != T || chats->Hq != Nq_heads || chats->bwd_tiles == nullptr)
            throw std::runtime_error("gqa_flash_attn_backward_qkv: chats plan was built for another shape "
                                     "(B, T, Nq_heads) or is empty");
    }

    // Hopper (H100/H200) causal hd==128 -> v44 (wgmma/TMA, causal-correct).
    // This is the path bluscript's "split" attn_mode reaches.
    // BLUTRAIN_NO_SM90_GQA forces the generic path below. Disabling this also
    // removes compute_drowsum_v22_sm90 and convert_dq_accum_to_bf16_v5_sm90,
    // which are fixed sub-steps of the v44 launch. As on the forward side, there
    // is no generic causal kernel for hd==128, so that config throws.
    if (hopper_ok) {
        gqa_hopper_causal_bwd_d128(
            dQ_in, dK_in, dV_in, dO, ddO, lse,
            dQ_out, dK_out, dV_out,
            B, Nq_heads, Nkv_heads, G, T, scale, seg_start, chats);
        return;
    }

    if (is_causal && hd == 64) {
        if (T % 128 != 0) {
            throw std::runtime_error(
                "gqa_flash_attn_backward_qkv: causal path requires T to be a multiple of 128");
        }
        if (get_arch() == ArchFamily::Blackwell_B300) {
            gqa_backward_v17_kv_launch_d64(
                dQ_in, dK_in, dV_in, dO, ddO, lse,
                dQ_out, dK_out, dV_out,
                B, Nq_heads, Nkv_heads, G, T, scale);
            return;
        }
        launch_gqa_backward<16, 32, 64>(
            dQ_in, dK_in, dV_in, dO, ddO, lse,
            dQ_out, dK_out, dV_out,
            B, Nq_heads, Nkv_heads, G, T, scale);
        return;
    }

    if (is_causal) {
        throw std::runtime_error(
            "gqa_flash_attn_backward_qkv: causal attention is only implemented for head_dim 64");
    }

    if (T % 32 != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_backward_qkv: T must be a multiple of 32 (tile size)");
    }

    switch (hd) {
        case 64:
            launch_gqa_backward<16, 32, 64>(
                dQ_in, dK_in, dV_in, dO, ddO, lse, dQ_out, dK_out, dV_out,
                B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        case 128:
            launch_gqa_backward<16, 32, 128>(
                dQ_in, dK_in, dV_in, dO, ddO, lse, dQ_out, dK_out, dV_out,
                B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        default:
            throw std::runtime_error(
                "gqa_flash_attn_backward_qkv: only head_dim 64 or 128 are compiled");
    }
}

} // namespace cuda
} // namespace OwnTensor
