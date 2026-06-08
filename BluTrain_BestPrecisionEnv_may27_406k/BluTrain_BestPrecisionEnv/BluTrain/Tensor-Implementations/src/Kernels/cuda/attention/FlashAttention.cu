#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "TensorLib.h"
#include "autograd/operations/FlashAttentionOps.h"
using namespace OwnTensor;
using namespace autograd;
namespace OwnTensor {
   namespace autograd {
     // ── Kernel tuning constants ────────────────────────────────────────────────────
// BLOCK_M  : Q rows per block (= warps per block).  8 → each K/V tile is reused
//            by 8 Q rows instead of 4, halving K/V HBM reads.  smem is unchanged.
// WARP_SIZE: threads per warp (hardware constant = 32)
//
// BlockN = min(64, 4096 / HeadDim)
//   Capped at 64 so s_tile[BlockN] never exceeds 64 register floats per thread.
//   smem = 2 * BlockN * HeadDim * 4 ≤ 32 KB for all supported HeadDim values.
//
// LocalN = ceil(HeadDim / WARP_SIZE)
//   Each lane only accumulates its stride-WARP_SIZE slice of Q and O, so we
//   allocate q_local[LocalN] and o_local[LocalN] instead of q_reg[HeadDim] and
//   o_reg[HeadDim].  For d=128 this cuts register arrays from 256 to 8 floats/thread,
//   eliminating the spilling that was the primary bottleneck at that size.
#define BLOCK_M   8
#define WARP_SIZE 32

// ── Warp reductions ────────────────────────────────────────────────────────────
// __shfl_xor_sync distributes the result to ALL lanes (no separate broadcast needed)

__inline__ __device__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    return val;
}

__inline__ __device__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset));
    return val;
}

// ── Flash Attention V2 kernel ──────────────────────────────────────────────────
//
// Implements Algorithm 1 (FlashAttention), steps 9-12, with the FA2 variant
// that keeps O un-normalised throughout and divides by ℓ only once at the end
// (avoids repeated HBM writes for O_i).
//
// Steps 9-11 are performed at tile granularity:
//   Step  9: S_{ij} = Q_i · K_j^T / sqrt(d)   — full tile score matrix
//   Step 10: m̃ = rowmax(S),  P̃ = exp(S − m̃),  ℓ̃ = rowsum(P̃)
//   Step 11: m_new = max(m_i, m̃),  ℓ_new = e^(m_i−m_new)·ℓ_i + e^(m̃−m_new)·ℓ̃
//   Step 12: O ← e^(m_i−m_new)·O + e^(m̃−m_new)·P̃·V  (un-normalised FA2 form)
//
// Template on HeadDim so all compile-time arrays (q_local[LocalN], o_local[LocalN],
// s_tile[BlockN]) are sized at compile time without any dynamic allocation.
//
// Grid : x = ceil(N / BLOCK_M),  y = B * H
// Block: BLOCK_M * WARP_SIZE threads  →  BLOCK_M warps, one warp per Q row

template<int HeadDim, bool Causal = false>
__global__ void flash_attn_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ L_out,   // [B*H, N]  log-sum-exp per row (may be nullptr)
    int B, int H, int N
) {
    constexpr int BlockN = (HeadDim < 64) ? 64 : (4096 / HeadDim);
    constexpr int LocalN = (HeadDim + WARP_SIZE - 1) / WARP_SIZE;

    // ── vec4 constants ────────────────────────────────────────────────────────
    // UseVec4 : true when HeadDim % 4 == 0.  Every global and smem access is
    //           widened to float4, issuing LDG.128/STG.128 instructions and
    //           reducing the transaction count by 4×.
    // HD4     : HeadDim expressed in float4 units  (e.g. 128 → 32).
    // LocalN4 : ceil(HD4 / WARP_SIZE) — float4 slices per lane.
    //           Lane l owns output columns {4(l+i·32), …, 4(l+i·32)+3} for
    //           i = 0 … LocalN4−1.  For HeadDim=64 only lanes 0–15 are active
    //           (col4 = lane_id < HD4=16); inactive lanes are guarded by the
    //           `col4 < HD4` check and contribute zero to every reduction.
    constexpr bool UseVec4 = (HeadDim % 4 == 0);
    constexpr int  HD4     = HeadDim / 4;
    constexpr int  LocalN4 = UseVec4 ? (HD4 + WARP_SIZE - 1) / WARP_SIZE : 1;

    // ── shared memory: K_tile | V_tile ────────────────────────────────────────
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + BlockN * HeadDim;

    // ── identify (batch, head) and query row ──────────────────────────────────
    int bh = blockIdx.y;
    int b  = bh / H;
    int h  = bh % H;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int q_row = blockIdx.x * BLOCK_M + warp_id;
    if (q_row >= N) return;

    const long long bh_off = (long long)(b * H + h) * N * HeadDim;
    const float* Q_ptr = Q + bh_off;
    const float* K_ptr = K + bh_off;
    const float* V_ptr = V + bh_off;
    float*       O_ptr = O + bh_off;

    const float scale = 1.0f / sqrtf((float)HeadDim);

    // ═══════════════════════════════════════════════════════════════════════════
    // PATH A — float4  (HeadDim divisible by 4)
    // ═══════════════════════════════════════════════════════════════════════════
    if constexpr (UseVec4) {

        // ── Q load: each lane l owns LocalN4 float4 slices ───────────────────
        // q_v4[i] = Q[q_row,  4*(lane_id + i·32) … 4*(lane_id + i·32)+3 ]
        // LDG.128 instruction: 4× fewer global transactions vs scalar.
        float4 q_v4[LocalN4];
        #pragma unroll
        for (int i = 0; i < LocalN4; ++i) {
            int col4 = lane_id + i * WARP_SIZE;
            q_v4[i] = (col4 < HD4)
                ? __ldg(reinterpret_cast<const float4*>(Q_ptr + q_row * HeadDim) + col4)
                : make_float4(0.f, 0.f, 0.f, 0.f);
        }

        float  m_i = -INFINITY;
        float  l_i = 0.0f;
        float4 o_v4[LocalN4];
        #pragma unroll
        for (int i = 0; i < LocalN4; ++i) o_v4[i] = make_float4(0.f, 0.f, 0.f, 0.f);

        // ── loop over KV tiles ────────────────────────────────────────────────
        for (int kv = 0; kv < N; kv += BlockN) {
            if constexpr (Causal) { if (kv > q_row) break; }

            // ── K/V tile load (float4 — LDG.128, 4× fewer HBM transactions) ──
            const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int row = warp_id; row < BlockN; row += BLOCK_M) {
                int g_row = kv + row;
                const float4* Ksrc = reinterpret_cast<const float4*>(K_ptr + g_row * HeadDim);
                const float4* Vsrc = reinterpret_cast<const float4*>(V_ptr + g_row * HeadDim);
                float4*       Kdst = reinterpret_cast<float4*>(K_tile + row * HeadDim);
                float4*       Vdst = reinterpret_cast<float4*>(V_tile + row * HeadDim);
                #pragma unroll
                for (int col4 = lane_id; col4 < HD4; col4 += WARP_SIZE) {
                    Kdst[col4] = (g_row < N) ? __ldg(Ksrc + col4) : zero4;
                    Vdst[col4] = (g_row < N) ? __ldg(Vsrc + col4) : zero4;
                }
            }
            __syncthreads();

            int   tile_n = min(BlockN, N - kv);
            float s_tile[BlockN];

            // ── Step 9: S = Q·K^T / sqrt(d) — outer-product block multiply ───────
            // Loop order: outer = HeadDim slices (i), inner = K-tile rows (j).
            // q_v4[i] is loaded once per slice and multiplied against every K[j]
            // that shares the same float4 column offset, so the register value is
            // fully reused across all tile_n K vectors.  Partial sums accumulate
            // in s_tile[0..BlockN-1] (register-resident); warp_sum is deferred to
            // after the accumulation loop, keeping the inner FMA body free of
            // synchronisation barriers and letting the scheduler overlap FMAs from
            // consecutive j iterations for better instruction-level parallelism.
            #pragma unroll
            for (int j = 0; j < BlockN; ++j) s_tile[j] = 0.0f;

            #pragma unroll
            for (int i = 0; i < LocalN4; ++i) {
                int col4 = lane_id + i * WARP_SIZE;
                if (col4 < HD4) {
                    float4 q4 = q_v4[i];            // hoisted: reused for all j
                    for (int j = 0; j < tile_n; ++j) {
                        float4 k4 = reinterpret_cast<const float4*>(K_tile + j * HeadDim)[col4];
                        s_tile[j] += q4.x*k4.x + q4.y*k4.y + q4.z*k4.z + q4.w*k4.w;
                    }
                }
            }

            // Reduce across HeadDim lanes, apply scale + causal mask, find row-max
            float m_tilde = -INFINITY;
            for (int j = 0; j < tile_n; ++j) {
                float score = warp_sum(s_tile[j]) * scale;
                if constexpr (Causal) { if ((kv + j) > q_row) score = -INFINITY; }
                s_tile[j] = score;
                m_tilde   = fmaxf(m_tilde, score);
            }

            // ── Step 10: P̃ = exp(S − m̃),  ℓ̃ = rowsum(P̃) ────────────────────
            float l_tilde = 0.0f;
            for (int j = 0; j < tile_n; ++j) {
                s_tile[j] = __expf(s_tile[j] - m_tilde);
                l_tilde  += s_tile[j];
            }

            // ── Step 11: update running max and normalizer ───────────────────
            float m_new      = fmaxf(m_i, m_tilde);
            float alpha_i    = __expf(m_i     - m_new);
            float alpha_tile = __expf(m_tilde - m_new);
            l_i = alpha_i * l_i + alpha_tile * l_tilde;
            m_i = m_new;

            // ── Step 12: O ← alpha_i·O + alpha_tile·P̃·V (float4 V access) ───
            // Each lane accumulates its 4-wide column slice of the P·V product.
            #pragma unroll
            for (int i = 0; i < LocalN4; ++i) {
                int col4 = lane_id + i * WARP_SIZE;
                if (col4 < HD4) {
                    float4 pv4 = make_float4(0.f, 0.f, 0.f, 0.f);
                    for (int j = 0; j < tile_n; ++j) {
                        float4 v4 = reinterpret_cast<const float4*>(V_tile + j * HeadDim)[col4];
                        pv4.x += s_tile[j] * v4.x;  pv4.y += s_tile[j] * v4.y;
                        pv4.z += s_tile[j] * v4.z;  pv4.w += s_tile[j] * v4.w;
                    }
                    o_v4[i].x = o_v4[i].x * alpha_i + alpha_tile * pv4.x;
                    o_v4[i].y = o_v4[i].y * alpha_i + alpha_tile * pv4.y;
                    o_v4[i].z = o_v4[i].z * alpha_i + alpha_tile * pv4.z;
                    o_v4[i].w = o_v4[i].w * alpha_i + alpha_tile * pv4.w;
                }
            }

            __syncthreads();
        }

        // ── normalize and store (STG.128) ─────────────────────────────────────
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int i = 0; i < LocalN4; ++i) {
            int col4 = lane_id + i * WARP_SIZE;
            if (col4 < HD4) {
                float4 out = make_float4(o_v4[i].x * inv_l, o_v4[i].y * inv_l,
                                         o_v4[i].z * inv_l, o_v4[i].w * inv_l);
                reinterpret_cast<float4*>(O_ptr + q_row * HeadDim)[col4] = out;
            }
        }

        // ── write log-sum-exp L[row] = m + log(l) ───────────────────────────
        if (L_out && lane_id == 0) {
            long long L_idx = (long long)(b * H + h) * N + q_row;
            L_out[L_idx] = m_i + logf(l_i);
        }

    // ═══════════════════════════════════════════════════════════════════════════
    // PATH B — scalar fallback  (HeadDim not divisible by 4, e.g. HeadDim=2)
    // ═══════════════════════════════════════════════════════════════════════════
    } else {

        float q_local[LocalN];
        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * WARP_SIZE;
            q_local[i] = (k < HeadDim) ? __ldg(&Q_ptr[q_row * HeadDim + k]) : 0.0f;
        }

        float m_i = -INFINITY;
        float l_i = 0.0f;
        float o_local[LocalN];
        #pragma unroll
        for (int i = 0; i < LocalN; ++i) o_local[i] = 0.0f;

        for (int kv = 0; kv < N; kv += BlockN) {
            if constexpr (Causal) { if (kv > q_row) break; }

            for (int row = warp_id; row < BlockN; row += BLOCK_M) {
                int g_row = kv + row;
                #pragma unroll
                for (int col = lane_id; col < HeadDim; col += WARP_SIZE) {
                    K_tile[row * HeadDim + col] = (g_row < N) ? __ldg(&K_ptr[g_row * HeadDim + col]) : 0.0f;
                    V_tile[row * HeadDim + col] = (g_row < N) ? __ldg(&V_ptr[g_row * HeadDim + col]) : 0.0f;
                }
            }
            __syncthreads();

            int   tile_n = min(BlockN, N - kv);
            float s_tile[BlockN];

            // Outer-product accumulation — same loop-order rationale as PATH A.
            #pragma unroll
            for (int j = 0; j < BlockN; ++j) s_tile[j] = 0.0f;

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * WARP_SIZE;
                if (k < HeadDim) {
                    float q_val = q_local[i];       // hoisted: reused for all j
                    for (int j = 0; j < tile_n; ++j)
                        s_tile[j] += q_val * K_tile[j * HeadDim + k];
                }
            }

            float m_tilde = -INFINITY;
            for (int j = 0; j < tile_n; ++j) {
                float score = warp_sum(s_tile[j]) * scale;
                if constexpr (Causal) { if ((kv + j) > q_row) score = -INFINITY; }
                s_tile[j] = score;
                m_tilde   = fmaxf(m_tilde, score);
            }

            float l_tilde = 0.0f;
            for (int j = 0; j < tile_n; ++j) {
                s_tile[j] = __expf(s_tile[j] - m_tilde);
                l_tilde  += s_tile[j];
            }

            float m_new      = fmaxf(m_i, m_tilde);
            float alpha_i    = __expf(m_i     - m_new);
            float alpha_tile = __expf(m_tilde - m_new);
            l_i = alpha_i * l_i + alpha_tile * l_tilde;
            m_i = m_new;

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * WARP_SIZE;
                if (k < HeadDim) {
                    float pv = 0.0f;
                    for (int j = 0; j < tile_n; ++j)
                        pv += s_tile[j] * V_tile[j * HeadDim + k];
                    o_local[i] = o_local[i] * alpha_i + alpha_tile * pv;
                }
            }

            __syncthreads();
        }

        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * WARP_SIZE;
            if (k < HeadDim)
                O_ptr[q_row * HeadDim + k] = o_local[i] * inv_l;
        }

        // ── write log-sum-exp L[row] = m + log(l) ───────────────────────────
        if (L_out && lane_id == 0) {
            long long L_idx = (long long)(b * H + h) * N + q_row;
            L_out[L_idx] = m_i + logf(l_i);
        }
    }
}

// ── Host launcher — dispatches to the right HeadDim instantiation ─────────────

void launch_flash_attn(
    const float* Q, const float* K, const float* V, float* O, float* L,
    int B, int H, int N, int d, bool causal
) {
    dim3 block(BLOCK_M * WARP_SIZE);
    dim3 grid((N + BLOCK_M - 1) / BLOCK_M, B * H);
    // Mirror the kernel's BlockN = min(64, 4096/d) so smem matches exactly.
    int    block_n = (d < 64) ? 64 : (4096 / d);
    size_t smem    = 2ULL * block_n * d * sizeof(float);

    // One macro to avoid repeating the causal branch for every head_dim.
#define LAUNCH(D) \
    do { \
        if (causal) flash_attn_v2<D, true> <<<grid, block, smem>>>(Q, K, V, O, L, B, H, N); \
        else        flash_attn_v2<D, false><<<grid, block, smem>>>(Q, K, V, O, L, B, H, N); \
    } while (0)

    switch (d) {
        case   2: LAUNCH(  2); break;
        case   4: LAUNCH(  4); break;
        case   8: LAUNCH(  8); break;
        case  16: LAUNCH( 16); break;
        case  24: LAUNCH( 24); break;
        case  32: LAUNCH( 32); break;
        case  40: LAUNCH( 40); break;
        case  48: LAUNCH( 48); break;
        case  56: LAUNCH( 56); break;
        case  64: LAUNCH( 64); break;
        case  80: LAUNCH( 80); break;
        case  96: LAUNCH( 96); break;
        case 128: LAUNCH(128); break;
        case 160: LAUNCH(160); break;
        case 192: LAUNCH(192); break;
        case 256: LAUNCH(256); break;
        default:
            printf("Unsupported head_dim %d — must be an even value in {8,16,...,96,128,160,192,256}.\n", d);
            return;
    }
#undef LAUNCH

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
}
   }
}

// ══════════════════════════════════════════════════════════════════════════════
//  BACKWARD PASS
// ══════════════════════════════════════════════════════════════════════════════

// ── Precompute D[i] = Σ_k dO[i,k]·O[i,k] ──────────────────────────────────
// Grid: (ceil(T/BLOCK_M), BH)    Block: BLOCK_M * WARP_SIZE
// Each warp handles one row — same parallelism pattern as the forward kernel
template<int HeadDim>
__global__ void flash_attn_bwd_precompute_D(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    int T)
{
    constexpr int LocalN = (HeadDim + WARP_SIZE - 1) / WARP_SIZE;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * BLOCK_M + warp_id;
    if (row >= T) return;

    const long long off = (long long)bh * T * HeadDim + row * HeadDim;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        int k = lane_id + i * WARP_SIZE;
        if (k < HeadDim)
            sum += dO[off + k] * O[off + k];
    }
    sum = warp_sum(sum);

    if (lane_id == 0)
        D[(long long)bh * T + row] = sum;
}

// ── Flash Attention backward kernel (warp-parallel, KV-tile-centric) ────────
//
// Grid  : (ceil(T / BlockN), BH)
// Block : BLOCK_M * WARP_SIZE threads  =  BLOCK_M warps
//
// Shared memory: K_tile + V_tile + dK_acc + dV_acc  (4 × BlockN × HeadDim)
//
// Each block owns one KV tile.  The outer loop sweeps Q rows BLOCK_M at a
// time — each warp independently handles one Q row using warp_sum (no
// __syncthreads inside the hot loop).  dK/dV use shared-memory atomicAdd
// (8-way contention across warps; native float atomicAdd on sm_80+).
// dQ is accumulated per-warp in registers, written to global via atomicAdd.
template<int HeadDim, bool Causal = false>
__global__ void flash_attn_bwd_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,     // [BH, T]
    const float* __restrict__ D,     // [BH, T]
    float* __restrict__ dQ,          // [BH, T, HeadDim]  zeroed
    float* __restrict__ dK,          // [BH, T, HeadDim]
    float* __restrict__ dV,          // [BH, T, HeadDim]
    int T, float scale)
{
    // Half the forward's BlockN: 4 smem arrays (K,V,dK,dV) + scratch must fit in 48KB.
    constexpr int FwdBlockN = (HeadDim < 64) ? 64 : (4096 / HeadDim);
    constexpr int BlockN = FwdBlockN / 2;
    constexpr int LocalN = (HeadDim + WARP_SIZE - 1) / WARP_SIZE;

    // ── shared memory layout ────────────────────────────────────────────────
    // Phase A stores ds_buf[BLOCK_M*BlockN] and p_buf[BLOCK_M*BlockN] for
    // cross-warp reduction in Phase B, eliminating shared-memory atomicAdd.
    // Also stores Q_buf[BLOCK_M*HeadDim] and dO_buf[BLOCK_M*HeadDim] loaded
    // per q_base iteration so Phase B can access any warp's Q/dO.
    extern __shared__ float smem[];
    float* Ks    = smem;                                     // [BlockN * HeadDim]
    float* Vs    = Ks    + BlockN * HeadDim;                 // [BlockN * HeadDim]
    float* dKs   = Vs    + BlockN * HeadDim;                 // [BlockN * HeadDim]
    float* dVs   = dKs   + BlockN * HeadDim;                 // [BlockN * HeadDim]
    float* ds_buf = dVs  + BlockN * HeadDim;                 // [BLOCK_M * BlockN]
    float* p_buf  = ds_buf + BLOCK_M * BlockN;               // [BLOCK_M * BlockN]
    float* Q_buf  = p_buf  + BLOCK_M * BlockN;               // [BLOCK_M * HeadDim]
    float* dO_buf = Q_buf  + BLOCK_M * HeadDim;              // [BLOCK_M * HeadDim]

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, T - tile_start);
    if (tile_start >= T) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const long long bh_off = (long long)bh * T * HeadDim;
    const long long bh_T   = (long long)bh * T;

    const float* Q_bh  = Q  + bh_off;
    const float* K_bh  = K  + bh_off;
    const float* V_bh  = V  + bh_off;
    const float* dO_bh = dO + bh_off;
    const float* L_bh  = L  + bh_T;
    const float* D_bh  = D  + bh_T;
    float*       dQ_bh = dQ + bh_off;
    float*       dK_bh = dK + bh_off;
    float*       dV_bh = dV + bh_off;

    // ── cooperative load K_tile, V_tile; zero dKs, dVs ──────────────────────
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim;
        int g_row = tile_start + r;
        Ks [idx] = (g_row < T) ? K_bh[g_row * HeadDim + (idx % HeadDim)] : 0.0f;
        Vs [idx] = (g_row < T) ? V_bh[g_row * HeadDim + (idx % HeadDim)] : 0.0f;
        dKs[idx] = 0.0f;
        dVs[idx] = 0.0f;
    }
    __syncthreads();

    // ── outer loop: sweep Q rows, BLOCK_M at a time ────────────────────────
    int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < T; q_base += BLOCK_M) {
        int qi = q_base + warp_id;
        bool valid = (qi < T);

        // Load Q[qi], dO[qi] into lane-local registers AND shared Q_buf/dO_buf
        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            int k = lane_id + i * WARP_SIZE;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * HeadDim + k];
                dov = dO_bh[qi * HeadDim + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
            // Store in shared for Phase B cooperative reduction
            if (k < HeadDim) {
                Q_buf [warp_id * HeadDim + k] = qv;
                dO_buf[warp_id * HeadDim + k] = dov;
            }
        }
        if (valid) {
            L_qi = L_bh[qi];
            D_qi = D_bh[qi];
        }

        // ── Phase A: each warp computes ds[j], p[j] for its Q row ──────────
        //    Also accumulates dQ in registers (no contention).
        for (int j = 0; j < tile_size; ++j) {

            // s = dot(Q[qi], K[j]) * scale
            float dot_qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * WARP_SIZE;
                if (k < HeadDim)
                    dot_qk += q_local[i] * Ks[j * HeadDim + k];
            }
            float s = warp_sum(dot_qk) * scale;

            // P = exp(S − L), with causal mask
            float p;
            if (!valid) {
                p = 0.0f;
            } else if (Causal && (tile_start + j) > qi) {
                p = 0.0f;
            } else {
                p = __expf(s - L_qi);
            }

            // dpv = dot(dO[qi], V[j])
            float dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * WARP_SIZE;
                if (k < HeadDim)
                    dot_dov += do_local[i] * Vs[j * HeadDim + k];
            }
            float dpv = warp_sum(dot_dov);

            float ds = p * (dpv - D_qi);

            // dQ[qi] += scale · dS · K[j]  (register — no contention)
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * WARP_SIZE;
                if (k < HeadDim)
                    dq_local[i] += ds * scale * Ks[j * HeadDim + k];
            }

            // Store ds, p for Phase B (lane 0 writes — all lanes have same value)
            if (lane_id == 0) {
                ds_buf[warp_id * BlockN + j] = ds;
                p_buf [warp_id * BlockN + j] = p;
            }
        } // end j loop (Phase A)

        __syncthreads();  // Q_buf, dO_buf, ds_buf, p_buf ready

        // ── Phase B: cooperatively update dK, dV (no atomics) ───────────────
        // dK[j][k] += Σ_w ds_buf[w][j] · scale · Q_buf[w][k]
        // dV[j][k] += Σ_w  p_buf[w][j] ·          dO_buf[w][k]
        // 256 threads ÷ (tile_size × HeadDim) work items
        for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
            int j = idx / HeadDim;
            int k = idx % HeadDim;
            float dk_acc = 0.0f, dv_acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < BLOCK_M; ++w) {
                float ds_wj = ds_buf[w * BlockN + j];
                float p_wj  = p_buf [w * BlockN + j];
                dk_acc += ds_wj * Q_buf [w * HeadDim + k];
                dv_acc += p_wj  * dO_buf[w * HeadDim + k];
            }
            dKs[idx] += dk_acc * scale;
            dVs[idx] += dv_acc;
        }

        __syncthreads();  // dKs, dVs updated; safe for next q_base

        // Write dQ[qi] to global (atomicAdd: multiple KV-tile blocks contribute)
        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                int k = lane_id + i * WARP_SIZE;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * HeadDim + k], dq_local[i]);
            }
        }
    } // end q_base loop

    __syncthreads();   // all warps done accumulating into dKs, dVs

    // ── write dK_tile, dV_tile to global (one block per tile, no contention) ─
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim;
        dK_bh[(tile_start + r) * HeadDim + (idx % HeadDim)] = dKs[idx];
        dV_bh[(tile_start + r) * HeadDim + (idx % HeadDim)] = dVs[idx];
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// dispatch_flash_attention_bwd
//
//   Q, K, V, O : [B, n_heads, T, head_dim]   (from forward)
//   dO         : [B, n_heads, T, head_dim]   (upstream gradient)
//   L          : [B*n_heads, T]              (log-sum-exp from forward)
//   scale      : attention scale
//
//   Returns {dQ, dK, dV}  each [B, n_heads, T, head_dim]
// ──────────────────────────────────────────────────────────────────────────────
namespace OwnTensor {

std::vector<Tensor> dispatch_flash_attention_bwd(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& O,
    const Tensor& dO,
    const Tensor& L,
    int B, int n_heads, int T, int head_dim,
    float scale)
{
    if (scale <= 0.0f)
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    TensorOptions opts = TensorOptions()
        .with_device(Device::CUDA).with_dtype(Dtype::Float32);

    // dQ must be zeroed (atomicAdd target); dK, dV written by blocks directly
    Tensor dQ_t = Tensor::zeros({{B, n_heads, T, head_dim}}, opts);
    Tensor dK_t = Tensor::zeros({{B, n_heads, T, head_dim}}, opts);
    Tensor dV_t = Tensor::zeros({{B, n_heads, T, head_dim}}, opts);

    const int BH = B * n_heads;

    // D buffer [BH, T] for precomputed row-wise dot(dO, O)
    Tensor D_t = Tensor::zeros({{BH, T}}, opts);

    const float* q_base  = Q.data<float>();
    const float* k_base  = K.data<float>();
    const float* v_base  = V.data<float>();
    const float* o_base  = O.data<float>();
    const float* do_base = dO.data<float>();
    const float* l_base  = L.data<float>();
    float*       d_base  = D_t.data<float>();
    float*       dq_base = dQ_t.data<float>();
    float*       dk_base = dK_t.data<float>();
    float*       dv_base = dV_t.data<float>();

    // Block/grid config matches forward: BLOCK_M warps per block
    dim3 block_cfg(BLOCK_M * WARP_SIZE);  // 256 threads

    // BwdBlockN = half of forward's BlockN to keep smem under 48KB
    int fwd_block_n = (head_dim < 64) ? 64 : (4096 / head_dim);
    int bwd_block_n = fwd_block_n / 2;

    // Step 1: precompute D — grid: (ceil(T/BLOCK_M), BH)
    dim3 grid_D((T + BLOCK_M - 1) / BLOCK_M, BH);

    // Step 2: main backward — grid: (kv_tiles, BH)
    int kv_tiles = (T + bwd_block_n - 1) / bwd_block_n;
    dim3 grid_bwd(kv_tiles, BH);
    // smem: 4*BlockN*d (K,V,dK,dV) + 2*BLOCK_M*BlockN (ds,p) + 2*BLOCK_M*d (Q,dO)
    size_t shmem_bwd = (4ULL * bwd_block_n * head_dim
                      + 2ULL * BLOCK_M * bwd_block_n
                      + 2ULL * BLOCK_M * head_dim) * sizeof(float);

    // Causal = true (this training script always uses causal attention)
#define LAUNCH_BWD(D) \
    do { \
        flash_attn_bwd_precompute_D<D><<<grid_D, block_cfg>>>( \
            do_base, o_base, d_base, T); \
        flash_attn_bwd_v2<D, true><<<grid_bwd, block_cfg, shmem_bwd>>>( \
            q_base, k_base, v_base, do_base, \
            l_base, d_base, \
            dq_base, dk_base, dv_base, \
            T, scale); \
    } while (0)

    switch (head_dim) {
        case   8: LAUNCH_BWD(  8); break;
        case  16: LAUNCH_BWD( 16); break;
        case  24: LAUNCH_BWD( 24); break;
        case  32: LAUNCH_BWD( 32); break;
        case  40: LAUNCH_BWD( 40); break;
        case  48: LAUNCH_BWD( 48); break;
        case  56: LAUNCH_BWD( 56); break;
        case  64: LAUNCH_BWD( 64); break;
        case  80: LAUNCH_BWD( 80); break;
        case  96: LAUNCH_BWD( 96); break;
        case 128: LAUNCH_BWD(128); break;
        case 160: LAUNCH_BWD(160); break;
        case 192: LAUNCH_BWD(192); break;
        case 256: LAUNCH_BWD(256); break;
        default:
            printf("Unsupported head_dim %d for backward\n", head_dim);
            break;
    }
#undef LAUNCH_BWD

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error after bwd kernel launch: %s\n", cudaGetErrorString(err));

    // No cudaDeviceSynchronize — downstream kernels on the same stream
    // will implicitly wait for these to complete.
    return {dQ_t, dK_t, dV_t};
}

}

// int main()
// {
//     // ── GPT-2 config from gpt2_attn_fixed.cpp ────────────────────────────────
//     // GPTConfig  (main): n_embd=384, n_heads=6, n_layers=3
//     // Attention::forward (lines 152-154):
//     //   q = transpose(reshape(q, [B, T, n_heads, head_dim]), 1, 2)
//     //   → q, k, v each become [B, n_heads, T, head_dim]
//     //
//     // For testing we mirror that 4-D shape with B=1, T=4 (small for display).
//     // ─────────────────────────────────────────────────────────────────────────
//     const int B        = 8;    // batch        (8 in training)
//     const int n_heads  = 6;    // GPTConfig::n_heads
//     const int n_embd   = 768;  // GPTConfig::n_embd
//     const int head_dim = n_embd / n_heads;  // 64 = dk
//     const int T        = 1024;    // seq length   (1024 in training; 4 here for display)

//     // scale = 1/sqrt(dk), matches Attention constructor line 124-125:
//     //   scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_))
//     const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

//     // std::cout << "=== Flash Attention Kernel Test (Algorithm 1 compliant) ===\n"
//     //           << "  Source      : gpt2_attn_fixed.cpp\n"
//     //           << "  n_embd      : " << n_embd   << "  (GPTConfig)\n"
//     //           << "  n_heads     : " << n_heads  << "  (GPTConfig)\n"
//     //           << "  head_dim    : " << head_dim << "  (dk = n_embd / n_heads)\n"
//     //           << "  scale       : 1/sqrt(" << head_dim << ") = " << scale << "\n"
//     //           << "  Q/K/V shape : [B=" << B
//     //           << ", n_heads=" << n_heads
//     //           << ", T=" << T
//     //           << ", head_dim=" << head_dim << "]"
//     //           << "  ← mirrors lines 152-154 in gpt2_attn_fixed.cpp\n\n";

//     OwnTensor::TensorOptions opts = OwnTensor::TensorOptions()
//         .with_device(Device::CUDA).with_dtype(Dtype::Float32);

//     // 4-D tensors [B, n_heads, T, head_dim] — exact shape from gpt2_attn_fixed.cpp
//     // after reshape([B, T, n_heads, head_dim]) + transpose(1, 2)
//     OwnTensor::Tensor Q = Tensor::randn({{B, n_heads, T, head_dim}}, opts);
//     OwnTensor::Tensor K = Tensor::randn({{B, n_heads, T, head_dim}}, opts);
//     OwnTensor::Tensor V = Tensor::randn({{B, n_heads, T, head_dim}}, opts);

    
//     std::cout << "Q shape [" << B << ", " << n_heads << ", " << T << ", " << head_dim << "]\n";
//     std::cout << "K shape [" << B << ", " << n_heads << ", " << T << ", " << head_dim << "]\n";
//     std::cout << "V shape [" << B << ", " << n_heads << ", " << T << ", " << head_dim << "]\n\n";

//     OwnTensor::Tensor O = dispatch_flash_attention(Q, K, V, B, n_heads, T, head_dim, scale);

//     std::cout << "=== Flash Attention Output  ["
//               << B << ", " << n_heads << ", " << T << ", " << head_dim << "] ===\n";
//     O.display();

   
//     // std::cout << "\n=== Reference: fused_tril_softmax on [T x T] scores (one head) ===\n";
//     // OwnTensor::Tensor scores_h0 = Tensor::randn({{T, T}}, opts);
//     // std::cout << "Score matrix Q_scaled @ K^T  [" << T << " x " << T << "]:\n";
//     // scores_h0.display();
//     // OwnTensor::Tensor attn_w = OwnTensor::fused_tril_softmax(scores_h0, 0, -INFINITY);
//     // std::cout << "Causal attention weights after fused_tril_softmax:\n";
//     // attn_w.display();
// }
