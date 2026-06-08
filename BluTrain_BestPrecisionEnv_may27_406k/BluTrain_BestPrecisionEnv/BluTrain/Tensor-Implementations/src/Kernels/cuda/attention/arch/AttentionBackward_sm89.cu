#include "ops/helpers/AttentionKernels.h"
#include "ops/cuda/attention/AttentionCommon.cuh"

namespace OwnTensor {

// =============================================================================
// Ada Lovelace SM89 Fused Attention Backward — PTX mma.sync.m16n8k8 TF32
// =============================================================================
//
// Replaces WMMA (nvcuda::wmma) with explicit PTX:
//   mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
//
// WMMA sites converted:
//   Phase A  — QK^T and dO·V^T  (NT layout, warps 0–3)
//   Phase B1 — dQ = ds@K        (NN layout, warps 0–3, per-Q-tile)
//   Phase B1 — dK = ds_kd@Q     (NN layout, warps 4–7, persistent across Q loop)
//   Phase B2 — dV = p_kd@dO     (NN layout, warps 4–7, persistent across Q loop)
//
// Each 16×16 WMMA tile → 2 MMA calls (left N-half 0..7, right 8..15).
// A uses ldmatrix.x4; B uses scalar smem reads.
// All smem layouts and scalar post-processing are unchanged.
// =============================================================================

static constexpr int SM89_BWD_WARP_SZ    = 32;
static constexpr int SM89_BWD_NUM_THREADS = 256;   // 8 warps
static constexpr int SM89_BWD_BLOCK_M_D  = 8;

__inline__ __device__ float sm89_bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = SM89_BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ── PTX mma.sync.aligned.m16n8k8 (TF32→f32 accumulate) ──────────────────────
__device__ __forceinline__
void bwd_mma_tf32(float& d0, float& d1, float& d2, float& d3,
                  uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                  uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ── Precompute D = rowsum(dO ⊙ O) ────────────────────────────────────────────
template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D_sm89(MemEfficientBwdParams params)
{
    const float* __restrict__ dO = params.dO;
    const float* __restrict__ O  = params.O;
    float*       __restrict__ D  = params.D;
    const int64_t T              = params.T;
    const int nh                 = params.nh;

    constexpr int LocalN = (HeadDim + SM89_BWD_WARP_SZ - 1) / SM89_BWD_WARP_SZ;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
    const int lane_id = threadIdx.x % SM89_BWD_WARP_SZ;
    const int b       = bh / nh;
    const int h       = bh - b * nh;

    const int base_row = blockIdx.x * (SM89_BWD_BLOCK_M_D * 2) + warp_id * 2;
    const int row0     = base_row;
    const int row1     = base_row + 1;
    const bool v0      = (row0 < T);
    const bool v1      = (row1 < T);

    const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
    const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
    const long long dO_off0 = (long long)row0 * params.do_strideM;
    const long long dO_off1 = (long long)row1 * params.do_strideM;
    const long long O_off0  = (long long)row0 * params.o_strideM;
    const long long O_off1  = (long long)row1 * params.o_strideM;

    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        const int k = lane_id + i * SM89_BWD_WARP_SZ;
        if (k < HeadDim) {
            if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);
            if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);
        }
    }
    sum0 = sm89_bwd_warp_sum(sum0);
    sum1 = sm89_bwd_warp_sum(sum1);
    if (lane_id == 0) {
        float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
        if (v0) D_bh[row0] = sum0;
        if (v1) D_bh[row1] = sum1;
    }
}

// =============================================================================
// exp12: KV-tile-centric backward, TF32 MMA, dK/dV atomic-free
// BM=32, BN=16 — sweet-spot for SM89 (43.8 KB smem, 2 blocks/SM, 16 active warps)
// dK and dV accumulated in persistent register arrays; dQ uses atomicAdd.
// =============================================================================
template <int HeadDim, bool Causal>
__launch_bounds__(256, 2)
__global__ void mem_efficient_bwd_unified_kernel_exp12(MemEfficientBwdParams params)
{
    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 32;
    constexpr int BM_TILES  = BM_WMMA / 16;    // 2 row-groups
    constexpr int HD_CHUNKS = HeadDim / 16;     // e.g. 4 for HD=64
    constexpr int HD_PAD    = HeadDim + 4;
    constexpr int BKN_PAD   = BlockN  + 4;
    constexpr int BM_PAD    = BM_WMMA + 4;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem_f[];
    float* Q_sm    = smem_f;
    float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
    float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
    float* Vs      = Ks      + BlockN   * HD_PAD;
    float* ds_qd   = Vs      + BlockN   * HD_PAD;
    float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;
    float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;
    float* p_kd    = ds_kd   + BlockN   * BM_PAD;
    float* tile_st = p_kd    + BlockN   * BM_PAD;
    float* LSE_sm  = tile_st + BM_WMMA  * HD_PAD;
    float* D_sm    = LSE_sm  + BM_WMMA;

    const int bh           = blockIdx.y;
    const int kv_tile      = blockIdx.x;
    const int kv_base      = kv_tile * BlockN;
    const int kv_tile_size = min(BlockN, params.T - kv_base);
    if (kv_base >= params.T) return;

    const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
    const int lane    = threadIdx.x % SM89_BWD_WARP_SZ;
    const int chunk   = warp_id % HD_CHUNKS;
    const int b       = bh / params.nh;
    const int h       = bh - b * params.nh;

    const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
    const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
    const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
    const float* D_bh   = params.D   + b * params.d_strideB   + h * params.d_strideH;
    float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;
    float*       dK_bh  = params.dK  + b * params.dk_strideB  + h * params.dk_strideH;
    float*       dV_bh  = params.dV  + b * params.dv_strideB  + h * params.dv_strideH;

    const int64_t q_sM  = params.q_strideM;
    const int64_t k_sM  = params.k_strideM;
    const int64_t v_sM  = params.v_strideM;
    const int64_t do_sM = params.do_strideM;
    const int64_t dq_sM = params.dq_strideM;
    const int64_t dk_sM = params.dk_strideM;
    const int64_t dv_sM = params.dv_strideM;

    // ── MMA helpers ───────────────────────────────────────────────────────────
    // loadA: ldmatrix.x4 from row-major smem into 4 TF32 registers.
    //   ptr = smem[(base_row + lane%16)*stride + k8 + (lane/16)*4]
    //
    // loadB_nt: NT B load — B[k,n] = Bs[n*stride+k].
    //   lane: n = n_base+lane>>2, k = k8+lane&3
    //
    // loadB_nn: NN B load — B[k,n] = Bs[k*stride+n].
    //   lane: n = n_base+lane>>2, k = k8+lane&3
    //
    // scatter16x8: write d[4] to 16×8 smem tile at (r_base, c_base).
    // gather16x8:  read  d[4] from 16×8 smem tile at (r_base, c_base).
    //   lane owns rows (lane>>2),(lane>>2)+8 and cols (lane&3)*2,(lane&3)*2+1.

    auto loadA = [&](uint32_t r[4], int m_base, int k8,
                     const float* As, int stride) __attribute__((always_inline)) {
        const int row  = m_base + (lane & 15);
        const int col  = k8 + (lane >> 4) * 4;
        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[row * stride + col]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };

    auto loadB_nt = [&](uint32_t r[2], int n_base, int k8,
                        const float* Bs, int stride) __attribute__((always_inline)) {
        const int n = n_base + (lane >> 2);
        const int k = k8 + (lane & 3);
        r[0] = *(const uint32_t*)(&Bs[n * stride + k]);
        r[1] = *(const uint32_t*)(&Bs[n * stride + k + 4]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    auto loadB_nn = [&](uint32_t r[2], int n_base, int k8,
                        const float* Bs, int stride) __attribute__((always_inline)) {
        const int n = n_base + (lane >> 2);
        const int k = k8 + (lane & 3);
        r[0] = *(const uint32_t*)(&Bs[k       * stride + n]);
        r[1] = *(const uint32_t*)(&Bs[(k + 4) * stride + n]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    auto scatter = [&](const float d[4], float* dst, int r_base, int c_base,
                       int stride) __attribute__((always_inline)) {
        const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
        const int c  = c_base + (lane & 3) * 2;
        dst[r0 * stride + c]     = d[0];
        dst[r0 * stride + c + 1] = d[1];
        dst[r1 * stride + c]     = d[2];
        dst[r1 * stride + c + 1] = d[3];
    };

    // Step 0: load K and V tiles into smem (persistent for entire Q loop)
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = kv_base + r;
        Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
    }
    __syncthreads();

    // Persistent dK/dV accumulators for warps HD_CHUNKS..2*HD_CHUNKS-1.
    // Each covers a 16×8 half of the [BlockN=16 × 16] output chunk.
    // dk_acc[0..3] = left N-half (chunk*16..+7), dk_acc[4..7] = right (chunk*16+8..+15).
    float dk_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float dv_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // Only warps HD_CHUNKS..2*HD_CHUNKS-1 use these; warps 0..HD_CHUNKS-1 ignore them.

    const int q_loop_start = Causal ? kv_base : 0;

    for (int q_base = q_loop_start; q_base < params.T; q_base += BM_WMMA) {
        __syncthreads();  // (a)

        const int q_tile_size = min(BM_WMMA, params.T - q_base);

        // Load Q, dO, LSE, D for this Q tile
        for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
            const int r  = idx / HeadDim, k = idx % HeadDim;
            const int qi = q_base + r;
            const bool vq = (qi < params.T);
            Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
            dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
        }
        if (threadIdx.x < BM_WMMA) {
            const int qi  = q_base + threadIdx.x;
            const bool vq = (qi < params.T);
            LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
            D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
        }
        __syncthreads();  // (b)

        // ── Phase A: warps 0–3 active ──────────────────────────────────────────
        // warps 0–1: S   = Q[32×HD] @ K^T[HD×16] → ds_qd [32×16]  (NT layout)
        // warps 2–3: DPV = dO[32×HD] @ V^T[HD×16] → DPV_sm[32×16] (NT layout)
        // Each warp handles one row-group (16 rows) of the 32-row tile.
        if (warp_id < 2 * BM_TILES) {
            const int  rg       = warp_id % BM_TILES;   // 0 or 1 (row group)
            const bool is_qk    = (warp_id < BM_TILES);
            const float* src_sm = (is_qk ? Q_sm  : dO_sm) + rg * 16 * HD_PAD;
            const float* kv_sm  =  is_qk ? Ks    : Vs;
            float*       dst_sm = (is_qk ? ds_qd : DPV_sm) + rg * 16 * BKN_PAD;

            // Accumulate result for all 16 K-steps (HD=64 → 8 steps of 8).
            // NT B: B[k,n] = kv_sm[n * HD_PAD + k].
            float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
            float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t frA[4], frB_l[2], frB_r[2];

            constexpr int KS_TOTAL = 2 * HD_CHUNKS;
            #pragma unroll
            for (int ks = 0; ks < KS_TOTAL; ks++) {
                const int k8 = ks * 8;
                loadA   (frA,   0, k8, src_sm, HD_PAD);
                // B covers BlockN=16 output N cols: left 0..7, right 8..15
                loadB_nt(frB_l, 0,     k8, kv_sm, HD_PAD);
                loadB_nt(frB_r, 8,     k8, kv_sm, HD_PAD);
                bwd_mma_tf32(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
                             frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                bwd_mma_tf32(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
                             frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
            // Scatter 16×16 result into dst_sm[0..16][0..16] with stride BKN_PAD.
            scatter(acc_l, dst_sm, 0, 0, BKN_PAD);
            scatter(acc_r, dst_sm, 0, 8, BKN_PAD);
        }
        __syncthreads();  // (c1)

        // Scalar post-process: compute p, ds; fill ds_qd (for dQ/dK), ds_kd, p_kd.
        for (int elem = threadIdx.x; elem < BM_WMMA * BlockN; elem += blockDim.x) {
            const int qi_local = elem / BlockN;
            const int j_local  = elem % BlockN;
            const float raw_s  = ds_qd [qi_local * BKN_PAD + j_local];
            const float dpv    = DPV_sm[qi_local * BKN_PAD + j_local];
            const float L      = LSE_sm[qi_local];
            const float D_val  = D_sm  [qi_local];
            const bool qi_ok   = ((q_base + qi_local) < params.T);
            const bool j_ok    = (j_local < kv_tile_size);
            const bool cok     = !Causal || ((kv_base + j_local) <= (q_base + qi_local));
            float p = 0.f;
            if (qi_ok && j_ok && cok)
                p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));
            const float ds = p * (dpv - D_val) * params.scale;
            ds_qd[qi_local * BKN_PAD + j_local] = ds;
            ds_kd[j_local  * BM_PAD  + qi_local] = ds;
            p_kd [j_local  * BM_PAD  + qi_local] = p;
        }
        __syncthreads();  // (c2)

        // ── Phase B1: warps 0–3 → dQ per tile, warps 4–7 → dK (persistent) ───
        //
        // dQ: A = ds_qd[32×16], B = Ks[16×16-chunk]  (NN layout, 2 k-steps of 8)
        //   Each warp (chunk 0..3) handles one 16-column head-dim group of dQ.
        //   Result stored to tile_st[rg*16..+16][chunk*16..+16].
        //
        // dK: A = ds_kd[16×32], B = Q_sm[32×16-chunk] (NN layout, 4 k-steps of 8)
        //   Persistent acc accumulates [BlockN=16 × 16] tile into dk_acc[8].
        {
            uint32_t frA[4], frB_l[2], frB_r[2];

            if (warp_id < HD_CHUNKS) {
                // dQ for this Q-tile: 2 row groups × 1 HD-chunk per warp
                const int n_base = chunk * 16;   // head-dim column base
                float dq_l[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                float dq_r[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

                #pragma unroll
                for (int rg = 0; rg < BM_TILES; rg++) {
                    const float* a_base = ds_qd + rg * 16 * BKN_PAD;
                    // 2 k-steps of 8 (BlockN=16 k-dim)
                    #pragma unroll
                    for (int ks = 0; ks < 2; ks++) {
                        const int k8 = ks * 8;
                        loadA   (frA,   0, k8, a_base, BKN_PAD);
                        loadB_nn(frB_l, n_base,     k8, Ks, HD_PAD);
                        loadB_nn(frB_r, n_base + 8, k8, Ks, HD_PAD);
                        bwd_mma_tf32(dq_l[rg][0], dq_l[rg][1], dq_l[rg][2], dq_l[rg][3],
                                     frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                        bwd_mma_tf32(dq_r[rg][0], dq_r[rg][1], dq_r[rg][2], dq_r[rg][3],
                                     frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                    }
                    scatter(dq_l[rg], tile_st, rg * 16, n_base,     HD_PAD);
                    scatter(dq_r[rg], tile_st, rg * 16, n_base + 8, HD_PAD);
                }
            } else {
                // dK (persistent): A = ds_kd[16×32], B = Q_sm[32×16-chunk]
                // NN layout: B[k,n] = Q_sm[k * HD_PAD + n].
                // 4 k-steps of 8 to cover BM_WMMA=32 k-dim.
                const int n_base = chunk * 16;
                #pragma unroll
                for (int ks = 0; ks < BM_TILES * 2; ks++) {
                    const int k8 = ks * 8;
                    loadA   (frA,   0, k8, ds_kd, BM_PAD);
                    loadB_nn(frB_l, n_base,     k8, Q_sm, HD_PAD);
                    loadB_nn(frB_r, n_base + 8, k8, Q_sm, HD_PAD);
                    bwd_mma_tf32(dk_acc[0], dk_acc[1], dk_acc[2], dk_acc[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                    bwd_mma_tf32(dk_acc[4], dk_acc[5], dk_acc[6], dk_acc[7],
                                 frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                }
            }
        }
        __syncthreads();  // (d): tile_st has dQ for this Q-tile

        // All 256 threads atomicAdd tile_st → global dQ
        for (int idx = threadIdx.x; idx < q_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dQ_bh[(q_base + r) * dq_sM + k], tile_st[r * HD_PAD + k]);
        }

        // ── Phase B2: warps 4–7 → dV (persistent) ────────────────────────────
        // A = p_kd[16×32], B = dO_sm[32×16-chunk] (NN layout, 4 k-steps).
        // Accumulates into dv_acc[8] across Q-tile loop.
        if (warp_id >= HD_CHUNKS) {
            uint32_t frA[4], frB_l[2], frB_r[2];
            const int n_base = chunk * 16;
            #pragma unroll
            for (int ks = 0; ks < BM_TILES * 2; ks++) {
                const int k8 = ks * 8;
                loadA   (frA,   0, k8, p_kd, BM_PAD);
                loadB_nn(frB_l, n_base,     k8, dO_sm, HD_PAD);
                loadB_nn(frB_r, n_base + 8, k8, dO_sm, HD_PAD);
                bwd_mma_tf32(dv_acc[0], dv_acc[1], dv_acc[2], dv_acc[3],
                             frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                bwd_mma_tf32(dv_acc[4], dv_acc[5], dv_acc[6], dv_acc[7],
                             frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
        }
        // sync(a) at top of next iteration covers dO_sm and atomicAdd reads.
    } // end Q-tile loop

    // ── Final: scatter dk_acc/dv_acc → tile_st → global dK/dV ───────────────
    __syncthreads();

    // Warps HD_CHUNKS..2*HD_CHUNKS-1 own the dK and dV accumulators.
    // Scatter each 8-register 16×16 block into tile_st[0..16][chunk*16..+16].
    if (warp_id >= HD_CHUNKS) {
        const int n_base = chunk * 16;
        scatter(dk_acc,     tile_st, 0, n_base,     HD_PAD);
        scatter(dk_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(kv_base + r) * dk_sM + k] = tile_st[r * HD_PAD + k];
    }
    __syncthreads();

    if (warp_id >= HD_CHUNKS) {
        const int n_base = chunk * 16;
        scatter(dv_acc,     tile_st, 0, n_base,     HD_PAD);
        scatter(dv_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dV_bh[(kv_base + r) * dv_sM + k] = tile_st[r * HD_PAD + k];
    }
}

// =============================================================================
// SM89 public entry point
// =============================================================================
void mem_efficient_attn_backward_sm89_cuda(
    const float* query,       int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,         int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value,       int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    const float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
    const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
    float* grad_query,        int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
    float* grad_key,          int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
    float* grad_value,        int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
    float* D_buf,             int64_t d_strideB, int64_t d_strideH,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    bool skip_grad_zero)
{
    float scale = 1.0f / sqrtf(static_cast<float>(hd));
    const int BH = (int)(B * nh);
    dim3 block_cfg(SM89_BWD_NUM_THREADS);
    dim3 grid_D(((int)T + (SM89_BWD_BLOCK_M_D * 2) - 1) / (SM89_BWD_BLOCK_M_D * 2), BH);

    MemEfficientBwdParams params;
    params.Q        = query;
    params.K        = key;
    params.V        = value;
    params.O        = output;
    params.dO       = grad_output;
    params.LSE      = lse;
    params.D        = D_buf;
    params.dQ       = grad_query;
    params.dK       = grad_key;
    params.dV       = grad_value;
    params.B        = (int)B;
    params.nh       = (int)nh;
    params.T        = (int)T;
    params.scale    = scale;
    params.is_causal = is_causal;
    params.q_strideB  = q_strideB;  params.q_strideM  = q_strideM;  params.q_strideH  = q_strideH;
    params.k_strideB  = k_strideB;  params.k_strideM  = k_strideM;  params.k_strideH  = k_strideH;
    params.v_strideB  = v_strideB;  params.v_strideM  = v_strideM;  params.v_strideH  = v_strideH;
    params.o_strideB  = o_strideB;  params.o_strideM  = o_strideM;  params.o_strideH  = o_strideH;
    params.do_strideB = do_strideB; params.do_strideM = do_strideM; params.do_strideH = do_strideH;
    params.dq_strideB = dq_strideB; params.dq_strideM = dq_strideM; params.dq_strideH = dq_strideH;
    params.dk_strideB = dk_strideB; params.dk_strideM = dk_strideM; params.dk_strideH = dk_strideH;
    params.dv_strideB = dv_strideB; params.dv_strideM = dv_strideM; params.dv_strideH = dv_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
    params.d_strideB   = d_strideB;   params.d_strideH   = d_strideH;

#define LAUNCH_MEM_BWD_EXP12(HD) \
    do { \
        constexpr int BN12 = 16, BM12 = 32; \
        const size_t shmem12 = \
            (2ULL * BM12 * ((HD) + 4)   \
           + 2ULL * BN12 * ((HD) + 4)   \
           + 2ULL * BM12 * ((BN12) + 4) \
           + 2ULL * BN12 * ((BM12) + 4) \
           + 1ULL * BM12 * ((HD) + 4)   \
           + 2ULL * BM12                \
            ) * sizeof(float); \
        const int kv12 = ((int)T + BN12 - 1) / BN12; \
        dim3 grid_kv12(kv12, BH); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp12<HD, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp12<HD, true>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
        if (!skip_grad_zero) { \
            cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
        } \
        mem_efficient_bwd_precompute_D_sm89<HD><<<grid_D, block_cfg>>>(params); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp12<HD, true> \
                <<<grid_kv12, block_cfg, shmem12>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp12<HD, false> \
                <<<grid_kv12, block_cfg, shmem12>>>(params); \
        } \
    } while (0)

    switch ((int)hd) {
        case  16: LAUNCH_MEM_BWD_EXP12( 16); break;
        case  32: LAUNCH_MEM_BWD_EXP12( 32); break;
        case  48: LAUNCH_MEM_BWD_EXP12( 48); break;
        case  64: LAUNCH_MEM_BWD_EXP12( 64); break;
        case  80: LAUNCH_MEM_BWD_EXP12( 80); break;
        case  96: LAUNCH_MEM_BWD_EXP12( 96); break;
        case 128: LAUNCH_MEM_BWD_EXP12(128); break;
        case 160: LAUNCH_MEM_BWD_EXP12(160); break;
        case 192: LAUNCH_MEM_BWD_EXP12(192); break;
        case 256: LAUNCH_MEM_BWD_EXP12(256); break;
        default:
            printf("mem_efficient_attn_backward_sm89: unsupported head_dim %d\n", (int)hd);
            break;
    }
#undef LAUNCH_MEM_BWD_EXP12
}

} // namespace OwnTensor





































// #include "ops/helpers/AttentionKernels.h"
// #include "ops/cuda/attention/AttentionCommon.cuh"

// namespace OwnTensor {

// // =============================================================================
// // Ada Lovelace SM89 Fused Attention Backward — PTX mma.sync.m16n8k8 TF32
// // =============================================================================
// //
// // Replaces WMMA (nvcuda::wmma) with explicit PTX:
// //   mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
// //
// // WMMA sites converted:
// //   Phase A  — QK^T and dO·V^T  (NT layout, warps 0–3)
// //   Phase B1 — dQ = ds@K        (NN layout, warps 0–3, per-Q-tile)
// //   Phase B1 — dK = ds_kd@Q     (NN layout, warps 4–7, persistent across Q loop)
// //   Phase B2 — dV = p_kd@dO     (NN layout, warps 4–7, persistent across Q loop)
// //
// // Each 16×16 WMMA tile → 2 MMA calls (left N-half 0..7, right 8..15).
// // A uses ldmatrix.x4; B uses scalar smem reads.
// // All smem layouts and scalar post-processing are unchanged.
// // =============================================================================

// static constexpr int SM89_BWD_WARP_SZ    = 32;
// static constexpr int SM89_BWD_NUM_THREADS = 256;   // 8 warps
// static constexpr int SM89_BWD_BLOCK_M_D  = 8;

// __inline__ __device__ float sm89_bwd_warp_sum(float val) {
//     #pragma unroll
//     for (int offset = SM89_BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
//         val += __shfl_xor_sync(0xffffffff, val, offset);
//     return val;
// }

// // ── PTX mma.sync.aligned.m16n8k8 (TF32→f32 accumulate) ──────────────────────
// __device__ __forceinline__
// void bwd_mma_tf32(float& d0, float& d1, float& d2, float& d3,
//                   uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
//                   uint32_t b0, uint32_t b1)
// {
//     asm volatile(
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
//         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
//         : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
//         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
// }

// // ── Precompute D = rowsum(dO ⊙ O) ────────────────────────────────────────────
// template<int HeadDim>
// __global__ void mem_efficient_bwd_precompute_D_sm89(MemEfficientBwdParams params)
// {
//     const float* __restrict__ dO = params.dO;
//     const float* __restrict__ O  = params.O;
//     float*       __restrict__ D  = params.D;
//     const int64_t T              = params.T;
//     const int nh                 = params.nh;

//     constexpr int LocalN = (HeadDim + SM89_BWD_WARP_SZ - 1) / SM89_BWD_WARP_SZ;

//     const int bh      = blockIdx.y;
//     const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
//     const int lane_id = threadIdx.x % SM89_BWD_WARP_SZ;
//     const int b       = bh / nh;
//     const int h       = bh - b * nh;

//     const int base_row = blockIdx.x * (SM89_BWD_BLOCK_M_D * 2) + warp_id * 2;
//     const int row0     = base_row;
//     const int row1     = base_row + 1;
//     const bool v0      = (row0 < T);
//     const bool v1      = (row1 < T);

//     const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
//     const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
//     const long long dO_off0 = (long long)row0 * params.do_strideM;
//     const long long dO_off1 = (long long)row1 * params.do_strideM;
//     const long long O_off0  = (long long)row0 * params.o_strideM;
//     const long long O_off1  = (long long)row1 * params.o_strideM;

//     float sum0 = 0.f, sum1 = 0.f;
//     #pragma unroll
//     for (int i = 0; i < LocalN; ++i) {
//         const int k = lane_id + i * SM89_BWD_WARP_SZ;
//         if (k < HeadDim) {
//             if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);
//             if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);
//         }
//     }
//     sum0 = sm89_bwd_warp_sum(sum0);
//     sum1 = sm89_bwd_warp_sum(sum1);
//     if (lane_id == 0) {
//         float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
//         if (v0) D_bh[row0] = sum0;
//         if (v1) D_bh[row1] = sum1;
//     }
// }

// // =============================================================================
// // exp12: KV-tile-centric backward, TF32 MMA, dK/dV atomic-free
// // BM=32, BN=16 — sweet-spot for SM89 (43.8 KB smem, 2 blocks/SM, 16 active warps)
// // dK and dV accumulated in persistent register arrays; dQ uses atomicAdd.
// // =============================================================================
// template <int HeadDim, bool Causal>
// __launch_bounds__(256, 2)
// __global__ void mem_efficient_bwd_unified_kernel_exp12(MemEfficientBwdParams params)
// {
//     constexpr int BlockN    = 16;
//     constexpr int BM_WMMA   = 32;
//     constexpr int BM_TILES  = BM_WMMA / 16;    // 2 row-groups
//     constexpr int HD_CHUNKS = HeadDim / 16;     // e.g. 4 for HD=64
//     constexpr int HD_PAD    = HeadDim + 4;
//     constexpr int BKN_PAD   = BlockN  + 4;
//     constexpr int BM_PAD    = BM_WMMA + 4;
//     constexpr float BWD_LOG2E = 1.4426950408889634074f;

//     extern __shared__ float smem_f[];
//     float* Q_sm    = smem_f;
//     float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
//     float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
//     float* Vs      = Ks      + BlockN   * HD_PAD;
//     float* ds_qd   = Vs      + BlockN   * HD_PAD;
//     float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;
//     float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;
//     float* p_kd    = ds_kd   + BlockN   * BM_PAD;
//     float* tile_st = p_kd    + BlockN   * BM_PAD;
//     float* LSE_sm  = tile_st + BM_WMMA  * HD_PAD;
//     float* D_sm    = LSE_sm  + BM_WMMA;

//     const int bh           = blockIdx.y;
//     const int kv_tile      = blockIdx.x;
//     const int kv_base      = kv_tile * BlockN;
//     const int kv_tile_size = min(BlockN, params.T - kv_base);
//     if (kv_base >= params.T) return;

//     const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
//     const int lane    = threadIdx.x % SM89_BWD_WARP_SZ;
//     const int chunk   = warp_id % HD_CHUNKS;
//     const int b       = bh / params.nh;
//     const int h       = bh - b * params.nh;

//     const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
//     const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
//     const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
//     const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
//     const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
//     const float* D_bh   = params.D   + b * params.d_strideB   + h * params.d_strideH;
//     float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;
//     float*       dK_bh  = params.dK  + b * params.dk_strideB  + h * params.dk_strideH;
//     float*       dV_bh  = params.dV  + b * params.dv_strideB  + h * params.dv_strideH;

//     const int64_t q_sM  = params.q_strideM;
//     const int64_t k_sM  = params.k_strideM;
//     const int64_t v_sM  = params.v_strideM;
//     const int64_t do_sM = params.do_strideM;
//     const int64_t dq_sM = params.dq_strideM;
//     const int64_t dk_sM = params.dk_strideM;
//     const int64_t dv_sM = params.dv_strideM;

//     // ── MMA helpers ───────────────────────────────────────────────────────────
//     // loadA: ldmatrix.x4 from row-major smem into 4 TF32 registers.
//     //   ptr = smem[(base_row + lane%16)*stride + k8 + (lane/16)*4]
//     //
//     // loadB_nt: NT B load — B[k,n] = Bs[n*stride+k].
//     //   lane: n = n_base+lane>>2, k = k8+lane&3
//     //
//     // loadB_nn: NN B load — B[k,n] = Bs[k*stride+n].
//     //   lane: n = n_base+lane>>2, k = k8+lane&3
//     //
//     // scatter16x8: write d[4] to 16×8 smem tile at (r_base, c_base).
//     // gather16x8:  read  d[4] from 16×8 smem tile at (r_base, c_base).
//     //   lane owns rows (lane>>2),(lane>>2)+8 and cols (lane&3)*2,(lane&3)*2+1.

//     auto loadA = [&](uint32_t r[4], int m_base, int k8,
//                      const float* As, int stride) __attribute__((always_inline)) {
//         const int row  = m_base + (lane & 15);
//         const int col  = k8 + (lane >> 4) * 4;
//         const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[row * stride + col]);
//         asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
//             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
//         r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
//     };

//     auto loadB_nt = [&](uint32_t r[2], int n_base, int k8,
//                         const float* Bs, int stride) __attribute__((always_inline)) {
//         const int n = n_base + (lane >> 2);
//         const int k = k8 + (lane & 3);
//         r[0] = *(const uint32_t*)(&Bs[n * stride + k]);
//         r[1] = *(const uint32_t*)(&Bs[n * stride + k + 4]);
//         r[0] += 0x1000u; r[1] += 0x1000u;
//     };

//     auto loadB_nn = [&](uint32_t r[2], int n_base, int k8,
//                         const float* Bs, int stride) __attribute__((always_inline)) {
//         const int n = n_base + (lane >> 2);
//         const int k = k8 + (lane & 3);
//         r[0] = *(const uint32_t*)(&Bs[k       * stride + n]);
//         r[1] = *(const uint32_t*)(&Bs[(k + 4) * stride + n]);
//         r[0] += 0x1000u; r[1] += 0x1000u;
//     };

//     auto scatter = [&](const float d[4], float* dst, int r_base, int c_base,
//                        int stride) __attribute__((always_inline)) {
//         const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
//         const int c  = c_base + (lane & 3) * 2;
//         dst[r0 * stride + c]     = d[0];
//         dst[r0 * stride + c + 1] = d[1];
//         dst[r1 * stride + c]     = d[2];
//         dst[r1 * stride + c + 1] = d[3];
//     };

//     // Step 0: load K and V tiles into smem (persistent for entire Q loop)
//     for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         const int g = kv_base + r;
//         Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
//         Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
//     }
//     __syncthreads();

//     // Persistent dK/dV accumulators for warps HD_CHUNKS..2*HD_CHUNKS-1.
//     // Each covers a 16×8 half of the [BlockN=16 × 16] output chunk.
//     // dk_acc[0..3] = left N-half (chunk*16..+7), dk_acc[4..7] = right (chunk*16+8..+15).
//     float dk_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
//     float dv_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
//     // Only warps HD_CHUNKS..2*HD_CHUNKS-1 use these; warps 0..HD_CHUNKS-1 ignore them.

//     const int q_loop_start = Causal ? kv_base : 0;

//     for (int q_base = q_loop_start; q_base < params.T; q_base += BM_WMMA) {
//         __syncthreads();  // (a)

//         const int q_tile_size = min(BM_WMMA, params.T - q_base);

//         // Load Q, dO, LSE, D for this Q tile
//         for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
//             const int r  = idx / HeadDim, k = idx % HeadDim;
//             const int qi = q_base + r;
//             const bool vq = (qi < params.T);
//             Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
//             dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
//         }
//         if (threadIdx.x < BM_WMMA) {
//             const int qi  = q_base + threadIdx.x;
//             const bool vq = (qi < params.T);
//             LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
//             D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
//         }
//         __syncthreads();  // (b)

//         // ── Phase A: warps 0–3 active ──────────────────────────────────────────
//         // warps 0–1: S   = Q[32×HD] @ K^T[HD×16] → ds_qd [32×16]  (NT layout)
//         // warps 2–3: DPV = dO[32×HD] @ V^T[HD×16] → DPV_sm[32×16] (NT layout)
//         // Each warp handles one row-group (16 rows) of the 32-row tile.
//         if (warp_id < 2 * BM_TILES) {
//             const int  rg       = warp_id % BM_TILES;   // 0 or 1 (row group)
//             const bool is_qk    = (warp_id < BM_TILES);
//             const float* src_sm = (is_qk ? Q_sm  : dO_sm) + rg * 16 * HD_PAD;
//             const float* kv_sm  =  is_qk ? Ks    : Vs;
//             float*       dst_sm = (is_qk ? ds_qd : DPV_sm) + rg * 16 * BKN_PAD;

//             // Accumulate result for all 16 K-steps (HD=64 → 8 steps of 8).
//             // NT B: B[k,n] = kv_sm[n * HD_PAD + k].
//             float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
//             float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
//             uint32_t frA[4], frB_l[2], frB_r[2];

//             constexpr int KS_TOTAL = 2 * HD_CHUNKS;
//             #pragma unroll
//             for (int ks = 0; ks < KS_TOTAL; ks++) {
//                 const int k8 = ks * 8;
//                 loadA   (frA,   0, k8, src_sm, HD_PAD);
//                 // B covers BlockN=16 output N cols: left 0..7, right 8..15
//                 loadB_nt(frB_l, 0,     k8, kv_sm, HD_PAD);
//                 loadB_nt(frB_r, 8,     k8, kv_sm, HD_PAD);
//                 bwd_mma_tf32(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
//                              frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
//                 bwd_mma_tf32(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
//                              frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
//             }
//             // Scatter 16×16 result into dst_sm[0..16][0..16] with stride BKN_PAD.
//             scatter(acc_l, dst_sm, 0, 0, BKN_PAD);
//             scatter(acc_r, dst_sm, 0, 8, BKN_PAD);
//         }
//         __syncthreads();  // (c1)

//         // Scalar post-process: compute p, ds; fill ds_qd (for dQ/dK), ds_kd, p_kd.
//         for (int elem = threadIdx.x; elem < BM_WMMA * BlockN; elem += blockDim.x) {
//             const int qi_local = elem / BlockN;
//             const int j_local  = elem % BlockN;
//             const float raw_s  = ds_qd [qi_local * BKN_PAD + j_local];
//             const float dpv    = DPV_sm[qi_local * BKN_PAD + j_local];
//             const float L      = LSE_sm[qi_local];
//             const float D_val  = D_sm  [qi_local];
//             const bool qi_ok   = ((q_base + qi_local) < params.T);
//             const bool j_ok    = (j_local < kv_tile_size);
//             const bool cok     = !Causal || ((kv_base + j_local) <= (q_base + qi_local));
//             float p = 0.f;
//             if (qi_ok && j_ok && cok)
//                 p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));
//             const float ds = p * (dpv - D_val) * params.scale;
//             ds_qd[qi_local * BKN_PAD + j_local] = ds;
//             ds_kd[j_local  * BM_PAD  + qi_local] = ds;
//             p_kd [j_local  * BM_PAD  + qi_local] = p;
//         }
//         __syncthreads();  // (c2)

//         // ── Phase B1: warps 4–7 → dK (persistent) ────────────────────────────
//         // dQ removed from this kernel — handled by mem_efficient_bwd_qtile_dq_sm89
//         // which is Q-tile-centric and writes dQ with direct assignment (no atomicAdd).
//         //
//         // dK: A = ds_kd[16×32], B = Q_sm[32×16-chunk] (NN layout, 4 k-steps of 8)
//         //   Persistent acc accumulates [BlockN=16 × 16] tile into dk_acc[8].
//         if (warp_id >= HD_CHUNKS) {
//             uint32_t frA[4], frB_l[2], frB_r[2];
//             const int n_base = chunk * 16;
//             #pragma unroll
//             for (int ks = 0; ks < BM_TILES * 2; ks++) {
//                 const int k8 = ks * 8;
//                 loadA   (frA,   0, k8, ds_kd, BM_PAD);
//                 loadB_nn(frB_l, n_base,     k8, Q_sm, HD_PAD);
//                 loadB_nn(frB_r, n_base + 8, k8, Q_sm, HD_PAD);
//                 bwd_mma_tf32(dk_acc[0], dk_acc[1], dk_acc[2], dk_acc[3],
//                              frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
//                 bwd_mma_tf32(dk_acc[4], dk_acc[5], dk_acc[6], dk_acc[7],
//                              frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
//             }
//         }
//         __syncthreads();  // (d)

//         // ── Phase B2: warps 4–7 → dV (persistent) ────────────────────────────
//         // A = p_kd[16×32], B = dO_sm[32×16-chunk] (NN layout, 4 k-steps).
//         // Accumulates into dv_acc[8] across Q-tile loop.
//         if (warp_id >= HD_CHUNKS) {
//             uint32_t frA[4], frB_l[2], frB_r[2];
//             const int n_base = chunk * 16;
//             #pragma unroll
//             for (int ks = 0; ks < BM_TILES * 2; ks++) {
//                 const int k8 = ks * 8;
//                 loadA   (frA,   0, k8, p_kd, BM_PAD);
//                 loadB_nn(frB_l, n_base,     k8, dO_sm, HD_PAD);
//                 loadB_nn(frB_r, n_base + 8, k8, dO_sm, HD_PAD);
//                 bwd_mma_tf32(dv_acc[0], dv_acc[1], dv_acc[2], dv_acc[3],
//                              frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
//                 bwd_mma_tf32(dv_acc[4], dv_acc[5], dv_acc[6], dv_acc[7],
//                              frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
//             }
//         }
//         // sync(a) at top of next iteration covers dO_sm and atomicAdd reads.
//     } // end Q-tile loop

//     // ── Final: scatter dk_acc/dv_acc → tile_st → global dK/dV ───────────────
//     __syncthreads();

//     // Warps HD_CHUNKS..2*HD_CHUNKS-1 own the dK and dV accumulators.
//     // Scatter each 8-register 16×16 block into tile_st[0..16][chunk*16..+16].
//     if (warp_id >= HD_CHUNKS) {
//         const int n_base = chunk * 16;
//         scatter(dk_acc,     tile_st, 0, n_base,     HD_PAD);
//         scatter(dk_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
//     }
//     __syncthreads();

//     for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         dK_bh[(kv_base + r) * dk_sM + k] = tile_st[r * HD_PAD + k];
//     }
//     __syncthreads();

//     if (warp_id >= HD_CHUNKS) {
//         const int n_base = chunk * 16;
//         scatter(dv_acc,     tile_st, 0, n_base,     HD_PAD);
//         scatter(dv_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
//     }
//     __syncthreads();

//     for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         dV_bh[(kv_base + r) * dv_sM + k] = tile_st[r * HD_PAD + k];
//     }
// }

// // =============================================================================
// // Q-tile-centric dQ kernel — no atomicAdd.
// //
// // Complements mem_efficient_bwd_unified_kernel_exp12 (which writes dK, dV).
// // Each block owns one Q-tile → accumulates dQ in persistent registers across
// // all KV tiles → writes dQ once with direct assignment (no competing writes).
// //
// // Grid: dim3(q_tiles, BH)  where q_tiles = ceil(T / BM_WMMA)
// // BM=32 (Q rows per block), BN=16 (KV tile width), same PTX/smem style as exp12.
// //
// // Smem layout (no ds_kd / p_kd needed since we only compute dQ):
// //   Q_sm   [BM × HD_PAD]   — persistent Q tile
// //   dO_sm  [BM × HD_PAD]   — persistent dO tile
// //   K_sm   [BN × HD_PAD]   — reloaded each KV iteration
// //   V_sm   [BN × HD_PAD]   — reloaded each KV iteration (for DPV = dO @ V^T)
// //   ds_sm  [BM × BKN_PAD]  — QK^T → ds after scalar post-process
// //   DPV_sm [BM × BKN_PAD]  — dO @ V^T (intermediate)
// //   LSE_sm [BM]             — persistent log-sum-exp
// //   D_sm   [BM]             — persistent rowsum(dO ⊙ O)
// //   tile_st[BM × HD_PAD]   — final dQ staging (scatter dq_acc → here → global)
// // =============================================================================
// template <int HeadDim, bool Causal>
// __launch_bounds__(256, 2)
// __global__ void mem_efficient_bwd_qtile_dq_sm89(MemEfficientBwdParams params)
// {
//     constexpr int BlockN    = 16;
//     constexpr int BM_WMMA   = 32;
//     constexpr int BM_TILES  = BM_WMMA / 16;
//     constexpr int HD_CHUNKS = HeadDim / 16;
//     constexpr int HD_PAD    = HeadDim + 4;
//     constexpr int BKN_PAD   = BlockN  + 4;
//     constexpr float BWD_LOG2E = 1.4426950408889634074f;

//     extern __shared__ float smem_f[];
//     float* Q_sm    = smem_f;
//     float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
//     float* K_sm    = dO_sm   + BM_WMMA * HD_PAD;
//     float* V_sm    = K_sm    + BlockN  * HD_PAD;
//     float* ds_sm   = V_sm    + BlockN  * HD_PAD;   // QK^T result, then ds
//     float* DPV_sm  = ds_sm   + BM_WMMA * BKN_PAD;  // dO·V^T
//     float* LSE_sm  = DPV_sm  + BM_WMMA * BKN_PAD;
//     float* D_sm    = LSE_sm  + BM_WMMA;
//     float* tile_st = D_sm    + BM_WMMA;             // dQ staging [BM × HD_PAD]

//     const int bh     = blockIdx.y;
//     const int q_tile = blockIdx.x;
//     const int q_base = q_tile * BM_WMMA;
//     if (q_base >= params.T) return;

//     const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
//     const int lane    = threadIdx.x % SM89_BWD_WARP_SZ;
//     const int chunk   = warp_id % HD_CHUNKS;
//     const int b       = bh / params.nh;
//     const int h       = bh - b * params.nh;

//     const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
//     const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
//     const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
//     const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
//     const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
//     const float* D_bh   = params.D   + b * params.d_strideB   + h * params.d_strideH;
//     float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;

//     const int64_t q_sM  = params.q_strideM;
//     const int64_t k_sM  = params.k_strideM;
//     const int64_t v_sM  = params.v_strideM;
//     const int64_t do_sM = params.do_strideM;
//     const int64_t dq_sM = params.dq_strideM;

//     // Same PTX helpers as the parent kernel.
//     auto loadA = [&](uint32_t r[4], int m_base, int k8,
//                      const float* As, int stride) __attribute__((always_inline)) {
//         const int row  = m_base + (lane & 15);
//         const int col  = k8 + (lane >> 4) * 4;
//         const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[row * stride + col]);
//         asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
//             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
//         r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
//     };

//     auto loadB_nt = [&](uint32_t r[2], int n_base, int k8,
//                         const float* Bs, int stride) __attribute__((always_inline)) {
//         const int n = n_base + (lane >> 2);
//         const int k = k8 + (lane & 3);
//         r[0] = *(const uint32_t*)(&Bs[n * stride + k]);
//         r[1] = *(const uint32_t*)(&Bs[n * stride + k + 4]);
//         r[0] += 0x1000u; r[1] += 0x1000u;
//     };

//     auto loadB_nn = [&](uint32_t r[2], int n_base, int k8,
//                         const float* Bs, int stride) __attribute__((always_inline)) {
//         const int n = n_base + (lane >> 2);
//         const int k = k8 + (lane & 3);
//         r[0] = *(const uint32_t*)(&Bs[k       * stride + n]);
//         r[1] = *(const uint32_t*)(&Bs[(k + 4) * stride + n]);
//         r[0] += 0x1000u; r[1] += 0x1000u;
//     };

//     auto scatter = [&](const float d[4], float* dst, int r_base, int c_base,
//                        int stride) __attribute__((always_inline)) {
//         const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
//         const int c  = c_base + (lane & 3) * 2;
//         dst[r0 * stride + c]     = d[0];
//         dst[r0 * stride + c + 1] = d[1];
//         dst[r1 * stride + c]     = d[2];
//         dst[r1 * stride + c + 1] = d[3];
//     };

//     // Load Q, dO, LSE, D for this Q tile — persistent across the entire KV loop.
//     for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         const int qi = q_base + r;
//         const bool vq = (qi < params.T);
//         Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
//         dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
//     }
//     if (threadIdx.x < BM_WMMA) {
//         const int qi  = q_base + threadIdx.x;
//         const bool vq = (qi < params.T);
//         LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
//         D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
//     }
//     __syncthreads();

//     // Persistent dQ accumulators in registers (warps HD_CHUNKS..2*HD_CHUNKS-1).
//     // Two row groups (BM_TILES=2) × left/right N-halves × 4 MMA output regs = 16 regs.
//     float dq_acc_l[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
//     float dq_acc_r[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};

//     const int kv_tiles = (params.T + BlockN - 1) / BlockN;
//     // For causal: only KV tiles where kv_base <= q_base + BM_WMMA - 1 contribute.
//     const int kv_end = Causal ? min(kv_tiles, (q_base + BM_WMMA + BlockN - 1) / BlockN)
//                                : kv_tiles;

//     for (int kv_base = 0; kv_base < kv_end * BlockN; kv_base += BlockN) {
//         __syncthreads();

//         const int kv_tile_size = min(BlockN, params.T - kv_base);

//         // Load K and V for this KV tile (reloaded each iteration).
//         for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             const int g = kv_base + r;
//             K_sm[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
//             V_sm[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
//         }
//         __syncthreads();

//         // ── Phase A: warps 0–3 ────────────────────────────────────────────────
//         // warps 0–1: S   = Q_sm  @ K_sm^T → ds_sm  [BM×BN] (NT layout)
//         // warps 2–3: DPV = dO_sm @ V_sm^T → DPV_sm [BM×BN] (NT layout)
//         if (warp_id < 2 * BM_TILES) {
//             const int  rg       = warp_id % BM_TILES;
//             const bool is_qk    = (warp_id < BM_TILES);
//             const float* src_sm = (is_qk ? Q_sm  : dO_sm) + rg * 16 * HD_PAD;
//             const float* kv_sm  =  is_qk ? K_sm  : V_sm;
//             float*       dst_sm = (is_qk ? ds_sm : DPV_sm) + rg * 16 * BKN_PAD;

//             float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
//             float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
//             uint32_t frA[4], frB_l[2], frB_r[2];

//             constexpr int KS_TOTAL = 2 * HD_CHUNKS;
//             #pragma unroll
//             for (int ks = 0; ks < KS_TOTAL; ks++) {
//                 const int k8 = ks * 8;
//                 loadA   (frA,   0, k8, src_sm, HD_PAD);
//                 loadB_nt(frB_l, 0,     k8, kv_sm, HD_PAD);
//                 loadB_nt(frB_r, 8,     k8, kv_sm, HD_PAD);
//                 bwd_mma_tf32(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
//                              frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
//                 bwd_mma_tf32(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
//                              frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
//             }
//             scatter(acc_l, dst_sm, 0, 0, BKN_PAD);
//             scatter(acc_r, dst_sm, 0, 8, BKN_PAD);
//         }
//         __syncthreads();

//         // Scalar post-process: ds[i,j] = p[i,j] * (DPV[i,j] - D[i]) * scale
//         for (int elem = threadIdx.x; elem < BM_WMMA * BlockN; elem += blockDim.x) {
//             const int qi_local = elem / BlockN;
//             const int j_local  = elem % BlockN;
//             const float raw_s  = ds_sm [qi_local * BKN_PAD + j_local];
//             const float dpv    = DPV_sm[qi_local * BKN_PAD + j_local];
//             const float L      = LSE_sm[qi_local];
//             const float D_val  = D_sm  [qi_local];
//             const bool qi_ok   = ((q_base + qi_local) < params.T);
//             const bool j_ok    = (j_local < kv_tile_size);
//             const bool cok     = !Causal || ((kv_base + j_local) <= (q_base + qi_local));
//             float p = 0.f;
//             if (qi_ok && j_ok && cok)
//                 p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));
//             ds_sm[qi_local * BKN_PAD + j_local] = p * (dpv - D_val) * params.scale;
//         }
//         __syncthreads();

//         // ── Phase B: warps 4–7 accumulate dQ persistently across KV tiles ────
//         // dQ[BM×HD] += ds[BM×BN] @ K[BN×HD]  (NN layout, 2 k-steps of 8)
//         // Each warp owns one HD chunk (16 cols) and both row groups.
//         if (warp_id >= HD_CHUNKS) {
//             uint32_t frA[4], frB_l[2], frB_r[2];
//             const int n_base = chunk * 16;
//             #pragma unroll
//             for (int rg = 0; rg < BM_TILES; rg++) {
//                 const float* a_base = ds_sm + rg * 16 * BKN_PAD;
//                 #pragma unroll
//                 for (int ks = 0; ks < 2; ks++) {
//                     const int k8 = ks * 8;
//                     loadA   (frA,   0, k8, a_base, BKN_PAD);
//                     loadB_nn(frB_l, n_base,     k8, K_sm, HD_PAD);
//                     loadB_nn(frB_r, n_base + 8, k8, K_sm, HD_PAD);
//                     bwd_mma_tf32(dq_acc_l[rg][0], dq_acc_l[rg][1],
//                                  dq_acc_l[rg][2], dq_acc_l[rg][3],
//                                  frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
//                     bwd_mma_tf32(dq_acc_r[rg][0], dq_acc_r[rg][1],
//                                  dq_acc_r[rg][2], dq_acc_r[rg][3],
//                                  frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
//                 }
//             }
//         }
//     } // end KV-tile loop

//     // ── Write dQ: scatter dq_acc → tile_st → global dQ (direct assignment) ──
//     __syncthreads();
//     if (warp_id >= HD_CHUNKS) {
//         const int n_base = chunk * 16;
//         #pragma unroll
//         for (int rg = 0; rg < BM_TILES; rg++) {
//             scatter(dq_acc_l[rg], tile_st, rg * 16, n_base,     HD_PAD);
//             scatter(dq_acc_r[rg], tile_st, rg * 16, n_base + 8, HD_PAD);
//         }
//     }
//     __syncthreads();

//     const int q_tile_size = min(BM_WMMA, params.T - q_base);
//     for (int idx = threadIdx.x; idx < q_tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         // Direct assignment — this block is the sole owner of these Q rows.
//         dQ_bh[(q_base + r) * dq_sM + k] = tile_st[r * HD_PAD + k];
//     }
// }

// // =============================================================================
// // SM89 public entry point
// // =============================================================================
// void mem_efficient_attn_backward_sm89_cuda(
//     const float* query,       int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     const float* key,         int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     const float* value,       int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     const float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
//     const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
//     float* grad_query,        int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
//     float* grad_key,          int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
//     float* grad_value,        int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
//     float* D_buf,             int64_t d_strideB, int64_t d_strideH,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal,
//     bool skip_grad_zero)
// {
//     float scale = 1.0f / sqrtf(static_cast<float>(hd));
//     const int BH = (int)(B * nh);
//     dim3 block_cfg(SM89_BWD_NUM_THREADS);
//     dim3 grid_D(((int)T + (SM89_BWD_BLOCK_M_D * 2) - 1) / (SM89_BWD_BLOCK_M_D * 2), BH);

//     MemEfficientBwdParams params;
//     params.Q        = query;
//     params.K        = key;
//     params.V        = value;
//     params.O        = output;
//     params.dO       = grad_output;
//     params.LSE      = lse;
//     params.D        = D_buf;
//     params.dQ       = grad_query;
//     params.dK       = grad_key;
//     params.dV       = grad_value;
//     params.B        = (int)B;
//     params.nh       = (int)nh;
//     params.T        = (int)T;
//     params.scale    = scale;
//     params.is_causal = is_causal;
//     params.q_strideB  = q_strideB;  params.q_strideM  = q_strideM;  params.q_strideH  = q_strideH;
//     params.k_strideB  = k_strideB;  params.k_strideM  = k_strideM;  params.k_strideH  = k_strideH;
//     params.v_strideB  = v_strideB;  params.v_strideM  = v_strideM;  params.v_strideH  = v_strideH;
//     params.o_strideB  = o_strideB;  params.o_strideM  = o_strideM;  params.o_strideH  = o_strideH;
//     params.do_strideB = do_strideB; params.do_strideM = do_strideM; params.do_strideH = do_strideH;
//     params.dq_strideB = dq_strideB; params.dq_strideM = dq_strideM; params.dq_strideH = dq_strideH;
//     params.dk_strideB = dk_strideB; params.dk_strideM = dk_strideM; params.dk_strideH = dk_strideH;
//     params.dv_strideB = dv_strideB; params.dv_strideM = dv_strideM; params.dv_strideH = dv_strideH;
//     params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
//     params.d_strideB   = d_strideB;   params.d_strideH   = d_strideH;

// #define LAUNCH_MEM_BWD_EXP12(HD) \
//     do { \
//         constexpr int BN12 = 16, BM12 = 32; \
//         /* dK/dV kernel shmem: Q,dO + K,V + ds,DPV (QK-layout) + ds_kd,p_kd (KQ-layout) + tile_st + LSE,D */ \
//         const size_t shmem12 = \
//             (2ULL * BM12 * ((HD) + 4)   \
//            + 2ULL * BN12 * ((HD) + 4)   \
//            + 2ULL * BM12 * ((BN12) + 4) \
//            + 2ULL * BN12 * ((BM12) + 4) \
//            + 1ULL * BM12 * ((HD) + 4)   \
//            + 2ULL * BM12                \
//             ) * sizeof(float); \
//         /* dQ kernel shmem: Q,dO (persistent) + K,V (per-tile) + ds,DPV + LSE,D + tile_st */ \
//         /* No ds_kd/p_kd buffers — those are only needed for dK/dV accumulation.         */ \
//         const size_t shmem_dq = \
//             (3ULL * BM12 * ((HD) + 4)   \
//            + 2ULL * BN12 * ((HD) + 4)   \
//            + 2ULL * BM12 * ((BN12) + 4) \
//            + 2ULL * BM12                \
//             ) * sizeof(float); \
//         const int kv12 = ((int)T + BN12 - 1) / BN12; \
//         const int q12  = ((int)T + BM12 - 1) / BM12; \
//         dim3 grid_kv12(kv12, BH); \
//         dim3 grid_q12 (q12,  BH); \
//         cudaFuncSetAttribute( \
//             mem_efficient_bwd_unified_kernel_exp12<HD, false>, \
//             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
//         cudaFuncSetAttribute( \
//             mem_efficient_bwd_unified_kernel_exp12<HD, true>, \
//             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
//         cudaFuncSetAttribute( \
//             mem_efficient_bwd_qtile_dq_sm89<HD, false>, \
//             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_dq); \
//         cudaFuncSetAttribute( \
//             mem_efficient_bwd_qtile_dq_sm89<HD, true>, \
//             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_dq); \
//         mem_efficient_bwd_precompute_D_sm89<HD><<<grid_D, block_cfg>>>(params); \
//         if (is_causal) { \
//             mem_efficient_bwd_unified_kernel_exp12<HD, true> \
//                 <<<grid_kv12, block_cfg, shmem12>>>(params); \
//             mem_efficient_bwd_qtile_dq_sm89<HD, true> \
//                 <<<grid_q12, block_cfg, shmem_dq>>>(params); \
//         } else { \
//             mem_efficient_bwd_unified_kernel_exp12<HD, false> \
//                 <<<grid_kv12, block_cfg, shmem12>>>(params); \
//             mem_efficient_bwd_qtile_dq_sm89<HD, false> \
//                 <<<grid_q12, block_cfg, shmem_dq>>>(params); \
//         } \
//     } while (0)

//     switch ((int)hd) {
//         case  16: LAUNCH_MEM_BWD_EXP12( 16); break;
//         case  32: LAUNCH_MEM_BWD_EXP12( 32); break;
//         case  48: LAUNCH_MEM_BWD_EXP12( 48); break;
//         case  64: LAUNCH_MEM_BWD_EXP12( 64); break;
//         case  80: LAUNCH_MEM_BWD_EXP12( 80); break;
//         case  96: LAUNCH_MEM_BWD_EXP12( 96); break;
//         case 128: LAUNCH_MEM_BWD_EXP12(128); break;
//         case 160: LAUNCH_MEM_BWD_EXP12(160); break;
//         case 192: LAUNCH_MEM_BWD_EXP12(192); break;
//         case 256: LAUNCH_MEM_BWD_EXP12(256); break;
//         default:
//             printf("mem_efficient_attn_backward_sm89: unsupported head_dim %d\n", (int)hd);
//             break;
//     }
// #undef LAUNCH_MEM_BWD_EXP12
// }

// } // namespace OwnTensor