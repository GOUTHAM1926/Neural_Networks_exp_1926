#include "ops/cuda/attention/AttentionCommon.cuh"
#include "ops/helpers/AttentionKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "autograd/backward/AttentionBackward.h"

namespace OwnTensor{

template<typename FragType>
__device__ __forceinline__ void round_tf32_wmma_frag(FragType& frag) {
    #pragma unroll
    for (int i = 0; i < (int)FragType::num_elements; i++)
        reinterpret_cast<uint32_t&>(frag.x[i]) += 0x1000u;
}

    // ============================================================================
// LSE Kernel
// ============================================================================

__global__ void compute_row_lse_kernel(
    const float* __restrict__ scores,
    float* __restrict__ lse,
    int64_t T,
    bool is_causal)
{
    const int64_t row = blockIdx.x;
    const int64_t qi  = row % T;
    const float* row_ptr = scores + row * T;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    extern __shared__ float smem[];

    float local_max = -INFINITY;
    for (int j = tid; j < T; j += bdim) {
        float val = row_ptr[j];
        if (is_causal && j > qi) val = -INFINITY;
        local_max = fmaxf(local_max, val);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int j = tid; j < T; j += bdim) {
        float val = row_ptr[j];
        if (is_causal && j > qi) val = -INFINITY;
        local_sum += expf(val - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];

    if (tid == 0)
        lse[row] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
}


// ============================================================================
// Backward Pass
// ============================================================================

static constexpr int BWD_BLOCK_M   = 8;
static constexpr int BWD_WARP_SZ   = 32;
static constexpr int BWD_NUM_THREADS = BWD_BLOCK_M * BWD_WARP_SZ;  // 256
static constexpr int BWD_BLOCK_M_D = 8; 

__inline__ __device__ float bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}


template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D(MemEfficientBwdParams params)
{
    const float* __restrict__ dO = params.dO;
    const float* __restrict__ O  = params.O;
    float*       __restrict__ D  = params.D;
    const int T                  = (int)params.T;
    const int nh                 = params.nh;

    constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;
    const int b       = bh / nh;
    const int h       = bh - b * nh;

    // OPT-B + OPT-C: each warp covers 2 rows; block covers BWD_BLOCK_M_D*2 rows.
    // row0 and row1 are the two global row indices this warp processes.
    const int base_row = blockIdx.x * (BWD_BLOCK_M_D * 2) + warp_id * 2;
    const int row0     = base_row;
    const int row1     = base_row + 1;
    const bool v0      = (row0 < T);
    const bool v1      = (row1 < T);

    // Strided base pointers for this (b, h) pair; head-dim stride is always 1.
    const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
    const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
    const long long dO_off0 = (long long)row0 * params.do_strideM;
    const long long dO_off1 = (long long)row1 * params.do_strideM;
    const long long O_off0  = (long long)row0 * params.o_strideM;
    const long long O_off1  = (long long)row1 * params.o_strideM;

    // OPT-B: two independent accumulators — no dependency between them.
    // The compiler sees sum0 and sum1 as entirely separate register chains
    // and can schedule their FMAs in an interleaved fashion, keeping the
    // FP32 pipelines fed while waiting on memory.
    float sum0 = 0.f, sum1 = 0.f;

    // OPT-D: __ldg() on all read-only inputs.
    // dO and O are marked __restrict__ (no aliasing), but __ldg additionally
    // routes the load through the read-only data cache on Ampere, improving
    // L2 hit rate when the same cache lines are reused by exp9.
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        const int k = lane_id + i * BWD_WARP_SZ;
        if (k < HeadDim) {
            // OPT-B: both rows computed in the same loop body.
            // The two load pairs (dO/O for row0 and row1) are independent
            // → the memory subsystem can issue them concurrently.
            if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);  // OPT-D
            if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);  // OPT-D
        }
    }

    // OPT-B: two back-to-back warp_sum calls with no data dependency.
    // PTX scheduler pipelines the two shfl_xor sequences (5 rounds each),
    // filling the ~20-cycle stall slots that dropped fadd utilisation to 55%.
    sum0 = bwd_warp_sum(sum0);
    sum1 = bwd_warp_sum(sum1);

    // Write results — only lane 0 holds the fully-reduced scalar.
    if (lane_id == 0) {
        float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
        if (v0) D_bh[row0] = sum0;
        if (v1) D_bh[row1] = sum1;
    }
}
// ============================================================================
// Exp7: scalar KV-outer backward, fallback for non-multiple-of-16 HeadDims
// ============================================================================

template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp7(MemEfficientBwdParams params)
{
    constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
    constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
    constexpr int HD_PAD  = HeadDim + 1;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    // smem: ONLY K and V tiles — same as exp6
    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = Ks + BlockN * HD_PAD;

    const int bh         = blockIdx.y;
    const int kv_tile    = blockIdx.x;
    const int tile_start = kv_tile * BlockN;
    const int tile_size  = min(BlockN, params.T - tile_start);
    if (tile_start >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
    const int lane_id = threadIdx.x % BWD_WARP_SZ;
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

    // Step 0: zero this block's dK/dV rows in global memory (scalar, coalesced)
    // Scalar avoids any float4 alignment edge cases; still fully coalesced.
    for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim;
        const int k = idx % HeadDim;
        dK_bh[(tile_start + r) * dk_sM + k] = 0.f;
        dV_bh[(tile_start + r) * dv_sM + k] = 0.f;
    }

    // Bank-conflict-free K/V smem load:
    // Thread idx → row r = idx/HeadDim, col k = idx%HeadDim.
    // Same-warp threads differ in k (consecutive) → consecutive smem banks → no conflict.
    // (float4 had stride d4*4 within row → threads t and t+8 collided on the same banks.)
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r     = idx / HeadDim;
        const int k     = idx % HeadDim;
        const int g_row = tile_start + r;
        Ks[r * HD_PAD + k] = (g_row < params.T) ? K_bh[g_row * k_sM + k] : 0.f;
        Vs[r * HD_PAD + k] = (g_row < params.T) ? V_bh[g_row * v_sM + k] : 0.f;
    }

    // Barrier: K/V loaded, dK/dV zeroed — safe to proceed
    __syncthreads();

    const int q_start = Causal ? tile_start : 0;

    for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
        const int qi     = q_base + warp_id;
        const bool valid = (qi < params.T);

        float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
        float L_qi = 0.0f, D_qi = 0.0f;

        #pragma unroll
        for (int i = 0; i < LocalN; ++i) {
            const int k = lane_id + i * BWD_WARP_SZ;
            float qv = 0.0f, dov = 0.0f;
            if (valid && k < HeadDim) {
                qv  = Q_bh [qi * q_sM  + k];
                dov = dO_bh[qi * do_sM + k];
            }
            q_local[i]  = qv;
            do_local[i] = dov;
            dq_local[i] = 0.0f;
        }
        if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

        // Fused Phase A+B: same as exp6 — merged i-loop, global atomics for dK/dV.
        // Smem reads (Ks/Vs) are bank-conflict-free here since k varies within a warp.
        #pragma unroll 4
        for (int j = 0; j < BlockN; ++j) {
            const bool j_valid = (j < tile_size);

            float dot_qk = 0.0f, dot_dov = 0.0f;
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
                    dot_dov += do_local[i] * Vs[j * HD_PAD + k];
                }
            }

            const float s   = bwd_warp_sum(dot_qk) * params.scale;
            const float dpv = bwd_warp_sum(dot_dov);

            float p = 0.0f;
            if (j_valid && valid && !(Causal && (tile_start + j) > qi))
                p = expf(s - L_qi);

            const float ds = p * (dpv - D_qi);

            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim) {
                    dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
                    atomicAdd(&dK_bh[(tile_start + j) * dk_sM + k],
                              ds * params.scale * q_local[i]);
                    atomicAdd(&dV_bh[(tile_start + j) * dv_sM + k],
                              p  *                do_local[i]);
                }
            }
        }

        if (valid) {
            #pragma unroll
            for (int i = 0; i < LocalN; ++i) {
                const int k = lane_id + i * BWD_WARP_SZ;
                if (k < HeadDim)
                    atomicAdd(&dQ_bh[qi * dq_sM + k], dq_local[i]);
            }
        }
    } // q_base loop
}
// ============================================================================
// Exp11: KV-tile-centric backward, TF32 WMMA, dK/dV atomic-free (HD%16==0)
//
// Each block owns one [BN×HD] KV tile.  dK and dV are accumulated in
// persistent WMMA register fragments across the Q loop and written once
// directly to global memory — no atomicAdd, deterministic.
// dQ uses atomicAdd (many KV blocks → same Q rows); error is ~1 ULP per
// KV-block contribution, vs. ~T·eps for the old Q-centric dV design.
//
// Smem layout is identical to the former Q-centric kernel — tile_st [BM×HD]
// is used for dQ staging per Q-tile and reused after the loop to stage
// dk_frag / dv_frag for the non-atomic global writes.
// ============================================================================

template <int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp11(MemEfficientBwdParams params)
{
    using namespace nvcuda;

    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 16;
    constexpr int HD_CHUNKS = HeadDim / 16;
    constexpr int HD_PAD    = HeadDim;
    constexpr int BKN_PAD   = BlockN;
    constexpr int BM_PAD    = BM_WMMA;

    extern __shared__ float smem_f[];
    float* Q_sm    = smem_f;
    float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
    float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
    float* Vs      = Ks      + BlockN   * HD_PAD;
    float* ds_qd   = Vs      + BlockN   * HD_PAD;   // [BM×BKN_PAD]
    float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;  // [BM×BKN_PAD]
    float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;  // [BN×BM_PAD]
    float* p_kd    = ds_kd   + BlockN   * BM_PAD;   // [BN×BM_PAD]
    float* tile_st = p_kd    + BlockN   * BM_PAD;   // [BM×HD] dQ per Q-tile / dk+dv final stage
    float* LSE_sm  = tile_st + BM_WMMA  * HeadDim;  // [BM_WMMA] per-iteration LSE
    float* D_sm    = LSE_sm  + BM_WMMA;             // [BM_WMMA] per-iteration D

    // Grid: blockIdx.x = KV-tile, blockIdx.y = batch-head
    const int bh           = blockIdx.y;
    const int kv_tile      = blockIdx.x;
    const int kv_base      = kv_tile * BlockN;
    const int kv_tile_size = min(BlockN, params.T - kv_base);
    if (kv_base >= params.T) return;

    const int warp_id = threadIdx.x / BWD_WARP_SZ;
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

    // ── Step 0: load K, V for this KV-tile into smem (persistent) ─────────────
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = kv_base + r;
        Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
    }
    __syncthreads();

    // ── Persistent WMMA accumulators for dK and dV ────────────────────────────
    // Warps HD_CHUNKS..2*HD_CHUNKS-1 accumulate these across the entire Q loop.
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> dk_frag, dv_frag;
    if (warp_id >= HD_CHUNKS) {
        wmma::fill_fragment(dk_frag, 0.0f);
        wmma::fill_fragment(dv_frag, 0.0f);
    }

    // Causal: only Q-tiles at q_base >= kv_base attend to this KV tile.
    const int q_loop_start = Causal ? kv_base : 0;

    // ── Main Q loop ───────────────────────────────────────────────────────────
    for (int q_base = q_loop_start; q_base < params.T; q_base += BM_WMMA) {

        // sync(a): prior atomicAdd reads of tile_st and dv_frag MMA reads of
        //          dO_sm/p_kd are done — safe to overwrite Q_sm and dO_sm.
        __syncthreads();

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
        __syncthreads();  // sync(b): Q_sm, dO_sm, LSE_sm, D_sm ready

        // ── Phase A: TF32 WMMA GEMMs ──────────────────────────────────────────
        //
        // Warp 0: S   = Q_sm × K^T  → ds_qd  [BM×BKN_PAD]   (col_major B = K^T)
        // Warp 1: DPV = dO_sm × V^T → DPV_sm [BM×BKN_PAD]   (col_major B = V^T)
        // K, V are persistent in Ks/Vs from Step 0.
        if (warp_id < 2) {
            const float* src_sm = (warp_id == 0) ? Q_sm   : dO_sm;
            const float* kv_sm  = (warp_id == 0) ? Ks     : Vs;
            float*       dst_sm = (warp_id == 0) ? ds_qd  : DPV_sm;

            wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> b_frag;

            #pragma unroll
            for (int ks = 0; ks < 2 * HD_CHUNKS; ++ks) {
                const int k_off = ks * 8;
                wmma::load_matrix_sync(a_frag, src_sm + k_off, HD_PAD);
                wmma::load_matrix_sync(b_frag, kv_sm  + k_off, HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            wmma::store_matrix_sync(dst_sm, acc_frag, BKN_PAD, wmma::mem_row_major);
        }
        __syncthreads();  // sync(c1): ds_qd (=S_sm) and DPV_sm fully written

        // Post-process: 256 threads × 1 element — full coverage of [BM×BN]
        {
            const int qi_local  = threadIdx.x / BlockN;
            const int j_local   = threadIdx.x % BlockN;
            const int qi_global = q_base  + qi_local;
            const int j_global  = kv_base + j_local;

            const float raw_s = ds_qd [qi_local * BKN_PAD + j_local];
            const float dpv   = DPV_sm[qi_local * BKN_PAD + j_local];
            const float L     = LSE_sm[qi_local];
            const float D_    = D_sm  [qi_local];

            const bool qi_valid  = (qi_global < params.T);
            const bool j_valid   = (j_local   < kv_tile_size);
            const bool causal_ok = !Causal || (j_global <= qi_global);

            float p = 0.f;
            if (qi_valid && j_valid && causal_ok)
                p = expf(raw_s * params.scale - L);

            const float ds = p * (dpv - D_) * params.scale;

            ds_qd[qi_local * BKN_PAD + j_local]  = ds;
            ds_kd[j_local  * BM_PAD  + qi_local] = ds;
            p_kd [j_local  * BM_PAD  + qi_local] = p;
        }
        __syncthreads();  // sync(c2): ds_qd, ds_kd, p_kd fully written

        // ── Phase B-1 ─────────────────────────────────────────────────────────
        //
        // Warps 0..HD_CHUNKS-1  : dQ WMMA → tile_st[BM×16-chunk]
        //   A = ds_qd [BM×BN],  B = Ks[:,chunk*16..+16]
        //
        // Warps HD_CHUNKS..2*HD-1: dK WMMA → dk_frag (persistent accumulator)
        //   A = ds_kd [BN×BM],  B = Q_sm[:,chunk*16..+16]
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            if (warp_id < HD_CHUNKS) {
                // dQ: A=ds_qd[BM×BN], B=Ks[BN×16], C→tile_st[BM×16]
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> dq_tile_frag;
                wmma::fill_fragment(dq_tile_frag, 0.0f);
                const float* b_ptr = Ks + chunk * 16;
                wmma::load_matrix_sync(a_frag, ds_qd,              BKN_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(dq_tile_frag, a_frag, b_frag, dq_tile_frag);
                wmma::load_matrix_sync(a_frag, ds_qd + 8,          BKN_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(dq_tile_frag, a_frag, b_frag, dq_tile_frag);
                wmma::store_matrix_sync(tile_st + chunk * 16, dq_tile_frag,
                                        HeadDim, wmma::mem_row_major);
            } else {
                // dK: A=ds_kd[BN×BM], B=Q_sm[BM×16], accumulate into dk_frag
                const float* b_ptr = Q_sm + chunk * 16;
                wmma::load_matrix_sync(a_frag, ds_kd,              BM_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
                wmma::load_matrix_sync(a_frag, ds_kd + 8,          BM_PAD);
                wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
            }
        }
        __syncthreads();  // sync(d): tile_st has dQ for this Q tile

        // All 256 threads atomicAdd tile_st → global dQ
        for (int idx = threadIdx.x; idx < q_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dQ_bh[(q_base + r) * dq_sM + k], tile_st[r * HeadDim + k]);
        }

        // ── Phase B-2: dV (warps >= HD_CHUNKS) ───────────────────────────────
        // Reads p_kd and dO_sm — no conflict with atomicAdd (different smem).
        // sync(a) at next iteration top ensures dO_sm is not overwritten while
        // this MMA is still in flight.
        if (warp_id >= HD_CHUNKS) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            const float* b_ptr = dO_sm + chunk * 16;
            wmma::load_matrix_sync(a_frag, p_kd,              BM_PAD);
            wmma::load_matrix_sync(b_frag, b_ptr,             HD_PAD);
            round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
            wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
            wmma::load_matrix_sync(a_frag, p_kd + 8,          BM_PAD);
            wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
            round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
            wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
        }
        // sync(a) at loop top covers atomicAdd reads and dv_frag MMA reads.
    } // Q loop

    // ── Final: dk_frag → tile_st → global dK (no atomicAdd) ──────────────────
    __syncthreads();  // ensure last atomicAdd reads and dv_frag MMA reads complete

    if (warp_id >= HD_CHUNKS) {
        wmma::store_matrix_sync(tile_st + chunk * 16, dk_frag,
                                HeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(kv_base + r) * dk_sM + k] = tile_st[r * HeadDim + k];
    }
    __syncthreads();

    // ── Final: dv_frag → tile_st → global dV (no atomicAdd) ──────────────────
    if (warp_id >= HD_CHUNKS) {
        wmma::store_matrix_sync(tile_st + chunk * 16, dv_frag,
                                HeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dV_bh[(kv_base + r) * dv_sM + k] = tile_st[r * HeadDim + k];
    }
}

// ============================================================================
// exp12 (SM89 Ada Lovelace) lives in arch/AttentionBackward_sm89.cu.
// Forward-declare the entry point for the arch-dispatch below.
// ============================================================================
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
    bool skip_grad_zero);

// ============================================================================
// Public API (namespace cuda)
// ============================================================================

namespace cuda {

void compute_row_lse(
    const float* scores, float* lse,
    int64_t batch, int64_t T, bool is_causal)
{
    int64_t total_rows = batch * T;
    int threads = 32;
    while (threads < T && threads < 1024) threads <<= 1;
    size_t smem = threads * sizeof(float);
    compute_row_lse_kernel<<<(int)total_rows, threads, smem>>>(
        scores, lse, T, is_causal);
}

void mem_efficient_attn_backward(
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
    dim3 block_cfg(BWD_NUM_THREADS);

    dim3 grid_D(((int)T + (BWD_BLOCK_M_D * 2) - 1) / (BWD_BLOCK_M_D * 2), BH);

    MemEfficientBwdParams params;
    params.Q     = query;
    params.K     = key;
    params.V     = value;
    params.O     = output;
    params.dO    = grad_output;
    params.LSE   = lse;
    params.D     = D_buf;
    params.dQ    = grad_query;
    params.dK    = grad_key;
    params.dV    = grad_value;
    params.B     = (int)B;
    params.nh    = (int)nh;
    params.T     = (int)T;
    params.scale = scale;
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

    // exp7: scalar KV-outer fallback for HeadDims not divisible by 16
    // exp7 is the only path that atomicAdds into dQ, so it must zero grad_query here.
#define LAUNCH_MEM_BWD_EXP7(HD) \
    do { \
        const int block_n7 = ((HD) < 64) ? 16 : (1024 / (HD)); \
        const size_t shmem_exp7 = 2ULL * block_n7 * ((HD) + 1) * sizeof(float); \
        const int kv_tiles7 = ((int)T + block_n7 - 1) / block_n7; \
        dim3 grid_bwd7(kv_tiles7, BH); \
        if (!skip_grad_zero) { \
            cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
        } \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp7<HD, true> \
                <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp7<HD, false> \
                <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
        } \
    } while (0)

    // exp11: KV-tile-centric, TF32 WMMA, dK/dV atomic-free, HD%16==0 only
#define LAUNCH_MEM_BWD_EXP11(HD) \
    do { \
        constexpr int BN11 = 16, BM11 = 16; \
        const size_t shmem11 = \
            (2ULL * BM11 * ((HD) + 1)   /* Q_sm + dO_sm   */ \
           + 2ULL * BN11 * ((HD) + 1)   /* Ks + Vs        */ \
           + 2ULL * BM11 * ((BN11) + 1) /* ds_qd + DPV_sm */ \
           + 2ULL * BN11 * ((BM11) + 1) /* ds_kd + p_kd   */ \
           + 1ULL * BM11 * (HD)         /* tile_st (dQ staging + dk/dv final) */ \
           + 2ULL * BM11                /* LSE_sm + D_sm  */ \
            ) * sizeof(float); \
        const int kv11 = ((int)T + BN11 - 1) / BN11; \
        dim3 grid_kv11(kv11, BH); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp11<HD, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem11); \
        if (!skip_grad_zero) { \
            cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
        } \
        mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp11<HD, true> \
                <<<grid_kv11, block_cfg, shmem11>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp11<HD, false> \
                <<<grid_kv11, block_cfg, shmem11>>>(params); \
        } \
    } while (0)

    // SM89 (Ada Lovelace): dispatch to exp12 (BM=32, bank-conflict-free, 2 blocks/SM).
    // Non-%16 HD falls through to exp7 below regardless of arch.
    if (hd % 16 == 0 && cuda::get_arch() == cuda::ArchFamily::Ada) {
      //if (false) {  
        mem_efficient_attn_backward_sm89_cuda(
            query,       q_strideB, q_strideM, q_strideH,
            key,         k_strideB, k_strideM, k_strideH,
            value,       v_strideB, v_strideM, v_strideH,
            output,      o_strideB, o_strideM, o_strideH,
            grad_output, do_strideB, do_strideM, do_strideH,
            lse,         lse_strideB, lse_strideH,
            grad_query,  dq_strideB, dq_strideM, dq_strideH,
            grad_key,    dk_strideB, dk_strideM, dk_strideH,
            grad_value,  dv_strideB, dv_strideM, dv_strideH,
            D_buf,       d_strideB, d_strideH,
            B, nh, T, hd, is_causal, skip_grad_zero);
        return;
    }

    switch ((int)hd) {
        case   8: LAUNCH_MEM_BWD_EXP7(  8); break;
        case  16: LAUNCH_MEM_BWD_EXP11( 16); break;
        case  24: LAUNCH_MEM_BWD_EXP7( 24); break;
        case  32: LAUNCH_MEM_BWD_EXP11( 32); break;
        case  40: LAUNCH_MEM_BWD_EXP7( 40); break;
        case  48: LAUNCH_MEM_BWD_EXP11( 48); break;
        case  56: LAUNCH_MEM_BWD_EXP7( 56); break;
        case  64: LAUNCH_MEM_BWD_EXP11( 64); break;
        case  80: LAUNCH_MEM_BWD_EXP11( 80); break;
        case  96: LAUNCH_MEM_BWD_EXP11( 96); break;
        case 128: LAUNCH_MEM_BWD_EXP11(128); break;
        case 160: LAUNCH_MEM_BWD_EXP11(160); break;
        case 192: LAUNCH_MEM_BWD_EXP11(192); break;
        case 256: LAUNCH_MEM_BWD_EXP11(256); break;
        default:
            printf("mem_efficient_attn_backward: unsupported head_dim %d\n", (int)hd);
            break;
    }
#undef LAUNCH_MEM_BWD_EXP7
#undef LAUNCH_MEM_BWD_EXP11
}

} // namespace cuda
} // namespace OwnTensor


// // #include "ops/cuda/attention/AttentionCommon.cuh"
// // #include "ops/helpers/AttentionKernels.h"
// // #include "ops/helpers/KernelDispatch.h"
// // #include "autograd/backward/AttentionBackward.h"

// // namespace OwnTensor{

// //     // ============================================================================
// // // LSE Kernel
// // // ============================================================================

// // __global__ void compute_row_lse_kernel(
// //     const float* __restrict__ scores,
// //     float* __restrict__ lse,
// //     int64_t T,
// //     bool is_causal)
// // {
// //     const int64_t row = blockIdx.x;
// //     const int64_t qi  = row % T;
// //     const float* row_ptr = scores + row * T;
// //     const int tid  = threadIdx.x;
// //     const int bdim = blockDim.x;

// //     extern __shared__ float smem[];

// //     float local_max = -INFINITY;
// //     for (int j = tid; j < T; j += bdim) {
// //         float val = row_ptr[j];
// //         if (is_causal && j > qi) val = -INFINITY;
// //         local_max = fmaxf(local_max, val);
// //     }
// //     smem[tid] = local_max;
// //     __syncthreads();
// //     for (int s = bdim / 2; s > 0; s >>= 1) {
// //         if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
// //         __syncthreads();
// //     }
// //     float row_max = smem[0];

// //     float local_sum = 0.0f;
// //     for (int j = tid; j < T; j += bdim) {
// //         float val = row_ptr[j];
// //         if (is_causal && j > qi) val = -INFINITY;
// //         local_sum += expf(val - row_max);
// //     }
// //     smem[tid] = local_sum;
// //     __syncthreads();
// //     for (int s = bdim / 2; s > 0; s >>= 1) {
// //         if (tid < s) smem[tid] += smem[tid + s];
// //         __syncthreads();
// //     }
// //     float row_sum = smem[0];

// //     if (tid == 0)
// //         lse[row] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
// // }


// // // ============================================================================
// // // Backward Pass
// // // ============================================================================

// // static constexpr int BWD_BLOCK_M   = 8;
// // static constexpr int BWD_WARP_SZ   = 32;
// // static constexpr int BWD_NUM_THREADS = BWD_BLOCK_M * BWD_WARP_SZ;  // 256
// // static constexpr int BWD_BLOCK_M_D = 8; 

// // __inline__ __device__ float bwd_warp_sum(float val) {
// //     #pragma unroll
// //     for (int offset = BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
// //         val += __shfl_xor_sync(0xffffffff, val, offset);
// //     return val;
// // }


// // template<int HeadDim>
// // __global__ void mem_efficient_bwd_precompute_D(MemEfficientBwdParams params)
// // {
// //     const float* __restrict__ dO = params.dO;
// //     const float* __restrict__ O  = params.O;
// //     float*       __restrict__ D  = params.D;
// //     const int T                  = (int)params.T;
// //     const int nh                 = params.nh;

// //     constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;

// //     const int bh      = blockIdx.y;
// //     const int warp_id = threadIdx.x / BWD_WARP_SZ;
// //     const int lane_id = threadIdx.x % BWD_WARP_SZ;
// //     const int b       = bh / nh;
// //     const int h       = bh - b * nh;

// //     // OPT-B + OPT-C: each warp covers 2 rows; block covers BWD_BLOCK_M_D*2 rows.
// //     // row0 and row1 are the two global row indices this warp processes.
// //     const int base_row = blockIdx.x * (BWD_BLOCK_M_D * 2) + warp_id * 2;
// //     const int row0     = base_row;
// //     const int row1     = base_row + 1;
// //     const bool v0      = (row0 < T);
// //     const bool v1      = (row1 < T);

// //     // Strided base pointers for this (b, h) pair; head-dim stride is always 1.
// //     const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
// //     const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
// //     const long long dO_off0 = (long long)row0 * params.do_strideM;
// //     const long long dO_off1 = (long long)row1 * params.do_strideM;
// //     const long long O_off0  = (long long)row0 * params.o_strideM;
// //     const long long O_off1  = (long long)row1 * params.o_strideM;

// //     // OPT-B: two independent accumulators — no dependency between them.
// //     // The compiler sees sum0 and sum1 as entirely separate register chains
// //     // and can schedule their FMAs in an interleaved fashion, keeping the
// //     // FP32 pipelines fed while waiting on memory.
// //     float sum0 = 0.f, sum1 = 0.f;

// //     // OPT-D: __ldg() on all read-only inputs.
// //     // dO and O are marked __restrict__ (no aliasing), but __ldg additionally
// //     // routes the load through the read-only data cache on Ampere, improving
// //     // L2 hit rate when the same cache lines are reused by exp9.
// //     #pragma unroll
// //     for (int i = 0; i < LocalN; ++i) {
// //         const int k = lane_id + i * BWD_WARP_SZ;
// //         if (k < HeadDim) {
// //             // OPT-B: both rows computed in the same loop body.
// //             // The two load pairs (dO/O for row0 and row1) are independent
// //             // → the memory subsystem can issue them concurrently.
// //             if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);  // OPT-D
// //             if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);  // OPT-D
// //         }
// //     }

// //     // OPT-B: two back-to-back warp_sum calls with no data dependency.
// //     // PTX scheduler pipelines the two shfl_xor sequences (5 rounds each),
// //     // filling the ~20-cycle stall slots that dropped fadd utilisation to 55%.
// //     sum0 = bwd_warp_sum(sum0);
// //     sum1 = bwd_warp_sum(sum1);

// //     // Write results — only lane 0 holds the fully-reduced scalar.
// //     if (lane_id == 0) {
// //         float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
// //         if (v0) D_bh[row0] = sum0;
// //         if (v1) D_bh[row1] = sum1;
// //     }
// // }
// // // ============================================================================
// // // Exp7: scalar KV-outer backward, fallback for non-multiple-of-16 HeadDims
// // // ============================================================================

// // template<int HeadDim, bool Causal>
// // __global__ void mem_efficient_bwd_unified_kernel_exp7(MemEfficientBwdParams params)
// // {
// //     constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
// //     constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
// //     constexpr int HD_PAD  = HeadDim + 1;
// //     constexpr float BWD_LOG2E = 1.4426950408889634074f;

// //     // smem: ONLY K and V tiles — same as exp6
// //     extern __shared__ float smem[];
// //     float* Ks = smem;
// //     float* Vs = Ks + BlockN * HD_PAD;

// //     const int bh         = blockIdx.y;
// //     const int kv_tile    = blockIdx.x;
// //     const int tile_start = kv_tile * BlockN;
// //     const int tile_size  = min(BlockN, params.T - tile_start);
// //     if (tile_start >= params.T) return;

// //     const int warp_id = threadIdx.x / BWD_WARP_SZ;
// //     const int lane_id = threadIdx.x % BWD_WARP_SZ;
// //     const int b       = bh / params.nh;
// //     const int h       = bh - b * params.nh;

// //     const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
// //     const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
// //     const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
// //     const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
// //     const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
// //     const float* D_bh   = params.D   + b * params.d_strideB   + h * params.d_strideH;
// //     float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;
// //     float*       dK_bh  = params.dK  + b * params.dk_strideB  + h * params.dk_strideH;
// //     float*       dV_bh  = params.dV  + b * params.dv_strideB  + h * params.dv_strideH;

// //     const int64_t q_sM  = params.q_strideM;
// //     const int64_t k_sM  = params.k_strideM;
// //     const int64_t v_sM  = params.v_strideM;
// //     const int64_t do_sM = params.do_strideM;
// //     const int64_t dq_sM = params.dq_strideM;
// //     const int64_t dk_sM = params.dk_strideM;
// //     const int64_t dv_sM = params.dv_strideM;

// //     // Step 0: zero this block's dK/dV rows in global memory (scalar, coalesced)
// //     // Scalar avoids any float4 alignment edge cases; still fully coalesced.
// //     for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
// //         const int r = idx / HeadDim;
// //         const int k = idx % HeadDim;
// //         dK_bh[(tile_start + r) * dk_sM + k] = 0.f;
// //         dV_bh[(tile_start + r) * dv_sM + k] = 0.f;
// //     }

// //     // Bank-conflict-free K/V smem load:
// //     // Thread idx → row r = idx/HeadDim, col k = idx%HeadDim.
// //     // Same-warp threads differ in k (consecutive) → consecutive smem banks → no conflict.
// //     // (float4 had stride d4*4 within row → threads t and t+8 collided on the same banks.)
// //     for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
// //         const int r     = idx / HeadDim;
// //         const int k     = idx % HeadDim;
// //         const int g_row = tile_start + r;
// //         Ks[r * HD_PAD + k] = (g_row < params.T) ? K_bh[g_row * k_sM + k] : 0.f;
// //         Vs[r * HD_PAD + k] = (g_row < params.T) ? V_bh[g_row * v_sM + k] : 0.f;
// //     }

// //     // Barrier: K/V loaded, dK/dV zeroed — safe to proceed
// //     __syncthreads();

// //     const int q_start = Causal ? tile_start : 0;

// //     for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
// //         const int qi     = q_base + warp_id;
// //         const bool valid = (qi < params.T);

// //         float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
// //         float L_qi = 0.0f, D_qi = 0.0f;

// //         #pragma unroll
// //         for (int i = 0; i < LocalN; ++i) {
// //             const int k = lane_id + i * BWD_WARP_SZ;
// //             float qv = 0.0f, dov = 0.0f;
// //             if (valid && k < HeadDim) {
// //                 qv  = Q_bh [qi * q_sM  + k];
// //                 dov = dO_bh[qi * do_sM + k];
// //             }
// //             q_local[i]  = qv;
// //             do_local[i] = dov;
// //             dq_local[i] = 0.0f;
// //         }
// //         if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

// //         // Fused Phase A+B: same as exp6 — merged i-loop, global atomics for dK/dV.
// //         // Smem reads (Ks/Vs) are bank-conflict-free here since k varies within a warp.
// //         #pragma unroll 4
// //         for (int j = 0; j < BlockN; ++j) {
// //             const bool j_valid = (j < tile_size);

// //             float dot_qk = 0.0f, dot_dov = 0.0f;
// //             #pragma unroll
// //             for (int i = 0; i < LocalN; ++i) {
// //                 const int k = lane_id + i * BWD_WARP_SZ;
// //                 if (k < HeadDim) {
// //                     dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
// //                     dot_dov += do_local[i] * Vs[j * HD_PAD + k];
// //                 }
// //             }

// //             const float s   = bwd_warp_sum(dot_qk) * params.scale;
// //             const float dpv = bwd_warp_sum(dot_dov);

// //             float p = 0.0f;
// //             if (j_valid && valid && !(Causal && (tile_start + j) > qi))
// //                 p = exp2f(BWD_LOG2E * (s - L_qi));

// //             const float ds = p * (dpv - D_qi);

// //             #pragma unroll
// //             for (int i = 0; i < LocalN; ++i) {
// //                 const int k = lane_id + i * BWD_WARP_SZ;
// //                 if (k < HeadDim) {
// //                     dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
// //                     atomicAdd(&dK_bh[(tile_start + j) * dk_sM + k],
// //                               ds * params.scale * q_local[i]);
// //                     atomicAdd(&dV_bh[(tile_start + j) * dv_sM + k],
// //                               p  *                do_local[i]);
// //                 }
// //             }
// //         }

// //         if (valid) {
// //             #pragma unroll
// //             for (int i = 0; i < LocalN; ++i) {
// //                 const int k = lane_id + i * BWD_WARP_SZ;
// //                 if (k < HeadDim)
// //                     atomicAdd(&dQ_bh[qi * dq_sM + k], dq_local[i]);
// //             }
// //         }
// //     } // q_base loop
// // }
// // // ============================================================================
// // // Exp11: Q-tile-centric backward, TF32 WMMA, dQ atomic-free (HD%16==0 only)
// // // ============================================================================

// // template <int HeadDim, bool Causal>
// // __global__ void mem_efficient_bwd_unified_kernel_exp11(MemEfficientBwdParams params)
// // {
// //     using namespace nvcuda;

// //     constexpr int BlockN    = 16;
// //     constexpr int BM_WMMA   = 16;
// //     constexpr int HD_CHUNKS = HeadDim / 16;
// //     constexpr int HD_PAD    = HeadDim;        // WMMA TF32 requires LDM to be multiple of 4 floats (16 bytes)
// //     constexpr int BKN_PAD   = BlockN;         // WMMA TF32 LDM constraint
// //     constexpr int BM_PAD    = BM_WMMA;        // WMMA TF32 LDM constraint
// //     constexpr float BWD_LOG2E = 1.4426950408889634074f;

// //     extern __shared__ float smem_f[];
// //     float* Q_sm    = smem_f;
// //     float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
// //     float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
// //     float* Vs      = Ks      + BlockN   * HD_PAD;
// //     float* ds_qd   = Vs      + BlockN   * HD_PAD;   // [BM×BKN_PAD] — S_sm / dQ WMMA A
// //     float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;  // [BM×BKN_PAD] — raw dO·V^T
// //     float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;  // [BN×BM_PAD]  — dK WMMA A
// //     float* p_kd    = ds_kd   + BlockN   * BM_PAD;   // [BN×BM_PAD]  — dV WMMA A
// //     float* tile_st = p_kd    + BlockN   * BM_PAD;   // [BN×HD] per-iter dK/dV + final dQ
// //     float* LSE_sm  = tile_st + BlockN   * HeadDim;  // [BM_WMMA] per-block LSE
// //     float* D_sm    = LSE_sm  + BM_WMMA;             // [BM_WMMA] per-block D

// //     const int bh           = blockIdx.y;
// //     const int q_tile       = blockIdx.x;
// //     const int q_tile_start = q_tile * BM_WMMA;
// //     const int tile_size    = min(BM_WMMA, params.T - q_tile_start);
// //     if (q_tile_start >= params.T) return;

// //     const int warp_id = threadIdx.x / BWD_WARP_SZ;
// //     const int chunk   = warp_id % HD_CHUNKS;  // column chunk of HeadDim [0, HD_CHUNKS)
// //     const int b       = bh / params.nh;
// //     const int h       = bh - b * params.nh;

// //     const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
// //     const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
// //     const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
// //     const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
// //     const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
// //     const float* D_bh   = params.D   + b * params.d_strideB   + h * params.d_strideH;
// //     float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;
// //     float*       dK_bh  = params.dK  + b * params.dk_strideB  + h * params.dk_strideH;
// //     float*       dV_bh  = params.dV  + b * params.dv_strideB  + h * params.dv_strideH;

// //     const int64_t q_sM  = params.q_strideM;
// //     const int64_t k_sM  = params.k_strideM;
// //     const int64_t v_sM  = params.v_strideM;
// //     const int64_t do_sM = params.do_strideM;
// //     const int64_t dq_sM = params.dq_strideM;
// //     const int64_t dk_sM = params.dk_strideM;
// //     const int64_t dv_sM = params.dv_strideM;

// //     // ── Step 0: load Q, dO, LSE, D for this Q-tile into smem (persistent) ────
// //     for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
// //         const int r  = idx / HeadDim, k = idx % HeadDim;
// //         const int qi = q_tile_start + r;
// //         const bool vq = (qi < params.T);
// //         Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
// //         dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
// //     }
// //     if (threadIdx.x < BM_WMMA) {
// //         const int qi  = q_tile_start + threadIdx.x;
// //         const bool vq = (qi < params.T);
// //         LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
// //         D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
// //     }
// //     __syncthreads();

// //     // ── Persistent TF32 WMMA accumulator for dQ ──────────────────────────────
// //     // Only warps 0..HD_CHUNKS-1 use this; others keep it at zero (never stored).
// //     wmma::fragment<wmma::accumulator, 16, 16, 8, float> dq_frag;
// //     wmma::fill_fragment(dq_frag, 0.0f);

// //     // Causal: skip KV tiles entirely past the Q-tile's last row
// //     const int kv_loop_end = Causal ? (q_tile_start + BM_WMMA) : params.T;

// //     // ── Main KV loop ──────────────────────────────────────────────────────────
// //     for (int kv_base = 0; kv_base < kv_loop_end; kv_base += BlockN) {

// //         // sync(a): prior dV atomicAdd reads of tile_st are done; prior Phase B
// //         //          reads of Ks are done — safe to overwrite both.
// //         __syncthreads();

// //         const int kv_tile_size = min(BlockN, params.T - kv_base);

// //         // Load K/V tile for this KV iteration
// //         for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
// //             const int r = idx / HeadDim, k = idx % HeadDim;
// //             const int g = kv_base + r;
// //             Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
// //             Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
// //         }
// //         __syncthreads();  // sync(b): Ks/Vs ready

// //         // ── Phase A: TF32 WMMA GEMMs + scalar post-process ───────────────────
// //         //
// //         // Warp 0: S   = Q_sm × K^T  → ds_qd  [BM×BKN_PAD]   (col_major B = K^T)
// //         // Warp 1: DPV = dO_sm × V^T → DPV_sm [BM×BKN_PAD]   (col_major B = V^T)
// //         // Warps 2..7 are idle during WMMA.
// //         //
// //         // After sync(c1), all 256 threads run the scalar post-process in parallel
// //         // (1 [qi, j] element each): compute p and ds from S, DPV, LSE_sm, D_sm,
// //         // then write ds_qd (overwrite S with ds), ds_kd, and p_kd.
// //         if (warp_id < 2) {
// //             const float* src_sm = (warp_id == 0) ? Q_sm   : dO_sm;
// //             const float* kv_sm  = (warp_id == 0) ? Ks     : Vs;
// //             float*       dst_sm = (warp_id == 0) ? ds_qd  : DPV_sm;

// //             wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
// //             wmma::fill_fragment(acc_frag, 0.0f);

// //             // col_major B fragment loads K^T (or V^T): b[k][j] = kv_sm[j*HD_PAD + k_off + k]
// //             wmma::fragment<wmma::matrix_a, 16, 16, 8,
// //                            wmma::precision::tf32, wmma::row_major> a_frag;
// //             wmma::fragment<wmma::matrix_b, 16, 16, 8,
// //                            wmma::precision::tf32, wmma::col_major> b_frag;

// //             // 2*HD_CHUNKS k-splits of 8 each (HD=64 → 8 splits total)
// //             #pragma unroll
// //             for (int ks = 0; ks < 2 * HD_CHUNKS; ++ks) {
// //                 const int k_off = ks * 8;
// //                 wmma::load_matrix_sync(a_frag, src_sm + k_off, HD_PAD);
// //                 wmma::load_matrix_sync(b_frag, kv_sm  + k_off, HD_PAD);
// //                 wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
// //             }
// //             wmma::store_matrix_sync(dst_sm, acc_frag, BKN_PAD, wmma::mem_row_major);
// //         }
// //         __syncthreads();  // sync(c1): ds_qd (=S_sm) and DPV_sm fully written

// //         // Post-process: 256 threads × 1 element — full coverage of [BM×BN]
// //         {
// //             const int qi_local  = threadIdx.x / BlockN;   // [0, BM_WMMA)
// //             const int j_local   = threadIdx.x % BlockN;   // [0, BlockN)
// //             const int qi_global = q_tile_start + qi_local;
// //             const int j_global  = kv_base + j_local;

// //             const float raw_s = ds_qd [qi_local * BKN_PAD + j_local];
// //             const float dpv   = DPV_sm[qi_local * BKN_PAD + j_local];
// //             const float L     = LSE_sm[qi_local];
// //             const float D_    = D_sm  [qi_local];

// //             const bool qi_valid  = (qi_global < params.T);
// //             const bool j_valid   = (j_local   < kv_tile_size);
// //             const bool causal_ok = !Causal || (j_global <= qi_global);

// //             float p = 0.f;
// //             if (qi_valid && j_valid && causal_ok)
// //                 p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));

// //             const float ds = p * (dpv - D_) * params.scale;

// //             ds_qd[qi_local * BKN_PAD + j_local]  = ds;   // overwrite S with ds
// //             ds_kd[j_local  * BM_PAD  + qi_local] = ds;
// //             p_kd [j_local  * BM_PAD  + qi_local] = p;
// //         }
// //         __syncthreads();  // sync(c2): ds_qd, ds_kd, p_kd fully written

// //         // ── Phase B — sub-phase 1 (parallel across warp groups) ──────────────
// //         //
// //         //   Warps 0..HD_CHUNKS-1 : dQ WMMA — acc dq_frag (persistent register)
// //         //     A = ds_qd [BM×BN],  B = Ks[:,chunk*16..+16]
// //         //
// //         //   Warps HD_CHUNKS..2*HD-1 : dK WMMA — write [BN×16] to tile_st
// //         //     A = ds_kd [BN×BM],  B = Q_sm[:,chunk*16..+16]
// //         //
// //         {
// //             wmma::fragment<wmma::matrix_a, 16, 16, 8,
// //                            wmma::precision::tf32, wmma::row_major> a_frag;
// //             wmma::fragment<wmma::matrix_b, 16, 16, 8,
// //                            wmma::precision::tf32, wmma::row_major> b_frag;

// //             if (warp_id < HD_CHUNKS) {
// //                 // dQ: A=ds_qd[BM×BN], B=Ks[BN×16], C=dq_frag[BM×16]
// //                 const float* b_ptr = Ks + chunk * 16;
// //                 wmma::load_matrix_sync(a_frag, ds_qd,              BKN_PAD);
// //                 wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
// //                 wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
// //                 wmma::load_matrix_sync(a_frag, ds_qd + 8,          BKN_PAD);
// //                 wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
// //                 wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
// //             } else {
// //                 // dK: A=ds_kd[BN×BM], B=Q_sm[BM×16], C→tile_st[BN×16]
// //                 wmma::fragment<wmma::accumulator, 16, 16, 8, float> dk_frag;
// //                 wmma::fill_fragment(dk_frag, 0.0f);
// //                 const float* b_ptr = Q_sm + chunk * 16;
// //                 wmma::load_matrix_sync(a_frag, ds_kd,              BM_PAD);
// //                 wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
// //                 wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
// //                 wmma::load_matrix_sync(a_frag, ds_kd + 8,          BM_PAD);
// //                 wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
// //                 wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
// //                 wmma::store_matrix_sync(tile_st + chunk * 16, dk_frag,
// //                                         HeadDim, wmma::mem_row_major);
// //             }
// //         }
// //         __syncthreads();  // sync(d): tile_st fully written with dK contribution

// //         // All 8 warps atomicAdd tile_st → global dK (coalesced per row)
// //         for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
// //             const int r = idx / HeadDim, k = idx % HeadDim;
// //             atomicAdd(&dK_bh[(kv_base + r) * dk_sM + k], tile_st[r * HeadDim + k]);
// //         }
// //         __syncthreads();  // sync(e): all tile_st reads done; safe to overwrite for dV

// //         // ── Phase B — sub-phase 2: dV WMMA (warps HD_CHUNKS..2*HD-1) ─────────
// //         //   A = p_kd [BN×BM],  B = dO_sm [BM×16], C → tile_st [BN×16]
// //         if (warp_id >= HD_CHUNKS) {
// //             wmma::fragment<wmma::matrix_a, 16, 16, 8,
// //                            wmma::precision::tf32, wmma::row_major> a_frag;
// //             wmma::fragment<wmma::matrix_b, 16, 16, 8,
// //                            wmma::precision::tf32, wmma::row_major> b_frag;
// //             wmma::fragment<wmma::accumulator, 16, 16, 8, float> dv_frag;
// //             wmma::fill_fragment(dv_frag, 0.0f);

// //             const float* b_ptr = dO_sm + chunk * 16;
// //             wmma::load_matrix_sync(a_frag, p_kd,              BM_PAD);
// //             wmma::load_matrix_sync(b_frag, b_ptr,             HD_PAD);
// //             wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
// //             wmma::load_matrix_sync(a_frag, p_kd + 8,          BM_PAD);
// //             wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
// //             wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
// //             wmma::store_matrix_sync(tile_st + chunk * 16, dv_frag,
// //                                     HeadDim, wmma::mem_row_major);
// //         }
// //         __syncthreads();  // sync(f): tile_st has dV contribution

// //         // All 8 warps atomicAdd tile_st → global dV (coalesced per row)
// //         for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
// //             const int r = idx / HeadDim, k = idx % HeadDim;
// //             atomicAdd(&dV_bh[(kv_base + r) * dv_sM + k], tile_st[r * HeadDim + k]);
// //         }
// //         // sync(a) of next iteration ensures dV atomicAdd reads of tile_st
// //         // are complete before tile_st is overwritten by the next dK WMMA.
// //     }

// //     // ── Final store: dq_frag → tile_st smem → global dQ (no atomicAdd) ───────
// //     __syncthreads();  // wait for last dV atomicAdd reads before reusing tile_st

// //     if (warp_id < HD_CHUNKS) {
// //         wmma::store_matrix_sync(tile_st + chunk * 16, dq_frag,
// //                                 HeadDim, wmma::mem_row_major);
// //     }
// //     __syncthreads();

// //     for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
// //         const int r = idx / HeadDim, k = idx % HeadDim;
// //         dQ_bh[(q_tile_start + r) * dq_sM + k] = tile_st[r * HeadDim + k];
// //     }
// // }

// // // ============================================================================
// // // exp12 (SM89 Ada Lovelace) lives in arch/AttentionBackward_sm89.cu.
// // // Forward-declare the entry point for the arch-dispatch below.
// // // ============================================================================
// // void mem_efficient_attn_backward_sm89_cuda(
// //     const float* query,       int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
// //     const float* key,         int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
// //     const float* value,       int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
// //     const float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
// //     const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
// //     const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
// //     float* grad_query,        int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
// //     float* grad_key,          int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
// //     float* grad_value,        int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
// //     float* D_buf,             int64_t d_strideB, int64_t d_strideH,
// //     int64_t B, int64_t nh, int64_t T, int64_t hd,
// //     bool is_causal,
// //     bool skip_grad_zero = false);

// // // ============================================================================
// // // Public API (namespace cuda)
// // // ============================================================================

// // namespace cuda {

// // void compute_row_lse(
// //     const float* scores, float* lse,
// //     int64_t batch, int64_t T, bool is_causal)
// // {
// //     int64_t total_rows = batch * T;
// //     int threads = 32;
// //     while (threads < T && threads < 1024) threads <<= 1;
// //     size_t smem = threads * sizeof(float);
// //     compute_row_lse_kernel<<<(int)total_rows, threads, smem>>>(
// //         scores, lse, T, is_causal);
// // }

// // void mem_efficient_attn_backward(
// //     const float* query,       int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
// //     const float* key,         int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
// //     const float* value,       int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
// //     const float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
// //     const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
// //     const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
// //     float* grad_query,        int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
// //     float* grad_key,          int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
// //     float* grad_value,        int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
// //     float* D_buf,             int64_t d_strideB, int64_t d_strideH,
// //     int64_t B, int64_t nh, int64_t T, int64_t hd,
// //     bool is_causal,
// //     bool skip_grad_zero)
// // {
// //     float scale = 1.0f / sqrtf(static_cast<float>(hd));
// //     const int BH = (int)(B * nh);
// //     dim3 block_cfg(BWD_NUM_THREADS);

// //     dim3 grid_D(((int)T + (BWD_BLOCK_M_D * 2) - 1) / (BWD_BLOCK_M_D * 2), BH);

// //     MemEfficientBwdParams params;
// //     params.Q     = query;
// //     params.K     = key;
// //     params.V     = value;
// //     params.O     = output;
// //     params.dO    = grad_output;
// //     params.LSE   = lse;
// //     params.D     = D_buf;
// //     params.dQ    = grad_query;
// //     params.dK    = grad_key;
// //     params.dV    = grad_value;
// //     params.B     = (int)B;
// //     params.nh    = (int)nh;
// //     params.T     = (int)T;
// //     params.scale = scale;
// //     params.is_causal = is_causal;
// //     params.q_strideB  = q_strideB;  params.q_strideM  = q_strideM;  params.q_strideH  = q_strideH;
// //     params.k_strideB  = k_strideB;  params.k_strideM  = k_strideM;  params.k_strideH  = k_strideH;
// //     params.v_strideB  = v_strideB;  params.v_strideM  = v_strideM;  params.v_strideH  = v_strideH;
// //     params.o_strideB  = o_strideB;  params.o_strideM  = o_strideM;  params.o_strideH  = o_strideH;
// //     params.do_strideB = do_strideB; params.do_strideM = do_strideM; params.do_strideH = do_strideH;
// //     params.dq_strideB = dq_strideB; params.dq_strideM = dq_strideM; params.dq_strideH = dq_strideH;
// //     params.dk_strideB = dk_strideB; params.dk_strideM = dk_strideM; params.dk_strideH = dk_strideH;
// //     params.dv_strideB = dv_strideB; params.dv_strideM = dv_strideM; params.dv_strideH = dv_strideH;
// //     params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
// //     params.d_strideB   = d_strideB;   params.d_strideH   = d_strideH;

// //     // exp7: scalar KV-outer fallback for HeadDims not divisible by 16
// //     // exp7 is the only path that atomicAdds into dQ, so it must zero grad_query here.
// // #define LAUNCH_MEM_BWD_EXP7(HD) \
// //     do { \
// //         const int block_n7 = ((HD) < 64) ? 16 : (1024 / (HD)); \
// //         const size_t shmem_exp7 = 2ULL * block_n7 * ((HD) + 1) * sizeof(float); \
// //         const int kv_tiles7 = ((int)T + block_n7 - 1) / block_n7; \
// //         dim3 grid_bwd7(kv_tiles7, BH); \
// //         if (!skip_grad_zero) { \
// //             cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
// //         } \
// //         mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
// //         if (is_causal) { \
// //             mem_efficient_bwd_unified_kernel_exp7<HD, true> \
// //                 <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
// //         } else { \
// //             mem_efficient_bwd_unified_kernel_exp7<HD, false> \
// //                 <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
// //         } \
// //     } while (0)

// //     // exp11: Q-tile-centric, TF32 WMMA, dQ atomic-free, HD%16==0 only
// // #define LAUNCH_MEM_BWD_EXP11(HD) \
// //     do { \
// //         constexpr int BN11 = 16, BM11 = 16; \
// //         const size_t shmem11 = \
// //             (2ULL * BM11 * ((HD) + 1)   /* Q_sm + dO_sm   */ \
// //            + 2ULL * BN11 * ((HD) + 1)   /* Ks + Vs        */ \
// //            + 2ULL * BM11 * ((BN11) + 1) /* ds_qd + DPV_sm */ \
// //            + 2ULL * BN11 * ((BM11) + 1) /* ds_kd + p_kd   */ \
// //            + 1ULL * BN11 * (HD)         /* tile_st        */ \
// //            + 2ULL * BM11                /* LSE_sm + D_sm  */ \
// //             ) * sizeof(float); \
// //         const int q11 = ((int)T + BM11 - 1) / BM11; \
// //         dim3 grid_q11(q11, BH); \
// //         cudaFuncSetAttribute( \
// //             mem_efficient_bwd_unified_kernel_exp11<HD, false>, \
// //             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem11); \
// //         if (!skip_grad_zero) { \
// //             cudaMemsetAsync(params.dK, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
// //             cudaMemsetAsync(params.dV, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
// //         } \
// //         mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
// //         if (is_causal) { \
// //             mem_efficient_bwd_unified_kernel_exp11<HD, true> \
// //                 <<<grid_q11, block_cfg, shmem11>>>(params); \
// //         } else { \
// //             mem_efficient_bwd_unified_kernel_exp11<HD, false> \
// //                 <<<grid_q11, block_cfg, shmem11>>>(params); \
// //         } \
// //     } while (0)

// //     // SM89 (Ada Lovelace): dispatch to exp12 (BM=32, bank-conflict-free, 2 blocks/SM).
// //     // Non-%16 HD falls through to exp7 below regardless of arch.
// //     if (hd % 16 == 0 && cuda::get_arch() == cuda::ArchFamily::Ada) {
// //     // if (false) {
// //         mem_efficient_attn_backward_sm89_cuda(
// //             query,       q_strideB, q_strideM, q_strideH,
// //             key,         k_strideB, k_strideM, k_strideH,
// //             value,       v_strideB, v_strideM, v_strideH,
// //             output,      o_strideB, o_strideM, o_strideH,
// //             grad_output, do_strideB, do_strideM, do_strideH,
// //             lse,         lse_strideB, lse_strideH,
// //             grad_query,  dq_strideB, dq_strideM, dq_strideH,
// //             grad_key,    dk_strideB, dk_strideM, dk_strideH,
// //             grad_value,  dv_strideB, dv_strideM, dv_strideH,
// //             D_buf,       d_strideB, d_strideH,
// //             B, nh, T, hd, is_causal, skip_grad_zero);
// //         return;
// //     }

// //     switch ((int)hd) {
// //         case   8: LAUNCH_MEM_BWD_EXP7(  8); break;
// //         case  16: LAUNCH_MEM_BWD_EXP11( 16); break;
// //         case  24: LAUNCH_MEM_BWD_EXP7( 24); break;
// //         case  32: LAUNCH_MEM_BWD_EXP11( 32); break;
// //         case  40: LAUNCH_MEM_BWD_EXP7( 40); break;
// //         case  48: LAUNCH_MEM_BWD_EXP11( 48); break;
// //         case  56: LAUNCH_MEM_BWD_EXP7( 56); break;
// //         case  64: LAUNCH_MEM_BWD_EXP11( 64); break;
// //         case  80: LAUNCH_MEM_BWD_EXP11( 80); break;
// //         case  96: LAUNCH_MEM_BWD_EXP11( 96); break;
// //         case 128: LAUNCH_MEM_BWD_EXP11(128); break;
// //         case 160: LAUNCH_MEM_BWD_EXP11(160); break;
// //         case 192: LAUNCH_MEM_BWD_EXP11(192); break;
// //         case 256: LAUNCH_MEM_BWD_EXP11(256); break;
// //         default:
// //             printf("mem_efficient_attn_backward: unsupported head_dim %d\n", (int)hd);
// //             break;
// //     }
// // #undef LAUNCH_MEM_BWD_EXP7
// // #undef LAUNCH_MEM_BWD_EXP11
// // }

// // } // namespace cuda
// // } // namespace OwnTensor


































// #include "ops/cuda/attention/AttentionCommon.cuh"
// #include "ops/helpers/AttentionKernels.h"
// #include "ops/helpers/KernelDispatch.h"
// #include "autograd/backward/AttentionBackward.h"

// namespace OwnTensor{

//     // ============================================================================
// // LSE Kernel
// // ============================================================================

// __global__ void compute_row_lse_kernel(
//     const float* __restrict__ scores,
//     float* __restrict__ lse,
//     int64_t T,
//     bool is_causal)
// {
//     const int64_t row = blockIdx.x;
//     const int64_t qi  = row % T;
//     const float* row_ptr = scores + row * T;
//     const int tid  = threadIdx.x;
//     const int bdim = blockDim.x;

//     extern __shared__ float smem[];

//     float local_max = -INFINITY;
//     for (int j = tid; j < T; j += bdim) {
//         float val = row_ptr[j];
//         if (is_causal && j > qi) val = -INFINITY;
//         local_max = fmaxf(local_max, val);
//     }
//     smem[tid] = local_max;
//     __syncthreads();
//     for (int s = bdim / 2; s > 0; s >>= 1) {
//         if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
//         __syncthreads();
//     }
//     float row_max = smem[0];

//     float local_sum = 0.0f;
//     for (int j = tid; j < T; j += bdim) {
//         float val = row_ptr[j];
//         if (is_causal && j > qi) val = -INFINITY;
//         local_sum += expf(val - row_max);
//     }
//     smem[tid] = local_sum;
//     __syncthreads();
//     for (int s = bdim / 2; s > 0; s >>= 1) {
//         if (tid < s) smem[tid] += smem[tid + s];
//         __syncthreads();
//     }
//     float row_sum = smem[0];

//     if (tid == 0)
//         lse[row] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
// }


// // ============================================================================
// // Backward Pass
// // ============================================================================

// static constexpr int BWD_BLOCK_M   = 8;
// static constexpr int BWD_WARP_SZ   = 32;
// static constexpr int BWD_NUM_THREADS = BWD_BLOCK_M * BWD_WARP_SZ;  // 256
// static constexpr int BWD_BLOCK_M_D = 8; 

// __inline__ __device__ float bwd_warp_sum(float val) {
//     #pragma unroll
//     for (int offset = BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
//         val += __shfl_xor_sync(0xffffffff, val, offset);
//     return val;
// }


// template<int HeadDim>
// __global__ void mem_efficient_bwd_precompute_D(MemEfficientBwdParams params)
// {
//     const float* __restrict__ dO = params.dO;
//     const float* __restrict__ O  = params.O;
//     float*       __restrict__ D  = params.D;
//     const int T                  = (int)params.T;
//     const int nh                 = params.nh;

//     constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;

//     const int bh      = blockIdx.y;
//     const int warp_id = threadIdx.x / BWD_WARP_SZ;
//     const int lane_id = threadIdx.x % BWD_WARP_SZ;
//     const int b       = bh / nh;
//     const int h       = bh - b * nh;

//     // OPT-B + OPT-C: each warp covers 2 rows; block covers BWD_BLOCK_M_D*2 rows.
//     // row0 and row1 are the two global row indices this warp processes.
//     const int base_row = blockIdx.x * (BWD_BLOCK_M_D * 2) + warp_id * 2;
//     const int row0     = base_row;
//     const int row1     = base_row + 1;
//     const bool v0      = (row0 < T);
//     const bool v1      = (row1 < T);

//     // Strided base pointers for this (b, h) pair; head-dim stride is always 1.
//     const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
//     const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
//     const long long dO_off0 = (long long)row0 * params.do_strideM;
//     const long long dO_off1 = (long long)row1 * params.do_strideM;
//     const long long O_off0  = (long long)row0 * params.o_strideM;
//     const long long O_off1  = (long long)row1 * params.o_strideM;

//     // OPT-B: two independent accumulators — no dependency between them.
//     // The compiler sees sum0 and sum1 as entirely separate register chains
//     // and can schedule their FMAs in an interleaved fashion, keeping the
//     // FP32 pipelines fed while waiting on memory.
//     float sum0 = 0.f, sum1 = 0.f;

//     // OPT-D: __ldg() on all read-only inputs.
//     // dO and O are marked __restrict__ (no aliasing), but __ldg additionally
//     // routes the load through the read-only data cache on Ampere, improving
//     // L2 hit rate when the same cache lines are reused by exp9.
//     #pragma unroll
//     for (int i = 0; i < LocalN; ++i) {
//         const int k = lane_id + i * BWD_WARP_SZ;
//         if (k < HeadDim) {
//             // OPT-B: both rows computed in the same loop body.
//             // The two load pairs (dO/O for row0 and row1) are independent
//             // → the memory subsystem can issue them concurrently.
//             if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);  // OPT-D
//             if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);  // OPT-D
//         }
//     }

//     // OPT-B: two back-to-back warp_sum calls with no data dependency.
//     // PTX scheduler pipelines the two shfl_xor sequences (5 rounds each),
//     // filling the ~20-cycle stall slots that dropped fadd utilisation to 55%.
//     sum0 = bwd_warp_sum(sum0);
//     sum1 = bwd_warp_sum(sum1);

//     // Write results — only lane 0 holds the fully-reduced scalar.
//     if (lane_id == 0) {
//         float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
//         if (v0) D_bh[row0] = sum0;
//         if (v1) D_bh[row1] = sum1;
//     }
// }
// // ============================================================================
// // Exp7: scalar KV-outer backward, fallback for non-multiple-of-16 HeadDims
// // ============================================================================

// template<int HeadDim, bool Causal>
// __global__ void mem_efficient_bwd_unified_kernel_exp7(MemEfficientBwdParams params)
// {
//     constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
//     constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
//     constexpr int HD_PAD  = HeadDim + 1;
//     constexpr float BWD_LOG2E = 1.4426950408889634074f;

//     // smem: ONLY K and V tiles — same as exp6
//     extern __shared__ float smem[];
//     float* Ks = smem;
//     float* Vs = Ks + BlockN * HD_PAD;

//     const int bh         = blockIdx.y;
//     const int kv_tile    = blockIdx.x;
//     const int tile_start = kv_tile * BlockN;
//     const int tile_size  = min(BlockN, params.T - tile_start);
//     if (tile_start >= params.T) return;

//     const int warp_id = threadIdx.x / BWD_WARP_SZ;
//     const int lane_id = threadIdx.x % BWD_WARP_SZ;
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

//     // Step 0: zero this block's dK/dV rows in global memory (scalar, coalesced)
//     // Scalar avoids any float4 alignment edge cases; still fully coalesced.
//     for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim;
//         const int k = idx % HeadDim;
//         dK_bh[(tile_start + r) * dk_sM + k] = 0.f;
//         dV_bh[(tile_start + r) * dv_sM + k] = 0.f;
//     }

//     // Bank-conflict-free K/V smem load:
//     // Thread idx → row r = idx/HeadDim, col k = idx%HeadDim.
//     // Same-warp threads differ in k (consecutive) → consecutive smem banks → no conflict.
//     // (float4 had stride d4*4 within row → threads t and t+8 collided on the same banks.)
//     for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
//         const int r     = idx / HeadDim;
//         const int k     = idx % HeadDim;
//         const int g_row = tile_start + r;
//         Ks[r * HD_PAD + k] = (g_row < params.T) ? K_bh[g_row * k_sM + k] : 0.f;
//         Vs[r * HD_PAD + k] = (g_row < params.T) ? V_bh[g_row * v_sM + k] : 0.f;
//     }

//     // Barrier: K/V loaded, dK/dV zeroed — safe to proceed
//     __syncthreads();

//     const int q_start = Causal ? tile_start : 0;

//     for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
//         const int qi     = q_base + warp_id;
//         const bool valid = (qi < params.T);

//         float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
//         float L_qi = 0.0f, D_qi = 0.0f;

//         #pragma unroll
//         for (int i = 0; i < LocalN; ++i) {
//             const int k = lane_id + i * BWD_WARP_SZ;
//             float qv = 0.0f, dov = 0.0f;
//             if (valid && k < HeadDim) {
//                 qv  = Q_bh [qi * q_sM  + k];
//                 dov = dO_bh[qi * do_sM + k];
//             }
//             q_local[i]  = qv;
//             do_local[i] = dov;
//             dq_local[i] = 0.0f;
//         }
//         if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

//         // Fused Phase A+B: same as exp6 — merged i-loop, global atomics for dK/dV.
//         // Smem reads (Ks/Vs) are bank-conflict-free here since k varies within a warp.
//         #pragma unroll 4
//         for (int j = 0; j < BlockN; ++j) {
//             const bool j_valid = (j < tile_size);

//             float dot_qk = 0.0f, dot_dov = 0.0f;
//             #pragma unroll
//             for (int i = 0; i < LocalN; ++i) {
//                 const int k = lane_id + i * BWD_WARP_SZ;
//                 if (k < HeadDim) {
//                     dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
//                     dot_dov += do_local[i] * Vs[j * HD_PAD + k];
//                 }
//             }

//             const float s   = bwd_warp_sum(dot_qk) * params.scale;
//             const float dpv = bwd_warp_sum(dot_dov);

//             float p = 0.0f;
//             if (j_valid && valid && !(Causal && (tile_start + j) > qi))
//                 p = exp2f(BWD_LOG2E * (s - L_qi));

//             const float ds = p * (dpv - D_qi);

//             #pragma unroll
//             for (int i = 0; i < LocalN; ++i) {
//                 const int k = lane_id + i * BWD_WARP_SZ;
//                 if (k < HeadDim) {
//                     dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
//                     atomicAdd(&dK_bh[(tile_start + j) * dk_sM + k],
//                               ds * params.scale * q_local[i]);
//                     atomicAdd(&dV_bh[(tile_start + j) * dv_sM + k],
//                               p  *                do_local[i]);
//                 }
//             }
//         }

//         if (valid) {
//             #pragma unroll
//             for (int i = 0; i < LocalN; ++i) {
//                 const int k = lane_id + i * BWD_WARP_SZ;
//                 if (k < HeadDim)
//                     atomicAdd(&dQ_bh[qi * dq_sM + k], dq_local[i]);
//             }
//         }
//     } // q_base loop
// }
// // ============================================================================
// // Exp11: Q-tile-centric backward, TF32 WMMA, dQ atomic-free (HD%16==0 only)
// // ============================================================================

// template <int HeadDim, bool Causal>
// __global__ void mem_efficient_bwd_unified_kernel_exp11(MemEfficientBwdParams params)
// {
//     using namespace nvcuda;

//     constexpr int BlockN    = 16;
//     constexpr int BM_WMMA   = 16;
//     constexpr int HD_CHUNKS = HeadDim / 16;
//     constexpr int HD_PAD    = HeadDim;        // WMMA TF32 requires LDM to be multiple of 4 floats (16 bytes)
//     constexpr int BKN_PAD   = BlockN;         // WMMA TF32 LDM constraint
//     constexpr int BM_PAD    = BM_WMMA;        // WMMA TF32 LDM constraint
//     constexpr float BWD_LOG2E = 1.4426950408889634074f;

//     extern __shared__ float smem_f[];
//     float* Q_sm    = smem_f;
//     float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
//     float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
//     float* Vs      = Ks      + BlockN   * HD_PAD;
//     float* ds_qd   = Vs      + BlockN   * HD_PAD;   // [BM×BKN_PAD] — S_sm / dQ WMMA A
//     float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;  // [BM×BKN_PAD] — raw dO·V^T
//     float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;  // [BN×BM_PAD]  — dK WMMA A
//     float* p_kd    = ds_kd   + BlockN   * BM_PAD;   // [BN×BM_PAD]  — dV WMMA A
//     float* tile_st = p_kd    + BlockN   * BM_PAD;   // [BN×HD] per-iter dK/dV + final dQ
//     float* LSE_sm  = tile_st + BlockN   * HeadDim;  // [BM_WMMA] per-block LSE
//     float* D_sm    = LSE_sm  + BM_WMMA;             // [BM_WMMA] per-block D

//     const int bh           = blockIdx.y;
//     const int q_tile       = blockIdx.x;
//     const int q_tile_start = q_tile * BM_WMMA;
//     const int tile_size    = min(BM_WMMA, params.T - q_tile_start);
//     if (q_tile_start >= params.T) return;

//     const int warp_id = threadIdx.x / BWD_WARP_SZ;
//     const int chunk   = warp_id % HD_CHUNKS;  // column chunk of HeadDim [0, HD_CHUNKS)
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

//     // ── Step 0: load Q, dO, LSE, D for this Q-tile into smem (persistent) ────
//     for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
//         const int r  = idx / HeadDim, k = idx % HeadDim;
//         const int qi = q_tile_start + r;
//         const bool vq = (qi < params.T);
//         Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
//         dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
//     }
//     if (threadIdx.x < BM_WMMA) {
//         const int qi  = q_tile_start + threadIdx.x;
//         const bool vq = (qi < params.T);
//         LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
//         D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
//     }
//     __syncthreads();

//     // ── Persistent TF32 WMMA accumulator for dQ ──────────────────────────────
//     // Only warps 0..HD_CHUNKS-1 use this; others keep it at zero (never stored).
//     wmma::fragment<wmma::accumulator, 16, 16, 8, float> dq_frag;
//     wmma::fill_fragment(dq_frag, 0.0f);

//     // Causal: skip KV tiles entirely past the Q-tile's last row
//     const int kv_loop_end = Causal ? (q_tile_start + BM_WMMA) : params.T;

//     // ── Main KV loop ──────────────────────────────────────────────────────────
//     for (int kv_base = 0; kv_base < kv_loop_end; kv_base += BlockN) {

//         // sync(a): prior dV atomicAdd reads of tile_st are done; prior Phase B
//         //          reads of Ks are done — safe to overwrite both.
//         __syncthreads();

//         const int kv_tile_size = min(BlockN, params.T - kv_base);

//         // Load K/V tile for this KV iteration
//         for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             const int g = kv_base + r;
//             Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
//             Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
//         }
//         __syncthreads();  // sync(b): Ks/Vs ready

//         // ── Phase A: TF32 WMMA GEMMs + scalar post-process ───────────────────
//         //
//         // Warp 0: S   = Q_sm × K^T  → ds_qd  [BM×BKN_PAD]   (col_major B = K^T)
//         // Warp 1: DPV = dO_sm × V^T → DPV_sm [BM×BKN_PAD]   (col_major B = V^T)
//         // Warps 2..7 are idle during WMMA.
//         //
//         // After sync(c1), all 256 threads run the scalar post-process in parallel
//         // (1 [qi, j] element each): compute p and ds from S, DPV, LSE_sm, D_sm,
//         // then write ds_qd (overwrite S with ds), ds_kd, and p_kd.
//         if (warp_id < 2) {
//             const float* src_sm = (warp_id == 0) ? Q_sm   : dO_sm;
//             const float* kv_sm  = (warp_id == 0) ? Ks     : Vs;
//             float*       dst_sm = (warp_id == 0) ? ds_qd  : DPV_sm;

//             wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
//             wmma::fill_fragment(acc_frag, 0.0f);

//             // col_major B fragment loads K^T (or V^T): b[k][j] = kv_sm[j*HD_PAD + k_off + k]
//             wmma::fragment<wmma::matrix_a, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> a_frag;
//             wmma::fragment<wmma::matrix_b, 16, 16, 8,
//                            wmma::precision::tf32, wmma::col_major> b_frag;

//             // 2*HD_CHUNKS k-splits of 8 each (HD=64 → 8 splits total)
//             #pragma unroll
//             for (int ks = 0; ks < 2 * HD_CHUNKS; ++ks) {
//                 const int k_off = ks * 8;
//                 wmma::load_matrix_sync(a_frag, src_sm + k_off, HD_PAD);
//                 wmma::load_matrix_sync(b_frag, kv_sm  + k_off, HD_PAD);
//                 wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//             }
//             wmma::store_matrix_sync(dst_sm, acc_frag, BKN_PAD, wmma::mem_row_major);
//         }
//         __syncthreads();  // sync(c1): ds_qd (=S_sm) and DPV_sm fully written

//         // Post-process: 256 threads × 1 element — full coverage of [BM×BN]
//         {
//             const int qi_local  = threadIdx.x / BlockN;   // [0, BM_WMMA)
//             const int j_local   = threadIdx.x % BlockN;   // [0, BlockN)
//             const int qi_global = q_tile_start + qi_local;
//             const int j_global  = kv_base + j_local;

//             const float raw_s = ds_qd [qi_local * BKN_PAD + j_local];
//             const float dpv   = DPV_sm[qi_local * BKN_PAD + j_local];
//             const float L     = LSE_sm[qi_local];
//             const float D_    = D_sm  [qi_local];

//             const bool qi_valid  = (qi_global < params.T);
//             const bool j_valid   = (j_local   < kv_tile_size);
//             const bool causal_ok = !Causal || (j_global <= qi_global);

//             float p = 0.f;
//             if (qi_valid && j_valid && causal_ok)
//                 p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));

//             const float ds = p * (dpv - D_) * params.scale;

//             ds_qd[qi_local * BKN_PAD + j_local]  = ds;   // overwrite S with ds
//             ds_kd[j_local  * BM_PAD  + qi_local] = ds;
//             p_kd [j_local  * BM_PAD  + qi_local] = p;
//         }
//         __syncthreads();  // sync(c2): ds_qd, ds_kd, p_kd fully written

//         // ── Phase B — sub-phase 1 (parallel across warp groups) ──────────────
//         //
//         //   Warps 0..HD_CHUNKS-1 : dQ WMMA — acc dq_frag (persistent register)
//         //     A = ds_qd [BM×BN],  B = Ks[:,chunk*16..+16]
//         //
//         //   Warps HD_CHUNKS..2*HD-1 : dK WMMA — write [BN×16] to tile_st
//         //     A = ds_kd [BN×BM],  B = Q_sm[:,chunk*16..+16]
//         //
//         {
//             wmma::fragment<wmma::matrix_a, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> a_frag;
//             wmma::fragment<wmma::matrix_b, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> b_frag;

//             if (warp_id < HD_CHUNKS) {
//                 // dQ: A=ds_qd[BM×BN], B=Ks[BN×16], C=dq_frag[BM×16]
//                 const float* b_ptr = Ks + chunk * 16;
//                 wmma::load_matrix_sync(a_frag, ds_qd,              BKN_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
//                 wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
//                 wmma::load_matrix_sync(a_frag, ds_qd + 8,          BKN_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
//                 wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
//             } else {
//                 // dK: A=ds_kd[BN×BM], B=Q_sm[BM×16], C→tile_st[BN×16]
//                 wmma::fragment<wmma::accumulator, 16, 16, 8, float> dk_frag;
//                 wmma::fill_fragment(dk_frag, 0.0f);
//                 const float* b_ptr = Q_sm + chunk * 16;
//                 wmma::load_matrix_sync(a_frag, ds_kd,              BM_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
//                 wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
//                 wmma::load_matrix_sync(a_frag, ds_kd + 8,          BM_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
//                 wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
//                 wmma::store_matrix_sync(tile_st + chunk * 16, dk_frag,
//                                         HeadDim, wmma::mem_row_major);
//             }
//         }
//         __syncthreads();  // sync(d): tile_st fully written with dK contribution

//         // All 8 warps atomicAdd tile_st → global dK (coalesced per row)
//         for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             atomicAdd(&dK_bh[(kv_base + r) * dk_sM + k], tile_st[r * HeadDim + k]);
//         }
//         __syncthreads();  // sync(e): all tile_st reads done; safe to overwrite for dV

//         // ── Phase B — sub-phase 2: dV WMMA (warps HD_CHUNKS..2*HD-1) ─────────
//         //   A = p_kd [BN×BM],  B = dO_sm [BM×16], C → tile_st [BN×16]
//         if (warp_id >= HD_CHUNKS) {
//             wmma::fragment<wmma::matrix_a, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> a_frag;
//             wmma::fragment<wmma::matrix_b, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> b_frag;
//             wmma::fragment<wmma::accumulator, 16, 16, 8, float> dv_frag;
//             wmma::fill_fragment(dv_frag, 0.0f);

//             const float* b_ptr = dO_sm + chunk * 16;
//             wmma::load_matrix_sync(a_frag, p_kd,              BM_PAD);
//             wmma::load_matrix_sync(b_frag, b_ptr,             HD_PAD);
//             wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
//             wmma::load_matrix_sync(a_frag, p_kd + 8,          BM_PAD);
//             wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
//             wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
//             wmma::store_matrix_sync(tile_st + chunk * 16, dv_frag,
//                                     HeadDim, wmma::mem_row_major);
//         }
//         __syncthreads();  // sync(f): tile_st has dV contribution

//         // All 8 warps atomicAdd tile_st → global dV (coalesced per row)
//         for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             atomicAdd(&dV_bh[(kv_base + r) * dv_sM + k], tile_st[r * HeadDim + k]);
//         }
//         // sync(a) of next iteration ensures dV atomicAdd reads of tile_st
//         // are complete before tile_st is overwritten by the next dK WMMA.
//     }

//     // ── Final store: dq_frag → tile_st smem → global dQ (no atomicAdd) ───────
//     __syncthreads();  // wait for last dV atomicAdd reads before reusing tile_st

//     if (warp_id < HD_CHUNKS) {
//         wmma::store_matrix_sync(tile_st + chunk * 16, dq_frag,
//                                 HeadDim, wmma::mem_row_major);
//     }
//     __syncthreads();

//     for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         dQ_bh[(q_tile_start + r) * dq_sM + k] = tile_st[r * HeadDim + k];
//     }
// }

// // ============================================================================
// // exp12 (SM89 Ada Lovelace) lives in arch/AttentionBackward_sm89.cu.
// // Forward-declare the entry point for the arch-dispatch below.
// // ============================================================================
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
//     bool skip_grad_zero);

// // 3xTF32 variant — same external signature, gated by USE_3XTF32=1 env var.
// // Matches PyTorch's CUTLASS kernel_backward.h OpMultiplyAddFastF32 on all 5
// // backward GEMMs (MatmulQK, MatmulDOIVJ, MatmulGradV, MatmulGradQ, MatmulGradK).
// void mem_efficient_attn_backward_sm89_3xtf32_cuda(
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
//     bool skip_grad_zero);

// // ============================================================================
// // Public API (namespace cuda)
// // ============================================================================

// namespace cuda {

// void compute_row_lse(
//     const float* scores, float* lse,
//     int64_t batch, int64_t T, bool is_causal)
// {
//     int64_t total_rows = batch * T;
//     int threads = 32;
//     while (threads < T && threads < 1024) threads <<= 1;
//     size_t smem = threads * sizeof(float);
//     compute_row_lse_kernel<<<(int)total_rows, threads, smem>>>(
//         scores, lse, T, is_causal);
// }

// void mem_efficient_attn_backward(
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
//     dim3 block_cfg(BWD_NUM_THREADS);

//     dim3 grid_D(((int)T + (BWD_BLOCK_M_D * 2) - 1) / (BWD_BLOCK_M_D * 2), BH);

//     MemEfficientBwdParams params;
//     params.Q     = query;
//     params.K     = key;
//     params.V     = value;
//     params.O     = output;
//     params.dO    = grad_output;
//     params.LSE   = lse;
//     params.D     = D_buf;
//     params.dQ    = grad_query;
//     params.dK    = grad_key;
//     params.dV    = grad_value;
//     params.B     = (int)B;
//     params.nh    = (int)nh;
//     params.T     = (int)T;
//     params.scale = scale;
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

//     // exp7: scalar KV-outer fallback for HeadDims not divisible by 16
//     // exp7 is the only path that atomicAdds into dQ, so it must zero grad_query here.
// #define LAUNCH_MEM_BWD_EXP7(HD) \
//     do { \
//         const int block_n7 = ((HD) < 64) ? 16 : (1024 / (HD)); \
//         const size_t shmem_exp7 = 2ULL * block_n7 * ((HD) + 1) * sizeof(float); \
//         const int kv_tiles7 = ((int)T + block_n7 - 1) / block_n7; \
//         dim3 grid_bwd7(kv_tiles7, BH); \
//         if (!skip_grad_zero) { \
//             cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
//         } \
//         mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
//         if (is_causal) { \
//             mem_efficient_bwd_unified_kernel_exp7<HD, true> \
//                 <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
//         } else { \
//             mem_efficient_bwd_unified_kernel_exp7<HD, false> \
//                 <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
//         } \
//     } while (0)

//     // exp11: Q-tile-centric, TF32 WMMA, dQ atomic-free, HD%16==0 only
// #define LAUNCH_MEM_BWD_EXP11(HD) \
//     do { \
//         constexpr int BN11 = 16, BM11 = 16; \
//         const size_t shmem11 = \
//             (2ULL * BM11 * ((HD) + 1)   /* Q_sm + dO_sm   */ \
//            + 2ULL * BN11 * ((HD) + 1)   /* Ks + Vs        */ \
//            + 2ULL * BM11 * ((BN11) + 1) /* ds_qd + DPV_sm */ \
//            + 2ULL * BN11 * ((BM11) + 1) /* ds_kd + p_kd   */ \
//            + 1ULL * BN11 * (HD)         /* tile_st        */ \
//            + 2ULL * BM11                /* LSE_sm + D_sm  */ \
//             ) * sizeof(float); \
//         const int q11 = ((int)T + BM11 - 1) / BM11; \
//         dim3 grid_q11(q11, BH); \
//         cudaFuncSetAttribute( \
//             mem_efficient_bwd_unified_kernel_exp11<HD, false>, \
//             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem11); \
//         if (!skip_grad_zero) { \
//             cudaMemsetAsync(params.dK, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
//             cudaMemsetAsync(params.dV, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
//         } \
//         mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
//         if (is_causal) { \
//             mem_efficient_bwd_unified_kernel_exp11<HD, true> \
//                 <<<grid_q11, block_cfg, shmem11>>>(params); \
//         } else { \
//             mem_efficient_bwd_unified_kernel_exp11<HD, false> \
//                 <<<grid_q11, block_cfg, shmem11>>>(params); \
//         } \
//     } while (0)

//     // SM89 (Ada Lovelace): dispatch to exp12 (BM=32, bank-conflict-free, 2 blocks/SM).
//     // Non-%16 HD falls through to exp7 below regardless of arch.
//     if (hd % 16 == 0 && cuda::get_arch() == cuda::ArchFamily::Ada) {
//       //if (false) {
//         // USE_3XTF32=1 routes to the CUTLASS-style OpMultiplyAddFastF32
//         // emulation kernel that mirrors PyTorch's kernel_backward.h on all
//         // 5 backward GEMMs. Smem caps it at hd <= 128 (~78 KB at HD=64 vs
//         // Ada's 99 KB opt-in limit); larger HDs or env unset fall back to
//         // the 1xTF32 path below.
//         static const bool kUse3xTF32 = []() {
//             const char* e = std::getenv("USE_3XTF32");
//             return e && e[0] == '1';
//         }();
//         if (kUse3xTF32 && hd <= 128) {
//             mem_efficient_attn_backward_sm89_3xtf32_cuda(
//                 query,       q_strideB, q_strideM, q_strideH,
//                 key,         k_strideB, k_strideM, k_strideH,
//                 value,       v_strideB, v_strideM, v_strideH,
//                 output,      o_strideB, o_strideM, o_strideH,
//                 grad_output, do_strideB, do_strideM, do_strideH,
//                 lse,         lse_strideB, lse_strideH,
//                 grad_query,  dq_strideB, dq_strideM, dq_strideH,
//                 grad_key,    dk_strideB, dk_strideM, dk_strideH,
//                 grad_value,  dv_strideB, dv_strideM, dv_strideH,
//                 D_buf,       d_strideB, d_strideH,
//                 B, nh, T, hd, is_causal, skip_grad_zero);
//             return;
//         }
//         mem_efficient_attn_backward_sm89_cuda(
//             query,       q_strideB, q_strideM, q_strideH,
//             key,         k_strideB, k_strideM, k_strideH,
//             value,       v_strideB, v_strideM, v_strideH,
//             output,      o_strideB, o_strideM, o_strideH,
//             grad_output, do_strideB, do_strideM, do_strideH,
//             lse,         lse_strideB, lse_strideH,
//             grad_query,  dq_strideB, dq_strideM, dq_strideH,
//             grad_key,    dk_strideB, dk_strideM, dk_strideH,
//             grad_value,  dv_strideB, dv_strideM, dv_strideH,
//             D_buf,       d_strideB, d_strideH,
//             B, nh, T, hd, is_causal, skip_grad_zero);
//         return;
//     }

//     switch ((int)hd) {
//         case   8: LAUNCH_MEM_BWD_EXP7(  8); break;
//         case  16: LAUNCH_MEM_BWD_EXP11( 16); break;
//         case  24: LAUNCH_MEM_BWD_EXP7( 24); break;
//         case  32: LAUNCH_MEM_BWD_EXP11( 32); break;
//         case  40: LAUNCH_MEM_BWD_EXP7( 40); break;
//         case  48: LAUNCH_MEM_BWD_EXP11( 48); break;
//         case  56: LAUNCH_MEM_BWD_EXP7( 56); break;
//         case  64: LAUNCH_MEM_BWD_EXP11( 64); break;
//         case  80: LAUNCH_MEM_BWD_EXP11( 80); break;
//         case  96: LAUNCH_MEM_BWD_EXP11( 96); break;
//         case 128: LAUNCH_MEM_BWD_EXP11(128); break;
//         case 160: LAUNCH_MEM_BWD_EXP11(160); break;
//         case 192: LAUNCH_MEM_BWD_EXP11(192); break;
//         case 256: LAUNCH_MEM_BWD_EXP11(256); break;
//         default:
//             printf("mem_efficient_attn_backward: unsupported head_dim %d\n", (int)hd);
//             break;
//     }
// #undef LAUNCH_MEM_BWD_EXP7
// #undef LAUNCH_MEM_BWD_EXP11
// }

// } // namespace cuda
// } // namespace OwnTensor

//-------------------------------------------------------------------------------------------------------

// #include "ops/cuda/attention/AttentionCommon.cuh"
// #include "ops/helpers/AttentionKernels.h"
// #include "ops/helpers/KernelDispatch.h"
// #include "autograd/backward/AttentionBackward.h"

// namespace OwnTensor{

//     // ============================================================================
// // LSE Kernel
// // ============================================================================

// __global__ void compute_row_lse_kernel(
//     const float* __restrict__ scores,
//     float* __restrict__ lse,
//     int64_t T,
//     bool is_causal)
// {
//     const int64_t row = blockIdx.x;
//     const int64_t qi  = row % T;
//     const float* row_ptr = scores + row * T;
//     const int tid  = threadIdx.x;
//     const int bdim = blockDim.x;

//     extern __shared__ float smem[];

//     float local_max = -INFINITY;
//     for (int j = tid; j < T; j += bdim) {
//         float val = row_ptr[j];
//         if (is_causal && j > qi) val = -INFINITY;
//         local_max = fmaxf(local_max, val);
//     }
//     smem[tid] = local_max;
//     __syncthreads();
//     for (int s = bdim / 2; s > 0; s >>= 1) {
//         if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
//         __syncthreads();
//     }
//     float row_max = smem[0];

//     float local_sum = 0.0f;
//     for (int j = tid; j < T; j += bdim) {
//         float val = row_ptr[j];
//         if (is_causal && j > qi) val = -INFINITY;
//         local_sum += expf(val - row_max);
//     }
//     smem[tid] = local_sum;
//     __syncthreads();
//     for (int s = bdim / 2; s > 0; s >>= 1) {
//         if (tid < s) smem[tid] += smem[tid + s];
//         __syncthreads();
//     }
//     float row_sum = smem[0];

//     if (tid == 0)
//         lse[row] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
// }


// // ============================================================================
// // Backward Pass
// // ============================================================================

// static constexpr int BWD_BLOCK_M   = 8;
// static constexpr int BWD_WARP_SZ   = 32;
// static constexpr int BWD_NUM_THREADS = BWD_BLOCK_M * BWD_WARP_SZ;  // 256
// static constexpr int BWD_BLOCK_M_D = 8; 

// __inline__ __device__ float bwd_warp_sum(float val) {
//     #pragma unroll
//     for (int offset = BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
//         val += __shfl_xor_sync(0xffffffff, val, offset);
//     return val;
// }


// template<int HeadDim>
// __global__ void mem_efficient_bwd_precompute_D(MemEfficientBwdParams params)
// {
//     const float* __restrict__ dO = params.dO;
//     const float* __restrict__ O  = params.O;
//     float*       __restrict__ D  = params.D;
//     const int T                  = (int)params.T;
//     const int nh                 = params.nh;

//     constexpr int LocalN = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;

//     const int bh      = blockIdx.y;
//     const int warp_id = threadIdx.x / BWD_WARP_SZ;
//     const int lane_id = threadIdx.x % BWD_WARP_SZ;
//     const int b       = bh / nh;
//     const int h       = bh - b * nh;

//     // OPT-B + OPT-C: each warp covers 2 rows; block covers BWD_BLOCK_M_D*2 rows.
//     // row0 and row1 are the two global row indices this warp processes.
//     const int base_row = blockIdx.x * (BWD_BLOCK_M_D * 2) + warp_id * 2;
//     const int row0     = base_row;
//     const int row1     = base_row + 1;
//     const bool v0      = (row0 < T);
//     const bool v1      = (row1 < T);

//     // Strided base pointers for this (b, h) pair; head-dim stride is always 1.
//     const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
//     const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
//     const long long dO_off0 = (long long)row0 * params.do_strideM;
//     const long long dO_off1 = (long long)row1 * params.do_strideM;
//     const long long O_off0  = (long long)row0 * params.o_strideM;
//     const long long O_off1  = (long long)row1 * params.o_strideM;

//     // OPT-B: two independent accumulators — no dependency between them.
//     // The compiler sees sum0 and sum1 as entirely separate register chains
//     // and can schedule their FMAs in an interleaved fashion, keeping the
//     // FP32 pipelines fed while waiting on memory.
//     float sum0 = 0.f, sum1 = 0.f;

//     // OPT-D: __ldg() on all read-only inputs.
//     // dO and O are marked __restrict__ (no aliasing), but __ldg additionally
//     // routes the load through the read-only data cache on Ampere, improving
//     // L2 hit rate when the same cache lines are reused by exp9.
//     #pragma unroll
//     for (int i = 0; i < LocalN; ++i) {
//         const int k = lane_id + i * BWD_WARP_SZ;
//         if (k < HeadDim) {
//             // OPT-B: both rows computed in the same loop body.
//             // The two load pairs (dO/O for row0 and row1) are independent
//             // → the memory subsystem can issue them concurrently.
//             if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);  // OPT-D
//             if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);  // OPT-D
//         }
//     }

//     // OPT-B: two back-to-back warp_sum calls with no data dependency.
//     // PTX scheduler pipelines the two shfl_xor sequences (5 rounds each),
//     // filling the ~20-cycle stall slots that dropped fadd utilisation to 55%.
//     sum0 = bwd_warp_sum(sum0);
//     sum1 = bwd_warp_sum(sum1);

//     // Write results — only lane 0 holds the fully-reduced scalar.
//     if (lane_id == 0) {
//         float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
//         if (v0) D_bh[row0] = sum0;
//         if (v1) D_bh[row1] = sum1;
//     }
// }
// // ============================================================================
// // Exp7: scalar KV-outer backward, fallback for non-multiple-of-16 HeadDims
// // ============================================================================

// template<int HeadDim, bool Causal>
// __global__ void mem_efficient_bwd_unified_kernel_exp7(MemEfficientBwdParams params)
// {
//     constexpr int BlockN  = (HeadDim < 64) ? 16 : (1024 / HeadDim);
//     constexpr int LocalN  = (HeadDim + BWD_WARP_SZ - 1) / BWD_WARP_SZ;
//     constexpr int HD_PAD  = HeadDim + 1;
//     constexpr float BWD_LOG2E = 1.4426950408889634074f;

//     // smem: ONLY K and V tiles — same as exp6
//     extern __shared__ float smem[];
//     float* Ks = smem;
//     float* Vs = Ks + BlockN * HD_PAD;

//     const int bh         = blockIdx.y;
//     const int kv_tile    = blockIdx.x;
//     const int tile_start = kv_tile * BlockN;
//     const int tile_size  = min(BlockN, params.T - tile_start);
//     if (tile_start >= params.T) return;

//     const int warp_id = threadIdx.x / BWD_WARP_SZ;
//     const int lane_id = threadIdx.x % BWD_WARP_SZ;
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

//     // Step 0: zero this block's dK/dV rows in global memory (scalar, coalesced)
//     // Scalar avoids any float4 alignment edge cases; still fully coalesced.
//     for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim;
//         const int k = idx % HeadDim;
//         dK_bh[(tile_start + r) * dk_sM + k] = 0.f;
//         dV_bh[(tile_start + r) * dv_sM + k] = 0.f;
//     }

//     // Bank-conflict-free K/V smem load:
//     // Thread idx → row r = idx/HeadDim, col k = idx%HeadDim.
//     // Same-warp threads differ in k (consecutive) → consecutive smem banks → no conflict.
//     // (float4 had stride d4*4 within row → threads t and t+8 collided on the same banks.)
//     for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
//         const int r     = idx / HeadDim;
//         const int k     = idx % HeadDim;
//         const int g_row = tile_start + r;
//         Ks[r * HD_PAD + k] = (g_row < params.T) ? K_bh[g_row * k_sM + k] : 0.f;
//         Vs[r * HD_PAD + k] = (g_row < params.T) ? V_bh[g_row * v_sM + k] : 0.f;
//     }

//     // Barrier: K/V loaded, dK/dV zeroed — safe to proceed
//     __syncthreads();

//     const int q_start = Causal ? tile_start : 0;

//     for (int q_base = q_start; q_base < params.T; q_base += BWD_BLOCK_M) {
//         const int qi     = q_base + warp_id;
//         const bool valid = (qi < params.T);

//         float q_local[LocalN], do_local[LocalN], dq_local[LocalN];
//         float L_qi = 0.0f, D_qi = 0.0f;

//         #pragma unroll
//         for (int i = 0; i < LocalN; ++i) {
//             const int k = lane_id + i * BWD_WARP_SZ;
//             float qv = 0.0f, dov = 0.0f;
//             if (valid && k < HeadDim) {
//                 qv  = Q_bh [qi * q_sM  + k];
//                 dov = dO_bh[qi * do_sM + k];
//             }
//             q_local[i]  = qv;
//             do_local[i] = dov;
//             dq_local[i] = 0.0f;
//         }
//         if (valid) { L_qi = LSE_bh[qi]; D_qi = D_bh[qi]; }

//         // Fused Phase A+B: same as exp6 — merged i-loop, global atomics for dK/dV.
//         // Smem reads (Ks/Vs) are bank-conflict-free here since k varies within a warp.
//         #pragma unroll 4
//         for (int j = 0; j < BlockN; ++j) {
//             const bool j_valid = (j < tile_size);

//             float dot_qk = 0.0f, dot_dov = 0.0f;
//             #pragma unroll
//             for (int i = 0; i < LocalN; ++i) {
//                 const int k = lane_id + i * BWD_WARP_SZ;
//                 if (k < HeadDim) {
//                     dot_qk  += q_local[i]  * Ks[j * HD_PAD + k];
//                     dot_dov += do_local[i] * Vs[j * HD_PAD + k];
//                 }
//             }

//             const float s   = bwd_warp_sum(dot_qk) * params.scale;
//             const float dpv = bwd_warp_sum(dot_dov);

//             float p = 0.0f;
//             if (j_valid && valid && !(Causal && (tile_start + j) > qi))
//                 p = exp2f(BWD_LOG2E * (s - L_qi));

//             const float ds = p * (dpv - D_qi);

//             #pragma unroll
//             for (int i = 0; i < LocalN; ++i) {
//                 const int k = lane_id + i * BWD_WARP_SZ;
//                 if (k < HeadDim) {
//                     dq_local[i] += ds * params.scale * Ks[j * HD_PAD + k];
//                     atomicAdd(&dK_bh[(tile_start + j) * dk_sM + k],
//                               ds * params.scale * q_local[i]);
//                     atomicAdd(&dV_bh[(tile_start + j) * dv_sM + k],
//                               p  *                do_local[i]);
//                 }
//             }
//         }

//         if (valid) {
//             #pragma unroll
//             for (int i = 0; i < LocalN; ++i) {
//                 const int k = lane_id + i * BWD_WARP_SZ;
//                 if (k < HeadDim)
//                     atomicAdd(&dQ_bh[qi * dq_sM + k], dq_local[i]);
//             }
//         }
//     } // q_base loop
// }
// // ============================================================================
// // Exp11: Q-tile-centric backward, TF32 WMMA, dQ atomic-free (HD%16==0 only)
// // ============================================================================

// template <int HeadDim, bool Causal>
// __global__ void mem_efficient_bwd_unified_kernel_exp11(MemEfficientBwdParams params)
// {
//     using namespace nvcuda;

//     constexpr int BlockN    = 16;
//     constexpr int BM_WMMA   = 16;
//     constexpr int HD_CHUNKS = HeadDim / 16;
//     constexpr int HD_PAD    = HeadDim;        // WMMA TF32 requires LDM to be multiple of 4 floats (16 bytes)
//     constexpr int BKN_PAD   = BlockN;         // WMMA TF32 LDM constraint
//     constexpr int BM_PAD    = BM_WMMA;        // WMMA TF32 LDM constraint
//     constexpr float BWD_LOG2E = 1.4426950408889634074f;

//     extern __shared__ float smem_f[];
//     float* Q_sm    = smem_f;
//     float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
//     float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
//     float* Vs      = Ks      + BlockN   * HD_PAD;
//     float* ds_qd   = Vs      + BlockN   * HD_PAD;   // [BM×BKN_PAD] — S_sm / dQ WMMA A
//     float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;  // [BM×BKN_PAD] — raw dO·V^T
//     float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;  // [BN×BM_PAD]  — dK WMMA A
//     float* p_kd    = ds_kd   + BlockN   * BM_PAD;   // [BN×BM_PAD]  — dV WMMA A
//     float* tile_st = p_kd    + BlockN   * BM_PAD;   // [BN×HD] per-iter dK/dV + final dQ
//     float* LSE_sm  = tile_st + BlockN   * HeadDim;  // [BM_WMMA] per-block LSE
//     float* D_sm    = LSE_sm  + BM_WMMA;             // [BM_WMMA] per-block D

//     const int bh           = blockIdx.y;
//     const int q_tile       = blockIdx.x;
//     const int q_tile_start = q_tile * BM_WMMA;
//     const int tile_size    = min(BM_WMMA, params.T - q_tile_start);
//     if (q_tile_start >= params.T) return;

//     const int warp_id = threadIdx.x / BWD_WARP_SZ;
//     const int chunk   = warp_id % HD_CHUNKS;  // column chunk of HeadDim [0, HD_CHUNKS)
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

//     // ── Step 0: load Q, dO, LSE, D for this Q-tile into smem (persistent) ────
//     for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
//         const int r  = idx / HeadDim, k = idx % HeadDim;
//         const int qi = q_tile_start + r;
//         const bool vq = (qi < params.T);
//         Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
//         dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
//     }
//     if (threadIdx.x < BM_WMMA) {
//         const int qi  = q_tile_start + threadIdx.x;
//         const bool vq = (qi < params.T);
//         LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
//         D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
//     }
//     __syncthreads();

//     // ── Persistent TF32 WMMA accumulator for dQ ──────────────────────────────
//     // Only warps 0..HD_CHUNKS-1 use this; others keep it at zero (never stored).
//     wmma::fragment<wmma::accumulator, 16, 16, 8, float> dq_frag;
//     wmma::fill_fragment(dq_frag, 0.0f);

//     // Causal: skip KV tiles entirely past the Q-tile's last row
//     const int kv_loop_end = Causal ? (q_tile_start + BM_WMMA) : params.T;

//     // ── Main KV loop ──────────────────────────────────────────────────────────
//     for (int kv_base = 0; kv_base < kv_loop_end; kv_base += BlockN) {

//         // sync(a): prior dV atomicAdd reads of tile_st are done; prior Phase B
//         //          reads of Ks are done — safe to overwrite both.
//         __syncthreads();

//         const int kv_tile_size = min(BlockN, params.T - kv_base);

//         // Load K/V tile for this KV iteration
//         for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             const int g = kv_base + r;
//             Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
//             Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
//         }
//         __syncthreads();  // sync(b): Ks/Vs ready

//         // ── Phase A: TF32 WMMA GEMMs + scalar post-process ───────────────────
//         //
//         // Warp 0: S   = Q_sm × K^T  → ds_qd  [BM×BKN_PAD]   (col_major B = K^T)
//         // Warp 1: DPV = dO_sm × V^T → DPV_sm [BM×BKN_PAD]   (col_major B = V^T)
//         // Warps 2..7 are idle during WMMA.
//         //
//         // After sync(c1), all 256 threads run the scalar post-process in parallel
//         // (1 [qi, j] element each): compute p and ds from S, DPV, LSE_sm, D_sm,
//         // then write ds_qd (overwrite S with ds), ds_kd, and p_kd.
//         if (warp_id < 2) {
//             const float* src_sm = (warp_id == 0) ? Q_sm   : dO_sm;
//             const float* kv_sm  = (warp_id == 0) ? Ks     : Vs;
//             float*       dst_sm = (warp_id == 0) ? ds_qd  : DPV_sm;

//             wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
//             wmma::fill_fragment(acc_frag, 0.0f);

//             // col_major B fragment loads K^T (or V^T): b[k][j] = kv_sm[j*HD_PAD + k_off + k]
//             wmma::fragment<wmma::matrix_a, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> a_frag;
//             wmma::fragment<wmma::matrix_b, 16, 16, 8,
//                            wmma::precision::tf32, wmma::col_major> b_frag;

//             // 2*HD_CHUNKS k-splits of 8 each (HD=64 → 8 splits total)
//             #pragma unroll
//             for (int ks = 0; ks < 2 * HD_CHUNKS; ++ks) {
//                 const int k_off = ks * 8;
//                 wmma::load_matrix_sync(a_frag, src_sm + k_off, HD_PAD);
//                 wmma::load_matrix_sync(b_frag, kv_sm  + k_off, HD_PAD);
//                 wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//             }
//             wmma::store_matrix_sync(dst_sm, acc_frag, BKN_PAD, wmma::mem_row_major);
//         }
//         __syncthreads();  // sync(c1): ds_qd (=S_sm) and DPV_sm fully written

//         // Post-process: 256 threads × 1 element — full coverage of [BM×BN]
//         {
//             const int qi_local  = threadIdx.x / BlockN;   // [0, BM_WMMA)
//             const int j_local   = threadIdx.x % BlockN;   // [0, BlockN)
//             const int qi_global = q_tile_start + qi_local;
//             const int j_global  = kv_base + j_local;

//             const float raw_s = ds_qd [qi_local * BKN_PAD + j_local];
//             const float dpv   = DPV_sm[qi_local * BKN_PAD + j_local];
//             const float L     = LSE_sm[qi_local];
//             const float D_    = D_sm  [qi_local];

//             const bool qi_valid  = (qi_global < params.T);
//             const bool j_valid   = (j_local   < kv_tile_size);
//             const bool causal_ok = !Causal || (j_global <= qi_global);

//             float p = 0.f;
//             if (qi_valid && j_valid && causal_ok)
//                 p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));

//             const float ds = p * (dpv - D_) * params.scale;

//             ds_qd[qi_local * BKN_PAD + j_local]  = ds;   // overwrite S with ds
//             ds_kd[j_local  * BM_PAD  + qi_local] = ds;
//             p_kd [j_local  * BM_PAD  + qi_local] = p;
//         }
//         __syncthreads();  // sync(c2): ds_qd, ds_kd, p_kd fully written

//         // ── Phase B — sub-phase 1 (parallel across warp groups) ──────────────
//         //
//         //   Warps 0..HD_CHUNKS-1 : dQ WMMA — acc dq_frag (persistent register)
//         //     A = ds_qd [BM×BN],  B = Ks[:,chunk*16..+16]
//         //
//         //   Warps HD_CHUNKS..2*HD-1 : dK WMMA — write [BN×16] to tile_st
//         //     A = ds_kd [BN×BM],  B = Q_sm[:,chunk*16..+16]
//         //
//         {
//             wmma::fragment<wmma::matrix_a, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> a_frag;
//             wmma::fragment<wmma::matrix_b, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> b_frag;

//             if (warp_id < HD_CHUNKS) {
//                 // dQ: A=ds_qd[BM×BN], B=Ks[BN×16], C=dq_frag[BM×16]
//                 const float* b_ptr = Ks + chunk * 16;
//                 wmma::load_matrix_sync(a_frag, ds_qd,              BKN_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
//                 wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
//                 wmma::load_matrix_sync(a_frag, ds_qd + 8,          BKN_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
//                 wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
//             } else {
//                 // dK: A=ds_kd[BN×BM], B=Q_sm[BM×16], C→tile_st[BN×16]
//                 wmma::fragment<wmma::accumulator, 16, 16, 8, float> dk_frag;
//                 wmma::fill_fragment(dk_frag, 0.0f);
//                 const float* b_ptr = Q_sm + chunk * 16;
//                 wmma::load_matrix_sync(a_frag, ds_kd,              BM_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr,              HD_PAD);
//                 wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
//                 wmma::load_matrix_sync(a_frag, ds_kd + 8,          BM_PAD);
//                 wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
//                 wmma::mma_sync(dk_frag, a_frag, b_frag, dk_frag);
//                 wmma::store_matrix_sync(tile_st + chunk * 16, dk_frag,
//                                         HeadDim, wmma::mem_row_major);
//             }
//         }
//         __syncthreads();  // sync(d): tile_st fully written with dK contribution

//         // All 8 warps atomicAdd tile_st → global dK (coalesced per row)
//         for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             atomicAdd(&dK_bh[(kv_base + r) * dk_sM + k], tile_st[r * HeadDim + k]);
//         }
//         __syncthreads();  // sync(e): all tile_st reads done; safe to overwrite for dV

//         // ── Phase B — sub-phase 2: dV WMMA (warps HD_CHUNKS..2*HD-1) ─────────
//         //   A = p_kd [BN×BM],  B = dO_sm [BM×16], C → tile_st [BN×16]
//         if (warp_id >= HD_CHUNKS) {
//             wmma::fragment<wmma::matrix_a, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> a_frag;
//             wmma::fragment<wmma::matrix_b, 16, 16, 8,
//                            wmma::precision::tf32, wmma::row_major> b_frag;
//             wmma::fragment<wmma::accumulator, 16, 16, 8, float> dv_frag;
//             wmma::fill_fragment(dv_frag, 0.0f);

//             const float* b_ptr = dO_sm + chunk * 16;
//             wmma::load_matrix_sync(a_frag, p_kd,              BM_PAD);
//             wmma::load_matrix_sync(b_frag, b_ptr,             HD_PAD);
//             wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
//             wmma::load_matrix_sync(a_frag, p_kd + 8,          BM_PAD);
//             wmma::load_matrix_sync(b_frag, b_ptr + 8 * HD_PAD, HD_PAD);
//             wmma::mma_sync(dv_frag, a_frag, b_frag, dv_frag);
//             wmma::store_matrix_sync(tile_st + chunk * 16, dv_frag,
//                                     HeadDim, wmma::mem_row_major);
//         }
//         __syncthreads();  // sync(f): tile_st has dV contribution

//         // All 8 warps atomicAdd tile_st → global dV (coalesced per row)
//         for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
//             const int r = idx / HeadDim, k = idx % HeadDim;
//             atomicAdd(&dV_bh[(kv_base + r) * dv_sM + k], tile_st[r * HeadDim + k]);
//         }
//         // sync(a) of next iteration ensures dV atomicAdd reads of tile_st
//         // are complete before tile_st is overwritten by the next dK WMMA.
//     }

//     // ── Final store: dq_frag → tile_st smem → global dQ (no atomicAdd) ───────
//     __syncthreads();  // wait for last dV atomicAdd reads before reusing tile_st

//     if (warp_id < HD_CHUNKS) {
//         wmma::store_matrix_sync(tile_st + chunk * 16, dq_frag,
//                                 HeadDim, wmma::mem_row_major);
//     }
//     __syncthreads();

//     for (int idx = threadIdx.x; idx < tile_size * HeadDim; idx += blockDim.x) {
//         const int r = idx / HeadDim, k = idx % HeadDim;
//         dQ_bh[(q_tile_start + r) * dq_sM + k] = tile_st[r * HeadDim + k];
//     }
// }

// // ============================================================================
// // exp12 (SM89 Ada Lovelace) lives in arch/AttentionBackward_sm89.cu.
// // Forward-declare the entry point for the arch-dispatch below.
// // ============================================================================
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
//     bool skip_grad_zero);

// // 3xTF32 variant — same external signature, gated by USE_3XTF32=1 env var.
// // Matches PyTorch's CUTLASS kernel_backward.h OpMultiplyAddFastF32 on all 5
// // backward GEMMs (MatmulQK, MatmulDOIVJ, MatmulGradV, MatmulGradQ, MatmulGradK).
// void mem_efficient_attn_backward_sm89_3xtf32_cuda(
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
//     bool skip_grad_zero);

// // KV-centric backward — SM86 (Ampere A10) and SM89 (Ada) 1xTF32 path, hd<=96.
// // dK/dV are atomic-free; dQ uses atomicAdd (launcher zeros dQ internally).
// // Calls precompute_D internally — do NOT call it separately before this.
// void mem_efficient_attn_backward_kvcentric_cuda(
//     const float* query,       int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     const float* key,         int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     const float* value,       int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     const float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
//     const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
//     float* grad_query,        int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
//     float* grad_key,          int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
//     float* grad_value,        int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
//     float* D_buf,             int64_t d_strideB,  int64_t d_strideH,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal,
//     bool skip_grad_zero);

// // ============================================================================
// // Public API (namespace cuda)
// // ============================================================================

// namespace cuda {

// void compute_row_lse(
//     const float* scores, float* lse,
//     int64_t batch, int64_t T, bool is_causal)
// {
//     int64_t total_rows = batch * T;
//     int threads = 32;
//     while (threads < T && threads < 1024) threads <<= 1;
//     size_t smem = threads * sizeof(float);
//     compute_row_lse_kernel<<<(int)total_rows, threads, smem>>>(
//         scores, lse, T, is_causal);
// }

// void mem_efficient_attn_backward(
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
//     dim3 block_cfg(BWD_NUM_THREADS);

//     dim3 grid_D(((int)T + (BWD_BLOCK_M_D * 2) - 1) / (BWD_BLOCK_M_D * 2), BH);

//     MemEfficientBwdParams params;
//     params.Q     = query;
//     params.K     = key;
//     params.V     = value;
//     params.O     = output;
//     params.dO    = grad_output;
//     params.LSE   = lse;
//     params.D     = D_buf;
//     params.dQ    = grad_query;
//     params.dK    = grad_key;
//     params.dV    = grad_value;
//     params.B     = (int)B;
//     params.nh    = (int)nh;
//     params.T     = (int)T;
//     params.scale = scale;
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

//     // exp7: scalar KV-outer fallback for HeadDims not divisible by 16
//     // exp7 is the only path that atomicAdds into dQ, so it must zero grad_query here.
// #define LAUNCH_MEM_BWD_EXP7(HD) \
//     do { \
//         const int block_n7 = ((HD) < 64) ? 16 : (1024 / (HD)); \
//         const size_t shmem_exp7 = 2ULL * block_n7 * ((HD) + 1) * sizeof(float); \
//         const int kv_tiles7 = ((int)T + block_n7 - 1) / block_n7; \
//         dim3 grid_bwd7(kv_tiles7, BH); \
//         if (!skip_grad_zero) { \
//             cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
//         } \
//         mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
//         if (is_causal) { \
//             mem_efficient_bwd_unified_kernel_exp7<HD, true> \
//                 <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
//         } else { \
//             mem_efficient_bwd_unified_kernel_exp7<HD, false> \
//                 <<<grid_bwd7, block_cfg, shmem_exp7>>>(params); \
//         } \
//     } while (0)

//     // exp11: Q-tile-centric, TF32 WMMA, dQ atomic-free, HD%16==0 only
// #define LAUNCH_MEM_BWD_EXP11(HD) \
//     do { \
//         constexpr int BN11 = 16, BM11 = 16; \
//         const size_t shmem11 = \
//             (2ULL * BM11 * ((HD) + 1)   /* Q_sm + dO_sm   */ \
//            + 2ULL * BN11 * ((HD) + 1)   /* Ks + Vs        */ \
//            + 2ULL * BM11 * ((BN11) + 1) /* ds_qd + DPV_sm */ \
//            + 2ULL * BN11 * ((BM11) + 1) /* ds_kd + p_kd   */ \
//            + 1ULL * BN11 * (HD)         /* tile_st        */ \
//            + 2ULL * BM11                /* LSE_sm + D_sm  */ \
//             ) * sizeof(float); \
//         const int q11 = ((int)T + BM11 - 1) / BM11; \
//         dim3 grid_q11(q11, BH); \
//         cudaFuncSetAttribute( \
//             mem_efficient_bwd_unified_kernel_exp11<HD, false>, \
//             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem11); \
//         if (!skip_grad_zero) { \
//             cudaMemsetAsync(params.dK, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
//             cudaMemsetAsync(params.dV, 0, (size_t)BH * (int)T * (HD) * sizeof(float)); \
//         } \
//         mem_efficient_bwd_precompute_D<HD><<<grid_D, block_cfg>>>(params); \
//         if (is_causal) { \
//             mem_efficient_bwd_unified_kernel_exp11<HD, true> \
//                 <<<grid_q11, block_cfg, shmem11>>>(params); \
//         } else { \
//             mem_efficient_bwd_unified_kernel_exp11<HD, false> \
//                 <<<grid_q11, block_cfg, shmem11>>>(params); \
//         } \
//     } while (0)

//     // KV-centric launcher (SM86 + SM89 1xTF32 path). Self-contained: zeros dQ
//     // and calls precompute_D internally. Supported hd: 16,32,48,64,80,96.
// #define LAUNCH_MEM_BWD_KVCENTRIC(HD) \
//     do { \
//         mem_efficient_attn_backward_kvcentric_cuda( \
//             query,       q_strideB, q_strideM, q_strideH, \
//             key,         k_strideB, k_strideM, k_strideH, \
//             value,       v_strideB, v_strideM, v_strideH, \
//             output,      o_strideB, o_strideM, o_strideH, \
//             grad_output, do_strideB, do_strideM, do_strideH, \
//             lse,         lse_strideB, lse_strideH, \
//             grad_query,  dq_strideB, dq_strideM, dq_strideH, \
//             grad_key,    dk_strideB, dk_strideM, dk_strideH, \
//             grad_value,  dv_strideB, dv_strideM, dv_strideH, \
//             D_buf,       d_strideB,  d_strideH, \
//             B, nh, T, hd, is_causal, skip_grad_zero); \
//     } while (0)

//     // SM86 (Ampere A10): KV-centric for hd%16==0 && hd<=96.
//     // hd>96 falls through to the generic exp11 switch below.
//     if (hd % 16 == 0 && hd <= 96 &&
//             cuda::get_arch() == cuda::ArchFamily::Ampere_A10) {
//         switch ((int)hd) {
//             case 16: LAUNCH_MEM_BWD_KVCENTRIC(16); break;
//             case 32: LAUNCH_MEM_BWD_KVCENTRIC(32); break;
//             case 48: LAUNCH_MEM_BWD_KVCENTRIC(48); break;
//             case 64: LAUNCH_MEM_BWD_KVCENTRIC(64); break;
//             case 80: LAUNCH_MEM_BWD_KVCENTRIC(80); break;
//             case 96: LAUNCH_MEM_BWD_KVCENTRIC(96); break;
//             default: break;
//         }
//         return;
//     }

//     // SM89 (Ada Lovelace): dispatch to exp12 (BM=32, bank-conflict-free, 2 blocks/SM).
//     // Non-%16 HD falls through to exp7 below regardless of arch.
//     if (hd % 16 == 0 && cuda::get_arch() == cuda::ArchFamily::Ada) {
//       //if (false) {
//         // USE_3XTF32=1 routes to the CUTLASS-style OpMultiplyAddFastF32
//         // emulation kernel that mirrors PyTorch's kernel_backward.h on all
//         // 5 backward GEMMs. Smem caps it at hd <= 128 (~78 KB at HD=64 vs
//         // Ada's 99 KB opt-in limit); larger HDs or env unset fall back to
//         // the 1xTF32 path below.
//         static const bool kUse3xTF32 = []() {
//             const char* e = std::getenv("USE_3XTF32");
//             return e && e[0] == '1';
//         }();
//         if (kUse3xTF32 && hd <= 128) {
//             mem_efficient_attn_backward_sm89_3xtf32_cuda(
//                 query,       q_strideB, q_strideM, q_strideH,
//                 key,         k_strideB, k_strideM, k_strideH,
//                 value,       v_strideB, v_strideM, v_strideH,
//                 output,      o_strideB, o_strideM, o_strideH,
//                 grad_output, do_strideB, do_strideM, do_strideH,
//                 lse,         lse_strideB, lse_strideH,
//                 grad_query,  dq_strideB, dq_strideM, dq_strideH,
//                 grad_key,    dk_strideB, dk_strideM, dk_strideH,
//                 grad_value,  dv_strideB, dv_strideM, dv_strideH,
//                 D_buf,       d_strideB, d_strideH,
//                 B, nh, T, hd, is_causal, skip_grad_zero);
//             return;
//         }
//         // hd<=96: KV-centric — atomic-free dK/dV, better SM89 occupancy at HD=64.
//         // hd>96: fall back to exp12 Q-centric (kvcentric smem exceeds 99KB limit).
//         if (hd <= 96) {
//             switch ((int)hd) {
//                 case 16: LAUNCH_MEM_BWD_KVCENTRIC(16); break;
//                 case 32: LAUNCH_MEM_BWD_KVCENTRIC(32); break;
//                 case 48: LAUNCH_MEM_BWD_KVCENTRIC(48); break;
//                 case 64: LAUNCH_MEM_BWD_KVCENTRIC(64); break;
//                 case 80: LAUNCH_MEM_BWD_KVCENTRIC(80); break;
//                 case 96: LAUNCH_MEM_BWD_KVCENTRIC(96); break;
//                 default: break;
//             }
//             return;
//         }
//         mem_efficient_attn_backward_sm89_cuda(
//             query,       q_strideB, q_strideM, q_strideH,
//             key,         k_strideB, k_strideM, k_strideH,
//             value,       v_strideB, v_strideM, v_strideH,
//             output,      o_strideB, o_strideM, o_strideH,
//             grad_output, do_strideB, do_strideM, do_strideH,
//             lse,         lse_strideB, lse_strideH,
//             grad_query,  dq_strideB, dq_strideM, dq_strideH,
//             grad_key,    dk_strideB, dk_strideM, dk_strideH,
//             grad_value,  dv_strideB, dv_strideM, dv_strideH,
//             D_buf,       d_strideB, d_strideH,
//             B, nh, T, hd, is_causal, skip_grad_zero);
//         return;
//     }

//     switch ((int)hd) {
//         case   8: LAUNCH_MEM_BWD_EXP7(  8); break;
//         case  16: LAUNCH_MEM_BWD_EXP11( 16); break;
//         case  24: LAUNCH_MEM_BWD_EXP7( 24); break;
//         case  32: LAUNCH_MEM_BWD_EXP11( 32); break;
//         case  40: LAUNCH_MEM_BWD_EXP7( 40); break;
//         case  48: LAUNCH_MEM_BWD_EXP11( 48); break;
//         case  56: LAUNCH_MEM_BWD_EXP7( 56); break;
//         case  64: LAUNCH_MEM_BWD_EXP11( 64); break;
//         case  80: LAUNCH_MEM_BWD_EXP11( 80); break;
//         case  96: LAUNCH_MEM_BWD_EXP11( 96); break;
//         case 128: LAUNCH_MEM_BWD_EXP11(128); break;
//         case 160: LAUNCH_MEM_BWD_EXP11(160); break;
//         case 192: LAUNCH_MEM_BWD_EXP11(192); break;
//         case 256: LAUNCH_MEM_BWD_EXP11(256); break;
//         default:
//             printf("mem_efficient_attn_backward: unsupported head_dim %d\n", (int)hd);
//             break;
//     }
// #undef LAUNCH_MEM_BWD_EXP7
// #undef LAUNCH_MEM_BWD_EXP11
// #undef LAUNCH_MEM_BWD_KVCENTRIC
// }

// } // namespace cuda
// } // namespace OwnTensor