#include "ops/helpers/AttentionKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/cuda/attention/AttentionCommon.cuh"

namespace OwnTensor {

// ============================================================================
// Constants & Prefetch Helpers
// ============================================================================

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_l16(void* smem_ptr, const void* global_ptr, bool pred) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"
        "}\n"
        : : "r"(smem_addr), "l"(global_ptr), "r"((int)pred) : "memory");
}

template<typename FragType>
__device__ __forceinline__ void round_tf32_wmma_frag(FragType& frag) {
    #pragma unroll
    for (int i = 0; i < (int)FragType::num_elements; i++)
        reinterpret_cast<uint32_t&>(frag.x[i]) += 0x1000u;
}

// static constexpr int MAX_HD = 256;   // max supported head dimension

// ============================================================================
// Forward Kernel: Fused Memory-Efficient Attention (shared-memory-centric)
// ============================================================================
//
// Computes: O = softmax(Q @ K^T / sqrt(hd)) @ V
// WITHOUT materializing the full T x T attention matrix.
//
// Design: MULTIPLE threads per query row (mirrors the backward kernel)
//
//   Grid : (ceil(T / FWD_BQ), B * nh)
//   Block: FWD_NUM_THREADS = 256
//
//   Thread mapping: 256 threads = 32 rows x 8 threads/row
//     qi_local  = tid / 8    -- which query row (0..31)
//     local_tid = tid % 8    -- position within the 8-thread row group
//
//   Shared memory layout (maximized -- everything shared lives here,
//   just like the backward kernel):
//
//     s_q      [BQ x hd]    -- Q tile, loaded once, persistent
//     s_kv     [BK x hd]    -- K tile then V tile (reused each iter)
//     s_scores [BQ x BK]    -- score/attention-weight matrix
//     s_m      [BQ]          -- running max per query row
//     s_l      [BQ]          -- running sum per query row
//
//   Per thread:
//     reg_out[hd/8]  -- output accumulator in registers (not smem)
//                       Each of 8 threads owns a DIFFERENT slice of hd.
//
//   Why s_scores in shared memory?
//     During Score GEMM: each thread computes 4 dot products (split by
//     key columns). During P@V GEMM: each thread accumulates hd/8 output
//     dims, reading ALL BK attention weights. Thread roles CHANGE between
//     phases -- s_scores is the communication channel, exactly like
//     ds_buf/p_buf in the backward kernel.
//
//   Why s_m/s_l in shared memory?
//     Only local_tid==0 writes the updated max/sum. All 4 threads in the
//     row group read it for rescaling. Mirrors LSE/D in the backward.
//
//   Algorithm (online softmax):
//     1. Cooperative load Q tile -> s_q
//     2. For each K/V tile:
//        a. Cooperative load K tile -> s_kv
//        b. Score GEMM: each thread computes 4 dots -> s_scores
//        c. Online softmax via warp shuffle across 4-thread groups:
//           partial max -> shuffle reduce -> update s_m
//           rescale reg_out by exp(m_old - m_new)
//           exp(scores) -> s_scores, partial sum -> shuffle -> s_l
//        d. Cooperative load V tile -> s_kv
//        e. P@V GEMM: read all BK weights from s_scores,
//           accumulate hd/8 dims per thread in reg_out
//     3. O = reg_out / s_l
//     4. LSE = s_m + log(s_l)
//

// Forward tuning constants
static constexpr int FWD_BQ              = 32;  // query rows per block
static constexpr int FWD_BK              = 32;  // key rows per tile
static constexpr int FWD_THREADS_PER_ROW = 8;   // threads per Q row (was 4, now 8 for occupancy)
static constexpr int FWD_NUM_THREADS     = FWD_BQ * FWD_THREADS_PER_ROW;  // 256
static constexpr int FWD_KJ_PER_THREAD   = FWD_BK / FWD_THREADS_PER_ROW; // 4
static constexpr int FWD_SMEM_PAD        = 1;   // +1 padding to eliminate bank conflicts
static constexpr int FWD_SCORE_STRIDE    = FWD_BK + FWD_SMEM_PAD;  // 33 (odd → no conflict)

template<int HeadDim>
__global__ void fused_attn_forward_kernel(MemEfficientFwdParams params)
{
    // Aliases so the body stays close to the pre-strided version.
    const float* __restrict__ Q             = params.Q;
    const float* __restrict__ K             = params.K;
    const float* __restrict__ V             = params.V;
    float*       __restrict__ O             = params.O;
    float*       __restrict__ LSE           = params.LSE;
    const int64_t T                         = params.T;
    const int     nh                        = params.nh;
    const float   scale                     = params.scale;
    const bool    is_causal                 = params.is_causal;
    const float   dropout_p                 = params.dropout_p;
    const float* __restrict__ dropout_mask  = params.dropout_mask;

    // All dimension-derived values are now compile-time constants
    constexpr uint HD_PAD       = HeadDim + FWD_SMEM_PAD;   // padded stride (e.g. 65)
    constexpr uint D_PER_THREAD = (HeadDim + FWD_THREADS_PER_ROW - 1) / FWD_THREADS_PER_ROW;

    const uint qi_block = (int64_t)blockIdx.x * FWD_BQ;
    const uint bnh      = blockIdx.y;
    const uint tid      = threadIdx.x;
    const int  b        = (int)bnh / nh;
    const int  h        = (int)bnh - b * nh;

    // ---- Thread-to-query-row mapping ----
    // 8 consecutive threads share one query row:
    //   threads 0-7     -> row 0
    //   threads 8-15    -> row 1
    //   ...
    //   threads 248-255 -> row 31
    //
    // NOTE: the warp_id/lane_id layout from the coalescing doc breaks the
    // warp shuffle reductions (they can't cross warp boundaries). The
    // PRACTICAL FIX in that doc keeps this mapping and stages output through
    // shared memory instead.
    const int qi_local  = tid / FWD_THREADS_PER_ROW;   // 0..31
    const int local_tid = tid % FWD_THREADS_PER_ROW;   // 0..7
    const int64_t qi_global = qi_block + qi_local;
    const bool qi_valid = (qi_global < T);

    // ---- Pointers for this batch*head (strided, last-dim stride = 1) ----
    const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
    float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
    float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

    const int64_t q_sM = params.q_strideM;
    const int64_t k_sM = params.k_strideM;
    const int64_t v_sM = params.v_strideM;
    const int64_t o_sM = params.o_strideM;

    // ===================================================================
    // Shared memory layout (padded to eliminate bank conflicts)
    // ===================================================================
    //
    //   Row strides are padded by +1 so successive rows map to different
    //   banks (stride 65 and 33 are odd → conflict-free across rows).
    //
    //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
    //   |  s_q              |  s_kv             |  s_scores       | s_m | s_l |  s_out            |
    //   |  [BQ x HD_PAD]    |  [BK x HD_PAD]    |  [BQ x (BK+1)] |[BQ] |[BQ] |  [BQ x HD_PAD]   |
    //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
    //
    //   s_out: staging buffer for coalesced global writeback (same padded
    //          stride as s_q → same conflict-free access pattern).
    //
    extern __shared__ float smem[];
    float* s_q      = smem;                                              // [BQ x HD_PAD]
    float* s_kv     = s_q      + FWD_BQ * HD_PAD;                       // [BK x HD_PAD]
    float* s_scores = s_kv     + FWD_BK * HD_PAD;                       // [BQ x FWD_SCORE_STRIDE]
    float* s_m      = s_scores + FWD_BQ * FWD_SCORE_STRIDE;             // [BQ]
    float* s_l      = s_m      + FWD_BQ;                                // [BQ]
    float* s_out    = s_l      + FWD_BQ;                                // [BQ x HD_PAD]

    // ---- Initialize running max and sum in shared memory ----
    if (local_tid == 0) {
        s_m[qi_local] = -INFINITY;
        s_l[qi_local] = 0.0f;
    }

    // ---- Output accumulator in registers ----
    // HeadDim is now a compile-time constant, so the compiler knows
    // D_PER_THREAD exactly and keeps reg_out entirely in ACTUAL registers
    // (no spilling to local memory).
    // Strided assignment: thread local_tid owns dimensions {local_tid, local_tid+8, local_tid+16, ...}
    // i.e., reg_out[dd] accumulates output for dimension (local_tid + dd * FWD_THREADS_PER_ROW)

    float reg_out[D_PER_THREAD];   // compile-time size → guaranteed registers
    #pragma unroll
    for (int i = 0; i < D_PER_THREAD; i++) reg_out[i] = 0.0f;

    // ===================================================================
    // Step 1: Cooperative load Q tile into s_q (ALL 256 threads help)
    // ===================================================================
    {
        constexpr int64_t total = (int64_t)FWD_BQ * HeadDim;
        for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
            int q = (int)(i / HeadDim);
            int d = (int)(i % HeadDim);
            s_q[q * HD_PAD + d] = (qi_block + q < T) ? Q_bnh[(qi_block + q) * q_sM + d] : 0.0f;
        }
    }
    __syncthreads();

    const int actual_q = ((int)(T - qi_block) < FWD_BQ)
                         ? (int)(T - qi_block) : FWD_BQ;
    if (actual_q <= 0) return;

    // ---- Causal early termination ----
    const int64_t max_kj = is_causal
        ? (((qi_block + actual_q) < T) ? (qi_block + actual_q) : T)
        : T;

    // ===================================================================
    // Step 2: Main loop over key/value tiles
    // ===================================================================
    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_BK) {
        const int block_len = ((int)(T - kj_block) < FWD_BK)
                              ? (int)(T - kj_block) : FWD_BK;

        // ---------------------------------------------------------------
        // Step 2a: Cooperative load K tile into s_kv (padded stride)
        // ---------------------------------------------------------------
        {
            const int64_t total = (int64_t)block_len * HeadDim;
            for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
                int k = (int)(i / HeadDim);
                int d = (int)(i % HeadDim);
                s_kv[k * HD_PAD + d] = K_bnh[(kj_block + k) * k_sM + d];
            }
        }
        __syncthreads();    // K tile ready in s_kv

        // ---------------------------------------------------------------
        // Step 2b: Score GEMM -- Q[BQ x hd] @ K[BK x hd]^T -> s_scores
        // ---------------------------------------------------------------
        {
            const int kj_start = local_tid * FWD_KJ_PER_THREAD;

            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                float dot;
                if (kj < block_len && qi_valid) {
                    dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HeadDim; d++)
                        dot += s_q[qi_local * HD_PAD + d] * s_kv[kj * HD_PAD + d];
                    dot *= scale;

                    // Causal mask
                    if (is_causal && (kj_block + kj) > qi_global)
                        dot = -INFINITY;
                } else {
                    dot = -INFINITY;
                }
                s_scores[qi_local * FWD_SCORE_STRIDE + kj] = dot;
            }
        }
        __syncthreads();    // all scores in s_scores

        // ---------------------------------------------------------------
        // Step 2c: Online softmax (warp shuffle across 8-thread groups)
        // ---------------------------------------------------------------
        {
            const int kj_start = local_tid * FWD_KJ_PER_THREAD;

            // (a) Partial row-max over this thread's 4 keys
            float partial_max = -INFINITY;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                if (kj < block_len)
                    partial_max = fmaxf(partial_max,
                                        s_scores[qi_local * FWD_SCORE_STRIDE + kj]);
            }

            // (b) Warp shuffle: reduce max across 8 threads in the group
            float row_max = partial_max;
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 2));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 4));

            // (c) Read old running max from shared memory
            float m_old = s_m[qi_local];
            float m_new = fmaxf(m_old, row_max);

            // (d) Correction factor for previous accumulations
            float alpha;
            if (m_old == -INFINITY)       alpha = 0.0f;
            else if (m_old == m_new)      alpha = 1.0f;
            else                          alpha = expf(m_old - m_new);

            // (e) Rescale this thread's output accumulator slice
            #pragma unroll
            for (int i = 0; i < D_PER_THREAD; i++)
                reg_out[i] *= alpha;

            // (f) Exponentiate scores, apply dropout, accumulate post-dropout sum.
            //
            // Step 2c is split so that s_l accumulates post-dropout weights
            // (walkthrough Option 2): compute exp → apply mask → sum masked values.
            // This ensures the final normalisation (reg_out / s_l) is correct
            // without any post-hoc correction of s_l.
            //
            // Dropout uses Option B (pre-generated mask tensor, shape [B*nh, T, T]):
            //   mask[bnh, qi, kj] already encodes the Bernoulli decision scaled by
            //   1/(1-p), so multiplying exp_s by the mask value both zeroes dropped
            //   entries and rescales surviving ones in one operation.
            float partial_sum = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                float exp_s;
                if (kj < block_len && m_new > -INFINITY) {
                    exp_s = expf(s_scores[qi_local * FWD_SCORE_STRIDE + kj] - m_new);
                    // Apply pre-generated dropout mask (scale-and-zero in one multiply).
                    // When dropout_p == 0 or no mask is provided this branch is skipped.
                    if (dropout_p > 0.0f && dropout_mask != nullptr) {
                        float m_val = dropout_mask[(bnh * T + qi_global) * T
                                                   + (kj_block + kj)];
                        exp_s *= m_val;
                    }
                } else {
                    exp_s = 0.0f;
                }
                s_scores[qi_local * FWD_SCORE_STRIDE + kj] = exp_s;
                partial_sum += exp_s;   // accumulate post-dropout sum → s_l correct
            }
            // Zero out padding
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                if (kj >= block_len)
                    s_scores[qi_local * FWD_SCORE_STRIDE + kj] = 0.0f;
            }

            // (g) Warp shuffle: reduce sum across 8 threads
            float row_sum = partial_sum;
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);

            // (h) Update shared running state (one writer per row)
            if (local_tid == 0) {
                s_l[qi_local] = alpha * s_l[qi_local] + row_sum;
                s_m[qi_local] = m_new;
            }
        }
        __syncthreads();    // s_scores holds P, s_m/s_l updated

        // ---------------------------------------------------------------
        // Step 2d: Cooperative load V tile into s_kv (overwrites K, padded)
        // ---------------------------------------------------------------
        {
            const int64_t total = (int64_t)block_len * HeadDim;
            for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
                int v = (int)(i / HeadDim);
                int d = (int)(i % HeadDim);
                s_kv[v * HD_PAD + d] = V_bnh[(kj_block + v) * v_sM + d];
            }
        }
        __syncthreads();    // V tile ready in s_kv

        // ---------------------------------------------------------------
        // Step 2e: P@V GEMM -- accumulate into reg_out
        // ---------------------------------------------------------------
        // kj iterates to FWD_BK (compile-time constant = 32) instead of
        // block_len (runtime), so #pragma unroll 4 produces 8 fully-pipelined
        // groups of 4 smem reads, overlapping latency with compute.
        // Boundary guard is inside the kj loop; padded elements are 0 (set in
        // Step 2c so this is safe). s_scores rows were already zeroed for kj
        // >= block_len in the softmax step, so no extra cost.
        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            // Strided assignment: thread local_tid owns every 8th dimension.
            // Thread 0: d=0,8,16,...   Thread 1: d=1,9,17,...   Thread 7: d=7,15,23,...
            // For fixed kj: bank(thread_i) = (kj*HD_PAD + dd*8 + i) % 32 → all 8 banks distinct → 0 conflicts.
            int d = local_tid + dd * FWD_THREADS_PER_ROW;
            if (d < HeadDim) {
                float acc = 0.0f;
                #pragma unroll 4
                for (int kj = 0; kj < FWD_BK; kj++) {
                    if (kj < block_len)
                        acc += s_scores[qi_local * FWD_SCORE_STRIDE + kj]
                             * s_kv[kj * HD_PAD + d];
                }
                reg_out[dd] += acc;
            }
        }
        __syncthreads();    // before next iteration overwrites s_kv/s_scores
    }

    // ===================================================================
    // Step 3-4: Normalize into s_out (staged), then coalesced global write
    // ===================================================================
    //
    // WHY TWO PHASES:
    //   Direct write: thread i writes O[qi_global * HeadDim + d_start + dd].
    //   Threads 0..7 all write to the SAME row (row 0), not adjacent rows,
    //   so within a warp the 32 threads write to 4 different rows with
    //   stride HeadDim*4 bytes → separate cache-line transactions per thread.
    //
    //   Phase A (scatter to s_out): each thread writes its d-slice into the
    //   shared staging buffer at s_out[qi_local][d_start..d_end]. Same access
    //   pattern as loading s_q → already proven conflict-free with HD_PAD.
    //
    //   Phase B (coalesced gather to global): all 256 threads stride linearly
    //   over s_out and write to O. Adjacent threads → adjacent global addresses
    //   → full 128-byte cache-line utilization per transaction.

    // ---- Phase A: normalize and scatter to shared staging buffer ----
    if (qi_valid) {
        float li    = s_l[qi_local];
        float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;

        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            int d = local_tid + dd * FWD_THREADS_PER_ROW;
            if (d < HeadDim)
                s_out[qi_local * HD_PAD + d] = reg_out[dd] * inv_l;
        }
    }
    __syncthreads();

    // ---- Phase B: cooperative coalesced writeback s_out -> O ----
    // Threads stride linearly: tid 0 writes (row0, d0), tid 1 writes (row0, d1), …
    // Adjacent tids → adjacent global memory addresses → coalesced.
    {
        const int64_t total = (int64_t)actual_q * HeadDim;
        for (int64_t i = (int64_t)tid; i < total; i += FWD_NUM_THREADS) {
            int q = (int)(i / HeadDim);
            int d = (int)(i % HeadDim);
            O_bnh[(qi_block + q) * o_sM + d] = s_out[q * HD_PAD + d];
        }
    }

    // ---- Write LSE: one thread per row (local_tid == 0) ----
    if (local_tid == 0 && qi_valid) {
        float m = s_m[qi_local];
        float l = s_l[qi_local];
        LSE_bnh[qi_global] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}


// ============================================================================
// Forward launch function
// ============================================================================
//
// Shared memory (padded, with s_out staging buffer):
//   ((BQ + BK + BQ) * (hd + 1) + BQ * (BK + 1) + 2 * BQ) * sizeof(float)
//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    s_q + s_kv + s_out, all with padded stride
//
//   hd=64:  (32+32+32)*65 + 32*33 + 64  =  7,360 floats =  29,440 bytes
//   hd=128: (32+32+32)*129 + 32*33 + 64 = 12,544 floats =  50,176 bytes (opt-in)
//   hd=256: (32+32+32)*257 + 32*33 + 64 = 24,768 floats =  99,072 bytes (opt-in)
//

static size_t compute_fwd_smem(int64_t hd) {
    int64_t hd_pad = hd + FWD_SMEM_PAD;
    return ((size_t)(FWD_BQ + FWD_BK + FWD_BQ) * hd_pad  // s_q + s_kv + s_out
          + (size_t)FWD_BQ * FWD_SCORE_STRIDE              // s_scores
          + (size_t)2 * FWD_BQ) * sizeof(float);           // s_m + s_l
}

static void launch_fwd_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
    float dropout_p, const float* dropout_mask,
    int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    int64_t lse_strideB, int64_t lse_strideH,
    int grid_y)
{
    // Build strided params once, reuse for all template dispatches below.
    MemEfficientFwdParams params;
    params.Q = Q; params.K = K; params.V = V;
    params.O = O; params.LSE = LSE;
    params.B = (int)B; params.nh = (int)nh; params.T = (int)T;
    params.scale = scale; params.is_causal = is_causal;
    params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
    params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
    params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
    params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
    params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
    // --- Query the device's shared memory limit ---
    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    size_t smem = compute_fwd_smem(hd);

    if ((int)smem > max_smem) {
        printf("fused_attn_forward: hd=%d needs %zu bytes smem, "
               "device max is %d. Cannot launch.\n",
               (int)hd, smem, max_smem);
        return;
    }

    // --- Grid: one block per query tile x one block per batch-head ---
    int grid_x = (int)((T + FWD_BQ - 1) / FWD_BQ);
    dim3 grid(grid_x, grid_y);

    //* print the grid and block dims
    // printf("The config for the kernel is: block{%d, 1} grid{%d, %d}\n", FWD_NUM_THREADS, grid.x, grid.y);
    // exit(1);

    // --- Template dispatch (same pattern as backward kernel) ---
    // HeadDim must be a compile-time constant so reg_out stays in registers.
    #define LAUNCH_FWD(HD) \
    do { \
        auto* kernel = fused_attn_forward_kernel<HD>; \
        cudaFuncSetAttribute(kernel, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             (int)smem); \
        kernel<<<grid, FWD_NUM_THREADS, smem>>>(params); \
    } while (0)

    switch ((int)hd) {
        case   8: LAUNCH_FWD(  8); break;
        case  16: LAUNCH_FWD( 16); break;
        case  24: LAUNCH_FWD( 24); break;
        case  32: LAUNCH_FWD( 32); break;
        case  40: LAUNCH_FWD( 40); break;
        case  48: LAUNCH_FWD( 48); break;
        case  56: LAUNCH_FWD( 56); break;
        case  64: LAUNCH_FWD( 64); break;
        case  80: LAUNCH_FWD( 80); break;
        case  96: LAUNCH_FWD( 96); break;
        case 128: LAUNCH_FWD(128); break;
        case 160: LAUNCH_FWD(160); break;
        case 192: LAUNCH_FWD(192); break;
        case 256: LAUNCH_FWD(256); break;
        default:
            printf("fused_attn_forward: unsupported head_dim %d\n", (int)hd);
            break;
    }
    #undef LAUNCH_FWD
}


static constexpr int FWD_TC_BQ          = 32;   // query rows per block
static constexpr int FWD_TC_BK          = 32;   // key/value rows per KV tile
static constexpr int FWD_TC_NUM_THREADS = 256;  // 8 warps × 32 lanes
static constexpr int FWD_TC_WMMA_M      = 16;   // WMMA tile rows
static constexpr int FWD_TC_WMMA_N      = 16;   // WMMA tile cols (HeadDim must be multiple)
static constexpr int FWD_TC_WMMA_K      = 8;    // WMMA tile K-dim for TF32
static constexpr int FWD_TC_SMEM_PAD    = 4;    // +4 padding: makes HD_PAD a multiple of 4
                                                //   (required by wmma::store_matrix_sync for
                                                //   FP32 accumulators) while avoiding bank
                                                //   conflicts (hd is always ≡0 mod 16, so
                                                //   HD_PAD=hd+4 is never a multiple of 32).

template<int HeadDim>
__global__ void fused_attn_forward_kernel_tc(MemEfficientFwdParams params)
{
    // Aliases so the body stays close to the pre-strided version.
    const float* __restrict__ Q             = params.Q;
    const float* __restrict__ K             = params.K;
    const float* __restrict__ V             = params.V;
    float*       __restrict__ O             = params.O;
    float*       __restrict__ LSE           = params.LSE;
    const int64_t T                         = params.T;
    const int     nh                        = params.nh;
    const float   scale                     = params.scale;
    const bool    is_causal                 = params.is_causal;
    const float   dropout_p                 = params.dropout_p;
    const float* __restrict__ dropout_mask  = params.dropout_mask;

    static_assert(HeadDim % FWD_TC_WMMA_N == 0,
                  "fused_attn_forward_kernel_tc: HeadDim must be divisible by 16");
    using namespace nvcuda;

    // ── Compile-time tile layout constants ───────────────────────────────────
    constexpr int HD_PAD        = HeadDim + FWD_TC_SMEM_PAD;
    // Score GEMM: Q[32×hd] @ K[32×hd]^T → s_scores[32×32]
    //   2×2 grid of 16×16 output tiles; K-dimension split into hd/8 k-tiles.
    constexpr int SCORE_N_TILES = FWD_TC_BK  / FWD_TC_WMMA_N;   // 2
    constexpr int SCORE_K_TILES = HeadDim    / FWD_TC_WMMA_K;   // hd/8
    // P@V GEMM: P[32×32] @ V[32×hd] → s_pv[32×hd]
    //   2×(hd/16) output tiles; K-dimension (=32) split into 4 k-tiles.
    constexpr int PV_N_TILES    = HeadDim    / FWD_TC_WMMA_N;   // hd/16
    constexpr int PV_K_TILES    = FWD_TC_BK  / FWD_TC_WMMA_K;  // 4
    constexpr int PV_TOTAL      = 2 * PV_N_TILES;               // total output tiles
    constexpr int PV_PASSES     = (PV_TOTAL + 7) / 8;           // passes over 8 warps
    constexpr int ROWS_PER_WARP = FWD_TC_BQ / (FWD_TC_NUM_THREADS / 32); // 4 rows/warp

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid     = threadIdx.x;

    const int64_t qi_block = (int64_t)blockIdx.x * FWD_TC_BQ;
    const int     bnh      = blockIdx.y;
    const int     b        = bnh / nh;
    const int     h        = bnh - b * nh;

    const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
    float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
    float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

    const int64_t q_sM = params.q_strideM;
    const int64_t k_sM = params.k_strideM;
    const int64_t v_sM = params.v_strideM;
    const int64_t o_sM = params.o_strideM;

    // ── Shared memory layout ──────────────────────────────────────────────────
    extern __shared__ float smem[];
    float* s_q      = smem;
    // Double-buffered s_kv: index stage by [0,1]
    float* s_kv_base = s_q      + FWD_TC_BQ * HD_PAD;
    float* s_kv[2]  = { s_kv_base, s_kv_base + FWD_TC_BK * HD_PAD };
    float* s_scores = s_kv_base + 2 * FWD_TC_BK * HD_PAD;  // stride = TC_BK (no pad)
    float* s_m      = s_scores + FWD_TC_BQ * FWD_TC_BK;
    float* s_l      = s_m      + FWD_TC_BQ;
    float* s_out    = s_l      + FWD_TC_BQ;
    float* s_pv     = s_out    + FWD_TC_BQ * HD_PAD;  // P@V result (separate buffer)

    // ── Initialize running state and output accumulator ───────────────────────
    for (int i = tid; i < FWD_TC_BQ; i += FWD_TC_NUM_THREADS) {
        s_m[i] = -INFINITY;
        s_l[i] =  0.0f;
    }
    for (int i = tid; i < FWD_TC_BQ * HD_PAD; i += FWD_TC_NUM_THREADS) {
        s_out[i] = 0.0f;
    }

    // ── Step 1: Cooperative load Q tile → s_q (Vectorized) ───────────────────
    {
        const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int q = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            const int64_t g_idx = (qi_block + q) * q_sM + d;
            
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (qi_block + q < T) {
                val = *(const float4*)&Q_bnh[g_idx];
            }
            *(float4*)&s_q[q * HD_PAD + d] = val;
        }
    }
    __syncthreads();

    const int actual_q = (int)(( (int64_t)FWD_TC_BQ < (T - qi_block) ) ? (int64_t)FWD_TC_BQ : (T - qi_block));
    if (actual_q <= 0) return;

    const int64_t max_kj = is_causal
        ? (( (qi_block + (int64_t)actual_q) < T ) ? (qi_block + (int64_t)actual_q) : T)
        : T;

    // ── Pre-fetch first K tile ───────────────────────────────────────────────
    {
        const int vec_total = (FWD_TC_BK * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int k = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            const int64_t g = 0LL + k;
            cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
        }
        cp_async_commit();
    }

    // ── Step 2: Main loop over KV tiles ──────────────────────────────────────
    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_TC_BK) {
        const int block_len = (int)(( (int64_t)FWD_TC_BK < (T - kj_block) ) ? (int64_t)FWD_TC_BK : (T - kj_block));
        const int64_t next_kj_block = kj_block + FWD_TC_BK;
        const bool has_next = (next_kj_block < max_kj);

        // ── 2a: Wait for K tile, start loading V tile ───────────────────────
        cp_async_wait_group<0>();
        __syncthreads(); // K tile ready in s_kv[0]

        // Load V tile into s_kv[1] async
        {
            const int vec_total = (FWD_TC_BK * HeadDim) / 4;
            for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                const int v = (i * 4) / HeadDim;
                const int d = (i * 4) % HeadDim;
                const int64_t g = kj_block + v;
                cp_async_l16(&s_kv[1][v * HD_PAD + d], &V_bnh[g * v_sM + d], g < T);
            }
            cp_async_commit();
        }

        // ── 2b: Score GEMM via WMMA — warps 0–3 ──────────────────────────────
        if (warp_id < 4) {
            const int m_tile = warp_id / SCORE_N_TILES;
            const int n_tile = warp_id % SCORE_N_TILES;
            wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k = 0; k < SCORE_K_TILES; ++k) {
                wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, s_q + m_tile * FWD_TC_WMMA_M * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
                wmma::load_matrix_sync(b_frag, s_kv[0] + n_tile * FWD_TC_WMMA_N * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + n_tile * FWD_TC_WMMA_N, acc, FWD_TC_BK, wmma::mem_row_major);
        }
        __syncthreads(); // s_scores ready

        // ── 2c: Online softmax ───────────────────────────────────────────────
        {
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int row = warp_id * ROWS_PER_WARP + r;
                const int64_t qi_global = qi_block + row;
                const bool qi_valid = (qi_global < T);

                float val = (lane_id < block_len && qi_valid) ? s_scores[row * FWD_TC_BK + lane_id] * scale : -INFINITY;
                if (is_causal && (kj_block + lane_id) > qi_global) val = -INFINITY;

                float row_max = val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

                const float m_old = s_m[row];
                const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : ((m_old == m_new) ? 1.0f : expf(m_old - m_new));

                #pragma unroll
                for (int d_base = 0; d_base < HeadDim; d_base += 128) {
                    int d = d_base + lane_id * 4;
                    if (d < HeadDim) {
                        float4* ptr = (float4*)&s_out[row * HD_PAD + d];
                        float4 v_out = *ptr;
                        v_out.x *= alpha; v_out.y *= alpha; v_out.z *= alpha; v_out.w *= alpha;
                        *ptr = v_out;
                    }
                }

                float exp_s = (lane_id < block_len && qi_valid && m_new > -INFINITY) ? expf(val - m_new) : 0.0f;
                if (dropout_p > 0.0f && dropout_mask != nullptr && qi_valid && lane_id < block_len)
                    exp_s *= dropout_mask[(bnh * T + qi_global) * T + (kj_block + lane_id)];
                s_scores[row * FWD_TC_BK + lane_id] = exp_s;

                float row_sum = exp_s;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

                if (lane_id == 0) {
                    s_l[row] = alpha * s_l[row] + row_sum;
                    s_m[row] = m_new;
                }
            }
        }
        __syncthreads();

        // ── 2d: Start pre-fetching next K tile if available ──────────────────
        if (has_next) {
            const int vec_total = (FWD_TC_BK * HeadDim) / 4;
            for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                const int k = (i * 4) / HeadDim;
                const int d = (i * 4) % HeadDim;
                const int64_t g = next_kj_block + k;
                cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
            }
            cp_async_commit();
        }

        // ── 2e: Wait for V tile, compute P@V GEMM ────────────────────────────
        cp_async_wait_group<1>(); // Wait for current V tile (s_kv[1])
        __syncthreads();

        {
            for (int pass = 0; pass < PV_PASSES; ++pass) {
                const int tile_id = warp_id + pass * 8;
                if (tile_id < PV_TOTAL) {
                    const int m_tile = tile_id / PV_N_TILES;
                    const int n_tile = tile_id % PV_N_TILES;
                    wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
                    wmma::fill_fragment(acc, 0.0f);

                    #pragma unroll
                    for (int k = 0; k < PV_K_TILES; ++k) {
                        wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(a_frag, s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + k * FWD_TC_WMMA_K, FWD_TC_BK);
                        wmma::load_matrix_sync(b_frag, s_kv[1] + k * FWD_TC_WMMA_K * HD_PAD + n_tile * FWD_TC_WMMA_N, HD_PAD);
                        round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                        wmma::mma_sync(acc, a_frag, b_frag, acc);
                    }
                    wmma::store_matrix_sync(s_pv + m_tile * FWD_TC_WMMA_M * HD_PAD + n_tile * FWD_TC_WMMA_N, acc, HD_PAD, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        // ── 2f: Accumulate P@V into s_out (Vectorized) ────────────────────────
        {
            const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
            for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                const int q = (i * 4) / HeadDim;
                const int d = (i * 4) % HeadDim;
                float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
                float4* pv_ptr  = (float4*)&s_pv[q * HD_PAD + d];
                float4 v_out = *out_ptr;
                float4 v_pv  = *pv_ptr;
                v_out.x += v_pv.x; v_out.y += v_pv.y; v_out.z += v_pv.z; v_out.w += v_pv.w;
                *out_ptr = v_out;
            }
        }
        __syncthreads();
    }

    // ── Step 3: Normalize s_out by s_l (Vectorized) ─────────────────────────
    {
        const int vec_total = (actual_q * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int q = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            const float li = s_l[q];
            const float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;
            
            float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
            float4 val = *out_ptr;
            val.x *= inv_l; val.y *= inv_l; val.z *= inv_l; val.w *= inv_l;
            *out_ptr = val;
        }
    }
    __syncthreads();

    // ── Step 4: Coalesced writeback s_out → global O (Vectorized) ───────────
    {
        const int vec_total = (actual_q * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int q = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            *(float4*)&O_bnh[(qi_block + q) * o_sM + d] = *(float4*)&s_out[q * HD_PAD + d];
        }
    }

    // ── Step 5: Write LSE ─────────────────────────────────────────────────────
    for (int i = tid; i < actual_q; i += FWD_TC_NUM_THREADS) {
        const float m = s_m[i];
        const float l = s_l[i];
        LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}

// ============================================================================
// TC Forward: Shared memory size
// ============================================================================
//   s_q + s_kv + s_out + s_pv : 4 × [TC_BQ × HD_PAD]
//   s_scores                  : [TC_BQ × TC_BK] (no padding)
//   s_m + s_l                 : 2 × TC_BQ
static size_t compute_fwd_tc_smem(int64_t hd) {
    const size_t hd_pad = (size_t)hd + FWD_TC_SMEM_PAD;
    return (5ULL * FWD_TC_BQ * hd_pad          // s_q, s_kv[2], s_out, s_pv
          + (size_t)FWD_TC_BQ * FWD_TC_BK      // s_scores
          + 2ULL * FWD_TC_BQ)                  // s_m, s_l
         * sizeof(float);
}

// ============================================================================
// TC Forward: Launch function
// ============================================================================
//   Falls back to the scalar kernel for HeadDim not divisible by 16 (WMMA_N=16)
//   or if the required smem exceeds the device limit.
static void launch_fwd_tc_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
    float dropout_p, const float* dropout_mask,
    int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    int64_t lse_strideB, int64_t lse_strideH,
    int grid_y)
{
    if (hd % FWD_TC_WMMA_N != 0) {
        launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
                          dropout_p, dropout_mask,
                          q_strideB, q_strideM, q_strideH,
                          k_strideB, k_strideM, k_strideH,
                          v_strideB, v_strideM, v_strideH,
                          o_strideB, o_strideM, o_strideH,
                          lse_strideB, lse_strideH,
                          grid_y);
        return;
    }

    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    const size_t smem = compute_fwd_tc_smem(hd);
    if ((int)smem > max_smem) {
        printf("fused_attn_forward_tc: hd=%d needs %zu bytes smem, "
               "device max is %d. Falling back to scalar kernel.\n",
               (int)hd, (size_t)smem, max_smem);
        launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
                          dropout_p, dropout_mask,
                          q_strideB, q_strideM, q_strideH,
                          k_strideB, k_strideM, k_strideH,
                          v_strideB, v_strideM, v_strideH,
                          o_strideB, o_strideM, o_strideH,
                          lse_strideB, lse_strideH,
                          grid_y);
        return;
    }

    const int grid_x = (int)((T + FWD_TC_BQ - 1) / FWD_TC_BQ);
    const dim3 grid(grid_x, grid_y);

    // Build strided params once.
    MemEfficientFwdParams params;
    params.Q = Q; params.K = K; params.V = V;
    params.O = O; params.LSE = LSE;
    params.B = (int)B; params.nh = (int)nh; params.T = (int)T;
    params.scale = scale; params.is_causal = is_causal;
    params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
    params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
    params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
    params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
    params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;

    #define LAUNCH_FWD_TC(HD) \
    do { \
        auto* kernel = fused_attn_forward_kernel_tc<HD>; \
        cudaFuncSetAttribute(kernel, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             (int)smem); \
        kernel<<<grid, FWD_TC_NUM_THREADS, smem>>>(params); \
    } while (0)

    switch ((int)hd) {
        case  16: LAUNCH_FWD_TC( 16); break;
        case  32: LAUNCH_FWD_TC( 32); break;
        case  48: LAUNCH_FWD_TC( 48); break;
        case  64: LAUNCH_FWD_TC( 64); break;
        case  80: LAUNCH_FWD_TC( 80); break;
        case  96: LAUNCH_FWD_TC( 96); break;
        case 128: LAUNCH_FWD_TC(128); break;
        case 160: LAUNCH_FWD_TC(160); break;
        case 192: LAUNCH_FWD_TC(192); break;
        case 256: LAUNCH_FWD_TC(256); break;
        default:
            launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
                              dropout_p, dropout_mask,
                              q_strideB, q_strideM, q_strideH,
                              k_strideB, k_strideM, k_strideH,
                              v_strideB, v_strideM, v_strideH,
                              o_strideB, o_strideM, o_strideH,
                              lse_strideB, lse_strideH,
                              grid_y);
            break;
    }
    #undef LAUNCH_FWD_TC
}

// Optimised kernel compiled for Ada Lovelace sm89 (A6000, RTX 4090, L40S).
// Defined in arch/AttentionForward_sm89.cu.
void fused_attn_forward_tc_sm89_cuda(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
    float dropout_p, const float* dropout_mask,
    int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    int64_t lse_strideB, int64_t lse_strideH,
    int grid_y,
    cudaStream_t stream = 0);

namespace cuda{

void mem_efficient_attn_forward(
    const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    float* lse,          int64_t lse_strideB, int64_t lse_strideH,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask)
    {
        if (hd > MAX_HD) {
            printf("mem_efficient_attn_forward: hd=%d exceeds MAX_HD=%d\n",
                (int)hd, MAX_HD);
            return;
        }
        float scale = 1.0f / sqrtf(static_cast<float>(hd));
        int grid_y = (int)(B * nh);
        launch_fwd_kernel(query, key, value, output, lse,
                        B, nh, T, hd, scale, is_causal,
                        dropout_p, dropout_mask,
                        q_strideB, q_strideM, q_strideH,
                        k_strideB, k_strideM, k_strideH,
                        v_strideB, v_strideM, v_strideH,
                        o_strideB, o_strideM, o_strideH,
                        lse_strideB, lse_strideH,
                        grid_y);
    }

void mem_efficient_attn_forward_tc(
    const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    float* lse,          int64_t lse_strideB, int64_t lse_strideH,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask)
    {
        if (hd > MAX_HD) {
            printf("mem_efficient_attn_forward_tc: hd=%d exceeds MAX_HD=%d\n",
                (int)hd, MAX_HD);
            return;
        }
        float scale = 1.0f / sqrtf(static_cast<float>(hd));
        int grid_y = (int)(B * nh);

        // Dispatch to Ada-tuned kernel when running on sm89 (A6000 Ada, RTX 4090, L40S).
        // The sm89 kernel uses BQ_TILE=64 for hd≤80 (all 8 warps active in
        // score GEMM, 2× K-reuse per block) and BQ_TILE=32 for hd≥96.
        // The original RTX 3060 path below is untouched.
        if (cuda::get_arch() == cuda::ArchFamily::Ada) {
        // if (false) {
            fused_attn_forward_tc_sm89_cuda(query, key, value, output, lse,
                                            B, nh, T, hd, scale, is_causal,
                                            dropout_p, dropout_mask,
                                            q_strideB, q_strideM, q_strideH,
                                            k_strideB, k_strideM, k_strideH,
                                            v_strideB, v_strideM, v_strideH,
                                            o_strideB, o_strideM, o_strideH,
                                            lse_strideB, lse_strideH,
                                            grid_y);
            return;
        }

        launch_fwd_tc_kernel(query, key, value, output, lse,
                            B, nh, T, hd, scale, is_causal,
                            dropout_p, dropout_mask,
                            q_strideB, q_strideM, q_strideH,
                            k_strideB, k_strideM, k_strideH,
                            v_strideB, v_strideM, v_strideH,
                            o_strideB, o_strideM, o_strideH,
                            lse_strideB, lse_strideH,
                            grid_y);
    }

} // namespace cuda
} // namespace OwnTensor 

// #include "ops/helpers/AttentionKernels.h"
// #include "ops/helpers/KernelDispatch.h"
// #include "ops/cuda/attention/AttentionCommon.cuh"

// namespace OwnTensor {

// // ============================================================================
// // Constants & Prefetch Helpers
// // ============================================================================

// __device__ __forceinline__ void cp_async_wait_all() {
//     asm volatile("cp.async.wait_all;\n" ::: "memory");
// }

// __device__ __forceinline__ void cp_async_commit() {
//     asm volatile("cp.async.commit_group;\n" ::: "memory");
// }

// template<int N>
// __device__ __forceinline__ void cp_async_wait_group() {
//     asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
// }

// __device__ __forceinline__ void cp_async_l16(void* smem_ptr, const void* global_ptr, bool pred) {
//     uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
//     asm volatile(
//         "{\n"
//         "  .reg .pred p;\n"
//         "  setp.ne.b32 p, %2, 0;\n"
//         "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
//         "  @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"
//         "}\n"
//         : : "r"(smem_addr), "l"(global_ptr), "r"((int)pred) : "memory");
// }

// // static constexpr int MAX_HD = 256;   // max supported head dimension

// // ============================================================================
// // Forward Kernel: Fused Memory-Efficient Attention (shared-memory-centric)
// // ============================================================================
// //
// // Computes: O = softmax(Q @ K^T / sqrt(hd)) @ V
// // WITHOUT materializing the full T x T attention matrix.
// //
// // Design: MULTIPLE threads per query row (mirrors the backward kernel)
// //
// //   Grid : (ceil(T / FWD_BQ), B * nh)
// //   Block: FWD_NUM_THREADS = 256
// //
// //   Thread mapping: 256 threads = 32 rows x 8 threads/row
// //     qi_local  = tid / 8    -- which query row (0..31)
// //     local_tid = tid % 8    -- position within the 8-thread row group
// //
// //   Shared memory layout (maximized -- everything shared lives here,
// //   just like the backward kernel):
// //
// //     s_q      [BQ x hd]    -- Q tile, loaded once, persistent
// //     s_kv     [BK x hd]    -- K tile then V tile (reused each iter)
// //     s_scores [BQ x BK]    -- score/attention-weight matrix
// //     s_m      [BQ]          -- running max per query row
// //     s_l      [BQ]          -- running sum per query row
// //
// //   Per thread:
// //     reg_out[hd/8]  -- output accumulator in registers (not smem)
// //                       Each of 8 threads owns a DIFFERENT slice of hd.
// //
// //   Why s_scores in shared memory?
// //     During Score GEMM: each thread computes 4 dot products (split by
// //     key columns). During P@V GEMM: each thread accumulates hd/8 output
// //     dims, reading ALL BK attention weights. Thread roles CHANGE between
// //     phases -- s_scores is the communication channel, exactly like
// //     ds_buf/p_buf in the backward kernel.
// //
// //   Why s_m/s_l in shared memory?
// //     Only local_tid==0 writes the updated max/sum. All 4 threads in the
// //     row group read it for rescaling. Mirrors LSE/D in the backward.
// //
// //   Algorithm (online softmax):
// //     1. Cooperative load Q tile -> s_q
// //     2. For each K/V tile:
// //        a. Cooperative load K tile -> s_kv
// //        b. Score GEMM: each thread computes 4 dots -> s_scores
// //        c. Online softmax via warp shuffle across 4-thread groups:
// //           partial max -> shuffle reduce -> update s_m
// //           rescale reg_out by exp(m_old - m_new)
// //           exp(scores) -> s_scores, partial sum -> shuffle -> s_l
// //        d. Cooperative load V tile -> s_kv
// //        e. P@V GEMM: read all BK weights from s_scores,
// //           accumulate hd/8 dims per thread in reg_out
// //     3. O = reg_out / s_l
// //     4. LSE = s_m + log(s_l)
// //

// // Forward tuning constants
// static constexpr int FWD_BQ              = 32;  // query rows per block
// static constexpr int FWD_BK              = 32;  // key rows per tile
// static constexpr int FWD_THREADS_PER_ROW = 8;   // threads per Q row (was 4, now 8 for occupancy)
// static constexpr int FWD_NUM_THREADS     = FWD_BQ * FWD_THREADS_PER_ROW;  // 256
// static constexpr int FWD_KJ_PER_THREAD   = FWD_BK / FWD_THREADS_PER_ROW; // 4
// static constexpr int FWD_SMEM_PAD        = 1;   // +1 padding to eliminate bank conflicts
// static constexpr int FWD_SCORE_STRIDE    = FWD_BK + FWD_SMEM_PAD;  // 33 (odd → no conflict)

// template<int HeadDim>
// __global__ void fused_attn_forward_kernel(MemEfficientFwdParams params)
// {
//     // Aliases so the body stays close to the pre-strided version.
//     const float* __restrict__ Q             = params.Q;
//     const float* __restrict__ K             = params.K;
//     const float* __restrict__ V             = params.V;
//     float*       __restrict__ O             = params.O;
//     float*       __restrict__ LSE           = params.LSE;
//     const int64_t T                         = params.T;
//     const int     nh                        = params.nh;
//     const float   scale                     = params.scale;
//     const bool    is_causal                 = params.is_causal;
//     const float   dropout_p                 = params.dropout_p;
//     const float* __restrict__ dropout_mask  = params.dropout_mask;

//     // All dimension-derived values are now compile-time constants
//     constexpr uint HD_PAD       = HeadDim + FWD_SMEM_PAD;   // padded stride (e.g. 65)
//     constexpr uint D_PER_THREAD = (HeadDim + FWD_THREADS_PER_ROW - 1) / FWD_THREADS_PER_ROW;

//     const uint qi_block = (int64_t)blockIdx.x * FWD_BQ;
//     const uint bnh      = blockIdx.y;
//     const uint tid      = threadIdx.x;
//     const int  b        = (int)bnh / nh;
//     const int  h        = (int)bnh - b * nh;

//     // ---- Thread-to-query-row mapping ----
//     // 8 consecutive threads share one query row:
//     //   threads 0-7     -> row 0
//     //   threads 8-15    -> row 1
//     //   ...
//     //   threads 248-255 -> row 31
//     //
//     // NOTE: the warp_id/lane_id layout from the coalescing doc breaks the
//     // warp shuffle reductions (they can't cross warp boundaries). The
//     // PRACTICAL FIX in that doc keeps this mapping and stages output through
//     // shared memory instead.
//     const int qi_local  = tid / FWD_THREADS_PER_ROW;   // 0..31
//     const int local_tid = tid % FWD_THREADS_PER_ROW;   // 0..7
//     const int64_t qi_global = qi_block + qi_local;
//     const bool qi_valid = (qi_global < T);

//     // ---- Pointers for this batch*head (strided, last-dim stride = 1) ----
//     const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
//     const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
//     const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
//     float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
//     float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

//     const int64_t q_sM = params.q_strideM;
//     const int64_t k_sM = params.k_strideM;
//     const int64_t v_sM = params.v_strideM;
//     const int64_t o_sM = params.o_strideM;

//     // ===================================================================
//     // Shared memory layout (padded to eliminate bank conflicts)
//     // ===================================================================
//     //
//     //   Row strides are padded by +1 so successive rows map to different
//     //   banks (stride 65 and 33 are odd → conflict-free across rows).
//     //
//     //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
//     //   |  s_q              |  s_kv             |  s_scores       | s_m | s_l |  s_out            |
//     //   |  [BQ x HD_PAD]    |  [BK x HD_PAD]    |  [BQ x (BK+1)] |[BQ] |[BQ] |  [BQ x HD_PAD]   |
//     //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
//     //
//     //   s_out: staging buffer for coalesced global writeback (same padded
//     //          stride as s_q → same conflict-free access pattern).
//     //
//     extern __shared__ float smem[];
//     float* s_q      = smem;                                              // [BQ x HD_PAD]
//     float* s_kv     = s_q      + FWD_BQ * HD_PAD;                       // [BK x HD_PAD]
//     float* s_scores = s_kv     + FWD_BK * HD_PAD;                       // [BQ x FWD_SCORE_STRIDE]
//     float* s_m      = s_scores + FWD_BQ * FWD_SCORE_STRIDE;             // [BQ]
//     float* s_l      = s_m      + FWD_BQ;                                // [BQ]
//     float* s_out    = s_l      + FWD_BQ;                                // [BQ x HD_PAD]

//     // ---- Initialize running max and sum in shared memory ----
//     if (local_tid == 0) {
//         s_m[qi_local] = -INFINITY;
//         s_l[qi_local] = 0.0f;
//     }

//     // ---- Output accumulator in registers ----
//     // HeadDim is now a compile-time constant, so the compiler knows
//     // D_PER_THREAD exactly and keeps reg_out entirely in ACTUAL registers
//     // (no spilling to local memory).
//     // Strided assignment: thread local_tid owns dimensions {local_tid, local_tid+8, local_tid+16, ...}
//     // i.e., reg_out[dd] accumulates output for dimension (local_tid + dd * FWD_THREADS_PER_ROW)

//     float reg_out[D_PER_THREAD];   // compile-time size → guaranteed registers
//     #pragma unroll
//     for (int i = 0; i < D_PER_THREAD; i++) reg_out[i] = 0.0f;

//     // ===================================================================
//     // Step 1: Cooperative load Q tile into s_q (ALL 256 threads help)
//     // ===================================================================
//     {
//         constexpr int64_t total = (int64_t)FWD_BQ * HeadDim;
//         for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
//             int q = (int)(i / HeadDim);
//             int d = (int)(i % HeadDim);
//             s_q[q * HD_PAD + d] = (qi_block + q < T) ? Q_bnh[(qi_block + q) * q_sM + d] : 0.0f;
//         }
//     }
//     __syncthreads();

//     const int actual_q = ((int)(T - qi_block) < FWD_BQ)
//                          ? (int)(T - qi_block) : FWD_BQ;
//     if (actual_q <= 0) return;

//     // ---- Causal early termination ----
//     const int64_t max_kj = is_causal
//         ? (((qi_block + actual_q) < T) ? (qi_block + actual_q) : T)
//         : T;

//     // ===================================================================
//     // Step 2: Main loop over key/value tiles
//     // ===================================================================
//     for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_BK) {
//         const int block_len = ((int)(T - kj_block) < FWD_BK)
//                               ? (int)(T - kj_block) : FWD_BK;

//         // ---------------------------------------------------------------
//         // Step 2a: Cooperative load K tile into s_kv (padded stride)
//         // ---------------------------------------------------------------
//         {
//             const int64_t total = (int64_t)block_len * HeadDim;
//             for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
//                 int k = (int)(i / HeadDim);
//                 int d = (int)(i % HeadDim);
//                 s_kv[k * HD_PAD + d] = K_bnh[(kj_block + k) * k_sM + d];
//             }
//         }
//         __syncthreads();    // K tile ready in s_kv

//         // ---------------------------------------------------------------
//         // Step 2b: Score GEMM -- Q[BQ x hd] @ K[BK x hd]^T -> s_scores
//         // ---------------------------------------------------------------
//         {
//             const int kj_start = local_tid * FWD_KJ_PER_THREAD;

//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 float dot;
//                 if (kj < block_len && qi_valid) {
//                     dot = 0.0f;
//                     #pragma unroll
//                     for (int d = 0; d < HeadDim; d++)
//                         dot += s_q[qi_local * HD_PAD + d] * s_kv[kj * HD_PAD + d];
//                     dot *= scale;

//                     // Causal mask
//                     if (is_causal && (kj_block + kj) > qi_global)
//                         dot = -INFINITY;
//                 } else {
//                     dot = -INFINITY;
//                 }
//                 s_scores[qi_local * FWD_SCORE_STRIDE + kj] = dot;
//             }
//         }
//         __syncthreads();    // all scores in s_scores

//         // ---------------------------------------------------------------
//         // Step 2c: Online softmax (warp shuffle across 8-thread groups)
//         // ---------------------------------------------------------------
//         {
//             const int kj_start = local_tid * FWD_KJ_PER_THREAD;

//             // (a) Partial row-max over this thread's 4 keys
//             float partial_max = -INFINITY;
//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 if (kj < block_len)
//                     partial_max = fmaxf(partial_max,
//                                         s_scores[qi_local * FWD_SCORE_STRIDE + kj]);
//             }

//             // (b) Warp shuffle: reduce max across 8 threads in the group
//             float row_max = partial_max;
//             row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
//             row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 2));
//             row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 4));

//             // (c) Read old running max from shared memory
//             float m_old = s_m[qi_local];
//             float m_new = fmaxf(m_old, row_max);

//             // (d) Correction factor for previous accumulations
//             float alpha;
//             if (m_old == -INFINITY)       alpha = 0.0f;
//             else if (m_old == m_new)      alpha = 1.0f;
//             else                          alpha = expf(m_old - m_new);

//             // (e) Rescale this thread's output accumulator slice
//             #pragma unroll
//             for (int i = 0; i < D_PER_THREAD; i++)
//                 reg_out[i] *= alpha;

//             // (f) Exponentiate scores, apply dropout, accumulate post-dropout sum.
//             //
//             // Step 2c is split so that s_l accumulates post-dropout weights
//             // (walkthrough Option 2): compute exp → apply mask → sum masked values.
//             // This ensures the final normalisation (reg_out / s_l) is correct
//             // without any post-hoc correction of s_l.
//             //
//             // Dropout uses Option B (pre-generated mask tensor, shape [B*nh, T, T]):
//             //   mask[bnh, qi, kj] already encodes the Bernoulli decision scaled by
//             //   1/(1-p), so multiplying exp_s by the mask value both zeroes dropped
//             //   entries and rescales surviving ones in one operation.
//             float partial_sum = 0.0f;
//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 float exp_s;
//                 if (kj < block_len && m_new > -INFINITY) {
//                     exp_s = expf(s_scores[qi_local * FWD_SCORE_STRIDE + kj] - m_new);
//                     // Apply pre-generated dropout mask (scale-and-zero in one multiply).
//                     // When dropout_p == 0 or no mask is provided this branch is skipped.
//                     if (dropout_p > 0.0f && dropout_mask != nullptr) {
//                         float m_val = dropout_mask[(bnh * T + qi_global) * T
//                                                    + (kj_block + kj)];
//                         exp_s *= m_val;
//                     }
//                 } else {
//                     exp_s = 0.0f;
//                 }
//                 s_scores[qi_local * FWD_SCORE_STRIDE + kj] = exp_s;
//                 partial_sum += exp_s;   // accumulate post-dropout sum → s_l correct
//             }
//             // Zero out padding
//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 if (kj >= block_len)
//                     s_scores[qi_local * FWD_SCORE_STRIDE + kj] = 0.0f;
//             }

//             // (g) Warp shuffle: reduce sum across 8 threads
//             float row_sum = partial_sum;
//             row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
//             row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
//             row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);

//             // (h) Update shared running state (one writer per row)
//             if (local_tid == 0) {
//                 s_l[qi_local] = alpha * s_l[qi_local] + row_sum;
//                 s_m[qi_local] = m_new;
//             }
//         }
//         __syncthreads();    // s_scores holds P, s_m/s_l updated

//         // ---------------------------------------------------------------
//         // Step 2d: Cooperative load V tile into s_kv (overwrites K, padded)
//         // ---------------------------------------------------------------
//         {
//             const int64_t total = (int64_t)block_len * HeadDim;
//             for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
//                 int v = (int)(i / HeadDim);
//                 int d = (int)(i % HeadDim);
//                 s_kv[v * HD_PAD + d] = V_bnh[(kj_block + v) * v_sM + d];
//             }
//         }
//         __syncthreads();    // V tile ready in s_kv

//         // ---------------------------------------------------------------
//         // Step 2e: P@V GEMM -- accumulate into reg_out
//         // ---------------------------------------------------------------
//         // kj iterates to FWD_BK (compile-time constant = 32) instead of
//         // block_len (runtime), so #pragma unroll 4 produces 8 fully-pipelined
//         // groups of 4 smem reads, overlapping latency with compute.
//         // Boundary guard is inside the kj loop; padded elements are 0 (set in
//         // Step 2c so this is safe). s_scores rows were already zeroed for kj
//         // >= block_len in the softmax step, so no extra cost.
//         #pragma unroll
//         for (int dd = 0; dd < D_PER_THREAD; dd++) {
//             // Strided assignment: thread local_tid owns every 8th dimension.
//             // Thread 0: d=0,8,16,...   Thread 1: d=1,9,17,...   Thread 7: d=7,15,23,...
//             // For fixed kj: bank(thread_i) = (kj*HD_PAD + dd*8 + i) % 32 → all 8 banks distinct → 0 conflicts.
//             int d = local_tid + dd * FWD_THREADS_PER_ROW;
//             if (d < HeadDim) {
//                 float acc = 0.0f;
//                 #pragma unroll 4
//                 for (int kj = 0; kj < FWD_BK; kj++) {
//                     if (kj < block_len)
//                         acc += s_scores[qi_local * FWD_SCORE_STRIDE + kj]
//                              * s_kv[kj * HD_PAD + d];
//                 }
//                 reg_out[dd] += acc;
//             }
//         }
//         __syncthreads();    // before next iteration overwrites s_kv/s_scores
//     }

//     // ===================================================================
//     // Step 3-4: Normalize into s_out (staged), then coalesced global write
//     // ===================================================================
//     //
//     // WHY TWO PHASES:
//     //   Direct write: thread i writes O[qi_global * HeadDim + d_start + dd].
//     //   Threads 0..7 all write to the SAME row (row 0), not adjacent rows,
//     //   so within a warp the 32 threads write to 4 different rows with
//     //   stride HeadDim*4 bytes → separate cache-line transactions per thread.
//     //
//     //   Phase A (scatter to s_out): each thread writes its d-slice into the
//     //   shared staging buffer at s_out[qi_local][d_start..d_end]. Same access
//     //   pattern as loading s_q → already proven conflict-free with HD_PAD.
//     //
//     //   Phase B (coalesced gather to global): all 256 threads stride linearly
//     //   over s_out and write to O. Adjacent threads → adjacent global addresses
//     //   → full 128-byte cache-line utilization per transaction.

//     // ---- Phase A: normalize and scatter to shared staging buffer ----
//     if (qi_valid) {
//         float li    = s_l[qi_local];
//         float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;

//         #pragma unroll
//         for (int dd = 0; dd < D_PER_THREAD; dd++) {
//             int d = local_tid + dd * FWD_THREADS_PER_ROW;
//             if (d < HeadDim)
//                 s_out[qi_local * HD_PAD + d] = reg_out[dd] * inv_l;
//         }
//     }
//     __syncthreads();

//     // ---- Phase B: cooperative coalesced writeback s_out -> O ----
//     // Threads stride linearly: tid 0 writes (row0, d0), tid 1 writes (row0, d1), …
//     // Adjacent tids → adjacent global memory addresses → coalesced.
//     {
//         const int64_t total = (int64_t)actual_q * HeadDim;
//         for (int64_t i = (int64_t)tid; i < total; i += FWD_NUM_THREADS) {
//             int q = (int)(i / HeadDim);
//             int d = (int)(i % HeadDim);
//             O_bnh[(qi_block + q) * o_sM + d] = s_out[q * HD_PAD + d];
//         }
//     }

//     // ---- Write LSE: one thread per row (local_tid == 0) ----
//     if (local_tid == 0 && qi_valid) {
//         float m = s_m[qi_local];
//         float l = s_l[qi_local];
//         LSE_bnh[qi_global] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
//     }
// }


// // ============================================================================
// // Forward launch function
// // ============================================================================
// //
// // Shared memory (padded, with s_out staging buffer):
// //   ((BQ + BK + BQ) * (hd + 1) + BQ * (BK + 1) + 2 * BQ) * sizeof(float)
// //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// //    s_q + s_kv + s_out, all with padded stride
// //
// //   hd=64:  (32+32+32)*65 + 32*33 + 64  =  7,360 floats =  29,440 bytes
// //   hd=128: (32+32+32)*129 + 32*33 + 64 = 12,544 floats =  50,176 bytes (opt-in)
// //   hd=256: (32+32+32)*257 + 32*33 + 64 = 24,768 floats =  99,072 bytes (opt-in)
// //

// static size_t compute_fwd_smem(int64_t hd) {
//     int64_t hd_pad = hd + FWD_SMEM_PAD;
//     return ((size_t)(FWD_BQ + FWD_BK + FWD_BQ) * hd_pad  // s_q + s_kv + s_out
//           + (size_t)FWD_BQ * FWD_SCORE_STRIDE              // s_scores
//           + (size_t)2 * FWD_BQ) * sizeof(float);           // s_m + s_l
// }

// static void launch_fwd_kernel(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y)
// {
//     // Build strided params once, reuse for all template dispatches below.
//     MemEfficientFwdParams params;
//     params.Q = Q; params.K = K; params.V = V;
//     params.O = O; params.LSE = LSE;
//     params.B = (int)B; params.nh = (int)nh; params.T = (int)T;
//     params.scale = scale; params.is_causal = is_causal;
//     params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
//     params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
//     params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
//     params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
//     params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
//     params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
//     // --- Query the device's shared memory limit ---
//     int deviceId;
//     cudaGetDevice(&deviceId);
//     int max_smem = 0;
//     cudaDeviceGetAttribute(&max_smem,
//                            cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

//     size_t smem = compute_fwd_smem(hd);

//     if ((int)smem > max_smem) {
//         printf("fused_attn_forward: hd=%d needs %zu bytes smem, "
//                "device max is %d. Cannot launch.\n",
//                (int)hd, smem, max_smem);
//         return;
//     }

//     // --- Grid: one block per query tile x one block per batch-head ---
//     int grid_x = (int)((T + FWD_BQ - 1) / FWD_BQ);
//     dim3 grid(grid_x, grid_y);

//     //* print the grid and block dims
//     // printf("The config for the kernel is: block{%d, 1} grid{%d, %d}\n", FWD_NUM_THREADS, grid.x, grid.y);
//     // exit(1);

//     // --- Template dispatch (same pattern as backward kernel) ---
//     // HeadDim must be a compile-time constant so reg_out stays in registers.
//     #define LAUNCH_FWD(HD) \
//     do { \
//         auto* kernel = fused_attn_forward_kernel<HD>; \
//         cudaFuncSetAttribute(kernel, \
//                              cudaFuncAttributeMaxDynamicSharedMemorySize, \
//                              (int)smem); \
//         kernel<<<grid, FWD_NUM_THREADS, smem>>>(params); \
//     } while (0)

//     switch ((int)hd) {
//         case   8: LAUNCH_FWD(  8); break;
//         case  16: LAUNCH_FWD( 16); break;
//         case  24: LAUNCH_FWD( 24); break;
//         case  32: LAUNCH_FWD( 32); break;
//         case  40: LAUNCH_FWD( 40); break;
//         case  48: LAUNCH_FWD( 48); break;
//         case  56: LAUNCH_FWD( 56); break;
//         case  64: LAUNCH_FWD( 64); break;
//         case  80: LAUNCH_FWD( 80); break;
//         case  96: LAUNCH_FWD( 96); break;
//         case 128: LAUNCH_FWD(128); break;
//         case 160: LAUNCH_FWD(160); break;
//         case 192: LAUNCH_FWD(192); break;
//         case 256: LAUNCH_FWD(256); break;
//         default:
//             printf("fused_attn_forward: unsupported head_dim %d\n", (int)hd);
//             break;
//     }
//     #undef LAUNCH_FWD
// }


// static constexpr int FWD_TC_BQ          = 32;   // query rows per block
// static constexpr int FWD_TC_BK          = 32;   // key/value rows per KV tile
// static constexpr int FWD_TC_NUM_THREADS = 256;  // 8 warps × 32 lanes
// static constexpr int FWD_TC_WMMA_M      = 16;   // WMMA tile rows
// static constexpr int FWD_TC_WMMA_N      = 16;   // WMMA tile cols (HeadDim must be multiple)
// static constexpr int FWD_TC_WMMA_K      = 8;    // WMMA tile K-dim for TF32
// static constexpr int FWD_TC_SMEM_PAD    = 4;    // +4 padding: makes HD_PAD a multiple of 4
//                                                 //   (required by wmma::store_matrix_sync for
//                                                 //   FP32 accumulators) while avoiding bank
//                                                 //   conflicts (hd is always ≡0 mod 16, so
//                                                 //   HD_PAD=hd+4 is never a multiple of 32).

// template<int HeadDim>
// __global__ void fused_attn_forward_kernel_tc(MemEfficientFwdParams params)
// {
//     // Aliases so the body stays close to the pre-strided version.
//     const float* __restrict__ Q             = params.Q;
//     const float* __restrict__ K             = params.K;
//     const float* __restrict__ V             = params.V;
//     float*       __restrict__ O             = params.O;
//     float*       __restrict__ LSE           = params.LSE;
//     const int64_t T                         = params.T;
//     const int     nh                        = params.nh;
//     const float   scale                     = params.scale;
//     const bool    is_causal                 = params.is_causal;
//     const float   dropout_p                 = params.dropout_p;
//     const float* __restrict__ dropout_mask  = params.dropout_mask;

//     static_assert(HeadDim % FWD_TC_WMMA_N == 0,
//                   "fused_attn_forward_kernel_tc: HeadDim must be divisible by 16");
//     using namespace nvcuda;

//     // ── Compile-time tile layout constants ───────────────────────────────────
//     constexpr int HD_PAD        = HeadDim + FWD_TC_SMEM_PAD;
//     // Score GEMM: Q[32×hd] @ K[32×hd]^T → s_scores[32×32]
//     //   2×2 grid of 16×16 output tiles; K-dimension split into hd/8 k-tiles.
//     constexpr int SCORE_N_TILES = FWD_TC_BK  / FWD_TC_WMMA_N;   // 2
//     constexpr int SCORE_K_TILES = HeadDim    / FWD_TC_WMMA_K;   // hd/8
//     // P@V GEMM: P[32×32] @ V[32×hd] → s_pv[32×hd]
//     //   2×(hd/16) output tiles; K-dimension (=32) split into 4 k-tiles.
//     constexpr int PV_N_TILES    = HeadDim    / FWD_TC_WMMA_N;   // hd/16
//     constexpr int PV_K_TILES    = FWD_TC_BK  / FWD_TC_WMMA_K;  // 4
//     constexpr int PV_TOTAL      = 2 * PV_N_TILES;               // total output tiles
//     constexpr int PV_PASSES     = (PV_TOTAL + 7) / 8;           // passes over 8 warps
//     constexpr int ROWS_PER_WARP = FWD_TC_BQ / (FWD_TC_NUM_THREADS / 32); // 4 rows/warp

//     const int warp_id = threadIdx.x / 32;
//     const int lane_id = threadIdx.x % 32;
//     const int tid     = threadIdx.x;

//     const int64_t qi_block = (int64_t)blockIdx.x * FWD_TC_BQ;
//     const int     bnh      = blockIdx.y;
//     const int     b        = bnh / nh;
//     const int     h        = bnh - b * nh;

//     const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
//     const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
//     const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
//     float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
//     float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

//     const int64_t q_sM = params.q_strideM;
//     const int64_t k_sM = params.k_strideM;
//     const int64_t v_sM = params.v_strideM;
//     const int64_t o_sM = params.o_strideM;

//     // ── Shared memory layout ──────────────────────────────────────────────────
//     extern __shared__ float smem[];
//     float* s_q      = smem;
//     // Double-buffered s_kv: index stage by [0,1]
//     float* s_kv_base = s_q      + FWD_TC_BQ * HD_PAD;
//     float* s_kv[2]  = { s_kv_base, s_kv_base + FWD_TC_BK * HD_PAD };
//     float* s_scores = s_kv_base + 2 * FWD_TC_BK * HD_PAD;  // stride = TC_BK (no pad)
//     float* s_m      = s_scores + FWD_TC_BQ * FWD_TC_BK;
//     float* s_l      = s_m      + FWD_TC_BQ;
//     float* s_out    = s_l      + FWD_TC_BQ;
//     float* s_pv     = s_out    + FWD_TC_BQ * HD_PAD;  // P@V result (separate buffer)

//     // ── Initialize running state and output accumulator ───────────────────────
//     for (int i = tid; i < FWD_TC_BQ; i += FWD_TC_NUM_THREADS) {
//         s_m[i] = -INFINITY;
//         s_l[i] =  0.0f;
//     }
//     for (int i = tid; i < FWD_TC_BQ * HD_PAD; i += FWD_TC_NUM_THREADS) {
//         s_out[i] = 0.0f;
//     }

//     // ── Step 1: Cooperative load Q tile → s_q (Vectorized) ───────────────────
//     {
//         const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int q = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             const int64_t g_idx = (qi_block + q) * q_sM + d;
            
//             float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
//             if (qi_block + q < T) {
//                 val = *(const float4*)&Q_bnh[g_idx];
//             }
//             *(float4*)&s_q[q * HD_PAD + d] = val;
//         }
//     }
//     __syncthreads();

//     const int actual_q = (int)(( (int64_t)FWD_TC_BQ < (T - qi_block) ) ? (int64_t)FWD_TC_BQ : (T - qi_block));
//     if (actual_q <= 0) return;

//     const int64_t max_kj = is_causal
//         ? (( (qi_block + (int64_t)actual_q) < T ) ? (qi_block + (int64_t)actual_q) : T)
//         : T;

//     // ── Pre-fetch first K tile ───────────────────────────────────────────────
//     {
//         const int vec_total = (FWD_TC_BK * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int k = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             const int64_t g = 0LL + k;
//             cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
//         }
//         cp_async_commit();
//     }

//     // ── Step 2: Main loop over KV tiles ──────────────────────────────────────
//     for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_TC_BK) {
//         const int block_len = (int)(( (int64_t)FWD_TC_BK < (T - kj_block) ) ? (int64_t)FWD_TC_BK : (T - kj_block));
//         const int64_t next_kj_block = kj_block + FWD_TC_BK;
//         const bool has_next = (next_kj_block < max_kj);

//         // ── 2a: Wait for K tile, start loading V tile ───────────────────────
//         cp_async_wait_group<0>();
//         __syncthreads(); // K tile ready in s_kv[0]

//         // Load V tile into s_kv[1] async
//         {
//             const int vec_total = (FWD_TC_BK * HeadDim) / 4;
//             for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//                 const int v = (i * 4) / HeadDim;
//                 const int d = (i * 4) % HeadDim;
//                 const int64_t g = kj_block + v;
//                 cp_async_l16(&s_kv[1][v * HD_PAD + d], &V_bnh[g * v_sM + d], g < T);
//             }
//             cp_async_commit();
//         }

//         // ── 2b: Score GEMM via WMMA — warps 0–3 ──────────────────────────────
//         if (warp_id < 4) {
//             const int m_tile = warp_id / SCORE_N_TILES;
//             const int n_tile = warp_id % SCORE_N_TILES;
//             wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
//             wmma::fill_fragment(acc, 0.0f);

//             #pragma unroll
//             for (int k = 0; k < SCORE_K_TILES; ++k) {
//                 wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
//                 wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;

//                 wmma::load_matrix_sync(a_frag, s_q + m_tile * FWD_TC_WMMA_M * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
//                 wmma::load_matrix_sync(b_frag, s_kv[0] + n_tile * FWD_TC_WMMA_N * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
//                 wmma::mma_sync(acc, a_frag, b_frag, acc);
//             }
//             wmma::store_matrix_sync(s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + n_tile * FWD_TC_WMMA_N, acc, FWD_TC_BK, wmma::mem_row_major);
//         }
//         __syncthreads(); // s_scores ready

//         // ── 2c: Online softmax ───────────────────────────────────────────────
//         {
//             for (int r = 0; r < ROWS_PER_WARP; ++r) {
//                 const int row = warp_id * ROWS_PER_WARP + r;
//                 const int64_t qi_global = qi_block + row;
//                 const bool qi_valid = (qi_global < T);

//                 float val = (lane_id < block_len && qi_valid) ? s_scores[row * FWD_TC_BK + lane_id] * scale : -INFINITY;
//                 if (is_causal && (kj_block + lane_id) > qi_global) val = -INFINITY;

//                 float row_max = val;
//                 #pragma unroll
//                 for (int off = 16; off > 0; off >>= 1)
//                     row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

//                 const float m_old = s_m[row];
//                 const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;
//                 const float alpha = (m_old == -INFINITY) ? 0.0f : ((m_old == m_new) ? 1.0f : expf(m_old - m_new));

//                 #pragma unroll
//                 for (int d_base = 0; d_base < HeadDim; d_base += 128) {
//                     int d = d_base + lane_id * 4;
//                     if (d < HeadDim) {
//                         float4* ptr = (float4*)&s_out[row * HD_PAD + d];
//                         float4 v_out = *ptr;
//                         v_out.x *= alpha; v_out.y *= alpha; v_out.z *= alpha; v_out.w *= alpha;
//                         *ptr = v_out;
//                     }
//                 }

//                 float exp_s = (lane_id < block_len && qi_valid && m_new > -INFINITY) ? expf(val - m_new) : 0.0f;
//                 if (dropout_p > 0.0f && dropout_mask != nullptr && qi_valid && lane_id < block_len)
//                     exp_s *= dropout_mask[(bnh * T + qi_global) * T + (kj_block + lane_id)];
//                 s_scores[row * FWD_TC_BK + lane_id] = exp_s;

//                 float row_sum = exp_s;
//                 #pragma unroll
//                 for (int off = 16; off > 0; off >>= 1)
//                     row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

//                 if (lane_id == 0) {
//                     s_l[row] = alpha * s_l[row] + row_sum;
//                     s_m[row] = m_new;
//                 }
//             }
//         }
//         __syncthreads();

//         // ── 2d: Start pre-fetching next K tile if available ──────────────────
//         if (has_next) {
//             const int vec_total = (FWD_TC_BK * HeadDim) / 4;
//             for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//                 const int k = (i * 4) / HeadDim;
//                 const int d = (i * 4) % HeadDim;
//                 const int64_t g = next_kj_block + k;
//                 cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
//             }
//             cp_async_commit();
//         }

//         // ── 2e: Wait for V tile, compute P@V GEMM ────────────────────────────
//         cp_async_wait_group<1>(); // Wait for current V tile (s_kv[1])
//         __syncthreads();

//         {
//             for (int pass = 0; pass < PV_PASSES; ++pass) {
//                 const int tile_id = warp_id + pass * 8;
//                 if (tile_id < PV_TOTAL) {
//                     const int m_tile = tile_id / PV_N_TILES;
//                     const int n_tile = tile_id % PV_N_TILES;
//                     wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
//                     wmma::fill_fragment(acc, 0.0f);

//                     #pragma unroll
//                     for (int k = 0; k < PV_K_TILES; ++k) {
//                         wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
//                         wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
//                         wmma::load_matrix_sync(a_frag, s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + k * FWD_TC_WMMA_K, FWD_TC_BK);
//                         wmma::load_matrix_sync(b_frag, s_kv[1] + k * FWD_TC_WMMA_K * HD_PAD + n_tile * FWD_TC_WMMA_N, HD_PAD);
//                         wmma::mma_sync(acc, a_frag, b_frag, acc);
//                     }
//                     wmma::store_matrix_sync(s_pv + m_tile * FWD_TC_WMMA_M * HD_PAD + n_tile * FWD_TC_WMMA_N, acc, HD_PAD, wmma::mem_row_major);
//                 }
//             }
//         }
//         __syncthreads();

//         // ── 2f: Accumulate P@V into s_out (Vectorized) ────────────────────────
//         {
//             const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
//             for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//                 const int q = (i * 4) / HeadDim;
//                 const int d = (i * 4) % HeadDim;
//                 float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
//                 float4* pv_ptr  = (float4*)&s_pv[q * HD_PAD + d];
//                 float4 v_out = *out_ptr;
//                 float4 v_pv  = *pv_ptr;
//                 v_out.x += v_pv.x; v_out.y += v_pv.y; v_out.z += v_pv.z; v_out.w += v_pv.w;
//                 *out_ptr = v_out;
//             }
//         }
//         __syncthreads();
//     }

//     // ── Step 3: Normalize s_out by s_l (Vectorized) ─────────────────────────
//     {
//         const int vec_total = (actual_q * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int q = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             const float li = s_l[q];
//             const float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;
            
//             float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
//             float4 val = *out_ptr;
//             val.x *= inv_l; val.y *= inv_l; val.z *= inv_l; val.w *= inv_l;
//             *out_ptr = val;
//         }
//     }
//     __syncthreads();

//     // ── Step 4: Coalesced writeback s_out → global O (Vectorized) ───────────
//     {
//         const int vec_total = (actual_q * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int q = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             *(float4*)&O_bnh[(qi_block + q) * o_sM + d] = *(float4*)&s_out[q * HD_PAD + d];
//         }
//     }

//     // ── Step 5: Write LSE ─────────────────────────────────────────────────────
//     for (int i = tid; i < actual_q; i += FWD_TC_NUM_THREADS) {
//         const float m = s_m[i];
//         const float l = s_l[i];
//         LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
//     }
// }

// // ============================================================================
// // TC Forward: Shared memory size
// // ============================================================================
// //   s_q + s_kv + s_out + s_pv : 4 × [TC_BQ × HD_PAD]
// //   s_scores                  : [TC_BQ × TC_BK] (no padding)
// //   s_m + s_l                 : 2 × TC_BQ
// static size_t compute_fwd_tc_smem(int64_t hd) {
//     const size_t hd_pad = (size_t)hd + FWD_TC_SMEM_PAD;
//     return (5ULL * FWD_TC_BQ * hd_pad          // s_q, s_kv[2], s_out, s_pv
//           + (size_t)FWD_TC_BQ * FWD_TC_BK      // s_scores
//           + 2ULL * FWD_TC_BQ)                  // s_m, s_l
//          * sizeof(float);
// }

// // ============================================================================
// // TC Forward: Launch function
// // ============================================================================
// //   Falls back to the scalar kernel for HeadDim not divisible by 16 (WMMA_N=16)
// //   or if the required smem exceeds the device limit.
// static void launch_fwd_tc_kernel(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y)
// {
//     if (hd % FWD_TC_WMMA_N != 0) {
//         launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
//                           dropout_p, dropout_mask,
//                           q_strideB, q_strideM, q_strideH,
//                           k_strideB, k_strideM, k_strideH,
//                           v_strideB, v_strideM, v_strideH,
//                           o_strideB, o_strideM, o_strideH,
//                           lse_strideB, lse_strideH,
//                           grid_y);
//         return;
//     }

//     int deviceId;
//     cudaGetDevice(&deviceId);
//     int max_smem = 0;
//     cudaDeviceGetAttribute(&max_smem,
//                            cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

//     const size_t smem = compute_fwd_tc_smem(hd);
//     if ((int)smem > max_smem) {
//         printf("fused_attn_forward_tc: hd=%d needs %zu bytes smem, "
//                "device max is %d. Falling back to scalar kernel.\n",
//                (int)hd, (size_t)smem, max_smem);
//         launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
//                           dropout_p, dropout_mask,
//                           q_strideB, q_strideM, q_strideH,
//                           k_strideB, k_strideM, k_strideH,
//                           v_strideB, v_strideM, v_strideH,
//                           o_strideB, o_strideM, o_strideH,
//                           lse_strideB, lse_strideH,
//                           grid_y);
//         return;
//     }

//     const int grid_x = (int)((T + FWD_TC_BQ - 1) / FWD_TC_BQ);
//     const dim3 grid(grid_x, grid_y);

//     // Build strided params once.
//     MemEfficientFwdParams params;
//     params.Q = Q; params.K = K; params.V = V;
//     params.O = O; params.LSE = LSE;
//     params.B = (int)B; params.nh = (int)nh; params.T = (int)T;
//     params.scale = scale; params.is_causal = is_causal;
//     params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
//     params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
//     params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
//     params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
//     params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
//     params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;

//     #define LAUNCH_FWD_TC(HD) \
//     do { \
//         auto* kernel = fused_attn_forward_kernel_tc<HD>; \
//         cudaFuncSetAttribute(kernel, \
//                              cudaFuncAttributeMaxDynamicSharedMemorySize, \
//                              (int)smem); \
//         kernel<<<grid, FWD_TC_NUM_THREADS, smem>>>(params); \
//     } while (0)

//     switch ((int)hd) {
//         case  16: LAUNCH_FWD_TC( 16); break;
//         case  32: LAUNCH_FWD_TC( 32); break;
//         case  48: LAUNCH_FWD_TC( 48); break;
//         case  64: LAUNCH_FWD_TC( 64); break;
//         case  80: LAUNCH_FWD_TC( 80); break;
//         case  96: LAUNCH_FWD_TC( 96); break;
//         case 128: LAUNCH_FWD_TC(128); break;
//         case 160: LAUNCH_FWD_TC(160); break;
//         case 192: LAUNCH_FWD_TC(192); break;
//         case 256: LAUNCH_FWD_TC(256); break;
//         default:
//             launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
//                               dropout_p, dropout_mask,
//                               q_strideB, q_strideM, q_strideH,
//                               k_strideB, k_strideM, k_strideH,
//                               v_strideB, v_strideM, v_strideH,
//                               o_strideB, o_strideM, o_strideH,
//                               lse_strideB, lse_strideH,
//                               grid_y);
//             break;
//     }
//     #undef LAUNCH_FWD_TC
// }

// // Optimised kernel compiled for Ada Lovelace sm89 (A6000, RTX 4090, L40S).
// // Defined in arch/AttentionForward_sm89.cu.
// void fused_attn_forward_tc_sm89_cuda(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y,
//     cudaStream_t stream = 0);

// // 3xTF32 (CUTLASS OpMultiplyAddFastF32) variant. Same signature.
// // Gated by USE_3XTF32=1 env var. Falls back to fused_attn_forward_tc_sm89_cuda
// // for HeadDim values it does not support (>80 with BQ=32,BK=32 smem layout).
// void fused_attn_forward_tc_sm89_3xtf32_cuda(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y,
//     cudaStream_t stream = 0);

// namespace cuda{

// void mem_efficient_attn_forward(
//     const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     float* lse,          int64_t lse_strideB, int64_t lse_strideH,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal,
//     float dropout_p, const float* dropout_mask)
//     {
//         if (hd > MAX_HD) {
//             printf("mem_efficient_attn_forward: hd=%d exceeds MAX_HD=%d\n",
//                 (int)hd, MAX_HD);
//             return;
//         }
//         float scale = 1.0f / sqrtf(static_cast<float>(hd));
//         int grid_y = (int)(B * nh);
//         launch_fwd_kernel(query, key, value, output, lse,
//                         B, nh, T, hd, scale, is_causal,
//                         dropout_p, dropout_mask,
//                         q_strideB, q_strideM, q_strideH,
//                         k_strideB, k_strideM, k_strideH,
//                         v_strideB, v_strideM, v_strideH,
//                         o_strideB, o_strideM, o_strideH,
//                         lse_strideB, lse_strideH,
//                         grid_y);
//     }

// void mem_efficient_attn_forward_tc(
//     const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     float* lse,          int64_t lse_strideB, int64_t lse_strideH,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal,
//     float dropout_p, const float* dropout_mask)
//     {
//         if (hd > MAX_HD) {
//             printf("mem_efficient_attn_forward_tc: hd=%d exceeds MAX_HD=%d\n",
//                 (int)hd, MAX_HD);
//             return;
//         }
//         float scale = 1.0f / sqrtf(static_cast<float>(hd));
//         int grid_y = (int)(B * nh);

//         // Dispatch to Ada-tuned kernel when running on sm89 (A6000 Ada, RTX 4090, L40S).
//         // The sm89 kernel uses BQ_TILE=64 for hd≤80 (all 8 warps active in
//         // score GEMM, 2× K-reuse per block) and BQ_TILE=32 for hd≥96.
//         // The original RTX 3060 path below is untouched.
//         if (cuda::get_arch() == cuda::ArchFamily::Ada) {
//         // if (false) {
//             // USE_3XTF32=1 routes to the CUTLASS-style OpMultiplyAddFastF32
//             // emulation that matches PyTorch's mem-efficient attention
//             // precision (FP32 inputs on sm80+). Default off (legacy 1xTF32).
//             // Read once at first call; cached for the rest of the run.
//             static const bool kUse3xTF32 = []() {
//                 const char* e = std::getenv("USE_3XTF32");
//                 return e && e[0] == '1';
//             }();
//             // The 3xTF32 kernel supports hd in {16, 32, 48, 64, 80}. For other
//             // hd values (or hd%16 != 0) it prints a warning and returns
//             // without launching, so fall back to the single-TF32 kernel here.
//             if (kUse3xTF32 && (hd % 16 == 0) && hd >= 16 && hd <= 80) {
//                 fused_attn_forward_tc_sm89_3xtf32_cuda(
//                     query, key, value, output, lse,
//                     B, nh, T, hd, scale, is_causal,
//                     dropout_p, dropout_mask,
//                     q_strideB, q_strideM, q_strideH,
//                     k_strideB, k_strideM, k_strideH,
//                     v_strideB, v_strideM, v_strideH,
//                     o_strideB, o_strideM, o_strideH,
//                     lse_strideB, lse_strideH,
//                     grid_y);
//                 return;
//             }
//             fused_attn_forward_tc_sm89_cuda(query, key, value, output, lse,
//                                             B, nh, T, hd, scale, is_causal,
//                                             dropout_p, dropout_mask,
//                                             q_strideB, q_strideM, q_strideH,
//                                             k_strideB, k_strideM, k_strideH,
//                                             v_strideB, v_strideM, v_strideH,
//                                             o_strideB, o_strideM, o_strideH,
//                                             lse_strideB, lse_strideH,
//                                             grid_y);
//             return;
//         }

//         launch_fwd_tc_kernel(query, key, value, output, lse,
//                             B, nh, T, hd, scale, is_causal,
//                             dropout_p, dropout_mask,
//                             q_strideB, q_strideM, q_strideH,
//                             k_strideB, k_strideM, k_strideH,
//                             v_strideB, v_strideM, v_strideH,
//                             o_strideB, o_strideM, o_strideH,
//                             lse_strideB, lse_strideH,
//                             grid_y);
//     }

// } // namespace cuda
// } // namespace OwnTensor 

//----------------------------------------------------------------------------------------------

// #include "ops/helpers/AttentionKernels.h"
// #include "ops/helpers/KernelDispatch.h"
// #include "ops/cuda/attention/AttentionCommon.cuh"

// namespace OwnTensor {

// // ============================================================================
// // Constants & Prefetch Helpers
// // ============================================================================

// __device__ __forceinline__ void cp_async_wait_all() {
//     asm volatile("cp.async.wait_all;\n" ::: "memory");
// }

// __device__ __forceinline__ void cp_async_commit() {
//     asm volatile("cp.async.commit_group;\n" ::: "memory");
// }

// template<int N>
// __device__ __forceinline__ void cp_async_wait_group() {
//     asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
// }

// __device__ __forceinline__ void cp_async_l16(void* smem_ptr, const void* global_ptr, bool pred) {
//     uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
//     asm volatile(
//         "{\n"
//         "  .reg .pred p;\n"
//         "  setp.ne.b32 p, %2, 0;\n"
//         "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
//         "  @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"
//         "}\n"
//         : : "r"(smem_addr), "l"(global_ptr), "r"((int)pred) : "memory");
// }

// // static constexpr int MAX_HD = 256;   // max supported head dimension

// // ============================================================================
// // Forward Kernel: Fused Memory-Efficient Attention (shared-memory-centric)
// // ============================================================================
// //
// // Computes: O = softmax(Q @ K^T / sqrt(hd)) @ V
// // WITHOUT materializing the full T x T attention matrix.
// //
// // Design: MULTIPLE threads per query row (mirrors the backward kernel)
// //
// //   Grid : (ceil(T / FWD_BQ), B * nh)
// //   Block: FWD_NUM_THREADS = 256
// //
// //   Thread mapping: 256 threads = 32 rows x 8 threads/row
// //     qi_local  = tid / 8    -- which query row (0..31)
// //     local_tid = tid % 8    -- position within the 8-thread row group
// //
// //   Shared memory layout (maximized -- everything shared lives here,
// //   just like the backward kernel):
// //
// //     s_q      [BQ x hd]    -- Q tile, loaded once, persistent
// //     s_kv     [BK x hd]    -- K tile then V tile (reused each iter)
// //     s_scores [BQ x BK]    -- score/attention-weight matrix
// //     s_m      [BQ]          -- running max per query row
// //     s_l      [BQ]          -- running sum per query row
// //
// //   Per thread:
// //     reg_out[hd/8]  -- output accumulator in registers (not smem)
// //                       Each of 8 threads owns a DIFFERENT slice of hd.
// //
// //   Why s_scores in shared memory?
// //     During Score GEMM: each thread computes 4 dot products (split by
// //     key columns). During P@V GEMM: each thread accumulates hd/8 output
// //     dims, reading ALL BK attention weights. Thread roles CHANGE between
// //     phases -- s_scores is the communication channel, exactly like
// //     ds_buf/p_buf in the backward kernel.
// //
// //   Why s_m/s_l in shared memory?
// //     Only local_tid==0 writes the updated max/sum. All 4 threads in the
// //     row group read it for rescaling. Mirrors LSE/D in the backward.
// //
// //   Algorithm (online softmax):
// //     1. Cooperative load Q tile -> s_q
// //     2. For each K/V tile:
// //        a. Cooperative load K tile -> s_kv
// //        b. Score GEMM: each thread computes 4 dots -> s_scores
// //        c. Online softmax via warp shuffle across 4-thread groups:
// //           partial max -> shuffle reduce -> update s_m
// //           rescale reg_out by exp(m_old - m_new)
// //           exp(scores) -> s_scores, partial sum -> shuffle -> s_l
// //        d. Cooperative load V tile -> s_kv
// //        e. P@V GEMM: read all BK weights from s_scores,
// //           accumulate hd/8 dims per thread in reg_out
// //     3. O = reg_out / s_l
// //     4. LSE = s_m + log(s_l)
// //

// // Forward tuning constants
// static constexpr int FWD_BQ              = 32;  // query rows per block
// static constexpr int FWD_BK              = 32;  // key rows per tile
// static constexpr int FWD_THREADS_PER_ROW = 8;   // threads per Q row (was 4, now 8 for occupancy)
// static constexpr int FWD_NUM_THREADS     = FWD_BQ * FWD_THREADS_PER_ROW;  // 256
// static constexpr int FWD_KJ_PER_THREAD   = FWD_BK / FWD_THREADS_PER_ROW; // 4
// static constexpr int FWD_SMEM_PAD        = 1;   // +1 padding to eliminate bank conflicts
// static constexpr int FWD_SCORE_STRIDE    = FWD_BK + FWD_SMEM_PAD;  // 33 (odd → no conflict)

// template<int HeadDim>
// __global__ void fused_attn_forward_kernel(MemEfficientFwdParams params)
// {
//     // Aliases so the body stays close to the pre-strided version.
//     const float* __restrict__ Q             = params.Q;
//     const float* __restrict__ K             = params.K;
//     const float* __restrict__ V             = params.V;
//     float*       __restrict__ O             = params.O;
//     float*       __restrict__ LSE           = params.LSE;
//     const int64_t T                         = params.T;
//     const int     nh                        = params.nh;
//     const float   scale                     = params.scale;
//     const bool    is_causal                 = params.is_causal;
//     const float   dropout_p                 = params.dropout_p;
//     const float* __restrict__ dropout_mask  = params.dropout_mask;

//     // All dimension-derived values are now compile-time constants
//     constexpr uint HD_PAD       = HeadDim + FWD_SMEM_PAD;   // padded stride (e.g. 65)
//     constexpr uint D_PER_THREAD = (HeadDim + FWD_THREADS_PER_ROW - 1) / FWD_THREADS_PER_ROW;

//     const uint qi_block = (int64_t)blockIdx.x * FWD_BQ;
//     const uint bnh      = blockIdx.y;
//     const uint tid      = threadIdx.x;
//     const int  b        = (int)bnh / nh;
//     const int  h        = (int)bnh - b * nh;

//     // ---- Thread-to-query-row mapping ----
//     // 8 consecutive threads share one query row:
//     //   threads 0-7     -> row 0
//     //   threads 8-15    -> row 1
//     //   ...
//     //   threads 248-255 -> row 31
//     //
//     // NOTE: the warp_id/lane_id layout from the coalescing doc breaks the
//     // warp shuffle reductions (they can't cross warp boundaries). The
//     // PRACTICAL FIX in that doc keeps this mapping and stages output through
//     // shared memory instead.
//     const int qi_local  = tid / FWD_THREADS_PER_ROW;   // 0..31
//     const int local_tid = tid % FWD_THREADS_PER_ROW;   // 0..7
//     const int64_t qi_global = qi_block + qi_local;
//     const bool qi_valid = (qi_global < T);

//     // ---- Pointers for this batch*head (strided, last-dim stride = 1) ----
//     const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
//     const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
//     const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
//     float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
//     float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

//     const int64_t q_sM = params.q_strideM;
//     const int64_t k_sM = params.k_strideM;
//     const int64_t v_sM = params.v_strideM;
//     const int64_t o_sM = params.o_strideM;

//     // ===================================================================
//     // Shared memory layout (padded to eliminate bank conflicts)
//     // ===================================================================
//     //
//     //   Row strides are padded by +1 so successive rows map to different
//     //   banks (stride 65 and 33 are odd → conflict-free across rows).
//     //
//     //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
//     //   |  s_q              |  s_kv             |  s_scores       | s_m | s_l |  s_out            |
//     //   |  [BQ x HD_PAD]    |  [BK x HD_PAD]    |  [BQ x (BK+1)] |[BQ] |[BQ] |  [BQ x HD_PAD]   |
//     //   +-------------------+-------------------+-----------------+-----+-----+-------------------+
//     //
//     //   s_out: staging buffer for coalesced global writeback (same padded
//     //          stride as s_q → same conflict-free access pattern).
//     //
//     extern __shared__ float smem[];
//     float* s_q      = smem;                                              // [BQ x HD_PAD]
//     float* s_kv     = s_q      + FWD_BQ * HD_PAD;                       // [BK x HD_PAD]
//     float* s_scores = s_kv     + FWD_BK * HD_PAD;                       // [BQ x FWD_SCORE_STRIDE]
//     float* s_m      = s_scores + FWD_BQ * FWD_SCORE_STRIDE;             // [BQ]
//     float* s_l      = s_m      + FWD_BQ;                                // [BQ]
//     float* s_out    = s_l      + FWD_BQ;                                // [BQ x HD_PAD]

//     // ---- Initialize running max and sum in shared memory ----
//     if (local_tid == 0) {
//         s_m[qi_local] = -INFINITY;
//         s_l[qi_local] = 0.0f;
//     }

//     // ---- Output accumulator in registers ----
//     // HeadDim is now a compile-time constant, so the compiler knows
//     // D_PER_THREAD exactly and keeps reg_out entirely in ACTUAL registers
//     // (no spilling to local memory).
//     // Strided assignment: thread local_tid owns dimensions {local_tid, local_tid+8, local_tid+16, ...}
//     // i.e., reg_out[dd] accumulates output for dimension (local_tid + dd * FWD_THREADS_PER_ROW)

//     float reg_out[D_PER_THREAD];   // compile-time size → guaranteed registers
//     #pragma unroll
//     for (int i = 0; i < D_PER_THREAD; i++) reg_out[i] = 0.0f;

//     // ===================================================================
//     // Step 1: Cooperative load Q tile into s_q (ALL 256 threads help)
//     // ===================================================================
//     {
//         constexpr int64_t total = (int64_t)FWD_BQ * HeadDim;
//         for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
//             int q = (int)(i / HeadDim);
//             int d = (int)(i % HeadDim);
//             s_q[q * HD_PAD + d] = (qi_block + q < T) ? Q_bnh[(qi_block + q) * q_sM + d] : 0.0f;
//         }
//     }
//     __syncthreads();

//     const int actual_q = ((int)(T - qi_block) < FWD_BQ)
//                          ? (int)(T - qi_block) : FWD_BQ;
//     if (actual_q <= 0) return;

//     // ---- Causal early termination ----
//     const int64_t max_kj = is_causal
//         ? (((qi_block + actual_q) < T) ? (qi_block + actual_q) : T)
//         : T;

//     // ===================================================================
//     // Step 2: Main loop over key/value tiles
//     // ===================================================================
//     for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_BK) {
//         const int block_len = ((int)(T - kj_block) < FWD_BK)
//                               ? (int)(T - kj_block) : FWD_BK;

//         // ---------------------------------------------------------------
//         // Step 2a: Cooperative load K tile into s_kv (padded stride)
//         // ---------------------------------------------------------------
//         {
//             const int64_t total = (int64_t)block_len * HeadDim;
//             for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
//                 int k = (int)(i / HeadDim);
//                 int d = (int)(i % HeadDim);
//                 s_kv[k * HD_PAD + d] = K_bnh[(kj_block + k) * k_sM + d];
//             }
//         }
//         __syncthreads();    // K tile ready in s_kv

//         // ---------------------------------------------------------------
//         // Step 2b: Score GEMM -- Q[BQ x hd] @ K[BK x hd]^T -> s_scores
//         // ---------------------------------------------------------------
//         {
//             const int kj_start = local_tid * FWD_KJ_PER_THREAD;

//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 float dot;
//                 if (kj < block_len && qi_valid) {
//                     dot = 0.0f;
//                     #pragma unroll
//                     for (int d = 0; d < HeadDim; d++)
//                         dot += s_q[qi_local * HD_PAD + d] * s_kv[kj * HD_PAD + d];
//                     dot *= scale;

//                     // Causal mask
//                     if (is_causal && (kj_block + kj) > qi_global)
//                         dot = -INFINITY;
//                 } else {
//                     dot = -INFINITY;
//                 }
//                 s_scores[qi_local * FWD_SCORE_STRIDE + kj] = dot;
//             }
//         }
//         __syncthreads();    // all scores in s_scores

//         // ---------------------------------------------------------------
//         // Step 2c: Online softmax (warp shuffle across 8-thread groups)
//         // ---------------------------------------------------------------
//         {
//             const int kj_start = local_tid * FWD_KJ_PER_THREAD;

//             // (a) Partial row-max over this thread's 4 keys
//             float partial_max = -INFINITY;
//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 if (kj < block_len)
//                     partial_max = fmaxf(partial_max,
//                                         s_scores[qi_local * FWD_SCORE_STRIDE + kj]);
//             }

//             // (b) Warp shuffle: reduce max across 8 threads in the group
//             float row_max = partial_max;
//             row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
//             row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 2));
//             row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 4));

//             // (c) Read old running max from shared memory
//             float m_old = s_m[qi_local];
//             float m_new = fmaxf(m_old, row_max);

//             // (d) Correction factor for previous accumulations
//             float alpha;
//             if (m_old == -INFINITY)       alpha = 0.0f;
//             else if (m_old == m_new)      alpha = 1.0f;
//             else                          alpha = expf(m_old - m_new);

//             // (e) Rescale this thread's output accumulator slice
//             #pragma unroll
//             for (int i = 0; i < D_PER_THREAD; i++)
//                 reg_out[i] *= alpha;

//             // (f) Exponentiate scores, apply dropout, accumulate post-dropout sum.
//             //
//             // Step 2c is split so that s_l accumulates post-dropout weights
//             // (walkthrough Option 2): compute exp → apply mask → sum masked values.
//             // This ensures the final normalisation (reg_out / s_l) is correct
//             // without any post-hoc correction of s_l.
//             //
//             // Dropout uses Option B (pre-generated mask tensor, shape [B*nh, T, T]):
//             //   mask[bnh, qi, kj] already encodes the Bernoulli decision scaled by
//             //   1/(1-p), so multiplying exp_s by the mask value both zeroes dropped
//             //   entries and rescales surviving ones in one operation.
//             float partial_sum = 0.0f;
//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 float exp_s;
//                 if (kj < block_len && m_new > -INFINITY) {
//                     exp_s = expf(s_scores[qi_local * FWD_SCORE_STRIDE + kj] - m_new);
//                     // Apply pre-generated dropout mask (scale-and-zero in one multiply).
//                     // When dropout_p == 0 or no mask is provided this branch is skipped.
//                     if (dropout_p > 0.0f && dropout_mask != nullptr) {
//                         float m_val = dropout_mask[(bnh * T + qi_global) * T
//                                                    + (kj_block + kj)];
//                         exp_s *= m_val;
//                     }
//                 } else {
//                     exp_s = 0.0f;
//                 }
//                 s_scores[qi_local * FWD_SCORE_STRIDE + kj] = exp_s;
//                 partial_sum += exp_s;   // accumulate post-dropout sum → s_l correct
//             }
//             // Zero out padding
//             #pragma unroll
//             for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
//                 int kj = kj_start + kk;
//                 if (kj >= block_len)
//                     s_scores[qi_local * FWD_SCORE_STRIDE + kj] = 0.0f;
//             }

//             // (g) Warp shuffle: reduce sum across 8 threads
//             float row_sum = partial_sum;
//             row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
//             row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
//             row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);

//             // (h) Update shared running state (one writer per row)
//             if (local_tid == 0) {
//                 s_l[qi_local] = alpha * s_l[qi_local] + row_sum;
//                 s_m[qi_local] = m_new;
//             }
//         }
//         __syncthreads();    // s_scores holds P, s_m/s_l updated

//         // ---------------------------------------------------------------
//         // Step 2d: Cooperative load V tile into s_kv (overwrites K, padded)
//         // ---------------------------------------------------------------
//         {
//             const int64_t total = (int64_t)block_len * HeadDim;
//             for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
//                 int v = (int)(i / HeadDim);
//                 int d = (int)(i % HeadDim);
//                 s_kv[v * HD_PAD + d] = V_bnh[(kj_block + v) * v_sM + d];
//             }
//         }
//         __syncthreads();    // V tile ready in s_kv

//         // ---------------------------------------------------------------
//         // Step 2e: P@V GEMM -- accumulate into reg_out
//         // ---------------------------------------------------------------
//         // kj iterates to FWD_BK (compile-time constant = 32) instead of
//         // block_len (runtime), so #pragma unroll 4 produces 8 fully-pipelined
//         // groups of 4 smem reads, overlapping latency with compute.
//         // Boundary guard is inside the kj loop; padded elements are 0 (set in
//         // Step 2c so this is safe). s_scores rows were already zeroed for kj
//         // >= block_len in the softmax step, so no extra cost.
//         #pragma unroll
//         for (int dd = 0; dd < D_PER_THREAD; dd++) {
//             // Strided assignment: thread local_tid owns every 8th dimension.
//             // Thread 0: d=0,8,16,...   Thread 1: d=1,9,17,...   Thread 7: d=7,15,23,...
//             // For fixed kj: bank(thread_i) = (kj*HD_PAD + dd*8 + i) % 32 → all 8 banks distinct → 0 conflicts.
//             int d = local_tid + dd * FWD_THREADS_PER_ROW;
//             if (d < HeadDim) {
//                 float acc = 0.0f;
//                 #pragma unroll 4
//                 for (int kj = 0; kj < FWD_BK; kj++) {
//                     if (kj < block_len)
//                         acc += s_scores[qi_local * FWD_SCORE_STRIDE + kj]
//                              * s_kv[kj * HD_PAD + d];
//                 }
//                 reg_out[dd] += acc;
//             }
//         }
//         __syncthreads();    // before next iteration overwrites s_kv/s_scores
//     }

//     // ===================================================================
//     // Step 3-4: Normalize into s_out (staged), then coalesced global write
//     // ===================================================================
//     //
//     // WHY TWO PHASES:
//     //   Direct write: thread i writes O[qi_global * HeadDim + d_start + dd].
//     //   Threads 0..7 all write to the SAME row (row 0), not adjacent rows,
//     //   so within a warp the 32 threads write to 4 different rows with
//     //   stride HeadDim*4 bytes → separate cache-line transactions per thread.
//     //
//     //   Phase A (scatter to s_out): each thread writes its d-slice into the
//     //   shared staging buffer at s_out[qi_local][d_start..d_end]. Same access
//     //   pattern as loading s_q → already proven conflict-free with HD_PAD.
//     //
//     //   Phase B (coalesced gather to global): all 256 threads stride linearly
//     //   over s_out and write to O. Adjacent threads → adjacent global addresses
//     //   → full 128-byte cache-line utilization per transaction.

//     // ---- Phase A: normalize and scatter to shared staging buffer ----
//     if (qi_valid) {
//         float li    = s_l[qi_local];
//         float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;

//         #pragma unroll
//         for (int dd = 0; dd < D_PER_THREAD; dd++) {
//             int d = local_tid + dd * FWD_THREADS_PER_ROW;
//             if (d < HeadDim)
//                 s_out[qi_local * HD_PAD + d] = reg_out[dd] * inv_l;
//         }
//     }
//     __syncthreads();

//     // ---- Phase B: cooperative coalesced writeback s_out -> O ----
//     // Threads stride linearly: tid 0 writes (row0, d0), tid 1 writes (row0, d1), …
//     // Adjacent tids → adjacent global memory addresses → coalesced.
//     {
//         const int64_t total = (int64_t)actual_q * HeadDim;
//         for (int64_t i = (int64_t)tid; i < total; i += FWD_NUM_THREADS) {
//             int q = (int)(i / HeadDim);
//             int d = (int)(i % HeadDim);
//             O_bnh[(qi_block + q) * o_sM + d] = s_out[q * HD_PAD + d];
//         }
//     }

//     // ---- Write LSE: one thread per row (local_tid == 0) ----
//     if (local_tid == 0 && qi_valid) {
//         float m = s_m[qi_local];
//         float l = s_l[qi_local];
//         LSE_bnh[qi_global] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
//     }
// }


// // ============================================================================
// // Forward launch function
// // ============================================================================
// //
// // Shared memory (padded, with s_out staging buffer):
// //   ((BQ + BK + BQ) * (hd + 1) + BQ * (BK + 1) + 2 * BQ) * sizeof(float)
// //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// //    s_q + s_kv + s_out, all with padded stride
// //
// //   hd=64:  (32+32+32)*65 + 32*33 + 64  =  7,360 floats =  29,440 bytes
// //   hd=128: (32+32+32)*129 + 32*33 + 64 = 12,544 floats =  50,176 bytes (opt-in)
// //   hd=256: (32+32+32)*257 + 32*33 + 64 = 24,768 floats =  99,072 bytes (opt-in)
// //

// static size_t compute_fwd_smem(int64_t hd) {
//     int64_t hd_pad = hd + FWD_SMEM_PAD;
//     return ((size_t)(FWD_BQ + FWD_BK + FWD_BQ) * hd_pad  // s_q + s_kv + s_out
//           + (size_t)FWD_BQ * FWD_SCORE_STRIDE              // s_scores
//           + (size_t)2 * FWD_BQ) * sizeof(float);           // s_m + s_l
// }

// static void launch_fwd_kernel(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y)
// {
//     // Build strided params once, reuse for all template dispatches below.
//     MemEfficientFwdParams params;
//     params.Q = Q; params.K = K; params.V = V;
//     params.O = O; params.LSE = LSE;
//     params.B = (int)B; params.nh = (int)nh; params.T = (int)T;
//     params.scale = scale; params.is_causal = is_causal;
//     params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
//     params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
//     params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
//     params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
//     params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
//     params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
//     // --- Query the device's shared memory limit ---
//     int deviceId;
//     cudaGetDevice(&deviceId);
//     int max_smem = 0;
//     cudaDeviceGetAttribute(&max_smem,
//                            cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

//     size_t smem = compute_fwd_smem(hd);

//     if ((int)smem > max_smem) {
//         printf("fused_attn_forward: hd=%d needs %zu bytes smem, "
//                "device max is %d. Cannot launch.\n",
//                (int)hd, smem, max_smem);
//         return;
//     }

//     // --- Grid: one block per query tile x one block per batch-head ---
//     int grid_x = (int)((T + FWD_BQ - 1) / FWD_BQ);
//     dim3 grid(grid_x, grid_y);

//     //* print the grid and block dims
//     // printf("The config for the kernel is: block{%d, 1} grid{%d, %d}\n", FWD_NUM_THREADS, grid.x, grid.y);
//     // exit(1);

//     // --- Template dispatch (same pattern as backward kernel) ---
//     // HeadDim must be a compile-time constant so reg_out stays in registers.
//     #define LAUNCH_FWD(HD) \
//     do { \
//         auto* kernel = fused_attn_forward_kernel<HD>; \
//         cudaFuncSetAttribute(kernel, \
//                              cudaFuncAttributeMaxDynamicSharedMemorySize, \
//                              (int)smem); \
//         kernel<<<grid, FWD_NUM_THREADS, smem>>>(params); \
//     } while (0)

//     switch ((int)hd) {
//         case   8: LAUNCH_FWD(  8); break;
//         case  16: LAUNCH_FWD( 16); break;
//         case  24: LAUNCH_FWD( 24); break;
//         case  32: LAUNCH_FWD( 32); break;
//         case  40: LAUNCH_FWD( 40); break;
//         case  48: LAUNCH_FWD( 48); break;
//         case  56: LAUNCH_FWD( 56); break;
//         case  64: LAUNCH_FWD( 64); break;
//         case  80: LAUNCH_FWD( 80); break;
//         case  96: LAUNCH_FWD( 96); break;
//         case 128: LAUNCH_FWD(128); break;
//         case 160: LAUNCH_FWD(160); break;
//         case 192: LAUNCH_FWD(192); break;
//         case 256: LAUNCH_FWD(256); break;
//         default:
//             printf("fused_attn_forward: unsupported head_dim %d\n", (int)hd);
//             break;
//     }
//     #undef LAUNCH_FWD
// }


// static constexpr int FWD_TC_BQ          = 32;   // query rows per block
// static constexpr int FWD_TC_BK          = 32;   // key/value rows per KV tile
// static constexpr int FWD_TC_NUM_THREADS = 256;  // 8 warps × 32 lanes
// static constexpr int FWD_TC_WMMA_M      = 16;   // WMMA tile rows
// static constexpr int FWD_TC_WMMA_N      = 16;   // WMMA tile cols (HeadDim must be multiple)
// static constexpr int FWD_TC_WMMA_K      = 8;    // WMMA tile K-dim for TF32
// static constexpr int FWD_TC_SMEM_PAD    = 4;    // +4 padding: makes HD_PAD a multiple of 4
//                                                 //   (required by wmma::store_matrix_sync for
//                                                 //   FP32 accumulators) while avoiding bank
//                                                 //   conflicts (hd is always ≡0 mod 16, so
//                                                 //   HD_PAD=hd+4 is never a multiple of 32).

// template<int HeadDim>
// __global__ void fused_attn_forward_kernel_tc(MemEfficientFwdParams params)
// {
//     // Aliases so the body stays close to the pre-strided version.
//     const float* __restrict__ Q             = params.Q;
//     const float* __restrict__ K             = params.K;
//     const float* __restrict__ V             = params.V;
//     float*       __restrict__ O             = params.O;
//     float*       __restrict__ LSE           = params.LSE;
//     const int64_t T                         = params.T;
//     const int     nh                        = params.nh;
//     const float   scale                     = params.scale;
//     const bool    is_causal                 = params.is_causal;
//     const float   dropout_p                 = params.dropout_p;
//     const float* __restrict__ dropout_mask  = params.dropout_mask;

//     static_assert(HeadDim % FWD_TC_WMMA_N == 0,
//                   "fused_attn_forward_kernel_tc: HeadDim must be divisible by 16");
//     using namespace nvcuda;

//     // ── Compile-time tile layout constants ───────────────────────────────────
//     constexpr int HD_PAD        = HeadDim + FWD_TC_SMEM_PAD;
//     // Score GEMM: Q[32×hd] @ K[32×hd]^T → s_scores[32×32]
//     //   2×2 grid of 16×16 output tiles; K-dimension split into hd/8 k-tiles.
//     constexpr int SCORE_N_TILES = FWD_TC_BK  / FWD_TC_WMMA_N;   // 2
//     constexpr int SCORE_K_TILES = HeadDim    / FWD_TC_WMMA_K;   // hd/8
//     // P@V GEMM: P[32×32] @ V[32×hd] → s_pv[32×hd]
//     //   2×(hd/16) output tiles; K-dimension (=32) split into 4 k-tiles.
//     constexpr int PV_N_TILES    = HeadDim    / FWD_TC_WMMA_N;   // hd/16
//     constexpr int PV_K_TILES    = FWD_TC_BK  / FWD_TC_WMMA_K;  // 4
//     constexpr int PV_TOTAL      = 2 * PV_N_TILES;               // total output tiles
//     constexpr int PV_PASSES     = (PV_TOTAL + 7) / 8;           // passes over 8 warps
//     constexpr int ROWS_PER_WARP = FWD_TC_BQ / (FWD_TC_NUM_THREADS / 32); // 4 rows/warp

//     const int warp_id = threadIdx.x / 32;
//     const int lane_id = threadIdx.x % 32;
//     const int tid     = threadIdx.x;

//     const int64_t qi_block = (int64_t)blockIdx.x * FWD_TC_BQ;
//     const int     bnh      = blockIdx.y;
//     const int     b        = bnh / nh;
//     const int     h        = bnh - b * nh;

//     const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
//     const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
//     const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
//     float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
//     float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

//     const int64_t q_sM = params.q_strideM;
//     const int64_t k_sM = params.k_strideM;
//     const int64_t v_sM = params.v_strideM;
//     const int64_t o_sM = params.o_strideM;

//     // ── Shared memory layout ──────────────────────────────────────────────────
//     extern __shared__ float smem[];
//     float* s_q      = smem;
//     // Double-buffered s_kv: index stage by [0,1]
//     float* s_kv_base = s_q      + FWD_TC_BQ * HD_PAD;
//     float* s_kv[2]  = { s_kv_base, s_kv_base + FWD_TC_BK * HD_PAD };
//     float* s_scores = s_kv_base + 2 * FWD_TC_BK * HD_PAD;  // stride = TC_BK (no pad)
//     float* s_m      = s_scores + FWD_TC_BQ * FWD_TC_BK;
//     float* s_l      = s_m      + FWD_TC_BQ;
//     float* s_out    = s_l      + FWD_TC_BQ;
//     float* s_pv     = s_out    + FWD_TC_BQ * HD_PAD;  // P@V result (separate buffer)

//     // ── Initialize running state and output accumulator ───────────────────────
//     for (int i = tid; i < FWD_TC_BQ; i += FWD_TC_NUM_THREADS) {
//         s_m[i] = -INFINITY;
//         s_l[i] =  0.0f;
//     }
//     for (int i = tid; i < FWD_TC_BQ * HD_PAD; i += FWD_TC_NUM_THREADS) {
//         s_out[i] = 0.0f;
//     }

//     // ── Step 1: Cooperative load Q tile → s_q (Vectorized) ───────────────────
//     {
//         const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int q = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             const int64_t g_idx = (qi_block + q) * q_sM + d;
            
//             float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
//             if (qi_block + q < T) {
//                 val = *(const float4*)&Q_bnh[g_idx];
//             }
//             *(float4*)&s_q[q * HD_PAD + d] = val;
//         }
//     }
//     __syncthreads();

//     const int actual_q = (int)(( (int64_t)FWD_TC_BQ < (T - qi_block) ) ? (int64_t)FWD_TC_BQ : (T - qi_block));
//     if (actual_q <= 0) return;

//     const int64_t max_kj = is_causal
//         ? (( (qi_block + (int64_t)actual_q) < T ) ? (qi_block + (int64_t)actual_q) : T)
//         : T;

//     // ── Pre-fetch first K tile ───────────────────────────────────────────────
//     {
//         const int vec_total = (FWD_TC_BK * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int k = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             const int64_t g = 0LL + k;
//             cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
//         }
//         cp_async_commit();
//     }

//     // ── Step 2: Main loop over KV tiles ──────────────────────────────────────
//     for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_TC_BK) {
//         const int block_len = (int)(( (int64_t)FWD_TC_BK < (T - kj_block) ) ? (int64_t)FWD_TC_BK : (T - kj_block));
//         const int64_t next_kj_block = kj_block + FWD_TC_BK;
//         const bool has_next = (next_kj_block < max_kj);

//         // ── 2a: Wait for K tile, start loading V tile ───────────────────────
//         cp_async_wait_group<0>();
//         __syncthreads(); // K tile ready in s_kv[0]

//         // Load V tile into s_kv[1] async
//         {
//             const int vec_total = (FWD_TC_BK * HeadDim) / 4;
//             for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//                 const int v = (i * 4) / HeadDim;
//                 const int d = (i * 4) % HeadDim;
//                 const int64_t g = kj_block + v;
//                 cp_async_l16(&s_kv[1][v * HD_PAD + d], &V_bnh[g * v_sM + d], g < T);
//             }
//             cp_async_commit();
//         }

//         // ── 2b: Score GEMM via WMMA — warps 0–3 ──────────────────────────────
//         if (warp_id < 4) {
//             const int m_tile = warp_id / SCORE_N_TILES;
//             const int n_tile = warp_id % SCORE_N_TILES;
//             wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
//             wmma::fill_fragment(acc, 0.0f);

//             #pragma unroll
//             for (int k = 0; k < SCORE_K_TILES; ++k) {
//                 wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
//                 wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;

//                 wmma::load_matrix_sync(a_frag, s_q + m_tile * FWD_TC_WMMA_M * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
//                 wmma::load_matrix_sync(b_frag, s_kv[0] + n_tile * FWD_TC_WMMA_N * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
//                 wmma::mma_sync(acc, a_frag, b_frag, acc);
//             }
//             wmma::store_matrix_sync(s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + n_tile * FWD_TC_WMMA_N, acc, FWD_TC_BK, wmma::mem_row_major);
//         }
//         __syncthreads(); // s_scores ready

//         // ── 2c: Online softmax ───────────────────────────────────────────────
//         {
//             for (int r = 0; r < ROWS_PER_WARP; ++r) {
//                 const int row = warp_id * ROWS_PER_WARP + r;
//                 const int64_t qi_global = qi_block + row;
//                 const bool qi_valid = (qi_global < T);

//                 float val = (lane_id < block_len && qi_valid) ? s_scores[row * FWD_TC_BK + lane_id] * scale : -INFINITY;
//                 if (is_causal && (kj_block + lane_id) > qi_global) val = -INFINITY;

//                 float row_max = val;
//                 #pragma unroll
//                 for (int off = 16; off > 0; off >>= 1)
//                     row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

//                 const float m_old = s_m[row];
//                 const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;
//                 const float alpha = (m_old == -INFINITY) ? 0.0f : ((m_old == m_new) ? 1.0f : expf(m_old - m_new));

//                 #pragma unroll
//                 for (int d_base = 0; d_base < HeadDim; d_base += 128) {
//                     int d = d_base + lane_id * 4;
//                     if (d < HeadDim) {
//                         float4* ptr = (float4*)&s_out[row * HD_PAD + d];
//                         float4 v_out = *ptr;
//                         v_out.x *= alpha; v_out.y *= alpha; v_out.z *= alpha; v_out.w *= alpha;
//                         *ptr = v_out;
//                     }
//                 }

//                 float exp_s = (lane_id < block_len && qi_valid && m_new > -INFINITY) ? expf(val - m_new) : 0.0f;
//                 if (dropout_p > 0.0f && dropout_mask != nullptr && qi_valid && lane_id < block_len)
//                     exp_s *= dropout_mask[(bnh * T + qi_global) * T + (kj_block + lane_id)];
//                 s_scores[row * FWD_TC_BK + lane_id] = exp_s;

//                 float row_sum = exp_s;
//                 #pragma unroll
//                 for (int off = 16; off > 0; off >>= 1)
//                     row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

//                 if (lane_id == 0) {
//                     s_l[row] = alpha * s_l[row] + row_sum;
//                     s_m[row] = m_new;
//                 }
//             }
//         }
//         __syncthreads();

//         // ── 2d: Start pre-fetching next K tile if available ──────────────────
//         if (has_next) {
//             const int vec_total = (FWD_TC_BK * HeadDim) / 4;
//             for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//                 const int k = (i * 4) / HeadDim;
//                 const int d = (i * 4) % HeadDim;
//                 const int64_t g = next_kj_block + k;
//                 cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
//             }
//             cp_async_commit();
//         }

//         // ── 2e: Wait for V tile, compute P@V GEMM ────────────────────────────
//         cp_async_wait_group<1>(); // Wait for current V tile (s_kv[1])
//         __syncthreads();

//         {
//             for (int pass = 0; pass < PV_PASSES; ++pass) {
//                 const int tile_id = warp_id + pass * 8;
//                 if (tile_id < PV_TOTAL) {
//                     const int m_tile = tile_id / PV_N_TILES;
//                     const int n_tile = tile_id % PV_N_TILES;
//                     wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
//                     wmma::fill_fragment(acc, 0.0f);

//                     #pragma unroll
//                     for (int k = 0; k < PV_K_TILES; ++k) {
//                         wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
//                         wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
//                         wmma::load_matrix_sync(a_frag, s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + k * FWD_TC_WMMA_K, FWD_TC_BK);
//                         wmma::load_matrix_sync(b_frag, s_kv[1] + k * FWD_TC_WMMA_K * HD_PAD + n_tile * FWD_TC_WMMA_N, HD_PAD);
//                         wmma::mma_sync(acc, a_frag, b_frag, acc);
//                     }
//                     wmma::store_matrix_sync(s_pv + m_tile * FWD_TC_WMMA_M * HD_PAD + n_tile * FWD_TC_WMMA_N, acc, HD_PAD, wmma::mem_row_major);
//                 }
//             }
//         }
//         __syncthreads();

//         // ── 2f: Accumulate P@V into s_out (Vectorized) ────────────────────────
//         {
//             const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
//             for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//                 const int q = (i * 4) / HeadDim;
//                 const int d = (i * 4) % HeadDim;
//                 float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
//                 float4* pv_ptr  = (float4*)&s_pv[q * HD_PAD + d];
//                 float4 v_out = *out_ptr;
//                 float4 v_pv  = *pv_ptr;
//                 v_out.x += v_pv.x; v_out.y += v_pv.y; v_out.z += v_pv.z; v_out.w += v_pv.w;
//                 *out_ptr = v_out;
//             }
//         }
//         __syncthreads();
//     }

//     // ── Step 3: Normalize s_out by s_l (Vectorized) ─────────────────────────
//     {
//         const int vec_total = (actual_q * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int q = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             const float li = s_l[q];
//             const float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;
            
//             float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
//             float4 val = *out_ptr;
//             val.x *= inv_l; val.y *= inv_l; val.z *= inv_l; val.w *= inv_l;
//             *out_ptr = val;
//         }
//     }
//     __syncthreads();

//     // ── Step 4: Coalesced writeback s_out → global O (Vectorized) ───────────
//     {
//         const int vec_total = (actual_q * HeadDim) / 4;
//         for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
//             const int q = (i * 4) / HeadDim;
//             const int d = (i * 4) % HeadDim;
//             *(float4*)&O_bnh[(qi_block + q) * o_sM + d] = *(float4*)&s_out[q * HD_PAD + d];
//         }
//     }

//     // ── Step 5: Write LSE ─────────────────────────────────────────────────────
//     for (int i = tid; i < actual_q; i += FWD_TC_NUM_THREADS) {
//         const float m = s_m[i];
//         const float l = s_l[i];
//         LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
//     }
// }

// // ============================================================================
// // TC Forward: Shared memory size
// // ============================================================================
// //   s_q + s_kv + s_out + s_pv : 4 × [TC_BQ × HD_PAD]
// //   s_scores                  : [TC_BQ × TC_BK] (no padding)
// //   s_m + s_l                 : 2 × TC_BQ
// static size_t compute_fwd_tc_smem(int64_t hd) {
//     const size_t hd_pad = (size_t)hd + FWD_TC_SMEM_PAD;
//     return (5ULL * FWD_TC_BQ * hd_pad          // s_q, s_kv[2], s_out, s_pv
//           + (size_t)FWD_TC_BQ * FWD_TC_BK      // s_scores
//           + 2ULL * FWD_TC_BQ)                  // s_m, s_l
//          * sizeof(float);
// }

// // ============================================================================
// // TC Forward: Launch function
// // ============================================================================
// //   Falls back to the scalar kernel for HeadDim not divisible by 16 (WMMA_N=16)
// //   or if the required smem exceeds the device limit.
// static void launch_fwd_tc_kernel(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y)
// {
//     if (hd % FWD_TC_WMMA_N != 0) {
//         launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
//                           dropout_p, dropout_mask,
//                           q_strideB, q_strideM, q_strideH,
//                           k_strideB, k_strideM, k_strideH,
//                           v_strideB, v_strideM, v_strideH,
//                           o_strideB, o_strideM, o_strideH,
//                           lse_strideB, lse_strideH,
//                           grid_y);
//         return;
//     }

//     int deviceId;
//     cudaGetDevice(&deviceId);
//     int max_smem = 0;
//     cudaDeviceGetAttribute(&max_smem,
//                            cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

//     const size_t smem = compute_fwd_tc_smem(hd);
//     if ((int)smem > max_smem) {
//         printf("fused_attn_forward_tc: hd=%d needs %zu bytes smem, "
//                "device max is %d. Falling back to scalar kernel.\n",
//                (int)hd, (size_t)smem, max_smem);
//         launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
//                           dropout_p, dropout_mask,
//                           q_strideB, q_strideM, q_strideH,
//                           k_strideB, k_strideM, k_strideH,
//                           v_strideB, v_strideM, v_strideH,
//                           o_strideB, o_strideM, o_strideH,
//                           lse_strideB, lse_strideH,
//                           grid_y);
//         return;
//     }

//     const int grid_x = (int)((T + FWD_TC_BQ - 1) / FWD_TC_BQ);
//     const dim3 grid(grid_x, grid_y);

//     // Build strided params once.
//     MemEfficientFwdParams params;
//     params.Q = Q; params.K = K; params.V = V;
//     params.O = O; params.LSE = LSE;
//     params.B = (int)B; params.nh = (int)nh; params.T = (int)T;
//     params.scale = scale; params.is_causal = is_causal;
//     params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
//     params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
//     params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
//     params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
//     params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
//     params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;

//     #define LAUNCH_FWD_TC(HD) \
//     do { \
//         auto* kernel = fused_attn_forward_kernel_tc<HD>; \
//         cudaFuncSetAttribute(kernel, \
//                              cudaFuncAttributeMaxDynamicSharedMemorySize, \
//                              (int)smem); \
//         kernel<<<grid, FWD_TC_NUM_THREADS, smem>>>(params); \
//     } while (0)

//     switch ((int)hd) {
//         case  16: LAUNCH_FWD_TC( 16); break;
//         case  32: LAUNCH_FWD_TC( 32); break;
//         case  48: LAUNCH_FWD_TC( 48); break;
//         case  64: LAUNCH_FWD_TC( 64); break;
//         case  80: LAUNCH_FWD_TC( 80); break;
//         case  96: LAUNCH_FWD_TC( 96); break;
//         case 128: LAUNCH_FWD_TC(128); break;
//         case 160: LAUNCH_FWD_TC(160); break;
//         case 192: LAUNCH_FWD_TC(192); break;
//         case 256: LAUNCH_FWD_TC(256); break;
//         default:
//             launch_fwd_kernel(Q, K, V, O, LSE, B, nh, T, hd, scale, is_causal,
//                               dropout_p, dropout_mask,
//                               q_strideB, q_strideM, q_strideH,
//                               k_strideB, k_strideM, k_strideH,
//                               v_strideB, v_strideM, v_strideH,
//                               o_strideB, o_strideM, o_strideH,
//                               lse_strideB, lse_strideH,
//                               grid_y);
//             break;
//     }
//     #undef LAUNCH_FWD_TC
// }

// // Optimised kernel compiled for Ada Lovelace sm89 (A6000, RTX 4090, L40S).
// // Defined in arch/AttentionForward_sm89.cu.
// void fused_attn_forward_tc_sm89_cuda(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y,
//     cudaStream_t stream = 0);

// // 3xTF32 (CUTLASS OpMultiplyAddFastF32) variant. Same signature.
// // Gated by USE_3XTF32=1 env var. Falls back to fused_attn_forward_tc_sm89_cuda
// // for HeadDim values it does not support (>80 with BQ=32,BK=32 smem layout).
// void fused_attn_forward_tc_sm89_3xtf32_cuda(
//     const float* Q, const float* K, const float* V,
//     float* O, float* LSE,
//     int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
//     float dropout_p, const float* dropout_mask,
//     int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     int64_t lse_strideB, int64_t lse_strideH,
//     int grid_y,
//     cudaStream_t stream = 0);

// namespace cuda{

// void mem_efficient_attn_forward(
//     const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     float* lse,          int64_t lse_strideB, int64_t lse_strideH,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal,
//     float dropout_p, const float* dropout_mask)
//     {
//         if (hd > MAX_HD) {
//             printf("mem_efficient_attn_forward: hd=%d exceeds MAX_HD=%d\n",
//                 (int)hd, MAX_HD);
//             return;
//         }
//         float scale = 1.0f / sqrtf(static_cast<float>(hd));
//         int grid_y = (int)(B * nh);
//         launch_fwd_kernel(query, key, value, output, lse,
//                         B, nh, T, hd, scale, is_causal,
//                         dropout_p, dropout_mask,
//                         q_strideB, q_strideM, q_strideH,
//                         k_strideB, k_strideM, k_strideH,
//                         v_strideB, v_strideM, v_strideH,
//                         o_strideB, o_strideM, o_strideH,
//                         lse_strideB, lse_strideH,
//                         grid_y);
//     }

// void mem_efficient_attn_forward_tc(
//     const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//     const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//     const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//     float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//     float* lse,          int64_t lse_strideB, int64_t lse_strideH,
//     int64_t B, int64_t nh, int64_t T, int64_t hd,
//     bool is_causal,
//     float dropout_p, const float* dropout_mask)
//     {
//         if (hd > MAX_HD) {
//             printf("mem_efficient_attn_forward_tc: hd=%d exceeds MAX_HD=%d\n",
//                 (int)hd, MAX_HD);
//             return;
//         }
//         float scale = 1.0f / sqrtf(static_cast<float>(hd));
//         int grid_y = (int)(B * nh);

//         // Dispatch to Ada-tuned kernel when running on sm89 (A6000 Ada, RTX 4090, L40S).
//         // The sm89 kernel uses BQ_TILE=64 for hd≤80 (all 8 warps active in
//         // score GEMM, 2× K-reuse per block) and BQ_TILE=32 for hd≥96.
//         // The original RTX 3060 path below is untouched.
//         if (cuda::get_arch() == cuda::ArchFamily::Ada) {
//         // if (false) {
//             // USE_3XTF32=1 routes to the CUTLASS-style OpMultiplyAddFastF32
//             // emulation that matches PyTorch's mem-efficient attention
//             // precision (FP32 inputs on sm80+). Default off (legacy 1xTF32).
//             // Read once at first call; cached for the rest of the run.
//             static const bool kUse3xTF32 = []() {
//                 const char* e = std::getenv("USE_3XTF32");
//                 return e && e[0] == '1';
//             }();
//             // The 3xTF32 kernel supports hd in {16, 32, 48, 64, 80}. For other
//             // hd values (or hd%16 != 0) it prints a warning and returns
//             // without launching, so fall back to the single-TF32 kernel here.
//             if (kUse3xTF32 && (hd % 16 == 0) && hd >= 16 && hd <= 80) {
//                 fused_attn_forward_tc_sm89_3xtf32_cuda(
//                     query, key, value, output, lse,
//                     B, nh, T, hd, scale, is_causal,
//                     dropout_p, dropout_mask,
//                     q_strideB, q_strideM, q_strideH,
//                     k_strideB, k_strideM, k_strideH,
//                     v_strideB, v_strideM, v_strideH,
//                     o_strideB, o_strideM, o_strideH,
//                     lse_strideB, lse_strideH,
//                     grid_y);
//                 return;
//             }
//             fused_attn_forward_tc_sm89_cuda(query, key, value, output, lse,
//                                             B, nh, T, hd, scale, is_causal,
//                                             dropout_p, dropout_mask,
//                                             q_strideB, q_strideM, q_strideH,
//                                             k_strideB, k_strideM, k_strideH,
//                                             v_strideB, v_strideM, v_strideH,
//                                             o_strideB, o_strideM, o_strideH,
//                                             lse_strideB, lse_strideH,
//                                             grid_y);
//             return;
//         }

//         launch_fwd_tc_kernel(query, key, value, output, lse,
//                             B, nh, T, hd, scale, is_causal,
//                             dropout_p, dropout_mask,
//                             q_strideB, q_strideM, q_strideH,
//                             k_strideB, k_strideM, k_strideH,
//                             v_strideB, v_strideM, v_strideH,
//                             o_strideB, o_strideM, o_strideH,
//                             lse_strideB, lse_strideH,
//                             grid_y);
//     }

// } // namespace cuda
// } // namespace OwnTensor 