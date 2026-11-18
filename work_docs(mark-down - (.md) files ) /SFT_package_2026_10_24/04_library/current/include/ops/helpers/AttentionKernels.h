#pragma once
#include <cstdint>
#include <memory>
#include "dtype/Types.h"   // bfloat16_t (host+device safe)

namespace OwnTensor {
namespace cuda {

//=========================
// MemoryEfficient Backend
//=========================

/**
* @brief Compute log-sum-exp per row of attention scores.
*
* Read-only kernel  256)
* @param is_causal     whether to apply lower-triangular causal mask
* @param dropout_p     dropout probability in [0, 1); pass 0 to disable
* @param dropout_mask  pre-generated mask tensor of shape [B*nh, T, T] where
*                      each element is either 0 (dropped) or 1/(1-dropout_p)
*                      (kept and rescaled); pass nullptr when dropout_p == 0
*/
void mem_efficient_attn_forward(
   const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
   const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
   const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
   float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
   float* lse,          int64_t lse_strideB, int64_t lse_strideH,
   int64_t B, int64_t nh, int64_t T, int64_t hd,
   bool is_causal,
  float dropout_p = 0.0f,
  const float* dropout_mask = nullptr,
  // Optional per-head max pre-softmax logit output [nh] for QK-Clip (MuonClip).
  // nullptr (default) disables the capture; the kernel path is unchanged.
  float* max_logits = nullptr
);

void mem_efficient_attn_forward_tc(
   const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
   const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
   const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
   float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
   float* lse,          int64_t lse_strideB, int64_t lse_strideH,
   int64_t B, int64_t nh, int64_t T, int64_t hd,
   bool is_causal,
  float dropout_p = 0.0f,
  const float* dropout_mask = nullptr,
  // Optional per-head max pre-softmax logit output [nh] for QK-Clip (MuonClip).
  float* max_logits = nullptr
);
/**
* @brief Unified memory-efficient attention backward pass for dQ, dK, dV.
*
* Architecture (PyTorch CUTLASS FMHA style):
*   1. Precompute D[i] = dot(dO[i], O[i]) in a lightweight kernel (one warp per row)
*   2. Single KV-tile-centric kernel computes dQ, dK, dV together:
*      - Each block owns one KV tile, sweeps all Q rows
*      - Phase A: warp-parallel score recompute + dQ accumulation in registers
*      - Phase B: cooperative dK/dV reduction (no per-element atomics)
*      - dQ written via atomicAdd; dK/dV written once from shared memory
*
* @param D_buf    (B*nh, T) scratch buffer for precomputed D values (pre-allocated)
*/
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
   bool skip_grad_zero = false  // when true, kernel skips its internal dQ/dK/dV
                                // memsets — caller must pre-zero (used by packed
                                // SDPA where dQ/dK/dV are strided views into one
                                // dqkv buffer and a contiguous memset would corrupt it)
);


//========================
// FlashAttention Backend (GQA, bf16 tensor-core)
//========================

/**
* @brief Grouped-query flash-attention forward (bf16, WMMA tensor cores).
*
* Reads ONE contiguous, block-concatenated, bf16 packed buffer:
*     [ Q : [B, Nq_heads,  T, hd]   (dense)
*       K : [B, Nkv_heads, T, hd]   (dense)
*       V : [B, Nkv_heads, T, hd] ] (dense)
* i.e. the whole Q tensor, then whole K, then whole V — NOT per-token
* interleaved. Nq_heads must be a multiple of Nkv_heads; query head h attends
* KV head h / (Nq_heads / Nkv_heads).
*
* Writes O as [B, Nq_heads, T, hd] (bf16) and LSE as [B, Nq_heads, T] (fp32).
* Constraints: hd in {64, 128}; non-causal: T a multiple of 32. Causal: Hopper
* v62 (hd 128, T % 256 == 0), B300 hd 64 (T % 256 == 0), every other case on any
* GPU but a B300 the generic causal WMMA kernel (since 2026-10-24), any T.
*/
void gqa_flash_attn_forward(
   const bfloat16_t* qkv,     // block-concatenated [Q | K | V]
   bfloat16_t*       output,  // [B, Nq_heads, T, hd]
   float*            lse,     // [B, Nq_heads, T]
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   bool is_causal
);

/**
* v2 packed rows (added 2026-10-10): several whole chats per row of T tokens, each attending only to
* itself. Given as a per-chat list -- design A, the cu_seqlens way of FlashAttention varlen,
* TE/Megatron THD, PyTorch varlen_attn, TRL and torchtitan VarlenInnerAttention: the kernels cut
* their work per segment instead of per row. (Design B, one seg_start value per token, was removed
* on 2026-10-15: A is faster and one representation keeps one tested path.)
* Build ONE plan per batch from the segments of every row -- chats AND each row's pad tail as its
* own segment; together they must cover every token of every row exactly once (checked, throws) --
* and pass the same plan to gqa_flash_attn_forward_qkv and gqa_flash_attn_backward_qkv.
* The plan owns device memory: free it with gqa_chat_plan_free (or hold it as a GqaChatPlanPtr).
* RoPE positions are not part of the plan: the caller gives fused_rms_rope positions that restart
* at 0 for every segment (as TRL's collator builds position_ids from the packed seq_lengths).
*/
struct GqaChatPlan {
    int32_t* fwd_units = nullptr;   // device, 4 ints per forward unit (v62): {b, t0, L, (cpl << 8) | hq}
                                    //   2026-11-13 (mixed units, see mx_hkv below): the 4th int w can also be
                                    //   bit 31 set = a MIXED unit: group 0 runs head (w & 0xff), group 1 runs head
                                    //     ((w >> 8) & 0xff), both the segment's LAST 64-row query tile;
                                    //   bit 30 set = a plain couple of a head whose last pair went to a mixed
                                    //     unit: its pair count is ceil(L/128) - 1, cpl = (w >> 8) & 0x3fffff.
    int      n_fwd_units = 0;
    int32_t* bwd_tiles = nullptr;   // device, 4 ints per Vj1d backward key tile (64 keys): {b, t0, L, kt}
    int      n_bwd_tiles = 0;
    int32_t* vj2_blocks = nullptr;  // device, 8 ints per Vj2 backward block (128 keys x one query head):
                                    //   {b, t0, L, kt, hq, dqt0, 0, 0}, dqt0 = padded dQ tile of its first step
    int      n_vj2_blocks = 0;
    int32_t* dq_tiles = nullptr;    // device, 4 ints per padded 64-row dQ tile of ONE query head (the same list
    int      n_dq_tiles = 0;        //   for every head): {b, row0, nrows, 0} (GQA_bwd_vj2_sm90.cu, bwd_prep_vj2_chats)
    int B = 0, S = 0, Hq = 0;       // the shape the plan was built for (checked at every call)
    int32_t* storage = nullptr;     // the ONE device allocation the lists point into
    // Generic (non-Hopper, non-B300) causal kernels, added 2026-10-24: one entry per segment, FA2 varlen's
    // cu_seqlens list. Each kernel cuts its own tiles from the segment start, so the list does not depend on
    // the tile size. Order: (row, start).
    int32_t* gen_segs = nullptr;    // device, 4 ints per segment: {b, t0, L, 0}
    int      n_gen_segs = 0;
    int      gen_max_len = 0;       // the longest segment (the grid's tile axis)
    // Mixed units, "MX" (added 2026-11-13; work doc 45.194, records profiling_reports/attn_short_chats_2026_11_13).
    // WHAT: a forward unit is one segment x one query head x a pair of 64-row query tiles, one tile for each of the two
    // compute warpgroups. A segment with an ODD number of 64-row tiles (every chat of <= 64 tokens, 129..192, ...) left
    // the second group of its last pair with no rows: it only ran the handshake. With mixed units that group runs the
    // same last tile of ANOTHER query head of the SAME KV head (same K / V, same key tiles: the loads do not change).
    // WHY (H200, 8 rows x 4,096, heads 18/6, forward, against the plan of before): -8.0% on rows of 32-160-token chats,
    // -2.7% on 300-1,500, -0.4..-1.05% on real SFT rows, no change on one chat per row; results unchanged (the suite's
    // "packed chat == the same chat alone, bit for bit" passes).
    // mx_hkv: 0 = this plan has no mixed unit (the forward runs exactly the kernel of before); otherwise the number of
    // KV heads the mixed units were built for. The forward refuses a plan built for another KV-head count: such a plan
    // would pair heads that read different K / V. The backward lists do not depend on it.
    int      mx_hkv = 0;
};
// seg_b/seg_t0/seg_len: HOST arrays of n_seg segments (row, start, length). Hq <= 255.
// The lists go up with one asynchronous copy on the current stream: the call does not wait for the GPU.
// 2026-11-13: Hkv = the number of KV heads of the attention call the plan is for. Hkv > 0 lets the builder make mixed
// units (see GqaChatPlan::mx_hkv; needs Hq / Hkv >= 2). Hkv = 0 (the default, so every caller of before compiles
// unchanged) gives the plan of before. The environment variable BLU_GQA_MX=0 turns the mixing off for a whole process
// (A/B tests), whatever Hkv is. The builder mixes only when the mixed units are at least 2% of the units the plain
// list would have (BLU_GQA_MX_MIN_PCT changes the limit; measured: below that share the longer unit decode of the
// mixed-unit kernel costs more than the shared tiles save), so a plan with few odd-length chats is the plan of before.
// Old declaration (kept, not deleted):
// void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
//                          const int32_t* seg_len, int n_seg, int B, int S, int Hq);
void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
                         const int32_t* seg_len, int n_seg, int B, int S, int Hq, int Hkv = 0);
void gqa_chat_plan_free(GqaChatPlan& plan);
// The plan as a shared handle (2026-10-15): the autograd attention op keeps it until the backward of the
// micro-step it was built for has run; it is freed when the last holder lets go.
using GqaChatPlanPtr = std::shared_ptr<const GqaChatPlan>;
// Old declaration (kept, not deleted):
// GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
//                                   int n_seg, int B, int S, int Hq);
GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
                                  int n_seg, int B, int S, int Hq, int Hkv = 0);   // Hkv: see gqa_chat_plan_build

/**
* @brief Grouped-query flash-attention forward, same kernels as
* gqa_flash_attn_forward but taking Q, K, V as three independent device
* buffers instead of one packed block-concatenated buffer. Q/K/V need not be
* contiguous with one another -- no packing copy is performed.
*
* Writes O as [B, Nq_heads, T, hd] (bf16) and LSE as [B, Nq_heads, T] (fp32).
* hd in {64, 128}. Causal: Hopper v62 (hd 128, T % 256 == 0), B300 hd 64 (T % 256 == 0),
* every other case on any GPU but a B300 (incl. Hopper with BLUTRAIN_NO_SM90[_GQA] set, hd 64,
* or T % 256 != 0) the generic causal WMMA kernel, any T. B300 hd 128 causal is refused.
* Non-causal: T a multiple of 32.
*
* chats (v2 packed rows, design A): optional per-chat plan built for this (B, T, Nq_heads)
* (see GqaChatPlan). nullptr (default) = plain causal over the whole row, exactly as before.
* A plan is accepted on the Hopper v62 path and on the generic causal path; on B300, or with
* is_causal = false, this throws instead of ignoring it. The backward takes the SAME plan; pass
* it to both, or the gradients mix the chats again.
*/
void gqa_flash_attn_forward_qkv(
   const bfloat16_t* q,       // [B, Nq_heads,  T, hd]
   const bfloat16_t* k,       // [B, Nkv_heads, T, hd]
   const bfloat16_t* v,       // [B, Nkv_heads, T, hd]
   bfloat16_t*       output,  // [B, Nq_heads, T, hd]
   float*            lse,     // [B, Nq_heads, T]
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   bool is_causal,
   const GqaChatPlan* chats = nullptr      // v2 packed rows (design A): per-chat plan, see above
);

/**
* @brief Grouped-query flash-attention backward (dQ, dK, dV).
*
* Same block-concatenated bf16 layout as the forward. dQ occupies the Q block
* (Nq_heads), dK/dV the K/V blocks (Nkv_heads). Because group_size =
* Nq_heads/Nkv_heads query heads share each KV head, dK/dV must ACCUMULATE
* across the group. D_buf is per query row: [B, Nq_heads, T] (fp32); it is
* currently unused (D[i]=rowsum(dO[i]⊙O[i]) is recomputed inside the kernels).
*
* Causal requests (is_causal=true): Hopper hd 128 -> the Hopper kernels; B300 hd 64
* -> the SM103 TMA+tcgen05 kernel (T a multiple of 128); every other case on any
* GPU but a B300 -> the generic causal WMMA kernels (since 2026-10-24), any T.
* Non-causal requests run the WMMA kernels (head_dim 64 or 128, T a multiple of 32).
*/
void gqa_flash_attn_backward(
   const bfloat16_t* qkv,         // block-concatenated [Q | K | V]
   const bfloat16_t* output,      // [B, Nq_heads, T, hd]
   const bfloat16_t* grad_output, // [B, Nq_heads, T, hd]
   const float*      lse,         // [B, Nq_heads, T]
   bfloat16_t*       grad_qkv,    // block-concatenated [dQ | dK | dV]
   float*            D_buf,       // [B, Nq_heads, T] scratch
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   bool is_causal
);

/**
* @brief Grouped-query flash-attention backward, same kernels as
* gqa_flash_attn_backward but taking Q, K, V, dQ, dK, dV as independent
* device buffers instead of one packed qkv / grad_qkv buffer. No packing
* copy is performed. dK/dV accumulation across the query-head group happens
* inside the kernel, so grad_k/grad_v need not be pre-zeroed by the caller.
*
* chats (v2 packed rows, design A): the SAME plan the forward got; output and lse must
* come from that packed forward. nullptr (default) = plain causal, exactly as before.
* A plan is accepted on the Hopper Vj2 (default since 2026-10-15; T % 128 == 0, else
* Vj1d runs) / Vj1d (USE_VJ2=0) paths (causal, hd 128, T % 64 == 0, neither USE_FLASH=1
* nor USE_DETERMINISTIC_DQ=1) and on the generic causal path (every causal call the Hopper
* kernels do not take, on any GPU but a B300; any T); anywhere else this throws instead of
* ignoring it.
*/
void gqa_flash_attn_backward_qkv(
   const bfloat16_t* q,           // [B, Nq_heads,  T, hd]
   const bfloat16_t* k,           // [B, Nkv_heads, T, hd]
   const bfloat16_t* v,           // [B, Nkv_heads, T, hd]
   const bfloat16_t* output,      // [B, Nq_heads, T, hd]
   const bfloat16_t* grad_output, // [B, Nq_heads, T, hd]
   const float*      lse,         // [B, Nq_heads, T]
   bfloat16_t*       grad_q,      // [B, Nq_heads,  T, hd]
   bfloat16_t*       grad_k,      // [B, Nkv_heads, T, hd]
   bfloat16_t*       grad_v,      // [B, Nkv_heads, T, hd]
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   bool is_causal,
   const GqaChatPlan* chats = nullptr      // v2 packed rows (design A): the SAME plan the forward got
);

/**
* @brief FUSED grouped-query flash-attention forward: QK-norm + RoPE + attention.
*
* Fuses, per (token, head) over head_dim, in ONE launch:
*   1. QK-norm (RMSNorm over hd) on Q and K:  rstd = 1/sqrt(mean(x^2)+eps),
*      x_hat = x * rstd * gamma.  Skipped for a plane if its gamma is nullptr.
*   2. RoPE rotation on the normalized Q and K (NeoX or GPT-J interleaved).
*   3. GQA flash attention on the resulting Q/K and the raw V.
* V is NOT normed or rotated. Values entering QK^T are RoPE(RMSNorm(Q/K)).
*
* Same block-concatenated bf16 packed layout as gqa_flash_attn_forward. The
* cos/sin cache is fp32 [cache_seq_len, hd] (built once via rope_build_cache_cuda,
* theta baked in) and reused across layers. q_rstd / k_rstd are written as extra
* fp32 outputs (mirroring LSE) so the backward can reuse them. q_gamma / k_gamma
* are fp32 [hd] (nullptr = skip that plane's QK-norm).
*
* bf16 IO, fp32 internal. Constraints inherited from gqa_flash_attn_forward
* (hd in {64,128}, T % 32 == 0, is_causal == false, Nq % Nkv == 0).
*/
void gqa_fused_flash_attn_forward(
   const bfloat16_t* qkv,           // raw [Q | K | V] (pre-norm, pre-rope)
   const float*      cos_sin_cache, // [cache_seq_len, hd] fp32
   const float*      q_gamma,       // [hd] fp32 or nullptr (skip Q QK-norm)
   const float*      k_gamma,       // [hd] fp32 or nullptr (skip K QK-norm)
   bfloat16_t*       output,        // [B, Nq_heads, T, hd]
   float*            lse,           // [B, Nq_heads, T]
   float*            q_rstd,        // [B, Nq_heads, T]  (saved for backward)
   float*            k_rstd,        // [B, Nkv_heads, T] (saved for backward)
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   int cache_seq_len, int pos_offset, float eps,
   bool interleaved, bool is_causal
);

/**
* @brief FUSED grouped-query flash-attention backward (chains attention-bwd ->
*        RoPE-bwd -> QK-norm-bwd).
*
* Takes the ORIGINAL raw qkv and internally re-applies QK-norm + RoPE (reusing
* the saved q_rstd / k_rstd) to reconstruct the post-norm-post-RoPE Q/K that the
* forward attention saw. Runs the attention backward to get grads wrt the
* rotated+normed Q/K, then applies RoPE-backward (R^T) and RMSNorm-backward to
* map them back to the raw qkv scale -> grad_qkv. Produces dq_gamma / dk_gamma
* ([hd] fp32, accumulated over all tokens/heads; zeroed inside). dV is unchanged.
*
* grad_qkv has the same block-concat layout as gqa_flash_attn_backward.
*/
void gqa_fused_flash_attn_backward(
   const bfloat16_t* qkv,           // raw [Q | K | V]
   const float*      cos_sin_cache, // [cache_seq_len, hd] fp32
   const float*      q_gamma,       // [hd] fp32 or nullptr
   const float*      k_gamma,       // [hd] fp32 or nullptr
   const bfloat16_t* output,        // [B, Nq_heads, T, hd]
   const bfloat16_t* grad_output,   // [B, Nq_heads, T, hd]
   const float*      lse,           // [B, Nq_heads, T]
   const float*      q_rstd,        // [B, Nq_heads, T]  (from forward)
   const float*      k_rstd,        // [B, Nkv_heads, T] (from forward)
   bfloat16_t*       grad_qkv,      // raw-scale [dQ | dK | dV]
   float*            dq_gamma,      // [hd] fp32, accumulated (nullptr = skip)
   float*            dk_gamma,      // [hd] fp32, accumulated (nullptr = skip)
   float*            D_buf,         // [B, Nq_heads, T] scratch
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   int cache_seq_len, int pos_offset, float eps,
   bool interleaved, bool is_causal
);


} // namespace cuda
} // namespace OwnTensor