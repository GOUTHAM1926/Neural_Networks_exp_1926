#pragma once
#include <cstdint>
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
* Constraints: hd in {64, 128}, T a multiple of 32. is_causal must be false
* (the kernel has no causal mask yet).
*/
void gqa_flash_attn_forward(
   const bfloat16_t* qkv,     // block-concatenated [Q | K | V]
   bfloat16_t*       output,  // [B, Nq_heads, T, hd]
   float*            lse,     // [B, Nq_heads, T]
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   bool is_causal
);

/**
* v2 packed rows, design A (per-segment work lists, the cu_seqlens way of FlashAttention varlen,
* TE/Megatron THD and torchtitan VarlenInnerAttention; added 2026-10-10). The same packed
* [B, H, T, D] rows as design B, but the kernels cut their work per segment instead of per row.
* Build ONE plan per batch from the segments of every row -- chats AND each row's pad tail as its
* own segment; together they must cover every token of every row exactly once (checked, throws) --
* and pass the same plan to gqa_flash_attn_forward_qkv and gqa_flash_attn_backward_qkv.
* The plan owns device memory: free it with gqa_chat_plan_free.
*/
struct GqaChatPlan {
    int32_t* fwd_units = nullptr;   // device, 4 ints per forward unit: {b, t0, L, (cpl << 8) | hq}
    int      n_fwd_units = 0;
    int32_t* bwd_tiles = nullptr;   // device, 4 ints per backward key tile: {b, t0, L, kt}
    int      n_bwd_tiles = 0;
    int B = 0, S = 0, Hq = 0;       // the shape the plan was built for (checked at every call)
};
// seg_b/seg_t0/seg_len: HOST arrays of n_seg segments (row, start, length). Hq <= 255.
void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
                         const int32_t* seg_len, int n_seg, int B, int S, int Hq);
void gqa_chat_plan_free(GqaChatPlan& plan);

/**
* @brief Grouped-query flash-attention forward, same kernels as
* gqa_flash_attn_forward but taking Q, K, V as three independent device
* buffers instead of one packed block-concatenated buffer. Q/K/V need not be
* contiguous with one another -- no packing copy is performed.
*
* Writes O as [B, Nq_heads, T, hd] (bf16) and LSE as [B, Nq_heads, T] (fp32).
* Same constraints as gqa_flash_attn_forward: hd in {64, 128}, T a multiple of
* 32 (256 for the causal hd==64 path). is_causal with hd!=64 is refused.
*
* seg_start (v2 packed rows, design B, added 2026-10-10): optional DEVICE int32
* [B, T]. Each row holds several chats; seg_start[b*T + t] is where the segment of
* token t starts inside row b (a chat, or the row's pad tail as its own segment).
* Token i then only attends keys j with seg_start[i] <= j <= i. Contract, checked by
* whoever builds the array: seg_start[t] <= t, and it equals t (a segment starts at t)
* or seg_start[t-1]. nullptr (default) = plain causal over the whole row, exactly as
* before. A non-null seg_start is only accepted on the Hopper v62 path (causal,
* hd 128, T % 256 == 0); anywhere else this throws instead of ignoring it.
* The backward takes the SAME seg_start (gqa_flash_attn_backward_qkv, added in
* Step 4 on 2026-10-10); pass it to both, or the gradients mix the chats again.
*/
void gqa_flash_attn_forward_qkv(
   const bfloat16_t* q,       // [B, Nq_heads,  T, hd]
   const bfloat16_t* k,       // [B, Nkv_heads, T, hd]
   const bfloat16_t* v,       // [B, Nkv_heads, T, hd]
   bfloat16_t*       output,  // [B, Nq_heads, T, hd]
   float*            lse,     // [B, Nq_heads, T]
   int B, int Nq_heads, int Nkv_heads, int T, int hd,
   bool is_causal,
   const int32_t*    seg_start = nullptr,  // v2 design B: [B, T] device, see above
   const GqaChatPlan* chats = nullptr      // v2 design A: per-segment plan (never with seg_start)
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
* Causal requests (is_causal=true) route to the SM103/B300 TMA+tcgen05 kernel,
* which requires head_dim==64 and T a multiple of 128. Non-causal requests fall
* back to the WMMA kernels (head_dim 64 or 128, T a multiple of 32). A causal
* request with head_dim!=64 is refused (no correct kernel).
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
* seg_start (v2 packed rows, design B, added 2026-10-10): optional DEVICE int32
* [B, T], the same array the forward got (see gqa_flash_attn_forward_qkv for the
* contract); output and lse must come from that packed forward. nullptr (default)
* = plain causal, exactly as before. A non-null seg_start is only accepted on the
* Hopper Vj1d path (causal, hd 128, T % 64 == 0, neither USE_FLASH=1 nor
* USE_DETERMINISTIC_DQ=1); anywhere else this throws instead of ignoring it.
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
   const int32_t*    seg_start = nullptr,  // v2 design B: [B, T] device, see above
   const GqaChatPlan* chats = nullptr      // v2 design A: the SAME plan the forward got
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