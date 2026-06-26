// #pragma once
// #include <cstdint>

// namespace OwnTensor {
// namespace cuda {

// /**
// * @brief Compute log-sum-exp per row of attention scores.
// *
// * Read-only kernel  256)
// * @param is_causal     whether to apply lower-triangular causal mask
// * @param dropout_p     dropout probability in [0, 1); pass 0 to disable
// * @param dropout_mask  pre-generated mask tensor of shape [B*nh, T, T] where
// *                      each element is either 0 (dropped) or 1/(1-dropout_p)
// *                      (kept and rescaled); pass nullptr when dropout_p == 0
// */
// void mem_efficient_attn_forward(
//    const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//    const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//    const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//    float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//    float* lse,          int64_t lse_strideB, int64_t lse_strideH,
//    int64_t B, int64_t nh, int64_t T, int64_t hd,
//    bool is_causal,
//   float dropout_p = 0.0f,
//   const float* dropout_mask = nullptr
// );

// void mem_efficient_attn_forward_tc(
//    const float* query,  int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//    const float* key,    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//    const float* value,  int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//    float* output,       int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//    float* lse,          int64_t lse_strideB, int64_t lse_strideH,
//    int64_t B, int64_t nh, int64_t T, int64_t hd,
//    bool is_causal,
//   float dropout_p = 0.0f,
//   const float* dropout_mask = nullptr
// );
// /**
// * @brief Unified memory-efficient attention backward pass for dQ, dK, dV.
// *
// * Architecture (PyTorch CUTLASS FMHA style):
// *   1. Precompute D[i] = dot(dO[i], O[i]) in a lightweight kernel (one warp per row)
// *   2. Single KV-tile-centric kernel computes dQ, dK, dV together:
// *      - Each block owns one KV tile, sweeps all Q rows
// *      - Phase A: warp-parallel score recompute + dQ accumulation in registers
// *      - Phase B: cooperative dK/dV reduction (no per-element atomics)
// *      - dQ written via atomicAdd; dK/dV written once from shared memory
// *
// * @param D_buf    (B*nh, T) scratch buffer for precomputed D values (pre-allocated)
// */
// void mem_efficient_attn_backward(
//    const float* query,       int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
//    const float* key,         int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
//    const float* value,       int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
//    const float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
//    const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
//    const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
//    float* grad_query,        int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
//    float* grad_key,          int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
//    float* grad_value,        int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
//    float* D_buf,             int64_t d_strideB, int64_t d_strideH,
//    int64_t B, int64_t nh, int64_t T, int64_t hd,
//    bool is_causal,
//    // When true, skip the per-tensor cudaMemsetAsync zero-init of dQ/dK/dV.
//    // The packed-QKV path zeros the entire dqkv buffer once externally; the
//    // launcher's per-tensor memsets assume contiguous dQ/dK/dV layout and
//    // would corrupt the packed buffer if left enabled there.
//    bool skip_grad_zero = false
// );

// } // namespace cuda
// } // namespace OwnTensor

















#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

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
  const float* dropout_mask = nullptr
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
  const float* dropout_mask = nullptr
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

} // namespace cuda
} // namespace OwnTensor