#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief LayerNorm with autograd support.
 *        Delegates to layer_norm_forward/backward in Normalizations.h.
 */
Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int normalized_shape,
    float eps = 1e-5f);

/**
 * @brief RMSNorm with autograd support.
 *        Delegates to rms_norm_forward/backward in Normalizations.h.
 */
Tensor rms_norm(
    const Tensor& input,
    const Tensor& weight,
    int normalized_shape,
    float eps = 1e-5f);

/**
 * @brief Fused QK-norm (RMSNorm) + RoPE with autograd support.
 *        Delegates to fused_rms_rope_forward/backward in Normalizations.h.
 * 
 * @param q              [B, Hq, T, hd] input Query tensor
 * @param k              [B, Hkv, T, hd] input Key tensor
 * @param q_gamma        [hd] scale parameter for Q RMSNorm
 * @param k_gamma        [hd] scale parameter for K RMSNorm
 * @param cos_sin_cache  [cache_seq_len, hd] RoPE cache
 * @param pos_offset     position offset for RoPE cache
 * @param eps            RMSNorm epsilon
 * @param interleaved    false for NeoX, true for GPT-J
 * @param positions      (v2 packing, added 2026-10-10) optional int32 CUDA tensor [B, T]
 *                       with each token's RoPE position (packed chats restart at 0).
 *                       Default = unchanged behaviour (position = t + pos_offset). It is
 *                       saved for the backward, which must rotate by the same positions.
 * @return std::pair<Tensor, Tensor> (q_out, k_out)
 */
std::pair<Tensor, Tensor> fused_rms_rope(
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int64_t pos_offset = 0,
    float eps = 1e-5f,
    bool interleaved = false,
    const Tensor& positions = Tensor());

/**
 * @brief Alias for fused_rms_rope.
 */
inline std::pair<Tensor, Tensor> fused_qk_norm_rope(
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int64_t pos_offset = 0,
    float eps = 1e-5f,
    bool interleaved = false)
{
    return fused_rms_rope(q, k, q_gamma, k_gamma, cos_sin_cache, pos_offset, eps, interleaved);
}

} // namespace autograd
} // namespace OwnTensor

