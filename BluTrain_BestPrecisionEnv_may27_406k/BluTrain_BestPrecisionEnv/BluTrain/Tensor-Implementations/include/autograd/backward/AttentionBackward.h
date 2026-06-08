// // #pragma once

// // #include "autograd/Node.h"
// // #include "core/Tensor.h"

// // namespace OwnTensor {
// // namespace autograd {

// // /**
// //  * @brief Backward function for memory-efficient scaled dot-product attention.
// //  *
// //  * Forward:
// //  *   O = softmax(Q @ K^T / sqrt(hd)) @ V
// //  *   (computed without materializing the full T×T attention matrix)
// //  *
// //  * Backward:
// //  *   Recomputes attention weights block-by-block from saved Q, K, V, and LSE.
// //  *   Uses separate kernels for dQ and dK/dV.
// //  *
// //  *   dS[i,j] = dO[i] @ V[j]^T
// //  *   D[i]    = dO[i] . O[i]
// //  *   dP[i,j] = p[i,j] * (dS[i,j] - D[i])       where p[i,j] = exp(s[i,j] - LSE[i])
// //  *   dQ[i]   = scale * sum_j dP[i,j] * K[j]
// //  *   dK[j]   = scale * sum_i dP[i,j] * Q[i]
// //  *   dV[j]   = sum_i p[i,j] * dO[i]
// //  *
// //  * Saves: Q, K, V (detached), O (detached), LSE (log-sum-exp per row)
// //  * Has 3 inputs: Q (edge 0), K (edge 1), V (edge 2)
// //  */
// // class MemEfficientAttentionBackward : public Node {
// // private:
// //     Tensor saved_query_;
// //     Tensor saved_key_;
// //     Tensor saved_value_;
// //     Tensor saved_output_;
// //     Tensor saved_lse_;
// //     int64_t B_, nh_, T_, hd_;
// //     bool is_causal_;

// // public:
// //     MemEfficientAttentionBackward(
// //         const Tensor& query, const Tensor& key, const Tensor& value,
// //         const Tensor& output, const Tensor& lse,
// //         int64_t B, int64_t nh, int64_t T, int64_t hd,
// //         bool is_causal);

// //     const char* name() const override { return "MemEfficientAttentionBackward"; }
// //     std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
// //     void release_saved_variables() override {
// //         saved_query_  = Tensor();
// //         saved_key_    = Tensor();
// //         saved_value_  = Tensor();
// //         saved_output_ = Tensor();
// //         saved_lse_    = Tensor();
// //     }
// // };

// // } // namespace autograd
// // } // namespace OwnTensor

// #pragma once

// #include "autograd/Node.h"
// #include "core/Tensor.h"

// namespace OwnTensor {
// namespace autograd {

// /**
//  * @brief Backward function for memory-efficient scaled dot-product attention.
//  *
//  * Forward:
//  *   O = softmax(Q @ K^T / sqrt(hd)) @ V
//  *   (computed without materializing the full T×T attention matrix)
//  *
//  * Backward:
//  *   Recomputes attention weights block-by-block from saved Q, K, V, and LSE.
//  *   Uses separate kernels for dQ and dK/dV.
//  *
//  *   dS[i,j] = dO[i] @ V[j]^T
//  *   D[i]    = dO[i] . O[i]
//  *   dP[i,j] = p[i,j] * (dS[i,j] - D[i])       where p[i,j] = exp(s[i,j] - LSE[i])
//  *   dQ[i]   = scale * sum_j dP[i,j] * K[j]
//  *   dK[j]   = scale * sum_i dP[i,j] * Q[i]
//  *   dV[j]   = sum_i p[i,j] * dO[i]
//  *
//  * Saves: Q, K, V (detached), O (detached), LSE (log-sum-exp per row)
//  * Has 3 inputs: Q (edge 0), K (edge 1), V (edge 2)
//  */
// class MemEfficientAttentionBackward : public Node {
// private:
//     Tensor saved_query_;
//     Tensor saved_key_;
//     Tensor saved_value_;
//     Tensor saved_output_;
//     Tensor saved_lse_;
//     int64_t B_, nh_, T_, hd_;
//     bool is_causal_;

// public:
//     MemEfficientAttentionBackward(
//         const Tensor& query, const Tensor& key, const Tensor& value,
//         const Tensor& output, const Tensor& lse,
//         int64_t B, int64_t nh, int64_t T, int64_t hd,
//         bool is_causal);

//     const char* name() const override { return "MemEfficientAttentionBackward"; }
//     std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
//     void release_saved_variables() override {
//         saved_query_  = Tensor();
//         saved_key_    = Tensor();
//         saved_value_  = Tensor();
//         saved_output_ = Tensor();
//         saved_lse_    = Tensor();
//     }
// };

// /**
//  * @brief Backward for scaled_dot_product_attention_packed.
//  *
//  * Forward saved a packed `qkv [B, T, 3*C]` and a packed output `O [B, T, C]`.
//  * In backward, allocates a single `dqkv [B, T, 3*C]` tensor, zeroes it once,
//  * and passes strided views (Q at offset 0, K at C, V at 2C — same stride layout
//  * as forward) as the dQ/dK/dV write targets to the existing memory-efficient
//  * backward kernel. No Tensor::cat is needed afterward; the backward returns the
//  * single packed dqkv tensor.
//  *
//  * Saves: qkv (detached), O (detached), LSE.
//  * Has 1 input edge: qkv (edge 0).
//  */
// class PackedSDPABackward : public Node {
// private:
//     Tensor saved_qkv_;
//     Tensor saved_output_;
//     Tensor saved_lse_;
//     int64_t B_, nh_, T_, hd_, C_;
//     bool is_causal_;

// public:
//     PackedSDPABackward(
//         const Tensor& qkv,
//         const Tensor& output, const Tensor& lse,
//         int64_t B, int64_t nh, int64_t T, int64_t hd,
//         bool is_causal);

//     const char* name() const override { return "PackedSDPABackward"; }
//     std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
//     void release_saved_variables() override {
//         saved_qkv_    = Tensor();
//         saved_output_ = Tensor();
//         saved_lse_    = Tensor();
//     }
// };

// } // namespace autograd
// } // namespace OwnTensor


















// #pragma once

// #include "autograd/Node.h"
// #include "core/Tensor.h"

// namespace OwnTensor {
// namespace autograd {

// /**
//  * @brief Backward function for memory-efficient scaled dot-product attention.
//  *
//  * Forward:
//  *   O = softmax(Q @ K^T / sqrt(hd)) @ V
//  *   (computed without materializing the full T×T attention matrix)
//  *
//  * Backward:
//  *   Recomputes attention weights block-by-block from saved Q, K, V, and LSE.
//  *   Uses separate kernels for dQ and dK/dV.
//  *
//  *   dS[i,j] = dO[i] @ V[j]^T
//  *   D[i]    = dO[i] . O[i]
//  *   dP[i,j] = p[i,j] * (dS[i,j] - D[i])       where p[i,j] = exp(s[i,j] - LSE[i])
//  *   dQ[i]   = scale * sum_j dP[i,j] * K[j]
//  *   dK[j]   = scale * sum_i dP[i,j] * Q[i]
//  *   dV[j]   = sum_i p[i,j] * dO[i]
//  *
//  * Saves: Q, K, V (detached), O (detached), LSE (log-sum-exp per row)
//  * Has 3 inputs: Q (edge 0), K (edge 1), V (edge 2)
//  */
// class MemEfficientAttentionBackward : public Node {
// private:
//     Tensor saved_query_;
//     Tensor saved_key_;
//     Tensor saved_value_;
//     Tensor saved_output_;
//     Tensor saved_lse_;
//     int64_t B_, nh_, T_, hd_;
//     bool is_causal_;

// public:
//     MemEfficientAttentionBackward(
//         const Tensor& query, const Tensor& key, const Tensor& value,
//         const Tensor& output, const Tensor& lse,
//         int64_t B, int64_t nh, int64_t T, int64_t hd,
//         bool is_causal);

//     const char* name() const override { return "MemEfficientAttentionBackward"; }
//     std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
//     void release_saved_variables() override {
//         saved_query_  = Tensor();
//         saved_key_    = Tensor();
//         saved_value_  = Tensor();
//         saved_output_ = Tensor();
//         saved_lse_    = Tensor();
//     }
// };

// } // namespace autograd
// } // namespace OwnTensor

#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for memory-efficient scaled dot-product attention.
 *
 * Forward:
 *   O = softmax(Q @ K^T / sqrt(hd)) @ V
 *   (computed without materializing the full T×T attention matrix)
 *
 * Backward:
 *   Recomputes attention weights block-by-block from saved Q, K, V, and LSE.
 *   Uses separate kernels for dQ and dK/dV.
 *
 *   dS[i,j] = dO[i] @ V[j]^T
 *   D[i]    = dO[i] . O[i]
 *   dP[i,j] = p[i,j] * (dS[i,j] - D[i])       where p[i,j] = exp(s[i,j] - LSE[i])
 *   dQ[i]   = scale * sum_j dP[i,j] * K[j]
 *   dK[j]   = scale * sum_i dP[i,j] * Q[i]
 *   dV[j]   = sum_i p[i,j] * dO[i]
 *
 * Saves: Q, K, V (detached), O (detached), LSE (log-sum-exp per row)
 * Has 3 inputs: Q (edge 0), K (edge 1), V (edge 2)
 */
class MemEfficientAttentionBackward : public Node {
private:
    Tensor saved_query_;
    Tensor saved_key_;
    Tensor saved_value_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, nh_, T_, hd_;
    bool is_causal_;

public:
    MemEfficientAttentionBackward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t nh, int64_t T, int64_t hd,
        bool is_causal);

    const char* name() const override { return "MemEfficientAttentionBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_query_  = Tensor();
        saved_key_    = Tensor();
        saved_value_  = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
    }
};

/**
 * @brief Backward for scaled_dot_product_attention_packed.
 *
 * Forward saved a packed `qkv [B, T, 3*C]` and a packed output `O [B, T, C]`.
 * Backward allocates one zeroed `dqkv [B, T, 3*C]`, passes strided views
 * (Q@0, K@C, V@2C — same stride layout as forward) into the unified
 * mem-efficient backward kernel with skip_grad_zero=true, and returns the
 * single packed dqkv. No Tensor::cat needed downstream.
 */
class PackedSDPABackward : public Node {
private:
    Tensor saved_qkv_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, nh_, T_, hd_, C_;
    bool is_causal_;

public:
    PackedSDPABackward(
        const Tensor& qkv,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t nh, int64_t T, int64_t hd,
        bool is_causal);

    const char* name() const override { return "PackedSDPABackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_qkv_    = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor