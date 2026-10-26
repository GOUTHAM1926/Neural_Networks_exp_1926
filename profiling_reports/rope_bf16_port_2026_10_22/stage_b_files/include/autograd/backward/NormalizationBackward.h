#pragma once

#include "autograd/Node.h"
#include "autograd/SavedVariable.h"

namespace OwnTensor {
namespace autograd {

class LayerNormBackward : public Node {
public:
    LayerNormBackward(
        Tensor input, Tensor mean, Tensor rstd, Tensor weight,
        int normalized_shape, float eps)
        : normalized_shape_(normalized_shape), eps_(eps)
    {
        input_ = SavedVariable(input, false);
        mean_  = SavedVariable(mean, false);
        rstd_  = SavedVariable(rstd, false);
        if (weight.is_valid())
            weight_ = SavedVariable(weight, false);
    }
    const char* name() const override { return "LayerNormBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        input_.reset(); mean_.reset(); rstd_.reset(); weight_.reset();
    }

private:
    SavedVariable input_, mean_, rstd_, weight_;
    int normalized_shape_;
    float eps_;
};

class RMSNormBackward : public Node {
public:
    RMSNormBackward(
        Tensor input, Tensor rstd, Tensor weight,
        int normalized_shape, float eps)
        : normalized_shape_(normalized_shape), eps_(eps)
    {
        input_ = SavedVariable(input, false);
        rstd_  = SavedVariable(rstd, false);
        if (weight.is_valid())
            weight_ = SavedVariable(weight, false);
    }
    const char* name() const override { return "RMSNormBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        input_.reset(); rstd_.reset(); weight_.reset();
    }

private:
    SavedVariable input_, rstd_, weight_;
    int normalized_shape_;
    float eps_;
};

class FusedRMSRoPEBackward : public Node {
public:
    // positions (v2 packing, added 2026-10-10): the optional int32 [B, T] per-token RoPE
    // positions the forward used (invalid Tensor = none). Kept as a plain Tensor reference
    // (no grad, never modified) so its buffer stays alive until this node runs.
    FusedRMSRoPEBackward(
        Tensor q, Tensor k, Tensor q_gamma, Tensor k_gamma,
        Tensor cos_sin_cache, Tensor q_rstd, Tensor k_rstd,
        int64_t pos_offset = 0, bool interleaved = false,
        Tensor positions = Tensor())
        : pos_offset_(pos_offset), interleaved_(interleaved), positions_(positions)
    {
        q_ = SavedVariable(q, false);
        k_ = SavedVariable(k, false);
        if (q_gamma.is_valid()) q_gamma_ = SavedVariable(q_gamma, false);
        if (k_gamma.is_valid()) k_gamma_ = SavedVariable(k_gamma, false);
        cos_sin_cache_ = SavedVariable(cos_sin_cache, false);
        q_rstd_ = SavedVariable(q_rstd, false);
        k_rstd_ = SavedVariable(k_rstd, false);
    }
    const char* name() const override { return "FusedRMSRoPEBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        q_.reset(); k_.reset(); q_gamma_.reset(); k_gamma_.reset();
        cos_sin_cache_.reset(); q_rstd_.reset(); k_rstd_.reset();
        positions_ = Tensor();   // v2: drop the reference with the other saved tensors
    }

private:
    SavedVariable q_, k_, q_gamma_, k_gamma_, cos_sin_cache_, q_rstd_, k_rstd_;
    int64_t pos_offset_;
    bool interleaved_;
    Tensor positions_;   // v2: optional per-token positions [B, T] (invalid = none)
};

class FusedRMSRoPEBf16Backward : public Node {
public:
    // positions (added 2026-10-22): as in FusedRMSRoPEBackward above.
    FusedRMSRoPEBf16Backward(
        Tensor q, Tensor k, Tensor q_gamma, Tensor k_gamma,
        Tensor cos_sin_cache, Tensor q_rstd, Tensor k_rstd,
        int64_t pos_offset = 0, bool interleaved = false,
        Tensor positions = Tensor())
        : pos_offset_(pos_offset), interleaved_(interleaved), positions_(positions)
    {
        q_ = SavedVariable(q, false);
        k_ = SavedVariable(k, false);
        q_gamma_ = SavedVariable(q_gamma, false);
        k_gamma_ = SavedVariable(k_gamma, false);
        cos_sin_cache_ = SavedVariable(cos_sin_cache, false);
        q_rstd_ = SavedVariable(q_rstd, false);
        k_rstd_ = SavedVariable(k_rstd, false);
    }
    const char* name() const override { return "FusedRMSRoPEBf16Backward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        q_.reset(); k_.reset(); q_gamma_.reset(); k_gamma_.reset();
        cos_sin_cache_.reset(); q_rstd_.reset(); k_rstd_.reset();
        positions_ = Tensor();
    }

private:
    SavedVariable q_, k_, q_gamma_, k_gamma_, cos_sin_cache_, q_rstd_, k_rstd_;
    int64_t pos_offset_;
    bool interleaved_;
    Tensor positions_;   // optional per-token positions [B, T] (invalid = none)
};

} // namespace autograd
} // namespace OwnTensor

