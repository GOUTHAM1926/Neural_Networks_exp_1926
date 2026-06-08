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

} // namespace autograd
} // namespace OwnTensor
