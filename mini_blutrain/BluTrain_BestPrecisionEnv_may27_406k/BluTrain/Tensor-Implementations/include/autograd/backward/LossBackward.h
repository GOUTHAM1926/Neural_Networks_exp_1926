#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for MSE loss: mean((pred - target)^2)
 * 
 * Backward: grad_pred = 2 * (pred - target) / numel
 */
class MSELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    MSELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    const char* name() const override { return "MSELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override;
};

/**
 * @brief Backward function for MAE loss: mean(|pred - target|)
 * 
 * Backward: grad_pred = sign(pred - target) / numel
 */
class MAELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    MAELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    const char* name() const override { return "MAELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override;
};

/**
 * @brief Backward function for binary cross entropy loss
 * 
 * Backward: grad_pred = -target/pred + (1-target)/(1-pred)
 */
class BCELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    BCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    const char* name() const override { return "BCELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override;
};

/**
 * @brief Backward function for categorical cross entropy loss
 * 
 * Backward: grad_pred = -target / pred
 */
class CCELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    CCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    const char* name() const override { return "CCELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override;
};

/**
 * @brief Backward function for sparse cross entropy loss with logits
 * 
 * Forward: softmax(logits), then -log(softmax[target])
 * Backward: grad_logits[i,c] = softmax[i,c] - (c == target[i] ? 1 : 0)
 */
class SparseCrossEntropyLossBackward : public Node {
private:
    Tensor saved_logits_;
    Tensor saved_targets_;
    // Optional: per-row (max, sum) saved by stats-aware forward. When valid,
    // backward runs the fast path that skips the redundant Reduce kernel.
    Tensor saved_max_;
    Tensor saved_sum_;
    int64_t batch_size_;
    int64_t num_classes_;

public:
    SparseCrossEntropyLossBackward(const Tensor& logits, const Tensor& targets,
                                    int64_t batch_size, int64_t num_classes);

    // Stats-aware ctor — forward saved (max, sum); backward skips Reduce kernel.
    SparseCrossEntropyLossBackward(const Tensor& logits, const Tensor& targets,
                                    const Tensor& saved_max, const Tensor& saved_sum,
                                    int64_t batch_size, int64_t num_classes);

    const char* name() const override { return "SparseCrossEntropyLossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        saved_logits_  = Tensor();
        saved_targets_ = Tensor();
        saved_max_     = Tensor();
        saved_sum_     = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor