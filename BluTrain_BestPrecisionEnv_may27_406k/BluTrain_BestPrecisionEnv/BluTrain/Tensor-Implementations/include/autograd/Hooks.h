#pragma once

#include <memory>
#include <functional>

namespace OwnTensor {

// Forward declaration
class Tensor;

/**
 * @brief Base class for pre-backward hooks.
 * 
 * Pre-hooks are called BEFORE the gradient computation for a tensor.
 * They can inspect or modify the gradient before it's used.
 * 
 * Common use cases:
 * - Gradient clipping
 * - Gradient logging/debugging
 * - Custom gradient transformations
 */
class FunctionPreHook {
public:
    virtual ~FunctionPreHook() = default;

    /**
     * @brief Execute the hook.
     * 
     * @param grad The gradient that will be used
     * @return Modified gradient (can return original unchanged)
     */
    virtual Tensor operator()(const Tensor& grad) = 0;
};

/**
 * @brief Lambda-based pre-hook for convenience.
 */
class LambdaPreHook : public FunctionPreHook {
public:
    using hook_fn = std::function<Tensor(const Tensor&)>;

    explicit LambdaPreHook(hook_fn fn);
    Tensor operator()(const Tensor& grad) override;

private:
    hook_fn fn_;
};

/**
 * @brief Base class for post-accumulation hooks.
 * 
 * Post-accumulation hooks are called AFTER the gradient has been
 * accumulated into a leaf tensor's .grad field.
 * 
 * Common use cases:
 * - Distributed gradient synchronization
 * - Gradient monitoring
 * - Custom post-processing
 */
class PostAccumulateGradHook {
public:
    virtual ~PostAccumulateGradHook() = default;

    /**
     * @brief Execute the hook after gradient accumulation.
     * 
     * For DDP, this is typically where gradient synchronization (all-reduce)
     * is triggered, as the gradient for this parameter is now "ready".
     * 
     * @param grad The accumulated gradient
     */
    virtual void operator()(const Tensor& grad) = 0;
};

/**
 * @brief Lambda-based post-accumulation hook.
 */
class LambdaPostAccHook : public PostAccumulateGradHook {
public:
    using hook_fn = std::function<void(const Tensor&)>;

    explicit LambdaPostAccHook(hook_fn fn);
    void operator()(const Tensor& grad) override;

private:
    hook_fn fn_;
};

// Helper functions for creating hooks

/**
 * @brief Create a pre-hook from a lambda.
 */
inline std::unique_ptr<FunctionPreHook> make_pre_hook(
    std::function<Tensor(const Tensor&)> fn) {
    return std::make_unique<LambdaPreHook>(std::move(fn));
}

/**
 * @brief Create a post-accumulation hook from a lambda.
 */
inline std::unique_ptr<PostAccumulateGradHook> make_post_acc_hook(
    std::function<void(const Tensor&)> fn) {
    return std::make_unique<LambdaPostAccHook>(std::move(fn));
}

} // namespace OwnTensor
