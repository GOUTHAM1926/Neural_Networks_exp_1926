#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include <optional>
#include "autograd/Hooks.h"  // Need full definition for unique_ptr

namespace OwnTensor {

// Forward declarations
class Tensor;
class TensorImpl;
class Node;
enum class Dtype;

/**
 * @brief Interface for autograd metadata.
 * 
 * This interface allows the core tensor implementation to remain independent
 * of the autograd system.
 */
struct AutogradMetaInterface {
    virtual ~AutogradMetaInterface() = default;

    /// Set whether this tensor requires gradient
    virtual void set_requires_grad(bool requires_grad, TensorImpl* self_impl) = 0;
    
    /// Check if this tensor requires gradient
    virtual bool requires_grad() const = 0;
    
    /// Get mutable reference to gradient tensor
    virtual Tensor& mutable_grad(TensorImpl* self_impl) = 0;
    
    /// Get const reference to gradient tensor
    virtual const Tensor& grad() const = 0;
    
    /// Check if gradient exists
    virtual bool has_grad() const = 0;
};

/**
 * @brief Full autograd metadata implementation (PyTorch-style).
 * 
 * AutogradMeta stores all information needed for automatic differentiation:
 * - Gradient storage (for leaf tensors)
 * - Computational graph connections (grad_fn, edges)
 * - Hooks for custom backward logic
 * - View tracking information
 * - Thread-safety primitives
 */
struct AutogradMeta : public AutogradMetaInterface {
    // =================================================================
    // GRADIENT STORAGE
    // =================================================================

    /// Accumulated gradient (for leaf tensors mainly)
    std::unique_ptr<Tensor> grad_;

    /// Function that created this tensor (nullptr for leaves)
    std::shared_ptr<Node> grad_fn_;

    /// Gradient accumulator for leaf tensors (weak to prevent cycles!)
    std::weak_ptr<Node> grad_accumulator_;

    // =================================================================
    // HOOKS
    // =================================================================

    /// Pre-backward hooks (run before gradient computation)
    std::vector<std::unique_ptr<FunctionPreHook>> hooks_;

    /// Post-accumulation hooks (run after .grad update on leaves)
    /// Suitable for DDP synchronization as it fires after full accumulation.
    std::vector<std::unique_ptr<PostAccumulateGradHook>> post_acc_grad_hooks_;

    // =================================================================
    // FLAGS & METADATA
    // =================================================================

    /// Only meaningful on leaf variables (must be false otherwise)
    bool requires_grad_{false};

    /// Whether non-leaf should retain gradient (normally cleared)
    bool retains_grad_{false};

    /// Is this tensor a view of another tensor?
    bool is_view_{false};

    /// Which output of grad_fn_ is this (for multi-output operations)
    uint32_t output_nr_{0};

    /// The dtype of the grad field; when nullopt, defaults to tensor's dtype
    std::optional<Dtype> grad_dtype_;

    /// When true, allows gradient dtype to be different from tensor dtype
    bool allow_grad_dtype_mismatch_{false};

    // =================================================================
    // THREAD SAFETY
    // =================================================================

    /// Mutex for thread-safe access to lazy fields
    mutable std::mutex mutex_;

    // =================================================================
    // CONSTRUCTORS
    // =================================================================

    /**
     * @brief Default constructor.
     */
    AutogradMeta() = default;

    /**
     * @brief Construct with requires_grad flag.
     */
    explicit AutogradMeta(bool requires_grad)
        : requires_grad_(requires_grad) {}

    /**
     * @brief Destructor.
     */
    ~AutogradMeta() override = default;

    // Move semantics
    AutogradMeta(AutogradMeta&& other) noexcept;
    AutogradMeta& operator=(AutogradMeta&& other) noexcept;

    // No copy
    AutogradMeta(const AutogradMeta&) = delete;
    AutogradMeta& operator=(const AutogradMeta&) = delete;

    // =================================================================
    // INTERFACE IMPLEMENTATION
    // =================================================================

    /**
     * @brief Set requires_grad property.
     */
    void set_requires_grad(bool requires_grad, TensorImpl* self_impl) override;

    /**
     * @brief Check if this tensor requires gradient.
     */
    bool requires_grad() const override {
        return requires_grad_ || grad_fn_ != nullptr;
    }

    /**
     * @brief Get mutable gradient.
     */
    Tensor& mutable_grad(TensorImpl* self_impl) override;

    /**
     * @brief Get const gradient.
     */
    const Tensor& grad() const override;

    // =================================================================
    // ADDITIONAL METHODS
    // =================================================================

    /**
     * @brief Get the gradient function.
     */
    const std::shared_ptr<Node>& grad_fn() const {
        return grad_fn_;
    }

    /**
     * @brief Set the gradient function.
     */
    void set_grad_fn(std::shared_ptr<Node> fn) {
        grad_fn_ = std::move(fn);
    }

    /**
     * @brief Check if this is a leaf tensor.
     */
    bool is_leaf() const {
        return grad_fn_ == nullptr;
    }

    /**
     * @brief Get output number.
     */
    uint32_t output_nr() const {
        return output_nr_;
    }

    /**
     * @brief Set output number.
     */
    void set_output_nr(uint32_t nr) {
        output_nr_ = nr;
    }

    /**
     * @brief Check if this is a view.
     */
    bool is_view() const {
        return is_view_;
    }

    /**
     * @brief Set view flag.
     */
    void set_is_view(bool is_view) {
        is_view_ = is_view;
    }

    /**
     * @brief Check if gradient should be retained (for non-leaves).
     */
    bool retains_grad() const {
        return retains_grad_;
    }

    /**
     * @brief Set whether to retain gradient.
     */
    void set_retains_grad(bool retains) {
        retains_grad_ = retains;
    }

    /**
     * @brief Add a pre-backward hook.
     */
    void add_hook(std::unique_ptr<FunctionPreHook> hook);

    /**
     * @brief Set post-accumulation hook.
     */
    void add_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook);

    /**
     * @brief Trigger all post-accumulation hooks.
     */
    void trigger_post_acc_hooks(const Tensor& grad);

    /**
     * @brief Clear all hooks.
     */
    void clear_hooks();

    /**
     * @brief Set gradient tensor (copy).
     */
    void set_grad(const Tensor& new_grad);

    /**
     * @brief Set gradient tensor (move).
     */
     void set_grad(Tensor&& new_grad);

    /**
     * @brief Accumulate gradient (sum += update) in a single locked step.
     */
    void accumulate_grad(Tensor&& update);

    /**
     * @brief Check if gradient exists.
     */
    bool has_grad() const;

    /**
     * @brief Reset gradient to uninitialized state.
     */
    void reset_grad();
};

} // namespace OwnTensor
