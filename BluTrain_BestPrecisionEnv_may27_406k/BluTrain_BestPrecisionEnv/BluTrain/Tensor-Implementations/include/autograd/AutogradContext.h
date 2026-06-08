#pragma once

/**
 * @file AutogradContext.h
 * @brief Context for custom autograd functions.
 * 
 * Used by custom backward functions to:
 * - Save tensors for backward pass
 * - Mark inputs as modified (dirty)
 * - Track non-differentiable outputs
 */

#include "autograd/SavedVariable.h"
#include <vector>
#include <unordered_set>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Context passed to custom autograd functions.
 * 
 * ## Usage in Custom Function
 * ```cpp
 * class MyFunction : public CppNode<MyFunction> {
 *     static variable_list forward(AutogradContext* ctx,
 *                                   const Tensor& input) {
 *         // Save for backward
 *         ctx->save_for_backward({input});
 *         
 *         // Compute output
 *         Tensor output = custom_op(input);
 *         return {output};
 *     }
 *     
 *     static variable_list backward(AutogradContext* ctx,
 *                                    const variable_list& grad_outputs) {
 *         // Retrieve saved tensors
 *         auto saved = ctx->get_saved_variables();
 *         Tensor input = saved[0];
 *         
 *         // Compute gradient
 *         return {grad_outputs[0] * custom_grad(input)};
 *     }
 * };
 * ```
 */
class AutogradContext {
private:
    /// Saved variables for backward pass
    std::vector<SavedVariable> saved_variables_;
    
    /// Set of tensors that were modified in-place
    std::unordered_set<TensorImpl*> dirty_inputs_;
    
    /// Whether this context is being used for backward
    bool is_backward_{false};
    
    /// The node this context belongs to
    std::shared_ptr<Node> grad_fn_;
    
public:
    AutogradContext() = default;
    
    // =========================================================================
    // Saving Tensors
    // =========================================================================
    
    /**
     * @brief Save tensors for use in backward pass.
     * 
     * @param tensors Tensors to save
     */
    void save_for_backward(const variable_list& tensors) {
        saved_variables_.clear();
        saved_variables_.reserve(tensors.size());
        for (const auto& t : tensors) {
            saved_variables_.emplace_back(t, false);
        }
    }
    
    /**
     * @brief Get saved tensors for backward computation.
     * 
     * @return Vector of unpacked tensors
     */
    variable_list get_saved_variables() {
        variable_list result;
        result.reserve(saved_variables_.size());
        for (auto& sv : saved_variables_) {
            result.push_back(sv.unpack(grad_fn_));
        }
        return result;
    }
    
    // =========================================================================
    // Dirty Tracking
    // =========================================================================
    
    /**
     * @brief Mark tensors as modified in-place.
     * 
     * This is used for operations that modify their inputs.
     */
    void mark_dirty(const variable_list& inputs) {
        for (const auto& t : inputs) {
            if (t.unsafeGetTensorImpl()) {
                dirty_inputs_.insert(t.unsafeGetTensorImpl());
            }
        }
    }
    
    /**
     * @brief Check if a tensor was marked dirty.
     */
    bool is_dirty(const Tensor& t) const {
        return dirty_inputs_.find(t.unsafeGetTensorImpl()) != dirty_inputs_.end();
    }
    
    // =========================================================================
    // State Management
    // =========================================================================
    
    /**
     * @brief Set the grad_fn associated with this context.
     */
    void set_grad_fn(std::shared_ptr<Node> fn) {
        grad_fn_ = std::move(fn);
    }
    
    /**
     * @brief Get the grad_fn.
     */
    std::shared_ptr<Node> grad_fn() const {
        return grad_fn_;
    }
    
    /**
     * @brief Release saved variables (after backward).
     */
    void release_variables() {
        for (auto& sv : saved_variables_) {
            sv.reset();
        }
        saved_variables_.clear();
    }
    
    /**
     * @brief Get number of saved variables.
     */
    size_t num_saved_variables() const {
        return saved_variables_.size();
    }
};

} // namespace autograd
} // namespace OwnTensor
