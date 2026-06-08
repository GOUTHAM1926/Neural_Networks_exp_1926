#pragma once

/**
 * @file SavedVariable.h
 * @brief SavedVariable for storing tensors across the forward/backward boundary.
 * 
 * SavedVariable is used by backward functions to save tensors that are needed
 * for gradient computation. It provides:
 * - Version checking to detect in-place modifications
 * - Proper reference management (weak_ptr for grad_fn to avoid cycles)
 * - Safe unpacking with error reporting
 * 
 * ## Why SavedVariable Matters
 * 
 * When computing gradients, backward functions need access to the original
 * input tensors. But if a tensor is modified in-place between the forward
 * pass and backward pass, the gradient computation would be incorrect.
 * SavedVariable detects this by checking the version counter.
 * 
 * ## Usage Example
 * 
 * ```cpp
 * class MulBackward : public Node {
 * private:
 *     SavedVariable saved_a_;
 *     SavedVariable saved_b_;
 *     
 * public:
 *     MulBackward(const Tensor& a, const Tensor& b)
 *         : Node(2),
 *           saved_a_(a, false),  // not an output
 *           saved_b_(b, false) {}
 *           
 *     variable_list apply(variable_list&& grads) override {
 *         // Unpack validates version hasn't changed
 *         Tensor a = saved_a_.unpack(shared_from_this());
 *         Tensor b = saved_b_.unpack(shared_from_this());
 *         
 *         return {grads[0] * b, grads[0] * a};
 *     }
 * };
 * ```
 */

#include "core/Tensor.h"
#include "autograd/Node.h"
#include <memory>
#include <stdexcept>
#include <sstream>

namespace OwnTensor {

/**
 * @brief Container for saving tensors across forward/backward boundary.
 * 
 * Saves the tensor data along with metadata needed to detect
 * if the tensor was modified in-place.
 */
class SavedVariable {
private:
    /// The saved tensor data
    Tensor data_;
    
    /// Version at save time (for detecting in-place modifications)
    uint32_t saved_version_{0};
    
    /// Original output number of this variable
    uint32_t output_nr_{0};
    
    /// Whether this variable was a leaf tensor
    bool was_leaf_{false};
    
    /// Whether this is an output of the function (special handling)
    bool is_output_{false};
    
    /// Whether tracking of gradients was enabled
    bool requires_grad_{false};
    
    /// The grad_fn at save time (weak to avoid cycles)
    std::weak_ptr<Node> weak_grad_fn_;

public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /// Default constructor - creates empty saved variable
    SavedVariable() = default;
    
    /**
     * @brief Save a tensor for later use in backward.
     * 
     * @param variable The tensor to save
     * @param is_output True if this variable is an output of the backward node
     * @param is_inplace_on_view True if this was created by an in-place op on a view
     */
    SavedVariable(const Tensor& variable, bool is_output, bool is_inplace_on_view = false);
    
    /// Moveable
    SavedVariable(SavedVariable&&) noexcept = default;
    SavedVariable& operator=(SavedVariable&&) noexcept = default;
    
    /// Copyable
    SavedVariable(const SavedVariable&) = default;
    SavedVariable& operator=(const SavedVariable&) = default;
    
    ~SavedVariable() = default;
    
    // =========================================================================
    // Unpacking
    // =========================================================================
    
    /**
     * @brief Unpack the saved variable for use in backward.
     * 
     * This validates that:
     * 1. The variable has not been modified in-place
     * 2. The variable is still valid
     * 
     * @param saved_for The Node that saved this variable (for error messages)
     * @return The unpacked tensor
     * @throws std::runtime_error if validation fails
     */
    Tensor unpack(std::shared_ptr<Node> saved_for = nullptr) const;
    
    /**
     * @brief Check if this saved variable is defined.
     */
    bool defined() const {
        return data_.unsafeGetTensorImpl() != nullptr;
    }
    
    // =========================================================================
    // Properties
    // =========================================================================
    
    /// Get the saved version
    uint32_t saved_version() const { return saved_version_; }
    
    /// Get output number
    uint32_t output_nr() const { return output_nr_; }
    
    /// Check if was a leaf
    bool was_leaf() const { return was_leaf_; }
    
    /// Check if is output
    bool is_output() const { return is_output_; }
    
    /// Check if required grad
    bool requires_grad() const { return requires_grad_; }
    
    /**
     * @brief Reset the saved variable (release resources).
     */
    void reset() {
        data_ = Tensor();
        weak_grad_fn_.reset();
    }
    
    /**
     * @brief Get the raw data (for debugging only).
     */
    const Tensor& raw_data() const { return data_; }
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Check if a tensor has been modified since version was recorded.
 * 
 * @param tensor The tensor to check
 * @param saved_version The version that was recorded
 * @return True if modified
 */
bool was_modified_inplace(const Tensor& tensor, uint32_t saved_version);

/**
 * @brief Increment the version counter on a tensor.
 * 
 * Call this when performing in-place operations.
 * 
 * @param tensor The tensor being modified
 */
void increment_version(const Tensor& tensor);

} // namespace OwnTensor
