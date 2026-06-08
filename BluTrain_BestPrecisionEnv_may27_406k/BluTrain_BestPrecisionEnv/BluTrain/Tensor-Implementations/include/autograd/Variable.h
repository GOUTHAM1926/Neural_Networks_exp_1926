#pragma once

/**
 * @file Variable.h
 * @brief Variable/Tensor autograd integration and helper functions.
 * 
 * This file provides the impl namespace functions that bridge Tensors with
 * the autograd system, following PyTorch's design pattern.
 * 
 * ## Overview
 * 
 * In PyTorch, Variable was historically a separate class from Tensor.
 * Now they are unified: `using Variable = Tensor`.
 * 
 * The `impl` namespace provides low-level helper functions for:
 * - Getting/setting gradient edges
 * - Accessing gradient accumulators
 * - Managing autograd metadata
 * 
 * ## Usage Examples
 * 
 * ### Getting gradient edge for a tensor
 * ```cpp
 * Tensor x = Tensor::randn({3, 3}, TensorOptions().with_req_grad(true));
 * Edge edge = impl::gradient_edge(x);  // Gets edge to grad accumulator
 * ```
 * 
 * ### Accessing autograd metadata
 * ```cpp
 * auto* meta = impl::get_autograd_meta(tensor);
 * if (meta) {
 *     // Use metadata
 * }
 * 
 * // Or ensure it exists:
 * auto* meta = impl::materialize_autograd_meta(tensor);
 * ```
 */

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "core/AutogradMeta.h"

namespace OwnTensor {

// Type alias (PyTorch compatibility)
using Variable = Tensor;

/**
 * @brief Implementation namespace for low-level autograd operations.
 * 
 * These functions provide direct access to autograd internals.
 * Use with caution - prefer high-level Tensor methods when possible.
 */
namespace impl {

// =============================================================================
// Autograd Metadata Access
// =============================================================================

/**
 * @brief Get autograd metadata for a tensor (may return nullptr).
 * 
 * ## When to Use
 * Use this when you need to check if a tensor has autograd metadata
 * without creating it if it doesn't exist.
 * 
 * ## Example
 * ```cpp
 * auto* meta = impl::get_autograd_meta(tensor);
 * if (meta && meta->requires_grad()) {
 *     // Tensor is tracking gradients
 * }
 * ```
 * 
 * @param self The tensor to get metadata from
 * @return Pointer to AutogradMeta, or nullptr if none exists
 */
inline AutogradMeta* get_autograd_meta(const Tensor& self) {
    if (!self.unsafeGetTensorImpl()) return nullptr;
    auto* interface = self.unsafeGetTensorImpl()->autograd_meta();
    return interface ? static_cast<AutogradMeta*>(interface) : nullptr;
}

/**
 * @brief Ensure autograd metadata exists (creates if needed).
 * 
 * ## When to Use
 * Use this when you need to modify autograd metadata and want to 
 * ensure it exists first.
 * 
 * ## Example
 * ```cpp
 * auto* meta = impl::materialize_autograd_meta(tensor);
 * meta->set_requires_grad(true);
 * ```
 * 
 * @param self The tensor to get/create metadata for
 * @return Pointer to AutogradMeta (never null for valid tensor)
 */
inline AutogradMeta* materialize_autograd_meta(const Tensor& self) {
    TensorImpl* impl = self.unsafeGetTensorImpl();
    if (!impl) return nullptr;
    
    if (!impl->has_autograd_meta()) {
        impl->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    return static_cast<AutogradMeta*>(impl->autograd_meta());
}

// =============================================================================
// Gradient Edge Functions
// =============================================================================

/**
 * @brief Get the gradient edge for a variable.
 * 
 * For **non-leaf tensors** (with grad_fn), returns edge to grad_fn.
 * For **leaf tensors** (requires_grad=true), returns edge to grad accumulator.
 * For tensors not requiring grad, returns invalid edge.
 * 
 * ## Why This Matters
 * When connecting the backward graph, each tensor's gradient edge tells
 * the engine where to send gradients during the backward pass.
 * 
 * ## Example
 * ```cpp
 * // For a leaf parameter
 * Tensor w = Tensor::randn({2, 2}, TensorOptions().with_req_grad(true));
 * Edge edge = impl::gradient_edge(w);  // Points to GradAccumulator
 * 
 * // For a computed tensor
 * Tensor y = autograd::matmul(x, w);
 * Edge edge = impl::gradient_edge(y);  // Points to MatmulBackward
 * ```
 * 
 * @param self The tensor to get gradient edge for
 * @return Edge connecting to gradient function or accumulator
 */
Edge gradient_edge(const Tensor& self);

/**
 * @brief Set the gradient edge for a variable.
 * 
 * This is called when creating a new tensor from an operation
 * to connect it to its gradient function.
 * 
 * @param self The tensor to set edge for
 * @param edge The edge to set
 */
void set_gradient_edge(const Tensor& self, Edge edge);

// =============================================================================
// Gradient Accumulator
// =============================================================================

/**
 * @brief Get or create gradient accumulator for a leaf variable.
 * 
 * Leaf tensors (parameters) use a GradAccumulator to collect gradients.
 * This function returns the existing accumulator or creates one.
 * 
 * ## Why Use Weak Pointer
 * The accumulator is stored as weak_ptr to prevent circular references.
 * The backward graph holds strong references during execution.
 * 
 * @param self The leaf tensor
 * @return Shared pointer to GradAccumulator
 */
std::shared_ptr<Node> grad_accumulator(const Tensor& self);

/**
 * @brief Set the gradient accumulator for a variable.
 * 
 * @param self The tensor to set accumulator for
 * @param acc The accumulator (stored as weak_ptr)
 */
void set_grad_accumulator(const Tensor& self, std::weak_ptr<Node> acc);

} // namespace impl

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * @brief Create a variable from tensor with gradient edge.
 * 
 * This creates a new tensor that shares storage with the input
 * but has a specific gradient edge attached.
 * 
 * ## When to Use
 * Used internally by operations to create output tensors that are
 * connected to the backward graph.
 * 
 * ## Example
 * ```cpp
 * Tensor data = compute_something();
 * auto grad_fn = std::make_shared<MyBackward>();
 * Tensor result = make_variable(data, Edge(grad_fn, 0));
 * ```
 * 
 * @param data The tensor data
 * @param gradient_edge Edge to attach
 * @return New tensor with the edge attached
 */
Tensor make_variable(const Tensor& data, Edge gradient_edge);

/**
 * @brief Create gradient edge between variable and function.
 * 
 * Convenience function that:
 * 1. Adds input metadata to the function
 * 2. Sets the gradient edge on the variable
 * 
 * @param variable The tensor to connect
 * @param function The backward function
 */
void create_gradient_edge(Tensor& variable, std::shared_ptr<Node> function);

} // namespace OwnTensor
