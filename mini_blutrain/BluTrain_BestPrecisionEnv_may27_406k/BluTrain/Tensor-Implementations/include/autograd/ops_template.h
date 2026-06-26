#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "core/AutogradMeta.h"
#include "checkpointing/GradMode.h"
#include "autograd/GraphRecorder.h"
#include <memory>
#include <type_traits>
#include "utils/Profiler.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Get edge to input tensor, creating/reusing GradAccumulator for leaves.
 * 
 * This is the shared helper for all autograd operations.
 */
Edge get_grad_edge(const Tensor& tensor);

// =============================================================================
// UNARY OPERATION TEMPLATE
// =============================================================================

/**
 * @brief Create a unary autograd operation with automatic graph building.
 * 
 * @tparam BackwardNode The backward node class (e.g., ReluBackward)
 * @tparam ForwardOp Lambda/function performing the forward computation
 * @tparam Args... Arguments to pass to BackwardNode constructor
 * 
 * @param x Input tensor
 * @param forward_op Function that computes forward(x)
 * @param backward_args Arguments for BackwardNode constructor
 * 
 * Example usage:
 * @code
 * Tensor relu(const Tensor& x) {
 *     return make_unary_op<ReluBackward>(x,
 *         [](const Tensor& input) { return where(input > 0, input, 0); },
 *         x);  // x passed to ReluBackward constructor
 * }
 * @endcode
 */
template<typename BackwardNode, typename ForwardOp, typename... Args>
Tensor make_unary_op(const Tensor& x, ForwardOp&& forward_op, Args&&... backward_args) {
    // 1. Forward pass
    AUTO_PROFILE("ForwardOp");
    Tensor result = forward_op(x);
    if (autograd::g_shape_debug)
        GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());

    // 2. Build graph if needed
    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<BackwardNode>(std::forward<Args>(backward_args)...);
        
        // Set up edge to input
        grad_fn->set_next_edge(0, get_grad_edge(x));
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

// =============================================================================
// BINARY OPERATION TEMPLATE
// =============================================================================

/**
 * @brief Create a binary autograd operation with automatic graph building.
 * 
 * @tparam BackwardNode The backward node class (e.g., AddBackward, MulBackward)
 * @tparam ForwardOp Lambda/function performing the forward computation
 * @tparam Args... Arguments to pass to BackwardNode constructor
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @param forward_op Function that computes forward(a, b)
 * @param backward_args Arguments for BackwardNode constructor
 * 
 * Example usage:
 * @code
 * Tensor add(const Tensor& a, const Tensor& b) {
 *     return make_binary_op<AddBackward>(a, b,
 *         [](const Tensor& x, const Tensor& y) { return x + y; });
 * }
 * 
 * Tensor mul(const Tensor& a, const Tensor& b) {
 *     return make_binary_op<MulBackward>(a, b,
 *         [](const Tensor& x, const Tensor& y) { return x * y; },
 *         a, b);  // a, b passed to MulBackward constructor
 * }
 * @endcode
 */
template<typename BackwardNode, typename ForwardOp, typename... Args>
Tensor make_binary_op(const Tensor& a, const Tensor& b, ForwardOp&& forward_op, Args&&... backward_args) {
    // 1. Forward pass
    AUTO_PROFILE("ForwardOp");
    Tensor result = forward_op(a, b);
    if (autograd::g_shape_debug)
        GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());

    // 2. Build graph if needed
    if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        auto grad_fn = std::make_shared<BackwardNode>(std::forward<Args>(backward_args)...);
        
        // Set up edges to inputs
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(a));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(b));
        }
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

} // namespace autograd
} // namespace OwnTensor
