#include "autograd/backward/BinaryBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Reduction.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor {
namespace autograd {

static Tensor reduce_to_shape(const Tensor& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) return grad;
    
    Tensor res = grad;
    int64_t grad_ndim = grad.ndim();
    int64_t target_ndim = target_shape.dims.size();
    
    std::vector<int64_t> dims_to_sum;
    
    // 1. Handle rank mismatch (leading dimensions)
    if (grad_ndim > target_ndim) {
        for (int64_t i = 0; i < grad_ndim - target_ndim; ++i) {
            dims_to_sum.push_back(i);
        }
    }
    
    // 2. Handle broadcasting in shared dimensions
    for (int64_t i = 0; i < target_ndim; ++i) {
        int64_t target_dim_idx = target_ndim - 1 - i;
        int64_t grad_dim_idx = grad_ndim - 1 - i;
        
        if (target_shape.dims[target_dim_idx] == 1 && grad.shape().dims[grad_dim_idx] > 1) {
            dims_to_sum.push_back(grad_dim_idx);
        }
    }
    
    if (!dims_to_sum.empty()) {
        res = reduce_sum(res, dims_to_sum, true);
    }
    
    if (res.shape() != target_shape) {
        res = res.reshape(target_shape);
    }
    
    return res;
}

// ============================================================================
// AddBackward
// ============================================================================

AddBackward::AddBackward(const Shape& a_shape, const Shape& b_shape)
    : Node(2), shape_a_(a_shape), shape_b_(b_shape) {}

std::vector<Tensor> AddBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("AddBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output, grad_b = grad_output
    return {reduce_to_shape(grad_output, shape_a_), 
            reduce_to_shape(grad_output, shape_b_)};
}

// ============================================================================
// MulBackward
// ============================================================================

MulBackward::MulBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> MulBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MulBackward: no gradients provided");
    }

    const Tensor& grad_output = grads[0];

    if (!grad_output.is_valid()) throw std::runtime_error("MulBackward: grad_output is invalid");

    const auto& edges = next_edges();
    bool need_grad_a = (edges.size() > 0 && edges[0].is_valid());
    bool need_grad_b = (edges.size() > 1 && edges[1].is_valid());

    Tensor grad_a, grad_b;

    if (need_grad_a) {
        grad_a = reduce_to_shape(grad_output * saved_b_, saved_a_.shape());
    }
    if (need_grad_b) {
        grad_b = reduce_to_shape(grad_output * saved_a_, saved_b_.shape());
    }

    return {grad_a, grad_b};
}

// ============================================================================
// SubBackward
// ============================================================================

SubBackward::SubBackward(const Shape& a_shape, const Shape& b_shape)
    : Node(2), shape_a_(a_shape), shape_b_(b_shape) {}

std::vector<Tensor> SubBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SubBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output, grad_b = -grad_output
    Tensor neg_grad = grad_output * -1.0;
    return {reduce_to_shape(grad_output, shape_a_), 
            reduce_to_shape(neg_grad, shape_b_)};
}

// ============================================================================
// DivBackward
// ============================================================================

DivBackward::DivBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> DivBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("DivBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output / b
    Tensor grad_a = grad_output / saved_b_;
    
    // grad_b = -grad_output * a / b^2
    Tensor term1 = grad_output * -1.0;
    Tensor term2 = term1 * saved_a_;
    Tensor b_sq = saved_b_ * saved_b_;
    Tensor grad_b = term2 / b_sq;
    
    return {reduce_to_shape(grad_a, saved_a_.shape()), 
            reduce_to_shape(grad_b, saved_b_.shape())};
}


} // namespace autograd
} // namespace OwnTensor