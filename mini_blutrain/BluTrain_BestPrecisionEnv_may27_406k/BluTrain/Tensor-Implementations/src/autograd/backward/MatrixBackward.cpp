#include "autograd/backward/MatrixBackward.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/UnaryOps/Reduction.h"
#include "device/DeviceCore.h"
#ifdef WITH_CUDA
#include "ops/MatmulBackward.cuh"
#endif
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

MatmulBackward::MatmulBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> MatmulBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MatmulBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    
#ifdef WITH_CUDA
    if (grad_output.is_cuda()) {
        // Case 1: Both operands are 2D [M,K] @ [K,N]
        if (saved_a_.ndim() == 2 && saved_b_.ndim() == 2 && grad_output.ndim() == 2) {
            Tensor grad_a(saved_a_.shape(), saved_a_.dtype(), saved_a_.device());
            Tensor grad_b(saved_b_.shape(), saved_b_.dtype(), saved_b_.device());
            cuda_matmul_backward(grad_output, saved_a_, saved_b_, grad_a, grad_b, 0);
            return {grad_a, grad_b};
        }

        // Case 2: Linear layer [B,T,Hidden] @ [Hidden,Out] -> [B,T,Out]
        if (saved_b_.ndim() == 2 && saved_a_.ndim() > 2 && grad_output.ndim() == saved_a_.ndim()) {
            int64_t hidden_dim = saved_a_.shape().dims.back();
            int64_t output_dim = grad_output.shape().dims.back();

            Tensor a_flat = saved_a_.reshape(Shape{{-1, hidden_dim}});
            Tensor g_flat = grad_output.reshape(Shape{{-1, output_dim}});

            Tensor grad_a_flat(a_flat.shape(), a_flat.dtype(), a_flat.device());
            Tensor grad_b(saved_b_.shape(), saved_b_.dtype(), saved_b_.device());

            cuda_matmul_backward(g_flat, a_flat, saved_b_, grad_a_flat, grad_b, 0);

            Tensor grad_a = grad_a_flat.reshape(saved_a_.shape());
            return {grad_a, grad_b};
        }

        // Case 3: Batched 4D matmul [B,H,T,K] @ [B,H,K,N] (attention Q@K^T, attn@V)
        // Make all inputs contiguous so cuBLAS can use a single batch=B*H call
        // instead of looping over outer batch dim with batch=H per iteration.
        if (saved_a_.ndim() == 4 && saved_b_.ndim() == 4) {
            Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
            Tensor a  = saved_a_.is_contiguous()    ? saved_a_    : saved_a_.contiguous();
            Tensor b  = saved_b_.is_contiguous()    ? saved_b_    : saved_b_.contiguous();
            Tensor grad_a(a.shape(), a.dtype(), a.device());
            Tensor grad_b(b.shape(), b.dtype(), b.device());
            device::set_cuda_device(grad_output.device().index);
            cuda_matmul_backward(go, a, b, grad_a, grad_b, 0);
            return {grad_a, grad_b};
        }
    }
#endif
    
    // TODO: CPU fallback
    // CPU/General CUDA path with explicit transpose
    // For 4D tensors, grad_output may be non-contiguous with broken batch strides
    // (e.g. after TransposeBackward on merge-heads), which cuBLAS strided batched
    // cannot handle. Make it contiguous to fix incorrect stride_a/stride_b estimates.
    const Tensor& grad_ref = grad_output;
    Tensor grad_contig;
    if (grad_output.ndim() == 4 && !grad_output.is_contiguous()) {
        grad_contig = grad_output.contiguous();
    }
    const Tensor& grad = grad_contig.is_valid() ? grad_contig : grad_ref;

    Tensor b_t = saved_b_.t();

    Tensor grad_a = OwnTensor::matmul(grad, b_t);
    grad_a = reduce_to_shape(grad_a, saved_a_.shape());

    Tensor grad_b;

    // Optimization for Linear Layer case: [Batch, T, Hidden] @ [Hidden, Out]
    // where we want to avoid materializing [Batch, Hidden, Out] before reduction
    if (saved_b_.ndim() == 2 && saved_a_.ndim() > 2) {
        int64_t hidden_dim = saved_a_.shape().dims.back();
        int64_t output_dim = grad.shape().dims.back();

        // Reshape [B, T, Hidden] -> [B*T, Hidden]
        Tensor a_flat = saved_a_.reshape(Shape{{-1, hidden_dim}});
        // Reshape [B, T, C] -> [B*T, C]
        Tensor g_flat = grad.reshape(Shape{{-1, output_dim}});

        // [Hidden, BT] @ [BT, C] -> [Hidden, C] (implicitly sums over B*T)
        grad_b = OwnTensor::matmul(a_flat.t(), g_flat);
    } else {
        // General case
        Tensor a_t = saved_a_.t();
        grad_b = OwnTensor::matmul(a_t, grad);
        grad_b = reduce_to_shape(grad_b, saved_b_.shape());
    }
    
    return {grad_a, grad_b};
}

AddmmBackward::AddmmBackward(const Tensor& input, const Tensor& mat1, const Tensor& mat2, float alpha, float beta)
    : Node(3), saved_mat1_(mat1), saved_mat2_(mat2), saved_input_shape_(input.shape()), alpha_(alpha), beta_(beta) {
    // Input is edge 0 (bias), mat1 is edge 1 (linear input), mat2 is edge 2 (weight)
}

std::vector<Tensor> AddmmBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("AddmmBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    Tensor grad_input;
    Tensor grad_mat1;
    Tensor grad_mat2;
    
    // 1. Gradient w.r.t input (bias): beta * grad_output, reduced to input shape
    if (beta_ != 0.0f) {
        if (beta_ == 1.0f) {
            grad_input = reduce_to_shape(grad_output, saved_input_shape_);
        } else {
            Tensor beta_t = Tensor::full(Shape{{1}}, grad_output.opts(), beta_);
            grad_input = reduce_to_shape(grad_output * beta_t, saved_input_shape_);
        }
    } else {
        grad_input = Tensor(); 
    }
    
    // 2. Gradient w.r.t mat1: alpha * grad_output @ mat2.T
    // 3. Gradient w.r.t mat2: alpha * mat1.T @ grad_output
    
    // 2. Gradient w.r.t mat1: alpha * grad_output @ mat2.T
    // 3. Gradient w.r.t mat2: alpha * mat1.T @ grad_output
    
    if (alpha_ != 0.0f) {
#ifdef WITH_CUDA
        // Optimized CUDA path (similar to MatmulBackward)
        // Currently only if alpha == 1.0 to avoid extra scaling kernel launches 
        // (though we could scale grad_output if needed, but keeping it simple for now as alpha is usually 1)
        if (alpha_ == 1.0f && grad_output.is_cuda() && saved_mat2_.ndim() == 2) {
             
             // Case 1: Pure 2D matmul [M,K] @ [K,N]
             if (saved_mat1_.ndim() == 2 && grad_output.ndim() == 2) {
                 grad_mat1 = Tensor(saved_mat1_.shape(), saved_mat1_.dtype(), saved_mat1_.device());
                 grad_mat2 = Tensor(saved_mat2_.shape(), saved_mat2_.dtype(), saved_mat2_.device());
                device::set_cuda_device(grad_output.device().index);
                cuda_matmul_backward(grad_output, saved_mat1_, saved_mat2_, grad_mat1, grad_mat2, 0);
                 
                 // If saved_mat1/2 required reduction (broadcasting), handle it? 
                 // cuda_matmul_backward assumes matching shapes for 2D.
                 // MatmulBackward doesn't do reduction after cuda_matmul_backward for 2D case, assumes strict match.
                 // But addmm might have broadcasted mat1/mat2? 
                 // The error "mat1 and mat2 must be 2D" implies strict 2D expectation for now in some paths,
                 // but GenMatmul supports broadcasting.
                 // However, MatmulBackward logic I copied assumes strict shapes for Case 1.
                 // Let's assume standard use case first.
             }
             // Case 2: Linear layer [B,T,Hidden] @ [Hidden,Out] -> [B,T,Out]
             else if (saved_mat1_.ndim() > 2 && grad_output.ndim() == saved_mat1_.ndim()) {
                 int64_t hidden_dim = saved_mat1_.shape().dims.back();
                 int64_t output_dim = grad_output.shape().dims.back();
                 
                 // Flatten: [B,T,Hidden] -> [B*T, Hidden], [B,T,Out] -> [B*T, Out]
                 Tensor a_flat = saved_mat1_.reshape(Shape{{-1, hidden_dim}});
                 Tensor g_flat = grad_output.reshape(Shape{{-1, output_dim}});
                 
                 Tensor grad_mat1_flat(a_flat.shape(), a_flat.dtype(), a_flat.device());
                 grad_mat2 = Tensor(saved_mat2_.shape(), saved_mat2_.dtype(), saved_mat2_.device());
                 
                 cuda_matmul_backward(g_flat, a_flat, saved_mat2_, grad_mat1_flat, grad_mat2, 0);
                 
                 grad_mat1 = grad_mat1_flat.reshape(saved_mat1_.shape());
             }
        }
#endif
        
        // Fallback or if not handled by optimized path (e.g. alpha != 1, or weird shapes)
        if (!grad_mat1.is_valid() || !grad_mat2.is_valid()) {
            Tensor alpha_t;
            if (alpha_ != 1.0f) {
                 alpha_t = Tensor::full(Shape{{1}}, grad_output.opts(), alpha_);
            }
    
            // grad_mat1
            Tensor mat2_t = saved_mat2_.t();
            grad_mat1 = OwnTensor::matmul(grad_output, mat2_t);
            if (alpha_ != 1.0f) grad_mat1 = grad_mat1 * alpha_t;
            grad_mat1 = reduce_to_shape(grad_mat1, saved_mat1_.shape());

            // grad_mat2
            Tensor mat1_t = saved_mat1_.t();
            grad_mat2 = OwnTensor::matmul(mat1_t, grad_output);
            if (alpha_ != 1.0f) grad_mat2 = grad_mat2 * alpha_t;
            grad_mat2 = reduce_to_shape(grad_mat2, saved_mat2_.shape());
        }
    }
    
    return {grad_input, grad_mat1, grad_mat2};
}

} // namespace autograd
} // namespace OwnTensor