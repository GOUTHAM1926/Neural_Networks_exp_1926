#include "autograd/operations/MatrixOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/MatrixBackward.h"
#include "autograd/backward/LinearBackward.h"
#ifdef WITH_CUDA
#include "ops/LinearKernels.cuh"
#endif
#include "ops/Kernels.h"
#include "ops/TensorOps.h"
#include "utils/Profiler.h"
#include "checkpointing/GradMode.h"
#include <algorithm>
#include "device/AllocationTracker.h"

namespace OwnTensor {
namespace autograd {

Tensor matmul(const Tensor& a, const Tensor& b) {
    GraphRecordMode::record_forward("MATRIX: matmul");
    // If a is [B, T, C] and b is [C, V], flatten a to [B*T, C] for a single giant GEMM
    TRACK_ALLOC_SCOPE("L19:autograd::matmul");
    if (a.ndim() == 3 && b.ndim() == 2 && a.shape().dims[2] == b.shape().dims[0]) {
        Shape original_a_shape = a.shape();
        int64_t B = original_a_shape.dims[0];
        int64_t T = original_a_shape.dims[1];
        int64_t C = original_a_shape.dims[2];
        int64_t V = b.shape().dims[1];
        
        // Flatten A to [B*T, C]
        Tensor a_flat = a.view(Shape{{B * T, C}});
        
        // Perform 2D matmul
        Tensor res_flat = make_binary_op<MatmulBackward>(a_flat, b,
            [](const Tensor& x, const Tensor& y) { return OwnTensor::matmul(x, y); },
            a_flat, b);
            
        // Reshape result back to [B, T, V]
        return res_flat.view(Shape{{B, T, V}});
    }
    
    return make_binary_op<MatmulBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return OwnTensor::matmul(x, y); },
        a, b);  // Pass a, b to MatmulBackward constructor
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    GraphRecordMode::record_forward("MATRIX: linear");
    // Forward: input @ weight + bias (bias handled by Tensor ops, not autograd ops)
    auto forward_fn = [](const Tensor& x, const Tensor& w, const Tensor& b) {
#ifdef WITH_CUDA
        if (x.is_cuda()) {
             Tensor out = Tensor::empty(Shape{{x.shape().dims[0], w.shape().dims[1]}}, x.opts()); 
             // Shape inference: x [..., In], w [In, Out]?
             // Actually, assuming w is correct for matmul(x,w).
             // We need to be careful about shape inference.
             // Best to utilize `cuda_linear_forward`'s logic or internal matmul helpers but we don't return new tensor there easily.
             
             // Since `cuda_linear_forward` in my implementation (defined below/above) 
             // does `output = matmul(x, w); add_bias(output, b);` internally,
             // we can just call it passing a dummy output tensor reference to be assigned, 
             // OR modify `cuda_linear_forward` to RETURN a tensor.
             
             // The implementation I wrote accepts `Tensor& output`.
             Tensor output; 
             {
                 AUTO_PROFILE_CUDA("Forward::Linear_Forward_CUDA");
                 cuda_linear_forward(x, w, b, output);
             }
             return output;
        }
#endif
        // We use raw Tensor operations here, not autograd wrappers
        // to avoid creating intermediate nodes
        Tensor out = OwnTensor::matmul(x, w);
        if (b.is_valid()) {
            out = out + b;
        }
        return out;
    };
    
    // Create output tensor with attached grad_fn
    Tensor result = forward_fn(input, weight, bias);
    
    if (GradMode::is_enabled() && (input.requires_grad() || weight.requires_grad())) {
        auto grad_fn = std::make_shared<LinearBackward>(input, weight);
        
        // Connect edges
        if (input.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(input));
        }
        if (weight.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(weight));
        }
        if (bias.is_valid() && bias.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(bias));
        }
        
        result.set_grad_fn(grad_fn);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}



Tensor addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2, float alpha, float beta) {
    TRACK_ALLOC_SCOPE("L108:autograd::addmm");
    GraphRecordMode::record_forward("MATRIX: addmm");
    Tensor result = OwnTensor::addmm(input, mat1, mat2, alpha, beta);
    
    if (GradMode::is_enabled() && (input.requires_grad() || mat1.requires_grad() || mat2.requires_grad())) {
        auto grad_fn = std::make_shared<AddmmBackward>(input, mat1, mat2, alpha, beta);
        
        if (input.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(input));
        }
        if (mat1.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(mat1));
        }
        if (mat2.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(mat2));
        }
        
        result.set_grad_fn(grad_fn);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

} // namespace autograd
} // namespace OwnTensor