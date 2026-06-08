#include "autograd/backward/FusedAttnSoftmaxMatmulBackward.h"
#include "ops/helpers/FusedKernels.h"
#include "ops/TensorOps.h"
#include "dtype/Types.h"
#ifdef WITH_CUDA
#include "ops/MatmulBackward.cuh"
#endif
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

FusedAttnSoftmaxMatmulBackward::FusedAttnSoftmaxMatmulBackward(
    const Tensor& attn_probs, const Tensor& v)
    : Node(2), saved_attn_probs_(attn_probs), saved_v_(v) {}

std::vector<Tensor> FusedAttnSoftmaxMatmulBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("FusedAttnSoftmaxMatmulBackward: no gradients provided");
    }

    const Tensor& grad_output = grads[0];  // [B, H, T, hd]
    const Tensor& attn_probs = saved_attn_probs_;  // [B, H, T, T]
    const Tensor& v = saved_v_;                     // [B, H, T, hd]

    if (!grad_output.device().is_cuda()) {
        throw std::runtime_error(
            "FusedAttnSoftmaxMatmulBackward: only CUDA tensors are supported");
    }

    // Step 1: Compute matmul gradients
    //   grad_attn_probs = grad_output @ v.T   [B,H,T,T]
    //   grad_v          = attn_probs.T @ grad_output  [B,H,T,hd]

    Tensor grad_attn_probs;
    Tensor grad_v;

#ifdef WITH_CUDA
    // 4D batched matmul backward: out = attn_probs @ v
    // Uses the same path as MatmulBackward Case 3
    {
        Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
        Tensor a  = attn_probs.is_contiguous()  ? attn_probs  : attn_probs.contiguous();
        Tensor b  = v.is_contiguous()           ? v           : v.contiguous();

        grad_attn_probs = Tensor(a.shape(), a.dtype(), a.device());
        grad_v = Tensor(b.shape(), b.dtype(), b.device());

        cuda_matmul_backward(go, a, b, grad_attn_probs, grad_v, 0);
    }
#else
    {
        Tensor v_t = v.t();
        grad_attn_probs = OwnTensor::matmul(grad_output, v_t);
        Tensor ap_t = attn_probs.t();
        grad_v = OwnTensor::matmul(ap_t, grad_output);
    }
#endif

    // Step 2: Softmax backward on grad_attn_probs using attn_probs
    //   grad_attn_weights[i] = attn_probs[i] * (grad_attn_probs[i] - dot)
    //   where dot = sum_j(grad_attn_probs[j] * attn_probs[j])
    // The tril mask zeros are handled automatically (attn_probs[i]=0 for masked).

    const int64_t cols = attn_probs.shape().dims.back();
    const int64_t rows = attn_probs.numel() / cols;

    Tensor grad_attn_weights(attn_probs.shape(), grad_attn_probs.opts());

    if (grad_attn_probs.dtype() == Dtype::Float32) {
        cuda::fused_tril_softmax_backward_cuda(
            grad_attn_probs.data<float>(), attn_probs.data<float>(),
            grad_attn_weights.data<float>(), rows, cols);
    } else if (grad_attn_probs.dtype() == Dtype::Float16) {
        cuda::fused_tril_softmax_backward_cuda(
            grad_attn_probs.data<float16_t>(), attn_probs.data<float16_t>(),
            grad_attn_weights.data<float16_t>(), rows, cols);
    } else if (grad_attn_probs.dtype() == Dtype::Bfloat16) {
        cuda::fused_tril_softmax_backward_cuda(
            grad_attn_probs.data<bfloat16_t>(), attn_probs.data<bfloat16_t>(),
            grad_attn_weights.data<bfloat16_t>(), rows, cols);
    } else {
        throw std::runtime_error("FusedAttnSoftmaxMatmulBackward: unsupported dtype");
    }

    // Edge 0 = grad_attn_weights, Edge 1 = grad_v
    return {grad_attn_weights, grad_v};
}

} // namespace autograd
} // namespace OwnTensor
