#include "autograd/backward/FusedTrilSoftmaxBackward.h"
#include "ops/helpers/FusedKernels.h"
#include "dtype/Types.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

FusedTrilSoftmaxBackward::FusedTrilSoftmaxBackward(
    const Tensor& output, int64_t trilDiag, double value)
    : Node(1), saved_output_(output), trilDiag_(trilDiag), value_(value) {}

std::vector<Tensor> FusedTrilSoftmaxBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("FusedTrilSoftmaxBackward: no gradients provided");
    }

    const Tensor& grad_output = grads[0];
    const Tensor& s = saved_output_;

    if (!grad_output.device().is_cuda()) {
        throw std::runtime_error(
            "FusedTrilSoftmaxBackward: only CUDA tensors are supported");
    }

    Tensor grad_input(s.shape(), grad_output.opts());

    const int64_t cols = s.shape().dims.back();
    const int64_t rows = s.numel() / cols;

    if (grad_output.dtype() == Dtype::Float32) {
        cuda::fused_tril_softmax_backward_cuda(grad_output.data<float>(), s.data<float>(), grad_input.data<float>(), rows, cols);
    } else if (grad_output.dtype() == Dtype::Float16) {
        cuda::fused_tril_softmax_backward_cuda(grad_output.data<float16_t>(), s.data<float16_t>(), grad_input.data<float16_t>(), rows, cols);
    } else if (grad_output.dtype() == Dtype::Bfloat16) {
        cuda::fused_tril_softmax_backward_cuda(grad_output.data<bfloat16_t>(), s.data<bfloat16_t>(), grad_input.data<bfloat16_t>(), rows, cols);
    } else {
        throw std::runtime_error("FusedTrilSoftmaxBackward: unsupported dtype");
    }

    return {grad_input};
}

} // namespace autograd
} // namespace OwnTensor
