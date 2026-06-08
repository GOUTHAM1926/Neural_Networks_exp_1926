#include "autograd/operations/ReshapeOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ReshapeBackward.h"
#include "autograd/backward/TransposeBackward.h"
#include "autograd/backward/ContiguousBackward.h"
#include "autograd/backward/ContiguousBackward.h"
#include "autograd/backward/ContiguousBackward.h"
#include "device/AllocationTracker.h"

namespace OwnTensor {
namespace autograd {

Tensor transpose(const Tensor& input, int dim0, int dim1) {
    GraphRecordMode::record_forward("RESHAPE: transpose");
    return make_unary_op<TransposeBackward>(input,
        [dim0, dim1](const Tensor& x) { return x.transpose(dim0, dim1); },
        dim0, dim1);
}

Tensor reshape(const Tensor& input, Shape new_shape) {
    GraphRecordMode::record_forward("RESHAPE: reshape");
    TRACK_ALLOC_SCOPE("L21:autograd::reshape");
    return make_unary_op<ReshapeBackward>(input,
        [&new_shape](const Tensor& x) { return x.reshape(new_shape); },
        input.shape());
}

Tensor view(const Tensor& input, Shape new_shape) {
    GraphRecordMode::record_forward("RESHAPE: view");
    return make_unary_op<ReshapeBackward>(input,
        [&new_shape](const Tensor& x) { return x.view(new_shape); },
        input.shape());
}

std::vector<Tensor> ReshapeBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    // Backward of reshape is reshape back to original shape
    return {grads[0].reshape(input_shape_)};
}

Tensor contiguous(const Tensor& input) {
    GraphRecordMode::record_forward("RESHAPE: contiguous");
    return make_unary_op<ContiguousBackward>(input,
        [](const Tensor& x) { return x.contiguous(); });
}

std::vector<Tensor> ContiguousBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    return {grads[0]};
}

} // namespace autograd
} // namespace OwnTensor
