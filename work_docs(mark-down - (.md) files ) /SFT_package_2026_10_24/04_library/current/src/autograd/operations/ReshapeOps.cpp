#include "autograd/operations/ReshapeOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ReshapeBackward.h"
#include "autograd/backward/TransposeBackward.h"
#include "autograd/backward/ContiguousBackward.h"
#include "autograd/backward/ContiguousBackward.h"
#include "autograd/backward/ContiguousBackward.h"
#include "profiler/AllocationTracker.h"
#include "profiler/RecordFunction.h"

namespace OwnTensor {
namespace autograd {

Tensor transpose(const Tensor& input, int dim0, int dim1) {
    PROFILER_SCOPE("autograd::transpose");
    GraphRecordMode::record_forward("RESHAPE: transpose");
    return make_unary_op<TransposeBackward>(input,
        [dim0, dim1](const Tensor& x) { return x.transpose(dim0, dim1); },
        dim0, dim1);
}

Tensor reshape(const Tensor& input, Shape new_shape) {
    PROFILER_SCOPE("autograd::reshape");
    GraphRecordMode::record_forward("RESHAPE: reshape");
    TRACK_ALLOC_SCOPE_DEEP("op::autograd::reshape");
    return make_unary_op<ReshapeBackward>(input,
        [&new_shape](const Tensor& x) { return x.reshape(new_shape); },
        input.shape());
}

Tensor view(const Tensor& input, Shape new_shape) {
    PROFILER_SCOPE("autograd::view");
    GraphRecordMode::record_forward("RESHAPE: view");
    return make_unary_op<ReshapeBackward>(input,
        [&new_shape](const Tensor& x) { return x.view(new_shape); },
        input.shape());
}

std::vector<Tensor> ReshapeBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("ReshapeBackward::apply");
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    // Backward of reshape is reshape back to original shape
    return {grads[0].reshape(input_shape_)};
}

Tensor contiguous(const Tensor& input) {
    PROFILER_SCOPE("autograd::contiguous");
    GraphRecordMode::record_forward("RESHAPE: contiguous");
    // 2026-11-07: already contiguous (offset 0) -> return the input itself and record nothing, the same no-op rule as
    // to_dtype below ("no-op preserves existing graph") and as PyTorch's contiguous().
    // Why: Tensor::contiguous() returns *this in that case (same object, no copy), and make_unary_op then stamped a new
    // ContiguousBackward onto that object -- i.e. onto the INPUT itself -- while its output_nr stayed what it was. For an
    // input that is output #1+ of a multi-output op (k_bf from fused_rms_rope_bf16 is output 1), the next op's edge became
    // (ContiguousBackward, slot 1), a slot that node does not have, and ContiguousBackward::apply returns nothing when
    // slot 0 is empty: that input's gradient was dropped silently. Found by the SFT context-parallel run (CP=2: step-0
    // grad norm 2.591 vs 2.800 without CP; k_gamma gradient 0); shown on one GPU with the real RoPE op in
    // test_contig_alias.cu (k and k_gamma gradient 0 through autograd::contiguous, correct without it).
    // Non-contiguous inputs are unchanged: a copy plus a ContiguousBackward node, as before.
    // The previous body, kept for reference (it took the make_unary_op path for every input):
    // return make_unary_op<ContiguousBackward>(input,
    //     [](const Tensor& x) { return x.contiguous(); });
    if (input.is_contiguous() && input.storage_offset() == 0) return input;
    return make_unary_op<ContiguousBackward>(input,
        [](const Tensor& x) { return x.contiguous(); });
}

std::vector<Tensor> ContiguousBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("ContiguousBackward::apply");
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    return {grads[0]};
}

Tensor to_dtype(const Tensor& input, Dtype target) {
    PROFILER_SCOPE("autograd::to_dtype");
    GraphRecordMode::record_forward("CAST: to_dtype");
    TRACK_ALLOC_SCOPE_DEEP("op::autograd::to_dtype");
    if (input.dtype() == target) return input;   // no-op preserves existing graph
    return make_unary_op<ToDtypeBackward>(input,
        [target](const Tensor& x) { return x.as_type(target); },
        input.dtype());   // ToDtypeBackward ctor: remember source dtype for the recast
}

} // namespace autograd
} // namespace OwnTensor
