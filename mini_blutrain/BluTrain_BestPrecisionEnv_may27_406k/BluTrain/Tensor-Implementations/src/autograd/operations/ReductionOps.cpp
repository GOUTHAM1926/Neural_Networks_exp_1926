#include "autograd/operations/ReductionOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ReductionBackward.h"
#include "ops/UnaryOps/Reduction.h"

namespace OwnTensor {
namespace autograd {

Tensor sum(const Tensor& x) {
    GraphRecordMode::record_forward("REDUCTION: sum");
    return make_unary_op<SumBackward>(x,
        [](const Tensor& input) { return reduce_sum(input); },
        x.shape());  // Pass shape to SumBackward constructor
}

Tensor mean(const Tensor& x) {
    GraphRecordMode::record_forward("REDUCTION: mean");
    return make_unary_op<MeanBackward>(x,
        [](const Tensor& input) { return reduce_mean(input); },
        x.shape(), x.numel());  // Pass shape and numel to MeanBackward constructor
}

} // namespace autograd
} // namespace OwnTensor
