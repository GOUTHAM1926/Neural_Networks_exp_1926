#include "autograd/operations/TrilOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/TrilBackward.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace autograd {

Tensor tril(const Tensor& input, int64_t diagonal, double value) {
    GraphRecordMode::record_forward("MATRIX: tril");
    return make_unary_op<TrilBackward>(input,
        [diagonal, value](const Tensor& x) {
            return OwnTensor::tril(x, diagonal, value);
        },
        input, diagonal, value);
}

} // namespace autograd
} // namespace OwnTensor


