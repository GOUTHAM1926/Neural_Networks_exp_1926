#include "autograd/operations/BinaryOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/BinaryBackward.h"
#include "ops/TensorOps.h"
#include "device/AllocationTracker.h"

namespace OwnTensor {
namespace autograd {

Tensor add(const Tensor& a, const Tensor& b) {
    TRACK_ALLOC_SCOPE("L11:autograd::add");
    GraphRecordMode::record_forward("ARITHMETIC: add");
    return make_binary_op<AddBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator+(x, y); },
        a.shape(), b.shape());
}

Tensor& add_(Tensor& a, const Tensor& b) {
    TRACK_ALLOC_SCOPE("L18:autograd::add_");
    GraphRecordMode::record_forward("ARITHMETIC: add_ (inplace)");

    bool needs_grad = GradMode::is_enabled() && (a.requires_grad() || b.requires_grad());

    // Capture edges BEFORE in-place mutation overwrites a's grad_fn
    Edge edge_a, edge_b;
    if (needs_grad) {
        edge_a = get_grad_edge(a);
        Tensor& b_mut = const_cast<Tensor&>(b);
        edge_b = get_grad_edge(b_mut);
    }

    Shape shape_a = a.shape();
    Shape shape_b = b.shape();

    // In-place add (uses existing operator+= with CUDA kernel)
    a += b;

    // Rewire autograd graph
    if (needs_grad) {
        auto grad_fn = std::make_shared<AddBackward>(shape_a, shape_b);
        grad_fn->set_next_edge(0, edge_a);
        grad_fn->set_next_edge(1, edge_b);
        a.set_grad_fn(grad_fn);
        a.set_requires_grad(true);
    }

    return a;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    GraphRecordMode::record_forward("ARITHMETIC: mul");
    TRACK_ALLOC_SCOPE("L19:autograd::mul");
    return make_binary_op<MulBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator*(x, y); },
        a, b);  // Pass a, b to MulBackward constructor
}

Tensor sub(const Tensor& a, const Tensor& b) {
    GraphRecordMode::record_forward("ARITHMETIC: sub");
    return make_binary_op<SubBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator-(x, y); },
        a.shape(), b.shape());
}

Tensor div(const Tensor& a, const Tensor& b) {
    GraphRecordMode::record_forward("ARITHMETIC: div");
    return make_binary_op<DivBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator/(x, y); },
        a, b);  // Pass a, b to DivBackward constructor
}

} // namespace autograd
} // namespace OwnTensor
