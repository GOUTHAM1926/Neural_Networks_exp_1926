#include "autograd/operations/TrigonometryOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/TrigonometryBackward.h"
#include "ops/UnaryOps/Trigonometry.h"

namespace OwnTensor {
namespace autograd {

Tensor sin(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:sin");
    return make_unary_op<SinBackward>(input,
        [](const Tensor& x) { return OwnTensor::sin(x); },
        input);
}

Tensor cos(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:cos");
    return make_unary_op<CosBackward>(input,
        [](const Tensor& x) { return OwnTensor::cos(x); },
        input);
}

Tensor tan(const Tensor& input) {
    // TanBackward needs output in my design (grad * (1+y^2))
    // Let's check my implementation of TanBackward in TrigonometryBackward.h
    // Yes: TanBackward(const Tensor& output);
    // But as noted in ArithmeticsOps, I can't easily pass result to constructor via make_unary_op...
    // WAIT. If I use result in backward, I need to construct BackwardNode AFTER result is computed.
    // make_unary_op does this: 
    // Tensor result = forward_op(x);
    // ... make_shared<BackwardNode>(args...)
    // BUT the args are passed to make_unary_op at the call site!
    // So 'input' is passed. 'result' is NOT available at call site.
    // So if TanBackward needs result, I cannot use make_unary_op as is.
    // I should modify TanBackward to take Input and compute Tan(x) again (or accept overhead).
    // Or I construct it manually here.
    
    // Let's implement manually for Tan to show how it's done or change TanBackward to Input.
    // Changing to Input is consistent with what I did for Sqrt.
    // Recalculating tan(x) is acceptable.
    
    // Actually, for Tan, y = tan(x). 1 + y^2 = 1 + tan^2(x) = sec^2(x) = 1/cos^2(x).
    // If I have input x, I can compute 1/cos^2(x).
    // Let's change TanBackward to take Input.
    GraphRecordMode::record_forward("TRIG:tan");
    
    return make_unary_op<TanBackward>(input,
        [](const Tensor& x) { return OwnTensor::tan(x); },
        input); // Pass input
}

Tensor asin(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:asin");
    return make_unary_op<AsinBackward>(input,
        [](const Tensor& x) { return OwnTensor::asin(x); },
        input);
}

Tensor acos(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:acos");
    return make_unary_op<AcosBackward>(input,
        [](const Tensor& x) { return OwnTensor::acos(x); },
        input);
}

Tensor atan(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:atan");
    return make_unary_op<AtanBackward>(input,
        [](const Tensor& x) { return OwnTensor::atan(x); },
        input);
}

Tensor sinh(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:sinh");
    return make_unary_op<SinhBackward>(input,
        [](const Tensor& x) { return OwnTensor::sinh(x); },
        input);
}

Tensor cosh(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:cosh");
    return make_unary_op<CoshBackward>(input,
        [](const Tensor& x) { return OwnTensor::cosh(x); },
        input);
}

Tensor tanh(const Tensor& input) {
    // TanhBackward takes Output in PyTorch usually (1-y^2).
    // I defined TanhBackward(const Tensor& output) in header.
    // I must change it to Input to use make_unary_op easily, or implement manually.
    // Let's change to Input for consistency.
    GraphRecordMode::record_forward("TRIG:tanh");
    return make_unary_op<TanhBackward>(input,
        [](const Tensor& x) { return OwnTensor::tanh(x); },
        input);
}

Tensor asinh(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:asinh");
    return make_unary_op<AsinhBackward>(input,
        [](const Tensor& x) { return OwnTensor::asinh(x); },
        input);
}

Tensor acosh(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:acosh");
    return make_unary_op<AcoshBackward>(input,
        [](const Tensor& x) { return OwnTensor::acosh(x); },
        input);
}

Tensor atanh(const Tensor& input) {
    GraphRecordMode::record_forward("TRIG:atanh");
    return make_unary_op<AtanhBackward>(input,
        [](const Tensor& x) { return OwnTensor::atanh(x); },
        input);
}

} // namespace autograd
} // namespace OwnTensor
