#include "autograd/operations/ArithmeticsOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ArithmeticsBackward.h"
#include "ops/UnaryOps/Arithmetics.h"

namespace OwnTensor {
namespace autograd {

Tensor square(const Tensor& input) {
    GraphRecordMode::record_forward("ARITHMETIC: square");
    return make_unary_op<SquareBackward>(input,
        [](const Tensor& x) { return OwnTensor::square(x); },
        input);
}

Tensor sqrt(const Tensor& input) {
    GraphRecordMode::record_forward("ARITHMETIC: sqrt");
    Tensor result = make_unary_op<SqrtBackward>(input,
        [](const Tensor& x) { return OwnTensor::sqrt(x); },
        Tensor()); // Placeholder, will fill later
        
    // We need to pass the *result* to the backward constructor, not input
    // The make_unary_op template creates the node before we have the result if we pass args directly?
    // Wait, create_node logic: make_shared<Node>(args...).
    // SqrtBackward needs 'output'.
    // Standard template passes args to constructor.
    // If we need output, we can't use the simple template easily if it constructs node before result?
    // Let's check ops_template.
    // ops_template: result = forward_op(x); THEN grad_fn = make_shared(args).
    // So 'result' is available, but we aren't passing it in 'args'.
    // We can't pass 'result' in args because it's computed inside make_unary_op.
    
    // Workaround: Use input for backward if possible, or modify template.
    // Sqrt backward can be: grad / (2*sqrt(x)) = grad / (2*y).
    // If we only have x, we can recompute sqrt(x) or store x.
    // Storing output is optimization. Let's store output.
    // But we can't pass output to template arguments.
    
    // Alternative: Implement manually without template for this one, OR
    // Just recompute sqrt(x) in backward (store x).
    // Let's modify SqrtBackward to take Input instead of Output.
    // It's safer/easier with current template.
    
    return make_unary_op<SqrtBackward>(input,
         [](const Tensor& x) { return OwnTensor::sqrt(x); },
         input); // Recompute approach or changed design
}

Tensor neg(const Tensor& input) {
    GraphRecordMode::record_forward("ARITHMETIC: neg");
    return make_unary_op<NegBackward>(input,
        [](const Tensor& x) { return OwnTensor::neg(x); });
}

Tensor abs(const Tensor& input) {
    GraphRecordMode::record_forward("ARITHMETIC: abs");
    return make_unary_op<AbsBackward>(input,
        [](const Tensor& x) { return OwnTensor::abs(x); },
        input);
}

Tensor reciprocal(const Tensor& input) {
    GraphRecordMode::record_forward("ARITHMETIC: reciprocal");
    // Reciprocal backward needs output or input.
    // d(1/x) = -1/x^2 = -y^2.
    // Let's modify ReciprocalBackward to take Input (safer for now).
    // Wait, actually let's just stick to the pattern.
    // I will modify Sqrt and Reciprocal backward to take INPUT and compute what they need.
    return make_unary_op<ReciprocalBackward>(input,
        [](const Tensor& x) { return OwnTensor::reciprocal(x); },
        input); // Pass input instead of output
}

Tensor pow(const Tensor& input, float exponent) {
    GraphRecordMode::record_forward("ARITHMETIC: pow");
    return make_unary_op<PowBackward>(input,
        [exponent](const Tensor& x) { return OwnTensor::pow(x, exponent); },
        input, exponent);
}

} // namespace autograd
} // namespace OwnTensor
