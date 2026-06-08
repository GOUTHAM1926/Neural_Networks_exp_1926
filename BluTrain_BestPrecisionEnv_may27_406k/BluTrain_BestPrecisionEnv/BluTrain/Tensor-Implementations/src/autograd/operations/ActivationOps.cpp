#include "autograd/operations/ActivationOps.h"
#include "autograd/backward/ActivationBackward.h"
#include "autograd/backward/TrilBackward.h"
#include "autograd/ops_template.h"
#include "ops/UnaryOps/Activations.h"
#include "dtype/DtypeTraits.h"

namespace OwnTensor {
namespace autograd {

// ============================================================================
// relu
// ============================================================================
Tensor relu(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: ReLU");

    Tensor output = relu_forward(x);

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<ReluBackward>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// gelu
// ============================================================================
Tensor gelu(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: GeLU");

    Tensor output = gelu_forward(x);

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<GeLUBackward>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// sigmoid
// ============================================================================
Tensor sigmoid(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: Sigmoid");

    Tensor output = sigmoid_forward(x);

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<SigmoidBackward>(output.detach());
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// softmax
// ============================================================================
Tensor softmax(const Tensor& x, int64_t dim) {
    GraphRecordMode::record_forward("ACTIVATION: Softmax");
    int64_t ndim = x.ndim();
    if (dim < 0) dim += ndim;

    // ── Fusion: tril + softmax → fused_tril_softmax ──────────────
    if (x.device().is_cuda() && is_float(x.dtype()) && dim == ndim - 1) {
        auto grad_fn_node = x.grad_fn();
        if (grad_fn_node) {
            auto* tril_node = dynamic_cast<TrilBackward*>(grad_fn_node.get());
            if (tril_node && tril_node->has_saved_input()) {
                Tensor original_input = tril_node->saved_input();
                Tensor& input_mut = const_cast<Tensor&>(original_input);
                return fused_tril_softmax(input_mut,
                                           tril_node->diagonal(),
                                           tril_node->value());
            }
        }
    }

    Tensor output = softmax_forward(x, dim);

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<SoftmaxBackward>(output.detach(), dim);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// fused_tril_softmax
// GPU: OwnTensor::fused_tril_softmax() in FusedKernels.cu handles autograd internally.
// CPU: composes tril() + softmax_forward() via fused_tril_softmax_forward().
// ============================================================================
Tensor fused_tril_softmax(const Tensor& x, int64_t diagonal, double value) {
    GraphRecordMode::record_forward("ACTIVATION: FusedTrilSoftmax");

    Tensor result = fused_tril_softmax_forward(x, diagonal, value);

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

// ============================================================================
// dropout
// ============================================================================
Tensor dropout(const Tensor& x, float p, bool training) {
    GraphRecordMode::record_forward("ACTIVATION: Dropout");

    if (!training || p == 0.0f) {
        if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(x.shape(), x.dtype());
        return x;
    }

    auto [output, mask] = dropout_forward(x, p);
    float scale = 1.0f / (1.0f - p);

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<DropoutBackward>(mask.detach(), scale);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// swiglu
// ============================================================================
Tensor swiglu(const Tensor &x) {
    GraphRecordMode::record_forward("ACTIVATION: SwiGLU");

    Tensor output = swiglu_forward(x);

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<SwiGLUBackward>(x);
        Tensor &x_mut = const_cast<Tensor &>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// fused_bias_gelu
// ============================================================================
Tensor fused_bias_gelu(const Tensor& input, const Tensor& bias) {
    GraphRecordMode::record_forward("ACTIVATION: FusedBiasGeLU");

    Tensor output = fused_bias_gelu_forward(input, bias);

    if (GradMode::is_enabled() && (input.requires_grad() || bias.requires_grad())) {
        auto grad_fn = std::make_shared<FusedBiasGeLUBackward>(input, bias);
        Tensor& input_mut = const_cast<Tensor&>(input);
        Tensor& bias_mut  = const_cast<Tensor&>(bias);
        grad_fn->set_next_edge(0, get_grad_edge(input_mut));
        grad_fn->set_next_edge(1, get_grad_edge(bias_mut));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

} // namespace autograd
} // namespace OwnTensor
