#include "autograd/operations/NormalizationOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/NormalizationBackward.h"
#include "autograd/AutogradContext.h"
#include "autograd/Variable.h"
#include "ops/UnaryOps/Normalizations.h"

namespace OwnTensor {
namespace autograd {

// ============================================================================
// layer_norm — thin autograd wrapper
// Computation is fully delegated to layer_norm_forward() in Normalizations.cpp
// ============================================================================
Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int normalized_shape,
    float eps)
{
    GraphRecordMode::record_forward("NORMALIZATION: layer_norm");

    // Pure math forward (CPU/GPU dispatch handled inside)
    auto result = layer_norm_forward(input, weight, bias, normalized_shape, eps);
    Tensor output = result.output;

    // Construct Autograd Graph
    if (GradMode::is_enabled() && (input.requires_grad() || (weight.is_valid() && weight.requires_grad()) || (bias.is_valid() && bias.requires_grad()))) {

        auto grad_fn = std::make_shared<LayerNormBackward>(
            input, result.mean, result.rstd, weight, normalized_shape, eps
        );

        if (input.requires_grad())
            grad_fn->set_next_edge(0, get_grad_edge(input));
        if (weight.is_valid() && weight.requires_grad())
            grad_fn->set_next_edge(1, get_grad_edge(weight));
        if (bias.is_valid() && bias.requires_grad())
            grad_fn->set_next_edge(2, get_grad_edge(bias));

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

// ============================================================================
// rms_norm — thin autograd wrapper
// ============================================================================
Tensor rms_norm(
    const Tensor& input,
    const Tensor& weight,
    int normalized_shape,
    float eps)
{
    GraphRecordMode::record_forward("NORMALIZATION: rms_norm");

    auto result = rms_norm_forward(input, weight, normalized_shape, eps);
    Tensor output = result.output;

    if (GradMode::is_enabled() && (input.requires_grad() || (weight.is_valid() && weight.requires_grad()))) {
        auto grad_fn = std::make_shared<RMSNormBackward>(
            input, result.rstd, weight, normalized_shape, eps
        );

        if (input.requires_grad())
            grad_fn->set_next_edge(0, get_grad_edge(input));
        if (weight.is_valid() && weight.requires_grad())
            grad_fn->set_next_edge(1, get_grad_edge(weight));

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

} // namespace autograd
} // namespace OwnTensor
