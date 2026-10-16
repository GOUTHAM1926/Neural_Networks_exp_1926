#include "autograd/operations/NormalizationOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/NormalizationBackward.h"
#include "autograd/AutogradContext.h"
#include "autograd/Variable.h"
#include "ops/UnaryOps/Normalizations.h"
#include "profiler/RecordFunction.h"

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
    PROFILER_SCOPE("autograd::layer_norm");
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
    PROFILER_SCOPE("autograd::rms_norm");
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

// ============================================================================
// fused_rms_rope — thin autograd wrapper
// ============================================================================
std::pair<Tensor, Tensor> fused_rms_rope(
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int64_t pos_offset,
    float eps,
    bool interleaved,
    const Tensor& positions)
{
    PROFILER_SCOPE("autograd::fused_rms_rope");
    GraphRecordMode::record_forward("NORMALIZATION: fused_rms_rope");

    // v2: positions (optional) is validated inside fused_rms_rope_forward.
    auto result = fused_rms_rope_forward(q, k, q_gamma, k_gamma, cos_sin_cache, pos_offset, eps, interleaved,
                                         positions);
    Tensor q_out = result.q_out;
    Tensor k_out = result.k_out;

    bool any_requires_grad = (q.requires_grad() || k.requires_grad() ||
                              (q_gamma.is_valid() && q_gamma.requires_grad()) ||
                              (k_gamma.is_valid() && k_gamma.requires_grad()));

    if (GradMode::is_enabled() && any_requires_grad) {
        // v2: the backward gets the SAME positions (it undoes the same rotation).
        auto grad_fn = std::make_shared<FusedRMSRoPEBackward>(
            q, k, q_gamma, k_gamma, cos_sin_cache, result.q_rstd, result.k_rstd, pos_offset, interleaved,
            positions
        );

        if (q.requires_grad())
            grad_fn->set_next_edge(0, get_grad_edge(q));
        if (k.requires_grad())
            grad_fn->set_next_edge(1, get_grad_edge(k));
        if (q_gamma.is_valid() && q_gamma.requires_grad())
            grad_fn->set_next_edge(2, get_grad_edge(q_gamma));
        if (k_gamma.is_valid() && k_gamma.requires_grad())
            grad_fn->set_next_edge(3, get_grad_edge(k_gamma));

        q_out.set_grad_fn(grad_fn);
        q_out.set_output_nr(0);
        k_out.set_grad_fn(grad_fn);
        k_out.set_output_nr(1);

        if (q.requires_grad() || (q_gamma.is_valid() && q_gamma.requires_grad())) {
            q_out.set_requires_grad(true);
        }
        if (k.requires_grad() || (k_gamma.is_valid() && k_gamma.requires_grad())) {
            k_out.set_requires_grad(true);
        }
    }

    if (autograd::g_shape_debug) {
        GraphRecordMode::attach_forward_shape(q_out.shape(), q_out.dtype());
        GraphRecordMode::attach_forward_shape(k_out.shape(), k_out.dtype());
    }
    return {q_out, k_out};
}

} // namespace autograd
} // namespace OwnTensor

