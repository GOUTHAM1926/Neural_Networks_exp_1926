#pragma once
// =============================================================================
// tp_ops.h — Tensor Parallel operations
//   - Async all-reduce helpers (compute-communication overlap)
//   - TpAllReduceBackward autograd node
//   - Column-parallel linear backward with overlap
//   - column_parallel_addmm / column_parallel_matmul
// =============================================================================

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/ops_template.h"
#include "../../dist/distributed.h"

using namespace OwnTensor;

namespace gpt2 {
namespace tp {

// =============================================================================
// Async All-Reduce Helpers
// =============================================================================

// Launch all-reduce on NCCL stream, return Work handle for deferred wait.
inline std::shared_ptr<Work> allreduce_launch(Tensor& t,
                                               std::shared_ptr<ProcessGroupNCCL>& pg) {
    cudaEvent_t compute_ready;
    cudaEventCreateWithFlags(&compute_ready, cudaEventDisableTiming);
    cudaEventRecord(compute_ready, OwnTensor::cuda::getCurrentStream());
    cudaStreamWaitEvent(pg->get_stream(), compute_ready, 0);
    cudaEventDestroy(compute_ready);

    auto work = pg->all_reduce_async(t.data<float>(), t.data<float>(),
                                     t.numel(), Dtype::Float32, ::sum, false);
    work->event_record();
    return work;
}

// Wait for a previously launched all-reduce
inline void allreduce_wait(std::shared_ptr<Work>& work) {
    work->streamWait(OwnTensor::cuda::getCurrentStream());
}

// Blocking all-reduce SUM (forward-only, not tracked by autograd)
inline void allreduce_inplace(Tensor& t, std::shared_ptr<ProcessGroupNCCL>& pg) {
    auto work = allreduce_launch(t, pg);
    allreduce_wait(work);
}

// =============================================================================
// Backward-only all-reduce autograd node
// Forward: identity. Backward: all-reduce SUM across TP group.
// =============================================================================

class TpAllReduceBackward : public Node {
    std::shared_ptr<ProcessGroupNCCL> pg_;
public:
    TpAllReduceBackward(std::shared_ptr<ProcessGroupNCCL> pg)
        : Node(1), pg_(pg) {}

    const char* name() const override { return "TpAllReduceBackward"; }

    variable_list apply(variable_list&& grads) override {
        if (grads.empty()) return {};
        Tensor g = grads[0];
        auto work = allreduce_launch(g, pg_);
        allreduce_wait(work);
        return {g};
    }

    void release_saved_variables() override { pg_.reset(); }
};

inline void insert_backward_allreduce(Tensor& t, std::shared_ptr<ProcessGroupNCCL>& pg) {
    if (!autograd::GradMode::is_enabled() || !t.requires_grad()) return;
    auto node = std::make_shared<TpAllReduceBackward>(pg);
    if (t.grad_fn()) {
        node->set_next_edge(0, Edge(t.grad_fn(), t.output_nr()));
    } else {
        node->set_next_edge(0, autograd::get_grad_edge(t));
    }
    t.set_grad_fn(node);
}

// =============================================================================
// Column-Parallel Linear Backward with Compute-Communication Overlap
//
// In backward, splits the matmul gradient into two phases:
//   Phase 1: grad_input  = grad_output @ W.T  (needs TP all-reduce)
//   Phase 2: grad_weight = input.T @ grad_output (overlaps with AR)
// =============================================================================

class ColumnParallelLinearBackward : public Node {
    Tensor saved_input_;
    Tensor saved_weight_;
    Shape saved_bias_shape_;
    bool has_bias_;
    std::shared_ptr<ProcessGroupNCCL> pg_;

public:
    ColumnParallelLinearBackward(const Tensor& input, const Tensor& weight,
                                 bool has_bias, const Shape& bias_shape,
                                 std::shared_ptr<ProcessGroupNCCL> pg)
        : Node(has_bias ? 3 : 2),
          saved_input_(input), saved_weight_(weight),
          saved_bias_shape_(bias_shape), has_bias_(has_bias), pg_(pg) {}

    const char* name() const override { return "ColumnParallelLinearBackward"; }

    variable_list apply(variable_list&& grads) override {
        if (grads.empty()) return {};
        const Tensor& grad_output = grads[0];

        int64_t hidden = saved_input_.shape().dims.back();
        int64_t out_dim = grad_output.shape().dims.back();

        Tensor input_flat = saved_input_.reshape(Shape{{-1, hidden}});
        Tensor g_flat = grad_output.reshape(Shape{{-1, out_dim}});

        // Phase 1: input gradient (needs TP all-reduce)
        Tensor grad_input_flat = OwnTensor::matmul(g_flat, saved_weight_.t());
        Tensor grad_input = grad_input_flat.reshape(saved_input_.shape());

        // Phase 2: launch async all-reduce
        auto work = allreduce_launch(grad_input, pg_);

        // Phase 3: weight gradient (overlaps with AR)
        Tensor grad_weight = OwnTensor::matmul(input_flat.t(), g_flat);

        Tensor grad_bias;
        if (has_bias_) {
            grad_bias = OwnTensor::reduce_sum(grad_output, {0, 1}, false);
        }

        // Phase 4: wait for all-reduce
        allreduce_wait(work);

        if (has_bias_) {
            return {grad_bias, grad_input, grad_weight};
        } else {
            return {grad_input, grad_weight};
        }
    }

    void release_saved_variables() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
        pg_.reset();
    }
};

// =============================================================================
// Column-parallel addmm: output = bias + input @ weight
// =============================================================================
inline Tensor column_parallel_addmm(Tensor& bias, Tensor& input, Tensor& weight,
                                     std::shared_ptr<ProcessGroupNCCL>& pg) {
    Tensor result = OwnTensor::addmm(bias, input, weight, 1.0f, 1.0f);

    if (autograd::GradMode::is_enabled() &&
        (bias.requires_grad() || input.requires_grad() || weight.requires_grad())) {

        auto grad_fn = std::make_shared<ColumnParallelLinearBackward>(
            input, weight, true, bias.shape(), pg);

        if (bias.requires_grad())
            grad_fn->set_next_edge(0, autograd::get_grad_edge(bias));
        if (input.requires_grad())
            grad_fn->set_next_edge(1, autograd::get_grad_edge(input));
        if (weight.requires_grad())
            grad_fn->set_next_edge(2, autograd::get_grad_edge(weight));

        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// =============================================================================
// Column-parallel matmul: output = input @ weight
// =============================================================================
inline Tensor column_parallel_matmul(Tensor& input, Tensor& weight,
                                      std::shared_ptr<ProcessGroupNCCL>& pg) {
    Tensor result = OwnTensor::matmul(input, weight);

    if (autograd::GradMode::is_enabled() &&
        (input.requires_grad() || weight.requires_grad())) {

        auto grad_fn = std::make_shared<ColumnParallelLinearBackward>(
            input, weight, false, Shape{}, pg);

        if (input.requires_grad())
            grad_fn->set_next_edge(0, autograd::get_grad_edge(input));
        if (weight.requires_grad())
            grad_fn->set_next_edge(1, autograd::get_grad_edge(weight));

        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

}  // namespace tp
}  // namespace gpt2
