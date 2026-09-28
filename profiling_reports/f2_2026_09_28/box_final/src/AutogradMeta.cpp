#include "core/AutogradMeta.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"
#include "autograd/Hooks.h"
#include "ops/TensorOps.h" // Needed for operator+
#ifdef WITH_CUDA
#include "ops/TensorOps.cuh"          // cuda_accumulate_lowp_into_fp32
#include "ops/helpers/KernelPolicy.h" // cuda::disable_fused_grad_accum
#include "device/DeviceCore.h"        // cuda::getCurrentStream
#endif
#include <stdexcept>
#include "profiler/RecordFunction.h"

namespace OwnTensor {

// ============================================================================
// Move Semantics
// ============================================================================

AutogradMeta::AutogradMeta(AutogradMeta&& other) noexcept
    : grad_(std::move(other.grad_)),
      grad_fn_(std::move(other.grad_fn_)),
      grad_accumulator_(std::move(other.grad_accumulator_)),
      hooks_(std::move(other.hooks_)),
      post_acc_grad_hooks_(std::move(other.post_acc_grad_hooks_)),
      requires_grad_(other.requires_grad_),
      retains_grad_(other.retains_grad_),
      is_view_(other.is_view_),
      output_nr_(other.output_nr_),
      grad_dtype_(other.grad_dtype_),
      allow_grad_dtype_mismatch_(other.allow_grad_dtype_mismatch_) {
    // mutex is not movable, but that's fine - each AutogradMeta has its own
}

AutogradMeta& AutogradMeta::operator=(AutogradMeta&& other) noexcept {
    PROFILER_SCOPE("AutogradMeta::operator=");
    if (this != &other) {
        std::lock_guard<std::mutex> lock1(mutex_);
        std::lock_guard<std::mutex> lock2(other.mutex_);
        
        grad_ = std::move(other.grad_);
        grad_fn_ = std::move(other.grad_fn_);
        grad_accumulator_ = std::move(other.grad_accumulator_);
        hooks_ = std::move(other.hooks_);
        post_acc_grad_hooks_ = std::move(other.post_acc_grad_hooks_);
        requires_grad_ = other.requires_grad_;
        retains_grad_ = other.retains_grad_;
        is_view_ = other.is_view_;
        output_nr_ = other.output_nr_;
        grad_dtype_ = other.grad_dtype_;
        allow_grad_dtype_mismatch_ = other.allow_grad_dtype_mismatch_;
    }
    return *this;
}

// ============================================================================
// Interface Implementation
// ============================================================================

void AutogradMeta::set_requires_grad(bool requires_grad, TensorImpl* self_impl) {
    PROFILER_SCOPE("AutogradMeta::set_requires_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    requires_grad_ = requires_grad;
}

Tensor& AutogradMeta::mutable_grad(TensorImpl* self_impl) {
    PROFILER_SCOPE("AutogradMeta::mutable_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!grad_) {
        if (!self_impl) {
            throw std::runtime_error("AutogradMeta::mutable_grad: self_impl is null");
        }
        
        // Lazy allocation: create gradient tensor with same shape/dtype/device
        grad_ = std::make_unique<Tensor>(
            self_impl->sizes(),
            self_impl->dtype(),
            self_impl->device(),
            false  // gradient itself doesn't require grad
        );
    }
    
    return *grad_;
}

const Tensor& AutogradMeta::grad() const {
    PROFILER_SCOPE("AutogradMeta::grad");
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!grad_) {
        throw std::runtime_error("AutogradMeta::grad: gradient has not been allocated");
    }
    
    return *grad_;
}

// ============================================================================
// Additional Methods
// ============================================================================

void AutogradMeta::set_grad(const Tensor& new_grad) {
    PROFILER_SCOPE("AutogradMeta::set_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(new_grad);
}

void AutogradMeta::set_grad(Tensor&& new_grad) {
    PROFILER_SCOPE("AutogradMeta::set_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(std::move(new_grad));
}

void AutogradMeta::accumulate_grad(Tensor&& update) {
    PROFILER_SCOPE("AutogradMeta::accumulate_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    // Case-3 mixed precision: accumulate low-precision (bf16/fp16) gradients in an
    // fp32 buffer (Megatron "main_grad" style). Summing many micro-step grads in
    // bf16 loses small increments to swamping; upcasting each contribution to fp32
    // keeps the running sum full-precision. fp32 params are unaffected.
    const bool lowp = (update.dtype() == Dtype::Bfloat16 || update.dtype() == Dtype::Float16);
    if (!grad_) {
        // Gradient Layout Contract — mirrors PyTorch's AccumulateGrad Case 1.5
        // (torch/csrc/autograd/utils/grad_layout_contract.h). All parameter
        // tensors in this codebase are rowmajor contiguous, so the contract
        // reduces to "the stashed grad must also be contiguous". If a backward
        // op returned a strided view (e.g. TransposeBackward feeding wte.weight
        // under weight tying), materialize it before adopting. Without this,
        // the stride-blind multi-tensor optimizer kernels walk param[i] and
        // grad[i] as flat arrays and scramble updates by a fixed permutation
        // (wte 1/5760 ratio incident, see Resolving GPT-2 Gradient Discrepancy.md).
        //
        // TODO(Gautam_1926): this is the MINIMAL implementation similar to PyTorch's  layout-contract. The
        // following PyTorch cases are intentionally NOT implemented because the
        // corresponding codepaths do not exist in our library today. Extend if
        // any of these get added in the future:
        //   1. Per-dim stride equality check against variable's strides
        //      (PyTorch obeys_layout_contract loop). Needed if we ever introduce
        //      non-contiguous parameter formats such as channels_last conv
        //      weights — today every param is rowmajor contiguous so
        //      is_contiguous() is equivalent to per-dim stride equality.
        //   2. Sparse / sparse_csr branches (PyTorch Case 1.2, 1.3). We have no
        //      sparse parameter or sparse gradient path; add if we ever
        //      implement sparse embedding gradients for very large vocabularies.
        //   3. MKLDNN branch (PyTorch Case 1.4). CUDA-only library, no oneDNN
        //      opaque tensors.
        //   4. is_tensor_stealable refcount safety check (PyTorch variable.h:201).
        //      GradAccumulator always hands us the unique reference via
        //      std::move, no python wrappers or retained-graph hooks compete
        //      for the storage, so std::move is unconditionally safe here.
        //      Reconsider if we add Python bindings or retained-graph support.
        //   5. Double-backward branch (PyTorch Case 3, GradMode::is_enabled()).
        //      We never run gradient-of-gradient computations; the in-place +=
        //      below is fine for first-order backward. Add an out-of-place
        //      branch if higher-order autograd is ever wired in.
        //   6. CHECK_RESULT tripwire after the in-place += (PyTorch macro at
        //      accumulate_grad.h:21). Warns once per process if an in-place add
        //      somehow produces a layout-violating result. Cheap to add and
        //      useful for debugging if we ever suspect contract drift.
        // Adopt a contiguous buffer (layout contract above); for low-precision
        // params store it upcast to fp32 so subsequent micro-steps accumulate in fp32.
        Tensor u = update.is_contiguous() ? std::move(update) : update.contiguous();
        grad_ = std::make_unique<Tensor>(lowp ? u.as_type(Dtype::Float32) : std::move(u));
    } else {
        // TODO(Gautam_1926): after the Engine.cpp:309 reducer materialize the
        // `update` arriving here is always contig, so this `*grad_ += update`
        // path always hits the fast vectorized add. The IF-branch contract
        // above is now redundant for autograd-driven calls (kept as a safety
        // net for any code path that calls accumulate_grad directly bypassing
        // the engine). Audit whether any non-engine caller of accumulate_grad
        // still exists; if not, the IF-branch's .contiguous() materialize can
        // be removed in favour of an assert. See
        // Resolving_Strided_Broadcast_Inplace_Add.md section 7b for the
        // single-boundary contract rationale.
        // grad_ is fp32 for low-precision params; upcast the incoming update to
        // match so the accumulation stays in fp32.
        if (update.dtype() != grad_->dtype()) {
#ifdef WITH_CUDA
            // F2 -- fused gradient accumulation.
            // What:   cuda_accumulate_lowp_into_fp32 (TensorOpsAdd.cu) widens the bf16/fp16
            //         update to fp32 in registers and adds it straight into grad_, one kernel.
            // Was:    only the `*grad_ += update.as_type(...)` line below: a new full-size fp32
            //         temporary per bf16 weight per micro-step, 2 kernels, 18 B/element of DRAM
            //         traffic. Now 10 B/element, 1 kernel, no temporary.
            // Why:    bf16/fp16 -> fp32 is exact and the add is the same FADD.FTZ, so the bits
            //         cannot change; the temporary only carried the widened value between kernels.
            // Proven: Tests/kernel_tests/test_fused_grad_accum.cu on H200: 0 mismatches against
            //         the old line after 4 accumulation steps, bf16 and fp16, 1 to 33.5M elements,
            //         with +-0, subnormals, +-inf, NaN and overflow; 1.41-1.97x faster per call.
            // Falls through (returns false) for CPU, non-contiguous or shape-mismatched updates.
            // BLUTRAIN_NO_FUSED_GRAD_ACC=1 forces the old line (A/B runs).
            if (!cuda::disable_fused_grad_accum() &&
                cuda_accumulate_lowp_into_fp32(*grad_, update, cuda::getCurrentStream())) {
                return;
            }
#endif
            *grad_ += update.as_type(grad_->dtype());
        } else {
            *grad_ += update;
        }
    }
}

void AutogradMeta::trigger_post_acc_hooks(const Tensor& grad) {
    PROFILER_SCOPE("AutogradMeta::trigger_post_acc_hooks");
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& hook : post_acc_grad_hooks_) {
        (*hook)(grad);
    }
}

bool AutogradMeta::has_grad() const {
    PROFILER_SCOPE("AutogradMeta::has_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    return grad_ != nullptr;
}

void AutogradMeta::reset_grad() {
    PROFILER_SCOPE("AutogradMeta::reset_grad");
    std::lock_guard<std::mutex> lock(mutex_);
    grad_.reset();
}

void AutogradMeta::add_hook(std::unique_ptr<FunctionPreHook> hook) {
    PROFILER_SCOPE("AutogradMeta::add_hook");
    std::lock_guard<std::mutex> lock(mutex_);
    hooks_.push_back(std::move(hook));
}

void AutogradMeta::add_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook) {
    PROFILER_SCOPE("AutogradMeta::add_post_acc_hook");
    std::lock_guard<std::mutex> lock(mutex_);
    post_acc_grad_hooks_.push_back(std::move(hook));
}

void AutogradMeta::clear_hooks() {
    PROFILER_SCOPE("AutogradMeta::clear_hooks");
    std::lock_guard<std::mutex> lock(mutex_);
    hooks_.clear();
    post_acc_grad_hooks_.clear();
}

} // namespace OwnTensor



// #include "core/AutogradMeta.h"
// #include "core/Tensor.h"
// #include "core/TensorImpl.h"
// #include "autograd/Hooks.h"
// #include "ops/TensorOps.h" // Needed for operator+
// #include <stdexcept>

// namespace OwnTensor {

// // ============================================================================
// // Move Semantics
// // ============================================================================

// AutogradMeta::AutogradMeta(AutogradMeta&& other) noexcept
//     : grad_(std::move(other.grad_)),
//       grad_fn_(std::move(other.grad_fn_)),
//       grad_accumulator_(std::move(other.grad_accumulator_)),
//       hooks_(std::move(other.hooks_)),
//       post_acc_grad_hooks_(std::move(other.post_acc_grad_hooks_)),
//       requires_grad_(other.requires_grad_),
//       retains_grad_(other.retains_grad_),
//       is_view_(other.is_view_),
//       output_nr_(other.output_nr_),
//       grad_dtype_(other.grad_dtype_),
//       allow_grad_dtype_mismatch_(other.allow_grad_dtype_mismatch_) {
//     // mutex is not movable, but that's fine - each AutogradMeta has its own
// }

// AutogradMeta& AutogradMeta::operator=(AutogradMeta&& other) noexcept {
//     if (this != &other) {
//         std::lock_guard<std::mutex> lock1(mutex_);
//         std::lock_guard<std::mutex> lock2(other.mutex_);
        
//         grad_ = std::move(other.grad_);
//         grad_fn_ = std::move(other.grad_fn_);
//         grad_accumulator_ = std::move(other.grad_accumulator_);
//         hooks_ = std::move(other.hooks_);
//         post_acc_grad_hooks_ = std::move(other.post_acc_grad_hooks_);
//         requires_grad_ = other.requires_grad_;
//         retains_grad_ = other.retains_grad_;
//         is_view_ = other.is_view_;
//         output_nr_ = other.output_nr_;
//         grad_dtype_ = other.grad_dtype_;
//         allow_grad_dtype_mismatch_ = other.allow_grad_dtype_mismatch_;
//     }
//     return *this;
// }

// // ============================================================================
// // Interface Implementation
// // ============================================================================

// void AutogradMeta::set_requires_grad(bool requires_grad, TensorImpl* self_impl) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     requires_grad_ = requires_grad;
// }

// Tensor& AutogradMeta::mutable_grad(TensorImpl* self_impl) {
//     std::lock_guard<std::mutex> lock(mutex_);
    
//     if (!grad_) {
//         if (!self_impl) {
//             throw std::runtime_error("AutogradMeta::mutable_grad: self_impl is null");
//         }
        
//         // Lazy allocation: create gradient tensor with same shape/dtype/device
//         grad_ = std::make_unique<Tensor>(
//             self_impl->sizes(),
//             self_impl->dtype(),
//             self_impl->device(),
//             false  // gradient itself doesn't require grad
//         );
//     }
    
//     return *grad_;
// }

// const Tensor& AutogradMeta::grad() const {
//     std::lock_guard<std::mutex> lock(mutex_);
    
//     if (!grad_) {
//         throw std::runtime_error("AutogradMeta::grad: gradient has not been allocated");
//     }
    
//     return *grad_;
// }

// // ============================================================================
// // Additional Methods
// // ============================================================================

// void AutogradMeta::set_grad(const Tensor& new_grad) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     grad_ = std::make_unique<Tensor>(new_grad);
// }

// void AutogradMeta::set_grad(Tensor&& new_grad) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     grad_ = std::make_unique<Tensor>(std::move(new_grad));
// }

// void AutogradMeta::accumulate_grad(Tensor&& update) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     if (!grad_) {
//         // Gradient Layout Contract — mirrors PyTorch's AccumulateGrad Case 1.5
//         // (torch/csrc/autograd/utils/grad_layout_contract.h). All parameter
//         // tensors in this codebase are rowmajor contiguous, so the contract
//         // reduces to "the stashed grad must also be contiguous". If a backward
//         // op returned a strided view (e.g. TransposeBackward feeding wte.weight
//         // under weight tying), materialize it before adopting. Without this,
//         // the stride-blind multi-tensor optimizer kernels walk param[i] and
//         // grad[i] as flat arrays and scramble updates by a fixed permutation
//         // (wte 1/5760 ratio incident, see Resolving GPT-2 Gradient Discrepancy.md).
//         //
//         // TODO(Gautam_1926): this is the MINIMAL port of PyTorch's contract. The
//         // following PyTorch cases are intentionally NOT implemented because the
//         // corresponding codepaths do not exist in our library today. Extend if
//         // any of these get added in the future:
//         //   1. Per-dim stride equality check against variable's strides
//         //      (PyTorch obeys_layout_contract loop). Needed if we ever introduce
//         //      non-contiguous parameter formats such as channels_last conv
//         //      weights — today every param is rowmajor contiguous so
//         //      is_contiguous() is equivalent to per-dim stride equality.
//         //   2. Sparse / sparse_csr branches (PyTorch Case 1.2, 1.3). We have no
//         //      sparse parameter or sparse gradient path; add if we ever
//         //      implement sparse embedding gradients for very large vocabularies.
//         //   3. MKLDNN branch (PyTorch Case 1.4). CUDA-only library, no oneDNN
//         //      opaque tensors.
//         //   4. is_tensor_stealable refcount safety check (PyTorch variable.h:201).
//         //      GradAccumulator always hands us the unique reference via
//         //      std::move, no python wrappers or retained-graph hooks compete
//         //      for the storage, so std::move is unconditionally safe here.
//         //      Reconsider if we add Python bindings or retained-graph support.
//         //   5. Double-backward branch (PyTorch Case 3, GradMode::is_enabled()).
//         //      We never run gradient-of-gradient computations; the in-place +=
//         //      below is fine for first-order backward. Add an out-of-place
//         //      branch if higher-order autograd is ever wired in.
//         //   6. CHECK_RESULT tripwire after the in-place += (PyTorch macro at
//         //      accumulate_grad.h:21). Warns once per process if an in-place add
//         //      somehow produces a layout-violating result. Cheap to add and
//         //      useful for debugging if we ever suspect contract drift.
//         if (update.is_contiguous()) {
//             grad_ = std::make_unique<Tensor>(std::move(update));
//         } else {
//             grad_ = std::make_unique<Tensor>(update.contiguous());
//         }
//     } else {
//         *grad_ += update;
//     }
// }

// void AutogradMeta::trigger_post_acc_hooks(const Tensor& grad) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     for (auto& hook : post_acc_grad_hooks_) {
//         (*hook)(grad);
//     }
// }

// bool AutogradMeta::has_grad() const {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return grad_ != nullptr;
// }

// void AutogradMeta::reset_grad() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     grad_.reset();
// }

// void AutogradMeta::add_hook(std::unique_ptr<FunctionPreHook> hook) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     hooks_.push_back(std::move(hook));
// }

// void AutogradMeta::add_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     post_acc_grad_hooks_.push_back(std::move(hook));
// }

// void AutogradMeta::clear_hooks() {
//     std::lock_guard<std::mutex> lock(mutex_);
//     hooks_.clear();
//     post_acc_grad_hooks_.clear();
// }

// } // namespace OwnTensor