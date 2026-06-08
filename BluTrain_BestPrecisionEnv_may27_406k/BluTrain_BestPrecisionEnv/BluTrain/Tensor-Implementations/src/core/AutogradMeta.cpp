#include "core/AutogradMeta.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"
#include "autograd/Hooks.h"
#include "ops/TensorOps.h" // Needed for operator+
#include <stdexcept>

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
    std::lock_guard<std::mutex> lock(mutex_);
    requires_grad_ = requires_grad;
}

Tensor& AutogradMeta::mutable_grad(TensorImpl* self_impl) {
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
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(new_grad);
}

void AutogradMeta::set_grad(Tensor&& new_grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(std::move(new_grad));
}

void AutogradMeta::accumulate_grad(Tensor&& update) {
    std::lock_guard<std::mutex> lock(mutex_);
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
        if (update.is_contiguous()) {
            grad_ = std::make_unique<Tensor>(std::move(update));
        } else {
            grad_ = std::make_unique<Tensor>(update.contiguous());
        }
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
        *grad_ += update;
    }
}

void AutogradMeta::trigger_post_acc_hooks(const Tensor& grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& hook : post_acc_grad_hooks_) {
        (*hook)(grad);
    }
}

bool AutogradMeta::has_grad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return grad_ != nullptr;
}

void AutogradMeta::reset_grad() {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_.reset();
}

void AutogradMeta::add_hook(std::unique_ptr<FunctionPreHook> hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    hooks_.push_back(std::move(hook));
}

void AutogradMeta::add_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    post_acc_grad_hooks_.push_back(std::move(hook));
}

void AutogradMeta::clear_hooks() {
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