#include "autograd/backward/LossBackward.h"
#include "utils/Profiler.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/helpers/ConditionalOps.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <stdexcept>

#ifdef WITH_CUDA
#include "ops/helpers/LossKernels.h"
#include "profiler/RecordFunction.h"
#endif

namespace OwnTensor {
namespace autograd {

// ============================================================================
// MSELossBackward
// ============================================================================

MSELossBackward::MSELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> MSELossBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("MSELossBackward::apply");
    if (grads.empty()) {
        throw std::runtime_error("MSELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    Tensor grad_pred;
    if (grad_output.device().is_cuda() && grad_output.dtype() == Dtype::Float32) {
         grad_pred = Tensor(saved_pred_.shape(), grad_output.opts());
         OwnTensor::device::set_cuda_device(grad_output.device().index);
         cuda::mse_loss_backward_cuda(grad_output.data<float>(), saved_pred_.data<float>(), saved_target_.data<float>(), grad_pred.data<float>(), numel_);
    } else {
         float scale = 2.0f / static_cast<float>(numel_);
         Tensor diff = saved_pred_ - saved_target_;
         grad_pred = diff * scale * grad_output;
    }
    
    return {grad_pred};
}

void MSELossBackward::release_saved_variables() {
    PROFILER_SCOPE("MSELossBackward::release_saved_variables");
    saved_pred_ = Tensor();
    saved_target_ = Tensor();
}

// ============================================================================
// MAELossBackward
// ============================================================================

MAELossBackward::MAELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> MAELossBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("MAELossBackward::apply");
    if (grads.empty()) {
        throw std::runtime_error("MAELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    Tensor grad_pred;
    if (grad_output.device().is_cuda() && grad_output.dtype() == Dtype::Float32) {
         grad_pred = Tensor(saved_pred_.shape(), grad_output.opts());
         OwnTensor::device::set_cuda_device(grad_output.device().index);
         cuda::mae_loss_backward_cuda(grad_output.data<float>(), saved_pred_.data<float>(), saved_target_.data<float>(), grad_pred.data<float>(), numel_);
    } else {
         float scale = 1.0f / static_cast<float>(numel_);

         Tensor diff = saved_pred_ - saved_target_;
         Tensor zero = Tensor::zeros(diff.shape(),
             TensorOptions().with_dtype(diff.dtype()).with_device(diff.device()));
         Tensor ones = Tensor::ones(diff.shape(),
             TensorOptions().with_dtype(diff.dtype()).with_device(diff.device()));
         Tensor neg_ones = ones * -1.0f;

         Tensor sign_diff = where(diff > zero, ones, where(diff < zero, neg_ones, zero));
         grad_pred = sign_diff * scale * grad_output;
    }
    
    return {grad_pred};
}

void MAELossBackward::release_saved_variables() {
    PROFILER_SCOPE("MAELossBackward::release_saved_variables");
    saved_pred_ = Tensor();
    saved_target_ = Tensor();
}

// ============================================================================
// BCELossBackward
// ============================================================================

BCELossBackward::BCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> BCELossBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("BCELossBackward::apply");
    if (grads.empty()) {
        throw std::runtime_error("BCELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // BCE: L = -mean(target * log(pred) + (1-target) * log(1-pred))
    // grad_pred = (-target/pred + (1-target)/(1-pred)) / numel
    Tensor grad_pred;
    if (grad_output.device().is_cuda() && grad_output.dtype() == Dtype::Float32) {
         grad_pred = Tensor(saved_pred_.shape(), grad_output.opts());
         OwnTensor::device::set_cuda_device(grad_output.device().index);
         cuda::bce_loss_backward_cuda(grad_output.data<float>(), saved_pred_.data<float>(), saved_target_.data<float>(), grad_pred.data<float>(), numel_);
    } else {
         float scale = 1.0f / static_cast<float>(numel_);

         Tensor ones = Tensor::ones(saved_pred_.shape(),
             TensorOptions().with_dtype(saved_pred_.dtype()).with_device(saved_pred_.device()));

         Tensor term1 = saved_target_ / saved_pred_ * -1.0f;
         Tensor term2 = (ones - saved_target_) / (ones - saved_pred_);
         grad_pred = (term1 + term2) * scale * grad_output;
    }
    
    return {grad_pred};
}

void BCELossBackward::release_saved_variables() {
    PROFILER_SCOPE("BCELossBackward::release_saved_variables");
    saved_pred_ = Tensor();
    saved_target_ = Tensor();
}

// ============================================================================
// CCELossBackward
// ============================================================================

CCELossBackward::CCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> CCELossBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("CCELossBackward::apply");
    if (grads.empty()) {
        throw std::runtime_error("CCELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // CCE: L = -mean(sum(target * log(pred), dim=1))
    // grad_pred = -target / pred / numel
    if (saved_pred_.device().is_cuda() && saved_pred_.dtype() == Dtype::Float32) {
         Tensor grad_pred = Tensor::zeros(saved_pred_.shape(), saved_pred_.opts());
         int64_t num_classes = saved_pred_.shape().dims.back();
         int64_t batch_size = saved_pred_.numel() / num_classes;

         const Tensor& g_out = (grad_output.device().is_cpu()) ? grad_output.to(saved_pred_.device()) : grad_output;

         OwnTensor::device::set_cuda_device(saved_pred_.device().index);
         cuda::categorical_cross_entropy_backward_cuda(
             g_out.data<float>(),
             saved_pred_.data<float>(),
             saved_target_.data<float>(),
             grad_pred.data<float>(),
             batch_size, num_classes
         );
         return {grad_pred};
    }

    float scale = 1.0f / static_cast<float>(numel_);
    Tensor grad_pred = saved_target_ / saved_pred_ * -1.0f;
    grad_pred = grad_pred * scale * grad_output;
    
    return {grad_pred};
}

void CCELossBackward::release_saved_variables() {
    PROFILER_SCOPE("CCELossBackward::release_saved_variables");
    saved_pred_ = Tensor();
    saved_target_ = Tensor();
}

// ============================================================================
// SparseCrossEntropyLossBackward
// ============================================================================

SparseCrossEntropyLossBackward::SparseCrossEntropyLossBackward(
    const Tensor& logits, const Tensor& targets, int64_t batch_size, int64_t num_classes,
    int64_t ignore_index, const Tensor& denom)
    : Node(1), saved_logits_(logits), saved_targets_(targets),
      batch_size_(batch_size), num_classes_(num_classes),
      ignore_index_(ignore_index), denom_(denom) {}

// Stats-aware ctor — backward will skip the Reduce kernel.
SparseCrossEntropyLossBackward::SparseCrossEntropyLossBackward(
    const Tensor& logits, const Tensor& targets,
    const Tensor& saved_max, const Tensor& saved_sum,
    int64_t batch_size, int64_t num_classes, int64_t ignore_index,
    const Tensor& denom)
    : Node(1), saved_logits_(logits), saved_targets_(targets),
      saved_max_(saved_max), saved_sum_(saved_sum),
      batch_size_(batch_size), num_classes_(num_classes),
      ignore_index_(ignore_index), denom_(denom) {}

std::vector<Tensor> SparseCrossEntropyLossBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("SparseCrossEntropyLossBackward::apply");
    if (grads.empty()) {
        throw std::runtime_error("SparseCrossEntropyLossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // The denominator the FORWARD divided by. The CUDA kernels take it as a
    // DEVICE pointer so neither pass has to sync to agree on it; the CPU path
    // reads the single element directly. When the node carries no denominator
    // (pretraining, or a node built before the SFT path existed) we fall back to
    // batch_size_, which is exactly the old behaviour.
    Tensor denom_tensor = denom_;
    if (!denom_tensor.is_valid()) {
        denom_tensor = Tensor::full(
            Shape{{1}},
            TensorOptions().with_dtype(Dtype::Float32).with_device(saved_logits_.device()),
            static_cast<float>(batch_size_));
    }
    // CPU path ONLY. Deliberately not initialised from denom_tensor here: on the
    // CUDA path that read would be a device->host copy on every backward, which
    // is the exact sync the device-pointer design exists to avoid. It is filled
    // in inside the is_cpu() branch below, before the lambda that uses it runs.
    float host_scale = 0.0f;
    
    // Handle both 2D [N, C] and 3D [B, T, C] logits
    // We need to flatten to 2D for computation, then reshape back
    auto logits_shape = saved_logits_.shape().dims;
    Shape original_shape = saved_logits_.shape();
    
    Tensor logits_2d = saved_logits_;
    if (logits_shape.size() == 3) {
        // Flatten [B, T, C] -> [B*T, C]
        logits_2d = saved_logits_.view(Shape{{batch_size_, num_classes_}});
    }
    
    // Compute softmax and gradient on 2D tensor
    TensorOptions opts = TensorOptions()
        .with_dtype(saved_logits_.dtype())
        .with_device(saved_logits_.device());
    
    // Allocated by the branch that fills it. 2026-10-17: until then it was
    // allocated here AND again in the CUDA branch, so for a moment the CUDA
    // backward held the logits plus TWO gradient-sized tensors (3.1 GiB each at
    // 8 x 4096 x 50,304 bf16), one of them never used.
    Tensor grad_logits_2d;

    // CPU implementation (templated on logits dtype, always computes in float)
    auto cpu_sparse_ce_backward = [&](auto* logits_data, auto* grad_data) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(logits_data)>>;
        // grad_output is a scalar — read as float regardless of dtype
        float grad_val = static_cast<float>(*grad_output.data<T>());

        for (int64_t i = 0; i < batch_size_; ++i) {
            float max_val = static_cast<float>(logits_data[i * num_classes_]);
            for (int64_t c = 1; c < num_classes_; ++c) {
                max_val = std::max(max_val, static_cast<float>(logits_data[i * num_classes_ + c]));
            }

            float sum_exp = 0.0f;
            std::vector<float> exp_vals(num_classes_);
            for (int64_t c = 0; c < num_classes_; ++c) {
                exp_vals[c] = std::exp(static_cast<float>(logits_data[i * num_classes_ + c]) - max_val);
                sum_exp += exp_vals[c];
            }

            int64_t target_class = 0;
            if (saved_targets_.dtype() == Dtype::Int64) {
                target_class = saved_targets_.data<int64_t>()[i];
            } else if (saved_targets_.dtype() == Dtype::Int32) {
                target_class = static_cast<int64_t>(saved_targets_.data<int32_t>()[i]);
            } else if (saved_targets_.dtype() == Dtype::UInt16) {
                target_class = static_cast<int64_t>(saved_targets_.data<uint16_t>()[i]);
            }

            // SFT label masking — must match the CUDA kernels and the forward
            // pass, otherwise CPU and GPU disagree and the model still learns
            // from the prompt tokens. The row is ZEROED, not left untouched:
            // grad_logits_2d comes from Tensor::empty and holds garbage.
            if (target_class == ignore_index_) {
                for (int64_t c = 0; c < num_classes_; ++c) {
                    grad_data[i * num_classes_ + c] = static_cast<T>(0.0f);
                }
                continue;
            }

            float total_scale = grad_val * host_scale;
            for (int64_t c = 0; c < num_classes_; ++c) {
                float softmax_val = exp_vals[c] / sum_exp;
                float one_hot = (c == target_class) ? 1.0f : 0.0f;
                grad_data[i * num_classes_ + c] = static_cast<T>((softmax_val - one_hot) * total_scale);
            }
        }
    };

    if (logits_2d.device().is_cpu()) {
        // denom_tensor lives on logits' device, so on this path it is already a
        // CPU tensor and to_cpu() is a no-op.
        host_scale = 1.0f / denom_tensor.to_cpu().data<float>()[0];
        grad_logits_2d = Tensor::empty(logits_2d.shape(), opts);
        if (logits_2d.dtype() == Dtype::Float32) {
            cpu_sparse_ce_backward(logits_2d.data<float>(), grad_logits_2d.data<float>());
        } else if (logits_2d.dtype() == Dtype::Float16) {
            cpu_sparse_ce_backward(logits_2d.data<float16_t>(), grad_logits_2d.data<float16_t>());
        } else if (logits_2d.dtype() == Dtype::Bfloat16) {
            cpu_sparse_ce_backward(logits_2d.data<bfloat16_t>(), grad_logits_2d.data<bfloat16_t>());
        }
    } else {
        #ifdef WITH_CUDA
        // Ensure all inputs are on the same device as logits_2d
        Tensor targets_cuda = saved_targets_;
        if (targets_cuda.device() != logits_2d.device()) {
            targets_cuda = targets_cuda.to(logits_2d.device());
        }
        
        Tensor grad_output_cuda = grad_output;
        if (grad_output_cuda.device() != logits_2d.device()) {
            grad_output_cuda = grad_output_cuda.to(logits_2d.device());
        }
        // grad_output is fp32 BY CONTRACT (see the note in LossKernels.h). The
        // loss scalar this op produces is always fp32, so autograd's gradient for
        // it is fp32 too -- but a caller can hand backward() its own tensor
        // (the trainer passes grad_scale), so convert rather than reinterpret.
        if (grad_output_cuda.dtype() != Dtype::Float32) {
            grad_output_cuda = grad_output_cuda.as_type(Dtype::Float32);
        }

        // Set device context
        device::set_cuda_device(logits_2d.device().index);
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

        AUTO_PROFILE_CUDA("Backward::SparseCrossEntropy_CUDA");

        // Fast path: forward saved (max, sum) → skip Reduce kernel.
        const bool stats_path = saved_max_.is_valid() && saved_sum_.is_valid();
        // Stats path (= training): the gradient overwrites the saved logits -- the
        // returned tensor shares their storage, no new [rows x vocab] buffer
        // (2026-10-17; the only behaviour since 2026-10-21, see LossOps.h).
        grad_logits_2d = stats_path ? logits_2d : Tensor::empty(logits_2d.shape(), opts);

        // Dispatch by logits dtype x target dtype
        // grad_out_ptr is `const float*` for every logits dtype -- it is NOT
        // derived from LogitT. Inferring it from the logits is what used to make
        // an fp32 grad_scale get read as bf16 with no error.
        auto dispatch_targets = [&](auto* logits_ptr, auto* grad_ptr, const float* grad_out_ptr) {
            using LogitT = std::remove_const_t<std::remove_pointer_t<decltype(logits_ptr)>>;
            if (stats_path) {   // grad_ptr == logits_ptr: the kernel reads each logit, then writes its gradient there
                const float* smax = saved_max_.data<float>();
                const float* ssum = saved_sum_.data<float>();
                if (targets_cuda.dtype() == Dtype::UInt16) {
                    cuda::sparse_ce_backward_cuda_vec_with_stats_inplace<LogitT, uint16_t>(
                        grad_ptr, targets_cuda.data<uint16_t>(), smax, ssum, batch_size_,
                        num_classes_, grad_out_ptr, denom_tensor.data<float>(), stream, ignore_index_);
                } else if (targets_cuda.dtype() == Dtype::Int64) {
                    cuda::sparse_ce_backward_cuda_vec_with_stats_inplace<LogitT, int64_t>(
                        grad_ptr, targets_cuda.data<int64_t>(), smax, ssum, batch_size_,
                        num_classes_, grad_out_ptr, denom_tensor.data<float>(), stream, ignore_index_);
                } else if (targets_cuda.dtype() == Dtype::Int32) {
                    cuda::sparse_ce_backward_cuda_vec_with_stats_inplace<LogitT, int32_t>(
                        grad_ptr, targets_cuda.data<int32_t>(), smax, ssum, batch_size_,
                        num_classes_, grad_out_ptr, denom_tensor.data<float>(), stream, ignore_index_);
                } else {
                    throw std::runtime_error("SparseCrossEntropyLossBackward: unsupported target dtype for CUDA (stats path)");
                }
                return;
            }
            // Original (no-stats) path — falls back when forward didn't save (max, sum).
            if (targets_cuda.dtype() == Dtype::UInt16) {
                cuda::sparse_cross_entropy_backward_cuda(logits_ptr, targets_cuda.data<uint16_t>(), grad_ptr, batch_size_, num_classes_, grad_out_ptr, denom_tensor.data<float>(), stream, ignore_index_);
            } else if (targets_cuda.dtype() == Dtype::Int64) {
                cuda::sparse_cross_entropy_backward_cuda(logits_ptr, targets_cuda.data<int64_t>(), grad_ptr, batch_size_, num_classes_, grad_out_ptr, denom_tensor.data<float>(), stream, ignore_index_);
            } else if (targets_cuda.dtype() == Dtype::Int32) {
                cuda::sparse_cross_entropy_backward_cuda(logits_ptr, targets_cuda.data<int32_t>(), grad_ptr, batch_size_, num_classes_, grad_out_ptr, denom_tensor.data<float>(), stream, ignore_index_);
            } else {
                throw std::runtime_error("SparseCrossEntropyLossBackward: unsupported target dtype for CUDA");
            }
        };

        if (logits_2d.dtype() == Dtype::Float32) {
            dispatch_targets(logits_2d.data<float>(), grad_logits_2d.data<float>(), grad_output_cuda.data<float>());
        } else if (logits_2d.dtype() == Dtype::Float16) {
            dispatch_targets(logits_2d.data<float16_t>(), grad_logits_2d.data<float16_t>(), grad_output_cuda.data<float>());
        } else if (logits_2d.dtype() == Dtype::Bfloat16) {
            dispatch_targets(logits_2d.data<bfloat16_t>(), grad_logits_2d.data<bfloat16_t>(), grad_output_cuda.data<float>());
        } else {
            throw std::runtime_error("SparseCrossEntropyLossBackward: unsupported logits dtype for CUDA");
        }
        
        // No sync needed - kernel will complete before next operation uses grad_logits_2d
        #else
        throw std::runtime_error("CUDA not available but tensor is on CUDA device");
        #endif
    }
    
    // Reshape gradient back to original shape if needed
    Tensor grad_logits = grad_logits_2d;
    if (logits_shape.size() == 3) {
        grad_logits = grad_logits_2d.view(original_shape);
    }
    
    return {grad_logits};
}

} // namespace autograd
} // namespace OwnTensor