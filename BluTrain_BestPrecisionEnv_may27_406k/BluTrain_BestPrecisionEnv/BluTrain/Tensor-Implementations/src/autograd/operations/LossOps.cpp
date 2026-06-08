#include "autograd/operations/LossOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/LossBackward.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/ScalarOps.h"
#include "device/DeviceCore.h"
#include "device/AllocationTracker.h"


#ifdef WITH_CUDA
#include "ops/helpers/LossKernels.h"
#endif

#include "utils/Profiler.h"
#include "checkpointing/GradMode.h"
#include <iostream>

namespace OwnTensor {
namespace autograd {

Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    GraphRecordMode::record_forward("LOSS: mse_loss");
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         {
             AUTO_PROFILE_CUDA("Forward::MSE_Loss_CUDA");
             device::set_cuda_device(predictions.device().index);
             cuda::mse_loss_forward_cuda(predictions.data<float>(), targets.data<float>(), result.data<float>(), predictions.numel());
         }
         
         if (GradMode::is_enabled() && predictions.requires_grad()) {
             auto grad_fn = std::make_shared<MSELossBackward>(predictions, targets, predictions.numel());
             grad_fn->set_next_edge(0, get_grad_edge(predictions));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
         return result;
    }

    // Forward: mean((pred - target)^2)
    Tensor diff = predictions - targets;
    Tensor sq_diff = OwnTensor::pow(diff, 2, 0);
    Tensor result = reduce_mean(sq_diff);
    
    // Build graph if predictions require grad
    if (GradMode::is_enabled() && predictions.requires_grad()) {
        auto grad_fn = std::make_shared<MSELossBackward>(predictions, targets, predictions.numel());
        grad_fn->set_next_edge(0, get_grad_edge(predictions));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor mae_loss(const Tensor& predictions, const Tensor& targets) {
    GraphRecordMode::record_forward("LOSS: mae_loss");
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         OwnTensor::device::set_cuda_device(predictions.device().index);
         cuda::mae_loss_forward_cuda(predictions.data<float>(), targets.data<float>(), result.data<float>(), predictions.numel());
         
         if (GradMode::is_enabled() && predictions.requires_grad()) {
             auto grad_fn = std::make_shared<MAELossBackward>(predictions, targets, predictions.numel());
             grad_fn->set_next_edge(0, get_grad_edge(predictions));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
         return result;
    }

    // Forward: mean(|pred - target|)
    Tensor diff = predictions - targets;
    Tensor abs_diff = OwnTensor::abs(diff, 0);
    Tensor result = reduce_mean(abs_diff);
    
    // Build graph if predictions require grad
    if (GradMode::is_enabled() && predictions.requires_grad()) {
        auto grad_fn = std::make_shared<MAELossBackward>(predictions, targets, predictions.numel());
        grad_fn->set_next_edge(0, get_grad_edge(predictions));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    GraphRecordMode::record_forward("LOSS: binary_cross_entropy");
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         OwnTensor::device::set_cuda_device(predictions.device().index);
         cuda::bce_loss_forward_cuda(predictions.data<float>(), targets.data<float>(), result.data<float>(), predictions.numel());
         
         if (GradMode::is_enabled() && predictions.requires_grad()) {
             auto grad_fn = std::make_shared<BCELossBackward>(predictions, targets, predictions.numel());
             grad_fn->set_next_edge(0, get_grad_edge(predictions));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
         return result;
    }

    float epsilon_val = 1e-7f;
    Tensor epsilon = Tensor::full(predictions.shape(),
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
    Tensor one_minus_epsilon = Tensor::full(predictions.shape(),
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

    // Clip predictions
    Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
    clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

    Tensor term1 = targets * OwnTensor::log(clipped_preds);
    Tensor ones = Tensor::ones(targets.shape(),
        TensorOptions().with_device(targets.device()).with_dtype(targets.dtype()));
    Tensor term2 = (ones - targets) * OwnTensor::log(ones - clipped_preds);
    Tensor sum_terms = term1 + term2;
    Tensor neg_one = Tensor::full({{1}},
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
    Tensor result = reduce_mean(sum_terms) * neg_one;

    // Build graph if predictions require grad
    if (GradMode::is_enabled() && predictions.requires_grad()) {
        auto grad_fn = std::make_shared<BCELossBackward>(clipped_preds, targets, predictions.numel());
        grad_fn->set_next_edge(0, get_grad_edge(predictions));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    GraphRecordMode::record_forward("LOSS: categorical_cross_entropy");
    if (predictions.device().is_cuda() && predictions.dtype() == Dtype::Float32) {
         Tensor result = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()));
         
         int64_t num_classes = predictions.shape().dims.back();
         int64_t batch_size = predictions.numel() / num_classes;
         OwnTensor::device::set_cuda_device(predictions.device().index);
         cuda::categorical_cross_entropy_forward_cuda(
             predictions.data<float>(),
             targets.data<float>(),
             result.data<float>(),
             batch_size, num_classes
         );
         
         // Build graph if predictions require grad
         if (GradMode::is_enabled() && predictions.requires_grad()) {
             auto grad_fn = std::make_shared<CCELossBackward>(predictions, targets, predictions.numel());
             grad_fn->set_next_edge(0, get_grad_edge(predictions));
             result.set_grad_fn(grad_fn);
             result.set_requires_grad(true);
         }
         if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
         return result;
    }

    float epsilon_val = 1e-7f;
    Tensor epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
    Tensor one_minus_epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

    // Clip predictions
    Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
    clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

    Tensor log_preds = OwnTensor::log(clipped_preds);
    Tensor target_log_probs = targets * log_preds;
    
    std::vector<int64_t> axis = {1};
    Tensor sample_losses = reduce_sum(target_log_probs, axis);
    Tensor neg_one = Tensor::full({{1}}, 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
    Tensor result = reduce_mean(sample_losses) * neg_one;
    
    // Build graph if predictions require grad
    if (GradMode::is_enabled() && predictions.requires_grad()) {
        auto grad_fn = std::make_shared<CCELossBackward>(clipped_preds, targets, predictions.numel());
        grad_fn->set_next_edge(0, get_grad_edge(predictions));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor sparse_cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    
    GraphRecordMode::record_forward("LOSS: sparse_cross_entropy");
    // Handle both 2D [N, C] and 3D [B, T, C] logits
    auto logits_shape = logits.shape().dims;
    int64_t batch_size, num_classes;
    TRACK_ALLOC_SCOPE("L201:autograd::sparse_cross_entropy_loss");
    Tensor logits_2d = logits;
    Tensor targets_1d = targets;
    
    if (logits_shape.size() == 3) {
        // 3D logits [B, T, C] -> flatten to [B*T, C]
        int64_t B = logits_shape[0];
        int64_t T = logits_shape[1];
        num_classes = logits_shape[2];
        batch_size = B * T;
        logits_2d = logits.view(Shape{{batch_size, num_classes}});
        
        // Flatten targets [B, T] -> [B*T] if needed
        auto targets_shape = targets.shape().dims;
        if (targets_shape.size() == 2) {
            targets_1d = targets.view(Shape{{batch_size}});
        }
    } else if (logits_shape.size() == 2) {
        batch_size = logits_shape[0];
        num_classes = logits_shape[1];
    } else {
        throw std::runtime_error("sparse_cross_entropy_loss: logits must be 2D [N, C] or 3D [B, T, C]");
    }
    
    auto targets_shape = targets_1d.shape().dims;
    if (targets_shape.size() != 1 || targets_shape[0] != batch_size) {
        throw std::runtime_error("sparse_cross_entropy_loss: targets shape mismatch");
    }
    
    // Compute softmax and cross entropy loss
    // For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    TensorOptions opts = TensorOptions()
        .with_dtype(logits.dtype())
        .with_device(logits.device());
    
    float total_loss = 0.0f;
    
    // CPU path — templated on logits dtype, always computes in float
    auto cpu_sparse_ce_forward = [&](auto* logits_data) {
        for (int64_t i = 0; i < batch_size; ++i) {
            float max_val = static_cast<float>(logits_data[i * num_classes]);
            for (int64_t c = 1; c < num_classes; ++c) {
                max_val = std::max(max_val, static_cast<float>(logits_data[i * num_classes + c]));
            }

            float sum_exp = 0.0f;
            for (int64_t c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(static_cast<float>(logits_data[i * num_classes + c]) - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);

            int64_t target_class = 0;
            if (targets_1d.dtype() == Dtype::Int64) {
                target_class = targets_1d.data<int64_t>()[i];
            } else if (targets_1d.dtype() == Dtype::Int32) {
                target_class = static_cast<int64_t>(targets_1d.data<int32_t>()[i]);
            } else if (targets_1d.dtype() == Dtype::UInt16) {
                target_class = static_cast<int64_t>(targets_1d.data<uint16_t>()[i]);
            }

            float loss_i = log_sum_exp - static_cast<float>(logits_data[i * num_classes + target_class]);
            total_loss += loss_i;
        }
    };

    if (logits_2d.device().is_cpu()) {
        if (logits_2d.dtype() == Dtype::Float16) {
            cpu_sparse_ce_forward(logits_2d.data<float16_t>());
        } else if (logits_2d.dtype() == Dtype::Bfloat16) {
            cpu_sparse_ce_forward(logits_2d.data<bfloat16_t>());
        } else {
            cpu_sparse_ce_forward(logits_2d.data<float>());
        }
    } else {
        // CUDA: Use optimized CUDA kernel
        #ifdef WITH_CUDA
        
        // Ensure targets are on the same device as logits
        if (targets_1d.device() != logits_2d.device()) {
             targets_1d = targets_1d.to(logits_2d.device());
        }

        // Set device context BEFORE any CUDA operations
        OwnTensor::device::set_cuda_device(logits_2d.device().index);

        // Keep result on GPU to avoid sync
        Tensor result = Tensor::zeros(Shape{{1}}, opts);

        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

        // Allocate (max, sum) scratch — only when grad will be computed; saved
        // into the autograd node so backward can skip the Reduce kernel.
        const bool will_save_stats = (GradMode::is_enabled() && logits.requires_grad());
        TensorOptions stats_opts = TensorOptions().with_dtype(Dtype::Float32).with_device(logits.device());
        Tensor saved_max, saved_sum;
        if (will_save_stats) {
            saved_max = Tensor::empty(Shape{{batch_size}}, stats_opts);
            saved_sum = Tensor::empty(Shape{{batch_size}}, stats_opts);
        }

        auto dispatch_targets = [&](auto* logits_ptr, auto* result_ptr) {
            using LogitT = std::remove_const_t<std::remove_pointer_t<decltype(logits_ptr)>>;
            float* smax_ptr = will_save_stats ? saved_max.data<float>() : nullptr;
            float* ssum_ptr = will_save_stats ? saved_sum.data<float>() : nullptr;

            auto fwd_save_stats = [&](auto* tgt_ptr) {
                cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<LogitT>(
                    logits_ptr, tgt_ptr, result_ptr,
                    smax_ptr, ssum_ptr, batch_size, num_classes, stream);
            };
            auto fwd_no_stats = [&](auto* tgt_ptr) {
                cuda::sparse_cross_entropy_forward_cuda_impl_vec(
                    logits_ptr, tgt_ptr, result_ptr, batch_size, num_classes, stream);
            };

            if (targets_1d.dtype() == Dtype::UInt16) {
                if (will_save_stats) fwd_save_stats(targets_1d.data<uint16_t>());
                else                  fwd_no_stats(targets_1d.data<uint16_t>());
            } else if (targets_1d.dtype() == Dtype::Int64) {
                if (will_save_stats) fwd_save_stats(targets_1d.data<int64_t>());
                else                  fwd_no_stats(targets_1d.data<int64_t>());
            } else if (targets_1d.dtype() == Dtype::Int32) {
                if (will_save_stats) fwd_save_stats(targets_1d.data<int32_t>());
                else                  fwd_no_stats(targets_1d.data<int32_t>());
            } else {
                throw std::runtime_error("sparse_cross_entropy_loss: unsupported target dtype for CUDA");
            }
        };

        if (logits_2d.dtype() == Dtype::Float32) {
            dispatch_targets(logits_2d.data<float>(), result.data<float>());
        } else if (logits_2d.dtype() == Dtype::Float16) {
            dispatch_targets(logits_2d.data<float16_t>(), result.data<float16_t>());
        } else if (logits_2d.dtype() == Dtype::Bfloat16) {
            dispatch_targets(logits_2d.data<bfloat16_t>(), result.data<bfloat16_t>());
        } else {
            throw std::runtime_error("sparse_cross_entropy_loss: unsupported logits dtype for CUDA");
        }

        // Division by batch_size on GPU (simple scalar op)
        result = result / static_cast<float>(batch_size);

        // Build autograd graph — pass saved (max, sum) so backward skips Reduce.
        if (will_save_stats) {
            auto grad_fn = std::make_shared<SparseCrossEntropyLossBackward>(
                logits, targets_1d, saved_max, saved_sum, batch_size, num_classes);
            grad_fn->set_next_edge(0, get_grad_edge(logits));
            result.set_grad_fn(grad_fn);
            result.set_requires_grad(true);
        }
        if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
        return result;
        #else
        throw std::runtime_error("CUDA not available but tensor is on CUDA device");
        #endif
    }
    
    // Average loss
    float mean_loss = total_loss / static_cast<float>(batch_size);
    
    // Create scalar result tensor
    Tensor result = Tensor::full(Shape{{1}}, opts, mean_loss);
    
    // Build autograd graph
    if (GradMode::is_enabled() && logits.requires_grad()) {
        auto grad_fn = std::make_shared<SparseCrossEntropyLossBackward>(
            logits, targets_1d, batch_size, num_classes);
        grad_fn->set_next_edge(0, get_grad_edge(logits));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

} // namespace autograd
} // namespace OwnTensor
