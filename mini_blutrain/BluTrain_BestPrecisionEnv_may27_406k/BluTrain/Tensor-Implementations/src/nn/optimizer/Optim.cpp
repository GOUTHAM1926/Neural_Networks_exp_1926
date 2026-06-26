// #include "nn/optimizer/Optim.h"
// #include "ops/TensorOps.h"
// #include "ops/ScalarOps.h"
// #include "ops/UnaryOps/Arithmetics.h"
// #include "ops/UnaryOps/Reduction.h"
// #include "core/Serialization.h"
// #include <fstream>
// #include <cmath>
// #include <iostream>
// #include "ops/helpers/AdamKernels.h"
// #include "ops/helpers/GradNormKernels.h"
// #include "ops/helpers/MultiTensorKernels.h"
// #include <cuda_runtime.h>
// #include "device/DeviceCore.h"
// #include "device/AllocationTracker.h"

// namespace OwnTensor {
// namespace nn {

// Optimizer::Optimizer(const std::vector<Tensor>& params) : params_(params) {
//    for (const auto& p : params_) {
//        void* key = p.unsafeGetTensorImpl();
//        if (p.requires_grad() && p.dtype() != Dtype::Float32) {
//            master_params_[key] = p.as_type(Dtype::Float32);
//        }
//    }
// }

// void Optimizer::zero_grad() {
//    // Fast path: every GPU-resident grad (any dtype) goes into one multi-tensor
//    // launch that zeros bytes via uint4 stores. Collapses N per-parameter
//    // cudaMemsetAsync calls into ~ceil(N/48) kernel launches. For GPT-2 (148
//    // params) that's 4 launches per zero_grad() instead of 148.
//    //
//    // Dtype-agnostic: works for FP32 / FP16 / BF16 / int / complex because
//    // numeric zero is all-zero bytes for every type we use. CPU-resident grads
//    // still use the per-param path (CPU loop).
//    std::vector<cuda::ZeroTensorInfo> gpu_grads;
//    gpu_grads.reserve(params_.size());

//    for (auto& p : params_) {
//        if (!p.requires_grad() || !p.has_grad()) continue;

//        if (p.device().is_cuda()) {
//            gpu_grads.push_back({
//                p.grad(),                                // void*
//                static_cast<int64_t>(p.grad_nbytes())    // total bytes of the grad
//            });
//        } else {
//            p.zero_grad();
//        }
//    }

//    if (!gpu_grads.empty()) {
//        cuda::multi_tensor_zero_cuda(gpu_grads);
//    }
// }

// Tensor* Optimizer::get_master_weight(const Tensor& v) {
//    void* key = v.unsafeGetTensorImpl();
//    auto it = master_params_.find(key);
//    if (it != master_params_.end()) {
//        return &it->second;
//    }
//    return nullptr;
// }

// void Optimizer::save_state(std::ostream& /*os*/) {
//     // Base class only handles params which are usually saved by Module::save_state_dict.
//     // However, we might want to save master_params if they exist.
//     // For now, base implementation does nothing.
// }

// void Optimizer::load_state(std::istream& /*is*/) {
//     // Base implementation does nothing.
// }

// SGDOptimizer::SGDOptimizer(const std::vector<Tensor>& params, float learning_rate, float momentum, float weight_decay)
//    : Optimizer(params), learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

// void SGDOptimizer::step() {
//    step_count_++;
//     if (!initialized_) {
//         for (auto& p : params_) {
//             // Always use Float32 for momentum buffers
//             TensorOptions opts = TensorOptions()
//                 .with_dtype(Dtype::Float32)
//                 .with_device(p.device());
//             momentum_buffer_.push_back(Tensor::zeros(p.shape(), opts));
//         }
//         initialized_ = true;
//     }

//    for (size_t i = 0; i < params_.size(); ++i) {
//        Tensor& p = params_[i];
//        if (!p.requires_grad() || !p.has_grad()) continue;

//        Tensor grad_f32;
//        try {
//            grad_f32 = p.grad_view();
//        } catch (...) {
//            continue;
//        }
      
//        // Ensure grad is float32
//        if (grad_f32.dtype() != Dtype::Float32) {
//            grad_f32 = grad_f32.as_type(Dtype::Float32);
//        }

//        // Unscale gradient using attached LossScaler
//        if (scaler_) {
//            grad_f32 *= (1.0f / scaler_->scale());
//        }

//        // Mixed precision: use FP32 master weight if available(new)
//        Tensor* master = get_master_weight(p);
//        Dtype orig_dtype = p.dtype();
//        bool needs_cast_back = (master != nullptr);
//        // work_tensor is the FP32 tensor we do all math on(new)
//        Tensor& work = master ? *master : p;

//        int64_t numel = p.numel();

//        // CPU implementation with direct data manipulation
//        if (p.device().is_cpu()) {
//            //float* param_data = p.data<float>();
//            float* param_data = work.data<float>();
//            const float* grad_data = grad_f32.data<float>();
//            float* buf_data = nullptr;
          
//            if (momentum_ != 0) {
//                Tensor& buf = momentum_buffer_[i];
//                if (buf.numel() == 0) {
//                    TensorOptions opts = TensorOptions()
//                        .with_dtype(Dtype::Float32)
//                        .with_device(p.device());
//                    buf = Tensor::zeros(p.shape(), opts);
//                }
//                buf_data = buf.data<float>();
//            }

//            for (int64_t j = 0; j < numel; ++j) {
//                float g = grad_data[j];
              
//                // Weight decay (L2 penalty)
//                if (weight_decay_ != 0) {
//                    g += weight_decay_ * param_data[j];
//                }
              
//                // Momentum
//                if (momentum_ != 0 && buf_data) {
//                    buf_data[j] = momentum_ * buf_data[j] + g;
//                    g = buf_data[j];
//                }
              
//                // Update parameter (in FP32 master)
//                param_data[j] -= learning_rate_ * g;
//            }


//            // Cast back: FP32 master → original dtype param(new)
//            if (needs_cast_back) {
//                p.copy_(work.as_type(orig_dtype));
//            }
//        } else {
//            // GPU path: use tensor operations on FP32 master
//            // Weight decay (L2 penalty)
//            if (weight_decay_ != 0) {
//                //Tensor p_f32 = p;
//                //if (p_f32.dtype() != Dtype::Float32) {
//                //    p_f32 = w_f32.as_type(Dtype::Float32);
//                Tensor w_f32 = work;
//                if (w_f32.dtype() != Dtype::Float32) {
//                    w_f32 = w_f32.as_type(Dtype::Float32);
//                }
//                //grad_f32 = grad_f32 + weight_decay_ * p_f32;
//                grad_f32 = grad_f32 + weight_decay_ * w_f32;
//            }

//            // Momentum
//            if (momentum_ != 0) {
//                Tensor& buf = momentum_buffer_[i];
//                if (buf.numel() == 0) {
//                    TensorOptions opts = TensorOptions()
//                        .with_dtype(Dtype::Float32)
//                        .with_device(p.device());
//                    buf = Tensor::zeros(p.shape(), opts);
//                }
//                buf = momentum_ * buf + grad_f32;
//                grad_f32 = buf;
//            }

//            // Update parameters
//            //p += -learning_rate_ * grad_f32;

//            // Update FP32 master (or param if already FP32)
//            work += -learning_rate_ * grad_f32;

//            // Cast back: FP32 master → original dtype param
//            if (needs_cast_back) {
//                p.copy_(work.as_type(orig_dtype));
//            }
//        }
//    }
// }

// void SGDOptimizer::save_state(std::ostream& os) {
//     os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));
//     int count = static_cast<int>(momentum_buffer_.size());
//     os.write(reinterpret_cast<const char*>(&count), sizeof(int));
//     for (const auto& t : momentum_buffer_) {
//         save_tensor(t, os);
//     }
// }

// void SGDOptimizer::load_state(std::istream& is) {
//     is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
//     int count;
//     is.read(reinterpret_cast<char*>(&count), sizeof(int));
//     if (momentum_buffer_.empty() && count > 0) {
//         momentum_buffer_.resize(count);
//     }
//     if (count != static_cast<int>(momentum_buffer_.size())) {
//         if (count == 0 && momentum_buffer_.empty()) return;
//         throw std::runtime_error("SGDOptimizer state count mismatch!");
//     }
//     for (int i = 0; i < count; ++i) {
//         momentum_buffer_[i] = load_tensor(is).to(params_[i].device());
//     }
//     initialized_ = true;
// }



// // **************************************************************************************
// // ======================== Adam Optimizer ==============================================
// // **************************************************************************************
// Adam::Adam(const std::vector<Tensor>& params,
//            float lr,
//            float beta1,
//            float beta2,
//            float eps,
//            float weight_decay)
//     : Optimizer(params),
//     lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), 
//     weight_decay_(weight_decay), step_count_(0),
//     initialized_(false) {}

// void Adam::step() {
//    step_count_++;
  
//    // Lazy initialization of momentum buffers — always FP32
//    if (!initialized_) {
//        m_.reserve(params_.size());
//        v_.reserve(params_.size());
//        for (auto& param : params_) {
//            TensorOptions opts = TensorOptions()
//                //.with_dtype(param.dtype())
//                .with_dtype(Dtype::Float32)
//                .with_device(param.device());
//            m_.push_back(Tensor::zeros(param.shape(), opts));
//            v_.push_back(Tensor::zeros(param.shape(), opts));
//        }
//        initialized_ = true;
//    }
  
//    // Bias correction factors
//    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
//    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

//    // Multi-tensor optimization: Collect GPU tensors for batched update
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_params_info;
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_m_info;
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_v_info;

//    // Track which GPU params need cast-back (index in params_ → original dtype)(new) 
//    std::vector<std::pair<size_t, Dtype>> gpu_cast_back;

//    // Reserve memory to avoid reallocations
//    gpu_params_info.reserve(params_.size());
//    gpu_grads_info.reserve(params_.size());
//    gpu_m_info.reserve(params_.size());
//    gpu_v_info.reserve(params_.size());

//    static const size_t MAX_CHUNK_SIZE = 512;

//    for (size_t i = 0; i < params_.size(); ++i) {
//        Tensor& param = params_[i];
//        if (!param.requires_grad() || !param.has_grad()) {
//            continue;
//        }
      
//        Tensor grad;
//        try{
//            grad = param.grad_view();
//        } catch(...){
//            continue;
//        }

//        // Ensure grad is FP32(new)
//        if (grad.dtype() != Dtype::Float32) {
//            grad = grad.as_type(Dtype::Float32);
//        }

//       // Unscale gradient using attached LossScaler
//       if (scaler_) {
//           grad *= (1.0f / scaler_->scale());
//       }

//        // Mixed precision: use FP32 master weight if available(new)
//        Tensor* master = get_master_weight(param);
//        Dtype orig_dtype = param.dtype();
//        bool needs_cast_back = (master != nullptr);
//        Tensor& work = master ? *master : param;

//        int64_t numel = param.numel();

//        // CPU implementation with direct data manipulation
//        if (param.device().is_cpu()) {
//            //float* param_data = param.data<float>();
//            //const float* grad_data = param.grad<float>();
//            float* param_data = work.data<float>();
//            const float* grad_data = grad.data<float>();
//            float* m_data = m_[i].data<float>();
//            float* v_data = v_[i].data<float>();
          
//            for (int64_t j = 0; j < numel; ++j) {
//                float g = grad_data[j];
              
//                // Apply weight decay to grad (coupled wt. decay)
//                if (weight_decay_ > 0.0f) {
//                    g += weight_decay_ * param_data[j];
//                }
              
//                // Update first moment: m = beta1 * m + (1 - beta1) * g
//                m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
              
//                // Update second moment: v = beta2 * v + (1 - beta2) * g^2
//                v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
              
//                // Bias-corrected estimates
//                float m_hat = m_data[j] / bias_correction1;
//                float v_hat = v_data[j] / bias_correction2;
              
//                // Update parameter (in FP32 master)
//                param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
//            }

//            // Cast back: FP32 master → original dtype param (new)
//            if (needs_cast_back) {
//                param.copy_(work.as_type(orig_dtype));
//            }
//        } else {
//            // Collect for Multi-Tensor Update — use master weight's FP32 data
//            //gpu_params_info.push_back({param.data<float>(), numel});
//            //gpu_grads_info.push_back({param.grad<float>(), numel});
//            gpu_params_info.push_back({work.data<float>(), numel});
//            gpu_grads_info.push_back({grad.data<float>(), numel});
//            gpu_m_info.push_back({m_[i].data<float>(), numel});
//            gpu_v_info.push_back({v_[i].data<float>(), numel});

//            if (needs_cast_back) {//new
//                gpu_cast_back.push_back({i, orig_dtype});
//            }
//        }
//    }

//    // Process GPU tensors in chunks
//    size_t total_gpu_tensors = gpu_params_info.size();
//    if (total_gpu_tensors > 0) {
//        for (size_t i = 0; i < total_gpu_tensors; i += MAX_CHUNK_SIZE) {
//            size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_gpu_tensors - i);

//            std::vector<OwnTensor::cuda::TensorInfo> p_chunk(gpu_params_info.begin() + i, gpu_params_info.begin() + i + chunk_size);
//            std::vector<OwnTensor::cuda::TensorInfo> g_chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
//            std::vector<OwnTensor::cuda::TensorInfo> m_chunk(gpu_m_info.begin() + i, gpu_m_info.begin() + i + chunk_size);
//            std::vector<OwnTensor::cuda::TensorInfo> v_chunk(gpu_v_info.begin() + i, gpu_v_info.begin() + i + chunk_size);

//            cuda::multi_tensor_adam_cuda(
//                p_chunk,
//                g_chunk,
//                m_chunk,
//                v_chunk,
//                lr_,
//                beta1_,
//                beta2_,
//                eps_,
//                weight_decay_,
//                bias_correction1,
//                bias_correction2,
//                false  // is_adamw = false for Adam
//            );
//        }

//        // Cast back GPU params: FP32 master → original dtype param
//        for (auto& [param_idx, dtype] : gpu_cast_back) {
//            Tensor* m = get_master_weight(params_[param_idx]);
//            if (m) {
//                params_[param_idx].copy_(m->as_type(dtype));
//            }
//        }
//    }
// }
// void Adam::save_state(std::ostream& os) {
//     os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));
//     int count = static_cast<int>(m_.size());
//     os.write(reinterpret_cast<const char*>(&count), sizeof(int));
//     for (const auto& t : m_) save_tensor(t, os);
//     for (const auto& t : v_) save_tensor(t, os);
// }

// void Adam::load_state(std::istream& is) {
//     is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
//     int count;
//     is.read(reinterpret_cast<char*>(&count), sizeof(int));
    
//     // Lazy init if needed
//     if (!initialized_) {
//         m_.resize(count);
//         v_.resize(count);
//         initialized_ = true;
//     }

//     if (count != static_cast<int>(m_.size())) {
//         throw std::runtime_error("Adam state count mismatch!");
//     }

//     for (int i = 0; i < count; ++i) m_[i] = load_tensor(is).to(params_[i].device());
//     for (int i = 0; i < count; ++i) v_[i] = load_tensor(is).to(params_[i].device());
// }


// } // namespace nn


// //======================================================================
// //-----------------Clip-Grad-Norm---------------------------------------
// //======================================================================
// namespace nn {

// float clip_grad_norm_(std::vector<Tensor>& params, float max_norm, float norm_type, bool error_if_nonfinite) {
//    // FAST GPU-based gradient clipping using multi-tensor CUDA kernels
  
//    // Persistent GPU buffers - allocated once, never freed (tiny size, acceptable leak)
//    static float* s_d_norm = nullptr;
//    static float* s_d_clip_coef = nullptr;
  
//    bool is_cuda = false;
  
//    // Check if any parameter is on CUDA
//    for (auto& param : params) {
//        if (!param.requires_grad() || !param.has_grad()) continue;
//        if (param.device().is_cuda()) {
//            is_cuda = true;
//            break;
//        }
//    }

//    // Handle norm_type
//    bool is_inf_norm = std::isinf(norm_type);
  
//    if (is_cuda) {
//        // GPU PATH: Use Multi-Tensor Kernels
      
//        // Lazy allocation of persistent buffers (one-time cost)
//        if (!s_d_norm) {
//            cudaMalloc(&s_d_norm, sizeof(float));
//            cudaMalloc(&s_d_clip_coef, sizeof(float));
//        }
      
//        // Reset accumulator (async, no sync needed)
//        cudaMemsetAsync(s_d_norm, 0, sizeof(float));

//        // Collect all GPU gradients
//        std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
//        gpu_grads_info.reserve(params.size());

//        for (auto& param : params) {
//            if (!param.requires_grad() || !param.has_grad()) continue;
          
//            try {
//                Tensor grad = param.grad_view();
//                if (grad.device().is_cuda() && grad.dtype() == Dtype::Float32) {
//                    gpu_grads_info.push_back({grad.data<float>(), grad.numel()});
//                }
//            } catch (...) {
//                continue;
//            }
//        }
      
//        // Phase 1: Compute Norm (Chunked)
//        static const size_t MAX_CHUNK_SIZE = 512;
//        size_t total_tensors = gpu_grads_info.size();
      
//        if (total_tensors > 0) {
//            if (is_inf_norm) {
//                // FIXME: Multi-tensor inf norm not implemented yet in this file, fallback or implement?
//                // For now, falling back to sequential for Inf norm, or assuming L2 for multi-tensor context primarily.
//                // Given the context of "Optimization Phase Bottleneck", L2 is the critical path.
//                // We will stick to the existing single-kernel loop for Inf norm to be safe, as multi-tensor API doesn't show inf norm.
              
//                for (auto& param : params) {
//                     if (!param.requires_grad() || !param.has_grad()) continue;
//                      try {
//                        Tensor grad = param.grad_view();
//                        if (grad.device().is_cuda() && grad.dtype() == Dtype::Float32) {
//                             cuda::grad_norm_inf_cuda(grad.data<float>(), s_d_norm, grad.numel());
//                        }
//                      } catch(...) {}
//                }

//            } else {
//                // L2 Norm - Multi-Tensor
//                for (size_t i = 0; i < total_tensors; i += MAX_CHUNK_SIZE) {
//                    size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_tensors - i);
//                    std::vector<OwnTensor::cuda::TensorInfo> chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
//                    cuda::multi_tensor_grad_norm_cuda(chunk, s_d_norm);
//                }
//            }
//        }
      
//        // Compute clip coefficient on GPU (also computes sqrt for L2 norm)
//        cuda::compute_clip_coef_cuda(s_d_norm, s_d_clip_coef, max_norm, is_inf_norm);
      
//        // Phase 2: Scale Gradients (Chunked)
//        if (total_tensors > 0) {
//             for (size_t i = 0; i < total_tensors; i += MAX_CHUNK_SIZE) {
//                size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_tensors - i);
//                std::vector<OwnTensor::cuda::TensorInfo> chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
//                cuda::multi_tensor_scale_cuda(chunk, s_d_clip_coef);
//            }
//        }
      
//        // Only now copy the total_norm to CPU for return value
//        float total_norm;
//        cudaMemcpy(&total_norm, s_d_norm, sizeof(float), cudaMemcpyDeviceToHost);
      
//        // No cudaFree - buffers are persistent

//        if (error_if_nonfinite && (std::isnan(total_norm) || std::isinf(total_norm))) {
//             throw std::runtime_error("The total norm of gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`");
//        }
      
//        return total_norm;
//    } else {
//        // CPU PATH: Simple loop
//        float total_norm_sq = 0.0f;
//        float total_norm_inf = 0.0f;
      
//        for (auto& param : params) {
//            if (!param.requires_grad() || !param.has_grad()) continue;
          
//            try {
//                Tensor grad = param.grad_view();
//                const float* data = grad.data<float>();
//                int64_t n = grad.numel();
              
//                if (is_inf_norm) {
//                    for (int64_t i = 0; i < n; ++i) {
//                        float val = std::abs(data[i]);
//                        if (val > total_norm_inf) total_norm_inf = val;
//                    }
//                } else {
//                    for (int64_t i = 0; i < n; ++i) {
//                        total_norm_sq += data[i] * data[i];
//                    }
//                }
//            } catch (...) {
//                continue;
//            }
//        }
      
//        float total_norm;
//        if (is_inf_norm) {
//            total_norm = total_norm_inf;
//        } else {
//            total_norm = std::sqrt(total_norm_sq);
//        }


//        if (error_if_nonfinite && (std::isnan(total_norm) || std::isinf(total_norm))) {
//             throw std::runtime_error("The total norm of gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`");
//        }
      
//        float clip_coef = max_norm / (total_norm + 1e-6f);
//        if (clip_coef < 1.0f) {
//            for (auto& param : params) {
//                if (!param.requires_grad() || !param.has_grad()) continue;
              
//                try {
//                    Tensor grad = param.grad_view();
//                    float* data = grad.data<float>();
//                    int64_t n = grad.numel();
//                    for (int64_t i = 0; i < n; ++i) {
//                        data[i] *= clip_coef;
//                    }
//                } catch (...) {
//                    continue;
//                }
//            }
//        }
//        return total_norm;
//    }
// }


// // *********************************************************************************************
// // ============================ AdamW Optimizer ================================================
// // *********************************************************************************************

// AdamW::AdamW(const std::vector<Tensor>& params,
//          float lr,
//          float beta1,
//          float beta2,
//          float eps,
//          float weight_decay)
//   : Optimizer(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
//     weight_decay_(weight_decay), step_count_(0),
//     initialized_(false) {}

// void AdamW::step() {
//   step_count_++;
//   TRACK_ALLOC_SCOPE("L601:nn::AdamW::step");
//   // Lazy initialization of momentum buffers — always FP32
//   if (!initialized_) {
//       m_.reserve(params_.size());
//       v_.reserve(params_.size());
//       for (auto& param : params_) {
//           TensorOptions opts = TensorOptions()
//               //.with_dtype(param.dtype())
//               .with_dtype(Dtype::Float32)
//               .with_device(param.device());
//           m_.push_back(Tensor::zeros(param.shape(), opts));
//           v_.push_back(Tensor::zeros(param.shape(), opts));
//       }
//       initialized_ = true;
//   }

//   // Bias correction factors
//   float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
//   float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
 
//   // Multi-tensor optimization for AdamW
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_params_info;
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_m_info;
//    std::vector<OwnTensor::cuda::TensorInfo> gpu_v_info;

//    // Track which GPU params need cast-back(new)
//    std::vector<std::pair<size_t, Dtype>> gpu_cast_back;

//    gpu_params_info.reserve(params_.size());
//    gpu_grads_info.reserve(params_.size());
//    gpu_m_info.reserve(params_.size());
//    gpu_v_info.reserve(params_.size());
  
//    static const size_t MAX_CHUNK_SIZE = 512;

//    for (size_t i = 0; i < params_.size(); ++i) {
//       Tensor& param = params_[i];
//       if (!param.requires_grad() || !param.has_grad()) {
//           continue;
//       }
//       Tensor grad;
//       try{
//           grad = param.grad_view();
//       } catch(...){
//           continue;
//       }

//       // Ensure grad is FP32(new)
//       if (grad.dtype() != Dtype::Float32) {
//           grad = grad.as_type(Dtype::Float32);
//       }
      
//       // Unscale gradient using attached LossScaler
//       if (scaler_) {
//           grad *= (1.0f / scaler_->scale());
//       }

//       // Mixed precision: use FP32 master weight if available(new)
//       Tensor* master = get_master_weight(param);
//       Dtype orig_dtype = param.dtype();
//       bool needs_cast_back = (master != nullptr);
//       Tensor& work = master ? *master : param;

//       int64_t numel = param.numel();
   
//       // CPU implementation with direct data manipulation
//       if (param.device().is_cpu()) {
//           //float* param_data = param.data<float>();
//           //const float* grad_data = param.grad<float>();
//           float* param_data = work.data<float>();
//           const float* grad_data = grad.data<float>();
//           float* m_data = m_[i].data<float>();
//           float* v_data = v_[i].data<float>();
       
//           for (int64_t j = 0; j < numel; ++j) {
//               float g = grad_data[j];
           
//                // Apply weight decay (Decoupled weight decay) on FP32 master
//                if (weight_decay_ > 0.0f) {
//                    param_data[j] *= (1.0f - lr_ * weight_decay_);
//                }
           
//               // Update first moment: m = beta1 * m + (1.0f - beta1) * g
//               m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
           
//               // Update second moment: v = beta2 * v + (1.0f - beta2) * g^2
//               v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
           
//               // Bias-corrected estimates
//               float m_hat = m_data[j] / bias_correction1;
//               float v_hat = v_data[j] / bias_correction2;
           
//               // Update parameter (in FP32 master)
//               param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
//           }

//           // Cast back: FP32 master → original dtype param(new)
//           if (needs_cast_back) {
//               param.copy_(work.as_type(orig_dtype));
//           }
//       } else {
//           // Collect for Multi-Tensor Update — use master weight's FP32 data
//           //gpu_params_info.push_back({param.data<float>(), numel});
//           //gpu_grads_info.push_back({param.grad<float>(), numel});
//           gpu_params_info.push_back({work.data<float>(), numel});
//           gpu_grads_info.push_back({grad.data<float>(), numel});
//           gpu_m_info.push_back({m_[i].data<float>(), numel});
//           gpu_v_info.push_back({v_[i].data<float>(), numel});

//           if (needs_cast_back) { //new
//               gpu_cast_back.push_back({i, orig_dtype});
//           }
//       }
//   }
 
//    // Process GPU tensors in chunks (AdamW shares the same kernel signature)
//    size_t total_gpu_tensors = gpu_params_info.size();
//    if (total_gpu_tensors > 0) {
//        for (size_t i = 0; i < total_gpu_tensors; i += MAX_CHUNK_SIZE) {
//            size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_gpu_tensors - i);

//            std::vector<OwnTensor::cuda::TensorInfo> p_chunk(gpu_params_info.begin() + i, gpu_params_info.begin() + i + chunk_size);
//            std::vector<OwnTensor::cuda::TensorInfo> g_chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
//            std::vector<OwnTensor::cuda::TensorInfo> m_chunk(gpu_m_info.begin() + i, gpu_m_info.begin() + i + chunk_size);
//            std::vector<OwnTensor::cuda::TensorInfo> v_chunk(gpu_v_info.begin() + i, gpu_v_info.begin() + i + chunk_size);

//            cuda::multi_tensor_adam_cuda(
//                p_chunk,
//                g_chunk,
//                m_chunk,
//                v_chunk,
//                lr_,
//                beta1_,
//                beta2_,
//                eps_,
//                weight_decay_,
//                bias_correction1,
//                bias_correction2,
//                true   // is_adamw = true for AdamW
//            );
//        }

//        // Cast back GPU params: FP32 master to original dtype param
//        for (auto& [param_idx, dtype] : gpu_cast_back) {
//            Tensor* m = get_master_weight(params_[param_idx]);
//            if (m) {
//                params_[param_idx].copy_(m->as_type(dtype));
//            }
//        }
//    }
// }

// void AdamW::save_state(std::ostream& os) {
//     os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));
//     int count = static_cast<int>(m_.size());
//     os.write(reinterpret_cast<const char*>(&count), sizeof(int));
//     for (const auto& t : m_) save_tensor(t, os);
//     for (const auto& t : v_) save_tensor(t, os);
// }

// void AdamW::load_state(std::istream& is) {
//     is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
//     int count;
//     is.read(reinterpret_cast<char*>(&count), sizeof(int));
    
//     // Lazy init if needed
//     if (!initialized_) {
//         m_.resize(count);
//         v_.resize(count);
//         initialized_ = true;
//     }

//     if (count != static_cast<int>(m_.size())) {
//         throw std::runtime_error("AdamW state count mismatch!");
//     }

//     for (int i = 0; i < count; ++i) m_[i] = load_tensor(is).to(params_[i].device());
//     for (int i = 0; i < count; ++i) v_[i] = load_tensor(is).to(params_[i].device());
// }

// } // namespace nn
// } // namespace OwnTensor






































#include "nn/optimizer/Optim.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Reduction.h"
#include "core/Serialization.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include "ops/helpers/AdamKernels.h"
#include "ops/helpers/GradNormKernels.h"
#include "ops/helpers/MultiTensorKernels.h"
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include "device/AllocationTracker.h"

namespace OwnTensor {
namespace nn {

Optimizer::Optimizer(const std::vector<Tensor>& params) : params_(params) {
   for (const auto& p : params_) {
       void* key = p.unsafeGetTensorImpl();
       if (p.requires_grad() && p.dtype() != Dtype::Float32) {
           master_params_[key] = p.as_type(Dtype::Float32);
       }
   }
}

void Optimizer::zero_grad() {
   // Fast path: every GPU-resident grad (any dtype) goes into one multi-tensor
   // launch that zeros bytes via uint4 stores. Collapses N per-parameter
   // cudaMemsetAsync calls into ~ceil(N/48) kernel launches. For GPT-2 (148
   // params) that's 4 launches per zero_grad() instead of 148.
   //
   // Dtype-agnostic: works for FP32 / FP16 / BF16 / int / complex because
   // numeric zero is all-zero bytes for every type we use. CPU-resident grads
   // still use the per-param path.
   std::vector<cuda::ZeroTensorInfo> gpu_grads;
   gpu_grads.reserve(params_.size());

   for (auto& p : params_) {
       if (!p.requires_grad() || !p.has_grad()) continue;

       if (p.device().is_cuda()) {
           gpu_grads.push_back({
               p.grad(),
               static_cast<int64_t>(p.grad_nbytes())
           });
       } else {
           p.zero_grad();
       }
   }

   if (!gpu_grads.empty()) {
       cuda::multi_tensor_zero_cuda(gpu_grads);
   }
}

Tensor* Optimizer::get_master_weight(const Tensor& v) {
   void* key = v.unsafeGetTensorImpl();
   auto it = master_params_.find(key);
   if (it != master_params_.end()) {
       return &it->second;
   }
   return nullptr;
}

void Optimizer::save_state(std::ostream& /*os*/) {
    // Base class only handles params which are usually saved by Module::save_state_dict.
    // However, we might want to save master_params if they exist.
    // For now, base implementation does nothing.
}

void Optimizer::load_state(std::istream& /*is*/) {
    // Base implementation does nothing.
}

SGDOptimizer::SGDOptimizer(const std::vector<Tensor>& params, float learning_rate, float momentum, float weight_decay)
   : Optimizer(params), learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

void SGDOptimizer::step() {
   step_count_++;
    if (!initialized_) {
        for (auto& p : params_) {
            // Always use Float32 for momentum buffers
            TensorOptions opts = TensorOptions()
                .with_dtype(Dtype::Float32)
                .with_device(p.device());
            momentum_buffer_.push_back(Tensor::zeros(p.shape(), opts));
        }
        initialized_ = true;
    }

   for (size_t i = 0; i < params_.size(); ++i) {
       Tensor& p = params_[i];
       if (!p.requires_grad() || !p.has_grad()) continue;

       Tensor grad_f32;
       try {
           grad_f32 = p.grad_view();
       } catch (...) {
           continue;
       }
      
       // Ensure grad is float32
       if (grad_f32.dtype() != Dtype::Float32) {
           grad_f32 = grad_f32.as_type(Dtype::Float32);
       }

       // Unscale gradient using attached LossScaler
       if (scaler_) {
           grad_f32 *= (1.0f / scaler_->scale());
       }

       // Mixed precision: use FP32 master weight if available(new)
       Tensor* master = get_master_weight(p);
       Dtype orig_dtype = p.dtype();
       bool needs_cast_back = (master != nullptr);
       // work_tensor is the FP32 tensor we do all math on(new)
       Tensor& work = master ? *master : p;

       int64_t numel = p.numel();

       // CPU implementation with direct data manipulation
       if (p.device().is_cpu()) {
           //float* param_data = p.data<float>();
           float* param_data = work.data<float>();
           const float* grad_data = grad_f32.data<float>();
           float* buf_data = nullptr;
          
           if (momentum_ != 0) {
               Tensor& buf = momentum_buffer_[i];
               if (buf.numel() == 0) {
                   TensorOptions opts = TensorOptions()
                       .with_dtype(Dtype::Float32)
                       .with_device(p.device());
                   buf = Tensor::zeros(p.shape(), opts);
               }
               buf_data = buf.data<float>();
           }

           for (int64_t j = 0; j < numel; ++j) {
               float g = grad_data[j];
              
               // Weight decay (L2 penalty)
               if (weight_decay_ != 0) {
                   g += weight_decay_ * param_data[j];
               }
              
               // Momentum
               if (momentum_ != 0 && buf_data) {
                   buf_data[j] = momentum_ * buf_data[j] + g;
                   g = buf_data[j];
               }
              
               // Update parameter (in FP32 master)
               param_data[j] -= learning_rate_ * g;
           }


           // Cast back: FP32 master → original dtype param(new)
           if (needs_cast_back) {
               p.copy_(work.as_type(orig_dtype));
           }
       } else {
           // GPU path: use tensor operations on FP32 master
           // Weight decay (L2 penalty)
           if (weight_decay_ != 0) {
               //Tensor p_f32 = p;
               //if (p_f32.dtype() != Dtype::Float32) {
               //    p_f32 = w_f32.as_type(Dtype::Float32);
               Tensor w_f32 = work;
               if (w_f32.dtype() != Dtype::Float32) {
                   w_f32 = w_f32.as_type(Dtype::Float32);
               }
               //grad_f32 = grad_f32 + weight_decay_ * p_f32;
               grad_f32 = grad_f32 + weight_decay_ * w_f32;
           }

           // Momentum
           if (momentum_ != 0) {
               Tensor& buf = momentum_buffer_[i];
               if (buf.numel() == 0) {
                   TensorOptions opts = TensorOptions()
                       .with_dtype(Dtype::Float32)
                       .with_device(p.device());
                   buf = Tensor::zeros(p.shape(), opts);
               }
               buf = momentum_ * buf + grad_f32;
               grad_f32 = buf;
           }

           // Update parameters
           //p += -learning_rate_ * grad_f32;

           // Update FP32 master (or param if already FP32)
           work += -learning_rate_ * grad_f32;

           // Cast back: FP32 master → original dtype param
           if (needs_cast_back) {
               p.copy_(work.as_type(orig_dtype));
           }
       }
   }
}

void SGDOptimizer::save_state(std::ostream& os) {
    os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));
    int count = static_cast<int>(momentum_buffer_.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(int));
    for (const auto& t : momentum_buffer_) {
        save_tensor(t, os);
    }
}

void SGDOptimizer::load_state(std::istream& is) {
    is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
    int count;
    is.read(reinterpret_cast<char*>(&count), sizeof(int));
    if (momentum_buffer_.empty() && count > 0) {
        momentum_buffer_.resize(count);
    }
    if (count != static_cast<int>(momentum_buffer_.size())) {
        if (count == 0 && momentum_buffer_.empty()) return;
        throw std::runtime_error("SGDOptimizer state count mismatch!");
    }
    for (int i = 0; i < count; ++i) {
        momentum_buffer_[i] = load_tensor(is).to(params_[i].device());
    }
    initialized_ = true;
}



// **************************************************************************************
// ======================== Adam Optimizer ==============================================
// **************************************************************************************
Adam::Adam(const std::vector<Tensor>& params,
           float lr,
           float beta1,
           float beta2,
           float eps,
           float weight_decay)
    : Optimizer(params),
    lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), 
    weight_decay_(weight_decay), step_count_(0),
    initialized_(false) {}

void Adam::step() {
   step_count_++;
  
   // Lazy initialization of momentum buffers — always FP32
   if (!initialized_) {
       m_.reserve(params_.size());
       v_.reserve(params_.size());
       for (auto& param : params_) {
           TensorOptions opts = TensorOptions()
               //.with_dtype(param.dtype())
               .with_dtype(Dtype::Float32)
               .with_device(param.device());
           m_.push_back(Tensor::zeros(param.shape(), opts));
           v_.push_back(Tensor::zeros(param.shape(), opts));
       }
       initialized_ = true;
   }
  
   // Bias correction factors
   float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
   float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

   // Multi-tensor optimization: Collect GPU tensors for batched update
   std::vector<OwnTensor::cuda::TensorInfo> gpu_params_info;
   std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
   std::vector<OwnTensor::cuda::TensorInfo> gpu_m_info;
   std::vector<OwnTensor::cuda::TensorInfo> gpu_v_info;

   // Track which GPU params need cast-back (index in params_ → original dtype)(new) 
   std::vector<std::pair<size_t, Dtype>> gpu_cast_back;

   // Reserve memory to avoid reallocations
   gpu_params_info.reserve(params_.size());
   gpu_grads_info.reserve(params_.size());
   gpu_m_info.reserve(params_.size());
   gpu_v_info.reserve(params_.size());

   static const size_t MAX_CHUNK_SIZE = 512;

   for (size_t i = 0; i < params_.size(); ++i) {
       Tensor& param = params_[i];
       if (!param.requires_grad() || !param.has_grad()) {
           continue;
       }
      
       Tensor grad;
       try{
           grad = param.grad_view();
       } catch(...){
           continue;
       }

       // Ensure grad is FP32(new)
       if (grad.dtype() != Dtype::Float32) {
           grad = grad.as_type(Dtype::Float32);
       }

      // Unscale gradient using attached LossScaler
      if (scaler_) {
          grad *= (1.0f / scaler_->scale());
      }

       // Mixed precision: use FP32 master weight if available(new)
       Tensor* master = get_master_weight(param);
       Dtype orig_dtype = param.dtype();
       bool needs_cast_back = (master != nullptr);
       Tensor& work = master ? *master : param;

       int64_t numel = param.numel();

       // CPU implementation with direct data manipulation
       if (param.device().is_cpu()) {
           //float* param_data = param.data<float>();
           //const float* grad_data = param.grad<float>();
           float* param_data = work.data<float>();
           const float* grad_data = grad.data<float>();
           float* m_data = m_[i].data<float>();
           float* v_data = v_[i].data<float>();
          
           for (int64_t j = 0; j < numel; ++j) {
               float g = grad_data[j];
              
               // Apply weight decay to grad (coupled wt. decay)
               if (weight_decay_ > 0.0f) {
                   g += weight_decay_ * param_data[j];
               }
              
               // Update first moment: m = beta1 * m + (1 - beta1) * g
               m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
              
               // Update second moment: v = beta2 * v + (1 - beta2) * g^2
               v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
              
               // Bias-corrected estimates
               float m_hat = m_data[j] / bias_correction1;
               float v_hat = v_data[j] / bias_correction2;
              
               // Update parameter (in FP32 master)
               param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
           }

           // Cast back: FP32 master → original dtype param (new)
           if (needs_cast_back) {
               param.copy_(work.as_type(orig_dtype));
           }
       } else {
           // Collect for Multi-Tensor Update — use master weight's FP32 data
           //gpu_params_info.push_back({param.data<float>(), numel});
           //gpu_grads_info.push_back({param.grad<float>(), numel});
           gpu_params_info.push_back({work.data<float>(), numel});
           gpu_grads_info.push_back({grad.data<float>(), numel});
           gpu_m_info.push_back({m_[i].data<float>(), numel});
           gpu_v_info.push_back({v_[i].data<float>(), numel});

           if (needs_cast_back) {//new
               gpu_cast_back.push_back({i, orig_dtype});
           }
       }
   }

   // Process GPU tensors in chunks
   size_t total_gpu_tensors = gpu_params_info.size();
   if (total_gpu_tensors > 0) {
       for (size_t i = 0; i < total_gpu_tensors; i += MAX_CHUNK_SIZE) {
           size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_gpu_tensors - i);

           std::vector<OwnTensor::cuda::TensorInfo> p_chunk(gpu_params_info.begin() + i, gpu_params_info.begin() + i + chunk_size);
           std::vector<OwnTensor::cuda::TensorInfo> g_chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
           std::vector<OwnTensor::cuda::TensorInfo> m_chunk(gpu_m_info.begin() + i, gpu_m_info.begin() + i + chunk_size);
           std::vector<OwnTensor::cuda::TensorInfo> v_chunk(gpu_v_info.begin() + i, gpu_v_info.begin() + i + chunk_size);

           cuda::multi_tensor_adam_cuda(
               p_chunk,
               g_chunk,
               m_chunk,
               v_chunk,
               lr_,
               beta1_,
               beta2_,
               eps_,
               weight_decay_,
               bias_correction1,
               bias_correction2,
               false  // is_adamw = false for Adam
           );
       }

       // Cast back GPU params: FP32 master → original dtype param
       for (auto& [param_idx, dtype] : gpu_cast_back) {
           Tensor* m = get_master_weight(params_[param_idx]);
           if (m) {
               params_[param_idx].copy_(m->as_type(dtype));
           }
       }
   }
}
void Adam::save_state(std::ostream& os) {
    os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));
    int count = static_cast<int>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(int));
    for (const auto& t : m_) save_tensor(t, os);
    for (const auto& t : v_) save_tensor(t, os);
}

void Adam::load_state(std::istream& is) {
    is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
    int count;
    is.read(reinterpret_cast<char*>(&count), sizeof(int));
    
    // Lazy init if needed
    if (!initialized_) {
        m_.resize(count);
        v_.resize(count);
        initialized_ = true;
    }

    if (count != static_cast<int>(m_.size())) {
        throw std::runtime_error("Adam state count mismatch!");
    }

    for (int i = 0; i < count; ++i) m_[i] = load_tensor(is).to(params_[i].device());
    for (int i = 0; i < count; ++i) v_[i] = load_tensor(is).to(params_[i].device());
}


} // namespace nn


//======================================================================
//-----------------Clip-Grad-Norm---------------------------------------
//======================================================================
namespace nn {

namespace {

// Persistent device buffers shared between sync and async clip variants.
// Tiny size (8 bytes total), allocated once and never freed.
struct ClipBufs {
    float* d_norm = nullptr;
    float* d_clip_coef = nullptr;
};
static ClipBufs& clip_bufs() {
    static ClipBufs b;
    if (!b.d_norm) {
        cudaMalloc(&b.d_norm, sizeof(float));
        cudaMalloc(&b.d_clip_coef, sizeof(float));
    }
    return b;
}

// Run the full GPU pipeline of clip_grad_norm_ — norm reduction over all
// gradient tensors, clip-coefficient compute, multi-tensor scale — with no
// sync. Leaves the L2 norm in *bufs.d_norm. Caller decides how to retrieve
// it (sync cudaMemcpy or async).
void clip_grad_norm_gpu_inplace(std::vector<Tensor>& params,
                                float max_norm,
                                bool is_inf_norm,
                                ClipBufs& bufs) {
    cudaMemsetAsync(bufs.d_norm, 0, sizeof(float));

    std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
    gpu_grads_info.reserve(params.size());
    for (auto& param : params) {
        if (!param.requires_grad() || !param.has_grad()) continue;
        try {
            Tensor grad = param.grad_view();
            if (grad.device().is_cuda() && grad.dtype() == Dtype::Float32) {
                gpu_grads_info.push_back({grad.data<float>(), grad.numel()});
            }
        } catch (...) { continue; }
    }

    static const size_t MAX_CHUNK_SIZE = 512;
    const size_t total_tensors = gpu_grads_info.size();

    if (total_tensors > 0) {
        if (is_inf_norm) {
            for (auto& param : params) {
                if (!param.requires_grad() || !param.has_grad()) continue;
                try {
                    Tensor grad = param.grad_view();
                    if (grad.device().is_cuda() && grad.dtype() == Dtype::Float32) {
                        cuda::grad_norm_inf_cuda(grad.data<float>(), bufs.d_norm, grad.numel());
                    }
                } catch (...) {}
            }
        } else {
            for (size_t i = 0; i < total_tensors; i += MAX_CHUNK_SIZE) {
                size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_tensors - i);
                std::vector<OwnTensor::cuda::TensorInfo> chunk(
                    gpu_grads_info.begin() + i,
                    gpu_grads_info.begin() + i + chunk_size);
                cuda::multi_tensor_grad_norm_cuda(chunk, bufs.d_norm);
            }
        }
    }

    cuda::compute_clip_coef_cuda(bufs.d_norm, bufs.d_clip_coef, max_norm, is_inf_norm);

    if (total_tensors > 0) {
        for (size_t i = 0; i < total_tensors; i += MAX_CHUNK_SIZE) {
            size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_tensors - i);
            std::vector<OwnTensor::cuda::TensorInfo> chunk(
                gpu_grads_info.begin() + i,
                gpu_grads_info.begin() + i + chunk_size);
            cuda::multi_tensor_scale_cuda(chunk, bufs.d_clip_coef);
        }
    }
}

} // namespace

void clip_grad_norm_async_(std::vector<Tensor>& params,
                           float max_norm,
                           float* pinned_host_norm,
                           float norm_type) {
    bool is_cuda = false;
    for (auto& param : params) {
        if (!param.requires_grad() || !param.has_grad()) continue;
        if (param.device().is_cuda()) { is_cuda = true; break; }
    }
    if (!is_cuda) {
        // Fall back to the sync CPU path and write the result into the
        // host buffer — there is no async story for the CPU loop.
        if (pinned_host_norm) {
            *pinned_host_norm = clip_grad_norm_(params, max_norm, norm_type, false);
        } else {
            (void)clip_grad_norm_(params, max_norm, norm_type, false);
        }
        return;
    }

    const bool is_inf_norm = std::isinf(norm_type);
    ClipBufs& bufs = clip_bufs();
    clip_grad_norm_gpu_inplace(params, max_norm, is_inf_norm, bufs);

    // Non-blocking D2H of the scalar norm into caller-owned pinned memory.
    // Default stream → no explicit stream argument needed; the copy is
    // ordered after the multi-tensor scale that just got queued.
    if (pinned_host_norm) {
        cudaMemcpyAsync(pinned_host_norm, bufs.d_norm,
                        sizeof(float), cudaMemcpyDeviceToHost);
    }
}

float clip_grad_norm_(std::vector<Tensor>& params, float max_norm, float norm_type, bool error_if_nonfinite) {
   // FAST GPU-based gradient clipping using multi-tensor CUDA kernels
  
   // Persistent GPU buffers - allocated once, never freed (tiny size, acceptable leak)
   static float* s_d_norm = nullptr;
   static float* s_d_clip_coef = nullptr;
  
   bool is_cuda = false;
  
   // Check if any parameter is on CUDA
   for (auto& param : params) {
       if (!param.requires_grad() || !param.has_grad()) continue;
       if (param.device().is_cuda()) {
           is_cuda = true;
           break;
       }
   }

   // Handle norm_type
   bool is_inf_norm = std::isinf(norm_type);
  
   if (is_cuda) {
       // GPU PATH: Use Multi-Tensor Kernels
      
       // Lazy allocation of persistent buffers (one-time cost)
       if (!s_d_norm) {
           cudaMalloc(&s_d_norm, sizeof(float));
           cudaMalloc(&s_d_clip_coef, sizeof(float));
       }
      
       // Reset accumulator (async, no sync needed)
       cudaMemsetAsync(s_d_norm, 0, sizeof(float));

       // Collect all GPU gradients
       std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
       gpu_grads_info.reserve(params.size());

       for (auto& param : params) {
           if (!param.requires_grad() || !param.has_grad()) continue;
          
           try {
               Tensor grad = param.grad_view();
               if (grad.device().is_cuda() && grad.dtype() == Dtype::Float32) {
                   gpu_grads_info.push_back({grad.data<float>(), grad.numel()});
               }
           } catch (...) {
               continue;
           }
       }
      
       // Phase 1: Compute Norm (Chunked)
       static const size_t MAX_CHUNK_SIZE = 512;
       size_t total_tensors = gpu_grads_info.size();
      
       if (total_tensors > 0) {
           if (is_inf_norm) {
               // FIXME: Multi-tensor inf norm not implemented yet in this file, fallback or implement?
               // For now, falling back to sequential for Inf norm, or assuming L2 for multi-tensor context primarily.
               // Given the context of "Optimization Phase Bottleneck", L2 is the critical path.
               // We will stick to the existing single-kernel loop for Inf norm to be safe, as multi-tensor API doesn't show inf norm.
              
               for (auto& param : params) {
                    if (!param.requires_grad() || !param.has_grad()) continue;
                     try {
                       Tensor grad = param.grad_view();
                       if (grad.device().is_cuda() && grad.dtype() == Dtype::Float32) {
                            cuda::grad_norm_inf_cuda(grad.data<float>(), s_d_norm, grad.numel());
                       }
                     } catch(...) {}
               }

           } else {
               // L2 Norm - Multi-Tensor
               for (size_t i = 0; i < total_tensors; i += MAX_CHUNK_SIZE) {
                   size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_tensors - i);
                   std::vector<OwnTensor::cuda::TensorInfo> chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
                   cuda::multi_tensor_grad_norm_cuda(chunk, s_d_norm);
               }
           }
       }
      
       // Compute clip coefficient on GPU (also computes sqrt for L2 norm)
       cuda::compute_clip_coef_cuda(s_d_norm, s_d_clip_coef, max_norm, is_inf_norm);
      
       // Phase 2: Scale Gradients (Chunked)
       if (total_tensors > 0) {
            for (size_t i = 0; i < total_tensors; i += MAX_CHUNK_SIZE) {
               size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_tensors - i);
               std::vector<OwnTensor::cuda::TensorInfo> chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
               cuda::multi_tensor_scale_cuda(chunk, s_d_clip_coef);
           }
       }
      
       // Only now copy the total_norm to CPU for return value
       float total_norm;
       cudaMemcpy(&total_norm, s_d_norm, sizeof(float), cudaMemcpyDeviceToHost);
      
       // No cudaFree - buffers are persistent

       if (error_if_nonfinite && (std::isnan(total_norm) || std::isinf(total_norm))) {
            throw std::runtime_error("The total norm of gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`");
       }
      
       return total_norm;
   } else {
       // CPU PATH: Simple loop
       float total_norm_sq = 0.0f;
       float total_norm_inf = 0.0f;
      
       for (auto& param : params) {
           if (!param.requires_grad() || !param.has_grad()) continue;
          
           try {
               Tensor grad = param.grad_view();
               const float* data = grad.data<float>();
               int64_t n = grad.numel();
              
               if (is_inf_norm) {
                   for (int64_t i = 0; i < n; ++i) {
                       float val = std::abs(data[i]);
                       if (val > total_norm_inf) total_norm_inf = val;
                   }
               } else {
                   for (int64_t i = 0; i < n; ++i) {
                       total_norm_sq += data[i] * data[i];
                   }
               }
           } catch (...) {
               continue;
           }
       }
      
       float total_norm;
       if (is_inf_norm) {
           total_norm = total_norm_inf;
       } else {
           total_norm = std::sqrt(total_norm_sq);
       }


       if (error_if_nonfinite && (std::isnan(total_norm) || std::isinf(total_norm))) {
            throw std::runtime_error("The total norm of gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`");
       }
      
       float clip_coef = max_norm / (total_norm + 1e-6f);
       if (clip_coef < 1.0f) {
           for (auto& param : params) {
               if (!param.requires_grad() || !param.has_grad()) continue;
              
               try {
                   Tensor grad = param.grad_view();
                   float* data = grad.data<float>();
                   int64_t n = grad.numel();
                   for (int64_t i = 0; i < n; ++i) {
                       data[i] *= clip_coef;
                   }
               } catch (...) {
                   continue;
               }
           }
       }
       return total_norm;
   }
}


// *********************************************************************************************
// ============================ AdamW Optimizer ================================================
// *********************************************************************************************

AdamW::AdamW(const std::vector<Tensor>& params,
         float lr,
         float beta1,
         float beta2,
         float eps,
         float weight_decay)
  : Optimizer(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
    weight_decay_(weight_decay), step_count_(0),
    initialized_(false) {}

void AdamW::step() {
  step_count_++;
  TRACK_ALLOC_SCOPE("L601:nn::AdamW::step");
  // Lazy initialization of momentum buffers — always FP32
  if (!initialized_) {
      m_.reserve(params_.size());
      v_.reserve(params_.size());
      for (auto& param : params_) {
          TensorOptions opts = TensorOptions()
              //.with_dtype(param.dtype())
              .with_dtype(Dtype::Float32)
              .with_device(param.device());
          m_.push_back(Tensor::zeros(param.shape(), opts));
          v_.push_back(Tensor::zeros(param.shape(), opts));
      }
      initialized_ = true;
  }

  // Bias correction factors
  float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
  float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
 
  // Multi-tensor optimization for AdamW
   std::vector<OwnTensor::cuda::TensorInfo> gpu_params_info;
   std::vector<OwnTensor::cuda::TensorInfo> gpu_grads_info;
   std::vector<OwnTensor::cuda::TensorInfo> gpu_m_info;
   std::vector<OwnTensor::cuda::TensorInfo> gpu_v_info;

   // Track which GPU params need cast-back(new)
   std::vector<std::pair<size_t, Dtype>> gpu_cast_back;

   gpu_params_info.reserve(params_.size());
   gpu_grads_info.reserve(params_.size());
   gpu_m_info.reserve(params_.size());
   gpu_v_info.reserve(params_.size());
  
   static const size_t MAX_CHUNK_SIZE = 512;

   for (size_t i = 0; i < params_.size(); ++i) {
      Tensor& param = params_[i];
      if (!param.requires_grad() || !param.has_grad()) {
          continue;
      }
      Tensor grad;
      try{
          grad = param.grad_view();
      } catch(...){
          continue;
      }

      // Ensure grad is FP32(new)
      if (grad.dtype() != Dtype::Float32) {
          grad = grad.as_type(Dtype::Float32);
      }
      
      // Unscale gradient using attached LossScaler
      if (scaler_) {
          grad *= (1.0f / scaler_->scale());
      }

      // Mixed precision: use FP32 master weight if available(new)
      Tensor* master = get_master_weight(param);
      Dtype orig_dtype = param.dtype();
      bool needs_cast_back = (master != nullptr);
      Tensor& work = master ? *master : param;

      int64_t numel = param.numel();
   
      // CPU implementation with direct data manipulation
      if (param.device().is_cpu()) {
          //float* param_data = param.data<float>();
          //const float* grad_data = param.grad<float>();
          float* param_data = work.data<float>();
          const float* grad_data = grad.data<float>();
          float* m_data = m_[i].data<float>();
          float* v_data = v_[i].data<float>();
       
          for (int64_t j = 0; j < numel; ++j) {
              float g = grad_data[j];
           
               // Apply weight decay (Decoupled weight decay) on FP32 master
               if (weight_decay_ > 0.0f) {
                   param_data[j] *= (1.0f - lr_ * weight_decay_);
               }
           
              // Update first moment: m = beta1 * m + (1.0f - beta1) * g
              m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
           
              // Update second moment: v = beta2 * v + (1.0f - beta2) * g^2
              v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
           
              // Bias-corrected estimates
              float m_hat = m_data[j] / bias_correction1;
              float v_hat = v_data[j] / bias_correction2;
           
              // Update parameter (in FP32 master)
              param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
          }

          // Cast back: FP32 master → original dtype param(new)
          if (needs_cast_back) {
              param.copy_(work.as_type(orig_dtype));
          }
      } else {
          // Collect for Multi-Tensor Update — use master weight's FP32 data
          //gpu_params_info.push_back({param.data<float>(), numel});
          //gpu_grads_info.push_back({param.grad<float>(), numel});
          gpu_params_info.push_back({work.data<float>(), numel});
          gpu_grads_info.push_back({grad.data<float>(), numel});
          gpu_m_info.push_back({m_[i].data<float>(), numel});
          gpu_v_info.push_back({v_[i].data<float>(), numel});

          if (needs_cast_back) { //new
              gpu_cast_back.push_back({i, orig_dtype});
          }
      }
  }
 
   // Process GPU tensors in chunks (AdamW shares the same kernel signature)
   size_t total_gpu_tensors = gpu_params_info.size();
   if (total_gpu_tensors > 0) {
       for (size_t i = 0; i < total_gpu_tensors; i += MAX_CHUNK_SIZE) {
           size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_gpu_tensors - i);

           std::vector<OwnTensor::cuda::TensorInfo> p_chunk(gpu_params_info.begin() + i, gpu_params_info.begin() + i + chunk_size);
           std::vector<OwnTensor::cuda::TensorInfo> g_chunk(gpu_grads_info.begin() + i, gpu_grads_info.begin() + i + chunk_size);
           std::vector<OwnTensor::cuda::TensorInfo> m_chunk(gpu_m_info.begin() + i, gpu_m_info.begin() + i + chunk_size);
           std::vector<OwnTensor::cuda::TensorInfo> v_chunk(gpu_v_info.begin() + i, gpu_v_info.begin() + i + chunk_size);

           cuda::multi_tensor_adam_cuda(
               p_chunk,
               g_chunk,
               m_chunk,
               v_chunk,
               lr_,
               beta1_,
               beta2_,
               eps_,
               weight_decay_,
               bias_correction1,
               bias_correction2,
               true   // is_adamw = true for AdamW
           );
       }

       // Cast back GPU params: FP32 master to original dtype param
       for (auto& [param_idx, dtype] : gpu_cast_back) {
           Tensor* m = get_master_weight(params_[param_idx]);
           if (m) {
               params_[param_idx].copy_(m->as_type(dtype));
           }
       }
   }
}

void AdamW::save_state(std::ostream& os) {
    os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));
    int count = static_cast<int>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(int));
    for (const auto& t : m_) save_tensor(t, os);
    for (const auto& t : v_) save_tensor(t, os);
}

void AdamW::load_state(std::istream& is) {
    is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
    int count;
    is.read(reinterpret_cast<char*>(&count), sizeof(int));
    
    // Lazy init if needed
    if (!initialized_) {
        m_.resize(count);
        v_.resize(count);
        initialized_ = true;
    }

    if (count != static_cast<int>(m_.size())) {
        throw std::runtime_error("AdamW state count mismatch!");
    }

    for (int i = 0; i < count; ++i) m_[i] = load_tensor(is).to(params_[i].device());
    for (int i = 0; i < count; ++i) v_[i] = load_tensor(is).to(params_[i].device());
}

} // namespace nn
} // namespace OwnTensor