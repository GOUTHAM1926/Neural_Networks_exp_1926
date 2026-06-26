#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include "TensorLib.h"
#include <memory>
#include "nn/optimizer/Optim.h"
#include <thread>
#include <mpi.h>
#include "Error_logs.hpp"
#include "DataParallel.hpp"
#include "device/DeviceTransfer.h"
#include "ops/helpers/MultiTensorKernels.h"
#include "ops/helpers/GradNormKernels.h"
#include "profiler.hpp"

/*
    ZeRO Redundancy Optimizer (Stage 1):

        Implements ZeRO-1 from DeepSpeed: https://arxiv.org/pdf/1910.02054

    Implementation Details:

        Initialization:
            1. Flatten all model parameters into a single contiguous FP32 buffer.
            2. Pad the buffer so total elements are divisible by world_size.
            3. Partition: each rank owns [partition_start, partition_start + partition_size).
            4. Create FP32 master copy of this rank's partition.
            5. Create optimizer states (m, v) ONLY for this rank's partition (1/N memory saving).

        Each Step:
            1. Flatten all gradients into a contiguous buffer.
            2. AllReduce gradients (average across all ranks).
            3. Gradient clipping on globally-averaged gradients (correct norm).
            4. Extract this rank's gradient partition.
            5. Run AdamW update on the partitioned master weights using partitioned m, v.
            6. AllGather updated partitions to reconstruct the full parameter buffer.
            7. Copy updated flat buffer back into original model parameter tensors.

        Communication:
            - AllReduce for gradients (one call on the entire flat gradient buffer).
            - AllGather for updated parameters after the local optimizer step.

        Memory per rank:
            - Full parameters (FP32):          N elements
            - Full gradients (FP32):           N elements  (persistent, reused each step)
            - Partitioned master weights:      N / world_size elements
            - Partitioned m (first moment):    N / world_size elements
            - Partitioned v (second moment):   N / world_size elements
            Total optimizer state memory = 3 * N / world_size  (vs 3 * N without ZeRO)

*/

// first support to adamW.
template <typename Optimizer = OwnTensor::nn::AdamW>
struct zero_ops{
    float learning_rate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
    bool overlap_with_ddp = false;
    bool use_ddp_bucket = false;
    std::shared_ptr<ProcessGroupNCCL> pg_;

    zero_ops(float learning_rate = 0.0f, std::shared_ptr<ProcessGroupNCCL> pg = nullptr, bool overlap_with_ddp = false, bool use_ddp_bucket = false)
        : learning_rate(learning_rate), pg_(pg), overlap_with_ddp(overlap_with_ddp), use_ddp_bucket(use_ddp_bucket) {}

    zero_ops with_weight_decay(float wd){
        auto opts = *this;
        opts.weight_decay = wd;
        return opts;
    }

    zero_ops with_beta1(float b1){
        auto opts = *this;
        opts.beta1 = b1;
        return opts;
    }

    zero_ops with_beta2(float b2){
        auto opts = *this;
        opts.beta2 = b2;
        return opts;
    }

    zero_ops with_eps(float e){
        auto opts = *this;
        opts.eps = e;
        return opts;
    }
};

template struct zero_ops<OwnTensor::nn::Adam>;
template struct zero_ops<OwnTensor::nn::AdamW>;
template struct zero_ops<OwnTensor::nn::SGDOptimizer>;


inline bool DEBUG_MODE = false;
inline bool DEBUG_LOG = false;


template <typename Optimizer = OwnTensor::nn::AdamW>
class ZeROOptimizer : public OwnTensor::nn::Optimizer{
    public:
        ZeROOptimizer(std::vector<OwnTensor::Tensor>& model_parameters,
                        std::shared_ptr<ProcessGroupNCCL>& pg,
                        float learning_rate = 0.0f,
                        float beta1 = 0.9f,
                        float beta2 = 0.99f,
                        float eps = 1e-8f,
                        float weight_decay = 0.0f,
                        bool overlap_with_ddp = false,
                        bool ddp_bucket_params = false
                    )
            : OwnTensor::nn::Optimizer(model_parameters),
              model_parameters_(model_parameters),
              learning_rate_(learning_rate),
              beta1_(beta1),
              beta2_(beta2),
              eps_(eps),
              weight_decay_(weight_decay),
              pg_(pg),
              overlap_with_ddp_(overlap_with_ddp),
              ddp_bucket_params_(ddp_bucket_params)
        {
            optimizer_states_initialization();
	    const char* env = std::getenv("DEBUG_MODE");
	    if(env != nullptr){
		    if(std::string(env) == "true") DEBUG_MODE = true;
		    else if(std::string(env) == "LOG"){
			    DEBUG_LOG = true;
			    int r = static_cast<int>(pg_->get_rank());
			    if(r == 0){
				    log_csv = LOG_CSV("logs/performance_log_zero1.csv", "logs/trace_zero1.json", r);
			    }
		    }else if(std::string(env) == "false"){
			    DEBUG_MODE = false;
		    }
	    }
        }

        ZeROOptimizer(std::vector<OwnTensor::Tensor>& model_parameters,
                zero_ops<Optimizer>& ops
            )
            : OwnTensor::nn::Optimizer(model_parameters),
              model_parameters_(model_parameters),
              learning_rate_(ops.learning_rate),
              beta1_(ops.beta1),
              beta2_(ops.beta2),
              eps_(ops.eps),
              weight_decay_(ops.weight_decay),
              pg_(ops.pg_),
              overlap_with_ddp_(ops.overlap_with_ddp),
              ddp_bucket_params_(ops.use_ddp_bucket)
        {
            optimizer_states_initialization();
	    const char* env = std::getenv("DEBUG_MODE");
	    if(env != nullptr){
		    if(std::string(env) == "true") DEBUG_MODE = true;
		    else if(std::string(env) == "LOG"){
			    DEBUG_LOG = true;
			    int r = static_cast<int>(pg_->get_rank());
			    if(r == 0){
				    log_csv = LOG_CSV("logs/performance_log_zero1.csv", "logs/trace_zero1.json", r);
			    }
		    }else if(std::string(env) == "false"){
			    DEBUG_MODE = false;
		    }
	    }

        }

        ~ZeROOptimizer() {
            if (d_norm_sq_) cudaFree(d_norm_sq_);
            if (d_clip_coef_) cudaFree(d_clip_coef_);
            if (h_norm_pinned_) cudaFreeHost(h_norm_pinned_);
        }


        // =====================================================================
        // optimizer_states_initialization():
        //   Flatten parameters, partition across ranks, create FP32 master
        //   copy and optimizer states (m, v) for this rank's partition only.
        // =====================================================================
        void optimizer_states_initialization() {
	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
	    }
            int world_size = pg_->get_worldsize();
            int rank = pg_->get_rank();

            // Record original parameter shapes and sizes
            total_elements_ = 0;
            for (size_t i = 0; i < model_parameters_.size(); i++) {
                original_shapes_.push_back(model_parameters_[i].shape());
                original_numels_.push_back(model_parameters_[i].numel());
                total_elements_ += model_parameters_[i].numel();
            }

            // Pad total elements to be evenly divisible by world_size
            padded_total_ = total_elements_;
            if (padded_total_ % world_size != 0) {
                padded_total_ += (world_size - (padded_total_ % world_size));
            }

            partition_size_ = padded_total_ / world_size;
            partition_start_ = rank * partition_size_;

            // Get device from first parameter
            auto device = model_parameters_[0].device();
            auto fp32_opts = OwnTensor::TensorOptions()
                .with_dtype(OwnTensor::Dtype::Float32)
                .with_device(device);

            // --- Step 1: Flatten all parameters into a contiguous FP32 buffer ---
            OwnTensor::Tensor flat_unpadded = OwnTensor::Tensor::flatten_concat(model_parameters_);

            // Ensure FP32
            if (flat_unpadded.dtype() != OwnTensor::Dtype::Float32) {
                flat_unpadded = flat_unpadded.as_type(OwnTensor::Dtype::Float32);
            }

            // Allocate padded flat buffer (zeros fill padding region)
            if (padded_total_ > total_elements_) {
                flat_params_ = OwnTensor::Tensor::zeros(
                    OwnTensor::Shape{{1, static_cast<int64_t>(padded_total_)}}, fp32_opts);
                // Copy unpadded data into the beginning of the padded buffer
                OwnTensor::device::copy_memory(
                    flat_params_.data(), device.device,
                    flat_unpadded.data(), device.device,
                    total_elements_ * OwnTensor::Tensor::dtype_size(OwnTensor::Dtype::Float32));
            } else {
                flat_params_ = flat_unpadded;
            }

            // --- Step 2: Extract this rank's partition as FP32 master copy ---
            // slice() creates a deep copy of the partition
            partition_fp32_ = flat_params_.slice(partition_start_, partition_size_);
            if (partition_fp32_.dtype() != OwnTensor::Dtype::Float32) {
                partition_fp32_ = partition_fp32_.as_type(OwnTensor::Dtype::Float32);
            }

            // --- Step 3: Pre-allocate gradient buffers (reused every step) ---
            flat_grads_ = OwnTensor::Tensor::zeros(
                OwnTensor::Shape{{1, static_cast<int64_t>(padded_total_)}}, fp32_opts);

            partition_grads_ = OwnTensor::Tensor::zeros(
                OwnTensor::Shape{{1, static_cast<int64_t>(partition_size_)}}, fp32_opts);

            // --- Step 4: Create optimizer states for this rank's partition only ---
            // This is the core memory saving of ZeRO-1:
            // Each rank stores m and v for only 1/world_size of the parameters.
            partition_m_ = OwnTensor::Tensor::zeros(
                OwnTensor::Shape{{1, static_cast<int64_t>(partition_size_)}}, fp32_opts);
            partition_v_ = OwnTensor::Tensor::zeros(
                OwnTensor::Shape{{1, static_cast<int64_t>(partition_size_)}}, fp32_opts);

            // --- Step 5: Allocate persistent GPU scalars for gradient clipping ---
            cudaMalloc(&d_norm_sq_, sizeof(float));
            cudaMalloc(&d_clip_coef_, sizeof(float));
            cudaMallocHost(&h_norm_pinned_, sizeof(float));
            *h_norm_pinned_ = 0.0f;

            if (rank == 0) {
                std::cout << "[ZeRO-1] Initialized:" << std::endl;
                std::cout << "  Total parameters: " << total_elements_ << std::endl;
                std::cout << "  Padded total: " << padded_total_ << std::endl;
                std::cout << "  Partition size per rank: " << partition_size_ << std::endl;
                std::cout << "  World size: " << world_size << std::endl;
                std::cout << "  Optimizer states memory per rank: "
                          << (partition_size_ * sizeof(float) * 2) / (1024.0 * 1024.0)
                          << " MB (m + v)" << std::endl;
                std::cout << "  Full optimizer states (without ZeRO): "
                          << (total_elements_ * sizeof(float) * 2) / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0){
			    log_csv.stop_logging(__FUNCTION__);
		    }
	    }
        }


        // =====================================================================
        // step():
        //   1. Flatten gradients
        //   2. AllReduce gradients (average)
        //   3. Clip globally-averaged gradients (correct norm)
        //   4. Extract this rank's gradient partition
        //   5. Local AdamW step on partition
        //   6. AllGather updated partitions
        //   7. Copy back to original model parameters
        // =====================================================================
        void step() override {
    	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
	    }

            step_count_++;

            auto device = model_parameters_[0].device();
            size_t elem_size = OwnTensor::Tensor::dtype_size(OwnTensor::Dtype::Float32);

            // ---- Step 1: Flatten all gradients into the pre-allocated flat buffer ----
            {
                size_t offset = 0;
                for (size_t i = 0; i < model_parameters_.size(); i++) {
                    void* dst = static_cast<uint8_t*>(flat_grads_.data()) + offset * elem_size;

                    if (model_parameters_[i].has_grad()) {
                        OwnTensor::Tensor grad = model_parameters_[i].grad_view();
                        // Ensure FP32
                        if (grad.dtype() != OwnTensor::Dtype::Float32) {
                            grad = grad.as_type(OwnTensor::Dtype::Float32);
                        }
                        OwnTensor::device::copy_memory(
                            dst, device.device,
                            grad.data(), device.device,
                            original_numels_[i] * elem_size);
                    } else {
                        // No gradient for this parameter: zero out its region
                        cudaMemsetAsync(dst, 0, original_numels_[i] * elem_size,
                                        OwnTensor::cuda::getCurrentStream());
                    }
                    offset += original_numels_[i];
                }

                // Zero out padding region if any
                if (padded_total_ > total_elements_) {
                    void* pad_dst = static_cast<uint8_t*>(flat_grads_.data()) + total_elements_ * elem_size;
                    cudaMemsetAsync(pad_dst, 0, (padded_total_ - total_elements_) * elem_size,
                                    OwnTensor::cuda::getCurrentStream());
                }
            }

            // ---- Step 2: AllReduce gradients across all ranks (average) ----
            pg_->all_reduce(
                flat_grads_.data(), flat_grads_.data(),
                padded_total_, OwnTensor::Dtype::Float32, avg, true);

            // ---- Step 3: Gradient clipping on globally-averaged gradients ----
            // After AllReduce, all ranks have identical averaged gradients.
            // Clip the FULL averaged gradient buffer so the norm is correct.
            if (max_grad_norm_ > 0.0f) {
                clip_grads();
            }

            // ---- Step 4: Extract this rank's gradient partition ----
            {
                void* src = static_cast<uint8_t*>(flat_grads_.data()) + partition_start_ * elem_size;
                OwnTensor::device::copy_memory(
                    partition_grads_.data(), device.device,
                    src, device.device,
                    partition_size_ * elem_size);
            }

            // ---- Step 5: Local AdamW step on this rank's partition ----
            local_step();

            // ---- Step 6: AllGather updated partitions to reconstruct full parameters ----
            all_gather_params();

            // ---- Step 7: Copy updated flat buffer back into original model parameters ----
            {
                size_t offset = 0;
                for (size_t i = 0; i < model_parameters_.size(); i++) {
                    void* src = static_cast<uint8_t*>(flat_params_.data()) + offset * elem_size;
                    OwnTensor::device::copy_memory(
                        model_parameters_[i].data(), device.device,
                        src, device.device,
                        original_numels_[i] * elem_size);
                    offset += original_numels_[i];
                }
            }
    	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
	    }

        }


        // =====================================================================
        // clip_grads():
        //   Gradient clipping on the globally-averaged flat gradient buffer.
        //   Uses the same multi_tensor kernels as nn::clip_grad_norm_.
        //   Computes norm on the full flat_grads_ buffer (all ranks have
        //   identical data after AllReduce), so no extra AllReduce needed.
        // =====================================================================
        void clip_grads() {
	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
	    }

            // 1. Compute L2 norm squared of full averaged gradient buffer
            cudaMemsetAsync(d_norm_sq_, 0, sizeof(float));

            // Collect gradient info — use all model param gradient regions
            // from the flat buffer (skip padding, it's zero anyway)
            std::vector<OwnTensor::cuda::TensorInfo> grad_info = {
                {static_cast<float*>(flat_grads_.data()), static_cast<int64_t>(total_elements_)}
            };
            OwnTensor::cuda::multi_tensor_grad_norm_cuda(grad_info, d_norm_sq_);

            // 2. Compute clip coefficient (also stores total_norm back into d_norm_sq_)
            OwnTensor::cuda::compute_clip_coef_cuda(
                d_norm_sq_, d_clip_coef_, max_grad_norm_, false);

            // 3. Scale the FULL flat gradient buffer by clip coefficient
            OwnTensor::cuda::multi_tensor_scale_cuda(grad_info, d_clip_coef_);

            // 4. Async D2H of norm for get_grad_norm()
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cudaMemcpyAsync(h_norm_pinned_, d_norm_sq_, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            last_grad_norm_ = *h_norm_pinned_;
    	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
	    }

        }


        // =====================================================================
        // local_step():
        //   Run fused AdamW kernel on this rank's partition only.
        //   Uses the multi_tensor_adam_cuda kernel for GPU efficiency.
        //   Operates on:
        //     - partition_fp32_ (FP32 master weights for this partition)
        //     - partition_grads_ (averaged gradient for this partition)
        //     - partition_m_ (first moment / momentum)
        //     - partition_v_ (second moment / variance)
        // =====================================================================
        void local_step() {
	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
	    }

            float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
            float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

            // Single "tensor" covering the entire partition
            std::vector<OwnTensor::cuda::TensorInfo> p_info = {
                {partition_fp32_.data<float>(), static_cast<int64_t>(partition_size_)}
            };
            std::vector<OwnTensor::cuda::TensorInfo> g_info = {
                {partition_grads_.data<float>(), static_cast<int64_t>(partition_size_)}
            };
            std::vector<OwnTensor::cuda::TensorInfo> m_info = {
                {partition_m_.data<float>(), static_cast<int64_t>(partition_size_)}
            };
            std::vector<OwnTensor::cuda::TensorInfo> v_info = {
                {partition_v_.data<float>(), static_cast<int64_t>(partition_size_)}
            };

            OwnTensor::cuda::multi_tensor_adam_cuda(
                p_info, g_info, m_info, v_info,
                learning_rate_,
                beta1_, beta2_, eps_, weight_decay_,
                bias_correction1, bias_correction2,
                true  // is_adamw = true (decoupled weight decay)
            );
    	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
	    }

        }


        // =====================================================================
        // all_gather_params():
        //   AllGather: each rank sends its updated partition (partition_fp32_),
        //   receives all world_size partitions into flat_params_.
        //   After this call, flat_params_ contains the fully updated parameters.
        // =====================================================================
        void all_gather_params() {
	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
	    }

            pg_->all_gather(
                partition_fp32_.data(),   // sendbuff: this rank's updated partition
                flat_params_.data(),      // recvbuff: full flat buffer
                partition_size_,          // sendcount: elements per rank
                OwnTensor::Dtype::Float32,
                true                      // synchronous
            );
    	    if(DEBUG_LOG){
		    if(pg_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
	    }

        }


        float get_lr() { return learning_rate_; }

        void set_lr(float lr) override { learning_rate_ = lr; }

        void set_max_grad_norm(float max_norm) { max_grad_norm_ = max_norm; }

        // Returns the gradient norm computed during the last step().
        // This is the L2 norm of the globally-averaged gradients BEFORE clipping.
        float get_grad_norm() {
            return last_grad_norm_;
        }


    protected:
        struct Bucket_Params{
            std::vector<int> offset;
            std::vector<int> length;
            std::vector<OwnTensor::Tensor> params;
            OwnTensor::Tensor flatten_parameter;
            std::vector<OwnTensor::Tensor> views_in;
            std::vector<OwnTensor::Tensor> views_out;
        };


    private:
        // Reference to the original model parameter tensors
        std::vector<OwnTensor::Tensor> model_parameters_;

	LOG_CSV log_csv;
        // Hyperparameters
        float learning_rate_;
        float beta1_;
        float beta2_;
        float eps_;
        float weight_decay_;
        int64_t step_count_ = 0;

        // Communication
        std::shared_ptr<ProcessGroupNCCL> pg_;
        bool overlap_with_ddp_ = false;
        bool ddp_bucket_params_ = false;

        // ---- ZeRO-1 state ----

        // Total number of elements across all parameters (before and after padding)
        size_t total_elements_ = 0;
        size_t padded_total_ = 0;

        // This rank's partition bounds
        size_t partition_size_ = 0;
        size_t partition_start_ = 0;

        // [1, padded_total] - Full flattened parameter buffer (used as AllGather recv buffer)
        OwnTensor::Tensor flat_params_;

        // [1, padded_total] - Full flattened gradient buffer (reused each step)
        OwnTensor::Tensor flat_grads_;

        // [1, partition_size] - FP32 master copy of this rank's parameter partition
        OwnTensor::Tensor partition_fp32_;

        // [1, partition_size] - This rank's gradient partition (extracted from flat_grads_)
        OwnTensor::Tensor partition_grads_;

        // [1, partition_size] - First moment estimate (momentum) for this rank's partition
        OwnTensor::Tensor partition_m_;

        // [1, partition_size] - Second moment estimate (variance) for this rank's partition
        OwnTensor::Tensor partition_v_;

        // Original parameter metadata for unflattening
        std::vector<OwnTensor::Shape> original_shapes_;
        std::vector<size_t> original_numels_;

        // ---- Gradient clipping state ----
        float max_grad_norm_ = 0.0f;
        float last_grad_norm_ = 0.0f;

        // Persistent GPU scalars (allocated once, tiny)
        float* d_norm_sq_ = nullptr;
        float* d_clip_coef_ = nullptr;
        float* h_norm_pinned_ = nullptr;

};

// Explicit instantiation declarations (definitions in ZeRORedundancyOptimizer.cpp)
extern template class ZeROOptimizer<OwnTensor::nn::AdamW>;
extern template class ZeROOptimizer<OwnTensor::nn::Adam>;
extern template class ZeROOptimizer<OwnTensor::nn::SGDOptimizer>;
