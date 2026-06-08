#pragma once
#include <vector>
#include <cstdint>

namespace OwnTensor {
namespace cuda {

struct TensorInfo {
   float* ptr;
   int64_t numel;
};

// Dtype-agnostic buffer descriptor — used by multi_tensor_zero so it works for
// any tensor type (FP32 / FP16 / BF16 / int / complex / etc.). Zero is the same
// bit pattern for every numeric type, so the kernel just writes zero bytes.
struct ZeroTensorInfo {
   void*   ptr;
   int64_t nbytes;
};

/**
* @brief Compute total L2 norm squared of multiple tensors
*/
void multi_tensor_grad_norm_cuda(
   const std::vector<TensorInfo>& tensors,
   float* norm_sq_accumulator
);

/**
* @brief Scale multiple tensors by a single coefficient stored on GPU
*/
void multi_tensor_scale_cuda(
   const std::vector<TensorInfo>& tensors,
   const float* clip_coef
);

/**
* @brief Fused Adam update for multiple tensors
*/
void multi_tensor_adam_cuda(
   const std::vector<TensorInfo>& params,
   const std::vector<TensorInfo>& grads,
   const std::vector<TensorInfo>& ms,
   const std::vector<TensorInfo>& vs,
   float lr,
   float beta1,
   float beta2,
   float eps,
   float weight_decay,
   float bias_correction1,
   float bias_correction2,
   bool is_adamw = false
);

void multi_tensor_adam_sm89_cuda(
    const std::vector<TensorInfo>& params,
    const std::vector<TensorInfo>& grads,
    const std::vector<TensorInfo>& ms,
    const std::vector<TensorInfo>& vs,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2, bool is_adamw
);

void multi_tensor_scale_sm89_cuda(
    const std::vector<TensorInfo>& tensors, const float* clip_coef
);

/**
* @brief Zero multiple tensors in a single multi-tensor launch (sm89 fast path).
*
* Dtype-agnostic: each tensor is described by (void* ptr, int64_t nbytes).
* Kernel writes zero bytes via uint4 stores (STG.128 on sm_89). Works for any
* numeric dtype because bit-pattern 0 == numeric 0 for every floating, integer,
* and complex type we use.
*
* Each CUDA block owns one chunk of one tensor; up to 48 tensors and 320 blocks
* are dispatched per launch. Replaces the naive per-parameter cudaMemsetAsync
* loop used by the old Optimizer::zero_grad path.
*/
void multi_tensor_zero_sm89_cuda(
    const std::vector<ZeroTensorInfo>& tensors
);

/**
* @brief Portable multi-tensor zero. Forwards to the sm89 fast path.
*/
void multi_tensor_zero_cuda(
    const std::vector<ZeroTensorInfo>& tensors
);

} // namespace cuda
} // namespace OwnTensor