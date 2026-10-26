#include "device/CachingCudaAllocator.h"
#include "device/CudaKernelScope.h"
#include "dtype/Types.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/helpers/LossKernels.h"
#include "profiler/RecordFunction.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

namespace OwnTensor {
namespace cuda {

// SM103 (B300) sparse-CE forward — save-stats per-row kernel launcher.
// Defined in Kernels/cuda/loss/arch/CrossEntropyKernels_sm103.cu.
// Runs only the loss+(max,sum) kernel; the shared sum_reduction runs in the
// caller.
template <typename T, typename T_idx>
void sparse_ce_fwd_sm103_launch(const T *logits, const T_idx *targets,
                                float *losses, float *saved_max, float *saved_sum,
                                int64_t batch_size, int64_t vocab_size,
                                cudaStream_t stream);

// ============================================================================
// Forward Pass CUDA Kernels
// ============================================================================

/*
 * ###################################################################################################################################
 * ####################################   Implementation of the two kernel
 * approch  [Vectoirzed]   ###################################
 * ###################################################################################################################################
 */
__device__ __forceinline__ void cp_async_16(void *smem_ptr,
                                            const void *glob_ptr) {
  // Converts generic shared pointer to a 32-bit segment-specific address for
  // PTX [6, 7]
  unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(glob_ptr));
}

// ============================================================================
// 16-BYTE VECTOR ACCESS FOR A TEMPLATED DTYPE
//
// WIDTH is the number of ELEMENTS in 16 bytes, so the byte size is constant:
//     float -> 4      float16_t / bfloat16_t -> 8      double -> 2
//
// Why it must be derived from sizeof(T) and not hardcoded to 4:
//   the per-thread byte offset is  tid * WIDTH * sizeof(T) == tid * 16, which is
//   exactly what `cp.async.cg.shared.global ... 16` and a 16-byte vector load
//   both require. Hardcoding 4 elements is correct ONLY for 4-byte T. For a
//   2-byte T it does two things wrong at once:
//     1. the address becomes tid*8, so odd threads issue a MISALIGNED cp.async
//        and the kernel dies with "misaligned address";
//     2. even if it were aligned, 16 bytes of bf16 is 8 values, and reading them
//        back out of a `float` staging buffer reinterprets them as 4 floats.
//
// Same idea as AddVecTraits in src/TensorOps/cuda/TensorOpsAdd.cu (VEC=4 for
// float, VEC=8 for half/bf16) and PyTorch's
//   template <typename scalar_t, int vec_size>
//   struct alignas(sizeof(scalar_t) * vec_size) aligned_vector { scalar_t val[vec_size]; };
// (aten/src/ATen/native/cuda/MemoryAccess.cuh:182).
//
// The staging region is unchanged in BYTES (bdim*WIDTH*sizeof(T) == bdim*16 ==
// bdim*4 floats), so every `smax = s_data + bdim*4` offset and every launch-side
// shared-memory size stays exactly as it was.
// ============================================================================
template <typename T> struct alignas(16) LossVec {
  static constexpr int WIDTH = 16 / sizeof(T);
  T v[WIDTH];
};

// The compiler checks the invariant for us: every instantiation is exactly 16 B.
static_assert(sizeof(LossVec<float>) == 16 && LossVec<float>::WIDTH == 4, "");
static_assert(sizeof(LossVec<double>) == 16 && LossVec<double>::WIDTH == 2, "");
static_assert(sizeof(LossVec<float16_t>) == 16 && LossVec<float16_t>::WIDTH == 8, "");
static_assert(sizeof(LossVec<bfloat16_t>) == 16 && LossVec<bfloat16_t>::WIDTH == 8, "");
static_assert(alignof(LossVec<bfloat16_t>) == 16, "16-B vector loads need 16-B alignment");

// ============================================================================
// SFT denominator: count the targets that are NOT ignore_index.
//
// This is PyTorch's `total_weight`. LossNLL.cpp:288 computes
//     total_weight_val = batch_size - num_ignored
// writes it out of the forward (:304), and the backward divides by that SAME
// scalar (:384, :394) -- on CUDA it stays a device pointer
// (Loss.cu:441 `const scalar_t *total_weight`, :449 `*grad_output / *total_weight`)
// so there is never a host sync. Liger uses the same quantity (`n_non_ignore`).
//
// Clamped to >= 1: an all-masked micro-batch would otherwise divide by zero.
// Megatron guards the identical case with torch.clamp(num_tokens, min=1)
// (pipeline_parallel/schedules.py:339).
//
// NOTE the invariant that makes this safe to switch on by default: with masking
// OFF (ignore_index = -1, uint16 targets) nothing can equal -1, so the count is
// exactly batch_size and every pretraining run is bit-identical to before.
// ============================================================================
template <typename T_idx>
__global__ void count_valid_targets_kernel(const T_idx *__restrict__ targets,
                                           int64_t batch_size,
                                           int64_t ignore_index,
                                           float *__restrict__ out_count) {
  extern __shared__ float s_cnt[];
  const int tid = threadIdx.x;
  float local = 0.0f;
  for (int64_t i = tid; i < batch_size; i += blockDim.x)
    if (static_cast<int64_t>(targets[i]) != ignore_index)
      local += 1.0f;
  s_cnt[tid] = local;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      s_cnt[tid] += s_cnt[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    *out_count = fmaxf(s_cnt[0], 1.0f);
}

template <typename T_idx>
void count_valid_targets_cuda(const T_idx *targets, int64_t batch_size,
                              int64_t ignore_index, float *out_count,
                              cudaStream_t stream) {
  CUDA_KERNEL_SCOPE("count_valid_targets_cuda", stream);
  // batch_size == 0 needs no special case: every thread contributes 0 and the
  // clamp turns the result into 1.0f.
  const int threads = 256; // power of two, required by the tree reduction
  count_valid_targets_kernel<T_idx>
      <<<1, threads, threads * sizeof(float), stream>>>(targets, batch_size,
                                                        ignore_index, out_count);
}

template void count_valid_targets_cuda<uint16_t>(const uint16_t *, int64_t,
                                                 int64_t, float *, cudaStream_t);
template void count_valid_targets_cuda<int32_t>(const int32_t *, int64_t,
                                                int64_t, float *, cudaStream_t);
template void count_valid_targets_cuda<int64_t>(const int64_t *, int64_t,
                                                int64_t, float *, cudaStream_t);

// Sum-reduction with a SEPARATE input and output type.
//
// TIn is the accumulator type the per-row losses are stored in (float), TOut is
// the type the final scalar is written as. Keeping them apart is what lets the
// whole reduction stay fp32 and cast to the caller's dtype EXACTLY ONCE, on the
// final value — the same structure PyTorch uses in
// aten/src/ATen/native/cuda/Loss.cu:266
//     *output = static_cast<scalar_t>(sh_inputs[0] / acc_weight[0]);
// where sh_inputs/acc_weight are accscalar_t (= float for half/bfloat16, see
// AccumulateType.h: CUDA_ACC_TYPE(BFloat16, float)).
//
// Casting at every stage instead quantises the running total: for 8192 rows at
// loss ~2.5 the sum is ~20480, where bf16's representable values are 128 apart,
// so the total lands up to 64 off — a 0.31% error carried into every gradient.
//
// TOut defaults to TIn so all existing sum_reduction_kernel<T> / <float> call
// sites keep compiling unchanged.
// `denom` (optional, device pointer) divides the FINAL sum. Passing it only to
// the last stage is what makes the forward's divisor and the backward's divisor
// the same device scalar -- the whole point of PyTorch's total_weight. Stage-1
// partials must NOT be divided, or the denominator is applied reduce_blocks
// times. nullptr => no division (every other loss in this file).
template <typename TIn, typename TOut = TIn>
__global__ void sum_reduction_kernel(const TIn *input, TOut *output, int64_t n,
                                     const float *denom = nullptr) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (i < n) ? static_cast<float>(input[i]) : 0.0f;
  __syncthreads();

// Reduction in shared memory
#pragma unroll 4
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block
  if (tid == 0) {
    const float v = (denom != nullptr) ? (sdata[0] / (*denom)) : sdata[0];
    output[blockIdx.x] = static_cast<TOut>(v);
  }
}

//! forward pass sparse
template <typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_vec(
    // losses is the fp32 ACCUMULATOR buffer, never T — see sum_reduction_kernel.
    const T *logits, const T_idx *targets, float *losses, int64_t batch_size,
    int64_t vocab_size,
    int64_t ignore_index = -1 // SFT: target value meaning "skip this row"

) {
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const int bid = blockIdx.x;

  extern __shared__ float s_data[];
  int64_t row = blockIdx.x;
  if (row >= batch_size)
    return;

  // SFT label masking: a row whose target is ignore_index contributes nothing
  // to the loss. Skipping here avoids the whole vocab-wide online softmax for
  // that row, so masking is a speedup rather than an overhead.
  // No saved_max/saved_sum in this kernel, so only losses[row] needs writing.
  if (static_cast<int64_t>(targets[row]) == ignore_index) {
    if (tid == 0)
      losses[row] = 0.0f;
    return;
  }
  const T *row_logits = logits + row * vocab_size;

  //* Online softmax
  float local_max = -1e38f;
  float local_sum = 0.0f;

  // 16-byte vectors; WIDTH elements each -- 4 for float, 8 for bf16/half, 2 for
  // double. See LossVec above for why this must come from sizeof(T).
  constexpr int VEC = LossVec<T>::WIDTH;
  const int64_t vec_count = vocab_size / VEC; // FULL vectors only
  // Stage as T, never as float: the bytes have to be read back as the type they
  // were written as. This occupies bdim*VEC*sizeof(T) == bdim*16 bytes, i.e. the
  // same bdim*4 floats that the smax/ssum offsets below already assume.
  T *s_stage = reinterpret_cast<T *>(s_data);
  //* Row must be 16-B aligned. With VEC derived from sizeof(T) the per-thread
  //* address is row_base + v*16, so checking the row base is sufficient for
  //* every dtype (it was NOT, back when the element count was hardcoded to 4).
  if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
    //* v indexes FULL vectors only, so nothing is read past the row end and the
    //* tail loop below owns the remainder exactly once -- no double counting.
    for (int64_t v = tid; v < vec_count; v += bdim) {
      cp_async_16(&s_stage[tid * VEC], &row_logits[v * VEC]);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
// No __syncthreads() here: each thread only reads its own
// s_stage[tid*VEC .. tid*VEC+VEC-1] slot.
// __syncthreads() inside a variable-iteration loop causes a deadlock when
// vec_count is not a multiple of bdim.
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        float val = static_cast<float>(s_stage[tid * VEC + k]);
        if (val > local_max) {
          local_sum = local_sum * expf(local_max - val) + 1.0f;
          local_max = val;
        } else {
          local_sum += expf(val - local_max);
        }
      }
    }
  } else {
    //* Fallback to scalar for unaligned rows
    for (int64_t j = threadIdx.x; j < vec_count * VEC; j += blockDim.x) {
      float val = static_cast<float>(row_logits[j]);
      if (val > local_max) {
        local_sum = local_sum * expf(local_max - val) + 1.0f;
        local_max = val;
      } else {
        local_sum += expf(val - local_max);
      }
    }
  }

  //* Handle tail elements for float type
  for (int64_t j = vec_count * VEC + threadIdx.x; j < vocab_size;
       j += blockDim.x) {
    float val = static_cast<float>(row_logits[j]);
    if (val > local_max) {
      local_sum = local_sum * expf(local_max - val) + 1.0f;
      local_max = val;
    } else {
      local_sum += expf(val - local_max);
    }
  }

  //* Block reduction in shared memory
  // smax/ssum are placed AFTER the staging buffer to avoid a race condition:
  // with s_data aliased to the same base, smax[128] = s_data[128] would
  // conflict with thread 32's in-flight cp.async DMA to s_data[128..131].
  float *smax = s_data + bdim * 4; // starts after staging (bdim*4 floats)
  float *ssum = s_data + bdim * 4 + bdim;

  smax[tid] = local_max;
  ssum[tid] = local_sum;
  __syncthreads(); //* Barrier synchronization

  //* Standard reduction loop
  for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float other_max = smax[tid + s];
      float other_sum = ssum[tid + s];

      //* Merge logic (Online softmax)
      if (other_max > smax[tid]) {
        ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
        smax[tid] = other_max;
      } else {
        ssum[tid] += other_sum * expf(other_max - smax[tid]);
      }
    }
    __syncthreads();
  }

  //* Now index 0 in the shared mem contains the block wide max and sum
  float final_max = smax[0];
  float final_sum = ssum[0];

  //* Final loss computation
  if (tid == 0) {
    int64_t target_idx = static_cast<int64_t>(targets[row]);
    float target_logit = static_cast<float>(row_logits[target_idx]);

    //* Loss = log(sum(exp(logits - max))) + max - target_logit
    float loss = logf(final_sum) + final_max - target_logit;
    losses[row] = loss;
  }
}

// =============================================================================
// Variant of sparse_ce_forward_kernel_vec that ALSO writes per-row (max, sum)
// to scratch tensors. The autograd node holds these and the backward pass uses
// them directly, skipping the redundant Reduce kernel that recomputes them.
// =============================================================================
template <typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_vec_save_stats(
    // losses is the fp32 ACCUMULATOR buffer, never T — see sum_reduction_kernel.
    const T *logits, const T_idx *targets, float *losses, float *saved_max,
    float *saved_sum, int64_t batch_size, int64_t vocab_size,
    int64_t ignore_index = -1 // SFT : target value meaning "skip this row"
) {
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  extern __shared__ float s_data[];
  int64_t row = blockIdx.x;
  if (row >= batch_size)
    return;
  // SFT label masking: this row contributes nothing to the loss, so skip the
  // whole vocab-wide online softmax for it.
  // saved_sum MUST be 1.0f, never 0.0f — the backward computes 1.0f/saved_sum,
  // and a zero there gives inf, then inf*0 = NaN which spreads through the
  // entire gradient tensor. 1.0f makes the discarded value harmless.
  if (static_cast<int64_t>(targets[row]) == ignore_index) {
    if (tid == 0) {
      losses[row] = 0.0f;
      saved_max[row] = 0.0f;
      saved_sum[row] = 1.0f;
    }
    return;
  }

  const T *row_logits = logits + row * vocab_size;
  float local_max = -1e38f;
  float local_sum = 0.0f;

  // 16-byte vectors; WIDTH elements each. See LossVec for why this is derived
  // from sizeof(T) rather than hardcoded to 4.
  constexpr int VEC = LossVec<T>::WIDTH;
  const int64_t vec_count = vocab_size / VEC; // FULL vectors only
  // Stage as T so the bytes are read back as the type they were written as.
  // bdim*VEC*sizeof(T) == bdim*16 bytes == the bdim*4 floats smax/ssum assume.
  T *s_stage = reinterpret_cast<T *>(s_data);
  //* Row-base check is sufficient once VEC comes from sizeof(T): the per-thread
  //* address is row_base + v*16 for every dtype.
  if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
    //* FULL vectors only -- the tail loop owns the remainder exactly once.
    for (int64_t v = tid; v < vec_count; v += bdim) {
      cp_async_16(&s_stage[tid * VEC], &row_logits[v * VEC]);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        float val = static_cast<float>(s_stage[tid * VEC + k]);
        if (val > local_max) {
          local_sum = local_sum * expf(local_max - val) + 1.0f;
          local_max = val;
        } else {
          local_sum += expf(val - local_max);
        }
      }
    }
  } else {
    for (int64_t j = threadIdx.x; j < vec_count * VEC; j += blockDim.x) {
      float val = static_cast<float>(row_logits[j]);
      if (val > local_max) {
        local_sum = local_sum * expf(local_max - val) + 1.0f;
        local_max = val;
      } else {
        local_sum += expf(val - local_max);
      }
    }
  }
  for (int64_t j = vec_count * VEC + threadIdx.x; j < vocab_size;
       j += blockDim.x) {
    float val = static_cast<float>(row_logits[j]);
    if (val > local_max) {
      local_sum = local_sum * expf(local_max - val) + 1.0f;
      local_max = val;
    } else {
      local_sum += expf(val - local_max);
    }
  }

  float *smax = s_data + bdim * 4;
  float *ssum = s_data + bdim * 4 + bdim;
  smax[tid] = local_max;
  ssum[tid] = local_sum;
  __syncthreads();

  for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float other_max = smax[tid + s];
      float other_sum = ssum[tid + s];
      if (other_max > smax[tid]) {
        ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
        smax[tid] = other_max;
      } else {
        ssum[tid] += other_sum * expf(other_max - smax[tid]);
      }
    }
    __syncthreads();
  }
  float final_max = smax[0];
  float final_sum = ssum[0];

  if (tid == 0) {
    saved_max[row] = final_max; // ← saved for backward
    saved_sum[row] = final_sum; // ← saved for backward
    int64_t target_idx = static_cast<int64_t>(targets[row]);
    float target_logit = static_cast<float>(row_logits[target_idx]);
    float loss = logf(final_sum) + final_max - target_logit;
    losses[row] = loss;
  }
}

template <typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec(
    const T *logits, const T_idx *targets, float *loss_output, int64_t batch_size,
    int64_t vocab_size, cudaStream_t stream, int64_t ignore_index,
    const float *denom) {
  CUDA_KERNEL_SCOPE("sparse_cross_entropy_forward_cuda_impl_vec", stream);
  if (batch_size == 0) {
    // Set loss to 0
    cudaMemsetAsync(loss_output, 0, sizeof(float), stream);
    return;
  }

  // Allocate temporary buffer for per-sample losses
  float *d_losses = static_cast<float *>(CachingCUDAAllocator::instance().allocate(
      batch_size * sizeof(float), stream));

  // Kernel 1: Compute loss for each sample
  int threads = 512; // 256
  int blocks = batch_size;
  // 6*threads: bdim*4 for staging buffer + bdim for smax + bdim for ssum
  size_t shared_mem = 6 * threads * sizeof(float);
  sparse_ce_forward_kernel_vec<T, T_idx>
      <<<blocks, threads, shared_mem, stream>>>(
          logits, targets, d_losses, batch_size, vocab_size, ignore_index);

  // Kernel 2: Reduce to sum all losses.
  // sum_reduction_kernel loads one element per thread, so a single block
  // of reduce_threads can only sum reduce_threads elements — not 1024.
  {
    int reduce_threads = 256;
    int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;

    if (reduce_blocks == 1) {
      sum_reduction_kernel<float, float>
          <<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
              d_losses, loss_output, batch_size, denom);
    } else {
      float *d_partial = static_cast<float *>(CachingCUDAAllocator::instance().allocate(
          reduce_blocks * sizeof(float), stream));

      sum_reduction_kernel<float, float>
          <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float),
             stream>>>(d_losses, d_partial, batch_size);
      sum_reduction_kernel<float, float>
          <<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
              d_partial, loss_output, reduce_blocks, denom);

      CachingCUDAAllocator::instance().deallocate(d_partial);
    }
  }

  CachingCUDAAllocator::instance().deallocate(d_losses);
}

/*
 * ###################################################################################################################################
 * ###################################################################################################################################
 * ###################################################################################################################################
 */

/**
 * @brief Compute sparse cross entropy loss for each sample (forward pass)
 *
 * Uses numerically stable log-softmax:
 * loss[i] = log(sum(exp(logits[i] - max(logits[i])))) + max(logits[i]) -
 * logits[i, target[i]]
 */

/**
 * @brief Parallel reduction to sum all losses
 * Uses shared memory for efficient reduction
 */




// ============================================================================
// Backward Pass CUDA Kernels (existing)
// ============================================================================

/**
 * @brief CUDA kernel for sparse cross entropy backward pass
 *
 * Gradient formula (derived from calculus):
 *   For class j:
 *     grad[j] = softmax(logits)[j] * scale           if j != target
 *     grad[j] = (softmax(logits)[j] - 1.0) * scale   if j == target
 *
 * This is equivalent to: grad = (softmax - target_indicator) * scale
 * where target_indicator equals 1 only at the target class index.
 *
 * @param logits     Input logits [batch_size, vocab_size]
 * @param targets    Sparse target indices [batch_size] (NOT one-hot!)
 * @param grad       Output gradient [batch_size, vocab_size]
 * @param batch_size Number of samples
 * @param vocab_size Number of classes
 * @param scale      Gradient scaling factor (typically 1/batch_size)
 */
/*
 * ###################################################################################################################################
 * ####################################   Implementation of the two kernel
 * approch  [Vectoirzed]   ###################################
 * ###################################################################################################################################
 */

template <typename T, typename T_idx>
__global__ void sparseCEReduce_kernel_optimized(
    const T *__restrict__ logits, const T_idx *__restrict__ targets, T *grad,
    float *d_partial_max, float *d_partial_sum, int64_t batch_size,
    int64_t vocab_size, int64_t ignore_index = -1) {
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const int xbid = blockIdx.x;
  const int ybid = blockIdx.y;
  const int xgdim = gridDim.x;
  const int bid = xbid + ybid * xgdim;

  const int row = ybid;
  if (row >= batch_size)
    return;
  // SFT label masking: skip the vocab-wide online softmax for ignored rows.
  // Both blocks of this row (xbid 0 and 1) run this and each writes its OWN
  // slot bid = xbid + row*xgdim, so no partial is left uninitialised.
  // d_partial_sum MUST be 1.0f, never 0.0f: Normalize merges the partials and
  // takes 1.0f/final_sum. Two zeros merge to 0 -> inf -> inf*0 = NaN across the
  // whole gradient. Two 1.0f merge to 2.0 -> inv_sum 0.5, finite and harmless
  // since Normalize zeroes this row anyway.
  if (static_cast<int64_t>(targets[row]) == ignore_index) {
    if (tid == 0) {
      d_partial_max[bid] = 0.0f;
      d_partial_sum[bid] = 1.0f;
    }
    return;
  }
  // Identify the segment of the row for this block
  int elementsPerBlock = vocab_size / xgdim;
  int startCol = xbid * elementsPerBlock;
  int endCol = startCol + elementsPerBlock;

  const T *row_logits = logits + row * vocab_size;

  // --- SHARED MEMORY SETUP ---
  // We need 2 sections: 1 for staging loads and 1 for the reduction stats
  extern __shared__ float sdata[];
  // Staged as T, not float: cp.async moves raw bytes, so they must be read back
  // as the type they were written as (bdim*VEC*sizeof(T) == bdim*16 bytes, the
  // same bdim*4 floats the smax/ssum offsets below assume). See LossVec.
  T *s_stage = reinterpret_cast<T *>(sdata);
  float *smax = sdata + (bdim * 4); // Reduction max (bdim floats)
  float *ssum = smax + bdim;        // Reduction sum (bdim floats)

  float local_max = -1e38f;
  float local_sum = 0.0f;

  // Vectorized loop: 16 bytes at a time == VEC elements (4 float / 8 bf16).
  // vec_end stops at the last FULL vector so nothing is read past endCol; the
  // remainder is picked up by the scalar tail below. The old loop ran to endCol
  // while unconditionally reading 4 slots, which both overran and (for 2-byte T)
  // misaligned the cp.async address.
  constexpr int VEC = LossVec<T>::WIDTH;
  const int64_t vec_end = startCol + ((endCol - startCol) / VEC) * VEC;
  for (int64_t col = startCol + tid * VEC; col < vec_end; col += bdim * VEC) {

    // 1. ASYNC LOAD: Initiate 128-bit transfer from Global to SMEM [2, 3, 9]
    // This bypasses L1 cache (.cg) to avoid pollution [10, 11]
    cp_async_16(&s_stage[tid * VEC], &row_logits[col]);

    // 2. COORDINATION: Commit the batch and wait for arrival [12-14]
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
// No __syncthreads() here: each thread reads only its own
// s_stage[tid*4..tid*4+3] slot.
// __syncthreads() inside a variable-iteration loop causes deadlock when
// elementsPerBlock is not a multiple of bdim*4 (e.g. vocab_size=50304,
// bdim=256).

// 3. COMPUTE: Perform Online Softmax math from the staging buffer
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float val = static_cast<float>(s_stage[tid * VEC + k]);
      if (val > local_max) {
        local_sum = local_sum * expf(local_max - val) + 1.0f;
        local_max = val;
      } else {
        local_sum += expf(val - local_max);
      }
    }
  }

  // Scalar tail: the elementsPerBlock % VEC columns the vector loop cannot
  // cover. Without this they were silently dropped from the max/sum.
  for (int64_t col = vec_end + tid; col < endCol; col += bdim) {
    float val = static_cast<float>(row_logits[col]);
    if (val > local_max) {
      local_sum = local_sum * expf(local_max - val) + 1.0f;
      local_max = val;
    } else {
      local_sum += expf(val - local_max);
    }
  }

  // --- PHASE 2: BLOCK REDUCTION ---
  smax[tid] = local_max;
  ssum[tid] = local_sum;
  __syncthreads();

  for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float other_max = smax[tid + s];
      float other_sum = ssum[tid + s];

      if (other_max > smax[tid]) {
        ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
        smax[tid] = other_max;
      } else {
        ssum[tid] += other_sum * expf(other_max - smax[tid]);
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_partial_max[bid] = smax[0];
    d_partial_sum[bid] = ssum[0];
  }
}

template <typename T, typename T_idx>
__global__ void sparseCENormalize_from_stats(
    const T *__restrict__ logits, const T_idx *__restrict__ targets,
    T *__restrict__ grad, const float *__restrict__ saved_max,
    const float *__restrict__ saved_sum, int64_t batch_size, int64_t vocab_size,
    const float *grad_output,
    // DEVICE pointer to the loss denominator -- see note on
    // count_valid_targets_kernel. Forward divided by *denom; so must backward.
    const float *__restrict__ denom,
    int64_t ignore_index = -1 // SFT : target value meaning "skip this row "
) {
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const int row = blockIdx.x;
  if (row >= batch_size)
    return;
  // SFT label masking: an ignored row must receive a ZERO gradient. This is a
  // CORRECTNESS requirement, not an optimisation — without it the model trains
  // on prompt tokens.
  // Zero with a plain T loop, NOT the float4 path below: that path assumes
  // sizeof(T)==4, and this kernel is also instantiated for float16/bfloat16
  // where a float4 write would run 2x past the end of the row.
  if (static_cast<int64_t>(targets[row]) == ignore_index) {
    T *row_grad_z = grad + row * vocab_size;
    for (int64_t c = tid; c < vocab_size; c += bdim)
      row_grad_z[c] = static_cast<T>(0);
    return;
  }
  const T *row_logits = logits + row * vocab_size;
  T *row_grad = grad + row * vocab_size;
  int64_t target_idx = static_cast<int64_t>(targets[row]);

  const float final_max = saved_max[row];
  const float inv_sum = 1.0f / saved_sum[row];
  const float f_scale = static_cast<float>(*grad_output) / (*denom);

  // Vectorized, COALESCED access: each thread handles one 16-byte vector per
  // stride. Thread tid reads vector tid, tid+bdim, ... so a warp's 32 threads
  // cover 32 contiguous 16-B vectors = 512 contiguous bytes, i.e. exactly
  // 4 x 128B sectors per warp per iteration.
  //
  // VEC comes from sizeof(T) (4 float / 8 bf16 / 2 double) and the vector type
  // is LossVec<T>, so 16 bytes always hold VEC elements OF T. The old code cast
  // row_logits/row_grad to float4* unconditionally, which for a 2-byte T both
  // reinterpreted 8 bf16 values as 4 floats and ran 2x past the end of the row.
  constexpr int VEC = LossVec<T>::WIDTH;
  using vec_t = LossVec<T>;
  // Load and store are both 16 B, so both rows must be 16-B aligned; otherwise
  // vec_count is 0 and the scalar tail below covers the whole row. The old code
  // checked neither, and had no tail at all -- for vocab_size % 4 != 0 the
  // trailing columns of every gradient row were simply never written.
  const bool aligned16 =
      ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) &&
      ((reinterpret_cast<uintptr_t>(row_grad) & 0xF) == 0);
  const int64_t vec_count = aligned16 ? (vocab_size / VEC) : 0;
  const vec_t *logits_v = reinterpret_cast<const vec_t *>(row_logits);
  vec_t *grad_v = reinterpret_cast<vec_t *>(row_grad);

  for (int64_t v = tid; v < vec_count; v += bdim) {
    // Single coalesced 16-byte load per thread
    vec_t in = logits_v[v];
    const int64_t col_base = v * VEC;

    vec_t out_grad;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float val = static_cast<float>(in.v[k]);
      float prob = expf(val - final_max) * inv_sum;
      float indicator = ((col_base + k) == target_idx) ? 1.0f : 0.0f;
      out_grad.v[k] = static_cast<T>((prob - indicator) * f_scale);
    }

    grad_v[v] = out_grad;
  }

  // Scalar tail: the vocab_size % VEC trailing columns, and the whole row when
  // unaligned.
  for (int64_t j = vec_count * VEC + tid; j < vocab_size; j += bdim) {
    float val = static_cast<float>(row_logits[j]);
    float prob = expf(val - final_max) * inv_sum;
    float indicator = (j == target_idx) ? 1.0f : 0.0f;
    row_grad[j] = static_cast<T>((prob - indicator) * f_scale);
  }
}
template <typename T, typename T_idx>
__global__ void sparseCENormalize_kernel_optimized(
    const T *__restrict__ logits, const T_idx *__restrict__ targets,
    T *__restrict__ grad, const float *d_partial_max,
    const float *d_partial_sum, int64_t batch_size, int64_t vocab_size,
    const float *grad_output, const float *__restrict__ denom, int num_partials,
    int64_t ignore_index = -1 // SFT : target value menaing " skip this row"
) {
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const int bid = blockIdx.x;

  const int row = bid;
  if (row >= batch_size)
    return;
  // SFT label masking: an ignored row must receive a ZERO gradient.
  // CORRECTNESS, not an optimisation — without it the model trains on prompt
  // tokens. Placed BEFORE the __syncthreads() below. The condition is
  // block-uniform (it depends only on blockIdx.x), so all threads of this block
  // take the same branch and none is left waiting at that barrier. Zero with a
  // plain T loop, NOT the float4 store used below: that store assumes
  // sizeof(T)==4, and this kernel is also instantiated for float16/bfloat16
  // where a float4 write would run past the end of the row.
  if (static_cast<int64_t>(targets[row]) == ignore_index) {
    T *row_grad_z = grad + row * vocab_size;
    for (int64_t c = tid; c < vocab_size; c += bdim)
      row_grad_z[c] = static_cast<T>(0);
    return;
  }
  // Row-specific pointers and target info
  const T *row_logits = logits + row * vocab_size;
  T *row_grad = grad + row * vocab_size;
  int64_t target_idx = static_cast<int64_t>(targets[row]);

  // --- PHASE 1: STATS CONSOLIDATION ---
  __shared__ float final_max_shared;
  __shared__ float final_sum_shared;

  if (tid == 0) {
    float g_max = -1e38f;
    float g_sum = 0.0f;
    for (int i = 0; i < num_partials; ++i) {
      float p_max = d_partial_max[row * num_partials + i];
      float p_sum = d_partial_sum[row * num_partials + i];

      if (p_max > g_max) {
        g_sum = g_sum * expf(g_max - p_max) + p_sum;
        g_max = p_max;
      } else {
        g_sum += p_sum * expf(p_max - g_max);
      }
    }
    final_max_shared = g_max;
    final_sum_shared = g_sum;
  }
  __syncthreads();

  const float final_max = final_max_shared;
  const float inv_sum = 1.0f / final_sum_shared;
  const float f_scale = static_cast<float>(*grad_output) / (*denom);

  // --- PHASE 2: VECTORIZED NORMALIZATION & STORE ---
  extern __shared__ float sdata[];
  // 16-byte vectors, VEC elements each (4 float / 8 bf16 / 2 double). Staged as
  // T so the cp.async bytes read back as the type they were written as, and
  // stored through LossVec<T> so the write is typed too -- the old code cast
  // row_grad to float4*, which for a 2-byte T wrote 2x past the row.
  constexpr int VEC = LossVec<T>::WIDTH;
  using vec_t = LossVec<T>;
  T *s_stage = reinterpret_cast<T *>(sdata);
  // Load and store are both 16 B, so BOTH rows must be 16-B aligned. If either
  // is not, vec_count is 0 and the scalar tail covers the whole row. (The old
  // code checked neither.)
  const bool aligned16 =
      ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) &&
      ((reinterpret_cast<uintptr_t>(row_grad) & 0xF) == 0);
  const int64_t vec_count = aligned16 ? (vocab_size / VEC) : 0;
  vec_t *grad_v = reinterpret_cast<vec_t *>(row_grad);

  // v indexes FULL vectors: byte offset v*16, and the store is always entirely
  // inside the row, so no `j + 3 < vocab_size` guard is needed.
  for (int64_t v = tid; v < vec_count; v += bdim) {
    const int64_t j = v * VEC;

    // 1. ASYNC LOAD: Stream logits directly into Shared Memory
    // .cg qualifier avoids L1 pollution for data read only once [1-3].
    cp_async_16(&s_stage[tid * VEC], &row_logits[j]);

    // 2. COORDINATION: Commit and wait for the "conveyor belt" [4-7].
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    // No __syncthreads() here: each thread reads only its own
    // s_stage[tid*4..tid*4+3] slot.
    // __syncthreads() inside a variable-iteration loop causes deadlock when
    // vocab_size is not a multiple of bdim*4 (e.g. vocab_size=50304, bdim=256).

    // 3. COMPUTE & VECTORIZED STORE
    vec_t out_grad;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float val = static_cast<float>(s_stage[tid * VEC + k]);
      float prob = expf(val - final_max) * inv_sum;
      float indicator = ((j + k) == target_idx) ? 1.0f : 0.0f;
      out_grad.v[k] = static_cast<T>((prob - indicator) * f_scale);
    }

    // 4. Vectorized global store (128-bit), typed
    grad_v[v] = out_grad;
  }

  // Scalar tail: the vocab_size % VEC trailing columns, and the whole row when
  // unaligned. The old `if (j + 3 < vocab_size)` guard skipped these entirely,
  // leaving whatever Tensor::empty had left in the gradient buffer.
  for (int64_t j = vec_count * VEC + tid; j < vocab_size; j += bdim) {
    float val = static_cast<float>(row_logits[j]);
    float prob = expf(val - final_max) * inv_sum;
    float indicator = (j == target_idx) ? 1.0f : 0.0f;
    row_grad[j] = static_cast<T>((prob - indicator) * f_scale);
  }
}

//! launch function
template <typename T, typename Tidx>
void sparse_ce_backward_cuda_vec(const T *logits, const Tidx *targets,
                                 T *grad_logits, int64_t batch_size,
                                 int64_t vocab_size, const float *grad_output,
                                 const float *denom, cudaStream_t stream,
                                 int64_t ignore_index) {
  CUDA_KERNEL_SCOPE("sparse_ce_backward_cuda_vec", stream);
  if (batch_size == 0)
    return;
  dim3 block{256}; //! play around with this..
  //* one block per row
  dim3 grid_reduce{2, batch_size}; //* lets split and then do a 2d grid...
  dim3 grid_norm{(unsigned)batch_size};
  //* Buffers for multi-block online softmax
  float *d_partial_max, *d_partial_sum;
  const size_t partial_bytes =
      (size_t)batch_size * grid_reduce.x * sizeof(float);
  d_partial_max = static_cast<float *>(
      CachingCUDAAllocator::instance().allocate(partial_bytes, stream));
  d_partial_sum = static_cast<float *>(
      CachingCUDAAllocator::instance().allocate(partial_bytes, stream));
  //* same stream for serializing and avoiding synchronization
  //* by deafult stream 0
  sparseCEReduce_kernel_optimized<<<grid_reduce, block,
                                    6 * block.x * sizeof(float)>>>(
      logits, targets, grad_logits, d_partial_max, d_partial_sum, batch_size,
      vocab_size, ignore_index);
  sparseCENormalize_kernel_optimized<<<grid_norm, block,
                                       block.x * 4 * sizeof(float)>>>(
      logits, targets, grad_logits, d_partial_max, d_partial_sum, batch_size,
      vocab_size, grad_output, denom, (int)grid_reduce.x, ignore_index);

  CachingCUDAAllocator::instance().deallocate(d_partial_max);
  CachingCUDAAllocator::instance().deallocate(d_partial_sum);
}

// =============================================================================
// Forward launcher that ALSO writes per-row (max, sum) for use by the
// stats-aware backward. Allocator-allocates the two scratch tensors which
// the autograd node holds onto until backward fires.
// =============================================================================
template <typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec_save_stats(
    const T *logits, const T_idx *targets, float *loss_output,
    float *saved_max, // pre-allocated [batch_size]
    float *saved_sum, // pre-allocated [batch_size]
    int64_t batch_size, int64_t vocab_size, cudaStream_t stream,
    int64_t ignore_index, const float *denom) {
  CUDA_KERNEL_SCOPE("sparse_cross_entropy_forward_cuda_impl_vec_save_stats",
                    stream);
  if (batch_size == 0) {
    cudaMemsetAsync(loss_output, 0, sizeof(float), stream);
    return;
  }
  float *d_losses = static_cast<float *>(CachingCUDAAllocator::instance().allocate(
      batch_size * sizeof(float), stream));

  int threads = 512;
  int blocks = batch_size;
  // B300: route to the sm103 register-resident kernel (drops cp.async/smem
  // staging). Every other arch keeps the generic staged kernel unchanged.
  if (get_arch() == ArchFamily::Blackwell_B300) {
    // TODO(SFT): the sm103 kernel does NOT yet support ignore_index and on B300
    // , SFT label masking is silently a no-op and the model will train on
    // prompt tokens , port the ignore_index early-out into
    // arch/CrossEntropyKernels_sm103.cu before running SFT on B300.
    sparse_ce_fwd_sm103_launch<T, T_idx>(logits, targets, d_losses, saved_max,
                                         saved_sum, batch_size, vocab_size,
                                         stream);
  } else {
    size_t shared_mem = 6 * threads * sizeof(float);
    sparse_ce_forward_kernel_vec_save_stats<T, T_idx>
        <<<blocks, threads, shared_mem, stream>>>(
            logits, targets, d_losses, saved_max, saved_sum, batch_size,
            vocab_size, ignore_index);
  }

  int reduce_threads = 256;
  int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;
  if (reduce_blocks == 1) {
    sum_reduction_kernel<float, float>
        <<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, loss_output, batch_size, denom);
  } else {
    float *d_partial = static_cast<float *>(CachingCUDAAllocator::instance().allocate(
        reduce_blocks * sizeof(float), stream));
    sum_reduction_kernel<float, float>
        <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float),
           stream>>>(d_losses, d_partial, batch_size);
    sum_reduction_kernel<float, float>
        <<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_partial, loss_output, reduce_blocks, denom);
    CachingCUDAAllocator::instance().deallocate(d_partial);
  }
  CachingCUDAAllocator::instance().deallocate(d_losses);
}

// =============================================================================
// Backward launcher that REUSES forward's saved (max, sum). Only the Normalize
// kernel runs — Reduce is skipped entirely. ~3.7 ms / call faster.
// =============================================================================

template <typename T, typename Tidx>
void sparse_ce_backward_cuda_vec_with_stats(
    const T *logits, const Tidx *targets, T *grad_logits,
    const float *saved_max, const float *saved_sum, int64_t batch_size,
    int64_t vocab_size, const float *grad_output, const float *denom,
    cudaStream_t stream, int64_t ignore_index) {
  CUDA_KERNEL_SCOPE("sparse_ce_backward_cuda_vec_with_stats", stream);
  if (batch_size == 0)
    return;
  dim3 block{256};
  dim3 grid_norm{(unsigned)batch_size};
  sparseCENormalize_from_stats<<<grid_norm, block, 0, stream>>>(
      logits, targets, grad_logits, saved_max, saved_sum, batch_size,
      vocab_size, grad_output, denom, ignore_index);
}
/*
 * ###################################################################################################################################
 * ###################################################################################################################################
 * ###################################################################################################################################
 */

/*
 * ###################################################################################################################################
 * #########################################   Implementation of the two kernel
 * approch   ############################################
 * ###################################################################################################################################
 */

//! kerenl function for reduction

//! kernel function to normalize and compute loss

//! launch function

/*
 * ###################################################################################################################################
 * ###################################################################################################################################
 * ###################################################################################################################################
 */


// Specialization/Guard for non-numeric types if needed, but static_cast<float>
// should work for most primitives. Complex types will still fail to static_cast
// to float.

// We will restrict the instantiations to real numeric types.

template <typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(const T *logits, const T_idx *targets,
                                        T *grad_logits, int64_t batch_size,
                                        int64_t vocab_size,
                                        const float *grad_output, const float *denom,
                                        cudaStream_t stream,
                                        int64_t ignore_index) {
  CUDA_KERNEL_SCOPE("sparse_cross_entropy_backward_cuda", stream);
  // Forwards to the two-kernel vec path, whose Reduce + Normalize kernels both
  // carry the SFT guard. Without passing ignore_index through, this fallback
  // silently trains on the masked (prompt) rows.
  sparse_ce_backward_cuda_vec<T, T_idx>(logits, targets, grad_logits,
                                        batch_size, vocab_size, grad_output,
                                        denom, stream, ignore_index);
}

// Explicit instantiations for two-kernel backward

// Explicit instantiations for vectorized two-kernel backward
#define INSTANTIATE_TWO_KERNEL_VEC(T)                                          \
  template void sparse_ce_backward_cuda_vec<T, int8_t>(                        \
      const T *, const int8_t *, T *, int64_t, int64_t, const float *, const float *,      \
      cudaStream_t, int64_t);                                                  \
  template void sparse_ce_backward_cuda_vec<T, int16_t>(                       \
      const T *, const int16_t *, T *, int64_t, int64_t, const float *, const float *,     \
      cudaStream_t, int64_t);                                                  \
  template void sparse_ce_backward_cuda_vec<T, int32_t>(                       \
      const T *, const int32_t *, T *, int64_t, int64_t, const float *, const float *,     \
      cudaStream_t, int64_t);                                                  \
  template void sparse_ce_backward_cuda_vec<T, int64_t>(                       \
      const T *, const int64_t *, T *, int64_t, int64_t, const float *, const float *,     \
      cudaStream_t, int64_t);

INSTANTIATE_TWO_KERNEL_VEC(float)
INSTANTIATE_TWO_KERNEL_VEC(double)

// Stats-aware variants: forward writes (max, sum); backward reads them and
// skips Reduce.
// sparse_cross_entropy_forward_cuda_impl_vec used to be instantiated only as a
// SIDE EFFECT of sparse_cross_entropy_forward_cuda_vec calling it. That wrapper
// was dead code and has been removed, so the instantiation is now explicit --
// otherwise LossOps.cpp (a different TU) gets an undefined reference.
// These 9 combos are exactly what LossOps dispatches: fp32/fp16/bf16 logits x
// uint16/int32/int64 targets.
#define INSTANTIATE_FWD_IMPL_VEC(T)                                            \
  template void sparse_cross_entropy_forward_cuda_impl_vec<T, uint16_t>(       \
      const T *, const uint16_t *, float *, int64_t, int64_t, cudaStream_t,        \
      int64_t, const float *);                                                 \
  template void sparse_cross_entropy_forward_cuda_impl_vec<T, int32_t>(        \
      const T *, const int32_t *, float *, int64_t, int64_t, cudaStream_t,         \
      int64_t, const float *);                                                 \
  template void sparse_cross_entropy_forward_cuda_impl_vec<T, int64_t>(        \
      const T *, const int64_t *, float *, int64_t, int64_t, cudaStream_t,         \
      int64_t, const float *);
INSTANTIATE_FWD_IMPL_VEC(float)
INSTANTIATE_FWD_IMPL_VEC(float16_t)
INSTANTIATE_FWD_IMPL_VEC(bfloat16_t)

#define INSTANTIATE_FWD_SAVE_STATS(T)                                          \
  template void                                                                \
  sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, uint16_t>(          \
      const T *, const uint16_t *, float *, float *, float *, int64_t, int64_t,    \
      cudaStream_t, int64_t, const float *);                                                  \
  template void                                                                \
  sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, int32_t>(           \
      const T *, const int32_t *, float *, float *, float *, int64_t, int64_t,     \
      cudaStream_t, int64_t, const float *);                                                  \
  template void                                                                \
  sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, int64_t>(           \
      const T *, const int64_t *, float *, float *, float *, int64_t, int64_t,     \
      cudaStream_t, int64_t, const float *);
INSTANTIATE_FWD_SAVE_STATS(float)
INSTANTIATE_FWD_SAVE_STATS(float16_t)
INSTANTIATE_FWD_SAVE_STATS(bfloat16_t)

#define INSTANTIATE_BWD_WITH_STATS(T)                                          \
  template void sparse_ce_backward_cuda_vec_with_stats<T, uint16_t>(           \
      const T *, const uint16_t *, T *, const float *, const float *, int64_t, \
      int64_t, const float *, const float *, cudaStream_t, int64_t);                       \
  template void sparse_ce_backward_cuda_vec_with_stats<T, int32_t>(            \
      const T *, const int32_t *, T *, const float *, const float *, int64_t,  \
      int64_t, const float *, const float *, cudaStream_t, int64_t);                       \
  template void sparse_ce_backward_cuda_vec_with_stats<T, int64_t>(            \
      const T *, const int64_t *, T *, const float *, const float *, int64_t,  \
      int64_t, const float *, const float *, cudaStream_t, int64_t);
INSTANTIATE_BWD_WITH_STATS(float)
INSTANTIATE_BWD_WITH_STATS(float16_t)
INSTANTIATE_BWD_WITH_STATS(bfloat16_t)

// Explicit instantiations for supported types
#define INSTANTIATE_GIVEN_T(T)                                                \
  template void sparse_cross_entropy_backward_cuda<T, uint8_t>(               \
      const T *, const uint8_t *, T *, int64_t, int64_t, const float *, const float *,    \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, uint16_t>(              \
      const T *, const uint16_t *, T *, int64_t, int64_t, const float *, const float *,   \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, uint32_t>(              \
      const T *, const uint32_t *, T *, int64_t, int64_t, const float *, const float *,   \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, uint64_t>(              \
      const T *, const uint64_t *, T *, int64_t, int64_t, const float *, const float *,   \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, int8_t>(                \
      const T *, const int8_t *, T *, int64_t, int64_t, const float *, const float *,     \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, int16_t>(               \
      const T *, const int16_t *, T *, int64_t, int64_t, const float *, const float *,    \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, int32_t>(               \
      const T *, const int32_t *, T *, int64_t, int64_t, const float *, const float *,    \
      cudaStream_t, int64_t);                                                 \
  template void sparse_cross_entropy_backward_cuda<T, int64_t>(               \
      const T *, const int64_t *, T *, int64_t, int64_t, const float *, const float *,    \
      cudaStream_t, int64_t);

INSTANTIATE_GIVEN_T(float)
INSTANTIATE_GIVEN_T(double)
INSTANTIATE_GIVEN_T(float16_t)
INSTANTIATE_GIVEN_T(bfloat16_t)

// Explicit instantiations for forward pass

// Explicit instantiations for vectorized forward pass

// ============================================================================
// Categorical Cross Entropy Extensions
// ============================================================================

__global__ void cce_forward_kernel(const float *__restrict__ predictions,
                                   const float *__restrict__ targets,
                                   float *__restrict__ losses, // [N]
                                   int64_t batch_size, int64_t num_classes) {
  int64_t i = blockIdx.x; // One block per sample
  if (i >= batch_size)
    return;

  const float epsilon = 1e-7f;
  const float one_minus_epsilon = 1.0f - 1e-7f;

  const float *row_pred = predictions + i * num_classes;
  const float *row_targ = targets + i * num_classes;

  float sum = 0.0f;
#pragma unroll 4
  for (int64_t j = threadIdx.x; j < num_classes; j += blockDim.x) {
    float p = row_pred[j];
    float t = row_targ[j];

    // Clip
    if (p < epsilon)
      p = epsilon;
    else if (p > one_minus_epsilon)
      p = one_minus_epsilon;

    sum += t * logf(p);
  }

  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    losses[i] = -sdata[0]; // Negate here (-sum is loss)
  }
}

__global__ void scale_loss_kernel(float *val, float s) {
  if (threadIdx.x == 0)
    *val *= s;
}

void categorical_cross_entropy_forward_cuda(const float *predictions,
                                            const float *targets,
                                            float *loss_output,
                                            int64_t batch_size,
                                            int64_t num_classes) {
  CUDA_KERNEL_SCOPE("categorical_cross_entropy_forward_cuda");
  if (batch_size == 0)
    return;

  float *d_losses = static_cast<float *>(
      CachingCUDAAllocator::instance().allocate(batch_size * sizeof(float)));

  int threads = 256;
  int blocks = batch_size;

  // 1. Per-sample loss
  cce_forward_kernel<<<blocks, threads, threads * sizeof(float)>>>(
      predictions, targets, d_losses, batch_size, num_classes);

  // 2. Reduce sum
  int reduce_threads = 256;
  int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;

  if (reduce_blocks == 1) {
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, loss_output, batch_size);
  } else {
    float *d_partial =
        static_cast<float *>(CachingCUDAAllocator::instance().allocate(
            reduce_blocks * sizeof(float)));
    sum_reduction_kernel<float>
        <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, d_partial, batch_size);
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_partial, loss_output, reduce_blocks);
    CachingCUDAAllocator::instance().deallocate(d_partial);
  }

  // 3. Average (Divide by batch_size)
  scale_loss_kernel<<<1, 1>>>(loss_output,
                              1.0f / static_cast<float>(batch_size));

  CachingCUDAAllocator::instance().deallocate(d_losses);
}

__global__ void
cce_backward_kernel(const float *__restrict__ grad_output, // scalar
                    const float *__restrict__ predictions,
                    const float *__restrict__ targets,
                    float *__restrict__ grad_input, int64_t numel,
                    float scale // 1/N
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;

  const float epsilon = 1e-7f;
  const float one_minus_epsilon = 1.0f - 1e-7f;

  float p = predictions[i];
  float t = targets[i];
  float g = *grad_output; // Access scalar gradient (assumed on GPU)

  float grad = 0.0f;
  // d(log(clip(p))) = 1/clipped_p if not clamped, else 0
  if (p >= epsilon && p <= one_minus_epsilon) {
    grad = g * (-t / p);
  }

  grad_input[i] = grad * scale;
}

void categorical_cross_entropy_backward_cuda(
    const float *grad_output, const float *predictions, const float *targets,
    float *grad_input, int64_t batch_size, int64_t num_classes) {
  CUDA_KERNEL_SCOPE("categorical_cross_entropy_backward_cuda");
  int64_t numel = batch_size * num_classes;
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;

  cce_backward_kernel<<<blocks, threads>>>(
      grad_output, predictions, targets, grad_input, numel,
      1.0f / static_cast<float>(batch_size));
}

// ============================================================================
// MSE / MAE / BCE Extensions
// ============================================================================

// --- MSE ---
__global__ void mse_forward_kernel(const float *__restrict__ p,
                                   const float *__restrict__ t,
                                   float *__restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float diff = p[i] - t[i];
  out[i] = diff * diff;
}

__global__ void mse_backward_kernel(const float *__restrict__ grad_out,
                                    const float *__restrict__ p,
                                    const float *__restrict__ t,
                                    float *__restrict__ grad_in, int64_t n,
                                    float scale) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float g = *grad_out; // scalar
  float diff = p[i] - t[i];
  grad_in[i] = 2.0f * diff * g * scale;
}

void mse_loss_forward_cuda(const float *predictions, const float *targets,
                           float *loss_output, int64_t numel) {
  CUDA_KERNEL_SCOPE("mse_loss_forward_cuda");
  if (numel == 0)
    return;
  float *d_losses = static_cast<float *>(
      CachingCUDAAllocator::instance().allocate(numel * sizeof(float)));

  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  mse_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses,
                                          numel);

  // Reduce
  int reduce_threads = 256;
  int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;

  if (reduce_blocks == 1) {
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, loss_output, numel);
  } else {
    float *d_partial =
        static_cast<float *>(CachingCUDAAllocator::instance().allocate(
            reduce_blocks * sizeof(float)));
    sum_reduction_kernel<float>
        <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, d_partial, numel);
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_partial, loss_output, reduce_blocks);
    CachingCUDAAllocator::instance().deallocate(d_partial);
  }

  scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
  CachingCUDAAllocator::instance().deallocate(d_losses);
}

void mse_loss_backward_cuda(const float *grad_output, const float *predictions,
                            const float *targets, float *grad_input,
                            int64_t numel) {
  CUDA_KERNEL_SCOPE("mse_loss_backward_cuda");
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  mse_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets,
                                           grad_input, numel, 1.0f / numel);
}

// --- MAE ---
__global__ void mae_forward_kernel(const float *__restrict__ p,
                                   const float *__restrict__ t,
                                   float *__restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float diff = p[i] - t[i];
  out[i] = fabsf(diff);
}

__global__ void mae_backward_kernel(const float *__restrict__ grad_out,
                                    const float *__restrict__ p,
                                    const float *__restrict__ t,
                                    float *__restrict__ grad_in, int64_t n,
                                    float scale) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float g = *grad_out;
  float diff = p[i] - t[i];
  float sign = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
  grad_in[i] = sign * g * scale;
}

void mae_loss_forward_cuda(const float *predictions, const float *targets,
                           float *loss_output, int64_t numel) {
  CUDA_KERNEL_SCOPE("mae_loss_forward_cuda");
  if (numel == 0)
    return;
  float *d_losses = static_cast<float *>(
      CachingCUDAAllocator::instance().allocate(numel * sizeof(float)));

  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  mae_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses,
                                          numel);

  // Reduce (same logic as MSE, could abstract but copy-paste is safer for now
  // without templating spaghetti)
  int reduce_threads = 256;
  int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;

  if (reduce_blocks == 1) {
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, loss_output, numel);
  } else {
    float *d_partial =
        static_cast<float *>(CachingCUDAAllocator::instance().allocate(
            reduce_blocks * sizeof(float)));
    sum_reduction_kernel<float>
        <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, d_partial, numel);
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_partial, loss_output, reduce_blocks);
    CachingCUDAAllocator::instance().deallocate(d_partial);
  }

  scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
  CachingCUDAAllocator::instance().deallocate(d_losses);
}

void mae_loss_backward_cuda(const float *grad_output, const float *predictions,
                            const float *targets, float *grad_input,
                            int64_t numel) {
  CUDA_KERNEL_SCOPE("mae_loss_backward_cuda");
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  mae_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets,
                                           grad_input, numel, 1.0f / numel);
}

// --- BCE ---
__global__ void bce_forward_kernel(const float *__restrict__ p,
                                   const float *__restrict__ t,
                                   float *__restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float pi = p[i];
  float ti = t[i];

  // Clip
  const float eps = 1e-7f;
  const float one_minus_eps = 1.0f - 1e-7f;
  if (pi < eps)
    pi = eps;
  else if (pi > one_minus_eps)
    pi = one_minus_eps;

  // Loss = -(t * log(p) + (1-t) * log(1-p))
  out[i] = -(ti * logf(pi) + (1.0f - ti) * logf(1.0f - pi));
}

__global__ void bce_backward_kernel(const float *__restrict__ grad_out,
                                    const float *__restrict__ p,
                                    const float *__restrict__ t,
                                    float *__restrict__ grad_in, int64_t n,
                                    float scale) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float g = *grad_out;
  float pi = p[i];
  float ti = t[i];

  // Clip
  const float eps = 1e-7f;
  const float one_minus_eps = 1.0f - eps;
  float p_clipped = pi;
  if (p_clipped < eps)
    p_clipped = eps;
  else if (p_clipped > one_minus_eps)
    p_clipped = one_minus_eps;

  // grad = (-t/p + (1-t)/(1-p)) * scale * g
  // if clipped, grad might be 0 theoretically, but standard DL frameworks
  // usually pass gradient through clipped values or use logits. Here we
  // replicate strict derivative of the loss function formula with clipped p.

  float term1 = -ti / p_clipped;
  float term2 = (1.0f - ti) / (1.0f - p_clipped);
  grad_in[i] = (term1 + term2) * g * scale;
}

void bce_loss_forward_cuda(const float *predictions, const float *targets,
                           float *loss_output, int64_t numel) {
  CUDA_KERNEL_SCOPE("bce_loss_forward_cuda");
  if (numel == 0)
    return;
  float *d_losses = static_cast<float *>(
      CachingCUDAAllocator::instance().allocate(numel * sizeof(float)));

  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  bce_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses,
                                          numel);

  // Reduce
  int reduce_threads = 256;
  int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;

  if (reduce_blocks == 1) {
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, loss_output, numel);
  } else {
    float *d_partial =
        static_cast<float *>(CachingCUDAAllocator::instance().allocate(
            reduce_blocks * sizeof(float)));
    sum_reduction_kernel<float>
        <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, d_partial, numel);
    sum_reduction_kernel<float>
        <<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_partial, loss_output, reduce_blocks);
    CachingCUDAAllocator::instance().deallocate(d_partial);
  }

  scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
  CachingCUDAAllocator::instance().deallocate(d_losses);
}

void bce_loss_backward_cuda(const float *grad_output, const float *predictions,
                            const float *targets, float *grad_input,
                            int64_t numel) {
  CUDA_KERNEL_SCOPE("bce_loss_backward_cuda");
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  bce_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets,
                                           grad_input, numel, 1.0f / numel);
}

} // namespace cuda
} // namespace OwnTensor
