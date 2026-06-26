#include "autograd/backward/EmbeddingBackward.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"
#include "ops/helpers/EmbeddingKernels.h"
#include "dtype/Types.h"
#include <stdexcept>
#include <vector>
#include "device/DeviceCore.h"
#include <omp.h>  // OPTIMIZATION 3: OpenMP for CPU multi-threaded scatter-add

namespace OwnTensor {
namespace autograd {

EmbeddingBackward::EmbeddingBackward(const Tensor& indices, int64_t vocab_size, int64_t embed_dim, int padding_idx)
    : Node(1), 
      saved_indices_(indices, false),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      padding_idx_(padding_idx) {}

std::vector<Tensor> EmbeddingBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("EmbeddingBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];  // [B, T, C]
    Tensor indices = saved_indices_.unpack(shared_from_this());  // [B, T]
    
    // Create gradient tensor for weight [vocab_size, embed_dim]
    TensorOptions opts = TensorOptions()
        .with_dtype(grad_output.dtype())
        .with_device(grad_output.device());
    Tensor grad_weight = Tensor::zeros(Shape{{vocab_size_, embed_dim_}}, opts);
    
    // Get shapes
    auto grad_shape = grad_output.shape().dims;
    int64_t C = embed_dim_;
    
    // Scatter-add: grad_weight[indices[n], :] += grad_output[n, :]
    int64_t N = indices.numel();
    
    // ---- CPU scatter-add (templated on grad dtype) ----
    auto cpu_scatter_add_typed = [&](auto* grad_data, auto* weight_grad_data) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(grad_data)>>;

        auto scatter_add = [&](auto get_idx) {
            int num_threads = omp_get_max_threads();

            std::vector<std::vector<T>> private_grads(
                num_threads,
                std::vector<T>(static_cast<size_t>(vocab_size_) * C, T{})
            );

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T* local_grad = private_grads[tid].data();

                #pragma omp for schedule(static)
                for (int64_t n = 0; n < N; ++n) {
                    int64_t token_id = get_idx(n);
                    if (token_id == (int64_t)padding_idx_) continue;
                    if (token_id >= 0 && token_id < vocab_size_) {
                        const T* src = grad_data + n * C;
                        T*       dst = local_grad + token_id * C;

                        #pragma omp simd
                        for (int64_t c = 0; c < C; ++c) {
                            dst[c] += src[c];
                        }
                    }
                }
            }

            for (int t = 0; t < num_threads; ++t) {
                const T* local_grad = private_grads[t].data();
                for (int64_t row = 0; row < vocab_size_; ++row) {
                    T*       dst = weight_grad_data + row * C;
                    const T* src = local_grad       + row * C;
                    #pragma omp simd
                    for (int64_t c = 0; c < C; ++c) {
                        dst[c] += src[c];
                    }
                }
            }
        };

        if (indices.dtype() == Dtype::Int64) {
            const int64_t* idx_data = indices.data<int64_t>();
            scatter_add([idx_data](int64_t n) -> int64_t { return idx_data[n]; });
        } else if (indices.dtype() == Dtype::Int32) {
            const int32_t* idx_data = indices.data<int32_t>();
            scatter_add([idx_data](int64_t n) -> int64_t { return static_cast<int64_t>(idx_data[n]); });
        } else if (indices.dtype() == Dtype::UInt16) {
            const uint16_t* idx_data = indices.data<uint16_t>();
            scatter_add([idx_data](int64_t n) -> int64_t { return static_cast<int64_t>(idx_data[n]); });
        }
    };

    // ---- CUDA dispatch (calls overloaded embedding_backward_cuda) ----
    auto cuda_dispatch = [&](auto* grad_ptr, auto* weight_ptr) {
        Tensor indices_cuda = indices;
        if (indices.device().is_cpu()) {
            indices_cuda = indices.to(grad_output.device());
        }

        Tensor indices_u16 = (indices_cuda.dtype() == Dtype::UInt16)
            ? indices_cuda : indices_cuda.as_type(Dtype::UInt16);

        device::set_cuda_device(grad_output.device().index);
        cuda::embedding_backward_cuda(
            indices_u16.data<uint16_t>(),
            grad_ptr, weight_ptr,
            N, C, vocab_size_, padding_idx_,
            grad_weight.stride().strides[0], grad_weight.stride().strides[1]
        );
    };

    // ---- Dispatch by dtype ----
    if (grad_output.device().is_cpu()) {
        if (grad_output.dtype() == Dtype::Float32) {
            cpu_scatter_add_typed(grad_output.data<float>(), grad_weight.data<float>());
        } else if (grad_output.dtype() == Dtype::Float16) {
            cpu_scatter_add_typed(grad_output.data<float16_t>(), grad_weight.data<float16_t>());
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            cpu_scatter_add_typed(grad_output.data<bfloat16_t>(), grad_weight.data<bfloat16_t>());
        }
    } else {
#ifdef WITH_CUDA
        if (grad_output.dtype() == Dtype::Float32) {
            cuda_dispatch(grad_output.data<float>(), grad_weight.data<float>());
        } else if (grad_output.dtype() == Dtype::Float16) {
            cuda_dispatch(grad_output.data<float16_t>(), grad_weight.data<float16_t>());
        } else if (grad_output.dtype() == Dtype::Bfloat16) {
            cuda_dispatch(grad_output.data<bfloat16_t>(), grad_weight.data<bfloat16_t>());
        }
#endif
    }
    
    return {grad_weight};
}

} // namespace autograd
} // namespace OwnTensor