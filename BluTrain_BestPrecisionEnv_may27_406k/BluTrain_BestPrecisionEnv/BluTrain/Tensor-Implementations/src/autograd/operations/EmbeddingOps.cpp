#include "autograd/operations/EmbeddingOps.h"
#include "autograd/backward/EmbeddingBackward.h"
#include "autograd/ops_template.h"
#include "autograd/Variable.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "ops/helpers/EmbeddingKernels.h"
#include "device/DeviceCore.h"
#include <stdexcept>
#include <cstring>  // std::memcpy, std::memset
#include "device/AllocationTracker.h"
namespace OwnTensor {
namespace autograd {

Tensor embedding(const Tensor& weight, const Tensor& indices, int padding_idx) {
    GraphRecordMode::record_forward("EMBEDDING: embedding");
    TRACK_ALLOC_SCOPE("L17:autograd::embedding");
    // Get dimensions
    auto weight_shape = weight.shape().dims;
    if (weight_shape.size() != 2) {
        throw std::runtime_error("embedding: weight must be 2D [vocab_size, embed_dim]");
    }
    int64_t vocab_size = weight_shape[0];
    int64_t embed_dim = weight_shape[1];
    
    auto indices_shape = indices.shape().dims;
    if (indices_shape.size() < 1 || indices_shape.size() > 2) {
        throw std::runtime_error("embedding: indices must be 1D or 2D");
    }
    
    // Compute output shape: indices_shape + [embed_dim]
    std::vector<int64_t> output_dims = indices_shape;
    output_dims.push_back(embed_dim);
    Shape output_shape{output_dims};
    
    // Create output tensor
    TensorOptions opts = TensorOptions()
        .with_dtype(weight.dtype())
        .with_device(weight.device());
    Tensor output(output_shape, opts);
    
    // Get total number of lookups
    int64_t num_indices = indices.numel();
    
    // Forward pass: lookup weight rows by indices
    // CPU: templated scatter lookup; CUDA: dispatch to overloaded kernel
    auto cpu_scatter_lookup = [&](auto* weight_data, auto* output_data, auto get_idx) {
        using T = std::remove_pointer_t<decltype(output_data)>;
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t token_id = get_idx(i);
            T* out_row = output_data + i * embed_dim;
            if (token_id == (int64_t)padding_idx) {
                std::memset(out_row, 0, embed_dim * sizeof(T));
                continue;
            }
            if (token_id < 0 || token_id >= vocab_size) {
                throw std::runtime_error(
                    "embedding: index out of range: " + std::to_string(token_id));
            }
            const T* row = weight_data + token_id * embed_dim;
            std::memcpy(out_row, row, embed_dim * sizeof(T));
        }
    };

    auto cpu_dispatch_indices = [&](auto* weight_data, auto* output_data) {
        if (indices.dtype() == Dtype::Int64) {
            const int64_t* idx = indices.data<int64_t>();
            cpu_scatter_lookup(weight_data, output_data, [idx](int64_t i) -> int64_t { return idx[i]; });
        } else if (indices.dtype() == Dtype::Int32) {
            const int32_t* idx = indices.data<int32_t>();
            cpu_scatter_lookup(weight_data, output_data, [idx](int64_t i) -> int64_t { return static_cast<int64_t>(idx[i]); });
        } else if (indices.dtype() == Dtype::UInt16) {
            const uint16_t* idx = indices.data<uint16_t>();
            cpu_scatter_lookup(weight_data, output_data, [idx](int64_t i) -> int64_t { return static_cast<int64_t>(idx[i]); });
        } else {
            throw std::runtime_error("embedding: indices must be Int32, Int64, or UInt16");
        }
    };

    auto cuda_dispatch = [&](auto* weight_ptr, auto* output_ptr) {
        Tensor indices_gpu = indices.device().is_cpu() ? indices.to(weight.device()) : indices;
        Tensor indices_u16 = (indices_gpu.dtype() == Dtype::UInt16)
            ? indices_gpu : indices_gpu.as_type(Dtype::UInt16);

        device::set_cuda_device(weight.device().index);
        cuda::embedding_forward_cuda(
            indices_u16.data<uint16_t>(),
            weight_ptr, output_ptr,
            num_indices, embed_dim, vocab_size, padding_idx,
            weight.stride().strides[0], weight.stride().strides[1]
        );
    };

    if (weight.device().is_cpu()) {
        if (weight.dtype() == Dtype::Float16) {
            cpu_dispatch_indices(weight.data<float16_t>(), output.data<float16_t>());
        } else if (weight.dtype() == Dtype::Bfloat16) {
            cpu_dispatch_indices(weight.data<bfloat16_t>(), output.data<bfloat16_t>());
        } else {
            cpu_dispatch_indices(weight.data<float>(), output.data<float>());
        }
    } else {
        if (weight.dtype() == Dtype::Float16) {
            cuda_dispatch(weight.data<float16_t>(), output.data<float16_t>());
        } else if (weight.dtype() == Dtype::Bfloat16) {
            cuda_dispatch(weight.data<bfloat16_t>(), output.data<bfloat16_t>());
        } else {
            cuda_dispatch(weight.data<float>(), output.data<float>());
        }
    }
    
    // Set up autograd if needed
    if (GradMode::is_enabled() && weight.requires_grad()) {
        auto grad_fn = std::make_shared<EmbeddingBackward>(indices, vocab_size, embed_dim, padding_idx);
        
        grad_fn->set_next_edge(0, get_grad_edge(weight));
        
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    if (autograd::g_shape_debug)
        GraphRecordMode::attach_forward_shape(output.shape(), output.dtype());
    return output;
}

} // namespace autograd
} // namespace OwnTensor
