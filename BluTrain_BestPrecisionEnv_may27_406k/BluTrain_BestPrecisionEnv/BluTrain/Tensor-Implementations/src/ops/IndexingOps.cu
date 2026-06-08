#include "ops/IndexingOps.h"
#include "core/TensorDispatch.h"
#ifdef WITH_CUDA
#include "ops/helpers/IndexingKernels.h"
#include <cuda_runtime.h>
#endif
#include <stdexcept>
#include <vector>

namespace OwnTensor {

Tensor gather(const Tensor& input, int64_t dim, const Tensor& index) {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("gather: dimension out of range");
    }

    // Basic implementation for CPU
    // We assume index can be broadcasted or at least has matching dimensions except at dim
    // For simplicity, we handle the case where index has same rank as input
    // or index is one rank lower and we are gathering along 'dim'.
    
    // For now, let's implement the matching rank case (PyTorch style)
    // and a special case for 1D targets with 2D/3D logits if it's common.
    
    // RESHAPE index if it's one rank lower
    Tensor index_reshaped = index;
    if (index.ndim() == input.ndim() - 1) {
        std::vector<int64_t> new_shape = index.shape().dims;
        new_shape.insert(new_shape.begin() + dim, 1);
        index_reshaped = index.view(Shape{new_shape});
    }

    if (index_reshaped.ndim() != input.ndim()) {
        throw std::invalid_argument("gather: index rank must match input rank (or be rank-1)");
    }

    Shape out_shape = index_reshaped.shape();
    Tensor result(out_shape, input.dtype(), input.device());

    if (input.is_cpu()) {
        dispatch_by_dtype(input.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            const T* in_ptr = input.data<T>();
            T* out_ptr = result.data<T>();

            dispatch_by_integer_dtype(index_reshaped.dtype(), [&](auto dummy_idx) {
                using T_idx = decltype(dummy_idx);
                const T_idx* idx_ptr = index_reshaped.data<T_idx>();

                size_t total_out = result.numel();
                const auto& out_sizes = result.shape().dims;
                const auto& in_strides = input.stride().strides;
                const auto& idx_strides = index_reshaped.stride().strides;
                
                #pragma omp parallel for
                for (size_t i = 0; i < total_out; ++i) {
                    // Convert linear index to coordinates for 'result'
                    std::vector<int64_t> coords(ndim);
                    size_t temp = i;
                    for (int j = ndim - 1; j >= 0; --j) {
                        coords[j] = temp % out_sizes[j];
                        temp /= out_sizes[j];
                    }

                    // Get index value at this coordinate
                    // (Assuming index and result have same shape)
                    T_idx gathered_idx = idx_ptr[i]; 

                    // Check bounds
                    if (gathered_idx < 0 || gathered_idx >= input.shape().dims[dim]) {
                        // In a real implementation we might throw, but in parallel loop it's hard.
                        // We'll just clamp or let it crash for now to keep it simple.
                        continue; 
                    }

                    // Compute input coordinate
                    std::vector<int64_t> in_coords = coords;
                    in_coords[dim] = static_cast<int64_t>(gathered_idx);

                    // Compute input linear index
                    size_t in_lin_idx = 0;
                    for (int j = 0; j < ndim; ++j) {
                        in_lin_idx += in_coords[j] * in_strides[j];
                    }

                    out_ptr[i] = in_ptr[in_lin_idx];
                }
            });
        });
    } else if (input.is_cuda()) {
#ifdef WITH_CUDA
        cudaStream_t stream = 0; // Or get current stream
        int ndim = input.ndim();
        size_t numel = index_reshaped.numel();

        // Allocate and copy metadata to GPU
        int64_t* d_in_strides = nullptr;
        int64_t* d_idx_strides = nullptr;
        int64_t* d_in_dims = nullptr;
        int64_t* d_idx_dims = nullptr;

        cudaMallocAsync(&d_in_strides, ndim * sizeof(int64_t), stream);
        cudaMallocAsync(&d_idx_strides, ndim * sizeof(int64_t), stream);
        cudaMallocAsync(&d_in_dims, ndim * sizeof(int64_t), stream);
        cudaMallocAsync(&d_idx_dims, ndim * sizeof(int64_t), stream);

        cudaMemcpyAsync(d_in_strides, input.stride().strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_idx_strides, index_reshaped.stride().strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_in_dims, input.shape().dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_idx_dims, index_reshaped.shape().dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        dispatch_by_dtype(input.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            dispatch_by_integer_dtype(index_reshaped.dtype(), [&](auto dummy_idx) {
                using T_idx = decltype(dummy_idx);
                cuda::gather_cuda<T, T_idx>(
                    input.data<T>(),
                    index_reshaped.data<T_idx>(),
                    result.data<T>(),
                    dim,
                    d_in_strides,
                    d_idx_strides,
                    nullptr, // out_strides not used in the kernel yet
                    d_in_dims,
                    d_idx_dims,
                    ndim,
                    numel,
                    stream
                );
            });
        });

        cudaFreeAsync(d_in_strides, stream);
        cudaFreeAsync(d_idx_strides, stream);
        cudaFreeAsync(d_in_dims, stream);
        cudaFreeAsync(d_idx_dims, stream);
#else
        throw std::runtime_error("gather: CUDA support not compiled");
#endif
    } else {
        throw std::runtime_error("gather: Unknown device type");
    }

    return result;
}

} // namespace OwnTensor
