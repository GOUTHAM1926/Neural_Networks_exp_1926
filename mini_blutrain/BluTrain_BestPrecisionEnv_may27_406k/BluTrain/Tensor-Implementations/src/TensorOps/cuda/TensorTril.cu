#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include "ops/TensorOps.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/DeviceCore.h"

namespace OwnTensor
{
    template <typename T>
    __global__ void tril_kernel(const T* input, T* output, int64_t diagonal, T value, size_t H, size_t W, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            size_t row_idx = (idx / W) % H;
            size_t col_idx = idx % W;

            if (static_cast<int64_t>(col_idx) > static_cast<int64_t>(row_idx) + diagonal)
            {
                output[idx] = value;
            }
            else
            {
                output[idx] = input[idx];
            }
        }
    }

    void cuda_tril_tensor(const Tensor& input, Tensor& output, int64_t diagonal, double value, cudaStream_t stream)
    {
        size_t total_elems = input.numel();
        if (total_elems == 0) return;

        const auto& shape_dims = input.shape().dims;
        size_t ndim = shape_dims.size();
        size_t H = shape_dims[ndim - 2];
        size_t W = shape_dims[ndim - 1];

        // Ensure input is contiguous on GPU if possible
        Tensor input_contig = input.is_contiguous() ? input : input.contiguous();

        dispatch_by_dtype(input.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            const T* d_in = input_contig.data<T>();
            T* d_out = output.data<T>();
            
            T fill_val = static_cast<T>(value);

            int block_size = 256;
            int grid_size = (total_elems + block_size - 1) / block_size;

            tril_kernel<T><<<grid_size, block_size, 0, stream>>>(d_in, d_out, diagonal, fill_val, H, W, total_elems);
        });
    }
}
#endif
