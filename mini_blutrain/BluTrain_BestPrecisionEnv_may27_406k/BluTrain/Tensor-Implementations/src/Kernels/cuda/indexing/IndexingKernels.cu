#include "ops/helpers/IndexingKernels.h"
#include "dtype/Types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace OwnTensor {
namespace cuda {

// Explicit instantiations for all types used in dispatch
#define INSTANTIATE_GATHER_IDX(T, T_idx) \
template void gather_cuda<T, T_idx>( \
    const T* input, \
    const T_idx* indices, \
    T* output, \
    int64_t dim, \
    const int64_t* in_strides, \
    const int64_t* idx_strides, \
    const int64_t* out_strides, \
    const int64_t* in_dims, \
    const int64_t* idx_dims, \
    int ndim, \
    int64_t numel, \
    cudaStream_t stream \
);

#define INSTANTIATE_GATHER(T) \
    INSTANTIATE_GATHER_IDX(T, int16_t) \
    INSTANTIATE_GATHER_IDX(T, int32_t) \
    INSTANTIATE_GATHER_IDX(T, int64_t) \
    INSTANTIATE_GATHER_IDX(T, uint8_t) \
    INSTANTIATE_GATHER_IDX(T, uint16_t) \
    INSTANTIATE_GATHER_IDX(T, uint32_t) \
    INSTANTIATE_GATHER_IDX(T, uint64_t)

INSTANTIATE_GATHER(float)
INSTANTIATE_GATHER(double)
INSTANTIATE_GATHER(int16_t)
INSTANTIATE_GATHER(int32_t)
INSTANTIATE_GATHER(int64_t)
INSTANTIATE_GATHER(uint8_t)
INSTANTIATE_GATHER(uint16_t)
INSTANTIATE_GATHER(uint32_t)
INSTANTIATE_GATHER(uint64_t)
INSTANTIATE_GATHER(float16_t)
INSTANTIATE_GATHER(bfloat16_t)
INSTANTIATE_GATHER(bool)
INSTANTIATE_GATHER(complex32_t)
INSTANTIATE_GATHER(complex64_t)
INSTANTIATE_GATHER(complex128_t)

} // namespace cuda
} // namespace OwnTensor
