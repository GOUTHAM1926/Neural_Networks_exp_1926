#pragma once

#include "core/Tensor.h"
#ifdef WITH_CUDA//✨✨✨
#include <driver_types.h>
#endif

namespace OwnTensor
{
    Tensor matmul(const Tensor& A, const Tensor& B, [[maybe_unused]]cudaStream_t stream = 0);//✨✨✨
    Tensor addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2, float alpha = 1.0f, float beta = 1.0f, [[maybe_unused]]cudaStream_t stream = 0);
}



