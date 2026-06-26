#pragma once

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "dtype/Types.h"

namespace OwnTensor {
namespace detail {

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION TRAITS (Custom Struct → Native CUDA Type)
// ═══════════════════════════════════════════════════════════

template<typename T> struct ToCudaNative { using type = T; };
template<> struct ToCudaNative<float16_t> { using type = __half; };
template<> struct ToCudaNative<bfloat16_t> { using type = __nv_bfloat16; };

template<typename T>
using CudaNativeType = typename ToCudaNative<T>::type;

} // namespace detail
} // namespace OwnTensor

#endif // WITH_CUDA
