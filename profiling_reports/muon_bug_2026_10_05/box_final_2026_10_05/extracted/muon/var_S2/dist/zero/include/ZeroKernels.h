#pragma once
//======================================================================
// ZeroKernels.h — small elementwise / reduction kernels used by the ZeRO-1
// optimizers (dist/zero/src/ZeroKernels.cu, built into libdist).
//
// Kept free of the tensor library on purpose: the .cu behind this header is
// compiled by nvcc with the dist flags, and the tensor library's headers are
// C++20. Callers map OwnTensor::Dtype onto ElemKind with zero_elem_kind().
//======================================================================

#include <cstdint>
#include <cuda_runtime.h>

namespace owt {
namespace zero {
namespace kernels {

// Source element types the kernels below accept. UNSUPPORTED means "use the
// caller's generic fallback", never "fail".
enum class ElemKind : int { UNSUPPORTED = -1, F32 = 0, BF16 = 1, F16 = 2 };

// dst[i] = float(src[i]) * scale, for i in [0, n). One pass instead of an
// up-cast followed by an in-place scale.
//
// Bit-identical to that two-pass form as the tensor library builds it
// (--use_fast_math): the multiply is mul.rn.ftz.f32, and it is skipped when
// scale == 1.0f, exactly as the old path skipped the scale kernel.
void cast_to_f32_scaled(const void* src, ElemKind kind, float* dst,
                        int64_t n, float scale, cudaStream_t stream);

// *d_acc += sum over i in [0, n) of (float(src[i]) * scale)^2, each term being
// exactly the value cast_to_f32_scaled() would produce for that element.
//
// Deterministic, and independent of where src sits in memory: the grid and
// the element-to-thread assignment depend only on n, per-block partials go to
// d_partials, and one single-block pass folds them in index order and adds to
// *d_acc. No atomics, so calls serialized on one stream accumulate in a fixed
// order, and the same tensor gives the same bits in any buffer layout.
//
// d_partials must hold kSumsqPartials floats and must not be shared with a
// concurrent call on another stream.
constexpr int kSumsqPartials = 1024;
void sumsq_accumulate(const void* src, ElemKind kind, int64_t n, float scale,
                      float* d_acc, float* d_partials, cudaStream_t stream);

} // namespace kernels
} // namespace zero
} // namespace owt
