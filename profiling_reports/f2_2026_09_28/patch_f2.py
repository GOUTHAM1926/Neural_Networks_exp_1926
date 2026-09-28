#!/usr/bin/env python3
"""F2 + LinearBackward fix for Tensor-Implementations (dist_side_merge @ 3a08ade).

  F2   AutogradMeta::accumulate_grad: `grad_fp32 += update.as_type(Float32)` becomes one fused kernel
       (cuda_accumulate_lowp_into_fp32, TensorOpsAdd.cu). Same bits, no full-size fp32 temporary,
       10 B/element of DRAM traffic instead of 18 B, one launch instead of two.
  LB   LinearBackward.cpp Case 2 (3-D input) declared a local `Tensor grad_weight` that shadowed the outer
       one, so autograd::linear returned an empty weight gradient.

Usage (from anywhere):
  python3 patch_f2.py <Tensor-Implementations dir>               # patch library + copy tests
  python3 patch_f2.py <Tensor-Implementations dir> --tests-only  # only copy the two tests (for the BEFORE run)
  python3 patch_f2.py <Tensor-Implementations dir> --check       # verify md5s and anchors, write nothing

Every target file must have the md5 it had at 3a08ade, and every edit must match exactly once, or nothing is
written. Originals are kept as <file>.orig_f2.
"""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))

EXPECTED_MD5 = {
    "src/core/AutogradMeta.cpp":                "26736f60b0db5237a1f61788584d8591",
    "src/TensorOps/cuda/TensorOpsAdd.cu":       "7b52ea49fe0aa3473488036b2ecd3647",
    "include/ops/TensorOps.cuh":                "e19b9e770134b254f6436add7c51c20f",
    "include/ops/helpers/KernelPolicy.h":       "3ae82ba3b93760bb18a14ac113eec5ef",
    "src/autograd/backward/LinearBackward.cpp": "ff4dbdc1abef046bcc358aaf314a4396",
}

TESTS = ["test_fused_grad_accum.cu", "test_linear_backward_3d.cu"]

# --------------------------------------------------------------------------------------------- edits
EDITS = {}

EDITS["include/ops/helpers/KernelPolicy.h"] = [(
"""inline bool disable_sm90_add_flat() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_ADD_FLAT");
    return v;
}
""",
"""inline bool disable_sm90_add_flat() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_ADD_FLAT");
    return v;
}

// Fused bf16/fp16 -> fp32 gradient accumulation — cuda_accumulate_lowp_into_fp32,
// called from AutogradMeta::accumulate_grad. BLUTRAIN_NO_FUSED_GRAD_ACC=1 restores
// `grad += update.as_type(Float32)`. Both give the same bits, so the switch exists
// for A/B timing only. Not an sm_90 kernel, so BLUTRAIN_NO_SM90 does not affect it.
inline bool disable_fused_grad_accum() {
    static const bool v = detail::env_flag_enabled("BLUTRAIN_NO_FUSED_GRAD_ACC");
    return v;
}
""")]

EDITS["include/ops/TensorOps.cuh"] = [(
"""void cuda_add_tensor_inplace( Tensor& A, const Tensor& B, cudaStream_t stream = 0);
""",
"""void cuda_add_tensor_inplace( Tensor& A, const Tensor& B, cudaStream_t stream = 0);

// acc += float(update) in one kernel, for an fp32 acc and a bf16/fp16 update that
// are both contiguous CUDA tensors of the same shape. Returns false and touches
// nothing when those conditions do not hold, so the caller can fall back.
bool cuda_accumulate_lowp_into_fp32(Tensor& acc, const Tensor& update, cudaStream_t stream = 0);
// Successful cuda_accumulate_lowp_into_fp32 calls in this process (for tests and
// for BLUTRAIN_REPORT_FUSED_GRAD_ACC=1, which prints it at exit).
uint64_t fused_grad_accum_call_count();
""")]

EDITS["src/TensorOps/cuda/TensorOpsAdd.cu"] = [(
"""#include "ops/helpers/Vectorization.cuh"
#include "profiler/RecordFunction.h"
""",
"""#include "ops/helpers/Vectorization.cuh"
#include "profiler/RecordFunction.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
"""), (
"""            size_t grid_size = get_optimal_grid_size(add_inplace_kernel_broadcast<T>, block_size, total_elems);
            add_inplace_kernel_broadcast<<<grid_size, block_size, 0, stream>>>(
                a_ptr, b_ptr, meta, total_elems
            );
        });
    }
""",
"""            size_t grid_size = get_optimal_grid_size(add_inplace_kernel_broadcast<T>, block_size, total_elems);
            add_inplace_kernel_broadcast<<<grid_size, block_size, 0, stream>>>(
                a_ptr, b_ptr, meta, total_elems
            );
        });
    }

    /*########################################################
        FUSED LOW-PRECISION GRADIENT ACCUMULATION
    ##########################################################*/

    // F2 -- acc_fp32[i] += float(update[i]) for a bf16 or fp16 update, in one pass.
    //
    // What:   one kernel reads the low-precision update (16 B = 8 values) and the fp32
    //         accumulator (2 x 16 B), widens in registers, adds, and writes the fp32 back.
    // Was:    AutogradMeta::accumulate_grad did `acc += update.as_type(Float32)`, which wrote
    //         the widened update to a full-size fp32 temporary and read it back: 18 B of DRAM
    //         traffic per element and two launches, against 10 B and one launch here.
    // Why:    widening bf16/fp16 to fp32 is exact (bf16 is the top half of an fp32; every fp16
    //         value is an fp32 value), and the add below is the same single fp32 add as
    //         add_inplace_kernel_sm90<float> in this file -- same TU, same flags, FADD.FTZ in
    //         the SASS of both -- so the result is bit-identical to the old path.
    // Proven: Tests/kernel_tests/test_fused_grad_accum.cu on H200 (sm_90): 0 mismatches against
    //         the old path after 4 accumulation steps, bf16 and fp16, sizes 1 to 33,554,437
    //         (vector, tail and misaligned-scalar paths), with +-0, subnormals, +-inf, NaN and
    //         overflow planted; 1.41-1.97x faster per call at 5-34M elements.
    namespace grad_accum {

    template <bool IS_BF16>
    __device__ __forceinline__ float widen(uint32_t bits16) {
        if constexpr (IS_BF16) {
            return __uint_as_float(bits16 << 16);                  // bf16 = top half of an fp32
        } else {
            return __half2float(__ushort_as_half(static_cast<unsigned short>(bits16)));
        }
    }

    // 8 low-precision values (one int4) into 8 floats (two float4). Element 2k is
    // the low half of 32-bit word k (little-endian).
    template <bool IS_BF16>
    __device__ __forceinline__ void add8(float4& lo, float4& hi, const int4 u) {
        const uint32_t w0 = static_cast<uint32_t>(u.x), w1 = static_cast<uint32_t>(u.y);
        const uint32_t w2 = static_cast<uint32_t>(u.z), w3 = static_cast<uint32_t>(u.w);
        lo.x = lo.x + widen<IS_BF16>(w0 & 0xFFFFu);  lo.y = lo.y + widen<IS_BF16>(w0 >> 16);
        lo.z = lo.z + widen<IS_BF16>(w1 & 0xFFFFu);  lo.w = lo.w + widen<IS_BF16>(w1 >> 16);
        hi.x = hi.x + widen<IS_BF16>(w2 & 0xFFFFu);  hi.y = hi.y + widen<IS_BF16>(w2 >> 16);
        hi.z = hi.z + widen<IS_BF16>(w3 & 0xFFFFu);  hi.w = hi.w + widen<IS_BF16>(w3 >> 16);
    }

    // Both pointers 16-B aligned. Grid-stride, 2x unrolled, the same shape as
    // add_inplace_kernel_sm90.
    template <bool IS_BF16, int UNROLL = 2>
    __global__ __launch_bounds__(256)
    void accum_lowp_fp32_vec_kernel(float* __restrict__ acc,
                                    const uint16_t* __restrict__ upd,
                                    size_t n)
    {
        const size_t n_vec  = n / 8;
        const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        const size_t stride = (size_t)blockDim.x * gridDim.x;

        float4*     __restrict__ a_v = reinterpret_cast<float4*>(acc);
        const int4* __restrict__ u_v = reinterpret_cast<const int4*>(upd);

        size_t i = tid;
        for (; i + (UNROLL - 1) * stride < n_vec; i += UNROLL * stride) {
            int4 u[UNROLL];
            float4 lo[UNROLL], hi[UNROLL];
            #pragma unroll
            for (int k = 0; k < UNROLL; k++) {
                const size_t j = i + k * stride;
                u[k] = u_v[j]; lo[k] = a_v[2 * j]; hi[k] = a_v[2 * j + 1];
            }
            #pragma unroll
            for (int k = 0; k < UNROLL; k++) {
                const size_t j = i + k * stride;
                add8<IS_BF16>(lo[k], hi[k], u[k]);
                a_v[2 * j] = lo[k]; a_v[2 * j + 1] = hi[k];
            }
        }
        for (; i < n_vec; i += stride) {
            float4 lo = a_v[2 * i], hi = a_v[2 * i + 1];
            add8<IS_BF16>(lo, hi, u_v[i]);
            a_v[2 * i] = lo; a_v[2 * i + 1] = hi;
        }

        // Ragged tail (n % 8, at most 7 elements), first threads of the grid.
        const size_t tail_base  = n_vec * 8;
        const size_t tail_count = n - tail_base;
        if (tid < tail_count) {
            acc[tail_base + tid] = acc[tail_base + tid] + widen<IS_BF16>(upd[tail_base + tid]);
        }
    }

    // Either pointer not 16-B aligned (a view with an element offset). Same
    // arithmetic, one element per iteration.
    template <bool IS_BF16>
    __global__ __launch_bounds__(256)
    void accum_lowp_fp32_scalar_kernel(float* __restrict__ acc,
                                       const uint16_t* __restrict__ upd,
                                       size_t n)
    {
        const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        const size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = tid; i < n; i += stride) {
            acc[i] = acc[i] + widen<IS_BF16>(upd[i]);
        }
    }

    std::atomic<uint64_t> calls{0};

    void report_at_exit() {
        std::fprintf(stderr, "[grad-accum] fused bf16/fp16 -> fp32 accumulations: %llu\\n",
                     static_cast<unsigned long long>(calls.load()));
    }

    template <bool IS_BF16>
    void launch(float* a, const uint16_t* u, size_t n, bool aligned, cudaStream_t stream) {
        const int block = 256;
        if (aligned) {
            size_t grid = get_optimal_grid_size(accum_lowp_fp32_vec_kernel<IS_BF16, 2>, block, n / 8);
            accum_lowp_fp32_vec_kernel<IS_BF16, 2><<<grid, block, 0, stream>>>(a, u, n);
        } else {
            size_t grid = get_optimal_grid_size(accum_lowp_fp32_scalar_kernel<IS_BF16>, block, n);
            accum_lowp_fp32_scalar_kernel<IS_BF16><<<grid, block, 0, stream>>>(a, u, n);
        }
    }

    }  // namespace grad_accum

    uint64_t fused_grad_accum_call_count() { return grad_accum::calls.load(); }

    bool cuda_accumulate_lowp_into_fp32(Tensor& acc, const Tensor& update, cudaStream_t stream)
    {
        const Dtype ud = update.dtype();
        if (acc.dtype() != Dtype::Float32) return false;
        if (ud != Dtype::Bfloat16 && ud != Dtype::Float16) return false;
        if (!acc.device().is_cuda() || !update.device().is_cuda()) return false;
        if (acc.device().index != update.device().index) return false;
        if (!acc.is_contiguous() || !update.is_contiguous()) return false;
        if (acc.shape().dims != update.shape().dims) return false;

        static const bool report = [] {
            const bool on = cuda::detail::env_flag_enabled("BLUTRAIN_REPORT_FUSED_GRAD_ACC");
            if (on) std::atexit(grad_accum::report_at_exit);
            return on;
        }();
        (void)report;

        const size_t n = acc.numel();
        if (n == 0) return true;

        CUDA_KERNEL_SCOPE("cuda_accumulate_lowp_into_fp32", stream);
        float* a = static_cast<float*>(acc.data());
        const uint16_t* u = static_cast<const uint16_t*>(update.data());
        const bool aligned = (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(u) % 16 == 0);
        if (ud == Dtype::Bfloat16) grad_accum::launch<true >(a, u, n, aligned, stream);
        else                       grad_accum::launch<false>(a, u, n, aligned, stream);

        // A failed launch would silently drop a gradient; fail loudly instead.
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cuda_accumulate_lowp_into_fp32: launch failed: ") +
                                     cudaGetErrorString(err));
        }
        grad_accum::calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
""")]

EDITS["src/core/AutogradMeta.cpp"] = [(
# Two lines, so the commented-out copy of this file further down ("// #include ...") cannot match.
"""#include "autograd/Hooks.h"
#include "ops/TensorOps.h" // Needed for operator+
""",
"""#include "autograd/Hooks.h"
#include "ops/TensorOps.h" // Needed for operator+
#ifdef WITH_CUDA
#include "ops/TensorOps.cuh"          // cuda_accumulate_lowp_into_fp32
#include "ops/helpers/KernelPolicy.h" // cuda::disable_fused_grad_accum
#include "device/DeviceCore.h"        // cuda::getCurrentStream
#endif
"""), (
"""        if (update.dtype() != grad_->dtype()) {
            *grad_ += update.as_type(grad_->dtype());
        } else {
""",
"""        if (update.dtype() != grad_->dtype()) {
#ifdef WITH_CUDA
            // F2 -- fused gradient accumulation.
            // What:   cuda_accumulate_lowp_into_fp32 (TensorOpsAdd.cu) widens the bf16/fp16
            //         update to fp32 in registers and adds it straight into grad_, one kernel.
            // Was:    only the `*grad_ += update.as_type(...)` line below: a new full-size fp32
            //         temporary per bf16 weight per micro-step, 2 kernels, 18 B/element of DRAM
            //         traffic. Now 10 B/element, 1 kernel, no temporary.
            // Why:    bf16/fp16 -> fp32 is exact and the add is the same FADD.FTZ, so the bits
            //         cannot change; the temporary only carried the widened value between kernels.
            // Proven: Tests/kernel_tests/test_fused_grad_accum.cu on H200: 0 mismatches against
            //         the old line after 4 accumulation steps, bf16 and fp16, 1 to 33.5M elements,
            //         with +-0, subnormals, +-inf, NaN and overflow; 1.41-1.97x faster per call.
            // Falls through (returns false) for CPU, non-contiguous or shape-mismatched updates.
            // BLUTRAIN_NO_FUSED_GRAD_ACC=1 forces the old line (A/B runs).
            if (!cuda::disable_fused_grad_accum() &&
                cuda_accumulate_lowp_into_fp32(*grad_, update, cuda::getCurrentStream())) {
                return;
            }
#endif
            *grad_ += update.as_type(grad_->dtype());
        } else {
""")]

EDITS["src/autograd/backward/LinearBackward.cpp"] = [(
"""            Tensor grad_weight = Tensor(saved_weight_.shape(), saved_weight_.opts().with_device(grad_output.device()));
""",
"""            // What:   assign the OUTER grad_weight (declared at the top of apply()).
            // Was:    `Tensor grad_weight = ...` declared a new local here, so the outer one stayed
            //         empty and "2. Gradient for Weight" below returned it: autograd::linear with a
            //         3-D input gave the weight NO gradient. The engine skips empty gradients, so
            //         nothing reported it. In since 4a62424 (2026-03-10).
            // Why:    Case 1 above already assigns the outer variable; this makes Case 2 match.
            //         The trainer never reached it (nn::Linear uses autograd::matmul / addmm), but
            //         DTensor::Linear (dist/Tensor-Parallelism) calls autograd::linear.
            // Proven: Tests/kernel_tests/test_linear_backward_3d.cu on H200: before, L1/L2/L5 FAIL
            //         (no weight gradient); after, 16/16 PASS, and the weight gradient equals the
            //         autograd::matmul path bit for bit (fp32 and bf16, with and without bias).
            grad_weight = Tensor(saved_weight_.shape(), saved_weight_.opts().with_device(grad_output.device()));
""")]

# --------------------------------------------------------------------------------------------- driver
def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = os.path.abspath(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "--apply"
    if mode not in ("--apply", "--tests-only", "--check"):
        sys.exit(f"unknown mode {mode}")
    if not os.path.isfile(os.path.join(root, "src/core/AutogradMeta.cpp")):
        sys.exit(f"{root} does not look like Tensor-Implementations")

    # copy the tests (both modes except --check)
    if mode != "--check":
        for t in TESTS:
            src = os.path.join(HERE, "tests", t)
            dst = os.path.join(root, "Tests/kernel_tests", t)
            shutil.copyfile(src, dst)
            print(f"copied test  {dst}  md5={md5(dst)}")
    if mode == "--tests-only":
        return

    # 1. every target file must be at its expected md5 (unpatched, unmodified)
    bad = []
    for rel, want in EXPECTED_MD5.items():
        got = md5(os.path.join(root, rel))
        if got != want:
            bad.append(f"  {rel}: md5 {got}, expected {want}")
    if bad:
        sys.exit("md5 mismatch -- refusing to patch (already patched, or a different revision):\n" + "\n".join(bad))

    # 2. every edit must match exactly once; build all new contents in memory first
    new_text = {}
    for rel, edits in EDITS.items():
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            text = f.read()
        for i, (old, new) in enumerate(edits):
            n = text.count(old)
            if n != 1:
                sys.exit(f"{rel} edit {i + 1}: anchor found {n} times, expected exactly 1 -- nothing written")
            text = text.replace(old, new)
        new_text[rel] = text
    print("all md5s and anchors OK (%d files, %d edits)" % (len(EDITS), sum(len(e) for e in EDITS.values())))
    if mode == "--check":
        return

    # 3. write, keeping originals
    for rel, text in new_text.items():
        p = os.path.join(root, rel)
        shutil.copyfile(p, p + ".orig_f2")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"patched  {rel}  md5={md5(p)}  (original kept as {rel}.orig_f2)")

if __name__ == "__main__":
    main()
