#!/usr/bin/env python3
"""GPU as_type fixes in src/Kernels/cuda/misc/ConversionKernels.cu (Tensor-Implementations 97735ad).

  1. convert_type_kernel sent EVERY type through a float temporary. float holds 24 bits, so int32
     above 2^24, int64, uint32/64 and double lost digits (int32 16,777,217 -> int64 16,777,216;
     double 10,000,000,001 -> 10,000,000,000), and double -> bf16/fp16 was rounded twice.
     Now, like the CPU as_type (AsTypeTensor.cpp):
       - bf16/fp16 source: unchanged (their values are exact in float, the temporary loses nothing);
       - wide source -> bf16/fp16: Dst(input), the Types.h one-rounding constructors (GPU intrinsics);
       - wide source -> wide destination: static_cast<Dst>(input), direct.
  2. complex -> complex was silently SKIPPED (output left uninitialised, no error). Now converted
     component by component, as the CPU does. complex <-> real now throws (as_type already refuses it).
  3. The element index was a 32-bit product (blockIdx.x * blockDim.x), which wraps above 2^32
     elements. Now 64-bit.

Run from the Tensor-Implementations root:  python3 patch_conversion_kernels.py
"""
import hashlib, sys

PATH = "src/Kernels/cuda/misc/ConversionKernels.cu"
WANT = "8f1df8705ea479abeeb154963b32217e"

EDITS = [
# ---- 1 + 3: the generic kernel ------------------------------------------------------------
("""template<typename Src, typename Dst>
__global__ void convert_type_kernel(const Src* __restrict__ input,
                                  Dst* __restrict__ output,
                                  int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val;
        if constexpr (std::is_same_v<Src, float16_t>) {
            __half h;
            *reinterpret_cast<uint16_t*>(&h) = input[idx].raw_bits;
            val = __half2float(h);
        } else if constexpr (std::is_same_v<Src, bfloat16_t>) {
            __nv_bfloat16 b;
            *reinterpret_cast<uint16_t*>(&b) = input[idx].raw_bits;
            val = __bfloat162float(b);
        } else {
            val = static_cast<float>(input[idx]);
        }

        if constexpr (std::is_same_v<Dst, float16_t>) {
            __half h = __float2half(val);
            output[idx].raw_bits = *reinterpret_cast<uint16_t*>(&h);
        } else if constexpr (std::is_same_v<Dst, bfloat16_t>) {
            __nv_bfloat16 b = __float2bfloat16(val);
            output[idx].raw_bits = *reinterpret_cast<uint16_t*>(&b);
        } else {
            output[idx] = static_cast<Dst>(val);
        }
    }
}
""",
"""template<typename Src, typename Dst>
__global__ void convert_type_kernel(const Src* __restrict__ input,
                                  Dst* __restrict__ output,
                                  int64_t n) {
    // Modified: 64-bit index. It was blockIdx.x * blockDim.x (a 32-bit product), which wraps for
    // tensors above 2^32 elements.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Modified: the float temporary is now used ONLY for bf16/fp16 sources. It used to carry
        // every conversion, and float holds 24 bits, so every wider value lost digits on the way:
        // int32 above 2^24, int64, uint32/64 and double (e.g. int32 16,777,217 -> int64 16,777,216;
        // int64 or double 10,000,000,001 -> 10,000,000,000), and double -> bf16/fp16 was rounded
        // twice (double -> float -> bf16). The CPU as_type (AsTypeTensor.cpp) converts directly;
        // this now does the same, so GPU and CPU give the same result.
        if constexpr (std::is_same_v<Src, float16_t> || std::is_same_v<Src, bfloat16_t>) {
            // 16-bit source: unchanged. Every bf16/fp16 value is exact in float, so nothing is lost.
            float val;
            if constexpr (std::is_same_v<Src, float16_t>) {
                __half h;
                *reinterpret_cast<uint16_t*>(&h) = input[idx].raw_bits;
                val = __half2float(h);
            } else {
                __nv_bfloat16 b;
                *reinterpret_cast<uint16_t*>(&b) = input[idx].raw_bits;
                val = __bfloat162float(b);
            }
            if constexpr (std::is_same_v<Dst, float16_t>) {
                __half h = __float2half(val);
                output[idx].raw_bits = *reinterpret_cast<uint16_t*>(&h);
            } else if constexpr (std::is_same_v<Dst, bfloat16_t>) {
                __nv_bfloat16 b = __float2bfloat16(val);
                output[idx].raw_bits = *reinterpret_cast<uint16_t*>(&b);
            } else {
                output[idx] = static_cast<Dst>(val);
            }
        } else if constexpr (std::is_same_v<Dst, float16_t> || std::is_same_v<Dst, bfloat16_t>) {
            // Wide source -> bf16/fp16: the Types.h constructors, which round ONCE and use the GPU
            // intrinsics on the device (float: __float2bfloat16_rn / __float2half_rn, the same
            // instruction as before, so fp32 -> bf16/fp16 is bit-identical; double:
            // __double2bfloat16 / __double2half; integers: __ll2bfloat16_rn etc.).
            output[idx] = Dst(input[idx]);
        } else {
            // Wide source -> wide destination: direct, as the CPU does (no float in between).
            output[idx] = static_cast<Dst>(input[idx]);
        }
    }
}
"""),
# ---- 2: complex kernel + launcher (before the generic dispatcher) ---------------------------
("""void convert_type_cuda_generic(const void* input, Dtype src_dtype, void* output, Dtype dst_dtype, int64_t n, cudaStream_t stream) {""",
"""// Modified: new kernel. complex -> complex on the GPU used to be SKIPPED (see
// convert_type_cuda_generic below). Converts component by component, exactly as the CPU as_type
// does (AsTypeTensor.cpp: DstType(static_cast<R>(src.real()), static_cast<R>(src.imag()))), so a
// complex32 component gets the Types.h one-rounding fp16 conversion.
template<typename Src, typename Dst>
__global__ void convert_complex_kernel(const Src* __restrict__ input,
                                       Dst* __restrict__ output,
                                       int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        using R = decltype(output[idx].real());
        output[idx] = Dst(static_cast<R>(input[idx].real()), static_cast<R>(input[idx].imag()));
    }
}

template<typename Src, typename Dst>
static void convert_complex_cuda(const Src* input, Dst* output, int64_t n, cudaStream_t stream) {
    if (n == 0) return;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    convert_complex_kernel<Src, Dst><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(input, output, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("convert_complex_cuda kernel launch failed: ") + cudaGetErrorString(err));
}

void convert_type_cuda_generic(const void* input, Dtype src_dtype, void* output, Dtype dst_dtype, int64_t n, cudaStream_t stream) {"""),
# ---- 2: dispatcher: convert complex->complex, refuse complex<->real ---------------------------
("""            if constexpr (is_complex_t<SrcType>::value || is_complex_t<DstType>::value) {
                // Skip complex types for now on GPU
            } else {""",
"""            if constexpr (is_complex_t<SrcType>::value && is_complex_t<DstType>::value) {
                // Modified: complex -> complex is converted. It was skipped here ("Skip complex
                // types for now on GPU"): the output was left uninitialised and no error was raised.
                convert_complex_cuda<SrcType, DstType>(
                    reinterpret_cast<const SrcType*>(input),
                    reinterpret_cast<DstType*>(output),
                    n, stream);
            } else if constexpr (is_complex_t<SrcType>::value || is_complex_t<DstType>::value) {
                // Modified: complex <-> real is refused, as Tensor::as_type already refuses it
                // (AsTypeTensor.cpp). It was silently skipped, leaving the output uninitialised.
                throw std::runtime_error(
                    "convert_type_cuda_generic: cannot convert between complex and non-complex types "
                    "(dtype " + std::to_string(static_cast<int>(src_dtype)) + " -> " +
                    std::to_string(static_cast<int>(dst_dtype)) + ")");
            } else {"""),
]

def main():
    data = open(PATH, "rb").read()
    got = hashlib.md5(data).hexdigest()
    if got != WANT:
        sys.exit(f"ABORT: {PATH} md5 {got}, expected {WANT}. Nothing written.")
    text = data.decode()
    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            sys.exit(f"ABORT: edit {i}: pattern found {n} times (need 1). Nothing written.")
        text = text.replace(old, new)
    open(PATH, "w").write(text)
    print(f"patched {PATH}: {len(EDITS)} edits")

if __name__ == "__main__":
    main()
