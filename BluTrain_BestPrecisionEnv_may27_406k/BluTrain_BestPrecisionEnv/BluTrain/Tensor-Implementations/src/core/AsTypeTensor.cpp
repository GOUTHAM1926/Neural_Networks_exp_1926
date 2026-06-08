#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/DeviceTransfer.h"
#include "dtype/DtypeTraits.h"
#include "ops/helpers/ConversionKernels.cuh"
#include "device/DeviceCore.h"

namespace OwnTensor
{
    Tensor Tensor::as_type(Dtype new_dtype) const {
        // Edge Case: If types are the same, just return a clone
        if (new_dtype == this->dtype()) {
            return this->clone();
        }

        //  Validation: Prevent invalid complex/scalar conversions
        bool src_is_complex = is_complex(this->dtype());
        bool dst_is_complex = is_complex(new_dtype);
        if (src_is_complex != dst_is_complex) {
            throw std::runtime_error(
                "Cannot convert between complex and non-complex types. " +
                get_dtype_name(this->dtype()) + " -> " + get_dtype_name(new_dtype)
            );
        }

        // 1. Create the destination tensor on the SAME device
        Tensor new_tensor(this->shape(), TensorOptions{new_dtype, this->device()});

        // 2. Get element count
        const size_t n = this->numel();

        //  3. Handle CUDA tensors
        if (this->is_cuda()) {
            #ifdef WITH_CUDA
                // Launch GPU kernel
                device::set_cuda_device(this->device().index);
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                convert_type_cuda_generic(this->data(), this->dtype(), new_tensor.data(), new_dtype, n, stream);
            #else
                throw std::runtime_error("CUDA support not compiled");
            #endif
        } else {
            //  4. Handle CPU tensors (original logic)
            const auto* src_untyped_ptr = this->data();
            auto* dst_untyped_ptr = new_tensor.data();

            // Nested dispatch for type conversion
            dispatch_by_dtype(this->dtype(), [&](auto src_type_placeholder) {
                using SrcType = decltype(src_type_placeholder);
                const auto* src_data = reinterpret_cast<const SrcType*>(src_untyped_ptr);

                dispatch_by_dtype(new_dtype, [&](auto dst_type_placeholder) {
                    using DstType = decltype(dst_type_placeholder);
                    auto* dst_data = reinterpret_cast<DstType*>(dst_untyped_ptr);

                    // Compile-time check: both must be complex or both must be non-complex
                    constexpr bool src_is_complex = 
                        std::is_same_v<SrcType, complex32_t> ||
                        std::is_same_v<SrcType, complex64_t> ||
                        std::is_same_v<SrcType, complex128_t>;
                    
                    constexpr bool dst_is_complex = 
                        std::is_same_v<DstType, complex32_t> ||
                        std::is_same_v<DstType, complex64_t> ||
                        std::is_same_v<DstType, complex128_t>;

                    constexpr bool src_is_packed = std::is_same_v<SrcType, float4_e2m1_2x_t>;
                    constexpr bool dst_is_packed = std::is_same_v<DstType, float4_e2m1_2x_t>;

                    if constexpr (src_is_packed || dst_is_packed) {
                        if constexpr (std::is_same_v<SrcType, DstType>) {
                             for (size_t i = 0; i < n; ++i) dst_data[i] = src_data[i];
                        } else {
                             // Not supported
                             // We can't throw here at compile time, but we can at runtime.
                             // However, we must ensure this branch compiles.
                             // Since this lambda is instantiated for ALL types, we can't use static_cast here.
                             // We just throw exception at runtime if this path is taken.
                             // But we need to make sure the code is valid C++.
                             // throw is valid.
                             throw std::runtime_error("Conversion to/from packed FP4 type is not supported via as_type");
                        }
                    }
                    else if constexpr (src_is_complex == dst_is_complex) {
                        // Valid conversion: both complex or both non-complex
                        for (size_t i = 0; i < n; ++i) {
                            if constexpr (src_is_complex) {
                                // Manual complex-to-complex conversion by accessing components
                                dst_data[i] = DstType(
                                    static_cast<decltype(dst_data[i].real())>(src_data[i].real()),
                                    static_cast<decltype(dst_data[i].imag())>(src_data[i].imag())
                                );
                            } else {
                                // Non-complex types can use static_cast
                                dst_data[i] = static_cast<DstType>(src_data[i]);
                            }
                        }
                    }
                    // If compile-time check fails, this branch won't be instantiated
                });
            });
        }

        // 5. Return the newly created tensor
        return new_tensor;
    }
}