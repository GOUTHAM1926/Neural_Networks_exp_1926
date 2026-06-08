#include "core/Tensor.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "autograd/Node.h"
#include "autograd/Hooks.h"
#include "autograd/Engine.h"
#include "dtype/Types.h"
#include "dtype/fp4.h"
#include "device/AllocatorRegistry.h"
#include "device/DeviceTransfer.h"
#include "device/Device.h"
#include "device/PinnedCPUAllocator.h"
#include "core/Views/ViewUtils.h"
#include "ops/helpers/ConditionalOps.h"
#include "dtype/DtypeTraits.h"
#include "core/TensorDispatch.h"
#include "core/TensorDataManip.h"
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include "core/Views/contiguous_kernel.h"
#include "ops/helpers/ConversionKernels.cuh"

#endif

#ifdef WITH_DEBUG
#endif

namespace OwnTensor 
{
    // ========================================================================
    // Constructors - Now create TensorImpl
    // ========================================================================
    
    Tensor::Tensor(Shape shape, Dtype dtype, DeviceIndex device, bool requires_grad, Pinned_Flag pinten) {
        #ifdef WITH_DEBUG
        std::cout << "Tensor constructor: device=" << (device.is_cpu() ? "CPU" : "CUDA") << "\n" << std::endl;
        #endif

        // Silently ignore pinten for GPU tensors 
        if (!device.is_cpu()) {
            pinten = Pinned_Flag::None;
        }

        // == CUDA DEVICE SETTING AND CHECK == //
        if (device.is_cuda()) {
            #ifdef WITH_CUDA
            if (!device::cuda_available()) {
                throw std::runtime_error("CUDA is not available but CUDA device requested");
            }
            cudaError_t err = cudaSetDevice(device.index);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
            }
            #else   
            throw std::runtime_error("CUDA support not compiled");        
            #endif
        }

        // Validate shape
        if (shape.dims.empty()) {
            // Allow empty tensors
            impl_ = make_intrusive<TensorImpl>(shape, dtype, device, requires_grad);
            return;
        }

        for (size_t i = 0; i < shape.dims.size(); ++i) {
            if (shape.dims[i] < 0) {
                throw std::runtime_error("All dimensions must be non-negative, got dimension " + 
                                        std::to_string(i) + " = " + std::to_string(shape.dims[i]));
            }
            if (shape.dims[i] == 0) {        
                throw std::runtime_error("Zero dimensions are not allowed, got dimension " + 
                                        std::to_string(i) + " = 0");
            }
        }

        // Get pinned allocator if flag is set
        Allocator* alloc = nullptr;
        if (pinten != Pinned_Flag::None) {
            alloc = AllocatorRegistry::get_pinned_cpu_allocator(pinten);
        }

        // Create TensorImpl - it handles everything
        impl_ = make_intrusive<TensorImpl>(shape, dtype, device, requires_grad, alloc);
    }

    // Tensor Options constructor
    Tensor::Tensor(Shape shape, TensorOptions opts) {
        // Silently ignore pinten for GPU tensors.
        if (!opts.device.is_cpu()) {
            opts.pinten = Pinned_Flag::None;
        }

        // Validate shape (copied from main constructor)
        if (shape.dims.empty()) {
            impl_ = make_intrusive<TensorImpl>(shape, opts.dtype, opts.device, opts.requires_grad);
            return;
        }

        // == CUDA DEVICE SETTING AND CHECK == //
        if (opts.device.is_cuda()) {
            #ifdef WITH_CUDA
            if (!device::cuda_available()) {
                throw std::runtime_error("CUDA is not available but CUDA device requested");
            }
            cudaError_t err = cudaSetDevice(opts.device.index);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
            }
            #else   
            throw std::runtime_error("CUDA support not compiled");        
            #endif
        }

        // Check for negative/zero dims
        for (size_t i = 0; i < shape.dims.size(); ++i) {
             if (shape.dims[i] <= 0) { // simplified check
                 throw std::runtime_error("Invalid dimension size");
             }
        }

        Allocator* alloc = nullptr;
        if (opts.pinten != Pinned_Flag::None) {
             alloc = AllocatorRegistry::get_pinned_cpu_allocator(opts.pinten);
        }

        impl_ = make_intrusive<TensorImpl>(shape, opts.dtype, opts.device, opts.requires_grad, alloc);
    }

    // Private constructor for creating views (shares TensorImpl's storage)
    Tensor::Tensor(intrusive_ptr<TensorImpl> impl,
                   Shape shape,
                   Stride stride,
                   size_t offset) {
        // Create new TensorImpl with shared storage
        // Now safely shares the intrusive_ptr<Storage>
        impl_ = make_intrusive<TensorImpl>(
            impl->storage_ptr(),
            shape,
            stride,
            offset,
            impl->dtype(),
            impl->device()
            // impl // Pass base_impl to keep original alive (for views)
        );

        if (impl->requires_grad()) {
            impl_->set_requires_grad(true);
        }
    }

    
    // ========================================================================
    // Utility Methods - Delegate to TensorImpl
    // ========================================================================
    size_t Tensor::numel() const {
        if (!impl_) return 0;
        return impl_->numel();
    }

    size_t Tensor::nbytes() const {
        if (!impl_) return 0;
        return impl_->nbytes();
    }

    size_t Tensor::grad_nbytes() const {
        if (!impl_ || !impl_->requires_grad()) return 0;
        return impl_->nbytes();
    }

    // ========================================================================
    // Gradient Access Methods
    // ========================================================================
    
    void* Tensor::grad() {
        if (!impl_ || !impl_->has_autograd_meta()) {
            return nullptr;
        }
        return impl_->mutable_grad().data();
    }

    const void* Tensor::grad() const {
        if (!impl_ || !impl_->has_autograd_meta()) {
            return nullptr;
        }
        return impl_->grad().data();
    }

    template<typename T>
    T* Tensor::grad() {
        if (!impl_ || !impl_->has_autograd_meta()) {
            return nullptr;
        }
        return impl_->mutable_grad().data<T>();
    }

    template<typename T>
    const T* Tensor::grad() const {
        if (!impl_ || !impl_->has_autograd_meta()) {
            return nullptr;
        }
        return impl_->grad().data<T>();
    }

    bool Tensor::is_contiguous() const
    {
        if (!impl_) return true;
        
        // Check if strides match row-major layout
        int64_t expected_stride = 1;
        const auto& dims = impl_->sizes().dims;
        const auto& strides = impl_->strides().strides;
       
        for (int i = dims.size() - 1; i >= 0; --i)
        {
            if (strides[i] != expected_stride)
            {
                return false;
            }
            expected_stride *= dims[i];
        }
        return true;
    }

    void Tensor::print_meta() const {
        
        // Shape
        std::cout << "Shape: [ ";
        for (auto i : this->shape().dims) {
            std::cout << i << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Stride: [ ";
        for (auto i : this->stride().strides) {
            std::cout << i << " ";
        }
        std::cout << "]" << std::endl;

        // Dtype
        std::cout << "Dtype: " << get_dtype_name(this->dtype()) << std::endl;

        // Size
        std::cout << "Tensor Size: " << this->nbytes() << " bytes - " << this->nbytes() / (1024 * 1024) << " mb" << std::endl; 
        std::cout << "Grad Size: " << this->grad_nbytes() << " bytes - " << this->grad_nbytes() / (1024 * 1024) << " mb" << std::endl; 

        std::cout << "Storage Offset: " << this->storage_offset() << std::endl; 

    }

    void Tensor::print_shape(Shape s) {
        std::cout << "Shape: [ ";
        for (auto i : s.dims) {
            std::cout << i << " ";
        }
        std::cout << "]" << std::endl;
    }

    void Tensor::print_stride(Stride s) {
        std::cout << "Stride: [ ";
        for (auto i : s.strides) {
            std::cout << i << " ";
        }
        std::cout << "]" << std::endl;
    }
    
    // ========================================================================
    // Contiguous Method
    // ========================================================================
    
    Tensor Tensor::contiguous() const {
        if (!impl_) {
            throw std::runtime_error("contiguous: tensor is not initialized");
        }
        
        // If already contiguous with zero offset, return self (no copy needed)
        // If already contiguous with zero offset, return self (no copy needed)
        if (is_contiguous() && storage_offset() == 0) {
            // Tensor out(impl_->sizes(), dtype(), device(), requires_grad());
            // device::copy_memory(out.data(), impl_->device(), data(), impl_->device(), nbytes());
            // return out;
            return *this;
        }

        // Allocate destination with row-major layout on the same device
        Tensor out(impl_->sizes(), dtype(), device(), requires_grad());
        Allocator* alloc = AllocatorRegistry::get_allocator(impl_->device().device);

        const size_t bytes_per_elem = dtype_size(dtype());
        const int64_t total_elems = static_cast<int64_t>(numel());
        const size_t D = impl_->sizes().dims.size();

        if (is_cpu()) {
            std::vector<int64_t> idx(D, 0);

            auto bump = [&](std::vector<int64_t>& v)->bool {
                for (int d = int(D) - 1; d >= 0; --d) {
                    if (++v[d] < impl_->sizes().dims[d]) return true;
                    v[d] = 0;
                }
                return false;
            };

            uint8_t* dst = reinterpret_cast<uint8_t*>(out.data());
            size_t write_pos = 0;

            do {
                // Compute element offset in elements
                int64_t elem_off = 0;
                for (size_t d = 0; d < D; ++d) {
                    elem_off += idx[d] * impl_->strides().strides[d];
                }

                // data() already accounts for storage_offset
                const uint8_t* src_elem_ptr =
                    reinterpret_cast<const uint8_t*>(data())
                    + elem_off * bytes_per_elem;

                std::memcpy(dst + write_pos, src_elem_ptr, bytes_per_elem);
                write_pos += bytes_per_elem;

            } while (bump(idx));

            return out;
        }
        #ifdef WITH_CUDA
            else if (is_cuda()) {
                cudaStream_t stream = 0;

                // Pass dims/strides as host arrays — kernel receives them by value
                // (no cudaMallocAsync overhead)

                // Pass dims/strides as host arrays — kernel receives them by value
                // (no cudaMallocAsync overhead)
                contiguous_strided_copy_cuda(
                    data(), out.data(), total_elems,
                    impl_->sizes().dims.data(),
                    impl_->strides().strides.data(),
                    static_cast<int32_t>(D),
                    0,
                    static_cast<int32_t>(bytes_per_elem),
                    stream
                );


                return out;
            }
            #endif
            else {
                throw std::runtime_error("Unknown device in Tensor::contiguous()");
            }
        }

    
    // ========================================================================
    // Clone and Copy Methods
    // ========================================================================
    
    Tensor Tensor::clone() const {
        if (!impl_) {
            throw std::runtime_error("clone: tensor is not initialized");
        }
        
        // Edge case: Empty tensor
        if (numel() == 0) {
            return Tensor(impl_->sizes(), impl_->dtype(), impl_->device(), requires_grad());
        }
        
        // Edge case: Non-contiguous or has storage_offset - materialize first
        if (!is_contiguous() || storage_offset() != 0) {
            try {
                Tensor src_contig = contiguous();
                Tensor result(src_contig.shape(), dtype(), device(), requires_grad());
                device::copy_memory(result.data(), impl_->device(), src_contig.data(), impl_->device(), src_contig.nbytes());

                return result;
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("clone failed (contiguous): ") + e.what());
            }
        }
        
        // Contiguous path: direct clone
        try {
            Tensor result(impl_->sizes(), dtype(), device(), requires_grad());
            device::copy_memory(result.data(), impl_->device(), data(), impl_->device(), nbytes());
            
            return result;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("clone failed: ") + e.what());
        }
    }

    Tensor Tensor::detach() const {
        if (!impl_) {
            throw std::runtime_error("detach: tensor is not initialized");
        }
        
        // Create new TensorImpl sharing the same storage
        // But with NO base_impl, NO autograd meta, NO required grad
        // The new tensor is a leaf node detached from the graph
        auto new_impl = make_intrusive<TensorImpl>(
            impl_->storage_ptr(),
            impl_->sizes(),
            impl_->strides(),
            impl_->storage_offset(),
            impl_->dtype(),
            impl_->device()
            // base_impl defaults to empty
        );
        
        return Tensor(std::move(new_impl));
    }

    Tensor& Tensor::copy_(const Tensor& src) {
        if (!impl_ || !src.impl_) {
            throw std::runtime_error("copy_: tensor is not initialized");
        }
        
        // Edge case: Self-copy is no-op
        if (this == &src || data() == src.data()) return *this;
        
        // Edge case: Empty tensor
        if (numel() == 0 && src.numel() == 0) {
            return *this;
        }
        
        // Edge case: Size validation
        if (numel() != src.numel()) {
            throw std::runtime_error(
                "copy_: size mismatch. Destination has " + 
                std::to_string(numel()) + " elements but source has " + 
                std::to_string(src.numel())
            );
        }
        
        if (dtype() != src.dtype()) {
            throw std::runtime_error("copy_: dtype mismatch");
        }
        
        if (numel() == 0) return *this;
        
        if (!is_contiguous() || storage_offset() != 0) {
            throw std::runtime_error("copy_: destination must be contiguous");
        }
        
        // Materialize non-contiguous source
        const Tensor* src_ptr = &src;
        Tensor src_contig;
        if (!src.is_contiguous()) {
            src_contig = src.contiguous();
            src_ptr = &src_contig;
        }
        
        try {
            device::copy_memory(
                this->data(), this->device(),
                src_ptr->data(), src_ptr->device(),
                src_ptr->nbytes()
            );
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("copy_ failed: ") + e.what());
        }
        
        return *this;
    }

    // ========================================================================
    // Storage and Memory Info Methods
    // ========================================================================
    
    size_t Tensor::storage_offset() const {
        if (!impl_) return 0;
        return impl_->storage_offset();
    }
    
    bool Tensor::owns_data() const {
        // In new architecture, storage ownership is managed by Storage/TensorImpl
        // Views share storage but don't have unique ownership
        if (!impl_) return false;
        return impl_->use_count() == 1;
    }
    
    bool Tensor::owns_grad() const {
        if (!impl_ || !impl_->has_autograd_meta()) return false;
        return true;  // Gradient always owned by autograd_meta
    }
    
    bool Tensor::is_valid() const {
        return impl_ && impl_->storage().is_valid();
    }

    // Determine element size based on data type
    size_t Tensor::dtype_size(Dtype d) {
        switch(d) {
            case Dtype::Bool:           return 1;
            case Dtype::Int8:           return dtype_traits<Dtype::Int8>::size;
            case Dtype::Int16:          return dtype_traits<Dtype::Int16>::size;
            case Dtype::Int32:          return dtype_traits<Dtype::Int32>::size;
            case Dtype::Int64:          return dtype_traits<Dtype::Int64>::size;
            case Dtype::UInt8:          return dtype_traits<Dtype::UInt8>::size;
            case Dtype::UInt16:         return dtype_traits<Dtype::UInt16>::size;
            case Dtype::UInt32:         return dtype_traits<Dtype::UInt32>::size;
            case Dtype::UInt64:         return dtype_traits<Dtype::UInt64>::size;
            case Dtype::Bfloat16:       return dtype_traits<Dtype::Bfloat16>::size;
            case Dtype::Float16:        return dtype_traits<Dtype::Float16>::size;
            case Dtype::Float32:        return dtype_traits<Dtype::Float32>::size;
            case Dtype::Float64:        return dtype_traits<Dtype::Float64>::size;
            case Dtype::Complex32:      return dtype_traits<Dtype::Complex32>::size;
            case Dtype::Complex64:      return dtype_traits<Dtype::Complex64>::size;
            case Dtype::Complex128:     return dtype_traits<Dtype::Complex128>::size;
            case Dtype::Float4_e2m1:    return dtype_traits<Dtype::Float4_e2m1>::size;
            case Dtype::Float4_e2m1_2x: return dtype_traits<Dtype::Float4_e2m1_2x>::size;
            default: throw std::runtime_error("Unsupported data type");
        }
    }

    
    // ========================================================================
    // Device Transfer Methods
    // ========================================================================
    
    Tensor Tensor::to(DeviceIndex device) const {
        if (!impl_) {
            throw std::runtime_error("to: tensor is not initialized");
        }
        
        // Same device - just return a copy
        if (device.device == impl_->device().device && device.index == impl_->device().index) {
            return *this;
        }
        
        // Handle views: Must be contiguous before device transfer
        if (!is_contiguous()) {
            Tensor contig = contiguous();
            return contig.to(device);
        }
        
        // Create new tensor on target device
        Tensor result(impl_->sizes(), impl_->dtype(), device, requires_grad());
        
        // Copy data between devices
        device::copy_memory(
            result.data(), device,
            data(), impl_->device(),
            nbytes()
        );
        
        return result;
    }

    Tensor Tensor::to_cpu() const {
        return to(DeviceIndex(Device::CPU));
    }

    void Tensor::to_cpu_() {
        if (!impl_) {
            throw std::runtime_error("inplace to_cpu_: tensor is not initialized");
        }

        if (this->device().is_cpu()) {
            return;
        }

        Allocator* cpu_alloc = AllocatorRegistry::get_allocator(Device::CPU);
        if (!cpu_alloc) {
             throw std::runtime_error("to_cpu_: Failed to get CPU allocator");
        }

        void* new_ptr = cpu_alloc->allocate(this->nbytes());
        if (!new_ptr) {
             throw std::runtime_error("to_cpu_: Failed to allocate memory on CPU");
        }

        try {
            device::copy_memory(new_ptr, DeviceIndex(Device::CPU), this->data(), this->device(), this->nbytes());
        } catch (...) {
            cpu_alloc->deallocate(new_ptr);
            throw;
        }

        // Create DataPtr with proper deleter that uses the allocator
        DataPtr data_(static_cast<uint8_t*>(new_ptr), DataPtrDeleter(cpu_alloc));

        // Update Storage: This frees old memory and sets new one
        this->impl_->mutable_storage().set_data_ptr(std::move(data_));
        
        /*
        cudaStream_t cuda_stream = OwnTensor::cuda::getCurrentStream();
        cudaStreamSynchronize(cuda_stream);
        */
        
        this->impl_->mutable_storage().set_device(DeviceIndex(Device::CPU));
        this->impl_->mutable_storage().set_allocator(cpu_alloc);
        
        // Update TensorImpl metadata
        this->impl_->set_device(DeviceIndex(Device::CPU));
    }

    Tensor Tensor::to_cuda(int device_index) const {
        return to(DeviceIndex(Device::CUDA, device_index));
    }

    void Tensor::to_cuda_(int device_index) {
        if (!impl_) {
             throw std::runtime_error("inplace to_cuda_: tensor is not initialized");
        }

        DeviceIndex target_device(Device::CUDA, device_index);

        if (this->device().is_cuda() && this->device().index == device_index) {
            return;
        }

        #ifdef WITH_CUDA
        if (!device::cuda_available()) {
            throw std::runtime_error("CUDA is not available");
        }

        // Ensure we are on the right device for allocation/copy if needed
        device::set_cuda_device(device_index);
        
        Allocator* cuda_alloc = AllocatorRegistry::get_allocator(Device::CUDA);
        if (!cuda_alloc) {
             throw std::runtime_error("to_cuda_: Failed to get CUDA allocator");
        }
        
        void* new_ptr = cuda_alloc->allocate(this->nbytes());
        if (!new_ptr) {
             throw std::runtime_error("to_cuda_: Failed to allocate memory on CUDA");
        }

        try {
            device::copy_memory(new_ptr, DeviceIndex(Device::CUDA, device_index), this->data(), this->device(), this->nbytes());
        } catch (...) {
            cuda_alloc->deallocate(new_ptr);
            throw;
        }

        DataPtr data_(static_cast<uint8_t*>(new_ptr), DataPtrDeleter(cuda_alloc));

        this->impl_->mutable_storage().set_data_ptr(std::move(data_));
        
        /*
        cudaStream_t cuda_stream = OwnTensor::cuda::getCurrentStream();
        cudaStreamSynchronize(cuda_stream);
        */
        
        this->impl_->mutable_storage().set_device(target_device);
        this->impl_->mutable_storage().set_allocator(cuda_alloc);

        this->impl_->set_device(target_device);
        #else
        throw std::runtime_error("CUDA support not compiled");
        #endif
    }

    Tensor Tensor::pin_memory() const {
        if (!impl_) {
            throw std::runtime_error("pin_memory: tensor is not initialized");
        }

        // If active device is CUDA or already pinned, return self
        if (is_cuda() || is_pinned()) {
            return *this;
        }

        // Naive Implementation: In-Place Pinned Memory using cudaHostRegister
        // This registers the existing pageable memory as pinned, avoiding a copy.
        // It allows method chaining by returning the tensor itself.
        
        #ifdef WITH_CUDA
        void* ptr = impl_->mutable_data();
        size_t size = nbytes();
        
        // Use default flags (Portable is safer usually but user asked for naive default)
        cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault); // OR cudaHostRegisterPortable
        if (err != cudaSuccess) {
             throw std::runtime_error("cudaHostRegister failed: " + std::string(cudaGetErrorString(err)));
        }
        #else
        throw std::runtime_error("pin_memory requires CUDA support compiled.");
        #endif

        return *this;

        // TO-DO(Optimization): PyTorch Implementation
        // PyTorch creates a copy in new pinned host memory (via cudaHostAlloc) instead of registering.
        // This is often an optimization because allocating pinned memory upfront avoids 
        // fragmentation and page-locking overhead on arbitrary pointers.
    }

    bool Tensor::is_cpu() const {
        if (!impl_) return true;  // Default to CPU
        return impl_->device().is_cpu();
    }

    bool Tensor::is_cuda() const {
        if (!impl_) return false;
        return impl_->device().is_cuda();
    }

    bool Tensor::is_pinned() const {
        if (!impl_ || is_cuda()) return false;
        
        // Check pointer attributes
        #ifdef WITH_CUDA
        const void* ptr = data();
        if (!ptr) return false;
        
        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        if (err != cudaSuccess) {
            // Reset error if it was just "invalid value" (not a CUDA pointer)
            cudaGetLastError(); 
            return false; 
        }
        
        return (attr.type == cudaMemoryTypeHost);
        #else
        return false;
        #endif
    }

Tensor Tensor::to_bool() const {
    Tensor result({this->shape()}, TensorOptions()
        .with_dtype(Dtype::Bool)
        .with_device(this->device()));  // Preserve device
    
    if (this->is_cpu()) {
        // CPU path - use existing OpenMP code
        dispatch_by_dtype(this->dtype(), [&](auto T_val) {
            using T = decltype(T_val);
            const T* src = this->data<T>();
            bool* dst = result.data<bool>();
            
            #pragma omp parallel for
            for (size_t i = 0; i < this->numel(); ++i) {
                dst[i] = (src[i] != T(0.0f));
            }
        });
    }
#ifdef WITH_CUDA
    else if (this->is_cuda()) {
        dispatch_by_dtype(this->dtype(), [&](auto T_val) {
            using T = decltype(T_val);
            const T* src = this->data<T>();
            bool* dst = result.data<bool>();
            
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            
            // Launch conversion kernel
            convert_to_bool_cuda<T>(src, dst, this->numel(), stream);
            
            //  Synchronization is ALREADY in convert_to_bool_cuda
            // No need to sync again here (but it doesn't hurt)
        });
    }  //  ADD THIS CLOSING BRACE
#endif
    else {
        throw std::runtime_error("to_bool: Unknown device type");
    }
    
    return result;
}

void Tensor::set_requires_grad(bool req) {
    if (!impl_) {
        throw std::runtime_error("set_requires_grad: tensor is not initialized");
    }
    impl_->set_requires_grad(req);
}

Tensor Tensor::grad_view() const {
    if (!impl_ || !impl_->has_autograd_meta()) {
        throw std::runtime_error("grad_view(): Tensor has no gradient allocated.");
    }
    
    // Return the gradient tensor directly
    return impl_->grad();
}

void Tensor::zero_grad() {
    if (impl_) {
        impl_->zero_grad();
    }
}

void Tensor::set_grad(const Tensor& grad) {
    if (!impl_) {
        throw std::runtime_error("set_grad: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->set_grad(grad);
}

// ========================================================================
// Autograd Methods (PyTorch-style)
// ========================================================================

std::shared_ptr<Node> Tensor::grad_fn() const {
    if (!impl_ || !impl_->has_autograd_meta()) {
        return nullptr;
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    return meta->grad_fn();
}

void Tensor::set_grad_fn(std::shared_ptr<Node> fn) {
    if (!impl_) {
        throw std::runtime_error("set_grad_fn: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->set_grad_fn(std::move(fn));
}

uint32_t Tensor::output_nr() const {
    if (!impl_ || !impl_->has_autograd_meta()) {
        return 0;
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    return meta->output_nr();
}

void Tensor::set_output_nr(uint32_t nr) {
    if (!impl_) {
        throw std::runtime_error("set_output_nr: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->set_output_nr(nr);
}

bool Tensor::is_view() const {
    if (!impl_ || !impl_->has_autograd_meta()) {
        return false;
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    return meta->is_view();
}

void Tensor::set_is_view(bool is_view) {
    if (!impl_) {
        throw std::runtime_error("set_is_view: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->set_is_view(is_view);
}

bool Tensor::retains_grad() const {
    if (!impl_ || !impl_->has_autograd_meta()) {
        return false;
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    return meta->retains_grad();
}

void Tensor::set_retains_grad(bool retains) {
    if (!impl_) {
        throw std::runtime_error("set_retains_grad: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->set_retains_grad(retains);
}

bool Tensor::is_leaf() const {
    if (!impl_ || !impl_->has_autograd_meta()) {
        return true;  // Tensors without autograd meta are leaves
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    return meta->is_leaf();
}

void Tensor::register_hook(std::unique_ptr<FunctionPreHook> hook) {
    if (!impl_) {
        throw std::runtime_error("register_hook: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->add_hook(std::move(hook));
}

void Tensor::register_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook) {
    if (!impl_) {
        throw std::runtime_error("register_post_acc_hook: tensor is not initialized");
    }
    if (!impl_->has_autograd_meta()) {
        impl_->set_autograd_meta(std::make_unique<AutogradMeta>());
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->add_post_acc_hook(std::move(hook));
}

void Tensor::clear_hooks() {
    if (!impl_ || !impl_->has_autograd_meta()) {
        return;  // Nothing to clear
    }
    auto* meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->clear_hooks();
}

void Tensor::backward(const Tensor* grad_output) {
    autograd::backward(*this, grad_output);
}

// ========================================================================
// Release Method
// ========================================================================

void Tensor::release() {
    impl_.reset();
}

//  ========================================================================
// Explicit Template Instantiations
// ========================================================================

// Explicit instantiations for grad() templates
template bool* Tensor::grad<bool>();
template const bool* Tensor::grad<bool>() const;

template int8_t* Tensor::grad<int8_t>();
template const int8_t* Tensor::grad<int8_t>() const;

template int16_t* Tensor::grad<int16_t>();
template const int16_t* Tensor::grad<int16_t>() const;

template int32_t* Tensor::grad<int32_t>();
template const int32_t* Tensor::grad<int32_t>() const;

template int64_t* Tensor::grad<int64_t>();
template const int64_t* Tensor::grad<int64_t>() const;

template float* Tensor::grad<float>();
template const float* Tensor::grad<float>() const;

template double* Tensor::grad<double>();
template const double* Tensor::grad<double>() const;

template float16_t* Tensor::grad<float16_t>();
template const float16_t* Tensor::grad<float16_t>() const;

template bfloat16_t* Tensor::grad<bfloat16_t>();
template const bfloat16_t* Tensor::grad<bfloat16_t>() const;

template complex32_t* Tensor::grad<complex32_t>();
template const complex32_t* Tensor::grad<complex32_t>() const;

template complex64_t* Tensor::grad<complex64_t>();
template const complex64_t* Tensor::grad<complex64_t>() const;

template complex128_t* Tensor::grad<complex128_t>();
template const complex128_t* Tensor::grad<complex128_t>() const;
 
    template const bool* Tensor::data<bool>() const;
    template bool* Tensor::data<bool>();
// int8_t (short)
    template const int8_t* Tensor::data<int8_t>() const;
    template int8_t* Tensor::data<int8_t>();

    // int16_t (short)
    template const short* Tensor::data<short>() const;
    template short* Tensor::data<short>();

    // int32_t (int)
    template const int* Tensor::data<int>() const;
    template int* Tensor::data<int>();

    // int64_t (long/index type used for reduction output)
    template const int64_t* Tensor::data<int64_t>() const;
    template int64_t* Tensor::data<int64_t>(); 

    // float (float)
    template const float* Tensor::data<float>() const;
    template float* Tensor::data<float>();

    // double (double)
    template const double* Tensor::data<double>() const;
    template double* Tensor::data<double>();

    // Custom types (float16_t and bfloat16_t)
    // Assuming these types are correctly defined in dtype/Types.h
    template const float16_t* Tensor::data<float16_t>() const;
    template float16_t* Tensor::data<float16_t>();

    template const bfloat16_t* Tensor::data<bfloat16_t>() const;
    template bfloat16_t* Tensor::data<bfloat16_t>();

    template const float4_e2m1_t* Tensor::data<float4_e2m1_t>() const;
    template float4_e2m1_t* Tensor::data<float4_e2m1_t>();

    template const float4_e2m1_2x_t* Tensor::data<float4_e2m1_2x_t>() const;
    template float4_e2m1_2x_t* Tensor::data<float4_e2m1_2x_t>();
 
    // unsigned types 
    template const uint8_t* Tensor::data<uint8_t>() const;
    template uint8_t* Tensor::data<uint8_t>();

    template const uint16_t* Tensor::data<uint16_t>() const;
    template uint16_t* Tensor::data<uint16_t>();

    template const uint32_t* Tensor::data<uint32_t>() const;
    template uint32_t* Tensor::data<uint32_t>();

    template const uint64_t* Tensor::data<uint64_t>() const;
    template uint64_t* Tensor::data<uint64_t>();
    // Complex types
    template const complex32_t* Tensor::data<complex32_t>() const;
    template complex32_t* Tensor::data<complex32_t>();
    
    template const complex64_t* Tensor::data<complex64_t>() const;
    template complex64_t* Tensor::data<complex64_t>();
    
    template const complex128_t* Tensor::data<complex128_t>() const;
    template complex128_t* Tensor::data<complex128_t>();

    // Explicit instantiations for set_data
    template void Tensor::set_data<bool>(const std::vector<bool>&);
    template void Tensor::set_data<int8_t>(const std::vector<int8_t>&);
    template void Tensor::set_data<int16_t>(const std::vector<int16_t>&);
    template void Tensor::set_data<int32_t>(const std::vector<int32_t>&);
    template void Tensor::set_data<int64_t>(const std::vector<int64_t>&);
    template void Tensor::set_data<float>(const std::vector<float>&);
    template void Tensor::set_data<double>(const std::vector<double>&);
    template void Tensor::set_data<uint8_t>(const std::vector<uint8_t>&);
    template void Tensor::set_data<uint16_t>(const std::vector<uint16_t>&);
    template void Tensor::set_data<uint32_t>(const std::vector<uint32_t>&);
    template void Tensor::set_data<uint64_t>(const std::vector<uint64_t>&);
    template void Tensor::set_data<float16_t>(const std::vector<float16_t>&);
    template void Tensor::set_data<bfloat16_t>(const std::vector<bfloat16_t>&);
    template void Tensor::set_data<complex32_t>(const std::vector<complex32_t>&);
    template void Tensor::set_data<complex64_t>(const std::vector<complex64_t>&);
    template void Tensor::set_data<complex128_t>(const std::vector<complex128_t>&);
    template void Tensor::set_data<float>(const float*, size_t);
    template void Tensor::set_data<double>(const double*, size_t);
    template void Tensor::set_data<int32_t>(const int32_t*, size_t);
    template void Tensor::set_data<int64_t>(const int64_t*, size_t);
    template void Tensor::set_data<bool>(const bool*, size_t);

    template void Tensor::set_grad<float>(const float*, size_t);
    template void Tensor::set_grad<double>(const double*, size_t);
    template void Tensor::set_grad<int32_t>(const int32_t*, size_t);
    template void Tensor::set_grad<int64_t>(const int64_t*, size_t);
    template void Tensor::set_grad<bool>(const bool*, size_t);


    template void Tensor::set_data<bool>(std::initializer_list<bool>);
    template void Tensor::set_data<int8_t>(std::initializer_list<int8_t>);
    template void Tensor::set_data<int16_t>(std::initializer_list<int16_t>);
    template void Tensor::set_data<int32_t>(std::initializer_list<int32_t>);
    template void Tensor::set_data<int64_t>(std::initializer_list<int64_t>);
    template void Tensor::set_data<float>(std::initializer_list<float>);
    template void Tensor::set_data<double>(std::initializer_list<double>);
    template void Tensor::set_data<uint8_t>(std::initializer_list<uint8_t>);
    template void Tensor::set_data<uint16_t>(std::initializer_list<uint16_t>);
    template void Tensor::set_data<uint32_t>(std::initializer_list<uint32_t>);
    template void Tensor::set_data<uint64_t>(std::initializer_list<uint64_t>);
    template void Tensor::set_data<float16_t>(std::initializer_list<float16_t>);
    template void Tensor::set_data<bfloat16_t>(std::initializer_list<bfloat16_t>);
    template void Tensor::set_data<complex32_t>(std::initializer_list<complex32_t>);
    template void Tensor::set_data<complex64_t>(std::initializer_list<complex64_t>);
    template void Tensor::set_data<complex128_t>(std::initializer_list<complex128_t>);
    

    // Explicit instantiations for set_grad
    template void Tensor::set_grad<bool>(const std::vector<bool>&);
    template void Tensor::set_grad<int8_t>(const std::vector<int8_t>&);
    template void Tensor::set_grad<int16_t>(const std::vector<int16_t>&);
    template void Tensor::set_grad<int32_t>(const std::vector<int32_t>&);
    template void Tensor::set_grad<int64_t>(const std::vector<int64_t>&);
    template void Tensor::set_grad<float>(const std::vector<float>&);
    template void Tensor::set_grad<double>(const std::vector<double>&);
    template void Tensor::set_grad<uint8_t>(const std::vector<uint8_t>&);
    template void Tensor::set_grad<uint16_t>(const std::vector<uint16_t>&);
    template void Tensor::set_grad<uint32_t>(const std::vector<uint32_t>&);
    template void Tensor::set_grad<uint64_t>(const std::vector<uint64_t>&);
    template void Tensor::set_grad<float16_t>(const std::vector<float16_t>&);
    template void Tensor::set_grad<bfloat16_t>(const std::vector<bfloat16_t>&);
    template void Tensor::set_grad<complex32_t>(const std::vector<complex32_t>&);
    template void Tensor::set_grad<complex64_t>(const std::vector<complex64_t>&);
    template void Tensor::set_grad<complex128_t>(const std::vector<complex128_t>&);

    // Explicit instantiations for fill_grad
    template void Tensor::fill_grad<bool>(bool);
    template void Tensor::fill_grad<int16_t>(int16_t);
    template void Tensor::fill_grad<int32_t>(int32_t);
    template void Tensor::fill_grad<int64_t>(int64_t);
    template void Tensor::fill_grad<uint8_t>(uint8_t);
    template void Tensor::fill_grad<uint16_t>(uint16_t);
    template void Tensor::fill_grad<uint32_t>(uint32_t);
    template void Tensor::fill_grad<uint64_t>(uint64_t);
    template void Tensor::fill_grad<float>(float);
    template void Tensor::fill_grad<double>(double);
    template void Tensor::fill_grad<float16_t>(float16_t);
    template void Tensor::fill_grad<bfloat16_t>(bfloat16_t);
    template void Tensor::fill_grad<complex32_t>(complex32_t);
    template void Tensor::fill_grad<complex64_t>(complex64_t);
    template void Tensor::fill_grad<complex128_t>(complex128_t);

    // Explicit instantiations for fill
    template void Tensor::fill<bool>(bool);
    template void Tensor::fill<int8_t>(int8_t);
    template void Tensor::fill<int16_t>(int16_t);
    template void Tensor::fill<int32_t>(int32_t);
    template void Tensor::fill<int64_t>(int64_t);
    template void Tensor::fill<uint8_t>(uint8_t);
    template void Tensor::fill<uint16_t>(uint16_t);
    template void Tensor::fill<uint32_t>(uint32_t);
    template void Tensor::fill<uint64_t>(uint64_t);
    template void Tensor::fill<float>(float);
    template void Tensor::fill<double>(double);
    template void Tensor::fill<float16_t>(float16_t);
    template void Tensor::fill<bfloat16_t>(bfloat16_t);
    template void Tensor::fill<complex32_t>(complex32_t);
    template void Tensor::fill<complex64_t>(complex64_t);
    template void Tensor::fill<complex128_t>(complex128_t);
    template void Tensor::fill<float4_e2m1_t>(float4_e2m1_t);
    template void Tensor::fill<float4_e2m1_2x_t>(float4_e2m1_2x_t);

    // Specialization for fill with bool (Moved from header)
    template<>
    void Tensor::fill<bool>(bool value) {
        if (dtype() != Dtype::Bool) {
            throw std::runtime_error("Fill bool: dtype must be Bool");
        }

        uint8_t fill_value = value ? 1 : 0;

        if (device().is_cpu()) {
            uint8_t* data = reinterpret_cast<uint8_t*>(this->data());
            std::memset(data, fill_value, numel());
        } else {
            // For GPU
            #ifdef WITH_CUDA
            std::vector<uint8_t> temp_data(numel(), fill_value);
            device::copy_memory(data(), device(),
                            temp_data.data(), DeviceIndex(Device::CPU),
                            numel() * sizeof(uint8_t));
             #else
             throw std::runtime_error("CUDA not enabled");
             #endif
        }
    }


    // ========================================================================
    // TopK Implementation
    // ========================================================================
    std::pair<Tensor, Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
        if (k < 0) throw std::runtime_error("topk: k must be non-negative");
        
        int64_t ndim_ = ndim();
        if (dim < 0) dim += ndim_;
        if (dim < 0 || dim >= ndim_) throw std::runtime_error("topk: index out of range");
        
        int64_t n = shape().dims[dim];
        if (k > n) throw std::runtime_error("topk: k=" + std::to_string(k) + " is larger than dimension size=" + std::to_string(n));

        // If on GPU, CPU roundtrip
        if (is_cuda()) {
             Tensor cpu_in = this->to_cpu();
             auto result = cpu_in.topk(k, dim, largest, sorted);
             int dev_idx = device().index;
             return {result.first.to_cuda(dev_idx), result.second.to_cuda(dev_idx)};
        }

        // CPU implementation
        // 1. Permute to bring dim to last
        Tensor input = *this;
        bool permuted = false;
        if (dim != ndim_ - 1) {
            // Need transpose/permute. 
            // For now, simple transpose: swap(dim, last)
            input = input.transpose(dim, ndim_ - 1);
            permuted = true;
        }
        
        // Ensure contiguous
        input = input.contiguous();
        
        const auto& in_dims = input.shape().dims;
        int64_t last_dim = in_dims.back();
        int64_t outer_size = input.numel() / last_dim;
        
        // Output shapes
        Shape out_shape = input.shape();
        out_shape.dims.back() = k;
        
        Tensor values(out_shape, input.dtype(), device(), false);
        Tensor indices(out_shape, Dtype::Int64, device(), false);
        
        // Process rows
        dispatch_by_dtype(input.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            const T* in_ptr = input.data<T>();
            T* val_ptr = values.data<T>();
            int64_t* idx_ptr = indices.data<int64_t>();
            
            // Per row vector for sorting
            // Note: Parallelize this loop with OpenMP if available
            #pragma omp parallel for
            for (int64_t i = 0; i < outer_size; ++i) {
                // Must allocate vector inside loop for thread safety with OpenMP
                std::vector<std::pair<T, int64_t>> row(last_dim);
                const T* row_in = in_ptr + i * last_dim;
                
                for (int64_t j = 0; j < last_dim; ++j) {
                    row[j] = {row_in[j], j};
                }
                
                
                // Partial sort or full sort?
                if (sorted) {
                     // sort gives strict ordering
                    if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
                        throw std::runtime_error("topk not supported for complex types");
                    } else {
                        auto cmp = [largest](const std::pair<T, int64_t>& a, const std::pair<T, int64_t>& b) {
                            if (largest) return a.first > b.first; 
                            else return a.first < b.first;
                        };
                        std::partial_sort(row.begin(), row.begin() + k, row.end(), cmp);
                    }
                } else {
                    if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
                         throw std::runtime_error("topk not supported for complex types");
                    } else {
                        auto cmp = [largest](const std::pair<T, int64_t>& a, const std::pair<T, int64_t>& b) {
                            if (largest) return a.first > b.first; 
                            else return a.first < b.first;
                        };
                        std::nth_element(row.begin(), row.begin() + k, row.end(), cmp);
                    }
                }
                
                T* row_val = val_ptr + i * k;
                int64_t* row_idx = idx_ptr + i * k;
                
                for (int64_t j = 0; j < k; ++j) {
                    row_val[j] = row[j].first;
                    row_idx[j] = row[j].second;
                }
            }
        });
        
        // Un-permute results if needed
        if (permuted) {
            values = values.transpose(dim, ndim_ - 1).contiguous();
            indices = indices.transpose(dim, ndim_ - 1).contiguous();
        }
        
        return {values, indices};
    }

} // namespace OwnTensor