#include "core/Storage.h"
#include "device/AllocatorRegistry.h"
#include "device/DeviceSet.h"
#include <stdexcept>
#include <cstring>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#endif

namespace OwnTensor {

// ============================================================================
// Constructors
// ============================================================================

Storage::Storage()
    : dtype_(Dtype::Float32),
      nbytes_(0),
      allocator_(nullptr),
      device_(DeviceIndex(Device::CPU)) {
    // Uninitialized storage - data_ptr_ is nullptr
}

Storage::Storage(Storage&& other) noexcept
    :   intrusive_ptr_target(),
        data_ptr_(std::move(other.data_ptr_)),
        dtype_(other.dtype_),
        nbytes_(other.nbytes_),
        allocator_(other.allocator_),
        device_(other.device_)
{
    other.nbytes_ = 0;
    other.allocator_ = nullptr;
}

Storage::Storage(size_t nbytes, Dtype dtype, DeviceIndex device, Allocator* allocator)
    : dtype_(dtype),
      nbytes_(nbytes),
      device_(device) {
    
    // Get allocator from registry if not provided
    if (allocator == nullptr) {
        allocator_ = AllocatorRegistry::get_allocator(device.device);
    } else {
        allocator_ = allocator;
    }
    
    if (allocator_ == nullptr) {
        throw std::runtime_error("Storage: Failed to get allocator for device");
    }
    
    // Allocate memory if nbytes > 0
    if (nbytes_ > 0) {
#ifdef WITH_CUDA
        if (device_.is_cuda()) {
            device::set_cuda_device(device_.index);
        }
#endif
        void* raw_ptr = allocator_->allocate(nbytes_);
        if (raw_ptr == nullptr) {
            throw std::runtime_error("Storage: Failed to allocate " + 
                                   std::to_string(nbytes_) + " bytes");
        }
        
        // Initialize memory to zero
        #ifdef WITH_CUDA
        if (device_.is_cuda()) {
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            // allocator_->memsetAsync(raw_ptr, 0, nbytes_, stream);
        } else
        #endif
        {
            // allocator_->memset(raw_ptr, 0, nbytes_);
        }
        
        // Create DataPtr with custom deleter
        data_ptr_ = DataPtr(static_cast<uint8_t*>(raw_ptr), DataPtrDeleter(allocator_));
    }
}

Storage::Storage(DataPtr data_ptr, size_t nbytes, Dtype dtype, 
                 DeviceIndex device, Allocator* allocator)
    : data_ptr_(std::move(data_ptr)),
      dtype_(dtype),
      nbytes_(nbytes),
      allocator_(allocator),
      device_(device) {
    
    // If allocator is nullptr, try to get it from registry
    if (allocator_ == nullptr) {
        allocator_ = AllocatorRegistry::get_allocator(device.device);
    }
}

// ============================================================================
// Mutators
// ============================================================================

void Storage::set_data_ptr(DataPtr new_ptr) {
    // This will automatically deallocate old data via DataPtrDeleter
    // allocator_->deallocate(static_cast<void*>(data_ptr_.release()));
    
    data_ptr_ = std::move(new_ptr);
}

void Storage::set_device(DeviceIndex device)
{
    device_ = device;
}

void Storage::set_allocator(Allocator* alloc)
{
    allocator_ = alloc;
}

void Storage::reset() {
    data_ptr_.reset();  // Deallocates via custom deleter
    nbytes_ = 0;
}

} // namespace OwnTensor
