#pragma once

#include <cstddef>
#include <memory>
#include "device/Device.h"
#include "device/Allocator.h"
#include "dtype/Dtype.h"
#include "core/IntrusivePtr.h"

namespace OwnTensor {

// Custom deleter wrapper for allocator-based cleanup
struct DataPtrDeleter {
    Allocator* allocator;
    
    DataPtrDeleter() : allocator(nullptr) {}
    explicit DataPtrDeleter(Allocator* alloc) : allocator(alloc) {}
    
    void operator()(uint8_t* ptr) const {
        if (allocator && ptr) {
            allocator->deallocate(ptr);
        }
    }
};

// Smart pointer for raw data with custom allocator-based deletion
using DataPtr = std::unique_ptr<uint8_t[], DataPtrDeleter>;

/**
 * Storage class manages raw memory allocation and lifetime.
 * This is the lowest layer in the three-layer tensor architecture:
 * Storage -> TensorImpl -> Tensor
 * 
 * Storage is responsible for:
 * - Allocating memory via Allocator
 * - Tracking total bytes allocated
 * - Managing data lifetime via DataPtr
 * - Storing dtype and device information
 */
class Storage : public intrusive_ptr_target {
private:
    DataPtr data_ptr_;           // Smart pointer with custom deleter
    Dtype dtype_;                // Data type of elements
    size_t nbytes_;              // Total bytes allocated
    Allocator* allocator_;       // Allocator used for this storage
    DeviceIndex device_;         // Device where storage resides

public:
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /**
     * Default constructor - creates uninitialized storage
     */
    Storage();
    
    /**
     * Constructor: allocate memory using allocator
     * @param nbytes Number of bytes to allocate
     * @param dtype Data type for elements
     * @param device Device where storage should reside
     * @param allocator Allocator to use (if nullptr, uses AllocatorRegistry)
     */
    Storage(size_t nbytes, Dtype dtype, DeviceIndex device, Allocator* allocator = nullptr);
    
    /**
     * Constructor: from existing data pointer
     * Takes ownership of the data pointer
     * @param data_ptr Existing data pointer with deleter
     * @param nbytes Number of bytes in the data
     * @param dtype Data type for elements
     * @param device Device where storage resides
     * @param allocator Allocator used for this storage
     */
    Storage(DataPtr data_ptr, size_t nbytes, Dtype dtype, DeviceIndex device, Allocator* allocator);
    
    // Move semantics
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept = default;
    
    // No copy (storage should not be copied, only moved)
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /**
     * Get raw pointer to data (const version)
     */
    const void* data() const { return data_ptr_.get(); }
    
    /**
     * Get raw pointer to data (mutable version)
     */
    void* mutable_data() { return data_ptr_.get(); }
    
    /**
     * Get underlying uint8_t pointer (mutable)
     */
    uint8_t* data_ptr() { return data_ptr_.get(); }     //Why????????
    
    /**
     * Get underlying uint8_t pointer (const)
     */
    const uint8_t* data_ptr() const { return data_ptr_.get(); }   //Why????????
    
    /**
     * Get total number of bytes allocated
     */
    size_t nbytes() const { return nbytes_; }
    
    /**
     * Get data type
     */
    Dtype dtype() const { return dtype_; }
    
    /**
     * Get device index
     */
    DeviceIndex device() const { return device_; }
    
    /**
     * Get allocator used for this storage
     */
    Allocator* allocator() const { return allocator_; }
    
    /**
     * Check if storage is initialized (has data)
     */
    bool is_valid() const { return data_ptr_.get() != nullptr; }
    
    // ========================================================================
    // Mutators
    // ========================================================================
    
    /**
     * Set new data pointer (takes ownership)
     * Warning: existing data will be deallocated
     */
    void set_data_ptr(DataPtr new_ptr);

    
    void set_device(DeviceIndex device);

    void set_allocator(Allocator* alloc);


    /**
     * Reset storage to uninitialized state
     */
    void reset();
};

} // namespace OwnTensor
