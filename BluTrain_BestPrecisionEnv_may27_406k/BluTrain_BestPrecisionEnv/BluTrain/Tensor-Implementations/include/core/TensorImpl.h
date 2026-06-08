#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include "dtype/Dtype.h"
#include "device/Device.h"
#include "core/Storage.h"
#include "core/AutogradMeta.h"
#include "core/Shape.h"
#include "core/Stride.h"
#include "core/IntrusivePtr.h"

namespace OwnTensor {

// Forward declarations
class Tensor;
class Allocator;

// ============================================================================
// Version Counter for In-Place Operations
// ============================================================================

/**
 * Version counter to track in-place modifications
 * Incremented whenever tensor is modified in-place
 */
struct VariableVersion {
private:
    std::atomic<uint32_t> version_{0};
    
public:
    VariableVersion() = default;
    VariableVersion(uint32_t initial) : version_(initial) {}
    
    void bump() {
        ++version_;
    }
    
    uint32_t current_version() const {
        return version_.load();
    }
    
    void set_version(uint32_t version) {
        version_.store(version);
    }
};

// ============================================================================
// TensorImpl: Core Tensor Implementation
// ============================================================================

/**
 * TensorImpl is the core tensor data structure in the three-layer architecture:
 * Storage -> TensorImpl -> Tensor
 * 
 * TensorImpl is responsible for:
 * - Owning the Storage (actual data)
 * - Managing tensor metadata (shape, strides, offset, dtype, device)
 * - Managing AutogradMeta (gradient tracking)
 * - Reference counting for efficient memory management
 * - Version tracking for in-place operations
 */
class TensorImpl : public intrusive_ptr_target {
private:
    intrusive_ptr<Storage> storage_;                       // Shared storage
    std::unique_ptr<AutogradMetaInterface> autograd_meta_; // Autograd metadata (lazy)
    std::shared_ptr<VariableVersion> version_counter_;      // Version for in-place ops                // Original tensor if this is a view
    
    // Tensor metadata
    Shape shape_;                                // Tensor dimensions
    Stride stride_;                              // Strides for each dimension
    int64_t storage_offset_;                     // Offset into storage
    Dtype dtype_;                                // Data type
    DeviceIndex device_;                         // Device location

public:
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /**
     * Constructor: Create TensorImpl with shared Storage
     * Used for creating views that share storage
     * 
     * @param storage Storage to share
     * @param shape Tensor shape
     * @param stride Tensor strides
     * @param offset Offset into storage
     * @param dtype Data type
     * @param device Device location
     * @param base_impl Original TensorImpl to keep alive
     */
    TensorImpl(intrusive_ptr<Storage> storage,
               const Shape& shape,
               const Stride& stride,
               int64_t offset,
               Dtype dtype,
               DeviceIndex device);
    
    /**
     * Constructor: Create TensorImpl and allocate new Storage
     * Used for creating new tensors
     * 
     * @param shape Tensor shape
     * @param dtype Data type
     * @param device Device location
     * @param requires_grad Whether to track gradients
     */
    TensorImpl(const Shape& shape,
               Dtype dtype,
               DeviceIndex device,
               bool requires_grad,
               Allocator* allocator = nullptr);
    
    // Destructor
    ~TensorImpl() override;
    
    // No copy (use intrusive_ptr for sharing)
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;
    
    // ========================================================================
    // Storage Access
    // ========================================================================
    
    const Storage& storage() const { return *storage_; }
    Storage& mutable_storage() { return *storage_; }
    
    // Direct access to storage pointer if needed
    intrusive_ptr<Storage> storage_ptr() const { return storage_; }
    
    // ========================================================================
    // Metadata Accessors
    // ========================================================================
    
    const Shape& sizes() const { return shape_; }
    const Stride& strides() const { return stride_; }
    int64_t storage_offset() const { return storage_offset_; }
    Dtype dtype() const { return dtype_; }
    DeviceIndex device() const { return device_; }
    void set_device(DeviceIndex device);
    
    /**
     * Get total number of elements in tensor
     */
    size_t numel() const;
    
    /**
     * Get number of bytes used by tensor elements
     */
    size_t nbytes() const;
    
    /**
     * Get number of dimensions
     */
    int64_t ndim() const;
    
    // ========================================================================
    // Data Access
    // ========================================================================
    
    /**
     * Get pointer to tensor data (mutable)
     * Automatically accounts for storage_offset
     */
    void* mutable_data();
    
    /**
     * Get pointer to tensor data (const)
     * Automatically accounts for storage_offset
     */
    const void* data() const;
    
    /**
     * Get typed pointer to data (mutable)
     */
    template<typename T>
    T* data() {
        return reinterpret_cast<T*>(mutable_data());
    }
    
    /**
     * Get typed pointer to data (const)
     */
    template<typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(data());
    }
    
    // ========================================================================
    // Autograd Methods
    // ========================================================================
    
    /**
     * Set whether tensor requires gradients
     */
    void set_requires_grad(bool requires_grad);
    
    /**
     * Check if tensor requires gradients
     */
    bool requires_grad() const;
    
    /**
     * Get mutable reference to gradient tensor
     */
    Tensor& mutable_grad();
    
    /**
     * Get const reference to gradient tensor
     */
    const Tensor& grad() const;
    
    /**
     * Set autograd metadata
     */
    void set_autograd_meta(std::unique_ptr<AutogradMetaInterface> autograd_meta);
    
    /**
     * Get autograd metadata (may be nullptr)
     */
    AutogradMetaInterface* autograd_meta() const {
        return autograd_meta_.get();
    }
    
    /**
     * Check if autograd metadata exists
     */
    bool has_autograd_meta() const {
        return autograd_meta_ != nullptr;
    }
        /**
     * Check if gradient data exists
     */
    bool has_grad() const {
        if (!has_autograd_meta()) return false;
        return autograd_meta_->has_grad();
    }
    
    /**
     * Zero out the gradient
     */
    void zero_grad();
    
    // ========================================================================
    // Version Control
    // ========================================================================
    
    void bump_version() {
        if (version_counter_) version_counter_->bump();
    }
    
    uint32_t version() const {
        return version_counter_ ? version_counter_->current_version() : 0;
    }

    std::shared_ptr<VariableVersion> version_counter_ptr() const {
        return version_counter_;
    }

    void set_version_counter(std::shared_ptr<VariableVersion> counter) {
        version_counter_ = std::move(counter);
    }
    
    // ========================================================================
    // Metadata Mutation
    // ========================================================================
    
    /**
     * Set new shape and strides
     * Used for view operations
     */
    void set_sizes_and_strides(const Shape& new_shape, const Stride& new_stride);
    
    /**
     * Set storage offset
     */
    void set_storage_offset(int64_t offset) {
        storage_offset_ = offset;
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    /**
     * Release resources (called when refcount reaches 0)
     */
    void release_resources() override;

    // ========================================================================
    // Debugging / Memory Tracking
    // ========================================================================
    static std::atomic<int64_t> active_tensor_count_;
    static int64_t get_active_count() { return active_tensor_count_.load(); }
};

} // namespace OwnTensor
