#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

namespace OwnTensor {

// ============================================================================
// Intrusive Reference Counting Infrastructure
// ============================================================================

/**
 * Base class for intrusive reference counting
 * Objects inheriting from this can be used with intrusive_ptr
 */
class intrusive_ptr_target {
protected:
    mutable std::atomic<size_t> refcount_{0};
    
public:
    virtual ~intrusive_ptr_target() = default;
    
    void add_ref() const {
        refcount_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void release() const {
        if (refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }
    
    size_t use_count() const {
        return refcount_.load();
    }
    
    // Override to perform custom cleanup
    virtual void release_resources() {}
};

/**
 * Intrusive smart pointer implementation
 * More efficient than shared_ptr for frequently copied objects
 */
template<typename T>
class intrusive_ptr {
private:
    T* ptr_;
    
public:
    intrusive_ptr() : ptr_(nullptr) {}
    
    explicit intrusive_ptr(T* p) : ptr_(p) {
        if (ptr_) ptr_->add_ref();
    }
    
    intrusive_ptr(const intrusive_ptr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->add_ref();
    }
    
    T* release_pointer() noexcept {
      T* p = ptr_;
      ptr_ = nullptr;
      return p;
    }

    intrusive_ptr(intrusive_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    // template<typename U>
    // friend class intrusive_ptr;
    
    template<typename U>
    intrusive_ptr(const intrusive_ptr<U>& other) : ptr_(other.get()) {
        if (ptr_) ptr_->add_ref();
    }

    template<typename U>
    intrusive_ptr(intrusive_ptr<U>&& other) noexcept : ptr_(other.get()) {
        if (ptr_ != nullptr) {
           other.release_pointer(); // Custom helper or manually set null
           // However standard intrusive_ptr move ctor usually clears source
        }
        // Wait, the template move implementation above is safer if we just copy pointer and null out source manually.
        // But we can't clear private member of other type easily without friend.
        // For now sticking to homogenous move or rely on copy+reset.
        // Actually, let's keep it simple and safe:
        // Copy constructor handles add_ref, so we can just use that if we don't want to mess with friends.
        // Or for efficiency:
    }
    
    ~intrusive_ptr() {
        if (ptr_) ptr_->release();
    }
    
    intrusive_ptr& operator=(const intrusive_ptr& other) {
        if (ptr_ != other.ptr_) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            if (ptr_) ptr_->add_ref();
        }
        return *this;
    }
    
    intrusive_ptr& operator=(intrusive_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    void reset() {
        if (ptr_) {
            ptr_->release();
            ptr_ = nullptr;
        }
    }
};

/**
 * Factory function to create intrusive_ptr
 */
template<typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
    return intrusive_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace OwnTensor
