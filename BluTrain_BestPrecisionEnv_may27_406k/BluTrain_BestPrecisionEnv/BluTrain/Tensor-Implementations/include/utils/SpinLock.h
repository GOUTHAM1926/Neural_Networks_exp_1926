#pragma once

#include <atomic>

namespace OwnTensor {
namespace utils {

/**
 * @brief Lightweight spinlock using std::atomic_flag.
 * 
 * No OS kernel calls. Ideal for nanosecond-duration critical sections
 * like gradient accumulation where std::mutex is overkill.
 */
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin with pause hint to reduce pipeline stalls
            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
            #endif
        }
    }
    
    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }
};

/**
 * @brief RAII guard for SpinLock (like std::lock_guard but for SpinLock).
 */
class SpinLockGuard {
    SpinLock& lock_;
public:
    explicit SpinLockGuard(SpinLock& lock) noexcept : lock_(lock) { lock_.lock(); }
    ~SpinLockGuard() noexcept { lock_.unlock(); }
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;
};

} // namespace utils
} // namespace OwnTensor
