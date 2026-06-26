#include "device/PinnedCPUAllocator.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <mutex>

namespace OwnTensor {
namespace device {

    namespace {
        struct GlobalPinnedStats {
            std::unordered_map<void*, size_t> allocs;
            std::mutex mtx;
            size_t current = 0;
            size_t peak = 0;
        };
        GlobalPinnedStats& get_pinned_stats() {
            static GlobalPinnedStats stats;
            return stats;
        }
    }

    PinnedCPUAllocator::PinnedCPUAllocator(unsigned int flags) : flags_(flags) {}

    PinnedCPUAllocator::MemoryStats PinnedCPUAllocator::get_stats() {
        auto& stats = get_pinned_stats();
        std::lock_guard<std::mutex> lock(stats.mtx);
        return {stats.current, stats.peak};
    }

void* PinnedCPUAllocator::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    
    // TO-DO(Optimization): High-Throughput Scenarios
    // Current Implementation: cudaHostAlloc()
    // Limitation: The CUDA driver takes a global lock on the device context during allocation.
    // Proposed Optimization: Use standard malloc() + cudaHostRegister() to avoid driver lock.

#ifdef WITH_CUDA
    // cudaHostAlloc with flags (Default, Portable, Mapped, WriteCombined)
    cudaError_t err = cudaHostAlloc(&ptr, bytes, flags_);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Pinned Memory allocation failed.");
    }
#else
    
    throw std::runtime_error("Pinned Memory allocation failed: CUDA support not compiled.");
#endif
    {
        auto& stats = get_pinned_stats();
        std::lock_guard<std::mutex> lock(stats.mtx);
        stats.allocs[ptr] = bytes;
        stats.current += bytes;
        if (stats.current > stats.peak) stats.peak = stats.current;
    }
    return ptr;
}

void PinnedCPUAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    {
        auto& stats = get_pinned_stats();
        std::lock_guard<std::mutex> lock(stats.mtx);
        auto it = stats.allocs.find(ptr);
        if (it != stats.allocs.end()) {
            size_t bytes = it->second;
            if (stats.current >= bytes) stats.current -= bytes;
            stats.allocs.erase(it);
        }
    }

#ifdef WITH_CUDA
    cudaFreeHost(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace device
} // namespace OwnTensor

