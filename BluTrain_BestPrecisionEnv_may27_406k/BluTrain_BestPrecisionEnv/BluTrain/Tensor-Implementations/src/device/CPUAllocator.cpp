#include "device/CPUAllocator.h"
#include "device/AllocationTracker.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <new> // For std::bad_alloc

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{
    // Changes added by Grishma
    namespace {
        struct GlobalCPUStats {
            std::unordered_map<void*, size_t> allocs;
            std::mutex mtx;
            size_t current = 0;
            size_t peak = 0;
        };
        GlobalCPUStats& get_cpu_stats() {
            static GlobalCPUStats stats;
            return stats;
        }
    }

    void CPUAllocator::record_alloc(size_t bytes) {
        auto& stats = get_cpu_stats();
        std::lock_guard<std::mutex> lock(stats.mtx);
        stats.current += bytes;
        if (stats.current > stats.peak) stats.peak = stats.current;
    }

    void CPUAllocator::record_free(size_t bytes) {
        auto& stats = get_cpu_stats();
        std::lock_guard<std::mutex> lock(stats.mtx);
        if (stats.current >= bytes) stats.current -= bytes;
    }

    CPUAllocator::MemoryStats CPUAllocator::get_stats() {
        auto& stats = get_cpu_stats();
        std::lock_guard<std::mutex> lock(stats.mtx);
        return {stats.current, stats.peak};
    }

    void* CPUAllocator::allocate(size_t bytes)
    {
        // TODO(Optimization): Switch from malloc to posix_memalign (or aligned_alloc in C++17) 
        // for better alignment suitable for AVX/SIMD instructions.
        
        // Use std::malloc instead of new for consistency with other allocators
        void* ptr = std::malloc(bytes);
        
        // if (ptr == nullptr && bytes > 0) {
        //      throw std::bad_alloc();
        // }

        {
            auto& stats = get_cpu_stats();
            std::lock_guard<std::mutex> lock(stats.mtx);
            stats.allocs[ptr] = bytes;
            stats.current += bytes;
            if (stats.current > stats.peak) stats.peak = stats.current;
        }
        AllocationTracker::instance().on_alloc(ptr, bytes, bytes, -1); // -1 = CPU
        return ptr;
    }

    void CPUAllocator::deallocate(void* ptr)
    {
        if (!ptr) return;
        size_t freed_bytes = 0;
        {
            auto& stats = get_cpu_stats();
            std::lock_guard<std::mutex> lock(stats.mtx);
            auto it = stats.allocs.find(ptr);
            if (it != stats.allocs.end()) {
                freed_bytes = it->second;
                if (stats.current >= freed_bytes) stats.current -= freed_bytes;
                stats.allocs.erase(it);
            }
        }
        AllocationTracker::instance().on_free(ptr, -1); // -1 = CPU
        std::free(ptr);
    }

}