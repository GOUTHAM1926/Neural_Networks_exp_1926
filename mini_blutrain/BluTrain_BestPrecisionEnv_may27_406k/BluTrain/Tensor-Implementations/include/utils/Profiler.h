#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <algorithm>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "device/CachingCudaAllocator.h"
#endif

namespace OwnTensor {
namespace autograd {

/**
 * @brief Record for a single operation profiling event
 */
struct ProfileRecord {
    std::string name;
    double duration_ms;
    bool is_gpu;
    size_t count = 0;

    ProfileRecord() : duration_ms(0), is_gpu(false) {}
    ProfileRecord(std::string n, double d, bool gpu) 
        : name(std::move(n)), duration_ms(d), is_gpu(gpu), count(1) {}
};

/**
 * @brief Thread-safe profiler for Autograd operations with hardware & memory metrics
 */
class Profiler {
public:
    static Profiler& instance();

    void set_enabled(bool enabled);
    bool is_enabled() const;

    void record_op(const char* name, double duration_ms, bool is_gpu = false);
    void reset();
    void print_stats();

private:
    Profiler();
    bool enabled_;
    
    // Thread-local statistics map
    static std::unordered_map<std::string, ProfileRecord>& get_local_stats();
    
    // Registry of all thread-local maps (protected by mutex)
    std::vector<std::unordered_map<std::string, ProfileRecord>*> thread_stats_registry_;
    std::mutex mutex_;
};

/**
 * @brief RAII Guard for profiling a single block with NVTX integration
 */
class ProfileGuard {
public:
    ProfileGuard(const char* name, bool is_gpu = false) 
        : name_(name), is_gpu_(is_gpu) {
        if (Profiler::instance().is_enabled()) {
#ifdef WITH_CUDA
            // NVTX is very fast and safe for timelines
            nvtxRangePushA(name_);
#endif
            cpu_start_ = std::chrono::high_resolution_clock::now();
        }
    }

    ~ProfileGuard() {
        if (Profiler::instance().is_enabled()) {
#ifdef WITH_CUDA
            nvtxRangePop();
#endif
            auto cpu_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start_).count();
            // We only record the CPU launch/overhead time in the internal report
            // to avoid stalling the GPU pipeline.
            Profiler::instance().record_op(name_, ms, is_gpu_);
        }
    }

private:
    const char* name_;
    bool is_gpu_;
    std::chrono::high_resolution_clock::time_point cpu_start_;
};

} // namespace autograd
} // namespace OwnTensor

// Macros for high-level instrumentation
#ifdef AUTOGRAD_PROFILER_ENABLED
#define AUTO_PROFILE(name) OwnTensor::autograd::ProfileGuard prof_guard_##__LINE__(name, false)
#define AUTO_PROFILE_CUDA(name) OwnTensor::autograd::ProfileGuard prof_guard_##__LINE__(name, true)
#else
#define AUTO_PROFILE(name)
#define AUTO_PROFILE_CUDA(name)
#endif
