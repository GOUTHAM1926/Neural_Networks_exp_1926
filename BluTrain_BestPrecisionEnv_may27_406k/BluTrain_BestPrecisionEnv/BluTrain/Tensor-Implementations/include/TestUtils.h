#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <cstdint>
#include "core/Tensor.h"
#include <stdexcept>

namespace OwnTensor
{
    #define BY 1
    #define KB 1024
    #define MB 1024*1024
    #define GB MB*KB

    constexpr size_t operator"" _BY(unsigned long long v) { return v; }
    constexpr size_t operator"" _KB(unsigned long long v) { return v * KB; }
    constexpr size_t operator"" _MB(unsigned long long v) { return v * MB; }

    // Floating point version for 2.5_MB
    constexpr size_t operator"" _KB(long double v) { return static_cast<size_t>(v * 1024); }
    constexpr size_t operator"" _MB(long double v) { return static_cast<size_t>(v * 1024 * 1024); }

    #define ASSERT_NE(ptr1, ptr2) \
    do { \
        if (ptr1 == ptr2) { \
            std::cerr << "Allocation Failed! " << #ptr1 << " == " << #ptr2 << " is a nullptr " << "\n" \
                      << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
        std::cout << "ASSERTION PASSED - POINTERS ARE NOT POINTING TO THE SAME ADDRESS OR NULLPTR" << std::endl; \
    } while(0)

    #define ASSERT_EQ(ptr1, ptr2) \
    do { \
        if (ptr1 != ptr2) { \
            std::cerr << "Not the Same Address! " << #ptr1 << " == " << #ptr2 << " Cache miss " << "\n" \
                      << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
        std::cout << "ASSERTION PASSED - BOTH ARE POINTING TO THE SAME ADDRESS: "  << ptr1 << std::endl; \
    } while(0)

    void print_separator(const std::string & title)
    {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "  " << title << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }

    void print_tensor_info(const std::string & name, Tensor & t)
    {
        std::cout << "  " << name << ":" << std::endl;
        std::cout << "    Device:                     " << (t.is_cpu() ? "CPU" : "CUDA") << std::endl;
        std::cout << "    Address:                    " << t.data() << std::endl;
        std::cout << "    Numel:                      " << t.numel() << std::endl;
        std::cout << "    Nbytes:                     " << t.nbytes() << " Bytes" << std::endl;
        std::cout << "    Actual Bytes Allocated:     " << t.allocated_bytes() << " Bytes" << std::endl;
        std::cout << "    Storage Offset:             " << t.storage_offset() << std::endl;
        std::cout << "    Owns Data:                  " << (t.owns_data() ? "yes" : "no") << std::endl;
        std::cout << "    Is Contiguous:              " << (t.is_contiguous() ? "yes" : "no") << std::endl;
    }

    void print_tensor_data(const std::string & name, Tensor & t, int max_elems = 10)
    {
        std::cout << "  " << name << " data: [";
        float* ptr = t.data<float>();
        int count = std::min((int)t.numel(), max_elems);
        for (int i = 0; i < count; ++i)
        {
            std::cout << ptr[i] << (i < count - 1 ? ", " : "");
        }
        if ((int)t.numel() > max_elems) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    bool is_address_between(const void* address_to_check, const void* start_address, const void* end_address)
    {
        // Pointers can be directly compared in C++.
        // This checks if 'address_to_check' is greater than or equal to 'start_address'
        // AND less than 'end_address'.
        return address_to_check >= start_address && address_to_check < end_address;
    }

    // =============================================================================
    // CUDA GPU Memory Testing Utilities
    // =============================================================================

    #ifdef WITH_CUDA

    struct CudaMemorySnapshot {
        size_t free;
        size_t total;
        size_t used;

        CudaMemorySnapshot() : free(0), total(0), used(0)
        {
            cudaError_t err = cudaMemGetInfo(&free, &total);
            if (err == cudaSuccess)
            {
                used = total - free;
            }
        }

        size_t used_mb() const { return used / (1024 * 1024); }
        size_t free_mb() const { return free / (1024 * 1024); }
        size_t total_mb() const { return total / (1024 * 1024); }
    };

    inline CudaMemorySnapshot get_cuda_memory_snapshot()
    {
        return CudaMemorySnapshot();
    }

    inline void print_cuda_memory_info(const std::string & label = "")
    {
        CudaMemorySnapshot snap;
        std::cout << "  [GPU Memory" << (label.empty() ? "" : " - " + label) << "] "
            << "Used: " << snap.used_mb() << " MB, "
            << "Free: " << snap.free_mb() << " MB, "
            << "Total: " << snap.total_mb() << " MB" << std::endl;
    }

    inline void assert_gpu_memory_increased(const CudaMemorySnapshot & before,
        const CudaMemorySnapshot & after,
        size_t min_expected_bytes = 0)
    {
        if (after.used <= before.used)
        {
            std::cerr << "GPU memory did NOT increase as expected!\n"
                << "  Before: " << before.used_mb() << " MB\n"
                << "  After: " << after.used_mb() << " MB\n";
            std::abort();
        }
        if (min_expected_bytes > 0)
        {
            size_t actual_increase = after.used - before.used;
            if (actual_increase < min_expected_bytes)
            {
                std::cerr << "GPU memory increase less than expected!\n"
                    << "  Expected: >= " << min_expected_bytes / (1024 * 1024) << " MB\n"
                    << "  Actual: " << actual_increase / (1024 * 1024) << " MB\n";
                std::abort();
            }
        }
        std::cout << "✔ GPU memory increased: " << before.used_mb() << " MB -> "
            << after.used_mb() << " MB (+" << (after.used - before.used) / (1024 * 1024) << " MB)" << std::endl;
    }

    inline void assert_gpu_memory_decreased(const CudaMemorySnapshot & before,
        const CudaMemorySnapshot & after)
    {
        if (after.used >= before.used)
        {
            std::cerr << "GPU memory did NOT decrease as expected!\n"
                << "  Before: " << before.used_mb() << " MB\n"
                << "  After: " << after.used_mb() << " MB\n";
            std::abort();
        }
        std::cout << "✔ GPU memory decreased: " << before.used_mb() << " MB -> "
            << after.used_mb() << " MB (-" << (before.used - after.used) / (1024 * 1024) << " MB)" << std::endl;
    }

    inline void assert_gpu_memory_unchanged(const CudaMemorySnapshot & before,
        const CudaMemorySnapshot & after,
        size_t tolerance_bytes = 1024 * 1024)
    {
        size_t diff = (after.used > before.used) ? (after.used - before.used) : (before.used - after.used);
        if (diff > tolerance_bytes)
        {
            std::cerr << "GPU memory changed unexpectedly!\n"
                << "  Before: " << before.used_mb() << " MB\n"
                << "  After: " << after.used_mb() << " MB\n"
                << "  Tolerance: " << tolerance_bytes / (1024 * 1024) << " MB\n";
            std::abort();
        }
        std::cout << "✔ GPU memory unchanged (within tolerance): " << before.used_mb() << " MB -> "
            << after.used_mb() << " MB" << std::endl;
    }

    #else
    // Stub implementations when CUDA is not available
        struct CudaMemorySnapshot {
            size_t free = 0, total = 0, used = 0;
            size_t used_mb() const { return 0; }
            size_t free_mb() const { return 0; }
            size_t total_mb() const { return 0; }
        };
        inline CudaMemorySnapshot get_cuda_memory_snapshot() { return CudaMemorySnapshot(); }
        inline void print_cuda_memory_info(const std::string & = "")
        {
            std::cout << "  [GPU Memory] CUDA not available" << std::endl;
        }
        inline void assert_gpu_memory_increased(const CudaMemorySnapshot&, const CudaMemorySnapshot&, size_t = 0) { }
        inline void assert_gpu_memory_decreased(const CudaMemorySnapshot&, const CudaMemorySnapshot&) { }
        inline void assert_gpu_memory_unchanged(const CudaMemorySnapshot&, const CudaMemorySnapshot&, size_t = 0) { }
    #endif

    //===============================================================
    //  Allocator and Latency Benchmarking
    //===============================================================

    struct BenchResult {
        std::string name;
        double total_time_ms;
        double avg_time_us;
        double min_time_us;
        double max_time_us;
        double p50_time_us;
        double p95_time_us;
        double p99_time_us;
        size_t num_operations;
        size_t cache_hits;
        size_t cache_misses;
        size_t peak_memory_bytes;
        double throughput_ops_per_sec;
    };

    class Timer {
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point start_;

        public:
            void start() { start_ = Clock::now(); }

            double elapsed_us() const {
                Clock::time_point end = Clock::now();
                return std::chrono::duration<double, std::micro>(end - start_).count();
            }

            double elapsed_ms() const {
                return elapsed_us() / 1000.0f;
            }
    };

    class LatencyRecorder {
        std::vector<double> latencies_us_;

        public:
            void record(double latency_us) {
                latencies_us_.push_back(latency_us);
            }

            double average() const {
                if (latencies_us_.empty()) return 0;
                double sum = 0;
                for (double l : latencies_us_) sum += l;
                return sum / latencies_us_.size();
            }

            double min() const {
                if (latencies_us_.empty()) return 0;
                return *std::min_element(latencies_us_.begin(), latencies_us_.end());
            }

            double max() const {
                if (latencies_us_.empty()) return 0;
                return *std::max_element(latencies_us_.begin(), latencies_us_.end());
            }

            double percentile(double p) const {
                if (latencies_us_.empty()) return 0;
                std::vector<double> sorted = latencies_us_;
                std::sort(sorted.begin(), sorted.end());
                size_t idx = static_cast<size_t>(p * sorted.size() / 100.0);
                if (idx >= sorted.size()) idx = sorted.size() - 1;
                return sorted[idx];
            }

            size_t count() const {
                return latencies_us_.size();
            }
    };

    inline void print_result(const BenchResult& r) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(58) << r.name << "║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Operations:      " << std::setw(40) << r.num_operations << " ║\n";
    std::cout << "║ Total Time:      " << std::setw(37) << (std::to_string(r.total_time_ms) + " ms") << "    ║\n";
    std::cout << "║ Throughput:      " << std::setw(34) << (std::to_string((int)r.throughput_ops_per_sec) + " ops/s") << "       ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Avg Latency:     " << std::setw(37) << (std::to_string(r.avg_time_us) + " µs") << "     ║\n";
    std::cout << "║ Min Latency:     " << std::setw(37) << (std::to_string(r.min_time_us) + " µs") << "     ║\n";
    std::cout << "║ Max Latency:     " << std::setw(37) << (std::to_string(r.max_time_us) + " µs") << "     ║\n";
    std::cout << "║ P50 Latency:     " << std::setw(37) << (std::to_string(r.p50_time_us) + " µs") << "     ║\n";
    std::cout << "║ P95 Latency:     " << std::setw(37) << (std::to_string(r.p95_time_us) + " µs") << "     ║\n";
    std::cout << "║ P99 Latency:     " << std::setw(37) << (std::to_string(r.p99_time_us) + " µs") << "     ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Cache Hits:      " << std::setw(40) << r.cache_hits << " ║\n";
    std::cout << "║ Cache Misses:    " << std::setw(40) << r.cache_misses << " ║\n";
    double hit_rate = (r.cache_hits + r.cache_misses > 0) 
        ? 100.0 * r.cache_hits / (r.cache_hits + r.cache_misses) : 0;
    std::cout << "║ Hit Rate:        " << std::setw(38) << (std::to_string(hit_rate) + "%") << "   ║\n";
    std::cout << "║ Peak Memory:     " << std::setw(36) << (std::to_string((r.peak_memory_bytes) / (1024*1024)) + " MB") << "     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
}
}

#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "dtype/DtypeCastUtils.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace TestUtils {
using namespace OwnTensor;
// ============================================================================
// Compare two numeric values with absolute tolerance
// ============================================================================
template<typename T1, typename T2>
bool compare_values(T1 a, T2 b, double tolerance = 1e-6) {
    return std::abs(static_cast<double>(a) - static_cast<double>(b)) < tolerance;
}

// ============================================================================
// Compare with relative tolerance (better for exp, log, etc.)
// ============================================================================
template<typename T1, typename T2>
bool compare_values_relative(T1 a, T2 b, double rel_tol = 1e-3, double abs_tol = 1e-6) {
    double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
    double max_val = std::max(std::abs(static_cast<double>(a)), 
                              std::abs(static_cast<double>(b)));
    
    // Use relative tolerance for large values, absolute for small values
    return diff <= std::max(rel_tol * max_val, abs_tol);
}

// ============================================================================
// Verify tensor values match expected values with adaptive tolerance
// ============================================================================
// In your test_utils.h or wherever verify_tensor_values is defined:

bool verify_tensor_values(const Tensor& tensor, const std::vector<float>& expected, double tolerance) {
    if (tensor.numel() != expected.size()) return false;
    
    // Move to CPU if needed
    Tensor cpu_tensor = tensor.is_cpu() ? tensor : tensor.to(DeviceIndex(Device::CPU));
    
    // Handle different dtypes
    if (cpu_tensor.dtype() == Dtype::Float32) {
        const float* data = cpu_tensor.data<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(data[i] - expected[i]) > tolerance) return false;
        }
        return true;
    } 
    else if (cpu_tensor.dtype() == Dtype::Float64) {
        const double* data = cpu_tensor.data<double>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(data[i] - expected[i]) > tolerance) return false;
        }
        return true;
    }
    // ADD INTEGER SUPPORT:
    else if (cpu_tensor.dtype() == Dtype::Int16) {
        const int16_t* data = cpu_tensor.data<int16_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(data[i] - static_cast<int16_t>(expected[i])) > tolerance) return false;
        }
        return true;
    }
    else if (cpu_tensor.dtype() == Dtype::Int32) {
        const int32_t* data = cpu_tensor.data<int32_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(data[i] - static_cast<int32_t>(expected[i])) > tolerance) return false;
        }
        return true;
    }
    else if (cpu_tensor.dtype() == Dtype::Int64) {
        const int64_t* data = cpu_tensor.data<int64_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(data[i] - static_cast<int64_t>(expected[i])) > tolerance) return false;
        }
        return true;
    }
    else if (cpu_tensor.dtype() == Dtype::Float16 || cpu_tensor.dtype() == Dtype::Bfloat16) {
        // Convert to float32 for comparison
        Tensor temp = convert_half_to_float32(cpu_tensor);
        const float* data = temp.data<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(data[i] - expected[i]) > tolerance) return false;
        }
        return true;
    }
    else {
        std::cout << "Unsupported dtype for verify_tensor_values\n";
        return false;
    }
}

// ============================================================================
// Create tensor from float vector with automatic dtype conversion
// ============================================================================
inline OwnTensor::Tensor create_tensor_from_float(
    const std::vector<float>& data,
    const OwnTensor::DeviceIndex& device,
    OwnTensor::Dtype dtype
) {
    using namespace OwnTensor;
    
    // Create tensor (allocates memory on correct device)
    Tensor tensor({{static_cast<int64_t>(data.size())}}, dtype, device);
    
    // For GPU tensors, we MUST copy via cudaMemcpy, not set_data()
    if (device.is_cuda()) {
#ifdef WITH_CUDA
        // Convert data to target dtype on CPU first
        if (dtype == Dtype::Int16) {
            std::vector<int16_t> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<int16_t>(), converted.data(), 
                      converted.size() * sizeof(int16_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Int32) {
            std::vector<int32_t> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<int32_t>(), converted.data(), 
                      converted.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Int64) {
            std::vector<int64_t> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<int64_t>(), converted.data(), 
                      converted.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Float32) {
            cudaMemcpy(tensor.data<float>(), data.data(), 
                      data.size() * sizeof(float), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Float64) {
            std::vector<double> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<double>(), converted.data(), 
                      converted.size() * sizeof(double), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Float16) {
            std::vector<float16_t> converted;
            for (float f : data) converted.push_back(float16_t(f));
            cudaMemcpy(tensor.data<float16_t>(), converted.data(), 
                      converted.size() * sizeof(float16_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Bfloat16) {
            std::vector<bfloat16_t> converted;
            for (float f : data) converted.push_back(bfloat16_t(f));
            cudaMemcpy(tensor.data<bfloat16_t>(), converted.data(), 
                      converted.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice);
        }
        
        // CRITICAL: Synchronize after copy!
        cudaDeviceSynchronize();
#endif
    } else {
        // CPU path - use set_data() as before
        if (dtype == Dtype::Int16) {
            std::vector<int16_t> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Int32) {
            std::vector<int32_t> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Int64) {
            std::vector<int64_t> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Float32) {
            tensor.set_data(data);
        } else if (dtype == Dtype::Float64) {
            std::vector<double> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Float16) {
            std::vector<float16_t> converted;
            for (float f : data) converted.push_back(float16_t(f));
            tensor.set_data(converted);
        } else if (dtype == Dtype::Bfloat16) {
            std::vector<bfloat16_t> converted;
            for (float f : data) converted.push_back(bfloat16_t(f));
            tensor.set_data(converted);
        }
    }
    
    return tensor;
}

} // namespace TestUtils