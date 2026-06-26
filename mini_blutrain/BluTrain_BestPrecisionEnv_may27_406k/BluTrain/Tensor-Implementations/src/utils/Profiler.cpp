#include "utils/Profiler.h"
#include "device/CPUAllocator.h"
#include "device/PinnedCPUAllocator.h"

namespace OwnTensor {
namespace autograd {

Profiler& Profiler::instance() {
    static Profiler prof;
    return prof;
}

void Profiler::set_enabled(bool enabled) {
    enabled_ = enabled;
}

bool Profiler::is_enabled() const {
    return enabled_;
}

std::unordered_map<std::string, ProfileRecord>& Profiler::get_local_stats() {
    thread_local std::unordered_map<std::string, ProfileRecord> local_stats;
    static thread_local bool registered = false;
    if (!registered) {
        std::lock_guard<std::mutex> lock(Profiler::instance().mutex_);
        Profiler::instance().thread_stats_registry_.push_back(&local_stats);
        registered = true;
    }
    return local_stats;
}

void Profiler::record_op(const char* name, double duration_ms, bool is_gpu) {
    if (!enabled_) return;
    
    // No global lock here! Each thread records to its own map.
    auto& stats = get_local_stats();
    auto& record = stats[name];
    if (record.name.empty()) record.name = name;
    record.duration_ms += duration_ms;
    record.count++;
    record.is_gpu = is_gpu;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* stats_ptr : thread_stats_registry_) {
        if (stats_ptr) stats_ptr->clear();
    }
}

void Profiler::print_stats() {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Merge all thread-local stats into a temporary aggregate map
    std::unordered_map<std::string, ProfileRecord> aggregate_stats;
    for (auto* stats_ptr : thread_stats_registry_) {
        if (!stats_ptr) continue;
        for (auto& [name, record] : *stats_ptr) {
            auto& agg = aggregate_stats[name];
            if (agg.name.empty()) agg.name = name;
            agg.duration_ms += record.duration_ms;
            agg.count += record.count;
            agg.is_gpu = record.is_gpu;
        }
    }

    if (aggregate_stats.empty()) {
        std::cout << "\n[Autograd Profiler] No data collected.\n";
        return;
    }

    std::vector<ProfileRecord> sorted_stats;
    for (auto& [name, record] : aggregate_stats) {
        sorted_stats.push_back(record);
    }

    std::sort(sorted_stats.begin(), sorted_stats.end(), [](const auto& a, const auto& b) {
        return a.duration_ms > b.duration_ms;
    });

    std::cout << "\n" << std::string(85, '=') << "\n";
    std::cout << "                      AUTOGRAD PROFILER REPORT\n";
    std::cout << std::string(85, '-') << "\n";
    std::cout << std::left << std::setw(35) << "Operation" 
              << std::setw(10) << "Type"
              << std::setw(10) << "Count" 
              << std::setw(15) << "Total (ms)" 
              << std::setw(15) << "Avg (ms)" << "\n";
    std::cout << std::string(85, '-') << "\n";

    double grand_total = 0;
    for (const auto& record : sorted_stats) {
        double avg = record.duration_ms / (record.count > 0 ? record.count : 1);
        std::cout << std::left << std::setw(35) << record.name 
                  << std::setw(10) << (record.is_gpu ? "CUDA" : "CPU")
                  << std::setw(10) << record.count 
                  << std::fixed << std::setprecision(3)
                  << std::setw(15) << record.duration_ms 
                  << std::setw(15) << avg << "\n";
        grand_total += record.duration_ms;
    }
    
    std::cout << std::string(85, '-') << "\n";
#ifdef WITH_CUDA
    auto mem_stats = CachingCUDAAllocator::instance().get_stats();
    std::cout << "MEMORY METRICS (CUDA):\n";
    std::cout << "  - Peak Usage:     " << std::fixed << std::setprecision(2) 
              << (mem_stats.peak / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  - Current Alloc:  " << (mem_stats.allocated / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  - Cached (Pool):  " << (mem_stats.cached / 1024.0 / 1024.0) << " MB\n";
    double hit_rate = (100.0 * mem_stats.num_cache_hits / std::max(1UL, mem_stats.num_allocs));
    std::cout << "  - Cache Hit Rate: " << hit_rate << " %\n";
    std::cout << std::string(85, '-') << "\n";
#endif

    auto cpu_stats = CPUAllocator::get_stats();
    auto pinned_stats = device::PinnedCPUAllocator::get_stats();
    std::cout << "MEMORY METRICS (CPU):\n";
    std::cout << "  - Standard Peak:  " << std::fixed << std::setprecision(2) 
              << (cpu_stats.peak / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  - Pinned Peak:    " << (pinned_stats.peak / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  - Total CPU Peak: " << ((cpu_stats.peak + pinned_stats.peak) / 1024.0 / 1024.0) << " MB\n";
    std::cout << std::string(85, '-') << "\n";
    std::cout << "TOTAL TIME (Trace): " << std::fixed << std::setprecision(3) << grand_total << " ms\n";
    std::cout << std::string(85, '=') << "\n\n";
}

Profiler::Profiler() : enabled_(false) {}

} // namespace autograd
} // namespace OwnTensor
