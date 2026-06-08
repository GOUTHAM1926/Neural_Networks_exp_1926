#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace OwnTensor {

enum class AllocEvent { ALLOC, FREE };

struct AllocationRecord {
    uint64_t id;
    int device;           // -1 = CPU, 0+ = CUDA device
    void* address;
    size_t bytes;            // requested bytes (pre-rounding)
    size_t allocated_bytes;  // actual allocated bytes (post-rounding)
    uint64_t timestamp_ns;
    std::string location; // Macro-defined source location
};

struct LeakInfo {
    uint64_t id;
    size_t bytes;
    int device;
    uint64_t age_ns;
    std::string location;
};

class AllocationTracker {
public:
    static AllocationTracker& instance();

    void init(const char* csv_path);
    void shutdown();

    // Location tracking (called via macros)
    static void set_location(const char* loc);
    static void clear_location();
    static const char* get_location();

    // Allocator hooks
    void on_alloc(void* ptr, size_t bytes, size_t allocated_bytes, int device);
    void on_free(void* ptr, int device);

    // Reporting (file-based - no console output)
    void write_leak_report(const char* output_path) const;
    std::vector<LeakInfo> get_leaks() const;

    // Stats accessors
    size_t get_current_allocated(int device = -1) const;
    size_t get_peak_allocated(int device = -1) const;
    size_t get_total_allocations() const;
    size_t get_total_deallocations() const;

    void reset_peak();

private:
    AllocationTracker() = default;

    // Thread-local location storage
    static std::string& current_location() {
        thread_local std::string loc = "UNKNOWN";
        return loc;
    }

    mutable std::mutex mtx_;
    std::unordered_map<void*, AllocationRecord> live_allocs_;
    std::ofstream csv_;
    bool initialized_ = false;
    std::atomic<uint64_t> alloc_counter_{0};
    std::atomic<uint64_t> dealloc_counter_{0};
    std::unordered_map<int, size_t> current_bytes_;
    std::unordered_map<int, size_t> peak_bytes_;
};

} // namespace OwnTensor

// ============================================================
// MACROS FOR LOCATION TRACKING
// ============================================================

#define TRACK_ALLOC_LOCATION(loc) \
    OwnTensor::AllocationTracker::set_location(loc)

#define TRACK_ALLOC_CLEAR() \
    OwnTensor::AllocationTracker::clear_location()

// Scoped version - auto-clears on scope exit
#define TRACK_ALLOC_SCOPE(loc) \
    struct _AllocScope_##__LINE__ { \
        _AllocScope_##__LINE__() { OwnTensor::AllocationTracker::set_location(loc); } \
        ~_AllocScope_##__LINE__() { OwnTensor::AllocationTracker::clear_location(); } \
    } _alloc_scope_instance_##__LINE__ 

    