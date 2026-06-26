#pragma once

#include "device/TuningConfig.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace OwnTensor {

// ---------------------------------------------------------------------------
// SizeClassTuner
//
// Collects allocation size frequencies during a warm-up phase, then computes
// optimal size class boundaries for the large pool (>= 1MB) using 1D dynamic
// programming quantization.
//
// Usage:
//   1. Call record_allocation() on every large-pool allocation during warm-up
//   2. Call compute_optimal_boundaries() after warm-up completes
//   3. Apply returned boundaries via SizeClass::set_boundaries()
// ---------------------------------------------------------------------------
class SizeClassTuner {
public:
    explicit SizeClassTuner(const TuningConfig& config);

    // Record a single allocation's requested size (pre-rounding).
    // Thread-safe. Only records sizes >= 1MB (large pool).
    void record_allocation(size_t requested_bytes);

    // Compute optimal boundaries from collected data.
    // Returns sorted vector of bucket boundary sizes.
    // If insufficient data, returns default boundaries.
    std::vector<size_t> compute_optimal_boundaries() const;

    // Benchmark mode: generate all candidate boundary configs by sweeping
    // k in [k_min, k_max] and max_bucket_width_mb across the given widths.
    // Candidate 0 is always the hardcoded default baseline.
    // Must be called after warm-up (record_allocation phase).
    struct BenchmarkCandidate {
        size_t k;                       // number of buckets
        size_t max_width_mb;            // max_bucket_width_mb used
        std::vector<size_t> boundaries; // computed boundaries (bytes)
        double dp_waste;                // theoretical waste from DP
    };
    std::vector<BenchmarkCandidate> generate_candidates(
        size_t k_min, size_t k_max,
        const std::vector<size_t>& widths_mb) const;

    // Bounds-based candidate generation for benchmark mode.
    // Explores bracket placements within [max_request, max_request + overhead]
    // using sampling strategies for diverse config exploration.
    // If k_min==0 && k_max==0, explores all possible bracket counts [1..N].
    std::vector<BenchmarkCandidate> generate_benchmark_candidates(
        size_t k_min, size_t k_max,
        double overhead_bound_mb,
        size_t num_candidates,
        const std::string& sampling_strategy,
        uint64_t seed) const;

    // Stats
    size_t total_allocations() const;
    size_t unique_sizes() const;

    // Reset collected data
    void clear();

private:
    TuningConfig config_;

    mutable std::mutex mu_;
    std::unordered_map<size_t, size_t> size_frequency_;  // requested_size -> count
    size_t total_allocs_ = 0;

    static constexpr size_t kSmallSize = 1048576;  // 1MB -- large pool threshold
    static constexpr size_t ALIGN_MB_2 = 2097152;  // 2MB alignment for overflow

    // DP internals
    struct DPResult {
        double total_waste;
        std::vector<size_t> boundaries;
    };

    // Solve optimal k-partition for the given sorted sizes/freqs.
    DPResult solve_dp(const std::vector<size_t>& sizes,
                      const std::vector<size_t>& freqs,
                      size_t k) const;
};

} // namespace OwnTensor
