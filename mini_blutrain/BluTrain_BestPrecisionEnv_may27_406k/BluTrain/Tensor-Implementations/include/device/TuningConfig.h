#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace OwnTensor {

struct TuningConfig {
    // Master toggle
    bool enabled = false;

    // Warm-up control
    size_t warmup_steps = 25;

    // Defrag / empty cache during warm-up for cleaner profiling
    bool defrag_during_warmup = true;
    size_t defrag_interval_steps = 1;
    bool empty_cache_during_warmup = true;

    // Algorithm parameters
    size_t min_buckets = 5;         // Minimum buckets before early-stopping
    size_t max_buckets = 10;        // Maximum buckets allowed
    size_t max_bucket_width_mb = 0; // Max span of a bucket (0 = disables constraint)
    bool strict_waste_target = false; // Ignore diminishing returns to reach lower waste
    double waste_threshold = 0.15;  // 15% internal fragmentation target

    // Persistence
    bool persist_results = true;
    std::string persist_path = "size_class_config.json";
    bool load_persisted = true;

    // Profile mode -- DP-sweep analysis of unique-size-based configs
    bool profile_mode = false;
    size_t profile_steps_per_candidate = 10;
    std::string profile_log_path = "profile_results.csv";
    size_t profile_k_min = 2;
    size_t profile_k_max = 10;
    std::vector<size_t> profile_widths = {0};  // max_bucket_width_mb values to sweep
    double profile_min_cache_hit_rate = 0.0;   // minimum cache hit rate (%), 0 = disabled
    double profile_max_fragmentation = 100.0;  // maximum fragmentation (%), 100 = disabled

    // Benchmark mode -- bounds-based bracket search
    bool benchmark_mode = false;
    size_t benchmark_steps_per_candidate = 10;
    std::string benchmark_log_path = "benchmark_results.csv";
    size_t benchmark_k_min = 3;
    size_t benchmark_k_max = 10;
    double benchmark_overhead_bound_mb = 1.0;       // per-bracket overhead bound
    size_t benchmark_num_candidates = 200;           // max candidates to generate
    std::string benchmark_sampling_strategy = "latin_hypercube"; // "grid", "random", "latin_hypercube"
    uint64_t benchmark_seed = 0;                     // 0 = time-based seed
    double benchmark_min_cache_hit_rate = 0.0;
    double benchmark_max_fragmentation = 100.0;

    // Boundary alignment -- round all computed boundaries UP to the nearest
    // multiple of this value (in MB).  E.g. 0.25 means 250 KB alignment so
    // 3.474 MB and 3.432 MB both become 3.5 MB (one class instead of two).
    // 0 = disabled (keep exact values from the DP / overhead search).
    double boundary_align_mb = 0.0;

    // Deep mode -- full model reinit per candidate (used with benchmark_mode)
    bool deep_mode = false;

    // Load config from a JSON file. Returns default config if file not found.
    static TuningConfig load(const std::string& path);

    // Load persisted size class boundaries from persist_path.
    // Returns empty vector if file not found or invalid.
    static std::vector<size_t> load_persisted_boundaries(const std::string& path);

    // Save computed boundaries to persist_path.
    static bool save_boundaries(const std::string& path,
                                const std::vector<size_t>& boundaries,
                                size_t unique_sizes_profiled,
                                size_t total_allocations_profiled,
                                double waste_ratio);
};

} // namespace OwnTensor
