#include "device/SizeClassTuner.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace OwnTensor {

// ---------------------------------------------------------------------------
// align_boundaries -- round each boundary UP to the nearest multiple of
// align_bytes, deduplicate, and ensure ascending order.
// When align_bytes == 0 the vector is returned unchanged (just sorted).
// ---------------------------------------------------------------------------
static std::vector<size_t> align_boundaries(
        const std::vector<size_t>& raw, size_t align_bytes) {

    std::vector<size_t> out;
    out.reserve(raw.size());

    for (size_t b : raw) {
        if (align_bytes > 0) {
            b = ((b + align_bytes - 1) / align_bytes) * align_bytes;
        }
        out.push_back(b);
    }

    // Sort ascending (offsets in benchmark mode can break ordering)
    std::sort(out.begin(), out.end());

    // Deduplicate (adjacent equal values after rounding)
    out.erase(std::unique(out.begin(), out.end()), out.end());

    return out;
}

SizeClassTuner::SizeClassTuner(const TuningConfig& config) : config_(config) {}

void SizeClassTuner::record_allocation(size_t requested_bytes) {
    if (requested_bytes < kSmallSize) return;  // only large pool

    std::lock_guard<std::mutex> lock(mu_);
    size_frequency_[requested_bytes]++;
    total_allocs_++;
}

size_t SizeClassTuner::total_allocations() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_allocs_;
}

size_t SizeClassTuner::unique_sizes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return size_frequency_.size();
}

void SizeClassTuner::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    size_frequency_.clear();
    total_allocs_ = 0;
}

// ---------------------------------------------------------------------------
// DP-based optimal 1D quantization
//
// Given N sorted unique sizes with frequencies, partition into k contiguous
// groups to minimize total waste:
//   waste(group) = sum_i (group_max - size_i) * freq_i
//
// dp[j][i] = minimum waste for partitioning sizes[0..i] into j groups
// cost(l, r) = sizes[r] * sum_freq(l..r) - sum_sf(l..r)
//   where sum_sf = sum of (size * freq)
// ---------------------------------------------------------------------------

SizeClassTuner::DPResult SizeClassTuner::solve_dp(
        const std::vector<size_t>& sizes,
        const std::vector<size_t>& freqs,
        size_t k) const {

    size_t N = sizes.size();
    if (k > N) k = N;  // can't have more buckets than unique sizes

    // Prefix sums for O(1) cost queries
    // prefix_freq[i] = sum of freqs[0..i-1]
    // prefix_sf[i]   = sum of sizes[j]*freqs[j] for j in 0..i-1
    std::vector<double> prefix_freq(N + 1, 0.0);
    std::vector<double> prefix_sf(N + 1, 0.0);

    for (size_t i = 0; i < N; i++) {
        prefix_freq[i + 1] = prefix_freq[i] + (double)freqs[i];
        prefix_sf[i + 1]   = prefix_sf[i] + (double)sizes[i] * (double)freqs[i];
    }

    const double INF = std::numeric_limits<double>::max();

    // cost(l, r) = waste of putting sizes[l..r] in one bucket (bucket = sizes[r])
    auto cost = [&](size_t l, size_t r) -> double {
        if (config_.max_bucket_width_mb > 0) {
            if ((sizes[r] - sizes[l]) > config_.max_bucket_width_mb * 1024 * 1024) {
                return INF;
            }
        }
        double freq_sum = prefix_freq[r + 1] - prefix_freq[l];
        double sf_sum   = prefix_sf[r + 1] - prefix_sf[l];
        return (double)sizes[r] * freq_sum - sf_sum;
    };

    // dp[j][i] = min waste for sizes[0..i] in j groups
    // split[j][i] = optimal split point for backtracking
    std::vector<std::vector<double>> dp(k + 1, std::vector<double>(N, INF));
    std::vector<std::vector<size_t>> split(k + 1, std::vector<size_t>(N, 0));

    // Base case: 1 group covering sizes[0..i]
    for (size_t i = 0; i < N; i++) {
        dp[1][i] = cost(0, i);
    }

    // Fill DP table
    for (size_t j = 2; j <= k; j++) {
        for (size_t i = j - 1; i < N; i++) {
            for (size_t m = j - 2; m < i; m++) {
                double candidate = dp[j - 1][m] + cost(m + 1, i);
                if (candidate < dp[j][i]) {
                    dp[j][i] = candidate;
                    split[j][i] = m;
                }
            }
        }
    }

    // Backtrack to find partition points
    DPResult result;
    result.total_waste = dp[k][N - 1];

    std::vector<size_t> partition_ends;
    size_t cur = N - 1;
    for (size_t j = k; j >= 2; j--) {
        partition_ends.push_back(cur);
        cur = split[j][cur];
    }
    partition_ends.push_back(cur);  // first group ends at cur

    std::reverse(partition_ends.begin(), partition_ends.end());

    // Each group's bucket = sizes[group_end] (the max in that group)
    for (size_t end_idx : partition_ends) {
        result.boundaries.push_back(sizes[end_idx]);
    }

    return result;
}

// ---------------------------------------------------------------------------
// compute_optimal_boundaries
// ---------------------------------------------------------------------------

std::vector<size_t> SizeClassTuner::compute_optimal_boundaries() const {
    std::lock_guard<std::mutex> lock(mu_);

    // Default boundaries (matches current hardcoded config)
    // std::vector<size_t> default_bounds = {2097152, 10485760, 13631488, 20971520}; // 2MB, 10MB, 13MB, 20MB
    std::vector<size_t> default_bounds = { 2097152, 3145728, 4194304, 7340032, 10485760, 20971520 }; // 2,3,4,7,10,20

    if (size_frequency_.empty()) {
        std::cerr << "[SizeClassTuner] No allocations recorded -- keeping defaults\n";
        return default_bounds;
    }

    // Sort unique sizes
    std::vector<std::pair<size_t, size_t>> sorted_sizes(
        size_frequency_.begin(), size_frequency_.end());
    std::sort(sorted_sizes.begin(), sorted_sizes.end());

    std::vector<size_t> sizes, freqs;
    for (auto& [sz, freq] : sorted_sizes) {
        sizes.push_back(sz);
        freqs.push_back(freq);
    }

    size_t N = sizes.size();

    std::cerr << "[SizeClassTuner] Profiled " << total_allocs_ << " allocations, "
              << N << " unique sizes (>= 1MB)\n";
    std::cerr << "[SizeClassTuner] Size range: "
              << sizes.front() / (1024*1024) << "MB - "
              << sizes.back() / (1024*1024) << "MB\n";

    // Trivial cases
    if (N == 1) {
        std::cerr << "[SizeClassTuner] Single unique size -- one bucket\n";
        return {sizes[0]};
    }
    if (N == 2) {
        std::cerr << "[SizeClassTuner] Two unique sizes -- two buckets\n";
        return {sizes[0], sizes[1]};
    }

    // Compute total requested bytes (for waste ratio calculation)
    double total_requested = 0.0;
    for (size_t i = 0; i < N; i++) {
        total_requested += (double)sizes[i] * (double)freqs[i];
    }

    // Run DP for increasing k, find the sweet spot
    size_t max_k = std::min(config_.max_buckets, N);
    size_t min_k = std::min(config_.min_buckets, max_k);
    if (min_k < 2) min_k = 2; // Can't have fewer than 2 buckets (or fallback to logic later)
    DPResult best;
    best.total_waste = std::numeric_limits<double>::max();
    size_t best_k = 1;

    double prev_waste = std::numeric_limits<double>::max();

    for (size_t k = 2; k <= max_k; k++) {
        DPResult result = solve_dp(sizes, freqs, k);
        double waste_ratio = result.total_waste / total_requested;
        double improvement = (prev_waste - result.total_waste) / prev_waste;

        std::cerr << "[SizeClassTuner] k=" << k
                  << " waste=" << (size_t)result.total_waste
                  << " ratio=" << (waste_ratio * 100.0) << "%"
                  << " improvement=" << (improvement * 100.0) << "%\n";

        best = result;
        best_k = k;

        // Stop conditions
        if (k >= min_k) {
            if (waste_ratio < config_.waste_threshold) {
                std::cerr << "[SizeClassTuner] Waste ratio below threshold ("
                          << (config_.waste_threshold * 100.0) << "%) -- stopping at k=" << k << "\n";
                break;
            }
            if (k > 2 && !config_.strict_waste_target && improvement < 0.05) {
                std::cerr << "[SizeClassTuner] Diminishing returns (<5% improvement) -- stopping at k=" << k << "\n";
                break;
            }
        }

        prev_waste = result.total_waste;
    }

    // Validate: if computed boundaries produce worse waste than defaults,
    // keep defaults
    // Simulate default waste
    double default_waste = 0.0;
    for (size_t i = 0; i < N; i++) {
        size_t rounded = 0;
        for (size_t b : default_bounds) {
            if (sizes[i] <= b) { rounded = b; break; }
        }
        if (rounded == 0) {
            // Above all default bounds -- 2MB alignment
            rounded = ((sizes[i] + ALIGN_MB_2 - 1) / ALIGN_MB_2) * ALIGN_MB_2;
        }
        default_waste += (double)(rounded - sizes[i]) * (double)freqs[i];
    }

    double computed_ratio = best.total_waste / total_requested;
    double default_ratio = default_waste / total_requested;

    std::cerr << "[SizeClassTuner] Computed waste: " << (computed_ratio * 100.0)
              << "% (" << best_k << " buckets) vs Default waste: "
              << (default_ratio * 100.0) << "%\n";

    if (best.total_waste >= std::numeric_limits<double>::max() / 2.0) {
        std::cerr << "[SizeClassTuner] WARNING: Impossible to satisfy max_bucket_width_mb constraint with " 
                  << best_k << " buckets -- falling back to defaults\n";
        return default_bounds;
    }

    if (best.total_waste > default_waste) {
        std::cerr << "[SizeClassTuner] WARNING: Computed boundaries are worse than defaults -- keeping defaults\n";
        return default_bounds;
    }

    // Log final boundaries
    std::cerr << "[SizeClassTuner] Final boundaries: [";
    for (size_t i = 0; i < best.boundaries.size(); i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << best.boundaries[i] / (1024*1024) << "MB"
                  << " (" << best.boundaries[i] << "B)";
    }
    std::cerr << "]\n";

    double savings = default_waste - best.total_waste;
    std::cerr << "[SizeClassTuner] Estimated savings: "
              << (size_t)(savings / (1024*1024)) << "MB total waste reduction\n";

    // Apply boundary alignment if configured
    size_t align_bytes = (size_t)(config_.boundary_align_mb * 1024.0 * 1024.0);
    auto aligned = align_boundaries(best.boundaries, align_bytes);
    if (align_bytes > 0 && aligned.size() != best.boundaries.size()) {
        std::cerr << "[SizeClassTuner] Alignment (" << config_.boundary_align_mb
                  << "MB) merged " << best.boundaries.size()
                  << " boundaries down to " << aligned.size() << "\n";
    }

    return aligned;
}

// ---------------------------------------------------------------------------
// generate_candidates -- benchmark mode exhaustive sweep
// ---------------------------------------------------------------------------

std::vector<SizeClassTuner::BenchmarkCandidate> SizeClassTuner::generate_candidates(
        size_t k_min, size_t k_max,
        const std::vector<size_t>& widths_mb) const {

    std::lock_guard<std::mutex> lock(mu_);

    std::vector<BenchmarkCandidate> candidates;

    // Candidate 0: default baseline with ACTUALLY COMPUTED waste
    {
        BenchmarkCandidate baseline;
        baseline.k = 0;
        baseline.max_width_mb = 0;
        baseline.boundaries = { 2097152, 3145728, 4194304, 7340032, 10485760, 20971520 }; // 2,3,4,7,10,20
;

        // Compute default waste (same logic as compute_optimal_boundaries())
        double default_waste = 0.0;
        for (auto& [sz, freq] : size_frequency_) {
            size_t rounded = 0;
            for (size_t b : baseline.boundaries) {
                if (sz <= b) { rounded = b; break; }
            }
            if (rounded == 0) {
                // Above all default bounds -- 2MB alignment
                rounded = ((sz + ALIGN_MB_2 - 1) / ALIGN_MB_2) * ALIGN_MB_2;
            }
            default_waste += (double)(rounded - sz) * (double)freq;
        }
        baseline.dp_waste = default_waste;
        candidates.push_back(std::move(baseline));
    }

    if (size_frequency_.empty()) {
        std::cerr << "[SizeClassTuner] No allocations recorded -- only baseline candidate\n";
        return candidates;
    }

    // Sort unique sizes
    std::vector<std::pair<size_t, size_t>> sorted_sizes(
        size_frequency_.begin(), size_frequency_.end());
    std::sort(sorted_sizes.begin(), sorted_sizes.end());

    std::vector<size_t> sizes, freqs;
    for (auto& [sz, freq] : sorted_sizes) {
        sizes.push_back(sz);
        freqs.push_back(freq);
    }

    size_t N = sizes.size();

    // Sweep all (k, width) combinations
    for (size_t width_mb : widths_mb) {
        // Temporarily override config width for DP solver
        TuningConfig tmp_config = config_;
        tmp_config.max_bucket_width_mb = width_mb;

        // Build a temporary tuner with this config to use solve_dp
        // (solve_dp reads config_.max_bucket_width_mb internally)
        // We use a const_cast-free approach: just call solve_dp directly
        // since we have access to sizes/freqs already

        size_t effective_k_max = std::min(k_max, N);
        size_t effective_k_min = std::min(k_min, effective_k_max);
        if (effective_k_min < 2) effective_k_min = 2;

        for (size_t k = effective_k_min; k <= effective_k_max; k++) {
            // We need to call solve_dp with the width constraint.
            // solve_dp uses config_.max_bucket_width_mb, so we temporarily
            // swap it. This is safe because we hold the mutex.
            size_t old_width = config_.max_bucket_width_mb;
            const_cast<TuningConfig&>(config_).max_bucket_width_mb = width_mb;

            DPResult result = solve_dp(sizes, freqs, k);

            const_cast<TuningConfig&>(config_).max_bucket_width_mb = old_width;

            // Skip if DP couldn't satisfy constraints (INF waste)
            if (result.total_waste >= std::numeric_limits<double>::max() / 2.0) {
                std::cerr << "[Benchmark] k=" << k << " width=" << width_mb
                          << "MB -- infeasible, skipping\n";
                continue;
            }

            BenchmarkCandidate c;
            c.max_width_mb = width_mb;
            size_t align_bytes = (size_t)(config_.boundary_align_mb * 1024.0 * 1024.0);
            c.boundaries = align_boundaries(result.boundaries, align_bytes);
            c.k = c.boundaries.size();
            c.dp_waste = result.total_waste;
            candidates.push_back(std::move(c));
        }
    }

    std::cerr << "[SizeClassTuner] Generated " << candidates.size()
              << " benchmark candidates (including baseline)\n";
    return candidates;
}

// ---------------------------------------------------------------------------
// generate_benchmark_candidates -- bounds-based bracket search
//
// Instead of using unique allocation sizes as brackets (profile mode),
// this explores bracket placements within [max_request, max_request + bound].
// Generates diverse configs via partition perturbation + sampling.
// ---------------------------------------------------------------------------

std::vector<SizeClassTuner::BenchmarkCandidate> SizeClassTuner::generate_benchmark_candidates(
        size_t k_min, size_t k_max,
        double overhead_bound_mb,
        size_t num_candidates,
        const std::string& sampling_strategy,
        uint64_t seed) const {

    std::lock_guard<std::mutex> lock(mu_);

    std::vector<BenchmarkCandidate> candidates;

    // Candidate 0: default baseline
    {
        BenchmarkCandidate baseline;
        baseline.k = 0;
        baseline.max_width_mb = 0;
        baseline.boundaries = {2097152, 10485760, 13631488, 20971520};

        double default_waste = 0.0;
        for (auto& [sz, freq] : size_frequency_) {
            size_t rounded = 0;
            for (size_t b : baseline.boundaries) {
                if (sz <= b) { rounded = b; break; }
            }
            if (rounded == 0) {
                rounded = ((sz + ALIGN_MB_2 - 1) / ALIGN_MB_2) * ALIGN_MB_2;
            }
            default_waste += (double)(rounded - sz) * (double)freq;
        }
        baseline.dp_waste = default_waste;
        candidates.push_back(std::move(baseline));
    }

    if (size_frequency_.empty()) {
        std::cerr << "[SizeClassTuner] No allocations recorded -- only baseline candidate\n";
        return candidates;
    }

    // Sort unique sizes
    std::vector<std::pair<size_t, size_t>> sorted_sizes(
        size_frequency_.begin(), size_frequency_.end());
    std::sort(sorted_sizes.begin(), sorted_sizes.end());

    std::vector<size_t> sizes, freqs;
    for (auto& [sz, freq] : sorted_sizes) {
        sizes.push_back(sz);
        freqs.push_back(freq);
    }

    size_t N = sizes.size();
    size_t overhead_bytes = (size_t)(overhead_bound_mb * 1024.0 * 1024.0);

    // Resolve k bounds: if both 0, explore all possible
    if (k_min == 0 && k_max == 0) {
        k_min = 1;
        k_max = N;
    }
    if (k_min < 1) k_min = 1;
    if (k_max > N) k_max = N;
    if (k_min > k_max) k_min = k_max;

    // Prefix sums for O(1) waste computation
    std::vector<double> prefix_freq(N + 1, 0.0);
    std::vector<double> prefix_sf(N + 1, 0.0);
    for (size_t i = 0; i < N; i++) {
        prefix_freq[i + 1] = prefix_freq[i] + (double)freqs[i];
        prefix_sf[i + 1]   = prefix_sf[i] + (double)sizes[i] * (double)freqs[i];
    }

    // Setup RNG
    uint64_t actual_seed = seed;
    if (actual_seed == 0) {
        actual_seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }
    std::mt19937_64 rng(actual_seed);

    // Compute waste for a given set of boundaries against all allocation sizes
    auto compute_waste = [&](const std::vector<size_t>& bounds) -> double {
        double waste = 0.0;
        for (size_t i = 0; i < N; i++) {
            size_t rounded = 0;
            for (size_t b : bounds) {
                if (sizes[i] <= b) { rounded = b; break; }
            }
            if (rounded == 0) {
                rounded = ((sizes[i] + ALIGN_MB_2 - 1) / ALIGN_MB_2) * ALIGN_MB_2;
            }
            waste += (double)(rounded - sizes[i]) * (double)freqs[i];
        }
        return waste;
    };

    // For a given partition (split points), compute group max sizes
    // split_points: indices into sizes[] where groups end (inclusive)
    // e.g., for k=3, N=10: split_points = {2, 6, 9} means
    //   group 0: sizes[0..2], group 1: sizes[3..6], group 2: sizes[7..9]
    // Returns k max sizes (one per group)
    auto get_group_maxes = [&](const std::vector<size_t>& split_points)
            -> std::vector<size_t> {
        std::vector<size_t> maxes;
        for (size_t sp : split_points) {
            maxes.push_back(sizes[sp]);
        }
        return maxes;
    };

    // Generate bracket boundaries from group maxes + overhead offsets
    // offsets are in [0, 1] range, scaled to [0, overhead_bytes]
    // Brackets are aligned to boundary_align_mb if set (>0), otherwise
    // to 512-byte minimum alignment for CUDA.
    size_t cfg_align_bytes = (size_t)(config_.boundary_align_mb * 1024.0 * 1024.0);
    size_t min_align = (cfg_align_bytes > 0) ? cfg_align_bytes : 512;
    auto make_boundaries = [&](const std::vector<size_t>& group_maxes,
                               const std::vector<double>& offsets)
            -> std::vector<size_t> {
        std::vector<size_t> bounds;
        for (size_t g = 0; g < group_maxes.size(); g++) {
            size_t raw = group_maxes[g] + (size_t)(offsets[g] * overhead_bytes);
            size_t bracket = ((raw + min_align - 1) / min_align) * min_align;
            bounds.push_back(bracket);
        }
        // Sort + deduplicate (alignment can collapse close boundaries)
        return align_boundaries(bounds, 0);
    };

    // Struct to hold raw candidates before filtering
    struct RawCandidate {
        std::vector<size_t> boundaries;
        double waste;
        size_t k;
    };
    std::vector<RawCandidate> raw_candidates;

    // Budget allocation: distribute candidates across k values
    size_t k_range = k_max - k_min + 1;
    size_t budget_per_k = std::max((size_t)1, (num_candidates - 1) / k_range);

    for (size_t k = k_min; k <= k_max; k++) {
        if (k > N) break;

        // Get DP-optimal partition for this k
        size_t old_width = config_.max_bucket_width_mb;
        const_cast<TuningConfig&>(config_).max_bucket_width_mb = 0;
        DPResult dp_result = solve_dp(sizes, freqs, k);
        const_cast<TuningConfig&>(config_).max_bucket_width_mb = old_width;

        if (dp_result.total_waste >= std::numeric_limits<double>::max() / 2.0) {
            continue;
        }

        // Convert DP boundaries back to split point indices
        std::vector<size_t> dp_split_points;
        for (size_t b : dp_result.boundaries) {
            for (size_t i = 0; i < N; i++) {
                if (sizes[i] == b) {
                    dp_split_points.push_back(i);
                    break;
                }
            }
        }

        if (dp_split_points.size() != k) {
            // Fallback: evenly distribute split points
            dp_split_points.clear();
            for (size_t g = 0; g < k; g++) {
                dp_split_points.push_back(((g + 1) * N - 1) / k);
            }
        }

        // Collect all partitions to sample from: DP-optimal + perturbations
        std::vector<std::vector<size_t>> partitions;
        partitions.push_back(dp_split_points);

        // Perturb split points by +/-1, +/-2
        for (size_t sp_idx = 0; sp_idx < dp_split_points.size() - 1; sp_idx++) {
            for (int delta : {-2, -1, 1, 2}) {
                auto perturbed = dp_split_points;
                int new_val = (int)perturbed[sp_idx] + delta;

                // Validate: must stay within bounds and maintain ordering
                int lower = (sp_idx == 0) ? 0 : (int)perturbed[sp_idx - 1] + 1;
                int upper = (int)perturbed[sp_idx + 1] - 1;

                if (new_val >= lower && new_val <= upper && new_val >= 0 && (size_t)new_val < N) {
                    perturbed[sp_idx] = (size_t)new_val;
                    partitions.push_back(perturbed);
                }
            }
        }

        // For each partition, sample bracket placements within overhead bounds
        size_t samples_per_partition = std::max((size_t)1, budget_per_k / partitions.size());

        for (auto& partition : partitions) {
            auto group_maxes = get_group_maxes(partition);

            if (sampling_strategy == "grid") {
                // Grid: discretize each bracket's overhead range into steps
                size_t grid_steps = std::max((size_t)2,
                    (size_t)std::round(std::pow((double)samples_per_partition, 1.0 / k)));

                // For small k, do full Cartesian product
                if (k <= 4) {
                    // Recursive grid generation
                    std::vector<std::vector<double>> grid_offsets(k);
                    for (size_t g = 0; g < k; g++) {
                        for (size_t s = 0; s < grid_steps; s++) {
                            grid_offsets[g].push_back((double)s / (double)(grid_steps - 1));
                        }
                    }

                    // Cartesian product via iterative approach
                    std::vector<size_t> indices(k, 0);
                    bool done = false;
                    while (!done) {
                        std::vector<double> offsets(k);
                        for (size_t g = 0; g < k; g++) {
                            offsets[g] = grid_offsets[g][indices[g]];
                        }
                        auto bounds = make_boundaries(group_maxes, offsets);
                        double waste = compute_waste(bounds);
                        raw_candidates.push_back({bounds, waste, bounds.size()});

                        // Increment indices
                        size_t carry = k - 1;
                        while (true) {
                            indices[carry]++;
                            if (indices[carry] < grid_steps) break;
                            indices[carry] = 0;
                            if (carry == 0) { done = true; break; }
                            carry--;
                        }
                    }
                } else {
                    // For large k, sample random grid points
                    std::uniform_int_distribution<size_t> step_dist(0, grid_steps - 1);
                    for (size_t s = 0; s < samples_per_partition; s++) {
                        std::vector<double> offsets(k);
                        for (size_t g = 0; g < k; g++) {
                            offsets[g] = (double)step_dist(rng) / (double)(grid_steps - 1);
                        }
                        auto bounds = make_boundaries(group_maxes, offsets);
                        double waste = compute_waste(bounds);
                        raw_candidates.push_back({bounds, waste, bounds.size()});
                    }
                }
            } else if (sampling_strategy == "random") {
                std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
                for (size_t s = 0; s < samples_per_partition; s++) {
                    std::vector<double> offsets(k);
                    for (size_t g = 0; g < k; g++) {
                        offsets[g] = unit_dist(rng);
                    }
                    auto bounds = make_boundaries(group_maxes, offsets);
                    double waste = compute_waste(bounds);
                    raw_candidates.push_back({bounds, waste, bounds.size()});
                }
            } else {
                // Latin Hypercube Sampling (default)
                size_t m = samples_per_partition;
                if (m < 2) m = 2;

                // Create k permutation arrays
                std::vector<std::vector<size_t>> perms(k);
                for (size_t g = 0; g < k; g++) {
                    perms[g].resize(m);
                    std::iota(perms[g].begin(), perms[g].end(), 0);
                    std::shuffle(perms[g].begin(), perms[g].end(), rng);
                }

                std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
                for (size_t s = 0; s < m; s++) {
                    std::vector<double> offsets(k);
                    for (size_t g = 0; g < k; g++) {
                        offsets[g] = ((double)perms[g][s] + unit_dist(rng)) / (double)m;
                    }
                    auto bounds = make_boundaries(group_maxes, offsets);
                    double waste = compute_waste(bounds);
                    raw_candidates.push_back({bounds, waste, bounds.size()});
                }
            }
        }

        // Also add the exact DP-optimal boundaries (offset = 0 for all)
        {
            std::vector<double> zero_offsets(k, 0.0);
            auto group_maxes = get_group_maxes(dp_split_points);
            auto bounds = make_boundaries(group_maxes, zero_offsets);
            double waste = compute_waste(bounds);
            raw_candidates.push_back({bounds, waste, bounds.size()});
        }
    }

    // Sort by waste and keep top num_candidates
    std::sort(raw_candidates.begin(), raw_candidates.end(),
              [](const RawCandidate& a, const RawCandidate& b) {
                  return a.waste < b.waste;
              });

    // Deduplicate: skip candidates with identical boundaries
    std::vector<RawCandidate> unique_raw;
    for (auto& rc : raw_candidates) {
        bool dup = false;
        for (auto& existing : unique_raw) {
            if (existing.boundaries == rc.boundaries) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            unique_raw.push_back(std::move(rc));
            if (unique_raw.size() >= num_candidates) break;
        }
    }

    // Convert to BenchmarkCandidate
    for (auto& rc : unique_raw) {
        BenchmarkCandidate c;
        c.k = rc.boundaries.size();
        c.max_width_mb = 0;
        c.boundaries = std::move(rc.boundaries);
        c.dp_waste = rc.waste;
        candidates.push_back(std::move(c));
    }

    std::cerr << "[SizeClassTuner] Generated " << candidates.size()
              << " benchmark candidates (including baseline)"
              << " | k=[" << k_min << ".." << k_max << "]"
              << " overhead=" << overhead_bound_mb << "MB"
              << " strategy=" << sampling_strategy << "\n";
    return candidates;
}

} // namespace OwnTensor
