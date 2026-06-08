#include "device/TuningConfig.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace OwnTensor {

// ---------------------------------------------------------------------------
// Minimal JSON value extraction helpers (no external dependency)
// These only handle the flat structure of our tuning config.
// ---------------------------------------------------------------------------

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Extract value string for a given key from JSON text.
// Handles: "key": value  where value is bool, number, or quoted string.
static std::string extract_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++; // skip colon

    // Skip whitespace
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
    if (pos >= json.size()) return "";

    // Quoted string
    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    }

    // Bool or number: read until comma, }, or whitespace
    auto end = json.find_first_of(",}\r\n", pos);
    if (end == std::string::npos) end = json.size();
    return trim(json.substr(pos, end - pos));
}

static bool parse_bool(const std::string& val, bool fallback) {
    if (val == "true") return true;
    if (val == "false") return false;
    return fallback;
}

static size_t parse_size(const std::string& val, size_t fallback) {
    if (val.empty()) return fallback;
    try { return std::stoull(val); }
    catch (...) { return fallback; }
}

static double parse_double(const std::string& val, double fallback) {
    if (val.empty()) return fallback;
    try { return std::stod(val); }
    catch (...) { return fallback; }
}

static uint64_t parse_uint64(const std::string& val, uint64_t fallback) {
    if (val.empty()) return fallback;
    try { return std::stoull(val); }
    catch (...) { return fallback; }
}

// Parse a JSON array of integers: [0, 10, 50, ...]
static std::vector<size_t> parse_size_array(const std::string& json, const std::string& key) {
    std::vector<size_t> result;
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;

    auto arr_start = json.find('[', pos);
    auto arr_end = json.find(']', arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) return result;

    std::string arr = json.substr(arr_start + 1, arr_end - arr_start - 1);
    std::istringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            try { result.push_back(std::stoull(token)); }
            catch (...) {}
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// TuningConfig::load
// ---------------------------------------------------------------------------

TuningConfig TuningConfig::load(const std::string& path) {
    TuningConfig cfg;
    std::string json = read_file(path);
    if (json.empty()) {
        std::cerr << "[TuningConfig] Config file not found: " << path
                  << " -- using defaults (tuning disabled)\n";
        return cfg;
    }

    cfg.enabled                  = parse_bool(extract_value(json, "enabled"), cfg.enabled);
    cfg.warmup_steps             = parse_size(extract_value(json, "warmup_steps"), cfg.warmup_steps);
    cfg.defrag_during_warmup     = parse_bool(extract_value(json, "defrag_during_warmup"), cfg.defrag_during_warmup);
    cfg.defrag_interval_steps    = parse_size(extract_value(json, "defrag_interval_steps"), cfg.defrag_interval_steps);
    cfg.empty_cache_during_warmup = parse_bool(extract_value(json, "empty_cache_during_warmup"), cfg.empty_cache_during_warmup);
    cfg.min_buckets              = parse_size(extract_value(json, "min_buckets"), cfg.min_buckets);
    cfg.max_buckets              = parse_size(extract_value(json, "max_buckets"), cfg.max_buckets);
    cfg.max_bucket_width_mb      = parse_size(extract_value(json, "max_bucket_width_mb"), cfg.max_bucket_width_mb);
    cfg.strict_waste_target      = parse_bool(extract_value(json, "strict_waste_target"), cfg.strict_waste_target);
    cfg.waste_threshold          = parse_double(extract_value(json, "waste_threshold"), cfg.waste_threshold);
    cfg.persist_results          = parse_bool(extract_value(json, "persist_results"), cfg.persist_results);
    cfg.persist_path             = extract_value(json, "persist_path");
    if (cfg.persist_path.empty()) cfg.persist_path = "size_class_config.json";
    cfg.load_persisted           = parse_bool(extract_value(json, "load_persisted"), cfg.load_persisted);

    // Profile mode fields (was benchmark_mode)
    cfg.profile_mode                  = parse_bool(extract_value(json, "profile_mode"), cfg.profile_mode);
    cfg.profile_steps_per_candidate   = parse_size(extract_value(json, "profile_steps_per_candidate"), cfg.profile_steps_per_candidate);
    std::string pr_log = extract_value(json, "profile_log_path");
    if (!pr_log.empty()) cfg.profile_log_path = pr_log;
    cfg.profile_k_min                 = parse_size(extract_value(json, "profile_k_min"), cfg.profile_k_min);
    cfg.profile_k_max                 = parse_size(extract_value(json, "profile_k_max"), cfg.profile_k_max);
    cfg.profile_min_cache_hit_rate    = parse_double(extract_value(json, "profile_min_cache_hit_rate"), cfg.profile_min_cache_hit_rate);
    cfg.profile_max_fragmentation     = parse_double(extract_value(json, "profile_max_fragmentation"), cfg.profile_max_fragmentation);

    auto pw = parse_size_array(json, "profile_widths");
    if (!pw.empty()) cfg.profile_widths = std::move(pw);

    // Benchmark mode fields (new bounds-based search)
    cfg.benchmark_mode                  = parse_bool(extract_value(json, "benchmark_mode"), cfg.benchmark_mode);
    cfg.benchmark_steps_per_candidate   = parse_size(extract_value(json, "benchmark_steps_per_candidate"), cfg.benchmark_steps_per_candidate);
    std::string bm_log = extract_value(json, "benchmark_log_path");
    if (!bm_log.empty()) cfg.benchmark_log_path = bm_log;
    cfg.benchmark_k_min                 = parse_size(extract_value(json, "benchmark_k_min"), cfg.benchmark_k_min);
    cfg.benchmark_k_max                 = parse_size(extract_value(json, "benchmark_k_max"), cfg.benchmark_k_max);
    cfg.benchmark_overhead_bound_mb     = parse_double(extract_value(json, "benchmark_overhead_bound_mb"), cfg.benchmark_overhead_bound_mb);
    cfg.benchmark_num_candidates        = parse_size(extract_value(json, "benchmark_num_candidates"), cfg.benchmark_num_candidates);
    std::string bm_strat = extract_value(json, "benchmark_sampling_strategy");
    if (!bm_strat.empty()) cfg.benchmark_sampling_strategy = bm_strat;
    cfg.benchmark_seed                  = parse_uint64(extract_value(json, "benchmark_seed"), cfg.benchmark_seed);
    cfg.benchmark_min_cache_hit_rate    = parse_double(extract_value(json, "benchmark_min_cache_hit_rate"), cfg.benchmark_min_cache_hit_rate);
    cfg.benchmark_max_fragmentation     = parse_double(extract_value(json, "benchmark_max_fragmentation"), cfg.benchmark_max_fragmentation);

    // Boundary alignment
    cfg.boundary_align_mb = parse_double(extract_value(json, "boundary_align_mb"), cfg.boundary_align_mb);

    // Deep mode
    cfg.deep_mode = parse_bool(extract_value(json, "deep_mode"), cfg.deep_mode);

    std::cerr << "[TuningConfig] Loaded from " << path
              << " | enabled=" << cfg.enabled
              << " warmup_steps=" << cfg.warmup_steps
              << " min_buckets=" << cfg.min_buckets
              << " max_buckets=" << cfg.max_buckets
              << " width_max_mb=" << cfg.max_bucket_width_mb
              << " strict_waste=" << cfg.strict_waste_target
              << " waste_threshold=" << cfg.waste_threshold;
    if (cfg.profile_mode) {
        std::cerr << " | PROFILE k=[" << cfg.profile_k_min << ".." << cfg.profile_k_max
                  << "] widths=[";
        for (size_t i = 0; i < cfg.profile_widths.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << cfg.profile_widths[i];
        }
        std::cerr << "] steps/candidate=" << cfg.profile_steps_per_candidate
                  << " min_hit_rate=" << cfg.profile_min_cache_hit_rate
                  << "% max_frag=" << cfg.profile_max_fragmentation << "%";
    }
    if (cfg.benchmark_mode) {
        std::cerr << " | BENCHMARK k=[" << cfg.benchmark_k_min << ".." << cfg.benchmark_k_max
                  << "] overhead=" << cfg.benchmark_overhead_bound_mb << "MB"
                  << " candidates=" << cfg.benchmark_num_candidates
                  << " strategy=" << cfg.benchmark_sampling_strategy
                  << " steps/candidate=" << cfg.benchmark_steps_per_candidate
                  << " min_hit_rate=" << cfg.benchmark_min_cache_hit_rate
                  << "% max_frag=" << cfg.benchmark_max_fragmentation << "%";
    }
    if (cfg.deep_mode) {
        std::cerr << " | DEEP_MODE";
    }
    if (cfg.boundary_align_mb > 0.0) {
        std::cerr << " | ALIGN=" << cfg.boundary_align_mb << "MB";
    }
    std::cerr << "\n";
    return cfg;
}

// ---------------------------------------------------------------------------
// Persistence: load / save boundaries
// ---------------------------------------------------------------------------

std::vector<size_t> TuningConfig::load_persisted_boundaries(const std::string& path) {
    std::vector<size_t> boundaries;
    std::string json = read_file(path);
    if (json.empty()) return boundaries;

    // Find "boundaries_bytes" array
    std::string key = "\"boundaries_bytes\"";
    auto pos = json.find(key);
    if (pos == std::string::npos) return boundaries;

    auto arr_start = json.find('[', pos);
    auto arr_end = json.find(']', arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) return boundaries;

    std::string arr = json.substr(arr_start + 1, arr_end - arr_start - 1);
    std::istringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            try { boundaries.push_back(std::stoull(token)); }
            catch (...) { /* skip bad value */ }
        }
    }

    if (!boundaries.empty()) {
        std::cerr << "[TuningConfig] Loaded persisted boundaries from " << path << ": [";
        for (size_t i = 0; i < boundaries.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << boundaries[i] / (1024*1024) << "MB";
        }
        std::cerr << "]\n";
    }
    return boundaries;
}

bool TuningConfig::save_boundaries(const std::string& path,
                                   const std::vector<size_t>& boundaries,
                                   size_t unique_sizes_profiled,
                                   size_t total_allocations_profiled,
                                   double waste_ratio) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[TuningConfig] Failed to write " << path << "\n";
        return false;
    }

    f << "{\n";
    f << "  \"boundaries_bytes\": [";
    for (size_t i = 0; i < boundaries.size(); i++) {
        if (i > 0) f << ", ";
        f << boundaries[i];
    }
    f << "],\n";
    f << "  \"computed_from\": {\n";
    f << "    \"unique_sizes_profiled\": " << unique_sizes_profiled << ",\n";
    f << "    \"total_allocations_profiled\": " << total_allocations_profiled << ",\n";
    f << "    \"waste_ratio\": " << waste_ratio << "\n";
    f << "  }\n";
    f << "}\n";

    std::cerr << "[TuningConfig] Saved boundaries to " << path << "\n";
    return true;
}

} // namespace OwnTensor
