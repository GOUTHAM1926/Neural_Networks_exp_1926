#pragma once

/**
 * @file AnomalyMode.h
 * @brief Anomaly detection for autograd debugging.
 * 
 * When enabled, stores stack traces at node creation and 
 * provides better error messages when gradient computation fails.
 */

#include <string>
#include <vector>
#include <atomic>
#include <mutex>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Global anomaly detection mode toggle.
 * 
 * ## Usage
 * ```cpp
 * AnomalyMode::set_enabled(true);
 * // ... perform operations ...
 * // If gradient computation fails, detailed info is shown
 * ```
 */
class AnomalyMode {
private:
    static std::atomic<bool> enabled_;
    
public:
    /// Check if anomaly detection is enabled
    static bool is_enabled() { return enabled_.load(); }
    
    /// Enable/disable anomaly detection
    static void set_enabled(bool enabled) { enabled_.store(enabled); }
};

/**
 * @brief Metadata for anomaly detection stored per-node.
 */
class AnomalyMetadata {
private:
    /// Stack trace at creation time (if anomaly mode enabled)
    std::vector<std::string> stack_trace_;
    
    /// Creation context description
    std::string creation_context_;
    
public:
    AnomalyMetadata() = default;
    
    /// Store current stack trace
    void store_stack() {
        // Simplified: just store context
        // Full implementation would capture actual stack
        creation_context_ = "Stack trace capture enabled";
    }
    
    /// Get stored stack trace
    const std::vector<std::string>& get_stack() const {
        return stack_trace_;
    }
    
    /// Get creation context
    const std::string& context() const {
        return creation_context_;
    }
    
    /// Set creation context
    void set_context(const std::string& ctx) {
        creation_context_ = ctx;
    }
};

} // namespace autograd
} // namespace OwnTensor
