#pragma once

#include <atomic>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Global state for autograd gradient recording.
 * 
 * Similar to torch.no_grad(), this class provides a way to globally
 * enable or disable the construction of the autograd graph.
 */
class GradMode {
public:
    /**
     * @brief Check if gradient recording is currently enabled.
     */
    static bool is_enabled();

    /**
     * @brief Enable or disable gradient recording globally for the current thread.
     */
    static void set_enabled(bool enabled);

private:
    static thread_local bool enabled_;
};

/**
 * @brief RAII guard to disable gradient recording.
 * 
 * Equivalent to torch.no_grad().
 */
class NoGradGuard {
public:
    NoGradGuard() : prev_mode_(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }

    ~NoGradGuard() {
        GradMode::set_enabled(prev_mode_);
    }

    // No copy or move
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_mode_;
};

/**
 * @brief RAII guard to enable gradient recording.
 * 
 * Useful for re-enabling gradients inside a no_grad block (e.g., for checkpointing).
 */
class GradModeGuard {
public:
    explicit GradModeGuard(bool enabled) : prev_mode_(GradMode::is_enabled()) {
        GradMode::set_enabled(enabled);
    }

    ~GradModeGuard() {
        GradMode::set_enabled(prev_mode_);
    }

    // No copy or move
    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool prev_mode_;
};

} // namespace autograd
} // namespace OwnTensor