#include "checkpointing/GradMode.h"

namespace OwnTensor {
namespace autograd {

// Initialize thread-local state to true (gradients enabled by default)
thread_local bool GradMode::enabled_ = true;

bool GradMode::is_enabled() {
    return enabled_;
}

void GradMode::set_enabled(bool enabled) {
    enabled_ = enabled;
}

} // namespace autograd
} // namespace OwnTensor