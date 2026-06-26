#include "autograd/GraphRecorder.h"

namespace OwnTensor {
namespace autograd {
    

// Default: nullptr — no recording, zero overhead.
thread_local GraphRecorder* GraphRecordMode::active_recorder_ = nullptr;

GraphRecorder* GraphRecordMode::get_active() {
    return active_recorder_;
}

void GraphRecordMode::set_active(GraphRecorder* recorder) {
    active_recorder_ = recorder;
}

} // namespace autograd
} // namespace OwnTensor
