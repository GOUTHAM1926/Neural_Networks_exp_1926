#include "autograd/AnomalyMode.h"

namespace OwnTensor {
namespace autograd {

// Static member initialization
std::atomic<bool> AnomalyMode::enabled_{false};

} // namespace autograd
} // namespace OwnTensor
