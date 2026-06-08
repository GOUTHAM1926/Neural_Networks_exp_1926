#pragma once

#include "core/Tensor.h"
#include <vector>

namespace OwnTensor {
namespace nn {

class LossScaler {
public:
    LossScaler(float init_scale = 65536.0f, int backoff_factor = 2, int growth_factor = 2, int growth_interval = 2000);

    float scale() const { return current_scale_; }
    
    // Scales the loss value
    Tensor scale_loss(Tensor loss);

    // Checks gradients for overflow (inf/nan) — does NOT mutate gradients
    bool check_overflow(const std::vector<Tensor>& params);

    // Updates the scale factor based on whether overflow was detected
    void update(bool overflow);

    // Check a single gradient tensor for inf/nan
    bool has_overflow(const Tensor& grad);

private:
    float current_scale_;
    int backoff_factor_;
    int growth_factor_;
    int growth_interval_;
    int steps_since_last_overflow_;
};

} // namespace nn
} // namespace OwnTensor
