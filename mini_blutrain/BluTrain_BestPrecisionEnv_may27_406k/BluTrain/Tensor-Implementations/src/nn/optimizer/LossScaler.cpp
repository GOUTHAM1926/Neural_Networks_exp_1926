#include "nn/optimizer/LossScaler.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"
#include "ops/ScalarOps.h"
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace OwnTensor {
namespace nn {

template<typename T> struct is_complex : std::false_type {};
template<> struct is_complex<complex32_t> : std::true_type {};
template<> struct is_complex<complex64_t> : std::true_type {};
template<> struct is_complex<complex128_t> : std::true_type {};

LossScaler::LossScaler(float init_scale, int backoff_factor, int growth_factor, int growth_interval)
    : current_scale_(init_scale), backoff_factor_(backoff_factor), growth_factor_(growth_factor), 
      growth_interval_(growth_interval), steps_since_last_overflow_(0) {}

Tensor LossScaler::scale_loss(Tensor loss) {
    return loss * current_scale_;
}

bool LossScaler::check_overflow(const std::vector<Tensor>& params) {
    for (const auto& p : params) {
        if (!p.requires_grad()) continue;
        
        Tensor g = p.grad_view();
        if (!g.is_valid()) continue;

        if (has_overflow(g)) {
            return true;
        }
    }
    return false;
}

void LossScaler::update(bool overflow) {
    if (overflow) {
        current_scale_ /= backoff_factor_;
        if (current_scale_ < 1.0f) current_scale_ = 1.0f;
        steps_since_last_overflow_ = 0;
    } else {
        steps_since_last_overflow_++;
        if (steps_since_last_overflow_ >= growth_interval_) {
            current_scale_ *= growth_factor_;
            steps_since_last_overflow_ = 0;
        }
    }
}

bool LossScaler::has_overflow(const Tensor& grad) {
    if (grad.is_cuda()) {
        // Fallback to CPU for overflow check to avoid complex kernel for now
        return has_overflow(grad.to_cpu());
    }

    bool overflow = false;
    
    dispatch_by_dtype(grad.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* data = grad.data<T>();
        size_t n = grad.numel();
        
        for (size_t i = 0; i < n; ++i) {
            if constexpr (is_complex<T>::value) {
                float r = static_cast<float>(data[i].real());
                float im = static_cast<float>(data[i].imag());
                if (std::isinf(r) || std::isnan(r) || std::isinf(im) || std::isnan(im)) {
                    overflow = true;
                    break;
                }
            } else if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                float val = static_cast<float>(data[i]);
                if (std::isinf(val) || std::isnan(val)) {
                    overflow = true;
                    break;
                }
            }
        }
    });
    
    return overflow;
}

} // namespace nn
} // namespace OwnTensor
