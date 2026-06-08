#include "core/Tensor.h"

namespace OwnTensor 
{
    // NOTE: These methods are now implemented in Tensor.cpp using the new
    // TensorImpl architecture. This file is kept for compatibility but
    // the actual implementations are in Tensor.cpp:
    //
    // - is_valid() - delegates to impl_ && impl_->storage().is_valid()
    // - release() - calls impl_.reset()
    //
    // No code needed here - methods defined in Tensor.cpp
}