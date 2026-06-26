#pragma once

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor{

  Tensor fused_tril_softmax(Tensor& input, int64_t trilDiag, double value = 0);

} // End of Owntensor