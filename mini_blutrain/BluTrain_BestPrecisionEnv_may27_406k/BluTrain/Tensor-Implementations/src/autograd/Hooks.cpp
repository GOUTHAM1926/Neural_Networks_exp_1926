#include "autograd/Hooks.h"
#include "core/Tensor.h"

namespace OwnTensor {

// LambdaPreHook implementation
LambdaPreHook::LambdaPreHook(hook_fn fn) : fn_(std::move(fn)) {}

Tensor LambdaPreHook::operator()(const Tensor& grad) {
    return fn_(grad);
}

// LambdaPostAccHook implementation
LambdaPostAccHook::LambdaPostAccHook(hook_fn fn) : fn_(std::move(fn)) {}

void LambdaPostAccHook::operator()(const Tensor& grad) {
    fn_(grad);
}

} // namespace OwnTensor
