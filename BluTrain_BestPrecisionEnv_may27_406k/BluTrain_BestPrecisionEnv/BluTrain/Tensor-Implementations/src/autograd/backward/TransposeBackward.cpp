#include "autograd/backward/TransposeBackward.h"

namespace OwnTensor {
namespace autograd {

TransposeBackward::TransposeBackward(int dim0, int dim1)
    : dim0_(dim0), dim1_(dim1) {}

std::vector<Tensor> TransposeBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    
    // The backward of transpose(dim0, dim1) is transpose(dim0, dim1)
    // because transpose is its own inverse for the same pair of dimensions.
    // Returns a strided view; the Gradient Layout Contract enforced in
    // AutogradMeta::accumulate_grad materialises it to contiguous before
    // adopting as a leaf gradient (mirrors PyTorch derivatives.yaml:1769).
    //
    // History — Gautam_1926:
    //   v1 (buggy):       return {grads[0].transpose(dim0_, dim1_)};
    //                     Returned a non-contiguous view. Under weight tying
    //                     this view landed in wte.weight.grad via AutogradMeta::
    //                     accumulate_grad's std::move adoption, the stride-blind
    //                     multi-tensor Adam kernel then walked param[i] and
    //                     grad[i] as flat arrays and scrambled updates by a
    //                     fixed permutation — wte/py grad ratio ≈ 1/5760,
    //                     loss still descended because 69% of params trained
    //                     correctly. See Resolving GPT-2 Gradient Discrepancy.md.
    //   v2 (workaround):  return {grads[0].transpose(dim0_, dim1_).contiguous()};
    //                     Per-site fix — materialised the view here so the
    //                     contiguous tensor reached accumulate_grad. Correct but
    //                     fragile: any other backward op returning a strided view
    //                     would silently hit the same scramble.
    //   v3 (current):     return {grads[0].transpose(dim0_, dim1_)};
    //                     Reverted to the simple form once the Gradient Layout
    //                     Contract was added at AutogradMeta::accumulate_grad.
    //                     The boundary check materialises any non-contiguous
    //                     leaf gradient regardless of which backward op produced
    //                     it — structurally protects every future view-returning
    //                     backward without per-site discipline. Matches PyTorch's
    //                     derivatives.yaml:1769 line-for-line.
    return {grads[0].transpose(dim0_, dim1_)};
}

} // namespace autograd
} // namespace OwnTensor