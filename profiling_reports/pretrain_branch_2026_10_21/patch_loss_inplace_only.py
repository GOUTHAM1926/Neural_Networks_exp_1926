# 2026-10-21 (user's decision): the loss backward ALWAYS writes the gradient over the logits on the CUDA training path (the
# forward saved its stats). No switch, no out-of-place alternative in autograd: removes grad_into_logits from
# sparse_cross_entropy_loss / SparseCrossEntropyLossBackward and the out-of-place stats branch. Bit-identical gradient (work
# doc 45.38: loss, W.grad, X.grad checksums identical), -2,888 MiB trainer peak on top of the double-allocation fix.
# Contract: nothing may read the logits after backward() (checked 2026-10-21: every trainer's logits feed only the loss).
# Unchanged: the CPU path and the CUDA no-stats fallback (training never uses them); the out-of-place CUDA kernel entry stays
# in LossKernels (the old-vs-new loss bench calls it), autograd no longer reaches it.
# usage: python3 patch_loss_inplace_only.py <Tensor-Implementations dir> [<bluscript_sft.cpp>]
import sys, os
TI = sys.argv[1]
def patch(rel, edits, base=TI):
    p = os.path.join(base, rel) if base else rel
    s = open(p).read()
    for old, new in edits:
        n = s.count(old)
        assert n == 1, f"{rel}: anchor found {n} times: {old[:70]!r}"
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched", rel)

patch("include/autograd/operations/LossOps.h", [
(''' * @param grad_into_logits (2026-10-17) false (default): the backward returns a
 *        new [rows x vocab] gradient tensor. true: the backward writes the
 *        gradient OVER the logits and returns a tensor sharing their storage
 *        (CUDA, when the forward saved its stats): bit-identical gradient, one
 *        logits-sized buffer less at the start of the backward pass. Only for a
 *        caller that never reads the logits after backward() -- e.g. the
 *        trainers' training step (the logits only feed this loss).
 */''',
''' *
 * The backward writes the gradient OVER the logits (CUDA, when the forward
 * saved its stats -- the training path): the returned gradient shares the
 * logits' storage, so no second [rows x vocab] buffer exists at the start of
 * the backward pass (2026-10-17, the default and only behaviour since
 * 2026-10-21). Bit-identical gradient. CONTRACT: do not read the logits after
 * backward(); in every trainer they feed only this loss.
 */'''),
('''                                 const Tensor *denom = nullptr,
                                 bool grad_into_logits = false);''',
'''                                 const Tensor *denom = nullptr);'''),
])
patch("src/autograd/operations/LossOps.cpp", [
('''                                 int64_t ignore_index, const Tensor *denom,
                                 bool grad_into_logits) {''',
'''                                 int64_t ignore_index, const Tensor *denom) {'''),
('''          ignore_index, denom_tensor, grad_into_logits);''',
'''          ignore_index, denom_tensor);'''),
])
patch("include/autograd/backward/LossBackward.h", [
('''    // 2026-10-17 (opt-in): write the gradient OVER the saved logits instead of
    // into a new [rows x vocab] tensor (CUDA, stats path only). Bit-identical
    // gradient; one logits-sized buffer less at the start of the backward pass.
    // Only valid when nothing reads the logits after this node runs.
    bool grad_into_logits_ = false;
''', ''),
('''                                    const Tensor& denom = Tensor(),
                                    bool grad_into_logits = false);''',
'''                                    const Tensor& denom = Tensor());'''),
])
patch("src/autograd/backward/LossBackward.cpp", [
('''    const Tensor& denom, bool grad_into_logits)''', '''    const Tensor& denom)'''),
('''      ignore_index_(ignore_index), denom_(denom),
      grad_into_logits_(grad_into_logits) {}''',
'''      ignore_index_(ignore_index), denom_(denom) {}'''),
('''        // 2026-10-17 (opt-in, stats path only): the gradient overwrites the saved
        // logits -- the returned tensor shares their storage, no new buffer.
        const bool inplace = grad_into_logits_ && stats_path;''',
'''        // Stats path (= training): the gradient overwrites the saved logits -- the
        // returned tensor shares their storage, no new [rows x vocab] buffer
        // (2026-10-17; the only behaviour since 2026-10-21, see LossOps.h).
        const bool inplace = stats_path;'''),
('''                    return;
                }
                if (targets_cuda.dtype() == Dtype::UInt16) {
                    cuda::sparse_ce_backward_cuda_vec_with_stats<LogitT, uint16_t>(
                        logits_ptr, targets_cuda.data<uint16_t>(), grad_ptr,
                        smax, ssum, batch_size_, num_classes_, grad_out_ptr,
                        denom_tensor.data<float>(), stream, ignore_index_);
                } else if (targets_cuda.dtype() == Dtype::Int64) {
                    cuda::sparse_ce_backward_cuda_vec_with_stats<LogitT, int64_t>(
                        logits_ptr, targets_cuda.data<int64_t>(), grad_ptr,
                        smax, ssum, batch_size_, num_classes_, grad_out_ptr,
                        denom_tensor.data<float>(), stream, ignore_index_);
                } else if (targets_cuda.dtype() == Dtype::Int32) {
                    cuda::sparse_ce_backward_cuda_vec_with_stats<LogitT, int32_t>(
                        logits_ptr, targets_cuda.data<int32_t>(), grad_ptr,
                        smax, ssum, batch_size_, num_classes_, grad_out_ptr,
                        denom_tensor.data<float>(), stream, ignore_index_);
                } else {
                    throw std::runtime_error("SparseCrossEntropyLossBackward: unsupported target dtype for CUDA (stats path)");
                }
                return;''',
'''                    return;
                }'''),
])
if len(sys.argv) > 2:   # our SFT trainer: the switch and its argument go
    patch(sys.argv[2], [
('''    // Loss gradient written over the logits (BLU_LOSS_GRAD_INPLACE=1, default off). See the micro loop.
    const bool loss_grad_inplace = getenv_int("BLU_LOSS_GRAD_INPLACE", 0) != 0;
    if (is_master && loss_grad_inplace)
        std::cout << "[loss] BLU_LOSS_GRAD_INPLACE=1: the loss gradient is written over the logits" << std::endl;
''', ''),
('''            // BLU_LOSS_GRAD_INPLACE=1 (2026-10-17): the loss backward writes the gradient over `logits`
            // instead of a new [B*T x vocab] tensor (bit-identical gradient, -3.1 GiB at B 8 x T 4096 bf16).
            // Safe here: `logits` feeds only this loss and is not read after backward().
            Tensor loss_raw = sft ? autograd::sparse_cross_entropy_loss(logits, batch.target,
                                                                         SftDataLoader::kIgnore, &sft_denom,
                                                                         loss_grad_inplace)
                                  : autograd::sparse_cross_entropy_loss(logits, batch.target, -1, nullptr,
                                                                         loss_grad_inplace);''',
'''            // The loss backward writes the gradient over `logits` (always, since 2026-10-21; bit-identical
            // gradient, -3.1 GiB at B 8 x T 4096 bf16). `logits` feeds only this loss and is not read after backward().
            Tensor loss_raw = sft ? autograd::sparse_cross_entropy_loss(logits, batch.target,
                                                                         SftDataLoader::kIgnore, &sft_denom)
                                  : autograd::sparse_cross_entropy_loss(logits, batch.target, -1, nullptr);'''),
], base=None)
