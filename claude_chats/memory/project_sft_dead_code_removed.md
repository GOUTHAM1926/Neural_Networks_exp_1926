---
name: project-sft-dead-code-removed
description: 469 lines of dead exported loss API deleted 2026-09-22; the near-miss was that the LIVE forward was only linking by accident
metadata:
  type: project
---

2026-09-22, branch `sft_gautam`. Deleted from `LossKernels.cu` / `LossKernels.h`:

| deleted | why it was a trap |
|---|---|
| `sparse_cross_entropy_forward_cuda` | plainest name in the header = most likely to be called |
| `sparse_cross_entropy_forward_cuda_impl` + `sparse_ce_forward_kernel_typed` | the 3-bf16-squash kernel, **no ignore_index**, **no denom** |
| `sparse_cross_entropy_forward_cuda_vec` | dead passthrough |
| `sparse_ce_backward_cuda` + `sparseCEReduce_kernel` + `sparseCENormalize_kernel` | dead backward pair |
| `sparse_cross_entropy_backward_cuda_impl` + `sparse_ce_backward_kernel_typed` | dead |

Plus 3 instantiation macros and 4 header decls. File 1995 -> 1526 lines. Verified in
the built `.so`: all dead symbols 0, all 6 live entry points present.

## THE NEAR-MISS — remember this one

`sparse_cross_entropy_forward_cuda_impl_vec` is what `LossOps.cpp` actually calls, and
it had **no explicit instantiation**. It existed only as a SIDE EFFECT of the dead
`sparse_cross_entropy_forward_cuda_vec` calling it. Deleting the dead wrapper would
have removed the symbol `LossOps.o` needs -> undefined reference at link.

Caught because `grep "template void ..._impl_vec<"` returned empty. Added
`INSTANTIATE_FWD_IMPL_VEC` for the 9 combos LossOps dispatches (fp32/fp16/bf16 x
uint16/int32/int64).

**The live production forward was linking on an accident, and had been for as long as
that dead wrapper existed.** Before deleting any exported symbol, check what its
removal de-instantiates, not just who calls it.

The header now exports exactly 6: `count_valid_targets_cuda`, `..._impl_vec`,
`..._impl_vec_save_stats`, `sparse_ce_backward_cuda_vec`, `..._vec_with_stats`,
`sparse_cross_entropy_backward_cuda`.

See [[project_sft_loss_done]], [[feedback_verify_reachability_not_presence]].
