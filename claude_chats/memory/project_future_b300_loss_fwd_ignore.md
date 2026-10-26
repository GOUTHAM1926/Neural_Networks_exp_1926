---
name: project-future-b300-loss-fwd-ignore
description: "FUTURE (found 2026-10-09, user said 'forget B300, we do H200'): the B300 sparse-CE forward kernel ignores ignore_index, so SFT masking is not applied in the forward pass on B300"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-09T14:14:34.179Z
---

Found 2026-10-09 while preparing the H200 loss-kernel benchmark (TI `d96b991`, `dist_side_merge`).

- In `sparse_cross_entropy_forward_cuda_impl_vec_save_stats` (`LossKernels.cu`), when `get_arch() == Blackwell_B300`, the code calls `sparse_ce_fwd_sm103_launch(...)`.
- That launcher's signature has **no `ignore_index` parameter** (`arch/CrossEntropyKernels_sm103.cu:140-143`), and the file contains no "ignore".
- Effect on B300: masked rows (target 65535) are computed as trained rows. With V=50304, it reads `logits[row*V + 65535]`, which is past the row. So the loss value and the saved stats are wrong.
- The backward uses the generic `sparseCENormalize_from_stats`, which does honour `ignore_index`, so the gradients for masked rows are still zero.
- H200 is unaffected: it uses the generic kernel.

**Why parked:** the user said "forget b300 bro, we are doing h200". Raise it when SFT moves to B300.

**How to apply:** before any SFT run on B300, either pass `ignore_index` into the sm103 forward, or route SFT to the generic kernel.
