---
name: project_future_size_split_copy_zero
description: "FUTURE (user 11-05 ~20:00 IST): per-tensor size split for TI's batched copy/zero (big tensors -> cudaMemcpyAsync/cudaMemsetAsync); 32 MB candidate on P032, P015 disagreed; work doc 45.169"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T14:35:57.641Z
---

**Decision (user, 2026-11-05 ~20:00 IST):** keep TI's batched copy (item 3) and zero (item 4) exactly as pushed
(TI 05e1d6c). The size split is a FUTURE task: "document and note it as a future task". Full record: work doc 45.169.

**What the split is:** the limit is checked PER TENSOR (one list entry), never on the whole list or flat buffer.
An entry at or above the limit gets its own driver call; the rest stay batched.

**Measured on P032 (H200, test_split2.cu, real lists).** Batched beats the old per-tensor loop in 7 of 8 lists.
- The one exception is the 1.6B zero_grad at 1 GPU: +21 us (2009.9 vs 1988.4 us), 0.0001% of a 21.6 s step.
- Multi-GPU runs zero the gradients with one memset over the flat buffer, so they never run that list.
- A split at 32 MB was best or within 1 us in all 8 lists.

**Not settled:**
- On P015 (another box and driver), splitting the 50-101 MB tensors was 7-15 us slower.
- Both tests are standalone timings, not the training step.
- The gain is small, at most 0.3 ms per step.

**The check costs nothing measurable:** 142.7 vs 143.0 us on a list with no big tensor.

**PyTorch has no split** (source checked 11-05). `_foreach_zero_` and `_foreach_copy_` batch tensors of any size
through `multi_tensor_apply`. Its default `zero_grad()` is `set_to_none=True`.

**Why:** the gain is tiny and depends on the box, so it was not worth a change today.

**How to apply:** when it is picked up:
1. Measure in the real step (nsys, 1 GPU, 135M SFT + 1.6B pretraining) on 2+ boxes.
2. Pick the limit from those numbers.
3. Implement it in `multi_tensor_copy_sm89_cuda` / `multi_tensor_zero_sm89_cuda`, old lines kept as comments
   ([[feedback_keep_old_code_as_comments]]).
4. Also consider the 32 KB kernel-argument space (PyTorch: 770 tensors / 2,240 blocks per launch on CUDA 13+).

Related: [[project_pretrain_before_after_1105]], [[feedback_isolated_measurements_mislead]].
