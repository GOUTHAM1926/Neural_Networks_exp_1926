---
name: project-sft-loss-preexisting-defects
description: Three pre-existing loss-kernel defects — two FIXED 2026-09-22 (bf16 vector path, vocab%4), one still open (reduction cap >65536)
metadata:
  type: project
---

Found 2026-09-22 while writing `test_sparse_ce_ignore_index.cu`. All verified against
`a4c80c1` (= what BluTrain production pins), so none were introduced by the SFT work.

## 1+3. bf16 vector path AND vocab%4 — BOTH FIXED 2026-09-22

They were the same root cause: the 16-byte vector accesses hardcoded **4 elements**
regardless of `sizeof(T)`. That did two things at once:
- **type pun** — 16 bytes of bf16 is 8 values, staged in a `float` array and read back
  as 4 floats. Garbage even when aligned.
- **misalignment** — per-thread offset `tid*8` instead of `tid*16` for 2-byte T, so
  `cp.async...16` died with "misaligned address". The row-base guard
  `(row_logits & 0xF) == 0` was VACUOUS: it checked the row, not the per-thread address.
- and the loops ran to `vocab_size` while reading 4 slots unconditionally, so the
  `vocab % 4` remainder was double-counted (fwd) or never written (grad).

**Fix:** `LossVec<T>` with `WIDTH = 16/sizeof(T)` (4 float / 8 bf16 / 2 double), typed
staging, `alignas(16)`, plus an explicit scalar tail in all 5 kernels. Deriving WIDTH
from the dtype makes the byte offset `v*16` for every dtype, so ONE change fixed both
the pun and the alignment. Shared-memory budget unchanged (`bdim*VEC*sizeof(T)` is
always `bdim*16` bytes), so no launch config moved.

Pattern copied from our own `AddVecTraits` (`src/TensorOps/cuda/TensorOpsAdd.cu`) and
PyTorch's `aligned_vector<scalar_t, vec_size>` (`MemoryAccess.cuh:182`).

Validated: 136 cases, 0 failures; all 26 fp16/bf16 cases at **ratio exactly 1.00**
(err_kernel == err_noisefloor to the last digit). Diagnostic XFAILs 32 -> 2.

## 2. batch_size > 65536 drops loss — STILL OPEN

`sum_reduction_kernel` loads one element per thread; stage 2 is `<<<1,256>>>` over
`ceil(B/256)` partials, so partials 256+ are never read. At B=131072 half the loss
vanishes (measured err 5.008e-01). Corrupts only the LOGGED loss — the backward
re-derives from saved_max/saved_sum. Production runs 8192-16384, so it does not fire.
Test case `batch_131072_half_dropped` pins it; 65537 only drops ONE row so it is a
boundary marker, not a demonstration.

See [[project_sft_loss_done]], [[project_sft_dead_code_removed]].

**2026-11-02 update:** the third defect (stage-2 reduction cap, >65,536 rows dropped from the loss sum) was FIXED and pushed
2026-10-30 (TI 6aac3bf, strided loop in sum_reduction_kernel, bit-identical <= 65,536). All three are fixed now.
