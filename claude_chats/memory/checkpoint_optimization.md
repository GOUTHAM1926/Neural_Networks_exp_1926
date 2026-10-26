---
name: Checkpoint optimization fixes applied
description: Async checkpointing enabled + optimizer buffer staging implemented to eliminate 225ms training blocks
type: project
originSessionId: 037f68e7-08f8-4537-ba3a-080a79e5360a
---
**Applied:** April 20, 2026

## What was the problem?

GPU idle gaps during checkpoint saves:
- Checkpoint transfers were 224.9 ms (114 + 110 ms) while PyTorch only does ~109 ms
- Training loop blocked at lines 677-680 every 6000 steps (CKPT_FREQ)
- Root cause: optimizer state (m_, v_ buffers) transferred without pre-staging to pinned memory

## What was changed?

**Change 1:** Enabled async checkpointing
- File: gpt2_attn_navin.cpp:471
- Changed `CheckpointManager ckpt_manager("checkpoints", "gpt2", 5);`
- To: `CheckpointManager ckpt_manager("checkpoints", "gpt2", 5, 0, false, true);`
- Effect: Checkpoint saves now happen in background thread, training doesn't block

**Change 2:** Applied staging to optimizer buffers
- File: src/nn/optimizer/Optim.cpp:755-810
- Added pre-staging of m_ and v_ tensors to pinned CPU memory before serialization
- Batch GPU→CPU transfers with single sync point (instead of per-buffer to_cpu())
- Effect: Optimizer transfer overhead reduced by ~50% (110 ms → ~60-80 ms)

**Why:** PyTorch stages ALL buffers (model + optimizer) to pinned memory upfront, then syncs once. We were only doing it for model params.

## Expected gains

- Training blocking on checkpoints: eliminated (0 ms instead of 225 ms)
- Checkpoint transfer time: slightly reduced
- Overall throughput: +10-15% during checkpoint steps
- GPU utilization: improved (no idle gaps between training and checkpoint)

## How to verify

1. Run training and check console output:
   - Should see "Async save started for: ..." messages
   - No training loop pause visible
   
2. Profile with nsys:
   - Memory transfers should be in background thread timeline
   - GPU kernel timeline should show no idle gaps during checkpoint

## Next optimization options

- Pre-allocate pinned buffers at CheckpointManager init (avoid per-checkpoint cudaHostAlloc)
- Batch transfer multiple tensors in single cudaMemcpyAsync call
- Expected additional gain: 15-20%

