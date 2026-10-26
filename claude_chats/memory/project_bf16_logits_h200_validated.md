---
name: project-bf16-logits-h200-validated
description: "2026-09-24 H200 A/B of BLU_BF16_LOGITS on dist_side_merge — memory, traffic, precision all MEASURED; the trainer is non-reproducible run-to-run; Bug D found in Types.h"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-24T12:56:46.537Z
---

Measured 2026-09-24 on Verda H200 (cold-bear-flows-fin-02). BluTrain `dist_side_merge` tip 03b94cf, submodule 8808f89, plus our 7 loss files; `bluscript_zero_ckpt.cpp` at B=4. B=8 OOMs on ONE GPU because ZeRO-1 shards nothing at world_size=1 (+15.9 GB optimizer state vs 8 GPUs).

- Their code + `BLU_BF16_LOGITS=1` → `misaligned address` at step 0; compute-sanitizer names `sparse_ce_forward_kernel_vec<bfloat16_t>`. Ours runs 20/20.
- Live peak (allocator `get_stats().active_peak`): 95,038 → 90,322 MiB = −4,716 = exactly 3 × 1,572 (the 3-tensor model). Reserved −5,876. Bit-identical across reps. ≈9.2 GiB live at B=8.
- DRAM traffic (ncu): 19.714 → 4.935 GB per micro-step = −74.97%.
- Throughput +0.60% (n=2/arm), below the 0.79–0.94% bytes-only prediction; cause not measured.
- Precision: `Tests/kernel_tests/test_cast_equivalence.cu` — 99.999% bit-identical grads, max 1 ulp, identical stats on RTX 3060 and H200. In training, fp32-vs-bf16 loss spread ≤ fp32-vs-fp32 spread.

**Why:** this is the evidence behind recommending `BLU_BF16_LOGITS=1`; don't re-derive it. The cast adds no information (bf16→fp32 is zero-padding, the grad is rounded back to bf16 before the matmul), so there is no loss benefit either way — the win is memory, traffic, and not crashing.

**How to apply:**
- The trainer is NOT reproducible run-to-run from step 1: same binary, clean GPU, step-3 loss 14.98 vs 11.17. Never compare single runs' early losses; ≥2 reps per arm. See [[feedback_paired_null_instrument]].
- nvidia-smi at 0.5 s under-read the peak by ~660 MiB vs the allocator's `reserved_peak`; use `CachingCUDAAllocator::get_stats()`.
- ncu on this trainer: validation runs first (3 domains × `val_loss_steps` forwards, ~11k converts) and burns `--launch-count`; needed a `BLU_VAL_STEPS` env override.
- **Bug D:** `include/dtype/Types.h:85` `float_to_bfloat16` uses the fp16 exponent limit (`127+15`), so any bf16 value ≥ 65,536 becomes ±Inf. Unfixed; blast radius unmeasured (every `static_cast<T>(float)` with T=bfloat16_t, on CPU AND GPU). HIGH PRIORITY, 2026-09-25: [[project-bug-d-bf16-overflow]].
- The box awk is mawk: it aborts on a missing input file — never hide that with `2>/dev/null`.
- Harness edits (B=4, `[MEM]` print, `BLU_VAL_STEPS`) exist only on the box copy — never commit them.
