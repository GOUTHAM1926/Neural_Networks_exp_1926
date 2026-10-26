---
name: project-casting-followups
description: "4 immediate future tasks after the loss fix (#1 wte round trip, #2 optimizer grad temp, #3 data<T> unchecked, #4 measure the missing speedup) — analysed 2026-09-24, nothing changed; done AFTER the SFT work (user's EOD 2026-09-24)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-24T15:56:41.341Z
---

Full write-up: `loss_kernel_dtype_bugs.md` §19 (master `~/Downloads/sft_research/`); the colleague runbook is §21.

- **#1:** `wte` is fp32 (G1), but the lm_head uses a bf16 copy (`bluscript_zero_ckpt.cpp:784-786`), so its grad goes fp32 → bf16 (`MatrixBackward.cpp:73`) → fp32. The switch is `BLU_BF16_EMBED=1`.
  - That switch is a REAL precision change (the input embedding loses bits). Unlike the logits case, it is not zero-padding.
  - The alternative is a bf16-in/fp32-out GEMM. Unknown whether BluBLAS supports one.
  - **CODE-CHECKED 2026-09-25 (doc §19.3.1): keep wte fp32, do NOT flip BLU_BF16_EMBED.** With bf16 wte, the embedding backward uses `atomicAdd` in bf16 (`EmbeddingKernels.cu:147-148`; fp32 gets the cooperative kernel, `:415-426`). Emulated: K=1000 contributions → ~4% row error, K=4000 → 7.5-20%. It is also non-deterministic. **PROVISIONAL — user finds it suspicious, parked for a deeper re-analysis (doc §19.3.2: assumptions + side findings: lying comment at :123, GPU silent garbage for tok>=V, cooperative grid = 18 blocks).** The halves sum (`Engine.cpp:342`) is one rounding; micro-step accumulation is already fp32 (`AutogradMeta.cpp:112-116`). #1 saves only ≈0.08% (est.) = low value.
- **#2:** `ShardedOptimizer::cast_shard_grad_to_fp32` makes an `as_type` temp (≈5.9 GB at W=1), then d2d-copies it. Fix: convert straight into `shard_grad_fp32_`, like `cast_master_range` already does.
  - Line numbers: dist_side_merge `:973`; local master `:948`, 25 lines earlier.
- **#3:** `data<T>()` is an unchecked reinterpret (827 sites). Research option 1 = `check_type()`.
- **#4:** speed gain +0.60% (2.35 ms per 392.2 ms micro-step) is below the bytes-only prediction 0.79–0.94%, so ~1 ms is unexplained. Hypothesis (unmeasured): the bf16 loss kernels are no longer purely bandwidth-bound. Plan: unprofiled nsys per-kernel durations of the 4 loss-path kernels, both arms, ≥2 reps; ncu locks clocks and gave bytes only. Doc §19.6.
- #1 and #2 do NOT conflict: they touch different groups. G0 = 1,472,200,704 bf16 Linear params; G1 = 116,029,184 fp32 (wte + norms).

**Why:** the user wants to go through these one by one; this stops me re-deriving them.

**Order (user's EOD, 2026-09-24):** SFT first. Next = test the version-0 SFT setup with Hugging Face SFT datasets, starting from our checkpoints. **#2 is TOMORROW (2026-09-25)** — the user's call: small, precedent in the same file. Bug D is also tomorrow ([[project-bug-d-bf16-overflow]]). #1, #3, #4 are done right after the SFT work.
- #2 checks: the new async convert must be on the same stream as the unscale/Adam that follow (the old `d2d` = `dev::copy_memory`); `shard_grad_fp32_` bit-identical before/after; allocator peak before/after (drops only if the optimizer step is the peak); speed ≈0.01%, est. `dist/zero` is in BluTrain (dist team's repo), not sft_gautam.

**How to apply:**
- Start #1 with a CODE-ONLY check: how does the embedding backward sum grads for a bf16 weight? If it sums in bf16, stop.
- Then a short A-B-A-B (`BLU_BF16_EMBED` 0/1, `BLU_BF16_LOGITS=1` on both, 2 reps). Then a long run with proper warmup.
- 20-step runs cannot show precision equivalence: the noise floor is ~0.4 loss. See [[project-bf16-logits-h200-validated]].
