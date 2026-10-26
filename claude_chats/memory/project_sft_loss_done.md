---
name: project-sft-loss-done
description: SFT masking + valid-token denominator + fp32 loss scalar DONE and validated (136 cases, 0 failures); trainer upcast removed; data path still does not exist
metadata:
  type: project
---

**Branch `sft_gautam`, validated by `Tests/kernel_tests/test_sparse_ce_ignore_index.cu`
on RTX 3060 sm_86: 136 cases, 0 SFT failures, 2 diagnostic XFAILs.** Uncommitted.

## What ships (library side)

- `ignore_index` masking in all 5 kernels, threaded end to end: `LossOps.cpp` -> the
  `SparseCrossEntropyLossBackward` node -> 6 CUDA call sites + the CPU loop.
- **Denominator = count of non-ignored targets**, produced as a DEVICE scalar by
  `count_valid_targets_kernel` (clamped >= 1), divided in the final reduction, and
  handed to the backward as `const float *denom`. No host sync. This is PyTorch's
  `total_weight` (`LossNLL.cpp:288` -> `:304` -> `:384`, CUDA `Loss.cu:441,449`).
  With `ignore_index = -1` the count equals batch_size, so pretraining is bit-identical.
- **`loss_output` and `grad_output` are fp32 by DECLARATION**, never the logits dtype.
  Removes a pre-existing bug where `LossBackward` bound `grad_output`'s type from the
  LOGITS, so an fp32 `grad_scale` was read as bf16 with no error. MEASURED gain:
  bf16 loss error 1.565e-03 -> 1.127e-05 (**139x**), fp16 2.007e-04 -> 8.629e-06 (23x).
- **Trainer upcast removed** (`bluscript_zero_wsd.cpp`, both call sites). Expected
  ~6.1 GB peak and ~75% loss traffic — **CALCULATED, NOT MEASURED**; needs an H200
  training run. The test never touches the trainer.

## Still open

1. **Measure the upcast removal** — the only unverified claim.
2. `batch_size > 65536` reduction cap (see [[project_sft_loss_preexisting_defects]]).
3. Target bounds check — PyTorch's `CHECK_INDEX_IN_CLASS` drops the `>= 0` half for
   unsigned targets via `if constexpr`; we have none.
4. sm103/B300 masking — one guard, `TODO(SFT)` in the code.
5. `check_type` in `Tensor::data<T>()` — logged as a RESEARCH/FUTURE TASK inside
   `LossKernels.h` with its 4 steps. Rejected for now: 840 call sites, 59 in loops,
   38 deliberate `__nv_bfloat16`/`__half` aliases, and untestable by a loss suite.

## THE THING THAT MATTERS

**SFT still cannot run.** The trainer calls `sparse_cross_entropy_loss(logits, target)`
with two arguments, so `ignore_index` defaults to -1 and nothing is masked — an "SFT
run" today is bit-identical to pretraining. There is NO preprocessing, NO format
decision, NO dataloader change (`Batch` is `{input, target}`, 0 refs to valid tokens),
and NO trainer wiring. Roughly 2 of 8 pipeline layers are done, and they are the
bottom two. See [[project_sft_dead_code_removed]].

Sentinel: targets are uint16 so PyTorch's `-100` is unrepresentable and silently a
no-op (pinned by test `F/sent_m100_u16_noop`). Use **65535**; `-1` is the safe default.
