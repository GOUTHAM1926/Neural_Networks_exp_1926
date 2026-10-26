---
name: project_embed_ablation_0101
description: Task 3 (D1/D2/D3 roundings) measured locally 2026-10-01 — bf16 table loses 4% (random dirs) to 52% (aligned) on top tokens; fp32 ~3e-8; gradients synthetic
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-01T04:15:44.893Z
---

Task 3 of the 2026-10-01 list was done locally on the RTX 3060 on 2026-10-01. It used dist_side_merge kernels, real flux token ids, and **synthetic** gradient values. It passed all 234 self-checks.

Records:
- Folder: `profiling_reports/embed_ablation_2026_10_01/` (holds the test copy, the `embed_precision` binary, `logs/run_1.log`, `logs/run_fp32atomic.log` and 4 CSVs).
- Test source: `Tensor-Implementations/Tests/kernel_tests/test_embedding_grad_precision.cu`, local-only (in info/exclude).
- Documented as §17 of `embedding_optimizer_zero_1_parallelism.md` (master + mirror). The user pushed it as `5c5007b` (Neural_Networks_exp_1926 main) on 2026-10-01, together with the records (forced past `.gitignore` with `-f`; the binary was left out).

Results:
- **D1, top tokens:**
  - The counts: `,` K=1290, `.` 1200, ` the` 1056.
  - The bf16 atomicAdd loses 4.2% with random-direction gradients and 52-55% with 90%-aligned ones. The fp32 table loses ~3e-8.
  - Kruthik's arithmetic (CPU emulation) gives a 0.17% floor at every K.
- **D2, the engine add in bf16:** the smaller part loses 21-39% when it is 100× smaller.
- **D3:** the bf16 table ends at 1-2.7%, almost all from D1. The send adds little. fp32 + BLU_BF16_REDUCE costs 0.16%.

**Why:** the user's question was whether a bf16 embedding is OK for SFT (task 4 decision, [[project_state_2026_09_30_resume]]).

**How to apply:**
- The real error is somewhere between 4% and ~55% and is unknown. It needs real gradients dumped from the trainer on an H200.
- Explaining this to the user: they understood only the plain version. Use "real token ids, made-up gradient numbers, real kernels". Avoid ratios like "1 million times" (they got confused).
- Next open item: the same test on real gradients dumped from the trainer (H200).
