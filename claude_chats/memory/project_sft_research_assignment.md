---
name: project-sft-research-assignment
description: From 2026-09-17 the user's assignment CHANGED - he now researches SFT (supervised fine-tuning) in POST-TRAINING, not pretraining kernels. BluBLAS/matmul is on hold.
metadata:
  type: project
---

**Changed at a meeting on 2026-09-17.** Gautam is moved off pretraining kernel work onto
**SFT (supervised fine-tuning) research in post-training**. Starting from zero on the topic -
first task is a full sweep of what exists (libraries, repos, blogs, papers, tooling).

**What this means:**
- [[project-state-2026-09-12-resume-thursday]] (BluBLAS at 99.80% of cuBLAS, ranked targets) is
  **ON HOLD**. Do not propose restarting it; he will say when.
- His background is still the advantage: CUDA kernels, autograd, performance engineering,
  profiling. Frame SFT work through that lens where it helps (throughput, memory, packing,
  fused kernels) rather than treating him as an ML-101 beginner.

**Team context unchanged:** Tensor & Ops team, 6 members, BluTrain framework (in-house DL
framework) + BluBLAS. Parallelisation/multi-GPU scaling is a DIFFERENT team's scope.
