---
name: feedback_separate_build_and_run
description: Always give PULL / BUILD / RUN as three separate command blocks — never merge build into the GPU run block
metadata:
  type: feedback
---

Give remote-box commands as THREE separate blocks, always:

1. **PULL** — `git pull` (seconds)
2. **BUILD** — `make -j$(nproc)`, `make test TEST=...` (CPU-only, MINUTES)
3. **RUN** — the GPU command, prefixed `CUDA_VISIBLE_DEVICES=1`

**Why:** building is CPU-only and takes minutes. If build and run are pasted as one block, the GPU
sits idle and LOCKED for the whole compile while the user waits — on a rented Verda box that is
paid time wasted, and it blocks a colleague who could have used the device. The user has corrected
this more than once ("did u forget or what, as building takes so much time that gpu should be
locked and cant be used while cpu takes that time").

**How to apply:** three fenced blocks with clear headings, never one combined block. Same rule for
rebuild-then-remeasure A/B loops: build both, then run both. Also never run GPU commands myself —
the user runs them on the box (see [[feedback_verify_claims_and_benchmark]]).
Related: [[feedback_gpu_pin_device_1]], [[feedback_test_log_format]].
