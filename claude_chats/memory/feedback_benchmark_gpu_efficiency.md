---
name: feedback-benchmark-gpu-efficiency
description: Benchmarks must not idle the GPU; measure per-run start-up vs busy time before any campaign
metadata:
  type: feedback
---
GPU time is scarce: the GPU should only run the measured work. Before a benchmark campaign, check from the first
runs how much of each run the GPU is actually busy (clock/util log), and fix the runner if start-up dominates.
**Why:** 18/11 the user was angry that the rivals benchmark took ~1 h: measured, each run spent 65-130 s in nsys
start-up and only 5-12 s on the GPU (~4 of 89 GPU-minutes measured). A layout-file bug also crashed every rival run
of rows 16-23 after ~2 min each, which a 1-second CPU check would have caught.
**How to apply:** validate inputs (layout files) on CPU first; start each contender once per round with all its
layouts inside (or one profiler session); run exports/summaries in the background; save each run as it ends; report
the busy share with the plan. Related: [[feedback-smoke-gate-before-every-campaign]], [[feedback-dont-burn-quota-on-agents]].
