---
name: project-f1-muon-v2-pushed
description: "F1 (NaN/Inf grad) fixed in ShardedMuon + SymmetricShardedOptimizer, pushed BluTrain dist_side_merge 23ae1e5 (2026-09-26); Muon matrices were SILENTLY frozen by a NaN, not NaN-weighted"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-26T15:39:01.174Z
---

BluTrain `dist_side_merge` `e3633e8..23ae1e5`. Author: company identity. Files:
`dist/zero/src/ShardedMuon.cpp`, `dist/zero/src/SymmetricShardedOptimizer.cpp`, and a comment-only
change in `bluscript_zero_ckpt.cpp`.

**Fix.** Same as v1's F1 (a264586):
- `if (sc && !isfinite(nsq))` becomes: throw `ZeroNonFiniteGradError` when there is no scaler (with a scaler, skip as before);
- a NaN norm is now logged as nan, not 0;
- a file-local `nonfinite_grad_msg` in each .cpp, so no header change.

The trainer's existing catch wraps `step()` for every group, so it covers all three optimizers: exit 4.

**What the old code did** (measured with a standalone test, 1x H200):
- **v2:** NaN written into the weight and into m/v.
- **Muon 1-D:** NaN written into the weight.
- **Muon 2-D:** SILENT. `muon_momentum_cuda` writes the NaN into m in place BEFORE `muon_has_nonfinite_cuda` rejects the Newton-Schulz output. The weights stay finite, but m keeps the NaN, so that matrix's update is rejected on every later step: it stops learning with no error.
- **`BLU_GLOBAL_CLIP=1`:** did not help. `reduce_and_measure` returns -1, then `step()` re-measures and goes on.

**Test:** `profiling_reports/f1_muon_v2_2026_09_26/` (`test_f1_nonfinite.cpp`, `run_f1_test.sh`, the patch, the logs).
- Before: v1 12/12; v2 and Muon fail all 16 no-scaler cases.
- After: 36/36.

**Test traps that cost 4 re-runs:**
- lr 3e-4 cannot move a bf16 weight in one step; use 1e-2.
- Muon test matrices must have dims that are multiples of 64. At 64x48 / 32x64, BluBLAS's transposed bf16 GEMM gives wrong numbers, so NS is rejected every step (see [[project-blublas-sub64-transposed-gemm]]).
- Add the healthy control BEFORE the first run ([[feedback_vacuous_checks]]).

**Not covered:**
- `max_grad_norm = 0` with no scaler: v1/v2 measure no norm at all.
- `bluscript_zero.cpp` and `train/production/bluscript.cpp` have no catch, so they stop via `std::terminate`.
- Not run with W>1.
