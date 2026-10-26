---
name: project-pdl-everywhere-future
description: FUTURE TASK — extend PDL to BluTrain's non-GEMM kernels and measure whether the chain reconnects
metadata:
  type: project
---

FUTURE TASK (parked 2026-08-25, do later): add PDL to the remaining BluTrain kernels
(convert_type, gqa, rms_norm, swiglu, add/add_inplace, strided_copy) and measure whether
there is any real gain.

**Why it is worth trying:** PDL is a property of the PAIR of consecutive launches, not of one
kernel — the consumer half (launch attribute + `griddep_wait`) is what opts in. Today only our
15 pinned GEMM configs opt in; our GEMM is 37.6% of GPU time and its neighbours (BluTrain's own
kernels, 62.4%, 121k+ launches) do not. So nearly every boundary is broken and PDL collects
almost nothing in training.

**Measured today (H200, day2 captures, 36 layers x 10 steps x grad_accum 4):**
- PDL real wall-clock saving in the whole trainer = **20.6 ms** (union 36,335.1 on vs 36,355.7 off) = 0.06%
- PDL span inflation of the per-kernel SUM = **381 ms**, which lands on dgrad and made our GEMM
  read 96.5% instead of 99.0% of cuBLAS
- In the standalone bench PDL is worth +0.93% mean (V1 +6.25 outlier) because there every
  boundary is our-PDL-kernel -> our-PDL-kernel

**How to apply:** the 0.06% is a LOWER BOUND — it was measured under nsys, which spreads kernels
apart and hides the gaps PDL would fill. Do not conclude PDL is useless in production from a
profiled run. The decisive test is unprofiled end-to-end tok/s with PDL on vs off. If PDL
everywhere is attempted, remember the beta!=0 gate (§H27zv-D): `beta != 0.0f` short-circuits
the epilogue C_in read, so beta==0 leaves the one branch PDL can break unexercised.

Related: [[project-sm90-dispatcher-done]] [[feedback-isolated-measurements-mislead]]
[[project-test2-measurement-traps]]
