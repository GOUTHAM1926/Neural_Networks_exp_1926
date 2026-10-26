---
name: project-instruction-cuts-do-not-convert
description: Two independent instruction cuts converted at ZERO. At locked clock the 192x192 pins are at cycle parity, so instructions cost power, not time - assume a cut pays nothing until measured.
metadata:
  type: project
---

**STAGE_CSEL (`cs`) on both 192x192 pins: NULL.** D = -0.075 +/- 0.231, t = -0.33 (§H53,
12-round crossover). It removes 7.8M instructions on gate_up-fwd and 10.8M on gate_up-wgrad -
real, source-attributed cuts - and buys nothing. The configs
`192x192x64_2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_cs` and `..._1x2_v_rs1_..._wlsm_cs` are compiled
and registered but **NOT pinned** (commit `c6839be`).

That is the SECOND independent instruction cut to convert at ~zero: §H27b4 cut 76% of the k-loop
vector ops and recovered 2.1% of the power gap.

**Why:** §H50 measured these kernels at **cycle parity** at locked clock (1.010x / 0.996x) with
IDENTICAL tensor-pipe instruction counts and 15-25% LESS DRAM traffic. Instructions cost POWER,
and power costs clock only under the 700 W cap - so a cut has to be large enough to move the
power draw before it moves anything at all.

**How to apply:** treat "fewer instructions" as a hypothesis about POWER, not about time, and
re-measure power after any cut rather than assuming conversion. The remaining big block is the
descriptor arithmetic at template lines 5533/5589/5431 (~65M, 5x what `cs` addressed) - but the
prior against it paying off is now high, so measure early and cheaply.
See [[feedback-paired-null-instrument]] and [[project-192-variant-search-closed]].
