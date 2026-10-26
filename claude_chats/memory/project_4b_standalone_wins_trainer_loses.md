---
name: project-4b-standalone-wins-trainer-loses
description: "On 4B we beat cuBLAS standalone (101.03%, 10/15 shapes) but lose in the trainer (96.07%) -- same box, same binary, same day, so the 5-point gap is real and lives between the GEMMs"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-10T09:00:47.948Z
---

MEASURED 2026-09-10 on `quiet-sun-closes-fin-03` (2x H200), GPU 1, free clocks, source
md5 `4b0655fd`.

| measurement | ours vs cuBLAS |
|---|---|
| standalone, per-GEMM, 15 shapes, 12 rounds | **101.03%** (10/15 at or above; median 100.29%; FLOP-weighted 100.30%) |
| trainer end-to-end, production regime (no BLU_MAX_STEPS) | **96.07%** |

**A 5-point swing, and this time it is NOT a cross-box artefact** -- same box, same binary, same
source md5, hours apart. Earlier versions of this contradiction were contaminated by comparing a
standalone number from a released box against a trainer number from a new one (that is why §H165
exists). This one is clean. And [[project_regime_effect_is_blas_specific]] rules out clock/power:
at the production regime both arms run the SAME ~1633 MHz at the SAME ~692 W.

**Conclusion: our GEMMs are genuinely faster than cuBLAS. The loss happens BETWEEN the GEMMs.**
That is the single most valuable open question on 4B.

Standalone reproduces across boxes -- 12 of 15 shapes within 0.5 points of 2026-09-09. Best
`downdgrad` 106.2%, `gate_up fwd` 104.2%, `K/V dgrad` 103.3%. Correctness 15/15, err=0.000e+00.

**One real regression, OPEN:** `lm_head dgrad` (8192x3072x50304, nt) fell **101.5% -> 98.7%**
(-2.8 pts). Only shape that moved, and the only one with K = vocab = 50304.

**Method notes worth keeping** (§H190-§H191):
- The 15 shapes were re-derived from the model config independently and matched the stored
  `tools/h147_cases.txt` exactly. Derivation *proves* gate_up is one fused GEMM (Cout=2*8192) and
  that Q/O and K/V each share a shape -- hence 5 sites x 3 ops, not 7.
- `h147_cases.txt` addresses variants by INDEX into `h92_all44_list.h`, a table that grew
  103 -> 570 entries. A shifted index would silently benchmark the wrong variant. Verify the
  index -> `toupper(lay)_name` mapping every time; 15/15 checked clean here.

Full write-up §H190-§H192 in `cublas_rev_eng_binary/analyzations_h200.md`.
Artefacts `h192_4b_standalone_2026_09_10/`. Scripts h191/h192/h193 in `box_rescue_2026_09_09/g2/`.
Relates to [[project_4b_dispatch_gap]], [[project_kota_blas_4b_headtohead]],
[[project_l2_pollution_parked]].
