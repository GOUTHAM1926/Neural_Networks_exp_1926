---
name: feedback-measure-against-dispatch-not-pins
description: pins.sh is NOT the dispatch table - they disagree on V2c and V6; any result meant to justify a production change must be measured against Bgemm_sm90_dispatch_table.h
metadata:
  type: feedback
---

**`tools/pins.sh` is not what production runs.** They agree on 13 of 15 shapes and **disagree
on V2c and V6**:

    var   pins.sh                                     dispatch table (production)
    V2c   256x128x64_2x1_h_ep32s2_pdl                 256x128x64_1x2_v_ep32s2
    V6    192x192x64_2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_wlsm   128x256x64_2x1_h_ep64swz128_pdl

**What this cost (2026-08-27, §H27bl):** a full day of V2c work was benchmarked against the
`pins.sh` config and reported as **+7.2 points**. Production dispatches a config that is
already **4 points better** than that baseline (94.6% vs 90.5% free). The real gain is
**+1.4 free / +2.2 locked** — a third of the claim.

Worse: `256x128x64_2x1_h_ep32s2_pdl_dh_dw_uad_ubd_cs`, which measured **+1.95 paired and
null-controlled** (§H27au), sits at **90.2%** against the dispatched config's **94.6%**.
**A gated, believable, bit-exact improvement to the wrong configuration is worth nothing.**

**RULE:** before quoting any standalone result as a production gain, check
`src/level3/batched_gemm/BF16/SM90/Bgemm_sm90_dispatch_table.h` for what that (M,N,K,layout)
actually routes to, and measure against THAT. `pins.sh` is a convenience list for sweeps, not
the production mapping.

**Also:** the TF annotations in the dispatch table comments can be stale — V2c's row reads
812.9 TF and the same config measured 678 TF free / 702 locked on 2026-08-27. Do not use the
comments as a baseline; measure.

**V6 has the same defect and has NOT been re-measured.** Its standalone numbers used the
pins.sh config. See [[project-trainer-parity-1p7b]], [[project-deadwork-removal-win]].

**THIRD OCCURRENCE, 2026-08-28 (§H37).** A full SASS instruction census was run on bare
`256x128x64_2x1_h_ep32s2_pdl` (the pins.sh config) and reported our overhead at **2.56x cuBLAS**
with `R2UR` as a 19% culprit. Production dispatches `..._dh_dw_uad_ubd_cs`, where the real
overhead is **1.67x** and R2UR is **already 94% gone** (4,194,304 -> 262,144). A work list built
on it would have sent us to re-fix something already fixed.

Running tally of wrong headlines from this one mistake: §H27bl (V2c +7.2 -> +1.4/+2.2),
§H31 (the "1.64x L2 requests" that is V1-only, mean is 1.312x), §H37 (2.56x -> 1.67x).

**Before ANY profiling run: look up the shape in `Bgemm_sm90_dispatch_table.h` and profile THAT
config.** `pins.sh` is a benchmarking convenience, not the production mapping.
