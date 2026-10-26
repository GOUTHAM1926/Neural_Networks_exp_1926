---
name: project_v1_drain_not_the_cost
description: V1's wgmma drain COUNT is not what costs it; halving drains doubled each drain and barrier stall rose
metadata:
  type: project
---

Measured 2026-08-20 (§H27zo-RESULT, H200 GPU 1, clock-locked 1200 MHz, 7 reps).

cuBLAS's V1 kernel (`nvjet_sm90_tst_128x128_64x6_1x2_h_bz_splitK_NTT`) gives BOTH m64 atoms of the
128x128 tile to ONE consumer warpgroup: 8 wgmma per drain, 4 left in flight across the back-edge,
release lagged one stage. We hardcode `NUM_CONSUMER_WARPGROUPS = 2` (template line 308), so each
warpgroup owns ONE atom -> 4 wgmma per drain, 0 in flight. That structural reading is VERIFIED.

**But the causal claim built on it was REFUTED by its own pre-registered falsifier.** Reaching
WG_MATOMS=2 (via a 256x128 tile at sk4, waves held at 0.97) hit the structural prediction exactly
— 4.00 -> 8.00 HGMMA per drain — and:

    tensor pipe active  81.94 -> 86.93  (+4.99, PREDICTED)
    stall_barrier       27.35 -> 33.51  (+6.16, PREDICTED TO FALL -- IT ROSE)
    cycles              83610 -> 79059  (-5.44%)
    instructions        13.60M -> 9.61M (-29.4%)

**Why: halving the drain COUNT doubles what each drain waits for.** DEPBAR's share of samples went
UP (30.76% -> 35.45%) with half as many DEPBARs. Fewer, longer drains; total barrier time did not
fall. So the 5.44% cycle gain is real but is NOT the drain being relieved.

Also settled: **WGMMA_SKEW is not V1's lever.** With WG_MATOMS=2 available, `_skew` was 89.61 vs
89.99 without, and ADDED 2.08M instructions. §H27zdb's V3 +1.62 does not transfer here.

And: **the GEMM kernel got 5.44% faster while end-to-end got WORSE** (90.37 -> 89.99). The ncu
ledger profiles `bgemm_sm90_kernel` only; sk4's reduce pass over four fp32 partial sets is
excluded, and `dram__bytes_write` was never captured. Always state that the ledger excludes the
reduce when quoting split-K cycle ratios.

CONFOUNDED: the probe changed tile, atoms/warpgroup, split factor AND depth (6->3) at once, so the
5.44% cannot be attributed to atoms-per-warpgroup. The only single-variable test is 128x128 at sk2
with `NUM_CONSUMER_WARPGROUPS = 1`, which IS the refactor -- so it is supported, not proven.

See [[project_sm90_two_lag_types]], [[feedback_isolated_measurements_mislead]],
[[feedback_verify_claims_and_benchmark]].
