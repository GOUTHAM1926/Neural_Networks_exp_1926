---
name: project-sm90-depth4-parked
description: SM90 depth-4 stays PARKED and is worth ~1.5pt on ONE shape (V6 only); the 'V5 is 17 points behind' line in here is STALE - V5 measured 100.5% locked on 2026-08-28
metadata:
  type: project
---

**Pinned (ships):** depth 3 + `_lsw` on all four 192x192 shapes — V3 99.44, V4 99.79, V6 98.53,
V5 82.46. Best measured everywhere.

**Parked (research target):** depth 4. Reached **96.37** on V6 (`ea96s1_rs8_dh_ubd_skew_lsw`,
2 chunks) from 84.80. It is the APPLES-TO-APPLES structure — cuBLAS runs
`nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN`, four stages. We knowingly ship the non-matching
depth 3 because it is faster today.

**Every depth-4 number is V6 / 16384x5632x2048 nt only.** Never run on V3/V4/V5.

**SCOPE CORRECTION — cuBLAS uses 3 STAGES on 3 of the 4 shapes; only V6 gets 4.** Captured per
shape: V3 `64x3_1x2_h_..._NNN`, V4 `64x3_2x1_v_..._NNN`, V5 `64x3_2x1_v_..._NTN`,
V6 `64x4_2x1_v_..._TNN`. Stage count tracks the LAYOUT (NNN/NTN -> 3, TNN -> 4). So depth 3 IS
apples-to-apples on V3/V4/V5 and those are genuine matched-structure results; only V6 is a
mismatch, making the parked depth-4 work worth ~1.5 pts on ONE of fifteen shapes.
**V5 runs the IDENTICAL structure to cuBLAS and is 17 points behind** -- a pure implementation
gap, now the largest and cleanest target.

**Why it is behind:** %cuB tracks MEMBAR count at ~0.83 pts each (1→98.56, 3→96.37, 6→94.54,
9→91.94). Depth 4 forces chunking so needs ≥3 MEMBARs vs depth 3's 1 — structurally ~1.7 behind.
cuBLAS pays 0 MEMBAR and 12 bare fences costing 0.06% total; they chunk FINER than us and it is
free. Our coarse chunking is forced by an expensive primitive.

**Blocked on:** no PTX spelling gives a bare shared-proxy fence (11 enumerated, §H27zdo); the only
bare one is `fence.proxy.async.global` — wrong proxy, silent-corruption risk. Revisit on newer
ptxas, or a tile whose WGMMA_N admits a 1-chunk depth-4 epilogue.

See §H27zdm/dn/do/dp in analyzations_h200.md. Related: [[project-step3-plan]],
[[feedback-apples-to-apples-never-substitute]], [[feedback-samples-not-static-counts]].

**STALE LINE CORRECTED 2026-08-28.** This file said *"V5 runs the IDENTICAL structure to cuBLAS
and is 17 points behind (82.46) -- a pure implementation gap, now the largest and cleanest
target."* **V5 is no longer behind.** Measured 2026-08-28 (§H28b standalone sweep,
`2048x11264x16384 tn`): **100.5% locked, 95.7% free** — ahead of cuBLAS at matched clock. Do not
open V5 as a target on the strength of the 82.46 figure.

**Depth-4 priority, restated:** by this file's own scope correction, cuBLAS runs 3 stages on
V3/V4/V5 (NNN/NTN) and 4 only on V6 (TNN), so depth 3 IS the matched structure on three of the
four. Depth-4 is worth ~1.5 points on ONE of fifteen shapes and is blocked on a PTX bare
shared-proxy fence that does not exist (11 spellings enumerated, §H27zdo). It is NOT the front
line for the drain problem in [[project-instruction-excess-worklist]] item 3b — that needs
pipeline SLACK generally (earlier stage release / more buffering), which depth-4-on-V6 does not
deliver.
