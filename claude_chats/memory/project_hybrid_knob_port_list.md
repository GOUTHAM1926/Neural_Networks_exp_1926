---
name: project_hybrid_knob_port_list
description: Hybrid BluBLAS port plan — his 19 knobs audited BY CODE MECHANISM (11 present / 3 partial / 7 absent), the 4 blockers in dependency order, and the 2 knobs that are dead in his tree too
metadata:
  type: project
---

22-agent adversarial audit (§H69) replaced the name-level knob comparison. Result: **11 PRESENT, 3 PARTIAL, 7 ABSENT** — not "3 missing of 18".

**Blockers, in dependency order:**
1. **`NUM_CONSUMER_WARPGROUPS` is hardcoded `2` in ours (:375)** — must become a real knob FIRST. It is the only absent knob that blocks expressing his configs, and `K_LOOP_TWO_LEVEL`'s BM>=256 restriction dissolves once it exists. Our own file at :2610-2611 already concedes the gap and :2623 says "must not be started" — that warning predates the hybrid.
2. **HIS template truncates K: `k_iters = (K/BK)/gridDim.y`, no ceil, no assert.** Routing any `K % BK != 0` shape to his kernel silently drops the K tail = wrong answer, no error. Our residue support must travel WITH the template.
3. **Split-K: take ours whole, drop his atomicAdd path.** See [[project_splitk_ours_vs_his]].
4. Cheap: `ldc` through to the kernel; `MIN_BLOCKS_PER_SM`; `EPI_DIRECT_STG`'s two UNGATED `BAR.SYNC` (his knob deletes them, ours doesn't — ours is a strict subset); `RASTER_SWIZZLE` range 0..4 -> 0..6 (he ships SUPER_M 32 and 64); 32-slot tmap cache vs our 1 entry.

**Skip on evidence — dead in HIS tree too:** `SPLITK_FP32_STORE` (only reader is smem sizing; its one config reverted 2026-07-31) and `K_STEP_UNROLL` (105 instantiations parsed, all pass 0; his comment says "NOT YET VALIDATED").

**Features, not knobs — deferred past the perf gate:** fused bias epilogue + `launch_bgemm_sm90_addmm` (`grep -ic bias` over our 8,427 lines = **0**); host-built Hilbert/tall-ratio tile map (0 hits in ours).

Still true: multicast A/B + our 7 cluster shapes must be ported as knobs; `grep -c cluster|multicast` over his tree = 0, so the feature sets are disjoint in that direction. Keep the 128x256x64 cluster routing.

**Why:** a name-level comparison was rejected by the user; only code-mechanism evidence counts. The refute pass overturned one verdict and the critic found 4 knobs the sweep never enumerated.
**How to apply:** do 1 before anything else; never route a non-K-aligned shape to his kernel body; cite §H69 for the ledger.
