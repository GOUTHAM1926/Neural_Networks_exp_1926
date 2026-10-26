---
name: project_warpgroup_knob_scope
description: NUM_CONSUMER_WARPGROUPS is DEMOTED — not a blocker, a perf option; both its justifications failed. Also: the refactor is BIGGER than I first scoped, and which /2 to leave alone
metadata:
  type: project
---

**DEMOTED 2026-09-03 — this is NOT a blocker, it is a performance option. Do it LAST.** Both justifications failed:
- *"unblocks KLOOP_UNROLL"* — KLOOP_UNROLL is already REFUTED AND CLOSED (§H27ay: total -6.8% but VECTOR +36%, perf flat, no-dw -13.4 pts; standing instruction is "do NOT extend it"). See [[project_kloop_unroll_future]]. And the direction was inverted anyway: MORE warpgroups makes `WG_MATOMS = MATOMS/NCW` SMALLER under an M-split; what would unlock BM=128 is NCW=**1**.
- *"express his configs"* — his `256x128x64_2WG` overrides to 2 warpgroups, which is ALREADY our default. Only his DERIVED defaults differ (BM=192->3, BM=256->4), and those are alternative arrangements of tiles our tile union already covers.

Also CORRECTED: "51 of his 53 tiles blocked at BM>=256" was wrong. Real values of `WG_MATOMS = SPLIT_M ? MATOMS/2 : MATOMS`: BM=64->1 blocked, BM=128->1 blocked, **BM=192->3 LEGAL**, BM=256->2, BM=320->5. Blocked set is BM in {64,128} = 45 of 53. Our own assert message says it: "192x192 has 3".

And the refactor is BIGGER than first scoped, not smaller — the work grid is an explicit PAIR (`NCHUNKS_WG0`/`NCHUNKS_WG1` ceil/floor, `WG_NCHUNKS`, `ASYMMETRIC_N`), plus the mainloop's 2-way specialisation `if (g==0) run_mainloop(IC<0>{}) else IC<1>` and the epilogue's `row0 = g*WG_MATOMS*64`. Clean generalisation exists (`nchunks(g) = NCHUNKS/NCW + (g < NCHUNKS%NCW)` reduces to today's ceil/floor at NCW=2) but it spans WGrid, Config, mainloop and epilogue.

Original scoping notes below. Our template hardcodes `NUM_CONSUMER_WARPGROUPS = 2` at line 382 and the file warns at :2617-2630 that changing it needs "a real WGrid/epilogue/barrier-accounting change" and "must not be started". **Scoped properly (§H70), that warning overstates it.**

**Already parametric, no work:** NUM_WARPGROUPS (:384), PRODUCER_WARPGROUP (:386), the register-budget assert (:413), `BM % NUM_CONSUMER_WARPGROUPS == 0` (:423), EPI_WG_SLOTSETS (:655), both STORE_BUF sizings (:816-817), the mbarrier arrive count (:5075). Only :382 hardcodes 2.

**The real work — three expressions, one root (the work-grid / epilogue split):**
- `WGrid::SPLIT_M = (MATOMS % 2 == 0)` (:3483)
- `EPI_SPLIT_M = (MATOMS % 2 == 0)` (:654) — must stay equal to WGrid::SPLIT_M, there is a static_assert at :3488
- `NCHUNKS_WG0 = SPLIT_M ? NCHUNKS : (NCHUNKS + 1) / 2` (:3522)

**DO NOT TOUCH — these `/2` are NOT the warpgroup count:** `CREGS = BN / 2` (:522) and `CREGS_CHUNK = WGMMA_N / 2` (:5447). That is the bf16 accumulator packing (two columns per register pair) and stays /2 whatever the warpgroup count becomes.

**Why it must go first:** it is the only absent knob that blocks expressing his configs, and the k-loop unroll's `BM>=256` restriction dissolves once it exists.

**Why:** the "must not be started" note was written when this was a side quest gated on the §H27zo probe. It is now the critical path, and the probe is no longer the only route.
