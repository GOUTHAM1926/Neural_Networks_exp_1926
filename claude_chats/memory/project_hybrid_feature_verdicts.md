---
name: project_hybrid_feature_verdicts
description: Hybrid step 8a DONE — all 13 shared features compared by code, ours 7 / his 3 / merge 3, with the concrete port list
metadata:
  type: project
---

All 13 mechanisms BOTH SM90 trees implement, compared by CODE (§H70). **Ours 7, his 3, merge 3.**

**TAKE FROM HIM (4 items):**
1. **32-slot TMA descriptor cache + `cuTensorMapReplaceAddress` fast path.** Biggest one. Training reuses the same shape every step with new activation pointers — our single-entry cache misses 100% of calls and rebuilds through the driver; he swaps 3 pointers.
2. **Tail drain after the persistent tile loop.** Both defer the store wait to the start of the next tile; that covers every copy except the LAST one. He drains it, we have no wait outside the loop at all. Not proven to break (a GEMM CTA fills a whole SM) but CUTLASS drains in the same place. One instruction.
3. **Per-value register asserts** (%8, 24..256) — we only check the total, he only checks the values. Take both.
4. **`BLUBLAS_NONPERSISTENT` escape + `MIN_BLOCKS_PER_SM` multiplier** (with his smem*MIN_BLOCKS assert).

**KEEP OURS (7):** integer divide (uniform-datapath mul-shift vs his fp64+fixup), pipeline depth (we reserve 1024B for the static mbars, he reserves nothing), TMA box >256, descriptor hoisting (3 modes / 30 live `_dh` rows vs his 1; DESC_WALK is the cuBLAS base+constant shape), **swizzle** (per-operand CUTLASS selector vs his ONE global welded to BK — same over-strict assert as §H66), k-loop unroll, chunked epilogue.

**MERGE (3):** registers, grid sizing (ours for clusters — his tree has ZERO clusters), tile order (our division-free maths + his group range).

**DONE this session:** RASTER_SWIZZLE cap widened 4 -> 6 so his SUPER_M 32/64 are expressible (`SUPER_M=S` == `RASTER=1, RASTER_SWIZZLE=log2 S`). Verified first that the maths is all masks/shifts on `1<<SW` and nothing else reads the constant.

**Two fixes owed in OURS:** the k-loop unroll's SKEW weld (forces BM>=256; 51 of his 53 tiles are BM<=192 — dissolves with the warpgroup knob), and the missing tail drain.

**PARKED perf idea:** he carries the epilogue slot index across tiles (`ring_base`) and drains once; we restart at slot 0 per tile.

See [[project_hybrid_knob_port_list]], [[project_splitk_ours_vs_his]].
