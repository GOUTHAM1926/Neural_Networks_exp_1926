---
name: project-raster-swizzle-win
description: The big one -- his SUPER_M=8 vs our RASTER_SWIZZLE=0 was the broad loss; setting it took all 77 kernels from 90.8% to 98.0% of his
metadata:
  type: project
---

**RASTER_SWIZZLE=3 (log2 group 8) == his SUPER_M=8. Setting it moved all 77 entry points from
90.8% to 98.0% of his** (median 92.4% -> 97.1%, free clocks, 12-round ABBA, 2772 measurements,
bit-exact on all 81 first). Kernels below 90%: 33 -> 2. At or above his: 8 -> 20. Biggest gains
nn 128x64x64 88.4->116.5, nn 128x32x64 84.5->111.5, tn 128x32x64 84.0->108.5.

**Mechanism, in his own words:** "Sweeps SUPER_M rows of tiles across all N before advancing,
so consecutive tiles share operand data in L2 instead of streaming past it." His default is 8
on EVERY config; ours was 0 (no grouping) on every config the 77 bind to -- although 111 of our
own configs already set it (`_rs8`). Gain scales inversely with tile width: the widest tiles
(128x192, 128x256) are a NULL because their L2 reuse was already fine.

**EXCEPTION -- pin RASTER_SWIZZLE=0 on the 2-CTA cluster configs** (cluster_/coopB/coopB_v2).
The global rswz=3 COSTS them 1.1-1.9 pts (nt 100.2->98.4, nn 100.2->98.5, tn 99.2->97.9); a
cluster already gets its L2 locality from the cluster itself. With the pin there are zero
regressions anywhere.

**HOW IT WAS FOUND, and why three earlier hypotheses were wrong:**
  1. structural config diff -- looked identical on 74/77 (WRONG: I never mapped his SUPER_M,
     param 5, to our RASTER_SWIZZLE, param 15; different names, same knob)
  2. SASS diff -- we issue HALF his vector instructions, same HGMMA/UTMALDG -> not instructions
  3. DESC_HOIST -- tightened the HGMMA dependency chain below his (23/13/6 -> 11/4/0,
     re-encode 17 -> 1) and bought only +0.45 median -> not the dependency chain
  4. **ncu** -- our scheduler is HEALTHIER than his (0.40 vs 0.31 eligible warps, 65% vs 72%
     starved) yet slower, with DRAM 17.2% vs his 13.5% -> MEMORY TRAFFIC, not instructions
  5. back to his source with that answer -> SUPER_M

**TWO MEASUREMENT LESSONS, both nearly cost the fix:**
  * **ncu single-launch duration said rswz was SLOWER** (85024 vs 83968 cycles). It profiles one
    cold-cache launch and does not see the L2-warm steady state. Trusting it would have binned
    a +12-point win. Use ncu for WHERE the bottleneck is, never for whether a fix works.
  * static SASS wins keep failing to convert -- this is the third (see
    [[project-kloop-unroll-future]], [[project-instruction-cuts-do-not-convert]]).
    Static analysis narrows the field; it does not identify the bottleneck.

Still behind after this: sw32 in nn/tn (84-90%, layout-specific, same signature as
[[project-noswizzle-descriptor-fix]]), 192x128/192x192 (92-93%, the NCW-must-divide-MATOMS
half-width wgmma issue), 128x192/128x256 (93-95%, unmoved by raster).
See [[project-44way-baseline]]. Detail: analyzations_h200.md.
