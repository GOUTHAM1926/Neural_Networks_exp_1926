---
name: project-v2c-pin-regated
description: V2c's dispatched pin is beaten by +1.384 +/- 0.121 (t=11.39) under a 12-round crossover - NOT yet pinned, pending a 1.7B trainer A/B.
metadata:
  type: project
---

**V2c = `lm_head-wgrad` = `2048x50304x16384 tn`.** The dispatch table selects
`TN_256x128x64_1x2_v_ep32s2` (line 630). Measured against that (not `pins.sh`), the knobbed
config **`256x128x64_1x2_v_ep32s2_pdl_dh_dw_uad_ubd_cs` wins by +1.384 +/- 0.121, t = +11.39**
(§H54, 2026-08-29, 12-round crossover, 144 runs, preheat on, bit-exact gated).

Believable because the two arrangements are equal and opposite (+1.475 / -1.292): flipping which
slots hold the candidate flipped the sign and preserved the size. Raw means 102.93% vs 104.27%.
§H47's earlier +1.03% (t=2.83, 6 rounds) was directionally right and slightly understated - the
statistic was not trustworthy but the effect was real.

**NOT PINNED.** §H27bn is the precedent where every rung gated bit-exact standalone and the tuned
dispatch was still SLOWER end-to-end, and this session had two instruction-level wins convert at
zero. **Next step: 1.7B trainer A/B on the dispatch arm**, snapshotted backends with the symbol
count verified before each run (never by build order). Pin only if it survives that.

See [[feedback-paired-null-instrument]] and [[feedback-measure-against-dispatch-not-pins]].
