---
name: project-192-ncw3-epi-swizzle
description: 192x192's deficit was our own NCW=2 default — NCW=3 unlocks WGMMA_N=192 + a swizzlable C box, +3.1/+3.6% on two production shapes; copying his regs 0/0 loses 8%.
metadata:
  type: project
---

**One root cause, three symptoms.** MATOMS = 192/64 = 3, our default NCW = 2, so 3 % 2 != 0 →
M-split illegal → warpgroups split along **N** (SPLIT_M=0) → NCHUNKS ≥ 2 required → WGMMA_N=192
illegal → picker takes **96** → C_BOX_N=96 → 96*2=192 is not a legal swizzle → **linear epilogue →
9.18M bank conflicts**. Half-width wgmma, 2x the wgmma count, and the conflicts all fall out of
one default.

His NCW = BM/64 = 3: he splits M, each warpgroup gets one m64 atom and the full 192-wide N.

**Fix = our own §H77 NCW knob.** NCW=3 → SPLIT_M, NCHUNKS=1, WGMMA_N=192, EPI_TILE_N=64,
C_BOX_N=64 → SW128. Two costs paid: register budget (40+3*232=736 > 512, so 40/**152**), and
WGMMA_SKEW must come off (needs ≥2 atoms/warpgroup; §C7517 valued skew at +0.53%).

```
shipped pin (NCW=2)          9,183,727 conflicts  91.23% tensor  834.6 TF
NCW=3 regs 0/0 (HIS setup)     465,577            82.85%         766.4 TF  -8.16%
NCW=3 regs 40/152 + swzEPI     303,055            93.37%         860.3 TF  +3.09%
```

**Copying his config exactly LOSES 8%** — his 0/0 disables the register split. Adopt his
*mechanism*, keep our knob values, measure. Bit-identical on both shapes (824M elements).
gate_up-fwd +3.09% (t=16.1), lm_head-fwd +3.63% (t=17.6).

**The swizzle is NOT worth fixing everywhere.** 31 of 33 pins are linear, but on the 256x128
pins (K=16384) it cut conflicts **85x for +0.06%** — the k-loop dominates and the epilogue is
negligible. Only fix it where K is short. See [[feedback-isolated-measurements-mislead]].

**NOT PINNED YET** — dispatch table still points at the old config.
Related: [[project-tile-order-verdicts]], [[project-hybrid-blublas-plan]].
