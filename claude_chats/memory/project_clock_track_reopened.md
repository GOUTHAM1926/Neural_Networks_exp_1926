---
name: project-clock-track-reopened
description: The "get the instruction ratio under 1.8x" target is CONTRADICTED - the 192x192 pins are at 1.57-1.59x already and still behind in the trainer. Do not plan against 1.8x.
metadata:
  type: project
---

§H27al fitted a zero-crossing of 1.77x/1.80x and set the target as "get the instruction ratio
below ~1.8x and the clock deficit disappears".

**§H50 (2026-08-29) contradicts it.** The two 192x192 pins measure **1.568x** and **1.594x**
total instruction ratio and are still behind in the trainer. Either the fitted threshold is
wrong or the relationship is not what it was fitted to.

At locked clock those same kernels are at **cycle parity** (1.010x and 0.996x) with **identical
tensor-pipe instruction counts**, so the excess costs POWER, not cycles - the chain to the clock
is intact, only the 1.8x number is not.

**How to apply:** do not use 1.8x as a goal or a stopping point. Settle the contradiction before
planning against any instruction-ratio threshold. See
[[feedback-vector-not-total-instructions]] for the per-pipe breakdown that replaces it.
