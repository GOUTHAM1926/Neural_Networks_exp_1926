---
name: project-instruction-track-refuted
description: Instruction excess IS the best predictor of real %cuB (r=-0.511) - the brief 2026-08-27 "refuted" note was itself wrong, built on an ncu counter that does not track throughput
metadata:
  type: project
---

**STATUS: the instruction hypothesis STANDS.** An earlier note the same day said REFUTED;
that note was wrong and is superseded (§H27as retracted by §H27at).

**Measured against REAL %cuB** (all 15 pins, one config per process, TARGET_MS=400,
kind-data-thinks-fin-03, CUDA 13.0):

    instruction ratio  vs real %cuB   r = -0.511   <- correct sign, best predictor
    log(K)             vs real %cuB   r = -0.381
    tensor gap         vs real %cuB   r = -0.207
    ncu cycle ratio    vs real %cuB   r = +0.197   <- USELESS, and what the bad note used

    instr >= 2.2 (V1,V2,V2b,V2c)  mean 93.72% of cuBLAS
    instr <= 1.6 (V3,V4,V5,V6)    mean 96.33%      difference 2.6 points
    robust: drop V2c -> r = -0.448 (n=14)

Real standings: mean **96.1%**, sd 2.27. Worst **V2c 89.1%**, then V1 94.5%, V2b 95.2%,
V5 95.4%. Best **V7c 99.4%**. Free-clock single shots carry a ~3-point noise floor, so
only V2c is clearly separated.

**Projection:** cross-shape slope ~2.7 points of %cuB per 1.0x instruction ratio. Full
parity on V1 (2.43x → 1.0x) ≈ **+3.9 points**. §H27am's intervention (2.43→2.08) measured
+0.58 clock points against +1.24 predicted — consistent within 2x, not a contradiction.
Single-digit, real, worth doing; does NOT reach 130%.

**METHOD RULE, earned twice on 2026-08-27:** correlate against measured %cuB or not at
all. §H27ak retracted a power conclusion taken in the wrong condition; §H27as retracted a
cycle-counter conclusion. **Profiler counters find MECHANISM after an effect is
established in real timing — never use them to establish the effect.**

**Also settled:** tile orientation is NOT the answer. cuBLAS runs 256x128 + 1x2 cluster
(multicasting A, the big operand) where we run 128x256 + 2x1. Matching them on V8 costs
5-15 points — our 128x256_2x1 pin is 98.2%, every 256x128 variant is 83-94%. Closes the
"tile-orientation unknown" that gated the dispatcher work.
See [[project-sm90-dispatcher-rollback]], [[project-kloop-unroll-future]].
