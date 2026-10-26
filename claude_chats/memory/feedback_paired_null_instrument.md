---
name: feedback-paired-null-instrument
description: A 6-round ABBA produced t=4.63 on a NULL effect. Use 12 rounds MINIMUM and run the pair BOTH WAYS (crossover) - a null arm alone is necessary but not sufficient.
metadata:
  type: feedback
---

ABBA-alternate the arms with a NULL arm (the incumbent against itself) - below the launch spread
`min>max` never passes and an unalternated repeat inherits any drift between the arms.

**BUT SIX ROUNDS IS NOT ENOUGH (§H53, 2026-08-29).** The same measurement, same shape, same
arrangement, same script:

    6 rounds    +0.982   t = +4.63      <- FALSE POSITIVE, the true effect is zero
    12 rounds   -0.263   t = -1.16
    swing between two runs of the SAME measurement: 1.245 points

Its null arm was wrong by the same magnitude the other way (-0.658, t=-2.17; re-measured +0.170).

**Why:** the rounds inside one run are CORRELATED - one process pool, one thermal state, one
session - so the within-run standard error badly understates run-to-run variance. **The paired t
from a single short ABBA is not a valid significance test.**

**How to apply:** 12 rounds minimum, and run the pair BOTH WAYS so the slot term is solved for
instead of assumed. With S = slot effect and D = true difference:
    arrangement 1 (inc in slots 1,4 ; cand in 2,3) -> obs1 = +D + S
    arrangement 2 (cand in 1,4 ; inc in 2,3)       -> obs2 = -D + S
    D = (obs1 - obs2)/2      S = (obs1 + obs2)/2
Script: `/tmp/h53_crossover.sh`. Measured floor 0.56 and null bias -0.01 from the older
`h27zdf_paired.sh` were 6-round numbers and are NOT reliable.

**Every 6-round gate is suspect**, including §H47's V2c (+1.03%, t=2.83) which was reported as
REAL and must be re-run as a 12-round crossover before its pin is trusted.
See [[project-instruction-cuts-do-not-convert]].
