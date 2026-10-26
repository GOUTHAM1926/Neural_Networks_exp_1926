---
name: project_power_is_operand_toggling
description: GEMM power is strongly operand-dependent — and a periodic benchmark fill faked a +42% gap that is really +2.73% FLOP-weighted
metadata:
  type: project
---

2026-09-11, §H219/§H219b, box 86.38.238.116, standalone 1x H200.

**THE REAL NUMBER: FLOP-weighted power gap vs cuBLAS is +2.73%** (15 production shapes,
independent random operands, locked 1200 MHz, 0.0% below-lock, 10/15 shapes worse).
Worst: **downdgrad 192x128 +11.20%**, **Q/Owgrad 192x192 +6.00%**, lm_head 256x128 +4.5-4.7%.
The **128x256 family is -1.2% to +4.2% and is NOT where the power goes**, despite carrying
10 of the 15 shapes.

**THE TRAP THAT NEARLY COST A DAY.** `h211_replay` filled ONE host buffer and copied it to
BOTH A and B, with a generator periodic mod 2048 -- at K=16384 every row of A is eight exact
repeats. cuBLAS's 256x128 stride landed in phase with that period, its operands barely
toggled, and it measured 439 W against our 620 W = **+41.3%**. With independent splitmix64
operands (different seeds per matrix) the same shape is **-0.8%**. The entire "+42% power gap"
was my own benchmark input.

**THE TELL, visible before I understood it:** our data-sensitivity (varied/constant power) was
a flat +82-86% on all four tiles; cuBLAS was +89-91% on three shapes and **+32.5% on exactly
one**. An isolated insensitivity on a single shape is an artefact signature, not a property.

**Power IS strongly data-dependent** -- constant operands drop both arms from ~630 W to
~335 W at the same clock and the same TFLOPS -- which is why this class of error is so
dangerous and why no performance counter can see it.

**Why:** it redirects the target list entirely (192x128 and 192x192, not 128x256) and
invalidates anything measured on the old fill.

**How to apply:**
- Use **fill=3** (independent, full-period, different seed per matrix). Re-measure any power
  result taken any other way.
- Lock BELOW the cap and **print the achieved clock and a below-lock %** every time: at
  1400 MHz with realistic data both arms sit at ~690-700 W, i.e. AT the cap, which clips and
  compresses the gap. 1200 MHz is clean. An earlier run was silently unlocked entirely
  (843 TFLOPS against a 735 ceiling).
- Still invalidated and needing re-measurement on fill=3: the §H213 K-sweep "cost is in the
  k-loop" conclusion, and §H214/§H215's per-iteration attribution.

See [[project_kloop_is_the_power]] (now SUSPECT), [[feedback_profiler_measurement_traps]],
[[project_instruction_cuts_do_not_convert]].
