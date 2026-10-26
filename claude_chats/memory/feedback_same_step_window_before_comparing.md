---
name: feedback-same-step-window-before-comparing
description: Never compare two runs' STEADY-STATE SUMMARY lines when their step counts differ — the summary averages a different window. Re-average the same steps, and de-profile before comparing absolutes.
metadata:
  type: feedback
---

The trainer's `STEADY-STATE SUMMARY` excludes the first 3 steps, so a 10-step run reports a 7-step
mean and a 15-step run reports a 12-step mean. **Comparing those two summary lines is a window
mismatch, not a comparison.** Re-average the same step range from the per-step lines first.

**Why:** doing this once turned "the deficit is 715.9 tok/s on both, identical to 0.1 tok/s" into a
false, very convincing coincidence. On the matched steps 3-9 window the two numbers are -773.6 and
-665.8 — not equal at all, and the conclusion drawn from the coincidence (a card difference) was
wrong too. The same class of error also produced a wrong nsys cost (1.34%) by pairing a plain log
with an nsys log **from a different build**; the matched pair gives **0.325%**.

**How to apply:** before any cross-run absolute comparison — (1) re-average the identical step
window from the `step N |` lines; (2) confirm both sides are equally profiled, and if not, de-profile
by a cost measured on a *matched same-build* plain/nsys pair; (3) prefer ratios computed *inside* one
run pair, where profiling and box state cancel. And log the build flags into the run output, or two
of your own logs become indistinguishable after the fact.
Related: [[feedback-apples-to-apples-never-substitute]], [[project-test2-measurement-traps]], [[feedback-verify-claims-and-benchmark]].
