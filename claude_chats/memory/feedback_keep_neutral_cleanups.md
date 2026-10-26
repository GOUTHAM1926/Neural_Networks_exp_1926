---
name: feedback_keep_neutral_cleanups
description: Don't revert a change that removes real redundant work just because it measured neutral; revert only if slower, incorrect, or adds complexity
metadata:
  type: feedback
---

On 2026-10-13 I reverted G2 (uniform warpgroup index: backward R2UR 59M -> 0.7M, IMAD 74M -> 41M, no time change,
identical errors) under a "keep only what wins" rule. The user objected: "don't revert the good changes... removing
redundant instructions is a good thing... maybe paired with some other optimization it gives results".

**Why:** a cleanup that is neutral alone can pay off once another bottleneck is removed (same idea as
[[feedback_sweep_combinations_not_single_knobs]]: a knob rejected alone is not rejected in combination).

**How to apply:** keep a change if it is correct (identical errors) AND removes genuinely redundant work AND is not
slower beyond noise. Revert only when it is slower, wrong, or adds complexity without removing work. Still measure every
change alone, by building the next change ON TOP of the kept one and comparing against that snapshot.
