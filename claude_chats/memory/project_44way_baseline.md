---
name: project-44way-baseline
description: First honest 44-way A/B vs Kota, 2026-09-04 -- mean 88.5% of his, 42 losses, and the two leads that fall out of it
metadata:
  type: project
---

All 44 entry points his dispatcher calls now exist in our tree and are exactly correct
(44/44 err=0.000e+00 both sides, 4 passes). Speed at locked 1785 MHz, 12-round ABBA,
one config per process, 1056 measurements:

  **wins 1 / ties 1 / losses 42 / mean 88.5% of his / worst 68.7%**

**LEAD 1 (biggest win available): the sw64/sw32 collapse is LAYOUT-specific.**
Same config, same shape: `128x128x32_sw64` nt 98.0% but nn 76.2% / tn 76.9%;
`128x128x16_sw32` nt 94.7% but nn 71.7% / tn 68.7%. A ~22-point drop when the identical
tile moves to nn/tn -- exactly the layouts where B is MN-major. Points at our MN-major
staging path at narrow BK.

**LEAD 2: `_regsplit` loses uniformly across all three layouts** (192x128 at 80.8/80.6/82.4).
Uniform across layouts = not a layout effect. It is the register-split knob his configs set
(PRODUCER_REGS 24 / CONSUMER_REGS 240; 192x128 uses 160) and our first-pass bindings leave at
our default 40/232.

**This CONTRADICTS [[project-hybrid-feature-verdicts]] / the tracker's "never copy his
settings"**, which rested on two tiles (entry #1 +20% ours, 192x192 his regs 0/0 -8.16%).
44 tiles say the opposite for the regsplit family. Two tiles is not a rule -- revisit it.

Only win: `tn_128x64x64_splitk` 105.8% (t=+34.8), the runtime split-K built that day.
Slot #1 went 94.8% -> 98.0%, but at a DIFFERENT shape, so the 3 points are not attributed.

Each of his 44 names is a one-line `using` binding on our side, so re-pointing a slot is
trivial -- the work is choosing the target. Full detail: analyzations_h200.md §H92.
Free-clock repeat of the same A/B was deferred to 2026-09-05. See [[project-hybrid-blublas-plan]].
