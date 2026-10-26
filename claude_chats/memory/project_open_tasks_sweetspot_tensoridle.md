---
name: project-open-tasks-sweetspot-tensoridle
description: Two open tasks opened 2026-08-26 after the H27ai instruction cut - find the IPC/clock sweet spot, and fill V1's idle tensor pipe
metadata:
  type: project
---

**TENSOR FILL CLOSED 2026-08-27 by measurement (the sweet-spot half is NOT closed — see [[project-instruction-track-refuted]], the instruction hypothesis stands against real %cuB) — see [[project-instruction-track-refuted]].**
- *Sweet spot*: there is none to find. The within-shape intervention put the real slope
  at 1.66%/1.0x and the all-15 correlation came back the WRONG SIGN (r = -0.305).
- *Idle tensor cores*: measured absolutely on all 15 — mean gap to cuBLAS **-0.75 points**,
  and we are AHEAD on V1 (+0.89) and V4 (+0.20). The absolute headroom is real but cuBLAS
  does not capture it either (V8: they sit at 75.38%); it is amortisation, driven by K.
- *KLOOP_UNROLL* is therefore NOT the lever either of these hoped for. It may still cut
  instructions, but instructions are not the deficit.

The live question instead: **low-K amortisation** — V8 (K=512) 1.079x, V7c/V8b/V6/V7
(K=2048) 1.022-1.032x. One mechanism, five variants.

Original text below, kept for the record.

Two OPEN tasks the user asked for on 2026-08-26, after §H27ai cut the k-loop
instruction count 24.4% per wgmma:

**1. Find the sweet spot between work-per-cycle and clock.** Not a tradeoff to
balance in our case — our excess instructions were integer address math, not
math, so removing them should raise clock AND cycles together. The task is to
map the actual curve: instructions/wgmma vs achieved clock vs TFLOPS, and find
where more work-per-cycle starts costing more clock than it buys. §H27ag's
slope is −6.82% clock per 1.0x instruction ratio; that is a fit, not a law, and
it was measured over a narrow range.

**2. Fill the idle tensor cores.** V1's tensor pipe runs 78.9–91.4% active —
the rest is idle. §H27zs attributed it: cuBLAS's mainloop idles ~0 cycles per
k-iteration, ours idles 34.8, which is the length of our 31-instruction
pre-first-HGMMA phase at ~1 IPC. cuBLAS's equivalent phase is 6 instructions
AND runs with 4 wgmma still in flight, so it is entirely hidden. This is the
real headroom for the 130% goal and is independent of the clock work.

Leading lever for both: `_KLOOP_UNROLL` (slot 33, already exists). cuBLAS fits
16 static HGMMA per loop body against our 4, amortising every loop-carried cost
4x better. See [[project-p1-campaign-state]].

**Why:** user's target is 130% of cuBLAS, not parity. §H27ai closes ~40% of the
k-loop instruction gap; neither of these two is addressed by it.

**How to apply:** treat both as gated experiments on the 8 pinned shapes, same
apples-to-apples discipline as §H27ai — one knob at a time, static SASS check
locally before any GPU run.
