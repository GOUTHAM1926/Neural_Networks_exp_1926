---
name: feedback_follow_recorded_decisions
description: Follow a recorded design decision (e.g. packing design A = default since 10-12); if I deviate, say so up front with the measured cost
metadata:
  type: feedback
---

When the work doc / memory records a decision (example: 2026-10-12 "design A (per-chat list, cu_seqlens way) is the
default", measured 2-16% faster fwd, 3-5% bwd), build on THAT path. If an easier path tempts me (10-15 morning I wired
design B / seg_start into Vj2 + the pipeline because it was quicker to port), I must say it before doing it, with the
measured cost, and let the user choose.

**Why:** 2026-10-15 the user found the new pipeline ran on B and was angry ("why are we doing seg_start if seqlens is
better?"); the morning's B work was then removed and A ported (work doc §45.7) — a day's detour.

**How to apply:** before porting/wiring a feature, grep the work doc + memory for an existing decision on it
([[project_packed_sft_pipeline_plan_1015]]); quote it in the plan; any deviation = an explicit question, not a silent choice.
