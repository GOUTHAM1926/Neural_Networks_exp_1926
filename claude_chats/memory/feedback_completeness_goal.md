---
name: feedback_completeness_goal
description: "The goal is a complete/correct/general BLAS, not just matching the 18 cuBLAS variants"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-24T11:12:57.730Z
---

The BluBLAS goal is a COMPLETE, correct, general BLAS library: all layouts (NN/NT/TN),
all cluster shapes, all strategies (multicast A/B, redundant, split-K, h/v raster,
addmm/bias epilogue, ...), implemented properly and in sync with future BLAS expansion.

Matching/beating cuBLAS on the captured shapes (the 18 nvjet variants / 46 shapes) is how
we DISCOVER THE SCOPE of what strategies exist and are worth having — it is NOT the ceiling
or the whole goal. Do not treat "only the 18 variants / 46 shapes" as the target and stop
there.

**Why:** user was explicit — "our final goal is to have all strategies and variants
implemented correctly ... that cuBLAS beating is just to collect the scope ... we aren't
doing partial things, we do completely and perfectly, and everything should be correct and
sync with future BLAS expansion."

**How to apply:** don't drop/defer a layout or variant as "not needed for the 46 shapes"
(e.g. NN/TN cluster). It's REQUIRED for completeness — sequence it, don't skip it.
Correctness of any variant is validatable NOW against a cuBLAS reference for that config;
don't gate implementation on perf-tuning inputs (like a benchmark sweep). Keep everything
general/templatized (one code path + config aliases) so it composes with future expansion.
Still achieve "complete + perfect" via incremental VALIDATED steps (correctness gate per
piece), not big-bang. Note: [[feedback_scope_8_shapes]] is the NARROWER B300-P1 perf-campaign
gating rule — different context; this is the overall library-completeness goal.
