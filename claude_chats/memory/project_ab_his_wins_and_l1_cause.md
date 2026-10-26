---
name: project_ab_his_wins_and_l1_cause
description: MEASURED 2026-09-03 — at identical knob values Kota's kernels beat ours on 10 of 14 configs (mean -8.30%); root cause is our per-atom TMA loop saturating L1 and starving the tensor pipe; static instruction counting predicted the OPPOSITE and must not be trusted
metadata:
  type: project
---

**THE RESULT (§H79).** His 14 dispatcher-reached configs vs the SAME 14 re-expressed in our template with identical knob values (all our extras off, so only the implementation differs). 12 rounds, order crossed over, clocks locked at 1980 MHz and verified, binary md5 checked, 168/168 rows valid.

**HIS wins 10 · OURS wins 3 · 1 tie. Mean -8.30%, median -9.62%.** t values up to -350, so not noise.
Worst for us: 128x64x64_regsplit -19.4%, 128x192x64_ws -17.9%, 128x16x64 -16.8%, 192x192x64 -15.5%, 128x160x64_regsplit -15.4%.
We win: 128x64x64 +11.3%, 128x32x64 +5.0%, 128x128x64_regsplit +2.5%.

**THE CAUSE (§H80, ncu).** On 128x160x64: our **executed** instructions 104.8M vs his 32.1M (**+227%**), tensor pipe 62.6% vs his 91.1%, **L1 throughput 98.3% = SATURATED** vs his 69%. Every stall metric is LOWER for us (barrier -68%, long-scoreboard -67%), so it is NOT synchronisation — we are shared-memory-bound and the tensor cores wait on L1. Occupancy, waves, registers (REG=168, 0 spill) and pipeline depth are identical on both sides.

**WHY (§H81, driver probe).** Our rank-3 MN-major descriptor sets `boxDim[0] = one swizzle atom`, so we must issue `MN/atom` TMA calls — 7 per k-step on 128x160 vs his 2. **The box CANNOT be widened**: `cuTensorMapEncodeTiled` returns "invalid argument" at 2x the atom on SW128/64/32 alike. His 4-D form makes the atom a gmem DIMENSION so the engine walks them in one instruction. Its price is the MN%64 truncation his host padding covers.

**THE FIX (proposed, NOT implemented, NOT measured):** add the 4-D MN-major descriptor as a second path picked at launch when `MN % 64 == 0`, keeping rank-3 otherwise — the same launch-time trick as [[project_a1b0_and_residue_fork]]. Every production shape is MN%64-aligned (0 of 15/18/17 need padding), so aligned shapes get his 2-TMA path and ragged shapes keep our correctness. Highest-value item found.

**METHOD LESSON — the big one.** Static instruction counting said we were leaner on 13 of 13 tiles. Hardware said he is faster on 10 of 14. Static counts the binary; what matters is what EXECUTES, and our small loop runs 3.3x more often. Combined with the A1B0 result (34% static cut -> +0.28% real), **static instruction counting has failed as a speed predictor twice in one day, in the same direction. Never quote it as evidence of performance.** See [[feedback_samples_not_static_counts]], [[feedback_vector_not_total_instructions]].

**CONSEQUENCE FOR THE HYBRID:** per the user's rule (take whoever wins), **10 of 14 common configs stay HIS**. We take ours on 3.
