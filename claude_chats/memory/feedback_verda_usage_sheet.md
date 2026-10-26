---
name: feedback_verda_usage_sheet
description: Always give a Verda GPU usage one-liner alongside the EOD report, without being asked — ONE line, dense technical facts only
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-30T16:02:13.826Z
---

"Verda" is the cloud GPU provider the user rents B300/H200 boxes from. At the end of each work
day, alongside the usual EOD report, the user also wants a **separate one-line "Verda usage
sheet" entry** summarizing what the rented GPU was actually used for that day — this feeds their
own external usage/accounting tracker, not this repo's documentation.

**Why this is easy to forget:** it's a short aside distinct from the main EOD report and from
`analyzations_*.md` documentation — the user has forgotten it themselves at least once too, and
got frustrated when I didn't proactively produce it either.

**FORMAT — strict. Corrected THREE times on 2026-07-29 because I kept over-writing it:**
- **ONE short line.** Not bullets (got "wtf did u do bro"), not prose (got "pure babbering"),
  not a long comma-chained sentence either (got "again u gave this much bluff").
- Shape: `ran <suite name> (<pass counts / key variants>) and <other suite> (<counts>) with
  <secondary tool use>`. Suite names + numbers carry the meaning — cut everything else.
- NO narrative verbs (validating/catching/confirming), NO findings, NO outcomes, NO caveats
  inside the line. Facts only. Lowercase, no fullstops.

Approved examples (copy this length, not longer):

```
ran the 84-case bf16 gemm suite (test_bgemm_sm90_nn_tn) 3 times across nn/tn on tiles 128x128,
128x256, 256x128, 128x192, 192x128, 192x192, 128x320 with beta=0/1, k512, multitile, batch2,
residue and square-2048 cases, plus full sm_90a rebuilds for ptxas register/spill data
```

```
ran test_bgemm_sm90_cluster_grid_gate (189/189, 378/378, 414/414) and test_bgemm_sm90_nn_tn
(19/19, 34/34, plus 192x192/128x320 nt) with full sm_90a rebuilds
```

**TWO one-liners are wanted, not one** (added 2026-07-30):
1. `Verda <B300|H200> usage — <date>` — what the rented GPU ran (format above).
2. `Work — <date>` — ONE SENTENCE saying what was done that day: what we set out to do and
   what actually happened. Plain technical English.
   **NOT a list of topic names.** Corrected 2026-07-30: I gave
   "rank-3 C TMA descriptor, producer tail, 346-config matrix test, RASTER_SWIZZLE knob,
   kernel_bench.sh, defect marked at 14 sites" and got "wtf is this bro, what are those?" —
   naming the artifacts says nothing about what was done to them.
   Approved shape: "ported all features onto every tile variant and built one matrix test
   covering all 346 kernel configs, ran it on H200, and fixed the two kernel bugs it exposed -
   the C descriptor overwriting the next batch when the tile does not divide M, and producer
   CTAs leaving a cluster while peers still wrote into their shared memory".
   Note the Verda usage line IS a facts/counts list; the work line is NOT. Different shapes.

**RE-READ THIS FILE BEFORE WRITING EITHER LINE.** Do not write them from a general memory of the
gist -- that is exactly how the format gets broken. Corrected AGAIN 2026-08-11: I wrote
"verda h200 usage today - roughly 2 h of single-gpu time, one device at a time (gpu 0 early when 1
was busy, then gpu 2 pinned idle for the a/b), 5 matrix sweeps + 3 ncu captures + 1 before/after
rebuild, no b300 today" and got "is this how u give verda usage bro". Everything wrong with it, and
all of it was already banned above: HOURS are not wanted (suite names + counts carry the meaning),
parenthetical explanations are narrative, "no b300 today" is a caveat, "5 matrix sweeps" is a
category count instead of the suite name and its pass counts. Same day the Work line was also wrong
-- I wrote findings and outcomes instead of one sentence on what was done.

**How to apply:** whenever giving an EOD report (asked directly, or the conversation is clearly
wrapping up for the day), ALSO produce BOTH one-liners, unasked. If a usage sheet entry was
missed for a PRIOR day, backfill it retroactively from that day's session history /
`analyzations_*.md` entries when asked — don't just apologize.
Related: [[feedback_test_log_format]], [[feedback_verify_claims_and_benchmark]].
