---
name: project-sm90-dispatcher-done
description: "SM90 BF16 dispatcher is measured-table based and validated, BUT every selection predates the 2026-08-13 barrier removal and must be re-measured before it is trusted again"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-08-14T09:45:12.534Z
---

SM90 BF16 dispatcher built 2026-08-04. `Bgemm_sm90_dispatch_table.h` (GENERATED, "do not
hand-edit") + `Bgemm_sm90_dispatcher.cu`. Two tiers, per `do_nn/do_nt/do_tn`:

    cfg = lookup(kTable_<lay>, M, N, K)       TIER 1  exact (M,N,K) table, now 437 entries
    if (cfg == 0xFFFF) cfg = tree_<lay>(...)  TIER 2  fitted decision tree (3, one per layout)

Originally 225 entries, validated 225/225 through the real `blublasBgemmSM90_dispatch_*`, replacing
`N%256 ? 128x256 : 128x128`, median +12.80%, 202/225 improved, none worse (tn most, +21.35%).
Built from Test 2 = 3 runs x 58,575 rows, plus a later 204-shape small-output campaign.

## THE WHOLE TABLE IS NOW SUSPECT -- re-measure before trusting it (added 2026-08-14)

Both campaigns ran BEFORE §H27za4 removed the per-tile cluster barrier on 2026-08-13. That barrier
held 33.9% of all PC-sampling stall samples and MASKED other costs. Proof it changes selections:
`128x256x64_2x1_h_swz64` measured **-0.2 pts** against its unswizzled twin at 09:27Z that day and
**+4.8 pts** after the barrier came out at ~19:13Z -- same config, same shape, opposite verdict.

So it is NOT merely that the table has zero `swz64` entries (it has 21 `ep32s2`). **All 437
selections and all 3 fitted trees were chosen against a kernel that no longer exists.**

**FUTURE TASK, agreed order (analyzations_h200.md §H27zb3-plan):**
  1. finish optimising against the step-3 pins -- measurement only, no production impact
  2. THEN add exact tier-1 entries for BluTrain's 15 shapes. Small: §H27z.53 measured the split as
     **1 TIER-1 / 14 TIER-2**, so 14 of 15 currently fall through to the fitted tree and exact
     entries capture all of them.
  3. SEPARATELY re-run the full dispatcher campaign with the post-barrier kernel and the new
     configs in the candidate set -- this is what fixes the other 422 entries and the trees.

Step-3 "pins" are a MEASUREMENT concept and do NOT change what BluTrain runs; only the dispatcher
does. Never conflate the two. Adoption is deliberately LAST: pins adopted now would be re-entered
later anyway, and a half-updated table is worse than a consistently stale one.

**REMIND THE USER at step 2** -- i.e. when the step-3 optimisation work is called finished, or when
anyone asks why BluTrain is not seeing the measured gains.

## Still true from before

**Why 3 runs:** a single run cannot pick a winner -- agreement is ~59% between two runs, 44.9%
across three, because top candidates sit inside measurement noise. Honest out-of-sample selection
loss is **~2.5% p90**, not the 0% the table scores in-sample.

**How to apply:** never quote in-sample selection loss; never claim we beat cuBLAS from sweep data
(in-sweep %cuB is unreliable -- verify on an idle GPU, see [[project-test2-measurement-traps]]).
2-D clusters are excluded from the table: they win 0/225 and cost 0.0000% to drop.

Older open items (§H27z.30): tn tile-coverage gap at M<=1024 (5 of 8 sub-80% cases, NOT a dispatch
error); 64x320 C7520; 128x144/128x112 silent-wrong configs.
Related: [[project-step3-plan]], [[feedback-apples-to-apples-never-substitute]].

## 2026-08-24 -- TIER-1 RE-PINNED FOR THE 15 BLUTRAIN SHAPES (§H27zM / §H27zN)

**All 15 were ALREADY in tier-1** -- none fell through to the tree. They just selected 2026-08-04
configs, i.e. pre-PDL, pre-barrier-removal, pre-swz64/warp-local. So this was a FIELD UPDATE on 15
existing rows, NOT an insertion: `lookup()` binary-searches (M,N,K), so row ORDER is load-bearing
and enum ids must be APPENDED, never renumbered (the table stores the numbers).

**RESULT: 14 rows now on our measured-best pins, 1 (V2c) left on its incumbent.**
Mean gain on the re-pinned rows **+3.90%**; V1 +11.16%, V4 +6.32%, V8 +5.62%.

**V2C IS THE EXCEPTION AND IT IS WELL EVIDENCED.** `256x128x64_1x2_v_ep32s2` = 736.16 TFLOPS
= **100.2% of cuBLAS**; our best `_pdl` pin manages 716.92 = **97.6%**. The incumbent beats cuBLAS
there and we do not. Verified: cuBLAS reference 734.74 (incumbent arm) vs 734.20 (pin arm) and
734.76 in an independent earlier run -- reproducible to 0.003%; clock flat 1410 on all 16 rows;
order effect measured at +0.85% on this shape and ALTERNATED away. Our pin has never exceeded
parity there on any card at any clock (GPU3 99.0/99.6/98.1, GPU1 97.6).
**LEAD: the winner is `1x2_v` where our pin is `2x1_h` -- a CLUSTER ORIENTATION difference, on the
widest-N shape we have (N=50304).**

**METHOD NOTE:** §H27zM ran the incumbent FIRST in every round (no alternation). The bias favoured
the INCUMBENT, so the 11 wins are if anything understated; only the ties needed re-testing, which
§H27zN did with the order alternating.

## FUTURE TASK -- the full variant x shape sweep, deliberately postponed

We have evaluated roughly **23 of 458** registered configs: 17 `_pdl` variants (§H27zI) plus the 6
tier-1 incumbents. **V2c proves an old non-`_pdl` config CAN beat our best pin**, so others may
exist on shapes we never tried. Done properly (ONE CONFIG PER PROCESS -- see
[[project-test2-measurement-traps]] trap 3) that is 458 x 15 x ~14 s = **~26 hours**.
Two cheaper attacks, in order of expected value:
  1. **Targeted**: test the `1x2_v` twins of our pins on the wide-N shapes -- ~20 min, and it
     directly tests the pattern V2c revealed, rather than brute force.
  2. **Shrink the binary**: the 13.75 s per invocation is mostly CUDA loading 1327 kernels. A
     trimmed build would bring the full sweep down to a few hours.

**NOT DONE YET:** full rebuild (the box holds a FAST_BUILD binary, which must never ship or sweep),
and an end-to-end validation that the dispatcher really selects the new configs.
**BluTrain's vendored copy is a STALE FORK** -- its template differs by 3662 lines and has none of
our configs, so syncing the table alone would break its build. Left untouched; its own task.
