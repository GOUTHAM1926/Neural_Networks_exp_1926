---
name: project-dispatched-not-best
description: The 135-shape sweep measures the config the DISPATCHER picks, not the best config we own - a tuned variant already in the tree beat the dispatched one by 4.7-10.5% on 6 of 6 shapes.
metadata:
  type: project
---

**Measuring "the dispatched config" is not measuring what we can do.** §H153 benchmarked all 135
production shapes on the entry point Kota's dispatcher selects. That answers "how good is the
hybrid today", NOT "how good is BluBLAS" — and the two differ a lot.

**Proof (§H159, 2026-09-08).** The 6 split-K K/V-wgrad shapes read 75.7-99.4% of cuBLAS on the
dispatched `TN_128x128x64_splitk(sk=2)` (clusterless). The tuned CLUSTERED family already built in
our tree — `128x128x64_1x2_v_sk2_dh_rot2_pdl_d6_shfl` — wins **6 of 6 by +4.72% to +10.46%**,
mean 85.63% -> **91.58%**, and 1.7B B4 goes to **104.07%, beating cuBLAS**. Gate 204/204 correct.
One config serves all six (91.58% mean vs 91.73% for per-shape pinning), so the rule is simple.

**Why it was missed for so long:** §H155's "34-arm sweep" contained ZERO of the 34 tuned clustered
sk configs — its four `_splitk` harness entries are all clusterless, and none of the tuned family
was in the h92 table. I then called that sweep comprehensive and built three further
investigations (§H156 nsys, §H157 BM=64, §H158 NCW=1) on the bad baseline, twice concluding "we are
at our ceiling". Gautam refused that conclusion three times and was right each time.

**Worse, the answer was already written down:** analyzations_h200.md:13734 records
`128x128x64_2x1_v_sk2` as cuBLAS's V1 kernel *in our coordinates*, and :17453 as the shipped
bit-exact V1 pin for `tn 2048x512x16384` — literally one of the six shapes.

**How to apply.** Before concluding any shape is a kernel problem: enumerate every already-built
config for that layout (510 tn configs exist; only ~96 have ever been measured) and check whether
a better one is simply not being dispatched. **Do this for the other 33 losing cases before
proposing kernel work.** Grep the template for `using Sm90Config_`, filter by the X-macro list for
the layout, and confirm the harness table actually contains them — presence in the tree is not
presence in the sweep. See [[feedback_verify_reachability_not_presence]],
[[feedback_measure_against_dispatch_not_pins]], [[project_p1_campaign_state]].
