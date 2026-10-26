---
name: feedback_apples_to_apples_never_substitute
description: "In an apples-to-apples comparison, pin the MATCHING config even when it is slower — a slower matching config is the finding, not a problem to route around"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-08-18T05:24:49.864Z
---

When comparing our kernel against cuBLAS (or any reference), **pin the configuration that MATCHES on
every field we can determine — even when we own a faster non-matching one.**

**The failure I made (2026-08-12, §H27z.92).** cuBLAS runs `192x192_64x4` at V6. Our matching
4-stage config measured 3.9 points slower than our own 3-stage. I pinned the 3-stage "because it is
our best" and labelled the mismatch in a footnote. The user's response was correct and blunt: that
HIDES the finding. The whole point of step 3 is to expose where our implementation of the SAME
configuration loses, so it can be FIXED. Substituting a different config converts a fixable defect
into an invisible one, and inflates the score while doing it.

**The rule — and the variant splits into TWO GROUPS, only one of which is matchable:**

  * **GROUP 1, always match:** tile, stage count, layout, split-K, schedule (coopA/coopB). All
    decodable from the nvjet name. A slower matching config is the RESULT — report the low number,
    it names the target. Note the layout suffix is TWO fields: letters 1-2 are the op flags
    (our nn→NN*, nt→TN*, tn→NT*, proven 3/3 single-variable §H27z.87); **letter 3 is a function of
    the TILE, not the layout** (it moved N→T with layout held constant; no tile carries both across
    164 shapes; 192x192 is always 'N'). Don't match on letter 3.
  * **GROUP 2, cluster/raster (h/v) — do NOT match, do NOT re-pin on it.** Not a convenience
    exception; four proofs: §H27z.90 the tile order is UNOBTAINABLE (positive control);
    §H27z.88/.89 cuBLAS doesn't even do a PURE walk — its fingerprint is GROUPED at ~4-8, so a
    pure-direction label and a grouped-walk label are not the same kind of object; §H27z.89b our
    v = M-fastest (re-reads A), h = N-fastest (re-reads B), "winner re-reads the smaller operand"
    holds 12/12 and our pins pick it 13/13; raster has ZERO duration effect (0.9%/0.5% across
    rs2/4/8/16). **Measured head to head (§H27z.95): raster-by-measurement 90.8% vs
    raster-by-name-mapping 88.1%, controls drifting −0.6.** The name-mapping is DISPROVED. Choosing
    raster by operand size is matching cuBLAS's BEHAVIOUR, not picking the fastest.
  * "our best config" and "the matching config" are DIFFERENT measurements answering different
    questions. Never quietly serve one when the other was asked for.

**I re-derived this WRONG a second time (2026-08-18)** — proposed "correcting" V3's pin `2x1_h`→
`2x1_v` from §H27z.86's name-mapping, and would have done V5/V7/V8 too, costing ~2.7 pts across 9
of 15 shapes. The user caught it. **Before proposing any re-pin, grep the doc for whether that
exact comparison has already been run.** §H27z.86's mapping is superseded by §H27z.95.

Related: [[project_step3_plan]], [[feedback_verify_claims_and_benchmark]].
