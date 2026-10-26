---
name: feedback_assertions_comments_can_break
description: A verification gate that greps a bare symbol name also matches prose in comments, so a comment edit turns it red and it stops being trusted
metadata:
  type: feedback
---

`verify_blutrain_wiring.sh` had TWO assertions already failing for reasons unrelated to any
build fault: it still required the tier-1/tier-2 table that §H197 deleted, and
`grep -c sm90_decline_reason` expected 6 while the file had 7 -- two of them comment prose.

**Why:** a gate that cries wolf gets ignored, and that is exactly how the `BLUBLAS_HAS_SM90`
SM89-fallback defect (-37%) survived a 9/9 static check twice.

**How to apply:** assert on the CALL SITE or the definition line, never a bare symbol
(`grep -c 'why = sm90_decline_reason('`, `grep -c '^static inline ... name('`). Before
trusting any gate, run its greps against the real file and confirm each returns what it
asserts. See [[feedback_smoke_gate_before_every_campaign]] and
[[feedback_verify_reachability_not_presence]].
