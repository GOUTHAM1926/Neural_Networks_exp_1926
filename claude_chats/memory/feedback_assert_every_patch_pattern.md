---
name: feedback-assert-every-patch-pattern
description: Every scripted find/replace must assert the pattern matched AND matched once, then re-read the file; a print statement is not a verification
metadata:
  type: feedback
---

2026-09-22: a python patch script used `s.replace(a, b, 1)` with **no assert** on three
edits to `LossOps.cpp`. The indentation was 2-space (that section sits outside the
`#ifdef WITH_CUDA`) not the 4-space I assumed, so all three silently did nothing — and
the script still printed "LossOps.cpp CPU path done". Cost a full build + test cycle.

**Why:** a success message that does not check anything is the same vacuous-check
pattern as [[feedback_vacuous_checks]], just in the tooling instead of the test.

**How to apply:**
- `assert a in s` AND `assert s.count(a) == 1` before every replace — ambiguity is as
  bad as a miss.
- Afterwards, `grep` the file back for the NEW text. Never report an edit from the
  script's own print.
- When a pattern misses, dump `repr()` of the real lines rather than eyeballing `sed`
  output — `sed 's/^/  /'` adds indentation and hides the true column.

The test caught it because the failures were exactly 2x (16/8), which named the cause
immediately. Keep assertions that produce a diagnosable ratio, not just pass/fail.

See [[feedback_verify_claims_and_benchmark]], [[feedback_verify_reachability_not_presence]].
