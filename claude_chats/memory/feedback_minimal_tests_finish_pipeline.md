---
name: feedback_minimal_tests_finish_pipeline
description: "From 10-15: time and GPU access are short - only MANDATORY tests (at most one per step), finish the full packed SFT pipeline first; extra/optional tests and comparisons come later"
metadata:
  type: feedback
---

User, 2026-10-15: "we have less time ... we should do less tests, only mandatory or at most needed ... dont be having gpu
access much time ... we need to wrap this sft-pipeline ... later we can do any number of tests ... if one test that should
be most needed, that only we will do".

**Why:** limited GPU access windows on the shared box and a deadline to get the packed SFT pipeline working end to end.

**How to apply:** for each step pick the ONE test that would catch a real break (prefer a cheap proof like a SASS
comparison or a CPU self-test over a GPU run); skip confirmatory re-runs and optional comparisons (e.g. error-value diffs
when the SASS is already identical); batch GPU checks into one short session at the end of a phase; list the skipped
tests as "later" instead of running them. Still never claim something was verified when it was not.
Related: [[feedback_no_unrequested_long_runs]], [[feedback_ask_before_every_gpu_run]].
